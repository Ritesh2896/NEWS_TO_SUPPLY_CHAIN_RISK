"""Graph Attention Network (GAT) Architecture for BDS-35 (Phase 9).

Implements a 2-layer Graph Attention Network using PyTorch Geometric (GATConv)
with multi-head self-attention, layer normalization, dropout, and attention weight extraction.

Architecture:
  Input Features (N x D)
  → GATConv Layer 1 (Multi-Head Attention)
  → Activation (ELU)
  → Dropout
  → GATConv Layer 2 (Single/Multi-Head Attention -> node_embeddings)
  → Activation (ELU)
  → Dropout
  → Output Layer (Linear -> Sigmoid)

Outputs:
  - node_embeddings: Topological latent representations [N, hidden_dim]
  - propagation_risk: Propagated risk scores [N, 1] bounded in [0.0, 1.0]
  - attention_weights: Optional tuple (edge_index, alpha) capturing neighbor attention coefficients

Academic Rigor:
  - Attention weights are strictly used for localized graph-neighborhood interpretation.
  - Non-Causality Disclaimer: Attention weights do NOT prove or establish causal disruption relationships.
"""
from __future__ import annotations

import json
import os
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gnn.config import ATTENTION_CAUSALITY_DISCLAIMER

try:
    from torch_geometric.nn import GATConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

    class GATConv(nn.Module):  # type: ignore
        """Pure-PyTorch multi-head GATConv fallback."""
        def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True, dropout: float = 0.0):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.heads = heads
            self.concat = concat
            self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
            self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
            self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
            nn.init.xavier_uniform_(self.att_src)
            nn.init.xavier_uniform_(self.att_dst)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_attention_weights: bool = False):
            N = x.size(0)
            H, C = self.heads, self.out_channels
            h = self.lin(x).view(N, H, C)
            row, col = edge_index[0], edge_index[1]

            alpha_src = (h[row] * self.att_src).sum(dim=-1)
            alpha_dst = (h[col] * self.att_dst).sum(dim=-1)
            alpha = F.leaky_relu(alpha_src + alpha_dst, negative_slope=0.2)

            # Softmax per target node
            alpha_exp = torch.exp(alpha - alpha.max(dim=0, keepdim=True)[0])
            deg = torch.zeros(N, H, device=x.device).scatter_add_(0, row.unsqueeze(-1).expand(-1, H), alpha_exp)
            deg = deg.clamp(min=1e-6)
            alpha_norm = alpha_exp / deg[row]

            out = torch.zeros(N, H, C, device=x.device)
            weighted_h = alpha_norm.unsqueeze(-1) * h[col]
            out = out.scatter_add(0, row.unsqueeze(-1).unsqueeze(-1).expand(-1, H, C), weighted_h)

            if self.concat:
                out = out.view(N, H * C)
            else:
                out = out.mean(dim=1)

            if return_attention_weights:
                return out, (edge_index, alpha_norm)
            return out


class GATPropagationModel(nn.Module):
    """Two-layer Graph Attention Network for supply-chain risk propagation."""

    def __init__(
        self,
        in_dim: int = 16,
        hidden_dim: int = 32,
        out_dim: int = 1,
        heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout_rate = dropout
        self.disclaimer = ATTENTION_CAUSALITY_DISCLAIMER

        # Layer 1: Multi-head attention (outputs heads * hidden_dim if concat)
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        # Layer 2: Single-head attention output (hidden_dim) -> produces node_embeddings
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Output linear projection head -> Propagated risk score
        self.output_head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass through 2-layer GAT.

        Args:
            x: Node feature tensor [N, in_dim].
            edge_index: Graph edge indices [2, E].
            return_attention_weights: If True, returns attention weights from Layer 2.

        Returns:
            tuple of (propagation_risk, node_embeddings, attention_weights):
              - propagation_risk: Tensor [N, out_dim] in [0.0, 1.0]
              - node_embeddings: Tensor [N, hidden_dim]
              - attention_weights: Optional tuple (edge_index, alpha)
        """
        # Block 1: Multi-head attention
        h1 = self.gat1(x, edge_index)
        h1 = self.norm1(h1)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)

        # Block 2: Attention layer returning node_embeddings and optional attention weights
        att_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        if return_attention_weights:
            try:
                h2, att_weights = self.gat2(h1, edge_index, return_attention_weights=True)
            except TypeError:
                # If PyG version doesn't accept return_attention_weights directly
                h2 = self.gat2(h1, edge_index)
        else:
            h2 = self.gat2(h1, edge_index)

        h2 = self.norm2(h2)
        node_embeddings = F.elu(h2)
        h2_drop = F.dropout(node_embeddings, p=self.dropout_rate, training=self.training)

        # Output head
        logits = self.output_head(h2_drop)
        propagation_risk = torch.sigmoid(logits)

        return propagation_risk, node_embeddings, att_weights

    def save_checkpoint(
        self,
        filepath: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save model state dict and architecture hyperparameter metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "heads": self.heads,
            "dropout": self.dropout_rate,
            "disclaimer": self.disclaimer,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, filepath)

        # Save sidecar metadata JSON
        json_path = filepath.rsplit(".", 1)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "in_dim": self.in_dim,
                "hidden_dim": self.hidden_dim,
                "out_dim": self.out_dim,
                "heads": self.heads,
                "dropout": self.dropout_rate,
                "disclaimer": self.disclaimer,
                "metadata": metadata or {},
            }, f, indent=2)

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str,
        map_location: str = "cpu",
    ) -> GATPropagationModel:
        """Load model from saved checkpoint file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
        model = cls(
            in_dim=checkpoint["in_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            out_dim=checkpoint["out_dim"],
            heads=checkpoint.get("heads", 2),
            dropout=checkpoint.get("dropout", 0.1),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
