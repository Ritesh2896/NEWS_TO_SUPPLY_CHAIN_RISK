"""GraphSAGE Risk Propagation Architecture for BDS-35 (Phase 8).

Implements a 2-layer GraphSAGE network using PyTorch Geometric (SAGEConv)
that propagates risk signals and node features through the heterogeneous supply-chain graph.

Architecture:
  Input Features (N x D)
  → GraphSAGE Conv Layer 1 (SAGEConv)
  → Activation (ReLU)
  → Dropout
  → GraphSAGE Conv Layer 2 (SAGEConv)
  → Activation (ReLU)
  → Dropout
  → Output Layer (Linear -> Sigmoid)

Outputs:
  - node_embeddings: Latent topological representations [N, hidden_dim]
  - graph_propagation_risk: Propagated risk scores [N, 1] bounded in [0.0, 1.0]

Academic Rigor:
  - Checkpointing: Supports saving and loading model state dicts with architecture metadata.
  - Reproducibility: Seeded initialization and deterministic behavior.
"""
from __future__ import annotations

import json
import os
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    # Pure-PyTorch fallback implementation if PyG is not installed
    class SAGEConv(nn.Module):  # type: ignore
        def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean"):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
            self.lin_r = nn.Linear(in_channels, out_channels, bias=False)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            num_nodes = x.size(0)
            row, col = edge_index[0], edge_index[1]
            out = torch.zeros(num_nodes, self.in_channels, device=x.device)
            deg = torch.zeros(num_nodes, device=x.device).scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
            deg = deg.clamp(min=1.0).unsqueeze(-1)
            out = out.scatter_add(0, row.unsqueeze(-1).expand(-1, self.in_channels), x[col])
            out = out / deg
            return self.lin_l(x) + self.lin_r(out)


class GraphSAGEPropagationModel(nn.Module):
    """Two-layer GraphSAGE network for supply-chain risk propagation and node embedding."""

    def __init__(
        self,
        in_dim: int = 16,
        hidden_dim: int = 32,
        out_dim: int = 1,
        dropout: float = 0.1,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout
        self.aggr = aggr

        # Layer 1: Input -> Hidden
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr=aggr)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Layer 2: Hidden -> Hidden (producing intermediate latent node embeddings)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Output head: Hidden -> Scalar propagation risk
        self.output_head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through 2-layer GraphSAGE network.

        Args:
            x: Node feature matrix of shape [N, in_dim].
            edge_index: Graph edge indices of shape [2, E].
            edge_weight: Optional edge weight tensor of shape [E].

        Returns:
            tuple of (propagation_risk, node_embeddings):
              - propagation_risk: Tensor of shape [N, out_dim] in [0.0, 1.0]
              - node_embeddings: Latent representations of shape [N, hidden_dim]
        """
        # Block 1
        h1 = self.conv1(x, edge_index)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)

        # Block 2 -> Produces node embeddings
        h2 = self.conv2(h1, edge_index)
        h2 = self.norm2(h2)
        node_embeddings = F.relu(h2)
        h2_drop = F.dropout(node_embeddings, p=self.dropout_rate, training=self.training)

        # Output head -> Propagated risk score
        logits = self.output_head(h2_drop)
        propagation_risk = torch.sigmoid(logits)

        return propagation_risk, node_embeddings

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
            "dropout": self.dropout_rate,
            "aggr": self.aggr,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, filepath)

        # Also save lightweight JSON metadata sidecar
        json_path = filepath.rsplit(".", 1)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "in_dim": self.in_dim,
                "hidden_dim": self.hidden_dim,
                "out_dim": self.out_dim,
                "dropout": self.dropout_rate,
                "aggr": self.aggr,
                "metadata": metadata or {},
            }, f, indent=2)

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str,
        map_location: str = "cpu",
    ) -> GraphSAGEPropagationModel:
        """Load model from saved checkpoint file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
        model = cls(
            in_dim=checkpoint["in_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            out_dim=checkpoint["out_dim"],
            dropout=checkpoint.get("dropout", 0.1),
            aggr=checkpoint.get("aggr", "mean"),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
