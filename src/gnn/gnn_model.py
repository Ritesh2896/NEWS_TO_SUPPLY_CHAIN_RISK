"""Lightweight GraphSAGE and GAT architectures implemented in pure PyTorch.

Academically grounded implementations:
  - GraphSAGE: Hamilton et al. (NeurIPS 2017) with neighbor mean-pooling and skip concatenation.
  - GAT: Veličković et al. (ICLR 2018) with multi-head self-attention coefficients and LeakyReLU.
CPU and GPU compatible without brittle C++ binary wheel dependencies.
"""
from __future__ import annotations
import json
import os
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """Single GraphSAGE layer: aggregates neighbor representations and concatenates with self."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Projection for concatenated [self || neighbor_agg]
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """Args:

        x: [N, in_dim] node feature matrix
        adj_norm: [N, N] row-normalized adjacency matrix
        """
        # Neighbor aggregation via normalized adjacency matrix multiply: [N, in_dim]
        neigh_h = torch.matmul(adj_norm, x)
        # Concatenate self and neighbor representations: [N, 2 * in_dim]
        concat_h = torch.cat([x, neigh_h], dim=-1)
        out = self.linear(concat_h)
        out = self.layer_norm(out)
        out = F.relu(out)
        return self.dropout(out)


class GraphSAGE(nn.Module):
    """Multi-layer GraphSAGE network for supply-chain risk scoring."""

    def __init__(self, in_dim: int = 12, hidden: int = 32, out_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layer1 = GraphSAGELayer(in_dim, hidden, dropout)
        self.layer2 = GraphSAGELayer(hidden, hidden, dropout)
        self.classifier = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for node risk."""
        # Ensure row-normalized adjacency
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-5)
        adj_norm = adj / deg

        h1 = self.layer1(x, adj_norm)
        h2 = self.layer2(h1, adj_norm)
        logits = self.classifier(h2).squeeze(-1)
        return logits


class GraphAttentionLayer(nn.Module):
    """Multi-head Graph Attention (GAT) layer computing attention coefficients alpha_ij."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 2, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.alpha = alpha

        # Per-head linear transformations
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.a_src = nn.Parameter(torch.zeros(n_heads, out_dim))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Args:

        x: [N, in_dim]
        adj: [N, N] binary adjacency matrix with self-loops
        """
        N = x.size(0)
        # Linear projection: [N, n_heads, out_dim]
        Wh = self.W(x).view(N, self.n_heads, self.out_dim)

        # Attentive energy components: e_ij = LeakyReLU(a_src * Wh_i + a_dst * Wh_j)
        score_src = (Wh * self.a_src).sum(dim=-1)  # [N, n_heads]
        score_dst = (Wh * self.a_dst).sum(dim=-1)  # [N, n_heads]

        # Outer sum: [n_heads, N, N]
        scores = score_src.T.unsqueeze(2) + score_dst.T.unsqueeze(1)
        scores = self.leaky_relu(scores)

        # Mask non-edges: where adj == 0, set to large negative value
        mask = (adj == 0).unsqueeze(0).expand(self.n_heads, N, N)
        scores = scores.masked_fill(mask, -1e9)

        # Attention coefficients: [n_heads, N, N]
        alpha = F.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)

        # Aggregate: [n_heads, N, out_dim]
        Wh_t = Wh.permute(1, 0, 2)  # [n_heads, N, out_dim]
        out = torch.bmm(alpha, Wh_t)  # [n_heads, N, out_dim]

        # Reshape or average heads: [N, n_heads * out_dim]
        out = out.permute(1, 0, 2).contiguous().view(N, -1)
        return F.elu(out)


class GAT(nn.Module):
    """Graph Attention Network (GAT) for supply-chain risk scoring."""

    def __init__(self, in_dim: int = 12, hidden: int = 16, n_heads: int = 2, out_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gat1 = GraphAttentionLayer(in_dim, hidden, n_heads=n_heads, dropout=dropout)
        # Second layer with single head
        self.gat2 = GraphAttentionLayer(hidden * n_heads, hidden, n_heads=1, dropout=dropout)
        self.classifier = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Add self-loops to adjacency if not already present
        N = adj.size(0)
        adj_loops = torch.clamp(adj + torch.eye(N, device=adj.device, dtype=adj.dtype), 0.0, 1.0)

        h1 = self.gat1(x, adj_loops)
        h2 = self.gat2(h1, adj_loops)
        logits = self.classifier(h2).squeeze(-1)
        return logits


def normalized_adjacency(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Construct self-looped symmetric normalized adjacency matrix."""
    a = np.eye(n, dtype=np.float32)
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            a[u, v] = 1.0
            a[v, u] = 1.0
    deg = a.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return a / deg


def topology_risk(features: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    """Explainable analytical fallback: own direct risk + neighbor shock diffusion + centrality."""
    x = np.asarray(features, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([])
    a = normalized_adjacency(n, edges)
    base = np.clip(x[:, 0], 0.0, 1.0)
    neighbor = np.clip(a @ base, 0.0, 1.0)
    centrality = np.clip(x[:, 8], 0.0, 1.0) if x.shape[1] > 8 else np.zeros(n)
    return np.clip(0.60 * base + 0.30 * neighbor + 0.10 * centrality, 0.0, 1.0)


def save_model(model: nn.Module, path: str, metadata: dict[str, Any]) -> None:
    """Persist PyTorch model weights and accompanying metadata JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    json_path = (path[:-3] if path.endswith(".pt") else path) + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
