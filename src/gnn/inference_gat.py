"""Inference and Attention Interpretation for Graph Attention Networks in BDS-35 (Phase 9).

Performs forward evaluation using trained GAT weights to produce:
  1. node_embeddings: High-order topological node vectors.
  2. propagation_risk: Risk scores propagating along attention-weighted pathways.
  3. attention_weights: Localized edge-level attention coefficients.

Academic Requirement:
  - Attention weights are used strictly for local graph-neighborhood interpretation.
  - Non-Causality Disclaimer: Attention weights do NOT prove causal relationships.
"""
from __future__ import annotations

import os
from typing import Any
import numpy as np
import torch

from src.gnn.config import ATTENTION_CAUSALITY_DISCLAIMER
from src.gnn.dataset import GraphDatasetBundle
from src.gnn.gat import GATPropagationModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def predict_gat_risk_propagation(
    dataset_bundle: GraphDatasetBundle,
    checkpoint_path: str = "models/gat_phase9.pt",
    return_attention: bool = True,
    top_k_neighbors: int = 3,
) -> dict[str, Any]:
    """Execute GAT forward pass to extract node embeddings, propagation risk, and attention weights.

    Args:
        dataset_bundle: GraphDatasetBundle with PyG Data and node mappings.
        checkpoint_path: Path to GAT model checkpoint.
        return_attention: If True, extracts attention weights for neighborhood interpretation.
        top_k_neighbors: Number of highest-attention incident edges to highlight per node.

    Returns:
        dict containing:
          - propagation_risk: dict mapping node_id -> float [0.0, 100.0]
          - normalized_risk: dict mapping node_id -> float [0.0, 1.0]
          - node_embeddings: dict mapping node_id -> list of float
          - attention_weights: list of dicts with (source, target, attention_weight)
          - neighborhood_interpretations: dict mapping node_id -> list of top-attended neighbors
          - causality_disclaimer: Non-causal academic disclaimer
    """
    full_path = os.path.join(ROOT, checkpoint_path) if not os.path.isabs(checkpoint_path) else checkpoint_path
    data = dataset_bundle.data

    # Load GAT model
    if os.path.exists(full_path):
        model = GATPropagationModel.load_checkpoint(full_path, map_location="cpu")
    else:
        in_dim = data.x.size(1)
        model = GATPropagationModel(in_dim=in_dim, hidden_dim=32, out_dim=1, heads=2, dropout=0.0)

    model.eval()
    with torch.no_grad():
        risk_tensor, embed_tensor, att_tuple = model(
            data.x, data.edge_index, return_attention_weights=return_attention
        )

    risk_vals = risk_tensor.squeeze(-1).cpu().numpy()
    embed_vals = embed_tensor.cpu().numpy()
    idx_to_id = dataset_bundle.idx_to_node_id

    propagation_risk: dict[str, float] = {}
    normalized_risk: dict[str, float] = {}
    node_embeddings: dict[str, list[float]] = {}

    for idx, nid in idx_to_id.items():
        if idx < len(risk_vals):
            r_norm = float(risk_vals[idx])
            propagation_risk[nid] = round(r_norm * 100.0, 2)
            normalized_risk[nid] = round(r_norm, 4)
            node_embeddings[nid] = [round(float(v), 5) for v in embed_vals[idx]]

    # Process attention weights for localized neighborhood interpretation
    attention_list: list[dict[str, Any]] = []
    neighborhood_map: dict[str, list[dict[str, Any]]] = {nid: [] for nid in idx_to_id.values()}

    if att_tuple is not None:
        att_edges, att_coeffs = att_tuple
        edges_np = att_edges.cpu().numpy()
        coeffs_np = att_coeffs.cpu().numpy()

        # If multi-head attention weights, average across heads
        if coeffs_np.ndim > 1:
            coeffs_np = coeffs_np.mean(axis=-1)

        num_edges = edges_np.shape[1]
        for e_idx in range(num_edges):
            u_idx = int(edges_np[0, e_idx])
            v_idx = int(edges_np[1, e_idx])
            u_id = idx_to_id.get(u_idx, str(u_idx))
            v_id = idx_to_id.get(v_idx, str(v_idx))
            score = round(float(coeffs_np[e_idx]), 4)

            entry = {
                "source_id": u_id,
                "target_id": v_id,
                "attention_weight": score,
            }
            attention_list.append(entry)

            if u_id in neighborhood_map:
                neighborhood_map[u_id].append({"neighbor_id": v_id, "attention_weight": score})

    # Sort each node's neighborhood by attention weight descending
    top_neighborhoods: dict[str, list[dict[str, Any]]] = {}
    for nid, nbrs in neighborhood_map.items():
        if nbrs:
            sorted_nbrs = sorted(nbrs, key=lambda x: x["attention_weight"], reverse=True)
            top_neighborhoods[nid] = sorted_nbrs[:top_k_neighbors]

    return {
        "propagation_risk": propagation_risk,
        "normalized_risk": normalized_risk,
        "node_embeddings": node_embeddings,
        "attention_weights": attention_list,
        "neighborhood_interpretations": top_neighborhoods,
        "causality_disclaimer": ATTENTION_CAUSALITY_DISCLAIMER,
        "metadata": {
            "checkpoint_path": checkpoint_path,
            "model_type": "GAT",
            "hidden_dim": model.hidden_dim,
            "heads": model.heads,
            "num_nodes": len(propagation_risk),
            "num_attention_edges": len(attention_list),
        },
    }
