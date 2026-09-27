"""Inference module for GNN risk propagation in BDS-35 (Phase 8).

Loads trained GraphSAGE model weights and performs forward propagation across the
supply-chain graph to produce:
  1. node_embedding: Dense latent representations for every node.
  2. graph_propagation_risk: Propagated risk scores reflecting structural exposure.

Maintains backward compatibility with legacy score_graph() callers.
"""
from __future__ import annotations

import os
import json
from typing import Any
import numpy as np
import pandas as pd
import torch

from src.gnn.dataset import GraphDatasetBundle
from src.gnn.feature_engineering import build_node_features
from src.gnn.gnn_model import GraphSAGE, GAT, topology_risk
from src.gnn.graphsage import GraphSAGEPropagationModel

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def predict_risk_propagation(
    dataset_bundle: GraphDatasetBundle,
    checkpoint_path: str | None = None,
    model: str = "graphsage",
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute GNN forward pass to obtain node embeddings and propagated risk scores.

    Supports model selection:
      - model = "graphsage"
      - model = "gat"

    Args:
        dataset_bundle: GraphDatasetBundle containing PyG Data and node mappings.
        checkpoint_path: Optional path to the trained model checkpoint.
        model: GNN architecture to evaluate ("graphsage" or "gat").
        **kwargs: Additional model-specific options (e.g. return_attention for GAT).

    Returns:
        dict containing propagation_risk, normalized_risk, node_embeddings, and metadata.
    """
    model_choice = model.lower().strip()
    if model_choice == "gat":
        from src.gnn.inference_gat import predict_gat_risk_propagation
        ckpt = checkpoint_path or "models/gat_phase9.pt"
        return predict_gat_risk_propagation(
            dataset_bundle=dataset_bundle,
            checkpoint_path=ckpt,
            **kwargs,
        )

    ckpt = checkpoint_path or "models/graphsage_phase8.pt"
    full_path = os.path.join(ROOT, ckpt) if not os.path.isabs(ckpt) else ckpt
    data = dataset_bundle.data

    # 1. Load trained GraphSAGE model if checkpoint exists
    if os.path.exists(full_path):
        sage_model = GraphSAGEPropagationModel.load_checkpoint(full_path, map_location="cpu")
    else:
        # Initialize default architecture if checkpoint has not yet been generated
        in_dim = data.x.size(1)
        sage_model = GraphSAGEPropagationModel(in_dim=in_dim, hidden_dim=32, out_dim=1, dropout=0.0)

    sage_model.eval()
    with torch.no_grad():
        risk_tensor, embed_tensor = sage_model(data.x, data.edge_index, data.edge_weight)

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

    return {
        "propagation_risk": propagation_risk,
        "normalized_risk": normalized_risk,
        "node_embeddings": node_embeddings,
        "metadata": {
            "checkpoint_path": ckpt,
            "model_type": "GraphSAGE",
            "hidden_dim": sage_model.hidden_dim,
            "num_nodes": len(propagation_risk),
            "disclaimer": dataset_bundle.experimental_setup["description"],
        },
    }


def score_graph(
    suppliers: pd.DataFrame,
    relationships: pd.DataFrame | None = None,
    alerts: pd.DataFrame | None = None,
    model_type: str = "graphsage",
    supplier_products: pd.DataFrame | None = None,
    supplier_locations: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Score graph propagation risk for all suppliers (backward compatible API).

    Returns dict mapping supplier_id -> graph_risk_score (0.0 to 100.0).
    """
    if suppliers is None or suppliers.empty:
        return {}

    # Extract features
    feats, edges = build_node_features(
        suppliers=suppliers,
        relationships=relationships,
        alerts=alerts,
        supplier_products=supplier_products,
        supplier_locations=supplier_locations,
    )

    if feats.empty:
        return {}

    cols = [c for c in feats.columns if c != "supplier_id"]
    X_mat = feats[cols].to_numpy(dtype=float)
    n = len(feats)

    # 1. Attempt to load trained weights
    model_file = os.path.join(ROOT, f"models/{model_type.lower()}_model.pt")
    if os.path.exists(model_file):
        try:
            in_dim = len(cols)
            if model_type.lower() == "gat":
                model = GAT(in_dim=in_dim, hidden=16, n_heads=2, dropout=0.0)
            else:
                model = GraphSAGE(in_dim=in_dim, hidden=32, dropout=0.0)

            state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            # Construct adjacency
            adj_mat = np.zeros((n, n), dtype=np.float32)
            for u, v in edges:
                if u < n and v < n:
                    adj_mat[u, v] = 1.0
                    adj_mat[v, u] = 1.0

            x_tensor = torch.tensor(X_mat, dtype=torch.float32)
            adj_tensor = torch.tensor(adj_mat, dtype=torch.float32)

            with torch.no_grad():
                logits = model(x_tensor, adj_tensor)
                probs = torch.sigmoid(logits).numpy()

            return {str(s): round(float(v) * 100.0, 2) for s, v in zip(feats["supplier_id"], probs)}
        except Exception:
            pass

    # 2. Transparent Topology-aware fallback
    scores = topology_risk(X_mat, edges)
    return {str(s): round(float(v) * 100.0, 2) for s, v in zip(feats["supplier_id"], scores)}
