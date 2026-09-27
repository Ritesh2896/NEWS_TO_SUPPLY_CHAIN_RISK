"""Supply-Chain Graph Dataset Loader & Feature Extractor for GNNs (Phase 8).

Builds PyTorch Geometric Data representations with academic integrity:
- Encodes all required input features:
    1. node_type_encoding (One-hot across SUPPLIER, PRODUCT, LOCATION, FACILITY, EVENT, NEWS)
    2. event_severity [0.0, 1.0]
    3. geographic_exposure [0.0, 1.0]
    4. supplier_criticality [0.0, 1.0]
    5. product_criticality [0.0, 1.0]
    6. dependency_strength [0.0, 1.0]
    7. deterministic_risk [0.0, 1.0]
    8. graph_structural_features (in_degree, out_degree, total_degree, local_centrality)
- Explicit experimental setup declaration:
    Never fabricates ground-truth disruption labels. Uses semi-supervised heuristic
    seed signals (from Phase 7 Deterministic Risk) or self-supervised structural loss.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import Data
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    # Lightweight Data class fallback if PyG data structure is not present
    class Data:  # type: ignore
        def __init__(self, x=None, edge_index=None, edge_weight=None, **kwargs):
            self.x = x
            self.edge_index = edge_index
            self.edge_weight = edge_weight
            for k, v in kwargs.items():
                setattr(self, k, v)


# Standard canonical node type ordering for consistent one-hot encoding
CANONICAL_NODE_TYPES: list[str] = [
    "SUPPLIER",
    "PRODUCT",
    "LOCATION",
    "FACILITY",
    "EVENT",
    "NEWS",
]

EXPERIMENTAL_SETUP_DISCLAIMER: dict[str, Any] = {
    "paradigm": "SEMI_SUPERVISED_RISK_PROPAGATION_DEMO",
    "ground_truth_claims": False,
    "description": (
        "Supervised empirical ground-truth disruption labels are not fabricated. "
        "The training regime uses deterministic heuristic risk signals from Phase 7 "
        "as initial seed targets, combined with self-supervised graph Laplacian smoothness "
        "regularization across supply dependency edges."
    ),
}


@dataclass
class GraphDatasetBundle:
    """Contains PyG Data object and node mapping metadata."""
    data: Data
    node_id_to_idx: dict[str, int]
    idx_to_node_id: dict[int, str]
    node_types: list[str]
    feature_names: list[str]
    experimental_setup: dict[str, Any]


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert any value to float, converting None/NaN/Inf to deterministic default."""
    if val is None or pd.isna(val):
        return default
    try:
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else f
    except (ValueError, TypeError):
        return default


def build_supply_chain_pyg_dataset(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
    risk_labels_df: pd.DataFrame | None = None,
) -> GraphDatasetBundle:
    """Build a PyTorch Geometric Data object from supply-chain graph tables.

    Args:
        nodes_df: Nodes DataFrame with columns: node_id, node_type, name, [features...].
        edges_df: Edges DataFrame with columns: source_id, target_id, edge_type, weight, [provenance].
        events_df: Optional processed events table for event severity annotations.
        risk_labels_df: Optional baseline risk labels (heuristic seed targets).

    Returns:
        GraphDatasetBundle containing PyG Data, node index maps, and feature descriptions.
    """
    if nodes_df.empty or edges_df.empty:
        raise ValueError("Cannot construct graph dataset from empty nodes or edges DataFrame.")

    # 1. Map node IDs to contiguous integer indices [0 .. N-1]
    node_ids = nodes_df["node_id"].astype(str).tolist()
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    idx_to_node_id = {idx: nid for idx, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    # 2. Compute graph structural features (in_degree, out_degree, total_degree)
    in_degrees = np.zeros(num_nodes, dtype=np.float32)
    out_degrees = np.zeros(num_nodes, dtype=np.float32)

    src_list: list[int] = []
    dst_list: list[int] = []
    weights_list: list[float] = []

    for _, e_row in edges_df.iterrows():
        u = str(e_row["source_id"]).strip()
        v = str(e_row["target_id"]).strip()
        if u in node_id_to_idx and v in node_id_to_idx:
            u_idx = node_id_to_idx[u]
            v_idx = node_id_to_idx[v]
            src_list.append(u_idx)
            dst_list.append(v_idx)

            w = float(e_row.get("weight", 1.0))
            weights_list.append(w)

            out_degrees[u_idx] += 1.0
            in_degrees[v_idx] += 1.0

    total_degrees = in_degrees + out_degrees
    max_deg = max(float(total_degrees.max()), 1.0)
    norm_in_deg = in_degrees / max_deg
    norm_out_deg = out_degrees / max_deg
    norm_total_deg = total_degrees / max_deg

    # Edge index tensor [2, E]
    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weight = torch.tensor(weights_list, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)

    # 3. Build node features matrix [N, D]
    feature_names: list[str] = [
        # One-hot node type encoding (6 dimensions)
        "is_supplier", "is_product", "is_location", "is_facility", "is_event", "is_news",
        # Disruption & physical exposure features (2 dimensions)
        "event_severity",
        "geographic_exposure",
        # Supply chain criticality features (2 dimensions)
        "supplier_criticality",
        "product_criticality",
        # Dependency & vulnerability features (2 dimensions)
        "dependency_strength",
        "single_source_dependency",
        # Baseline deterministic risk (1 dimension)
        "deterministic_risk",
        # Structural graph features (3 dimensions)
        "norm_in_degree",
        "norm_out_degree",
        "norm_total_degree",
    ]
    num_features = len(feature_names)
    X = np.zeros((num_nodes, num_features), dtype=np.float32)

    # Pre-index optional risk labels / heuristic scores for seed nodes
    seed_risks: dict[str, float] = {}
    if risk_labels_df is not None and not risk_labels_df.empty:
        id_col = "supplier_id" if "supplier_id" in risk_labels_df.columns else "entity_id"
        score_col = "deterministic_risk" if "deterministic_risk" in risk_labels_df.columns else "risk_score"
        if id_col in risk_labels_df.columns and score_col in risk_labels_df.columns:
            for _, r in risk_labels_df.iterrows():
                try:
                    s_val = float(r[score_col])
                    # Normalize to [0, 1] if on 0-100 scale
                    seed_risks[str(r[id_col])] = s_val / 100.0 if s_val > 1.0 else s_val
                except (ValueError, TypeError):
                    continue

    node_types: list[str] = []

    for idx, nid in enumerate(node_ids):
        n_row = nodes_df.iloc[idx]
        ntype = str(n_row.get("node_type", "SUPPLIER")).upper().strip()
        node_types.append(ntype)

        # 1. One-hot node type encoding
        for t_idx, c_type in enumerate(CANONICAL_NODE_TYPES):
            if ntype == c_type:
                X[idx, t_idx] = 1.0
                break

        # 2. Extract domain attributes safely (guards against NaNs)
        raw_sev = n_row.get("event_severity") if pd.notna(n_row.get("event_severity")) else n_row.get("severity")
        sev = _safe_float(raw_sev, 0.0)
        X[idx, 6] = sev / 5.0 if sev > 1.0 else sev  # event_severity

        raw_geo = n_row.get("geographic_exposure") if pd.notna(n_row.get("geographic_exposure")) else n_row.get("exposure_score")
        X[idx, 7] = _safe_float(raw_geo, 0.20)         # geographic_exposure

        X[idx, 8] = _safe_float(n_row.get("supplier_criticality"), 0.50 if ntype == "SUPPLIER" else 0.0)
        X[idx, 9] = _safe_float(n_row.get("product_criticality"), 0.50 if ntype == "PRODUCT" else 0.0)
        X[idx, 10] = _safe_float(n_row.get("dependency_strength"), 0.50)

        single = 1.0 if str(n_row.get("single_source", "FALSE")).upper() in ("TRUE", "YES", "1") else 0.0
        X[idx, 11] = single                           # single_source_dependency

        # Deterministic risk feature
        det_risk = seed_risks.get(nid, _safe_float(n_row.get("deterministic_risk"), 0.30))
        X[idx, 12] = det_risk

        # Structural features
        X[idx, 13] = norm_in_deg[idx]
        X[idx, 14] = norm_out_deg[idx]
        X[idx, 15] = norm_total_deg[idx]

    x_tensor = torch.tensor(X, dtype=torch.float32)

    # Seed risk tensor used for demonstration / semi-supervised training targets
    seed_targets = torch.tensor(
        [seed_risks.get(nid, 0.0) for nid in node_ids],
        dtype=torch.float32
    ).unsqueeze(-1)

    # Build PyG Data
    pyg_data = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=seed_targets,
        num_nodes=num_nodes,
    )

    return GraphDatasetBundle(
        data=pyg_data,
        node_id_to_idx=node_id_to_idx,
        idx_to_node_id=idx_to_node_id,
        node_types=node_types,
        feature_names=feature_names,
        experimental_setup=EXPERIMENTAL_SETUP_DISCLAIMER,
    )
