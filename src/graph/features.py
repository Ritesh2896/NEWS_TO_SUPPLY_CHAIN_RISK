"""Feature engineering and PyTorch Geometric HeteroData encoding for heterogeneous supply-chain graph."""
from __future__ import annotations
from typing import Any
import numpy as np
import networkx as nx
import torch

try:
    from torch_geometric.data import HeteroData
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

from src.graph.schema import NodeType, EdgeType, CANONICAL_EDGE_CONSTRAINTS


def extract_node_feature_vector(node_id: str, node_data: dict[str, Any], G: nx.DiGraph) -> list[float]:
    """Convert heterogeneous node attributes into numerical feature vectors."""
    nt = node_data.get("node_type", "UNKNOWN")
    in_deg = float(G.in_degree(node_id))
    out_deg = float(G.out_degree(node_id))

    # Base structural features for all nodes
    feat = [in_deg, out_deg]

    if nt == NodeType.SUPPLIER.value:
        tier = float(node_data.get("tier", 1))
        crit_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Critical": 4.0}
        crit = crit_map.get(node_data.get("criticality", "Medium"), 2.0)
        feat.extend([tier, crit, 1.0, 0.0, 0.0, 0.0])
    elif nt == NodeType.PRODUCT.value:
        crit_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Critical": 4.0}
        crit = crit_map.get(node_data.get("criticality", "Medium"), 2.0)
        feat.extend([crit, 0.0, 0.0, 1.0, 0.0, 0.0])
    elif nt == NodeType.LOCATION.value:
        lat = float(node_data.get("latitude", 0.0)) / 90.0
        lon = float(node_data.get("longitude", 0.0)) / 180.0
        feat.extend([lat, lon, 0.0, 0.0, 1.0, 0.0])
    elif nt == NodeType.EVENT.value:
        sev = float(node_data.get("severity", 3.0)) / 5.0
        feat.extend([sev, 1.0, 0.0, 0.0, 0.0, 1.0])
    elif nt == NodeType.NEWS.value:
        feat.extend([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    elif nt == NodeType.FACILITY.value:
        feat.extend([0.5, 0.5, 0.0, 1.0, 0.0, 0.0])
    else:
        feat.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return feat


def to_pyg_heterodata(G: nx.DiGraph) -> Any:
    """Convert heterogeneous NetworkX DiGraph into PyTorch Geometric HeteroData."""
    if not _HAS_PYG:
        raise ImportError("torch-geometric is required to build HeteroData representation.")

    data = HeteroData()

    # 1. Group nodes by type and build integer index maps
    type_nodes: dict[str, list[str]] = {}
    node_to_idx: dict[str, dict[str, int]] = {}

    for n, d in G.nodes(data=True):
        nt = d.get("node_type", "UNKNOWN")
        type_nodes.setdefault(nt, []).append(n)

    for nt, nodes in type_nodes.items():
        # Sort nodes deterministically
        nodes.sort()
        node_to_idx[nt] = {node_id: idx for idx, node_id in enumerate(nodes)}

        # Build feature matrix
        feats = [extract_node_feature_vector(nid, G.nodes[nid], G) for nid in nodes]
        x_tensor = torch.tensor(feats, dtype=torch.float)
        data[nt].x = x_tensor
        data[nt].node_ids = nodes

    # 2. Group edges by canonical edge relation (source_type, edge_type, target_type)
    edge_groups: dict[tuple[str, str, str], tuple[list[int], list[int], list[float]]] = {}

    for u, v, d in G.edges(data=True):
        u_type = G.nodes[u].get("node_type", "UNKNOWN")
        v_type = G.nodes[v].get("node_type", "UNKNOWN")
        etype = d.get("edge_type", "RELATION")
        w = float(d.get("weight", 1.0))

        if u_type in node_to_idx and v_type in node_to_idx:
            u_idx = node_to_idx[u_type].get(u)
            v_idx = node_to_idx[v_type].get(v)
            if u_idx is not None and v_idx is not None:
                key = (u_type, etype, v_type)
                if key not in edge_groups:
                    edge_groups[key] = ([], [], [])
                edge_groups[key][0].append(u_idx)
                edge_groups[key][1].append(v_idx)
                edge_groups[key][2].append(w)

    for (src_t, rel, dst_t), (src_indices, dst_indices, weights) in edge_groups.items():
        if src_indices:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            edge_weight = torch.tensor(weights, dtype=torch.float)
            data[src_t, rel, dst_t].edge_index = edge_index
            data[src_t, rel, dst_t].edge_weight = edge_weight

    return data
