"""BDS-35 Graph Processing and Construction Package.

Provides:
  - Heterogeneous graph schema and canonical edge types
  - HeterogeneousGraphBuilder for NetworkX graph construction and validation
  - PyTorch Geometric HeteroData encoding for GNNs
  - Topological risk propagation and subgraph extraction
  - Graph statistics calculation and CSV export
"""
from src.graph.schema import (
    NodeType,
    EdgeType,
    CANONICAL_EDGE_CONSTRAINTS,
    NODE_COLUMNS,
    EDGE_COLUMNS,
)
from src.graph.builder import HeterogeneousGraphBuilder
from src.graph.features import extract_node_feature_vector, to_pyg_heterodata
from src.graph.export import compute_graph_statistics, export_graph_to_csv
from src.graph.supply_chain_graph import (
    build_supply_chain_graph,
    compute_graph_metrics,
    get_ego_network,
    propagate_risk,
)

propagate_topological_risk = propagate_risk
extract_ego_subgraph = get_ego_network

__all__ = [
    "NodeType",
    "EdgeType",
    "CANONICAL_EDGE_CONSTRAINTS",
    "NODE_COLUMNS",
    "EDGE_COLUMNS",
    "HeterogeneousGraphBuilder",
    "extract_node_feature_vector",
    "to_pyg_heterodata",
    "compute_graph_statistics",
    "export_graph_to_csv",
    "build_supply_chain_graph",
    "propagate_risk",
    "propagate_topological_risk",
    "get_ego_network",
    "extract_ego_subgraph",
    "compute_graph_metrics",
]
