"""Unit tests for Phase 5: Graph Construction.

Tests:
  - Heterogeneous node types (NEWS, EVENT, SUPPLIER, PRODUCT, LOCATION, FACILITY)
  - Canonical edge types (7 types)
  - Node ID validation and dangling edge pruning
  - Edge weights and provenance preservation
  - Graph statistics computation (node counts, edge counts, degree stats, components, isolated nodes)
  - PyTorch Geometric HeteroData conversion
  - CSV export schema compliance
"""
from __future__ import annotations
import os
import networkx as nx
import pandas as pd
import pytest
import torch

from src.graph.builder import HeterogeneousGraphBuilder
from src.graph.export import compute_graph_statistics, export_graph_to_csv
from src.graph.features import extract_node_feature_vector, to_pyg_heterodata
from src.graph.schema import (
    NodeType,
    EdgeType,
    NODE_COLUMNS,
    EDGE_COLUMNS,
)


@pytest.fixture
def sample_graph_data():
    """Create sample DataFrames across all relational entity types."""
    suppliers_df = pd.DataFrame([
        {"supplier_id": "SUP00001", "supplier_name": "Apex Semiconductor", "tier": 1, "criticality": "High", "data_status": "SYNTHETIC_DEMO"},
        {"supplier_id": "SUP00002", "supplier_name": "Global Logistics", "tier": 2, "criticality": "Medium", "data_status": "SYNTHETIC_DEMO"},
    ])
    products_df = pd.DataFrame([
        {"product_id": "PROD00001", "product_name": "Microcontroller", "category": "Semiconductors", "criticality": "High", "data_status": "SYNTHETIC_DEMO"},
        {"product_id": "PROD00002", "product_name": "Lithium Battery", "category": "Energy", "criticality": "Medium", "data_status": "SYNTHETIC_DEMO"},
    ])
    locations_df = pd.DataFrame([
        {"location_id": "LOC00001", "location_name": "Port of Rotterdam", "city": "Rotterdam", "country": "Netherlands", "latitude": 51.92, "longitude": 4.47, "data_status": "SYNTHETIC_DEMO"},
    ])
    supplier_locations_df = pd.DataFrame([
        {"supplier_id": "SUP00001", "location_id": "LOC00001", "facility_type": "Manufacturing Plant", "ownership": "Owned", "data_status": "SYNTHETIC_DEMO"},
    ])
    supplier_products_df = pd.DataFrame([
        {"supplier_id": "SUP00001", "product_id": "PROD00001", "share_percent": 85.0, "data_status": "SYNTHETIC_DEMO"},
    ])
    news_df = pd.DataFrame([
        {"article_id": "NEWS00001", "title": "Port of Rotterdam faces logistics delays", "source_name": "Maritime Wire", "published_at": "2026-09-24T10:00:00Z", "data_status": "SYNTHETIC_DEMO"},
    ])
    events_df = pd.DataFrame([
        {"event_id": "EVT00001", "article_id": "NEWS00001", "location_id": "LOC00001", "event_type": "Port Disruption", "severity": 4, "data_status": "SYNTHETIC_DEMO"},
    ])

    return {
        "suppliers_df": suppliers_df,
        "products_df": products_df,
        "locations_df": locations_df,
        "supplier_locations_df": supplier_locations_df,
        "supplier_products_df": supplier_products_df,
        "news_df": news_df,
        "events_df": events_df,
    }


def test_heterogeneous_node_and_edge_types(sample_graph_data):
    """Verify construction of all 6 node types and canonical edge types."""
    builder = HeterogeneousGraphBuilder()
    G = builder.build_from_dataframes(**sample_graph_data)

    node_types = {d.get("node_type") for _, d in G.nodes(data=True)}
    expected_node_types = {
        NodeType.NEWS.value,
        NodeType.EVENT.value,
        NodeType.SUPPLIER.value,
        NodeType.PRODUCT.value,
        NodeType.LOCATION.value,
        NodeType.FACILITY.value,
    }
    assert expected_node_types.issubset(node_types)

    edge_types = {d.get("edge_type") for _, _, d in G.edges(data=True)}
    assert EdgeType.NEWS_HAS_EVENT.value in edge_types
    assert EdgeType.EVENT_AT_LOCATION.value in edge_types
    assert EdgeType.LOCATION_AFFECTS_FACILITY.value in edge_types
    assert EdgeType.FACILITY_BELONGS_TO_SUPPLIER.value in edge_types
    assert EdgeType.SUPPLIER_PROVIDES_PRODUCT.value in edge_types


def test_invalid_edge_pruning(sample_graph_data):
    """Verify that edges pointing to non-existent nodes are automatically detected and pruned."""
    builder = HeterogeneousGraphBuilder()
    G = builder.build_from_dataframes(**sample_graph_data)

    # Intentionally add a dangling edge referencing ghost nodes
    G.add_edge("SUP00001", "GHOST_NODE_999", edge_type="INVALID_EDGE", weight=1.0)
    G.add_edge("GHOST_NODE_888", "PROD00001", edge_type="INVALID_EDGE", weight=1.0)
    assert G.has_edge("SUP00001", "GHOST_NODE_999")

    # Run validation and pruning
    summary = builder.validate_and_prune()
    assert summary["invalid_edges_removed"] == 2
    assert not G.has_edge("SUP00001", "GHOST_NODE_999")
    assert not G.has_edge("GHOST_NODE_888", "PROD00001")


def test_edge_weights_and_provenance(sample_graph_data):
    """Verify edge weights and data provenance (data_status) are preserved."""
    builder = HeterogeneousGraphBuilder()
    G = builder.build_from_dataframes(**sample_graph_data)

    # Check supplier -> product edge
    edge_data = G.get_edge_data("SUP00001", "PROD00001")
    assert edge_data is not None
    assert edge_data["weight"] == 0.85
    assert edge_data["data_status"] == "SYNTHETIC_DEMO"

    # Check node provenance
    sup_node = G.nodes["SUP00001"]
    assert sup_node["data_status"] == "SYNTHETIC_DEMO"


def test_graph_statistics_export(sample_graph_data):
    """Verify graph statistics generation (nodes by type, degrees, components, isolated nodes)."""
    builder = HeterogeneousGraphBuilder()
    G = builder.build_from_dataframes(**sample_graph_data)

    stats = compute_graph_statistics(G)

    # 1. Node count by type
    assert "nodes_by_type" in stats
    assert stats["nodes_by_type"].get(NodeType.SUPPLIER.value, 0) == 2
    assert stats["nodes_by_type"].get(NodeType.LOCATION.value, 0) == 1

    # 2. Edge count by type
    assert "edges_by_type" in stats
    assert len(stats["edges_by_type"]) > 0

    # 3. Degree statistics
    assert "degree_statistics" in stats
    assert "in_degree" in stats["degree_statistics"]
    assert "out_degree" in stats["degree_statistics"]
    assert stats["degree_statistics"]["total_degree"]["max"] >= 1

    # 4. Connected components
    assert "connected_components" in stats
    assert stats["connected_components"]["weakly_connected"] >= 1

    # 5. Isolated nodes
    assert "isolated_nodes" in stats
    assert isinstance(stats["isolated_nodes"]["count"], int)


def test_pyg_heterodata_conversion(sample_graph_data):
    """Verify conversion of NetworkX DiGraph into PyTorch Geometric HeteroData."""
    builder = HeterogeneousGraphBuilder()
    G = builder.build_from_dataframes(**sample_graph_data)

    hetero_data = to_pyg_heterodata(G)

    # Verify node types exist in HeteroData
    assert "SUPPLIER" in hetero_data.node_types
    assert "PRODUCT" in hetero_data.node_types
    assert "LOCATION" in hetero_data.node_types

    # Verify feature tensor shapes
    sup_x = hetero_data["SUPPLIER"].x
    assert isinstance(sup_x, torch.Tensor)
    assert sup_x.shape[0] == 2  # 2 suppliers
    assert sup_x.shape[1] > 0   # features encoded

    # Verify edge relations exist
    edge_types = hetero_data.edge_types
    assert any("SUPPLIER_PROVIDES_PRODUCT" in str(et) for et in edge_types)


def test_csv_export_schema(tmp_path, sample_graph_data):
    """Verify export to nodes.csv and edges.csv matching required column schemas."""
    builder = HeterogeneousGraphBuilder()
    G = builder.build_from_dataframes(**sample_graph_data)

    nodes_file = str(tmp_path / "nodes.csv")
    edges_file = str(tmp_path / "edges.csv")

    n_count, e_count = export_graph_to_csv(G, nodes_path=nodes_file, edges_path=edges_file)
    assert n_count == G.number_of_nodes()
    assert e_count == G.number_of_edges()

    df_nodes = pd.read_csv(nodes_file)
    for col in NODE_COLUMNS:
        assert col in df_nodes.columns

    df_edges = pd.read_csv(edges_file)
    for col in EDGE_COLUMNS:
        assert col in df_edges.columns
