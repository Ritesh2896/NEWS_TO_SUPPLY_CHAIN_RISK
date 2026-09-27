"""Unit tests for Graph construction and Graph Neural Networks (GraphSAGE & GAT)."""
import torch
import numpy as np
import pandas as pd
import networkx as nx

from src.graph.supply_chain_graph import (
    build_supply_chain_graph,
    compute_graph_metrics,
    get_ego_network,
    propagate_risk,
)
from src.gnn.gnn_model import GraphSAGE, GAT, topology_risk
from src.gnn.feature_engineering import build_node_features


def test_graph_construction():
    sups = pd.DataFrame([{"supplier_id": "SUP001", "supplier_name": "Supplier 1", "tier": "Tier-1"}])
    prods = pd.DataFrame([{"product_id": "PROD001", "product_name": "Product 1"}])
    locs = pd.DataFrame([{"location_id": "LOC001", "location_name": "Location 1"}])
    sp = pd.DataFrame([{"supplier_id": "SUP001", "product_id": "PROD001", "dependency_level": "High"}])
    sl = pd.DataFrame([{"supplier_id": "SUP001", "location_id": "LOC001", "dependency": "Medium"}])

    G = build_supply_chain_graph(
        suppliers_df=sups,
        products_df=prods,
        locations_df=locs,
        supplier_products_df=sp,
        supplier_locations_df=sl,
    )

    metrics = compute_graph_metrics(G)
    assert metrics["num_nodes"] == 3
    assert metrics["num_edges"] >= 3
    assert "SUPPLIER" in metrics["node_types"]

    # Ego network
    ego = get_ego_network(G, "SUP001", radius=1)
    assert len(ego["nodes"]) >= 2


def test_topological_risk_propagation():
    G = nx.DiGraph()
    G.add_edge("LOC1", "SUP1", weight=1.0)
    G.add_edge("SUP1", "PROD1", weight=0.8)

    shocks = {"LOC1": 1.0}
    res = propagate_risk(G, shocks, damping=0.5, max_iter=10)

    assert "SUP1" in res
    assert 0.0 < res["SUP1"] <= 1.0
    assert 0.0 <= res["PROD1"] <= 1.0


def test_graphsage_forward_pass():
    model = GraphSAGE(in_dim=12, hidden=16, dropout=0.0)
    x = torch.randn(5, 12)
    adj = torch.eye(5)
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0

    logits = model(x, adj)
    assert logits.shape == (5,)
    probs = torch.sigmoid(logits)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_gat_forward_pass():
    model = GAT(in_dim=12, hidden=8, n_heads=2, dropout=0.0)
    x = torch.randn(4, 12)
    adj = torch.eye(4)
    adj[0, 2] = 1.0
    adj[2, 0] = 1.0

    logits = model(x, adj)
    assert logits.shape == (4,)
    probs = torch.sigmoid(logits)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_node_feature_engineering():
    sups = pd.DataFrame([
        {"supplier_id": "SUP001", "criticality": "High", "tier": "Tier-1", "single_source": "YES"},
        {"supplier_id": "SUP002", "criticality": "Low", "tier": "Tier-2", "single_source": "NO"},
    ])
    sp = pd.DataFrame([{"supplier_id": "SUP001", "product_id": "PROD001"}])
    feats, edges = build_node_features(sups, supplier_products=sp)

    assert len(feats) == 2
    assert "news_risk" in feats.columns
    assert "single_source_risk" in feats.columns
    assert feats.loc[feats["supplier_id"] == "SUP001", "single_source_risk"].iloc[0] == 1.0
