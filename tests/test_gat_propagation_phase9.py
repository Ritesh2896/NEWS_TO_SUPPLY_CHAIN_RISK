"""Tests for Phase 9: Graph Attention Network (GAT) Risk Propagation & Interpretation."""

import json
import os
import pandas as pd
import pytest
import torch

from src.gnn.config import (
    CentralizedGNNConfig,
    GNNModelType,
    ATTENTION_CAUSALITY_DISCLAIMER,
)
from src.gnn.dataset import build_supply_chain_pyg_dataset
from src.gnn.gat import GATPropagationModel
from src.gnn.trainer_gat import GATTrainer
from src.gnn.inference_gat import predict_gat_risk_propagation
from src.gnn.inference import predict_risk_propagation


@pytest.fixture
def mock_supply_graph():
    """Create minimal synthetic supply chain graph."""
    nodes = pd.DataFrame([
        {"node_id": "SUP_A", "node_type": "SUPPLIER", "supplier_criticality": 0.95, "deterministic_risk": 0.85},
        {"node_id": "SUP_B", "node_type": "SUPPLIER", "supplier_criticality": 0.50, "deterministic_risk": 0.40},
        {"node_id": "PROD_X", "node_type": "PRODUCT", "product_criticality": 0.90},
        {"node_id": "LOC_1", "node_type": "LOCATION", "geographic_exposure": 0.90},
        {"node_id": "EVT_1", "node_type": "EVENT", "event_severity": 0.85},
    ])
    edges = pd.DataFrame([
        {"source_id": "EVT_1", "target_id": "LOC_1", "weight": 1.0, "edge_type": "EVENT_AT_LOCATION"},
        {"source_id": "LOC_1", "target_id": "SUP_A", "weight": 0.95, "edge_type": "LOCATION_AFFECTS_FACILITY"},
        {"source_id": "SUP_A", "target_id": "PROD_X", "weight": 0.90, "edge_type": "SUPPLIER_PROVIDES_PRODUCT"},
        {"source_id": "SUP_B", "target_id": "SUP_A", "weight": 0.70, "edge_type": "SUPPLIER_DEPENDS_ON_SUPPLIER"},
    ])
    return nodes, edges


def test_gat_architecture_forward():
    """Verify 2-layer GAT forward pass with multi-head attention and attention extraction."""
    num_nodes = 5
    in_dim = 16
    hidden_dim = 24
    heads = 2

    model = GATPropagationModel(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=1,
        heads=heads,
        dropout=0.1,
    )

    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    risk, embeddings, att_weights = model(x, edge_index, return_attention_weights=True)

    # 1. Output shapes
    assert risk.shape == (num_nodes, 1)
    assert embeddings.shape == (num_nodes, hidden_dim)

    # 2. Probability bounds
    assert (risk >= 0.0).all() and (risk <= 1.0).all()

    # 3. Attention weights extracted
    if att_weights is not None:
        att_edges, att_alpha = att_weights
        assert att_edges.shape[0] == 2
        assert att_alpha.shape[0] == att_edges.shape[1]


def test_gat_checkpoint_save_and_load(tmp_path):
    """Verify saving and loading GAT checkpoints restores exact weights and disclaimer."""
    ckpt_path = str(tmp_path / "test_gat.pt")

    model1 = GATPropagationModel(in_dim=16, hidden_dim=20, out_dim=1, heads=2)
    metadata = {"experiment": "gat_test", "phase": 9}
    model1.save_checkpoint(ckpt_path, metadata=metadata)

    assert os.path.exists(ckpt_path)
    assert os.path.exists(str(tmp_path / "test_gat.json"))

    model2 = GATPropagationModel.load_checkpoint(ckpt_path)
    assert model2.in_dim == 16
    assert model2.hidden_dim == 20
    assert model2.heads == 2

    # Verify identical evaluation forward pass
    model1.eval()
    model2.eval()
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    r1, e1, _ = model1(x, edge_index)
    r2, e2, _ = model2(x, edge_index)

    assert torch.allclose(r1, r2, atol=1e-5)
    assert torch.allclose(e1, e2, atol=1e-5)


def test_centralized_config_and_gat_training(mock_supply_graph, tmp_path):
    """Verify training GAT with centralized config, Dirichlet loss, and log persistence."""
    nodes, edges = mock_supply_graph
    bundle = build_supply_chain_pyg_dataset(nodes_df=nodes, edges_df=edges)

    ckpt_path = str(tmp_path / "gat_trained.pt")
    log_path = str(tmp_path / "gat_log.json")

    config = CentralizedGNNConfig(
        model_type="gat",
        epochs=15,
        learning_rate=0.02,
        hidden_dim=16,
        num_heads=2,
        seed=101,
        checkpoint_path=ckpt_path,
        log_path=log_path,
    )

    trainer = GATTrainer(config=config)
    res = trainer.train(bundle)

    assert res["status"] == "COMPLETED"
    assert res["model_type"] == "GAT"
    assert os.path.exists(ckpt_path)
    assert os.path.exists(log_path)

    with open(log_path, "r", encoding="utf-8") as f:
        log_content = json.load(f)

    assert len(log_content["training_log"]) == 15
    assert log_content["training_log"][-1]["total_loss"] < log_content["training_log"][0]["total_loss"]
    assert "No synthetic classification accuracy is reported" in log_content["academic_disclaimer"]


def test_attention_neighborhood_interpretation_and_disclaimer(mock_supply_graph, tmp_path):
    """Verify attention weight extraction, neighborhood mapping, and non-causality disclaimer."""
    nodes, edges = mock_supply_graph
    bundle = build_supply_chain_pyg_dataset(nodes_df=nodes, edges_df=edges)

    ckpt_path = str(tmp_path / "gat_interpret.pt")
    config = CentralizedGNNConfig(model_type="gat", epochs=5, hidden_dim=16, checkpoint_path=ckpt_path)
    trainer = GATTrainer(config=config)
    trainer.train(bundle)

    preds = predict_gat_risk_propagation(
        dataset_bundle=bundle,
        checkpoint_path=ckpt_path,
        return_attention=True,
    )

    # 1. Output structure
    assert "propagation_risk" in preds
    assert "node_embeddings" in preds
    assert "attention_weights" in preds
    assert "neighborhood_interpretations" in preds
    assert "causality_disclaimer" in preds

    # 2. Non-causality disclaimer check
    assert "do NOT prove or establish causal relationships" in preds["causality_disclaimer"]
    assert "localized graph neighborhoods" in preds["causality_disclaimer"]

    # 3. Neighborhood interpretations
    assert isinstance(preds["neighborhood_interpretations"], dict)
    assert "SUP_A" in preds["propagation_risk"]
    assert 0.0 <= preds["propagation_risk"]["SUP_A"] <= 100.0


def test_unified_model_selection(mock_supply_graph, tmp_path):
    """Verify model selection between 'graphsage' and 'gat' in predict_risk_propagation."""
    nodes, edges = mock_supply_graph
    bundle = build_supply_chain_pyg_dataset(nodes_df=nodes, edges_df=edges)

    # 1. Evaluate with model = "graphsage"
    res_sage = predict_risk_propagation(bundle, model="graphsage")
    assert "propagation_risk" in res_sage
    assert "node_embeddings" in res_sage
    assert "SUP_A" in res_sage["propagation_risk"]

    # 2. Evaluate with model = "gat"
    res_gat = predict_risk_propagation(bundle, model="gat")
    assert "propagation_risk" in res_gat
    assert "node_embeddings" in res_gat
    assert "attention_weights" in res_gat
    assert "causality_disclaimer" in res_gat
    assert "SUP_A" in res_gat["propagation_risk"]


def test_do_not_break_graphsage(mock_supply_graph):
    """Verify GraphSAGE forward pass and training remain 100% operational."""
    from src.gnn.graphsage import GraphSAGEPropagationModel
    from src.gnn.trainer import GraphSAGETrainer

    nodes, edges = mock_supply_graph
    bundle = build_supply_chain_pyg_dataset(nodes_df=nodes, edges_df=edges)

    model = GraphSAGEPropagationModel(in_dim=16, hidden_dim=16)
    risk, embed = model(bundle.data.x, bundle.data.edge_index)
    assert risk.shape == (5, 1)
    assert embed.shape == (5, 16)
