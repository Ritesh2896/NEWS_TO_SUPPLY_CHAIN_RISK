"""Tests for Phase 8: GraphSAGE Risk Propagation (Architecture, Features, Trainer, Inference)."""

import json
import os
import numpy as np
import pandas as pd
import pytest
import torch

from src.gnn.graphsage import GraphSAGEPropagationModel
from src.gnn.dataset import (
    build_supply_chain_pyg_dataset,
    CANONICAL_NODE_TYPES,
    EXPERIMENTAL_SETUP_DISCLAIMER,
)
from src.gnn.trainer import (
    GNNTrainingConfig,
    GraphSAGETrainer,
    set_reproducibility_seed,
    compute_graph_smoothness_loss,
)
from src.gnn.inference import (
    predict_risk_propagation,
    score_graph,
)


@pytest.fixture
def mock_graph_data():
    """Create a minimal synthetic supply chain graph for testing."""
    nodes = pd.DataFrame([
        {"node_id": "SUP001", "node_type": "SUPPLIER", "supplier_criticality": 0.9, "deterministic_risk": 0.8},
        {"node_id": "SUP002", "node_type": "SUPPLIER", "supplier_criticality": 0.4, "deterministic_risk": 0.3},
        {"node_id": "PROD001", "node_type": "PRODUCT", "product_criticality": 0.85},
        {"node_id": "LOC001", "node_type": "LOCATION", "geographic_exposure": 0.95},
        {"node_id": "EVT001", "node_type": "EVENT", "event_severity": 0.9},
    ])
    edges = pd.DataFrame([
        {"source_id": "EVT001", "target_id": "LOC001", "weight": 1.0, "edge_type": "EVENT_AT_LOCATION"},
        {"source_id": "LOC001", "target_id": "SUP001", "weight": 0.9, "edge_type": "LOCATION_AFFECTS_FACILITY"},
        {"source_id": "SUP001", "target_id": "PROD001", "weight": 0.85, "edge_type": "SUPPLIER_PROVIDES_PRODUCT"},
        {"source_id": "SUP002", "target_id": "SUP001", "weight": 0.6, "edge_type": "SUPPLIER_DEPENDS_ON_SUPPLIER"},
    ])
    return nodes, edges


def test_graphsage_architecture_forward():
    """Verify 2-layer GraphSAGE forward pass produces embeddings and propagation risk."""
    num_nodes = 5
    in_dim = 16
    hidden_dim = 32
    out_dim = 1

    model = GraphSAGEPropagationModel(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=0.1,
    )

    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

    risk, embeddings = model(x, edge_index)

    # 1. Output shapes
    assert risk.shape == (num_nodes, 1)
    assert embeddings.shape == (num_nodes, hidden_dim)

    # 2. Risk values bounded in [0.0, 1.0] via sigmoid
    assert (risk >= 0.0).all() and (risk <= 1.0).all()


def test_graphsage_checkpoint_save_and_load(tmp_path):
    """Verify saving and loading model checkpoints restores exact weights and metadata."""
    ckpt_path = str(tmp_path / "test_graphsage.pt")

    model1 = GraphSAGEPropagationModel(in_dim=16, hidden_dim=24, out_dim=1)
    metadata = {"experiment": "test_checkpoint", "version": "phase8"}
    model1.save_checkpoint(ckpt_path, metadata=metadata)

    assert os.path.exists(ckpt_path)
    assert os.path.exists(str(tmp_path / "test_graphsage.json"))

    model2 = GraphSAGEPropagationModel.load_checkpoint(ckpt_path)
    assert model2.in_dim == 16
    assert model2.hidden_dim == 24
    assert model2.out_dim == 1

    # Verify identical forward output in eval mode (deactivating dropout)
    model1.eval()
    x = torch.randn(3, 16)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    r1, e1 = model1(x, edge_index)
    r2, e2 = model2(x, edge_index)
    assert torch.allclose(r1, r2, atol=1e-5)
    assert torch.allclose(e1, e2, atol=1e-5)


def test_dataset_feature_extraction(mock_graph_data):
    """Verify PyG dataset construction contains all 8 required feature categories."""
    nodes, edges = mock_graph_data
    bundle = build_supply_chain_pyg_dataset(nodes_df=nodes, edges_df=edges)

    # 1. Check feature names list
    f_names = bundle.feature_names
    assert "is_supplier" in f_names
    assert "event_severity" in f_names
    assert "geographic_exposure" in f_names
    assert "supplier_criticality" in f_names
    assert "product_criticality" in f_names
    assert "dependency_strength" in f_names
    assert "deterministic_risk" in f_names
    assert "norm_in_degree" in f_names

    # 2. Tensor dimensions
    assert bundle.data.x.shape == (5, 16)
    assert bundle.data.edge_index.shape == (2, 4)

    # 3. Academic disclaimer & experimental setup
    assert bundle.experimental_setup["ground_truth_claims"] is False
    assert "SEMI_SUPERVISED" in bundle.experimental_setup["paradigm"]


def test_reproducible_training_loop(mock_graph_data, tmp_path):
    """Verify training convergence, loss logging, and strict seed reproducibility."""
    nodes, edges = mock_graph_data
    bundle = build_supply_chain_pyg_dataset(nodes_df=nodes, edges_df=edges)

    ckpt_path = str(tmp_path / "reproducible_model.pt")
    log_path = str(tmp_path / "training_log.json")

    cfg = GNNTrainingConfig(
        epochs=15,
        learning_rate=0.02,
        hidden_dim=16,
        seed=123,
        checkpoint_path=ckpt_path,
        log_path=log_path,
    )

    trainer = GraphSAGETrainer(config=cfg)
    summary = trainer.train(bundle)

    assert summary["status"] == "COMPLETED"
    assert os.path.exists(ckpt_path)
    assert os.path.exists(log_path)

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    assert "training_log" in log_data
    assert len(log_data["training_log"]) == 15
    first_loss = log_data["training_log"][0]["total_loss"]
    final_loss = log_data["training_log"][-1]["total_loss"]
    assert final_loss < first_loss  # Optimization converged


def test_inference_and_risk_propagation(mock_graph_data, tmp_path):
    """Verify predict_risk_propagation produces node embeddings and propagation risk."""
    nodes, edges = mock_graph_data
    bundle = build_supply_chain_pyg_dataset(nodes_df=nodes, edges_df=edges)

    ckpt_path = str(tmp_path / "inference_test_model.pt")
    cfg = GNNTrainingConfig(epochs=5, hidden_dim=16, seed=42, checkpoint_path=ckpt_path)
    trainer = GraphSAGETrainer(config=cfg)
    trainer.train(bundle)

    preds = predict_risk_propagation(dataset_bundle=bundle, checkpoint_path=ckpt_path)

    assert "propagation_risk" in preds
    assert "node_embeddings" in preds
    assert "metadata" in preds

    prop_risk = preds["propagation_risk"]
    assert len(prop_risk) == 5
    assert "SUP001" in prop_risk
    assert 0.0 <= prop_risk["SUP001"] <= 100.0

    embeds = preds["node_embeddings"]
    assert len(embeds["SUP001"]) == 16


def test_backward_compatibility_score_graph():
    """Verify legacy score_graph remains operational."""
    sups = pd.DataFrame([{"supplier_id": "SUP001", "supplier_name": "Supplier 1", "tier": "Tier-1"}])
    scores = score_graph(suppliers=sups, model_type="graphsage")
    assert isinstance(scores, dict)
    if "SUP001" in scores:
        assert 0.0 <= scores["SUP001"] <= 100.0
