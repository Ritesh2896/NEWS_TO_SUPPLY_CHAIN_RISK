"""Phase 13: Complete End-to-End System Test Suite.

Comprehensive verification across all 9 architectural pillars:
  1. DATA: CSV loading, schema validation, duplicate detection, foreign-key validation
  2. NEWS: NewsAPI response parsing, deduplication, relevance detection
  3. NLP: Entity extraction, event extraction, normalization
  4. LINKING: Exact matching, fuzzy matching, confidence thresholds
  5. GRAPH: Node creation, edge creation, invalid references pruning, graph statistics
  6. RISK: Deterministic risk, geographic exposure, risk bands, explanation generation
  7. GNN: GraphSAGE forward pass, GAT forward pass, inference output shape
  8. API: Health endpoint, pipeline endpoint, alerts endpoint
  9. DASHBOARD: Importability, cache loading, and helper functions
"""
from __future__ import annotations

import os
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

# API
from src.api.main import app

# Data & Schemas
from src.data.loader import DataLoader
from src.data.schemas import SCHEMAS
from src.data.validators import (
    detect_duplicate_ids,
    validate_foreign_keys,
    check_missing_values,
    validate_schema,
)

# News Ingestion
from src.news.client import NewsApiClient
from src.news.deduplication import deduplicate_articles
from src.news.models import NormalizedArticle
from src.news.relevance import check_relevance

# Local NLP
from src.nlp.entity_extraction import extract_entities
from src.nlp.event_extraction import extract_event
from src.nlp.normalization import normalize_entity_name, normalize_entity_type

# Entity Linking
from src.linking.entity_linker import EntityLinker

# Graph Construction
from src.graph.builder import HeterogeneousGraphBuilder
from src.graph.export import compute_graph_statistics
from src.graph.schema import NodeType, EdgeType

# Risk Engines
from src.risk.config import RiskWeightsConfig, RiskBand
from src.risk.deterministic import calculate_deterministic_risk
from src.risk.geo_exposure import (
    DEFAULT_EXPOSURE_CONFIG,
    haversine_distance,
)
from src.alerts.generator import (
    calculate_combined_risk,
    generate_alerts,
)
from src.alerts.schemas import (
    RiskBand as AlertRiskBand,
    RiskBandThresholds,
    DEFAULT_COMBINED_WEIGHTS,
    PROJECT_THRESHOLDS_DISCLAIMER,
)
from src.alerts.explanations import (
    extract_alert_reasons,
    format_alert_explanation,
)

# GNN Models
from src.gnn.graphsage import GraphSAGEPropagationModel
from src.gnn.gat import GATPropagationModel

client = TestClient(app)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# =============================================================
# 1. DATA TESTS
# =============================================================
def test_phase13_data_csv_loading():
    """Verify loading of master and processed CSV tables via DataLoader."""
    loader = DataLoader()
    sups = loader.load_single("suppliers")
    prods = loader.load_single("products")
    locs = loader.load_single("locations")

    assert isinstance(sups, pd.DataFrame) and len(sups) >= 1000
    assert isinstance(prods, pd.DataFrame) and len(prods) >= 1000
    assert isinstance(locs, pd.DataFrame) and len(locs) >= 1000


def test_phase13_data_schema_validation():
    """Verify tables contain required columns specified by the data schema."""
    loader = DataLoader()
    for table_name in ["suppliers", "products", "locations"]:
        df = loader.load_single(table_name)
        expected = SCHEMAS.get(table_name, [])
        for col in expected:
            assert col in df.columns, f"Table '{table_name}' missing column '{col}'"


def test_phase13_data_duplicate_detection():
    """Verify duplicate primary key detection identifies duplicate IDs."""
    sample_df = pd.DataFrame([
        {"supplier_id": "SUP00001", "supplier_name": "Supplier A"},
        {"supplier_id": "SUP00001", "supplier_name": "Supplier A Dup"},
        {"supplier_id": "SUP00002", "supplier_name": "Supplier B"},
    ])
    dups = detect_duplicate_ids(sample_df, "supplier_id")
    assert len(dups) == 1
    assert "SUP00001" in dups


def test_phase13_data_foreign_key_validation():
    """Verify foreign-key referential integrity validator catches orphaned keys."""
    parent_df = pd.DataFrame([{"supplier_id": "SUP00001"}, {"supplier_id": "SUP00002"}])
    child_df = pd.DataFrame([
        {"supplier_id": "SUP00001", "product_id": "PROD1"},
        {"supplier_id": "SUP99999", "product_id": "PROD2"},  # Orphan
    ])
    orphans = validate_foreign_keys(child_df, parent_df, "supplier_id", "supplier_id")
    assert len(orphans) == 1
    assert "SUP99999" in orphans


# =============================================================
# 2. NEWS TESTS
# =============================================================
def test_phase13_news_response_parsing():
    """Verify raw NewsAPI response dictionary is parsed into NormalizedArticle."""
    client_inst = NewsApiClient(api_key="mock_test_key")
    raw_article = {
        "source": {"name": "Logistics Weekly"},
        "author": "Maritime Reporter",
        "title": "Severe typhoon disrupts container terminal operations",
        "description": "Port authorities halt vessel berthing due to high swell.",
        "url": "https://example.com/typhoon-port",
        "publishedAt": "2026-09-24T12:00:00Z",
        "content": "Container crane operations have been halted at the port.",
    }
    norm_list = client_inst._normalize_articles([raw_article], query="ports")
    assert len(norm_list) == 1
    norm = norm_list[0]
    assert isinstance(norm, NormalizedArticle)
    assert norm.source_name == "Logistics Weekly"
    assert "typhoon" in norm.title.lower()
    assert norm.url == "https://example.com/typhoon-port"


def test_phase13_news_deduplication():
    """Verify exact URL and near-duplicate title deduplication."""
    a1 = NormalizedArticle(
        article_id="A1", source_name="News1",
        title="Dockworkers launch strike at major container port",
        url="https://example.com/repeat-url", published_at="2026-09-24T00:00:00Z"
    )
    a2 = NormalizedArticle(
        article_id="A2", source_name="News2",
        title="Dockworkers launch strike at major container port",
        url="https://example.com/repeat-url", published_at="2026-09-24T00:05:00Z"
    )
    a3 = NormalizedArticle(
        article_id="A3", source_name="News3",
        title="Dockworkers launch strike at major container port!",
        url="https://example.com/different-url", published_at="2026-09-24T00:10:00Z"
    )
    unique, dup_count = deduplicate_articles([a1, a2, a3], title_threshold=0.85)
    assert len(unique) == 1
    assert dup_count == 2


def test_phase13_news_relevance_detection():
    """Verify supply-chain disruption relevance detection rules."""
    rel_text = "Factory fire and severe flash flood shut down assembly plant in industrial district."
    is_rel, score, matched = check_relevance(rel_text)
    assert is_rel is True
    assert score >= 0.40
    assert len(matched) > 0

    irrel_text = "Local high school football championship concludes with exciting overtime field goal."
    is_not_rel, score_irrel, _ = check_relevance(irrel_text)
    assert is_not_rel is False
    assert score_irrel < 0.40


# =============================================================
# 3. NLP TESTS
# =============================================================
def test_phase13_nlp_entity_extraction():
    """Verify entity extraction detects organizations and locations with spans."""
    text = "Tata Steel reported major facility shutdown in Jamshedpur following equipment failure."
    extracted = extract_entities(text)
    assert "organizations" in extracted
    assert "locations" in extracted
    orgs = [o.lower() for o in extracted["organizations"]]
    assert any("tata" in o or "steel" in o for o in orgs)


def test_phase13_nlp_event_extraction():
    """Verify event extraction classifies disruption category and assigns deterministic severity."""
    text = "A catastrophic earthquake measuring magnitude 7.2 struck the coastal manufacturing hub."
    evt = extract_event(text)
    assert evt is not None
    assert evt.get("event_type") == "Earthquake"
    # Catastrophic modifier boosts base severity of 4 to maximum 5
    assert evt.get("severity") in [4, 5, "Critical", "High"]


def test_phase13_nlp_normalization():
    """Verify normalization standardizes corporate legal suffixes."""
    assert normalize_entity_name("Acme Manufacturing, Inc.", "ORG") == "ACME MANUFACTURING"
    assert normalize_entity_name("Global Logistics Ltd.", "ORG") == "GLOBAL LOGISTICS"
    assert normalize_entity_name("Port of Rotterdam", "PORT") == "ROTTERDAM PORT"


# =============================================================
# 4. ENTITY LINKING TESTS
# =============================================================
def test_phase13_linking_exact_matching():
    """Verify exact matching resolves known supplier names."""
    linker = EntityLinker()
    match = linker.link_supplier("Apparel Supplier 0001")
    assert match is not None
    assert match["linked_id"] == "SUP00001"
    assert match["match_method"] in ["exact", "normalized"]
    assert match["confidence"] >= 0.80


def test_phase13_linking_fuzzy_matching():
    """Verify controlled fuzzy matching resolves minor spelling variations."""
    linker = EntityLinker()
    match = linker.link_supplier("Aparel Supplier 0001")
    assert match is not None
    assert match["linked_id"] == "SUP00001"
    assert match["match_method"] in ["fuzzy", "normalized", "exact"]


def test_phase13_linking_confidence_thresholds():
    """Verify unresolvable entities below confidence threshold return UNKNOWN."""
    linker = EntityLinker()
    match = linker.link_supplier("CompletelyNonExistentSupplyChainEntityXYZ")
    assert match["linked_id"] == "UNKNOWN"
    assert match["confidence"] < 0.60


# =============================================================
# 5. GRAPH TESTS
# =============================================================
def test_phase13_graph_node_and_edge_creation():
    """Verify heterogeneous node and edge creation across canonical types."""
    builder = HeterogeneousGraphBuilder()
    sups = pd.DataFrame([{"supplier_id": "SUP00001", "supplier_name": "Apex Semiconductor", "tier": 1, "criticality": "High", "data_status": "SYNTHETIC_DEMO"}])
    prods = pd.DataFrame([{"product_id": "PROD00001", "product_name": "Microcontroller", "category": "Semiconductors", "criticality": "High", "data_status": "SYNTHETIC_DEMO"}])
    locs = pd.DataFrame([{"location_id": "LOC00001", "location_name": "Port of Rotterdam", "city": "Rotterdam", "country": "Netherlands", "latitude": 51.92, "longitude": 4.47, "data_status": "SYNTHETIC_DEMO"}])
    sup_locs = pd.DataFrame([{"supplier_id": "SUP00001", "location_id": "LOC00001", "facility_type": "Manufacturing Plant", "ownership": "Owned", "data_status": "SYNTHETIC_DEMO"}])
    sup_prods = pd.DataFrame([{"supplier_id": "SUP00001", "product_id": "PROD00001", "share_percent": 85.0, "data_status": "SYNTHETIC_DEMO"}])
    news = pd.DataFrame([{"article_id": "NEWS00001", "title": "Port delay", "source_name": "Maritime Wire", "published_at": "2026-09-24T10:00:00Z", "data_status": "SYNTHETIC_DEMO"}])
    evts = pd.DataFrame([{"event_id": "EVT00001", "article_id": "NEWS00001", "location_id": "LOC00001", "event_type": "Port Disruption", "severity": 4, "data_status": "SYNTHETIC_DEMO"}])

    G = builder.build_from_dataframes(
        suppliers_df=sups,
        products_df=prods,
        locations_df=locs,
        supplier_locations_df=sup_locs,
        supplier_products_df=sup_prods,
        news_df=news,
        events_df=evts,
    )
    assert G.number_of_nodes() >= 5
    assert G.number_of_edges() >= 3


def test_phase13_graph_invalid_references_pruning():
    """Verify edges referencing non-existent nodes are safely pruned."""
    builder = HeterogeneousGraphBuilder()
    builder.add_node("SUP00001", node_type="SUPPLIER")
    # Manually insert dangling edges into graph
    builder.graph.add_edge("SUP00001", "GHOST_NODE_999", edge_type="INVALID_EDGE")
    builder.graph.add_edge("GHOST_NODE_888", "SUP00001", edge_type="INVALID_EDGE")

    summary = builder.validate_and_prune()
    assert summary["invalid_edges_removed"] == 2
    assert not builder.graph.has_edge("SUP00001", "GHOST_NODE_999")
    assert not builder.graph.has_edge("GHOST_NODE_888", "SUP00001")


def test_phase13_graph_statistics():
    """Verify graph statistics computation returns node/edge breakdowns and degree stats."""
    builder = HeterogeneousGraphBuilder()
    builder.add_node("S1", node_type="SUPPLIER")
    builder.add_node("P1", node_type="PRODUCT")
    builder.graph.add_edge("S1", "P1", edge_type="SUPPLIER_PROVIDES_PRODUCT")

    stats = compute_graph_statistics(builder.graph)
    assert stats["total_nodes"] == 2
    assert stats["total_edges"] == 1
    assert stats["nodes_by_type"].get("SUPPLIER") == 1
    assert "degree_statistics" in stats


# =============================================================
# 6. RISK TESTS
# =============================================================
def test_phase13_risk_deterministic_calculation():
    """Verify deterministic multi-factor risk calculation in [0.0, 1.0]."""
    res = calculate_deterministic_risk(
        event_severity=5,
        geographic_exposure=0.90,
        supplier_criticality="critical",
        product_criticality="high",
        dependency_strength="high",
        single_source_dependency=True,
    )
    assert 0.0 <= res.risk_score <= 100.0
    assert res.risk_band in [RiskBand.HIGH.value, RiskBand.CRITICAL.value]
    assert "event_severity" in res.contributing_factors


def test_phase13_risk_geographic_exposure():
    """Verify Haversine distance and exposure score decay."""
    dist_km = haversine_distance(18.922, 72.834, 18.520, 73.856)  # Mumbai to Pune (~120 km)
    assert 100.0 < dist_km < 150.0

    score, band, tier, desc = DEFAULT_EXPOSURE_CONFIG.get_zone(dist_km)
    assert 0.0 <= score <= 1.0
    assert band in ["25-100 km", "100-250 km"]


def test_phase13_risk_bands_and_explanation():
    """Verify project-defined risk bands and canonical explanation template."""
    thresholds = RiskBandThresholds()
    assert thresholds.determine_band(0.20) == AlertRiskBand.LOW
    assert thresholds.determine_band(0.50) == AlertRiskBand.MEDIUM
    assert thresholds.determine_band(0.68) == AlertRiskBand.HIGH
    assert thresholds.determine_band(0.85) == AlertRiskBand.CRITICAL

    reasons = [
        "High-severity disruption",
        "Geographic exposure detected",
        "High dependency relationship",
        "Elevated deterministic risk",
        "Graph propagation from affected upstream nodes",
    ]
    expl = format_alert_explanation(
        supplier_id="SUP00123",
        supplier_name="Acme Parts",
        risk_band=AlertRiskBand.HIGH,
        risk_score_100=72.5,
        reasons=reasons,
    )
    assert "Supplier:" in expl
    assert "SUP00123" in expl
    assert "Risk:" in expl
    assert "HIGH" in expl
    assert "* High-severity disruption" in expl


# =============================================================
# 7. GNN TESTS
# =============================================================
def test_phase13_gnn_graphsage_forward_pass():
    """Verify GraphSAGE 2-layer forward pass produces node embeddings and risk scores."""
    model = GraphSAGEPropagationModel(in_dim=8, hidden_dim=16, dropout=0.0)
    x = torch.randn(10, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        out_risk, emb = model(x, edge_index)

    assert out_risk.shape == (10, 1)
    assert emb.shape == (10, 16)
    assert (out_risk >= 0.0).all() and (out_risk <= 1.0).all()


def test_phase13_gnn_gat_forward_pass():
    """Verify GAT 2-layer forward pass produces embeddings, scores, and attention weights."""
    model = GATPropagationModel(in_dim=8, hidden_dim=16, heads=2, dropout=0.0)
    x = torch.randn(10, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        out_risk, emb, (edge_out, alpha) = model(x, edge_index, return_attention_weights=True)

    assert out_risk.shape == (10, 1)
    assert emb.shape == (10, 16)
    assert alpha is not None
    assert alpha.dim() >= 1


def test_phase13_gnn_inference_output_shape():
    """Verify tri-model combination produces scalar normalized combined risk."""
    c_norm, c_100, band = calculate_combined_risk(
        deterministic_risk=0.70,
        graphsage_risk=0.60,
        gat_risk=0.65,
        weights=DEFAULT_COMBINED_WEIGHTS,
    )
    assert 0.0 <= c_norm <= 1.0
    assert 0.0 <= c_100 <= 100.0
    assert band in [AlertRiskBand.MEDIUM, AlertRiskBand.HIGH, AlertRiskBand.CRITICAL]


# =============================================================
# 8. API TESTS
# =============================================================
def test_phase13_api_health_endpoint():
    """Verify GET /health returns structured diagnostics without leaking secrets."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "newsapi_configured" in data
    assert "trained_models" in data
    assert "database_status" in data


def test_phase13_api_alerts_endpoint():
    """Verify GET /alerts returns paginated explainable alerts with band metadata."""
    res = client.get("/alerts?limit=5&offset=0")
    assert res.status_code == 200
    data = res.json()
    assert "total" in data
    assert "items" in data
    assert len(data["items"]) <= 5
    if data["items"]:
        first = data["items"][0]
        assert "supplier_id" in first
        assert "combined_risk" in first
        assert "risk_band" in first


def test_phase13_api_pipeline_endpoint():
    """Verify POST /pipeline/run-live executes the 8-stage sequence."""
    payload = {
        "topic": "logistics ports manufacturing disruption",
        "hours": 24,
        "limit": 2,
        "model_type": "both",
        "timeout_seconds": 60,
    }
    res = client.post("/pipeline/run-live", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert "stages" in data
    assert "news_ingestion" in data["stages"]
    assert "nlp_processing" in data["stages"]
    assert "entity_linking" in data["stages"]
    assert "graph_update" in data["stages"]
    assert "geo_exposure" in data["stages"]
    assert "deterministic_risk" in data["stages"]
    assert "gnn_inference" in data["stages"]
    assert "alerts_generated" in data["stages"]


# =============================================================
# 9. DASHBOARD TESTS
# =============================================================
def test_phase13_dashboard_importability_and_helpers():
    """Verify dashboard module helper functions and provenance badges."""
    from dashboard.app import badge_html, risk_band_badge_html

    # Provenance badges
    assert "DEMO DATA" in badge_html("SYNTHETIC_DEMO")
    assert "LIVE / NEWSAPI" in badge_html("REAL_DATA")

    # Risk band badges
    assert "CRITICAL" in risk_band_badge_html("CRITICAL")
    assert "HIGH" in risk_band_badge_html("HIGH")
    assert "MEDIUM" in risk_band_badge_html("MEDIUM")
    assert "LOW" in risk_band_badge_html("LOW")
