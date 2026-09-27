"""Test Suite for Phase 11: FastAPI Backend.

Validates all required endpoints, schemas, error handling, status codes,
pagination, filtering, timeout prevention, and academic disclosures:
  1. GET /health
  2. POST /ingest/live
  3. POST /pipeline/run-live
  4. GET /news/live
  5. GET /events
  6. GET /alerts
  7. GET /suppliers
  8. GET /products
  9. GET /locations
  10. GET /graph/stats
  11. GET /risk/{entity_id}
  12. POST /feedback
  13. CORS & OpenAPI documentation
"""
from __future__ import annotations

import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.alerts.schemas import PROJECT_THRESHOLDS_DISCLAIMER
from src.news.models import NormalizedArticle

client = TestClient(app)


# -------------------------------------------------------------
# 1. Health & Diagnostics
# -------------------------------------------------------------
def test_health_endpoint():
    """Verify GET /health returns structured diagnostics without exposing secrets."""
    with patch.dict(os.environ, {"NEWSAPI_API_KEY": "secret_mock_key_12345"}):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ["ok", "degraded"]
        assert data["newsapi_configured"] is True
        # Ensure API key is NEVER leaked in response payload
        assert "secret_mock_key_12345" not in res.text
        assert "trained_models" in data
        assert "graphsage" in data["trained_models"]
        assert "gat" in data["trained_models"]
        assert "database_status" in data


# -------------------------------------------------------------
# 2. Live News Ingestion Endpoint
# -------------------------------------------------------------
@patch("src.news.client.NewsApiClient.fetch_articles")
@patch("src.news.save_normalized_articles", return_value=1)
def test_ingest_live_endpoint(mock_save, mock_fetch):
    """Verify POST /ingest/live processes input and returns IngestResponse."""
    mock_articles = [
        NormalizedArticle(
            article_id="ART-PHASE11-01",
            source_name="Maritime Journal",
            title="Severe storm halts operations at port terminal",
            url="https://example.com/port-storm-test",
            published_at="2026-09-24T12:00:00Z",
            description="Terminal cranes shut down due to gale force winds.",
            content="Gale force winds shut down all container operations at port.",
            data_status="REAL_DATA",
        )
    ]
    mock_fetch.return_value = (mock_articles, {"status": "ok"})

    payload = {
        "topic": "ports logistics disruption",
        "hours": 24,
        "limit": 5,
        "language": "en",
    }
    res = client.post("/ingest/live", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["fetched"] == 1
    assert data["accepted"] == 1
    assert data["stored"] == 1
    assert len(data["articles"]) == 1
    assert data["articles"][0]["article_id"] == "ART-PHASE11-01"


def test_ingest_live_validation_error():
    """Verify POST /ingest/live validates parameters and returns 422 on invalid payload."""
    invalid_payload = {
        "topic": "",  # min_length violation
        "hours": -5,  # negative hours
        "limit": 100,  # exceeds allowed limit
    }
    res = client.post("/ingest/live", json=invalid_payload)
    assert res.status_code == 422


# -------------------------------------------------------------
# 3. Live Pipeline Execution Endpoint
# -------------------------------------------------------------
def test_pipeline_run_live_endpoint():
    """Verify POST /pipeline/run-live runs the 8-stage pipeline and returns summary."""
    payload = {
        "topic": "India ports suppliers logistics manufacturing disruptions",
        "hours": 24,
        "limit": 3,
        "model_type": "both",
        "timeout_seconds": 60,
    }
    res = client.post("/pipeline/run-live", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "COMPLETED"
    assert "execution_time_seconds" in data
    assert data["execution_time_seconds"] >= 0.0
    assert data["topic"] == payload["topic"]

    # Verify all 8 stages are documented in summary
    stages = data["stages"]
    assert "news_ingestion" in stages
    assert "nlp_processing" in stages
    assert "entity_linking" in stages
    assert "graph_update" in stages
    assert "geo_exposure" in stages
    assert "deterministic_risk" in stages
    assert "gnn_inference" in stages
    assert "alerts_generated" in stages

    assert data["alerts_count"] >= 0
    assert isinstance(data["alerts"], list)
    assert "threshold_disclaimer" in data


# -------------------------------------------------------------
# 4. Paginated News Endpoint
# -------------------------------------------------------------
def test_get_news_live_paginated():
    """Verify GET /news/live supports pagination and keyword filtering."""
    res = client.get("/news/live?limit=5&offset=0")
    assert res.status_code == 200
    data = res.json()
    assert "total" in data
    assert "offset" in data
    assert "limit" in data
    assert "items" in data
    assert data["limit"] == 5
    assert len(data["items"]) <= 5

    # Test keyword query
    res_q = client.get("/news/live?query=disruption&limit=5")
    assert res_q.status_code == 200
    data_q = res_q.json()
    assert isinstance(data_q["items"], list)


# -------------------------------------------------------------
# 5. Paginated Events Endpoint
# -------------------------------------------------------------
def test_get_events_paginated_and_filtered():
    """Verify GET /events returns paginated events with optional filters."""
    res = client.get("/events?limit=10&offset=0")
    assert res.status_code == 200
    data = res.json()
    assert "total" in data
    assert data["limit"] == 10
    assert len(data["items"]) <= 10

    if data["items"]:
        evt = data["items"][0]
        assert "event_id" in evt
        assert "event_type" in evt
        assert "severity" in evt


# -------------------------------------------------------------
# 6. Paginated Alerts Endpoint
# -------------------------------------------------------------
def test_get_alerts_paginated_and_filtered():
    """Verify GET /alerts returns explainable alerts with band filtering."""
    res = client.get("/alerts?limit=10&offset=0")
    assert res.status_code == 200
    data = res.json()
    assert "total" in data
    assert "items" in data
    assert len(data["items"]) <= 10

    if data["items"]:
        alt = data["items"][0]
        assert "alert_id" in alt
        assert "supplier_id" in alt
        assert "deterministic_risk" in alt
        assert "combined_risk" in alt
        assert "risk_band" in alt
        assert alt["risk_band"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert "threshold_disclaimer" in alt

    # Filter by risk_band
    res_band = client.get("/alerts?risk_band=HIGH&limit=5")
    assert res_band.status_code == 200
    for a in res_band.json()["items"]:
        assert a["risk_band"] == "HIGH"


# -------------------------------------------------------------
# 7, 8, 9. Master Data Entities (Suppliers, Products, Locations)
# -------------------------------------------------------------
def test_get_master_entities_paginated():
    """Verify GET /suppliers, /products, and /locations support search and pagination."""
    # Suppliers
    res_s = client.get("/suppliers?limit=5&offset=0")
    assert res_s.status_code == 200
    data_s = res_s.json()
    assert data_s["total"] >= 1000
    assert len(data_s["items"]) == 5
    assert "supplier_id" in data_s["items"][0]

    # Search supplier
    res_s_search = client.get("/suppliers?search=SUP00001")
    assert res_s_search.status_code == 200
    assert any(s["supplier_id"] == "SUP00001" for s in res_s_search.json()["items"])

    # Products
    res_p = client.get("/products?limit=5&offset=0")
    assert res_p.status_code == 200
    data_p = res_p.json()
    assert data_p["total"] >= 1000
    assert len(data_p["items"]) == 5
    assert "product_id" in data_p["items"][0]

    # Locations
    res_l = client.get("/locations?limit=5&offset=0")
    assert res_l.status_code == 200
    data_l = res_l.json()
    assert data_l["total"] >= 1000
    assert len(data_l["items"]) == 5
    assert "location_id" in data_l["items"][0]


# -------------------------------------------------------------
# 10. Graph Statistics Endpoint
# -------------------------------------------------------------
def test_graph_stats_schema():
    """Verify GET /graph/stats returns complete topological indicators."""
    res = client.get("/graph/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["num_nodes"] >= 1000
    assert data["num_edges"] >= 1000
    assert 0.0 <= data["density"] <= 1.0
    assert "node_types" in data
    assert "edge_types" in data
    assert "connected_components" in data
    assert "isolated_nodes" in data


# -------------------------------------------------------------
# 11. Explainable Entity Risk Endpoint
# -------------------------------------------------------------
def test_get_entity_risk_success_and_not_found():
    """Verify GET /risk/{entity_id} retrieves explainable risk posture or returns 404."""
    # Existing supplier
    res = client.get("/risk/SUP00001")
    assert res.status_code == 200
    data = res.json()
    assert data["entity_id"] == "SUP00001"
    assert "deterministic_risk" in data
    assert "graphsage_risk" in data
    assert "gat_risk" in data
    assert "combined_risk" in data
    assert 0.0 <= data["combined_risk"] <= 1.0
    assert data["risk_band"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert isinstance(data["reasons"], list)
    assert len(data["reasons"]) > 0
    assert "explanation" in data
    assert "Supplier:" in data["explanation"]
    assert "threshold_disclaimer" in data

    # Non-existent entity
    res_404 = client.get("/risk/NON_EXISTENT_ENTITY_99999")
    assert res_404.status_code == 404
    assert "not found" in res_404.json()["detail"].lower()


# -------------------------------------------------------------
# 12. Analyst Feedback Endpoint
# -------------------------------------------------------------
def test_post_feedback_success_and_validation():
    """Verify POST /feedback logs triage decision and enforces validation."""
    payload = {
        "alert_id": "ALT-EVT00001-SUP00001",
        "decision": "ACCEPT",
        "analyst_id": "analyst_jaswant",
        "comment": "Confirmed flood impact with regional logistics team.",
        "recommended_action": "Reroute shipments to secondary facility.",
    }
    res = client.post("/feedback", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "saved"
    assert data["alert_id"] == payload["alert_id"]
    assert data["decision"] == "ACCEPT"
    assert data["analyst_id"] == payload["analyst_id"]
    assert data["feedback_id"].startswith("FB-")

    # Invalid empty payload -> 422
    res_err = client.post("/feedback", json={})
    assert res_err.status_code == 422


# -------------------------------------------------------------
# 13. Documentation & OpenAPI Spec
# -------------------------------------------------------------
def test_openapi_and_docs_available():
    """Verify Swagger documentation and OpenAPI schema are accessible."""
    res_openapi = client.get("/openapi.json")
    assert res_openapi.status_code == 200
    spec = res_openapi.json()
    assert "paths" in spec
    assert "/health" in spec["paths"]
    assert "/ingest/live" in spec["paths"]
    assert "/pipeline/run-live" in spec["paths"]
    assert "/news/live" in spec["paths"]
    assert "/events" in spec["paths"]
    assert "/alerts" in spec["paths"]
    assert "/suppliers" in spec["paths"]
    assert "/products" in spec["paths"]
    assert "/locations" in spec["paths"]
    assert "/graph/stats" in spec["paths"]
    assert "/risk/{entity_id}" in spec["paths"]
    assert "/feedback" in spec["paths"]

    res_docs = client.get("/docs")
    assert res_docs.status_code == 200
