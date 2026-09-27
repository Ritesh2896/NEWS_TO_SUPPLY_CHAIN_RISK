"""End-to-End integration and FastAPI endpoint tests for BDS-35."""
import os
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app
from src.linking.entity_linking import link_supplier, link_location, link_product
from src.pipeline.end_to_end_pipeline import run_pipeline
from src.risk.geospatial_exposure import calculate_geographic_exposure, haversine_km
from src.risk.risk_engine import calculate_risk

client = TestClient(app)


def test_entity_linking():
    sups = pd.DataFrame([{"supplier_id": "SUP00001", "supplier_name": "Apparel Supplier 0001"}])
    locs = pd.DataFrame([{"location_id": "LOC00001", "location_name": "Mumbai Industrial Location 0001", "city": "Mumbai", "country": "India"}])
    prods = pd.DataFrame([{"product_id": "PROD00001", "product_name": "Cotton knitwear 0001", "category": "Apparel"}])

    s_link = link_supplier(["Apparel Supplier 0001"], sups)
    assert s_link is not None
    assert s_link["supplier_id"] == "SUP00001"

    l_link = link_location(["Mumbai"], locs)
    assert l_link is not None
    assert l_link["location_id"] == "LOC00001"

    p_link = link_product(["Cotton knitwear"], prods)
    assert p_link is not None
    assert p_link["product_id"] == "PROD00001"


def test_haversine_and_exposure():
    # Mumbai to Pune (~120 km)
    d = haversine_km(18.922, 72.834, 18.520, 73.856)
    assert 100 < d < 150
    score, dist = calculate_geographic_exposure(18.922, 72.834, 18.520, 73.856)
    assert 0.0 <= score <= 1.0


def test_deterministic_risk_engine():
    res = calculate_risk(
        severity="High",
        dependency="High",
        criticality="High",
        geographic_exposure=1.0,
        single_source=True,
    )
    assert res["risk_score"] == 97.5  # 92.5 + 5.0 single-source penalty
    assert res["risk_level"] == "CRITICAL"
    assert "components" in res
    assert res["components"]["single_source_penalty"] == 5.0


def test_pipeline_execution():
    test_article = [{
        "news_id": "TEST-INT-01",
        "title": "Severe flooding disrupts operations at Apparel Supplier 0001 in Mumbai",
        "content": "Heavy flood waters entered facilities of Apparel Supplier 0001, causing immediate factory shutdown.",
        "source": "Global Supply Wire",
        "url": "https://example.com/flood-mumbai",
        "published_at": "2026-09-23T12:00:00Z",
        "data_status": "SYNTHETIC_DEMO",
    }]
    alerts_df = run_pipeline(test_article, model_type="graphsage")
    assert not alerts_df.empty
    top_alert = alerts_df.iloc[0]
    assert top_alert["supplier_id"] == "SUP00001"
    assert top_alert["combined_risk_score"] > 0
    assert top_alert["combined_risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_fastapi_endpoints():
    # Health check
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Root info
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["project"].startswith("BDS-35")

    # Graph stats
    r = client.get("/graph/stats")
    assert r.status_code == 200
    assert r.json()["num_nodes"] >= 1000

    # Ego network
    r = client.get("/graph/ego/SUP00001")
    assert r.status_code == 200
    assert "nodes" in r.json()

    # Models registry
    r = client.get("/models")
    assert r.status_code == 200
    assert "graphsage" in r.json()
    assert "gat" in r.json()
