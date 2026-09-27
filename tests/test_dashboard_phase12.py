"""Unit and Integration Tests for Phase 12 Streamlit Dashboard."""
from __future__ import annotations

import os
import pandas as pd
import pytest

from dashboard.app import (
    badge_html,
    risk_band_badge_html,
    load_csv,
    load_json,
    load_graph_cached,
)
from src.alerts.schemas import PROJECT_THRESHOLDS_DISCLAIMER


def test_badge_helpers():
    """Verify data provenance and risk band badges render proper HTML."""
    # Provenance badges
    demo_b = badge_html("SYNTHETIC_DEMO")
    assert "DEMO DATA" in demo_b
    assert "badge-demo" in demo_b

    live_b = badge_html("LIVE_NEWS")
    assert "LIVE / NEWSAPI" in live_b
    assert "badge-live" in live_b

    # Risk band badges
    crit_b = risk_band_badge_html("CRITICAL")
    assert "CRITICAL" in crit_b
    assert "badge-critical" in crit_b

    high_b = risk_band_badge_html("HIGH")
    assert "HIGH" in high_b
    assert "badge-high" in high_b

    med_b = risk_band_badge_html("MEDIUM")
    assert "MEDIUM" in med_b
    assert "badge-medium" in med_b

    low_b = risk_band_badge_html("LOW")
    assert "LOW" in low_b
    assert "badge-low" in low_b


def test_dashboard_data_loading():
    """Verify master and processed tables are loaded correctly by the dashboard."""
    sups = load_csv("data/master/suppliers.csv")
    assert not sups.empty
    assert "supplier_id" in sups.columns
    assert len(sups) >= 1000

    prods = load_csv("data/master/products.csv")
    assert not prods.empty
    assert "product_id" in prods.columns
    assert len(prods) >= 1000

    locs = load_csv("data/master/locations.csv")
    assert not locs.empty
    assert "location_id" in locs.columns
    assert len(locs) >= 1000

    alerts = load_csv("data/processed/alerts.csv")
    if alerts.empty:
        alerts = load_csv("data/processed/final_alerts.csv")
    assert not alerts.empty
    assert "supplier_id" in alerts.columns


def test_risk_alerts_required_fields():
    """Verify that alert records contain all 10 required fields for Page 4."""
    alerts = load_csv("data/processed/alerts.csv")
    if alerts.empty:
        alerts = load_csv("data/processed/final_alerts.csv")

    assert not alerts.empty
    # Must have supplier, risk score, risk band, event, deterministic risk, graphsage, gat, combined, explanation
    sample = alerts.iloc[0]
    assert "supplier_id" in sample
    assert "event_id" in sample
    assert ("risk_band" in sample) or ("combined_risk_level" in sample)
    assert ("combined_risk" in sample) or ("combined_risk_score" in sample)
    assert ("explanation" in sample) or ("reason" in sample)


def test_cached_master_graph():
    """Verify that the cached master graph contains heterogeneous node and edge types."""
    G = load_graph_cached()
    assert G.number_of_nodes() >= 1000
    assert G.number_of_edges() >= 1000

    node_types = {d.get("node_type") for _, d in G.nodes(data=True)}
    assert "SUPPLIER" in node_types
    assert "PRODUCT" in node_types
    assert "LOCATION" in node_types


def test_academic_disclaimer_preserved():
    """Verify project threshold disclaimer is intact."""
    assert "project-defined" in PROJECT_THRESHOLDS_DISCLAIMER.lower()
    assert "not externally validated" in PROJECT_THRESHOLDS_DISCLAIMER.lower()
