"""Tests for Phase 10: Combined Risk Engine and Explainable Alerts."""

import os
import pandas as pd
import pytest

from src.alerts.schemas import (
    RiskBand,
    CombinedRiskWeights,
    RiskBandThresholds,
    PROJECT_THRESHOLDS_DISCLAIMER,
)
from src.alerts.explanations import (
    extract_alert_reasons,
    format_alert_explanation,
)
from src.alerts.generator import (
    calculate_combined_risk,
    generate_alerts,
)
from src.alerts.alert_engine import create_alert


def test_combined_weights_validation():
    """Verify combination weights enforce non-negativity and summation to 1.0."""
    # Valid default weights
    weights = CombinedRiskWeights(alpha=0.50, beta=0.25, gamma=0.25)
    assert weights.alpha == 0.50
    assert weights.beta == 0.25
    assert weights.gamma == 0.25

    # Invalid weights sum
    with pytest.raises(ValueError, match="Combination weights must sum to 1.0"):
        CombinedRiskWeights(alpha=0.60, beta=0.60, gamma=0.10)

    # Negative weight
    with pytest.raises(ValueError, match="must be non-negative"):
        CombinedRiskWeights(alpha=-0.10, beta=0.60, gamma=0.50)


def test_calculate_combined_risk_formula():
    """Verify combined_risk = alpha*det + beta*sage + gamma*gat."""
    weights = CombinedRiskWeights(alpha=0.50, beta=0.30, gamma=0.20)

    # Test inputs in [0, 1]
    det = 0.80
    sage = 0.70
    gat = 0.60
    # Expected: 0.50 * 0.80 + 0.30 * 0.70 + 0.20 * 0.60 = 0.40 + 0.21 + 0.12 = 0.73
    c_norm, c_100, band = calculate_combined_risk(det, sage, gat, weights=weights)

    assert abs(c_norm - 0.73) < 1e-4
    assert abs(c_100 - 73.0) < 1e-2
    assert band == RiskBand.HIGH

    # Test inputs in 0-100 scale
    c_norm_100, c_100_scale, band_100 = calculate_combined_risk(80.0, 70.0, 60.0, weights=weights)
    assert abs(c_norm_100 - 0.73) < 1e-4
    assert band_100 == RiskBand.HIGH


def test_project_defined_risk_bands():
    """Verify project-defined risk bands: LOW, MEDIUM, HIGH, CRITICAL."""
    thresholds = RiskBandThresholds()

    # LOW: < 0.40
    _, _, b1 = calculate_combined_risk(0.20, 0.20, 0.20, thresholds=thresholds)
    assert b1 == RiskBand.LOW

    # MEDIUM: [0.40, 0.60)
    _, _, b2 = calculate_combined_risk(0.50, 0.50, 0.50, thresholds=thresholds)
    assert b2 == RiskBand.MEDIUM

    # HIGH: [0.60, 0.75)
    _, _, b3 = calculate_combined_risk(0.65, 0.70, 0.60, thresholds=thresholds)
    assert b3 == RiskBand.HIGH

    # CRITICAL: >= 0.75
    _, _, b4 = calculate_combined_risk(0.90, 0.85, 0.80, thresholds=thresholds)
    assert b4 == RiskBand.CRITICAL


def test_alert_reasons_extraction():
    """Verify alert reasons match user-specified high-risk conditions."""
    reasons = extract_alert_reasons(
        event_severity=0.85,          # High-severity disruption
        geographic_exposure=0.80,     # Geographic exposure detected
        dependency_strength=0.90,     # High dependency relationship
        deterministic_risk=0.75,      # Elevated deterministic risk
        graphsage_risk=0.65,          # Graph propagation
        gat_risk=0.60,
    )

    assert "High-severity disruption" in reasons
    assert "Geographic exposure detected" in reasons
    assert "High dependency relationship" in reasons
    assert "Elevated deterministic risk" in reasons
    assert "Graph propagation from affected upstream nodes" in reasons


def test_alert_explanation_canonical_template():
    """Verify alert explanation matches canonical user template."""
    reasons = [
        "High-severity disruption",
        "Geographic exposure detected",
        "High dependency relationship",
        "Elevated deterministic risk",
        "Graph propagation from affected upstream nodes",
    ]

    exp = format_alert_explanation(
        supplier_id="SUP00123",
        supplier_name=None,
        risk_band=RiskBand.HIGH,
        risk_score_100=72.5,
        reasons=reasons,
    )

    # Must contain required header blocks
    assert "Supplier:" in exp
    assert "SUP00123" in exp
    assert "Risk:" in exp
    assert "HIGH" in exp
    assert "Reasons:" in exp
    assert "* High-severity disruption" in exp
    assert "* Geographic exposure detected" in exp
    assert "* High dependency relationship" in exp
    assert "* Elevated deterministic risk" in exp
    assert "* Graph propagation from affected upstream nodes" in exp

    # Must contain the project threshold disclaimer
    assert PROJECT_THRESHOLDS_DISCLAIMER in exp


def test_generate_alerts_and_csv_persistence(tmp_path):
    """Verify batch alert generation and CSV schema persistence."""
    out_csv = str(tmp_path / "alerts.csv")

    risk_records = [
        {
            "supplier_id": "SUP00123",
            "supplier_name": "Apex Microelectronics",
            "event_id": "EVT00001",
            "event_type": "Flood",
            "event_severity": 0.85,
            "geographic_exposure": 0.90,
            "dependency_strength": 0.85,
            "deterministic_risk": 0.80,
            "graphsage_risk": 0.70,
            "gat_risk": 0.65,
        },
        {
            "supplier_id": "SUP00456",
            "supplier_name": "Standard Logistics",
            "event_id": "EVT00002",
            "event_type": "Port Disruption",
            "event_severity": 0.20,
            "geographic_exposure": 0.15,
            "dependency_strength": 0.25,
            "deterministic_risk": 0.25,
            "graphsage_risk": 0.20,
            "gat_risk": 0.20,
        },
    ]

    df_alerts = generate_alerts(
        risk_records=risk_records,
        output_csv_path=out_csv,
        save_csv=True,
    )

    assert len(df_alerts) == 2
    assert os.path.exists(out_csv)

    # Verify column schemas
    expected_cols = [
        "alert_id", "supplier_id", "supplier_name", "event_id", "event_type",
        "deterministic_risk", "graphsage_risk", "gat_risk", "combined_risk",
        "risk_score_100", "risk_band", "reasons", "explanation", "created_at", "data_status"
    ]
    for col in expected_cols:
        assert col in df_alerts.columns, f"Missing column: {col}"

    row_high = df_alerts[df_alerts["supplier_id"] == "SUP00123"].iloc[0]
    assert row_high["risk_band"] in ("HIGH", "CRITICAL")
    assert "High-severity disruption" in row_high["reasons"]
    assert "SUP00123" in row_high["explanation"]

    row_low = df_alerts[df_alerts["supplier_id"] == "SUP00456"].iloc[0]
    assert row_low["risk_band"] == "LOW"


def test_legacy_create_alert_backward_compatibility():
    """Verify legacy create_alert from alert_engine remains fully functional."""
    alert = create_alert(
        event_id="EVT_LEGACY_01",
        supplier_id="SUP00999",
        product_id="PROD0001",
        risk_result={"risk_score": 85.0, "risk_level": "CRITICAL"},
        graph_risk=75.0,
    )
    assert alert["alert_id"] == "ALT-EVT_LEGACY_01"
    assert alert["combined_level"] == "CRITICAL"
    assert alert["combined_risk"] > 70.0
