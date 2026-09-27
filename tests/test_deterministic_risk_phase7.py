"""Tests for Phase 7: Deterministic Risk Engine (Weights, Normalization, Explainability, Taxonomy)."""

import pandas as pd
import pytest

from src.risk.config import (
    RiskClassificationType,
    RiskBand,
    RiskWeightsConfig,
    DeterministicRiskConfig,
    DEFAULT_DETERMINISTIC_CONFIG,
)
from src.risk.explanation import (
    ContributingFactor,
    generate_risk_explanation,
    TAXONOMY_DISCLAIMER,
)
from src.risk.deterministic import (
    calculate_deterministic_risk,
    batch_calculate_deterministic_risk,
    _normalize_factor,
)


def test_weights_sum_validation():
    """Verify weights configuration enforces strict summation to 1.0."""
    # Valid default weights sum to 1.0
    valid_cfg = RiskWeightsConfig(
        event_severity=0.25,
        geographic_exposure=0.20,
        supplier_criticality=0.15,
        product_criticality=0.15,
        dependency_strength=0.15,
        single_source_dependency=0.10,
    )
    assert valid_cfg.event_severity == 0.25

    # Invalid weights sum (< 1.0 or > 1.0) must raise ValueError
    with pytest.raises(ValueError, match="Risk factor weights must sum to 1.0"):
        RiskWeightsConfig(
            event_severity=0.50,
            geographic_exposure=0.50,
            supplier_criticality=0.20,  # Sum = 1.20
            product_criticality=0.0,
            dependency_strength=0.0,
            single_source_dependency=0.0,
        )


def test_factor_normalization_deterministic():
    """Verify all candidate factors normalize into [0.0, 1.0] deterministically without randomness."""
    # Scale 1-5 with severity mapping
    from src.risk.config import SEVERITY_MAPPING, CRITICALITY_MAPPING, DEPENDENCY_MAPPING
    assert _normalize_factor(5, SEVERITY_MAPPING) == 1.00
    assert _normalize_factor(1, SEVERITY_MAPPING) == 0.20
    assert _normalize_factor(3, SEVERITY_MAPPING) == 0.60

    # Categorical strings
    from src.risk.config import SEVERITY_MAPPING, CRITICALITY_MAPPING, DEPENDENCY_MAPPING
    assert _normalize_factor("CRITICAL", SEVERITY_MAPPING) == 1.00
    assert _normalize_factor("high", SEVERITY_MAPPING) == 0.75
    assert _normalize_factor("Tier-1", CRITICALITY_MAPPING) == 1.00
    assert _normalize_factor("sole", DEPENDENCY_MAPPING) == 1.00

    # Booleans / Single source
    from src.risk.config import SINGLE_SOURCE_MAPPING
    assert _normalize_factor(True, SINGLE_SOURCE_MAPPING) == 1.00
    assert _normalize_factor(False, SINGLE_SOURCE_MAPPING) == 0.00
    assert _normalize_factor("single", SINGLE_SOURCE_MAPPING) == 1.00

    # Reproducibility: multiple calls yield exact same values
    v1 = _normalize_factor("severe", SEVERITY_MAPPING)
    v2 = _normalize_factor("severe", SEVERITY_MAPPING)
    assert v1 == v2 == 0.85


def test_deterministic_risk_calculation_and_bands():
    """Verify risk score computation and risk band assignment."""
    # 1. High-risk scenario
    res_high = calculate_deterministic_risk(
        event_severity=5,                    # 1.00 * 0.25 = 0.25
        geographic_exposure=0.90,             # 0.90 * 0.20 = 0.18
        supplier_criticality="critical",      # 1.00 * 0.15 = 0.15
        product_criticality="high",           # 0.85 * 0.15 = 0.1275
        dependency_strength="high",           # 0.85 * 0.15 = 0.1275
        single_source_dependency=True,        # 1.00 * 0.10 = 0.10
    )
    # Total ~ 0.935 -> 93.5/100
    assert res_high.risk_score >= 85.0
    assert res_high.risk_band == RiskBand.CRITICAL.value

    # 2. Low-risk scenario
    res_low = calculate_deterministic_risk(
        event_severity=1,                    # 0.20 * 0.25 = 0.05
        geographic_exposure=0.10,             # 0.10 * 0.20 = 0.02
        supplier_criticality="low",           # 0.25 * 0.15 = 0.0375
        product_criticality="low",            # 0.25 * 0.15 = 0.0375
        dependency_strength="low",            # 0.25 * 0.15 = 0.0375
        single_source_dependency=False,       # 0.00 * 0.10 = 0.00
    )
    # Total ~ 0.1825 -> 18.25/100
    assert res_low.risk_score < 30.0
    assert res_low.risk_band == RiskBand.LOW.value


def test_contributing_factors_and_explanation():
    """Verify explanation clearly lists reasons and top contributing factors."""
    res = calculate_deterministic_risk(
        event_severity="high",
        geographic_exposure=0.95,
        supplier_criticality="high",
        product_criticality="high",
        dependency_strength="high",
        single_source_dependency=False,
    )

    # 1. Contributing factors dict
    assert "event_severity" in res.contributing_factors
    assert "geographic_exposure" in res.contributing_factors
    assert "supplier_criticality" in res.contributing_factors
    assert "product_criticality" in res.contributing_factors
    assert "dependency_strength" in res.contributing_factors
    assert "single_source_dependency" in res.contributing_factors

    factors = res.contributing_factors
    assert factors["event_severity"]["weight"] == 0.25
    assert factors["geographic_exposure"]["weight"] == 0.20

    # 2. Natural language explanation
    assert "Risk increased because:" in res.explanation
    assert len(res.reasons) > 0
    # Must contain key reason strings
    reasons_text = " ".join(res.reasons)
    assert "event severity is high" in reasons_text
    assert "affected location is geographically close" in reasons_text
    assert "supplier has high dependency" in reasons_text
    assert "product criticality is high" in reasons_text


def test_taxonomy_distinction():
    """Verify HEURISTIC RISK is clearly distinguished from MODEL OUTPUT and GROUND TRUTH."""
    res = calculate_deterministic_risk(
        event_severity=4,
        geographic_exposure=0.80,
        supplier_criticality="high",
        product_criticality="medium",
        dependency_strength="high",
        single_source_dependency=True,
    )

    # Risk type must explicitly be HEURISTIC_RISK
    assert res.risk_type == RiskClassificationType.HEURISTIC_RISK.value

    # Disclaimer must explicitly declare it is not ground truth nor model output
    assert "CLASSIFICATION: HEURISTIC_RISK" in res.disclaimer
    assert "NOT an empirical ground-truth label" in res.disclaimer
    assert "machine-learned MODEL OUTPUT" in res.disclaimer
    assert res.disclaimer in res.explanation


def test_batch_calculate_deterministic_risk():
    """Verify batch DataFrame risk computation and output schema."""
    df_input = pd.DataFrame([
        {
            "supplier_id": "SUP001",
            "event_severity": 5,
            "geographic_exposure": 0.85,
            "supplier_criticality": "critical",
            "product_criticality": "high",
            "dependency_strength": "high",
            "single_source_dependency": True,
        },
        {
            "supplier_id": "SUP002",
            "event_severity": 1,
            "geographic_exposure": 0.15,
            "supplier_criticality": "low",
            "product_criticality": "low",
            "dependency_strength": "low",
            "single_source_dependency": False,
        },
    ])

    df_out = batch_calculate_deterministic_risk(df_input)

    assert len(df_out) == 2
    assert "risk_score" in df_out.columns
    assert "normalized_score" in df_out.columns
    assert "risk_band" in df_out.columns
    assert "risk_type" in df_out.columns
    assert "primary_driver" in df_out.columns
    assert "explanation" in df_out.columns
    assert "reasons" in df_out.columns

    # Row 1 is Critical, Row 2 is Low
    assert df_out.iloc[0]["risk_band"] == "CRITICAL"
    assert df_out.iloc[0]["risk_type"] == "HEURISTIC_RISK"
    assert df_out.iloc[1]["risk_band"] == "LOW"
    assert df_out.iloc[1]["risk_type"] == "HEURISTIC_RISK"


def test_custom_weights_configuration():
    """Verify custom weights configuration adjusts scores predictably."""
    # Config where single source has 0.50 weight
    custom_cfg = DeterministicRiskConfig(
        weights=RiskWeightsConfig(
            event_severity=0.10,
            geographic_exposure=0.10,
            supplier_criticality=0.10,
            product_criticality=0.10,
            dependency_strength=0.10,
            single_source_dependency=0.50,
        )
    )

    # When single source is False vs True
    r_no_single = calculate_deterministic_risk(
        event_severity=0, geographic_exposure=0, supplier_criticality=0,
        product_criticality=0, dependency_strength=0, single_source_dependency=False,
        config=custom_cfg,
    )
    r_single = calculate_deterministic_risk(
        event_severity=0, geographic_exposure=0, supplier_criticality=0,
        product_criticality=0, dependency_strength=0, single_source_dependency=True,
        config=custom_cfg,
    )

    assert r_no_single.risk_score == 0.0
    assert r_single.risk_score == 50.0  # 0.50 * 1.0 * 100
