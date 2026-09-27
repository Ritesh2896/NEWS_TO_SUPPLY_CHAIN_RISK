"""Deterministic Risk Engine for BDS-35 (Phase 7).

Calculates explainable, deterministic supply-chain disruption risk scores using
configurable domain weights across 6 candidate factors:
  1. event_severity
  2. geographic_exposure
  3. supplier_criticality
  4. product_criticality
  5. dependency_strength
  6. single_source_dependency

Formula:
  deterministic_risk = (
      w1 * event_severity
    + w2 * geographic_exposure
    + w3 * supplier_criticality
    + w4 * product_criticality
    + w5 * dependency_strength
    + w6 * single_source_dependency
  )

Normalizes all factors to [0.0, 1.0].
Distinguishes HEURISTIC RISK from MODEL OUTPUT and GROUND TRUTH.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from src.risk.config import (
    DeterministicRiskConfig,
    DEFAULT_DETERMINISTIC_CONFIG,
    RiskBand,
    RiskClassificationType,
    SEVERITY_MAPPING,
    CRITICALITY_MAPPING,
    DEPENDENCY_MAPPING,
    SINGLE_SOURCE_MAPPING,
)
from src.risk.explanation import (
    ContributingFactor,
    generate_risk_explanation,
    get_level_label,
    TAXONOMY_DISCLAIMER,
)


@dataclass
class DeterministicRiskResult:
    """Complete output of a deterministic risk calculation."""
    risk_score: float                  # Scaled score [0.0, 100.0]
    normalized_score: float            # Normalized score [0.0, 1.0]
    risk_band: str                     # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    risk_type: str                     # "HEURISTIC_RISK"
    contributing_factors: dict[str, dict[str, Any]]
    explanation: str
    reasons: list[str]
    weights_used: dict[str, float]
    disclaimer: str = TAXONOMY_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "risk_score": round(self.risk_score, 2),
            "normalized_score": round(self.normalized_score, 4),
            "risk_band": self.risk_band,
            "risk_type": self.risk_type,
            "contributing_factors": self.contributing_factors,
            "explanation": self.explanation,
            "reasons": self.reasons,
            "weights_used": self.weights_used,
            "disclaimer": self.disclaimer,
        }


def _normalize_factor(
    val: Any,
    mapping: dict[str, float] | None = None,
    default: float = 0.50,
) -> float:
    """Normalize input factor (numeric or categorical) into a strict [0.0, 1.0] float.

    Does NOT use random fallback values; uses documented deterministic mappings.
    """
    if val is None or pd.isna(val):
        return default

    # If boolean
    if isinstance(val, bool):
        return 1.0 if val else 0.0

    # If in categorical/discrete mapping (e.g. severity 1->0.20, 2->0.40, etc.)
    s_val = str(val).strip().lower()
    if mapping and s_val in mapping:
        return mapping[s_val]

    # If numeric
    if isinstance(val, (int, float)):
        f_val = float(val)
        if 0.0 <= f_val <= 1.0:
            return f_val
        if 1.0 < f_val <= 5.0:
            return round(f_val / 5.0, 4)
        if 0.0 <= f_val <= 100.0:
            return round(f_val / 100.0, 4)
        return min(1.0, max(0.0, f_val))

    # Try numeric conversion from string
    try:
        f_val = float(s_val)
        if 0.0 <= f_val <= 1.0:
            return f_val
        if 1.0 < f_val <= 5.0:
            return round(f_val / 5.0, 4)
        if 0.0 <= f_val <= 100.0:
            return round(f_val / 100.0, 4)
    except (ValueError, TypeError):
        pass

    return default


def calculate_deterministic_risk(
    event_severity: Any,
    geographic_exposure: Any,
    supplier_criticality: Any,
    product_criticality: Any,
    dependency_strength: Any,
    single_source_dependency: Any = False,
    config: DeterministicRiskConfig | None = None,
) -> DeterministicRiskResult:
    """Calculate deterministic supply-chain risk from 6 candidate factors.

    All factors are normalized to [0.0, 1.0] and combined via domain-expert weights.

    Args:
        event_severity: Severity of the disruption event (1-5, or "high", etc.).
        geographic_exposure: Physical exposure score [0.0, 1.0] or distance band.
        supplier_criticality: Tier / criticality of supplier ("Tier-1", "high", etc.).
        product_criticality: Importance of the affected product ("essential", "high", etc.).
        dependency_strength: Buyer dependency level on supplier ("sole", "high", etc.).
        single_source_dependency: Boolean or indicator whether supplier is sole/single source.
        config: Optional DeterministicRiskConfig with custom weights and thresholds.

    Returns:
        DeterministicRiskResult containing score, band, contributing factors, and explanation.
    """
    cfg = config or DEFAULT_DETERMINISTIC_CONFIG
    w = cfg.weights

    # 1. Normalize all candidate factors to [0.0, 1.0]
    n_sev = _normalize_factor(event_severity, SEVERITY_MAPPING, default=0.50)
    n_geo = _normalize_factor(geographic_exposure, default=0.50)
    n_s_crit = _normalize_factor(supplier_criticality, CRITICALITY_MAPPING, default=0.40)
    n_p_crit = _normalize_factor(product_criticality, CRITICALITY_MAPPING, default=0.40)
    n_dep = _normalize_factor(dependency_strength, DEPENDENCY_MAPPING, default=0.50)
    n_single = _normalize_factor(single_source_dependency, SINGLE_SOURCE_MAPPING, default=0.0)

    # 2. Compute weighted combination
    normalized_score = (
        w.event_severity * n_sev
        + w.geographic_exposure * n_geo
        + w.supplier_criticality * n_s_crit
        + w.product_criticality * n_p_crit
        + w.dependency_strength * n_dep
        + w.single_source_dependency * n_single
    )
    normalized_score = min(1.0, max(0.0, normalized_score))

    # Scale to [0.0, 100.0]
    score_100 = round(normalized_score * 100.0, 2)

    # 3. Determine risk band
    band = cfg.determine_band(normalized_score)

    # 4. Build detailed contributing factor records
    factors_list = [
        ContributingFactor(
            factor_name="event_severity",
            display_name="Event Severity",
            raw_value=event_severity,
            normalized_value=n_sev,
            weight=w.event_severity,
            weighted_impact=round(w.event_severity * n_sev, 4),
            level_label=get_level_label(n_sev),
            description=f"Disruption severity normalized to {n_sev:.2f} (weight: {w.event_severity:.2f})",
        ),
        ContributingFactor(
            factor_name="geographic_exposure",
            display_name="Geographic Exposure",
            raw_value=geographic_exposure,
            normalized_value=n_geo,
            weight=w.geographic_exposure,
            weighted_impact=round(w.geographic_exposure * n_geo, 4),
            level_label=get_level_label(n_geo),
            description=f"Geographic proximity exposure normalized to {n_geo:.2f} (weight: {w.geographic_exposure:.2f})",
        ),
        ContributingFactor(
            factor_name="supplier_criticality",
            display_name="Supplier Criticality",
            raw_value=supplier_criticality,
            normalized_value=n_s_crit,
            weight=w.supplier_criticality,
            weighted_impact=round(w.supplier_criticality * n_s_crit, 4),
            level_label=get_level_label(n_s_crit),
            description=f"Supplier criticality tier normalized to {n_s_crit:.2f} (weight: {w.supplier_criticality:.2f})",
        ),
        ContributingFactor(
            factor_name="product_criticality",
            display_name="Product Criticality",
            raw_value=product_criticality,
            normalized_value=n_p_crit,
            weight=w.product_criticality,
            weighted_impact=round(w.product_criticality * n_p_crit, 4),
            level_label=get_level_label(n_p_crit),
            description=f"Product importance tier normalized to {n_p_crit:.2f} (weight: {w.product_criticality:.2f})",
        ),
        ContributingFactor(
            factor_name="dependency_strength",
            display_name="Dependency Strength",
            raw_value=dependency_strength,
            normalized_value=n_dep,
            weight=w.dependency_strength,
            weighted_impact=round(w.dependency_strength * n_dep, 4),
            level_label=get_level_label(n_dep),
            description=f"Supply dependency level normalized to {n_dep:.2f} (weight: {w.dependency_strength:.2f})",
        ),
        ContributingFactor(
            factor_name="single_source_dependency",
            display_name="Single-Source Dependency",
            raw_value=single_source_dependency,
            normalized_value=n_single,
            weight=w.single_source_dependency,
            weighted_impact=round(w.single_source_dependency * n_single, 4),
            level_label="critical" if n_single > 0.5 else "low",
            description=f"Sole-sourcing vulnerability normalized to {n_single:.2f} (weight: {w.single_source_dependency:.2f})",
        ),
    ]

    contributing_dict = {f.factor_name: f.to_dict() for f in factors_list}

    # 5. Generate natural-language explainable rationale
    explanation, reasons = generate_risk_explanation(
        factors=factors_list,
        risk_score=score_100,
        risk_band=band,
    )

    return DeterministicRiskResult(
        risk_score=score_100,
        normalized_score=round(normalized_score, 4),
        risk_band=band.value,
        risk_type=RiskClassificationType.HEURISTIC_RISK.value,
        contributing_factors=contributing_dict,
        explanation=explanation,
        reasons=reasons,
        weights_used=w.to_dict(),
        disclaimer=TAXONOMY_DISCLAIMER,
    )


def batch_calculate_deterministic_risk(
    df: pd.DataFrame,
    config: DeterministicRiskConfig | None = None,
) -> pd.DataFrame:
    """Compute deterministic risk across all rows in a DataFrame.

    Maps column names flexibly (e.g. 'severity' -> 'event_severity', etc.).
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "risk_score", "normalized_score", "risk_band", "risk_type",
            "primary_driver", "explanation", "reasons"
        ])

    results: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        sev = row.get("event_severity", row.get("severity"))
        geo = row.get("geographic_exposure", row.get("geo_exposure", row.get("exposure_score")))
        s_crit = row.get("supplier_criticality", row.get("criticality"))
        p_crit = row.get("product_criticality", row.get("criticality"))
        dep = row.get("dependency_strength", row.get("dependency"))
        single = row.get("single_source_dependency", row.get("single_source", False))

        res = calculate_deterministic_risk(
            event_severity=sev,
            geographic_exposure=geo,
            supplier_criticality=s_crit,
            product_criticality=p_crit,
            dependency_strength=dep,
            single_source_dependency=single,
            config=config,
        )

        # Identify top primary driver
        primary_driver = max(
            res.contributing_factors.items(),
            key=lambda item: item[1]["weighted_impact"]
        )[1]["display_name"]

        results.append({
            "risk_score": res.risk_score,
            "normalized_score": res.normalized_score,
            "risk_band": res.risk_band,
            "risk_type": res.risk_type,
            "primary_driver": primary_driver,
            "explanation": res.explanation,
            "reasons": "; ".join(res.reasons),
        })

    out_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), out_df], axis=1)
