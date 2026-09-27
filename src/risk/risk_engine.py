"""Deterministic Risk Engine for BDS-35.

Calculates an explainable, non-random risk score combining:
  1. Event Severity (S)
  2. Dependency Level (D)
  3. Supplier / Product Criticality (C)
  4. Geographic Exposure (G)
  5. Single-Source Dependency Penalty (optional)

Formula:
  Score = (w_sev * S + w_dep * D + w_crit * C + w_geo * G) * 100 + penalty_single_source
Normalized to [0.0, 100.0] with explainable component breakdowns.
"""
from __future__ import annotations
from typing import Any

# Standardized factor mapping dictionaries
SEVERITY_SCORE = {
    "1": 0.20, "2": 0.40, "3": 0.60, "4": 0.80, "5": 1.00,
    "low": 0.25, "Low": 0.25, "LOW": 0.25,
    "medium": 0.50, "Medium": 0.50, "MEDIUM": 0.50,
    "high": 0.75, "High": 0.75, "HIGH": 0.75,
    "critical": 1.00, "Critical": 1.00, "CRITICAL": 1.00,
}

DEPENDENCY_SCORE = {
    "low": 0.25, "Low": 0.25, "LOW": 0.25,
    "medium": 0.50, "Medium": 0.50, "MEDIUM": 0.50,
    "high": 1.00, "High": 1.00, "HIGH": 1.00,
    "critical": 1.00, "Critical": 1.00, "CRITICAL": 1.00,
}

CRITICALITY_SCORE = {
    "unknown": 0.40, "UNKNOWN": 0.40,
    "low": 0.25, "Low": 0.25, "LOW": 0.25,
    "medium": 0.50, "Medium": 0.50, "MEDIUM": 0.50,
    "high": 1.00, "High": 1.00, "HIGH": 1.00,
    "critical": 1.00, "Critical": 1.00, "CRITICAL": 1.00,
}

# Default explainable weights summing to 1.0
DEFAULT_WEIGHTS = {
    "severity": 0.30,
    "dependency": 0.25,
    "criticality": 0.25,
    "geographic_exposure": 0.20,
}


def _parse_val(val: Any, mapping: dict[str, float], default: float = 0.50) -> float:
    """Parse numeric or categorical factor into normalized [0, 1] float."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        # If already normalized [0, 1]
        if 0.0 <= val <= 1.0:
            return float(val)
        # If 1-5 scale
        if 1.0 <= val <= 5.0:
            return float(val) / 5.0
        return float(min(1.0, max(0.0, val / 100.0)))
    val_str = str(val).strip()
    return mapping.get(val_str, mapping.get(val_str.lower(), default))


def calculate_risk(
    severity: Any = "Medium",
    dependency: Any = "Medium",
    criticality: Any = "Medium",
    geographic_exposure: float | int = 0.50,
    single_source: bool | str = False,
    weights: dict[str, float] | None = None,
    product_criticality: Any = None,
) -> dict[str, Any]:
    """Calculate deterministic risk score and return explainable component breakdown.

    Compatible with BDS-35 benchmark tests and adaptable for single-source penalty.
    """
    w = DEFAULT_WEIGHTS.copy()
    if weights:
        w.update(weights)

    s = _parse_val(severity, SEVERITY_SCORE, 0.50)
    d = _parse_val(dependency, DEPENDENCY_SCORE, 0.50)

    # Use product criticality if explicitly provided, else general criticality
    if product_criticality is not None:
        c_prod = _parse_val(product_criticality, CRITICALITY_SCORE, 0.50)
        c_sup = _parse_val(criticality, CRITICALITY_SCORE, 0.50)
        c = 0.5 * c_prod + 0.5 * c_sup
    else:
        c = _parse_val(criticality, CRITICALITY_SCORE, 0.50)

    try:
        g = float(geographic_exposure)
        g = max(0.0, min(1.0, g))
    except (ValueError, TypeError):
        g = 0.50

    # Base weighted sum
    base_score = (
        w["severity"] * s +
        w["dependency"] * d +
        w["criticality"] * c +
        w["geographic_exposure"] * g
    ) * 100.0

    # Single-source penalty (5% risk bump if supplier has no backup)
    is_single = False
    if isinstance(single_source, bool):
        is_single = single_source
    elif isinstance(single_source, str):
        is_single = single_source.upper() in {"YES", "TRUE", "1"}

    single_source_penalty = 5.0 if is_single else 0.0
    final_score = round(min(100.0, base_score + single_source_penalty), 2)

    # Risk level thresholding
    if final_score >= 80.0:
        level = "CRITICAL"
    elif final_score >= 60.0:
        level = "HIGH"
    elif final_score >= 40.0:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Human-readable explainable attribution
    primary_driver = max(
        [
            ("Event Severity", w["severity"] * s),
            ("Dependency Level", w["dependency"] * d),
            ("Criticality", w["criticality"] * c),
            ("Geographic Exposure", w["geographic_exposure"] * g),
        ],
        key=lambda x: x[1]
    )[0]

    explanation = (
        f"Deterministic risk: {final_score:.1f}/100 ({level}). "
        f"Primary factor: {primary_driver}. "
        f"Inputs: Sev={s:.2f}, Dep={d:.2f}, Crit={c:.2f}, GeoExp={g:.2f}"
        + (", Single-Source Penalty applied (+5.0)" if is_single else ".")
    )

    return {
        "risk_score": final_score,
        "risk_level": level,
        "primary_driver": primary_driver,
        "explanation": explanation,
        "components": {
            "severity": round(s, 3),
            "dependency": round(d, 3),
            "criticality": round(c, 3),
            "geographic_exposure": round(g, 3),
            "single_source_penalty": single_source_penalty,
        },
        "formula": "0.30*Sev + 0.25*Dep + 0.25*Crit + 0.20*GeoExp + SingleSourcePenalty",
    }


def batch_calculate_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate deterministic risk across a DataFrame of factor rows."""
    if df.empty:
        return df

    results = []
    for _, row in df.iterrows():
        res = calculate_risk(
            severity=row.get("severity"),
            dependency=row.get("dependency"),
            criticality=row.get("criticality"),
            geographic_exposure=row.get("geographic_exposure", row.get("geo_exposure", 0.5)),
            single_source=row.get("single_source", False),
        )
        results.append(res)

    res_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), res_df], axis=1)
