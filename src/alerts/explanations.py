"""Explanation Generator for Early-Warning Alerts in BDS-35 (Phase 10).

Constructs natural-language audit trails for combined risk alerts according to
the canonical explanation template:

  Supplier:
  SUP00123

  Risk:
  HIGH

  Reasons:
  * High-severity disruption
  * Geographic exposure detected
  * High dependency relationship
  * Elevated deterministic risk
  * Graph propagation from affected upstream nodes
"""
from __future__ import annotations

from typing import Any
from src.alerts.schemas import RiskBand, PROJECT_THRESHOLDS_DISCLAIMER


def extract_alert_reasons(
    event_severity: float,
    geographic_exposure: float,
    dependency_strength: float,
    deterministic_risk: float,
    graphsage_risk: float,
    gat_risk: float,
    supplier_criticality: float = 0.5,
    product_criticality: float = 0.5,
    single_source: bool = False,
) -> list[str]:
    """Identify qualitative disruption reasons based on normalized factor levels."""
    reasons: list[str] = []

    # 1. Event severity
    if event_severity >= 0.70:
        reasons.append("High-severity disruption")
    elif event_severity >= 0.40:
        reasons.append("Moderate disruption event detected")

    # 2. Geographic exposure
    if geographic_exposure >= 0.65:
        reasons.append("Geographic exposure detected")
    elif geographic_exposure >= 0.40:
        reasons.append("Moderate geographic proximity to event")

    # 3. Dependency relationship
    if dependency_strength >= 0.70:
        reasons.append("High dependency relationship")
    elif dependency_strength >= 0.45:
        reasons.append("Direct operational dependency")

    # 4. Deterministic risk
    if deterministic_risk >= 0.60:
        reasons.append("Elevated deterministic risk")
    elif deterministic_risk >= 0.40:
        reasons.append("Moderate baseline heuristic risk")

    # 5. Graph propagation
    if graphsage_risk >= 0.50 or gat_risk >= 0.50:
        reasons.append("Graph propagation from affected upstream nodes")

    # 6. Single-source sole supplier
    if single_source:
        reasons.append("Single-source sole supplier vulnerability")

    # 7. Product criticality
    if product_criticality >= 0.75:
        reasons.append("Essential product criticality")

    # Fallback if all factors are low
    if not reasons:
        reasons.append("Routine baseline operational monitoring")

    return reasons


def format_alert_explanation(
    supplier_id: str,
    supplier_name: str | None,
    risk_band: RiskBand | str,
    risk_score_100: float,
    reasons: list[str],
    include_disclaimer: bool = True,
) -> str:
    """Format structured natural-language alert explanation matching required academic template.

    Example Output:
      Supplier:
      SUP00123

      Risk:
      HIGH

      Reasons:
      * High-severity disruption
      * Geographic exposure detected
      * High dependency relationship
      * Elevated deterministic risk
      * Graph propagation from affected upstream nodes
    """
    band_str = risk_band.value if isinstance(risk_band, RiskBand) else str(risk_band).upper()

    lines = [
        "Supplier:",
        f"{supplier_id}" + (f" ({supplier_name})" if supplier_name else ""),
        "",
        "Risk:",
        f"{band_str}",
        "",
        "Reasons:",
    ]

    for r in reasons:
        lines.append(f"* {r}")

    if include_disclaimer:
        lines.append("")
        lines.append(PROJECT_THRESHOLDS_DISCLAIMER)

    return "\n".join(lines)
