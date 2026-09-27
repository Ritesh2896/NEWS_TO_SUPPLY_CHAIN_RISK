"""Combined Risk Engine and Alert Generator for BDS-35 (Phase 10).

Integrates:
  1. Deterministic heuristic risk (Phase 7)
  2. GraphSAGE topological propagation risk (Phase 8)
  3. GAT attention-weighted propagation risk (Phase 9)

Combines them via configurable weighted formula:
  combined_risk = alpha * deterministic_risk + beta * graphsage_risk + gamma * gat_risk

Normalizes strictly to [0.0, 1.0].
Assigns project-defined risk bands: LOW, MEDIUM, HIGH, CRITICAL.
Generates structured alerts with explanations and outputs to data/processed/alerts.csv.
"""
from __future__ import annotations

import os
from typing import Any, Sequence
import numpy as np
import pandas as pd

from src.alerts.schemas import (
    AlertRecord,
    CombinedRiskWeights,
    DEFAULT_COMBINED_WEIGHTS,
    DEFAULT_BAND_THRESHOLDS,
    PROJECT_THRESHOLDS_DISCLAIMER,
    RiskBand,
    RiskBandThresholds,
)
from src.alerts.explanations import (
    extract_alert_reasons,
    format_alert_explanation,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROCESSED_DIR = os.path.join(ROOT, "data/processed")


def calculate_combined_risk(
    deterministic_risk: float,
    graphsage_risk: float,
    gat_risk: float,
    weights: CombinedRiskWeights | None = None,
    thresholds: RiskBandThresholds | None = None,
) -> tuple[float, float, RiskBand]:
    """Combine deterministic risk, GraphSAGE propagation risk, and GAT propagation risk.

    Args:
        deterministic_risk: Heuristic risk score in [0.0, 1.0] or [0.0, 100.0].
        graphsage_risk: GraphSAGE propagation score in [0.0, 1.0] or [0.0, 100.0].
        gat_risk: GAT attention propagation score in [0.0, 1.0] or [0.0, 100.0].
        weights: Configurable weights (alpha, beta, gamma).
        thresholds: Project-defined band cutoffs.

    Returns:
        tuple of (combined_risk_norm, combined_risk_100, risk_band):
          - combined_risk_norm: Float in [0.0, 1.0]
          - combined_risk_100: Float in [0.0, 100.0]
          - risk_band: RiskBand enum (LOW, MEDIUM, HIGH, CRITICAL)
    """
    w = weights or DEFAULT_COMBINED_WEIGHTS
    t = thresholds or DEFAULT_BAND_THRESHOLDS

    # Normalize inputs to [0.0, 1.0] if provided on 0-100 scale
    d_norm = deterministic_risk / 100.0 if deterministic_risk > 1.0 else max(0.0, float(deterministic_risk))
    sage_norm = graphsage_risk / 100.0 if graphsage_risk > 1.0 else max(0.0, float(graphsage_risk))
    gat_norm = gat_risk / 100.0 if gat_risk > 1.0 else max(0.0, float(gat_risk))

    # Weighted combination
    combined_norm = (
        w.alpha * d_norm
        + w.beta * sage_norm
        + w.gamma * gat_norm
    )
    combined_norm = round(min(1.0, max(0.0, combined_norm)), 4)
    combined_100 = round(combined_norm * 100.0, 2)

    band = t.determine_band(combined_norm)
    return combined_norm, combined_100, band


def generate_alerts(
    risk_records: list[dict[str, Any]] | pd.DataFrame,
    weights: CombinedRiskWeights | None = None,
    thresholds: RiskBandThresholds | None = None,
    output_csv_path: str = "data/processed/alerts.csv",
    save_csv: bool = True,
) -> pd.DataFrame:
    """Generate explainable early-warning alert records from risk assessments.

    Args:
        risk_records: List of dictionaries or DataFrame containing factor inputs:
          - supplier_id, event_id, [supplier_name, event_type]
          - deterministic_risk
          - graphsage_risk
          - gat_risk
          - [event_severity, geographic_exposure, dependency_strength, etc.]
        weights: Configurable blending weights.
        thresholds: Project-defined band boundaries.
        output_csv_path: Relative or absolute path to write alerts.csv.
        save_csv: If True, persists DataFrame to disk.

    Returns:
        DataFrame containing generated AlertRecord objects.
    """
    if isinstance(risk_records, pd.DataFrame):
        records_list = risk_records.to_dict(orient="records")
    else:
        records_list = list(risk_records)

    if not records_list:
        empty_cols = [
            "alert_id", "supplier_id", "supplier_name", "event_id", "event_type",
            "deterministic_risk", "graphsage_risk", "gat_risk", "combined_risk",
            "risk_score_100", "risk_band", "reasons", "explanation", "created_at", "data_status"
        ]
        return pd.DataFrame(columns=empty_cols)

    alerts: list[dict[str, Any]] = []

    for idx, item in enumerate(records_list):
        sup_id = str(item.get("supplier_id", f"SUP_{idx:05d}")).strip()
        sup_name = item.get("supplier_name", f"Supplier {sup_id}")
        evt_id = str(item.get("event_id", f"EVT_{idx:05d}")).strip()
        evt_type = str(item.get("event_type", "Operational Disruption"))
        alert_id = f"ALT-{evt_id}-{sup_id}"

        # Risk components
        det_risk = float(item.get("deterministic_risk", 0.50))
        sage_risk = float(item.get("graphsage_risk", item.get("graph_risk", det_risk)))
        gat_risk = float(item.get("gat_risk", sage_risk))

        # Calculate combined risk
        c_norm, c_100, band = calculate_combined_risk(
            deterministic_risk=det_risk,
            graphsage_risk=sage_risk,
            gat_risk=gat_risk,
            weights=weights,
            thresholds=thresholds,
        )

        # Context factors for reasons
        sev = float(item.get("event_severity", item.get("severity", 0.5)))
        sev_norm = sev / 5.0 if sev > 1.0 else sev
        geo = float(item.get("geographic_exposure", item.get("exposure_score", 0.5)))
        dep = float(item.get("dependency_strength", item.get("dependency", 0.5)))
        s_crit = float(item.get("supplier_criticality", 0.5))
        p_crit = float(item.get("product_criticality", 0.5))
        single = bool(item.get("single_source_dependency", item.get("single_source", False)))

        # Extract reasons
        reasons = extract_alert_reasons(
            event_severity=sev_norm,
            geographic_exposure=geo,
            dependency_strength=dep,
            deterministic_risk=det_risk / 100.0 if det_risk > 1.0 else det_risk,
            graphsage_risk=sage_risk / 100.0 if sage_risk > 1.0 else sage_risk,
            gat_risk=gat_risk / 100.0 if gat_risk > 1.0 else gat_risk,
            supplier_criticality=s_crit,
            product_criticality=p_crit,
            single_source=single,
        )

        # Format canonical natural-language explanation
        explanation = format_alert_explanation(
            supplier_id=sup_id,
            supplier_name=sup_name,
            risk_band=band,
            risk_score_100=c_100,
            reasons=reasons,
        )

        record = AlertRecord(
            alert_id=alert_id,
            supplier_id=sup_id,
            supplier_name=sup_name,
            event_id=evt_id,
            event_type=evt_type,
            deterministic_risk=round(det_risk / 100.0 if det_risk > 1.0 else det_risk, 4),
            graphsage_risk=round(sage_risk / 100.0 if sage_risk > 1.0 else sage_risk, 4),
            gat_risk=round(gat_risk / 100.0 if gat_risk > 1.0 else gat_risk, 4),
            combined_risk=c_norm,
            risk_score_100=c_100,
            risk_band=band.value,
            reasons=reasons,
            explanation=explanation,
            contributing_factors={
                "deterministic_risk": det_risk,
                "graphsage_risk": sage_risk,
                "gat_risk": gat_risk,
                "event_severity": sev_norm,
                "geographic_exposure": geo,
                "dependency_strength": dep,
            },
            data_status=str(item.get("data_status", "SYNTHETIC_DEMO")),
        )
        alerts.append(record.to_dict())

    df = pd.DataFrame(alerts)

    if save_csv:
        full_out_path = os.path.join(ROOT, output_csv_path) if not os.path.isabs(output_csv_path) else output_csv_path
        os.makedirs(os.path.dirname(os.path.abspath(full_out_path)), exist_ok=True)
        df.to_csv(full_out_path, index=False)

    return df
