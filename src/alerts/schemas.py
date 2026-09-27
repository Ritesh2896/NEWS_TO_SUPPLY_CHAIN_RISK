"""Schemas and Configuration for Combined Risk and Alert Generation (Phase 10).

Defines:
- RiskBand: Project-defined risk bands (LOW, MEDIUM, HIGH, CRITICAL).
- CombinedRiskWeights: Configurable blending weights (alpha, beta, gamma).
- RiskBandThresholds: Project-defined boundary thresholds.
- AlertRecord: Structured schema for early-warning risk alerts.
- Academic Disclaimers on project-defined threshold status.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RiskBand(str, Enum):
    """Categorical risk tiers defined for project-level prioritization."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Explicit academic disclaimer mandated by methodology
PROJECT_THRESHOLDS_DISCLAIMER: str = (
    "PROJECT THRESHOLD DISCLAIMER: Risk bands (LOW, MEDIUM, HIGH, CRITICAL) "
    "are project-defined operational prioritization thresholds calibrated for early warning, "
    "not externally validated ground-truth labels."
)


@dataclass(frozen=True)
class CombinedRiskWeights:
    """Configurable weights combining Deterministic, GraphSAGE, and GAT risks.

    Formula:
      combined_risk = alpha * deterministic_risk + beta * graphsage_risk + gamma * gat_risk

    Constraint:
      alpha + beta + gamma == 1.0 (with non-negative components)
    """
    alpha: float = 0.50  # Deterministic Risk weight
    beta: float = 0.25   # GraphSAGE Propagation Risk weight
    gamma: float = 0.25  # GAT Attention Propagation Risk weight

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError("All combination weights must be non-negative.")
        total = self.alpha + self.beta + self.gamma
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Combination weights must sum to 1.0 (got {total:.4f}).")

    def to_dict(self) -> dict[str, float]:
        """Convert weights to dictionary."""
        return {
            "alpha_deterministic": self.alpha,
            "beta_graphsage": self.beta,
            "gamma_gat": self.gamma,
        }


@dataclass(frozen=True)
class RiskBandThresholds:
    """Project-defined cutoff thresholds for risk classification on [0.0, 1.0] scale."""
    low_upper: float = 0.40      # [0.00, 0.40) -> LOW
    medium_upper: float = 0.60   # [0.40, 0.60) -> MEDIUM
    high_upper: float = 0.75     # [0.60, 0.75) -> HIGH
    # [0.75, 1.00] -> CRITICAL

    def determine_band(self, score: float) -> RiskBand:
        """Map normalized score [0.0, 1.0] to project-defined RiskBand."""
        clamped = min(1.0, max(0.0, float(score)))
        if clamped >= self.high_upper:
            return RiskBand.CRITICAL
        if clamped >= self.medium_upper:
            return RiskBand.HIGH
        if clamped >= self.low_upper:
            return RiskBand.MEDIUM
        return RiskBand.LOW


DEFAULT_COMBINED_WEIGHTS = CombinedRiskWeights()
DEFAULT_BAND_THRESHOLDS = RiskBandThresholds()


@dataclass
class AlertRecord:
    """Structured representation of an explainable early-warning alert."""
    alert_id: str
    supplier_id: str
    supplier_name: str
    event_id: str
    event_type: str
    deterministic_risk: float
    graphsage_risk: float
    gat_risk: float
    combined_risk: float
    risk_score_100: float
    risk_band: str
    reasons: list[str]
    explanation: str
    contributing_factors: dict[str, Any]
    threshold_disclaimer: str = PROJECT_THRESHOLDS_DISCLAIMER
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data_status: str = "SYNTHETIC_DEMO"

    def to_dict(self) -> dict[str, Any]:
        """Convert alert record to dictionary matching alerts.csv schema."""
        return {
            "alert_id": self.alert_id,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "deterministic_risk": round(self.deterministic_risk, 4),
            "graphsage_risk": round(self.graphsage_risk, 4),
            "gat_risk": round(self.gat_risk, 4),
            "combined_risk": round(self.combined_risk, 4),
            "risk_score_100": round(self.risk_score_100, 2),
            "risk_band": self.risk_band,
            "reasons": "; ".join(self.reasons),
            "explanation": self.explanation,
            "created_at": self.created_at,
            "data_status": self.data_status,
        }
