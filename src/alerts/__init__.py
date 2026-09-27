"""Alert generation and risk combination module for BDS-35 (Phase 10)."""

from src.alerts.schemas import (
    RiskBand,
    CombinedRiskWeights,
    DEFAULT_COMBINED_WEIGHTS,
    RiskBandThresholds,
    DEFAULT_BAND_THRESHOLDS,
    PROJECT_THRESHOLDS_DISCLAIMER,
    AlertRecord,
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

__all__ = [
    "RiskBand",
    "CombinedRiskWeights",
    "DEFAULT_COMBINED_WEIGHTS",
    "RiskBandThresholds",
    "DEFAULT_BAND_THRESHOLDS",
    "PROJECT_THRESHOLDS_DISCLAIMER",
    "AlertRecord",
    "extract_alert_reasons",
    "format_alert_explanation",
    "calculate_combined_risk",
    "generate_alerts",
    "create_alert",
]
