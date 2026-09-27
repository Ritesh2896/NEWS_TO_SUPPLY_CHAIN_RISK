"""Confidence calibration and threshold management for entity linking."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LinkingThresholds:
    """Configurable thresholds and base confidence levels for entity linking."""

    supplier_threshold: float = 0.55
    location_threshold: float = 0.55
    product_threshold: float = 0.55
    
    # Base confidence assignments per matching method
    exact_base_confidence: float = 0.98
    normalized_base_confidence: float = 0.92
    alias_base_confidence: float = 0.88
    fuzzy_discount: float = 0.95


def calibrate_confidence(
    match_method: str, match_score: float, thresholds: LinkingThresholds | None = None
) -> float:
    """Calibrate a deterministic confidence score in [0.0, 1.0] based on match method and raw similarity.
    
    Methods:
      - 'exact': returns ~0.98 - 1.0
      - 'normalized': returns ~0.90 - 0.95
      - 'alias': returns ~0.85 - 0.90
      - 'fuzzy': scaled by match_score
      - 'unresolved': 0.0
    """
    if thresholds is None:
        thresholds = LinkingThresholds()

    if match_method == "exact":
        conf = thresholds.exact_base_confidence
    elif match_method == "normalized":
        conf = min(0.96, thresholds.normalized_base_confidence + (match_score - 0.90) * 0.4)
    elif match_method == "alias":
        conf = min(0.92, thresholds.alias_base_confidence + (match_score - 0.80) * 0.3)
    elif match_method == "fuzzy":
        conf = match_score * thresholds.fuzzy_discount
    else:
        conf = 0.0

    return round(float(max(0.0, min(1.0, conf))), 3)
