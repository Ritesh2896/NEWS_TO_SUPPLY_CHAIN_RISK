"""Explainability engine for the Deterministic Risk Module (Phase 7).

Generates human-readable, auditable explanations highlighting the top drivers
of supply chain disruption risk, along with explicit academic taxonomy disclaimers
distinguishing HEURISTIC RISK from MODEL OUTPUT and GROUND TRUTH.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.risk.config import RiskClassificationType, RiskBand


@dataclass
class ContributingFactor:
    """Structured breakdown of a single candidate factor's contribution to risk."""
    factor_name: str
    display_name: str
    raw_value: Any
    normalized_value: float  # [0.0, 1.0]
    weight: float            # Configured weight
    weighted_impact: float   # normalized_value * weight
    level_label: str         # "high", "medium", "low", etc.
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Convert factor details to dictionary."""
        return {
            "factor_name": self.factor_name,
            "display_name": self.display_name,
            "raw_value": self.raw_value,
            "normalized_value": round(self.normalized_value, 4),
            "weight": round(self.weight, 4),
            "weighted_impact": round(self.weighted_impact, 4),
            "level_label": self.level_label,
            "description": self.description,
        }


TAXONOMY_DISCLAIMER = (
    f"CLASSIFICATION: {RiskClassificationType.HEURISTIC_RISK.value}. "
    "This deterministic risk score is computed via domain-expert weighted rules. "
    "It is NOT an empirical ground-truth label, nor is it a machine-learned MODEL OUTPUT."
)


def get_level_label(val: float) -> str:
    """Map normalized factor score [0.0, 1.0] to descriptive qualitative label."""
    if val >= 0.75:
        return "high"
    if val >= 0.40:
        return "moderate"
    return "low"


def format_factor_reason(factor: ContributingFactor) -> str:
    """Format an individual factor's natural-language contribution statement."""
    name = factor.factor_name
    val = factor.normalized_value

    if name == "event_severity":
        if val >= 0.75:
            return "event severity is high"
        elif val >= 0.40:
            return "event severity is moderate"
        return "event severity is low"

    if name == "geographic_exposure":
        if val >= 0.75:
            return "affected location is geographically close"
        elif val >= 0.40:
            return "affected location is in moderate geographic proximity"
        return "affected location is geographically distant"

    if name == "supplier_criticality":
        if val >= 0.75:
            return "supplier criticality is high"
        elif val >= 0.40:
            return "supplier criticality is moderate"
        return "supplier criticality is low"

    if name == "product_criticality":
        if val >= 0.75:
            return "product criticality is high"
        elif val >= 0.40:
            return "product criticality is moderate"
        return "product criticality is low"

    if name == "dependency_strength":
        if val >= 0.75:
            return "supplier has high dependency"
        elif val >= 0.40:
            return "supplier has moderate dependency"
        return "supplier has low dependency"

    if name == "single_source_dependency":
        if val >= 0.75:
            return "single-source dependency exists with no immediate redundant supplier"
        return "multi-source supply redundancy is present"

    return f"{factor.display_name} is {factor.level_label}"


def generate_risk_explanation(
    factors: list[ContributingFactor],
    risk_score: float,
    risk_band: RiskBand,
) -> tuple[str, list[str]]:
    """Generate structured, human-interpretable explanation for deterministic risk.

    Ranks factors by weighted impact, identifies active upward drivers,
    and structures clear numbered bullet points followed by the taxonomy disclaimer.

    Returns:
        tuple of (full_explanation_text, list_of_bullet_points)
    """
    # Sort factors by weighted impact descending
    sorted_factors = sorted(factors, key=lambda f: f.weighted_impact, reverse=True)

    # Filter drivers that actively contribute elevated risk (normalized value >= 0.40)
    elevated_factors = [f for f in sorted_factors if f.normalized_value >= 0.40]

    # If no elevated factors, use top 2 factors
    driving_factors = elevated_factors if elevated_factors else sorted_factors[:2]

    bullet_points: list[str] = []
    for idx, f in enumerate(driving_factors, start=1):
        reason = format_factor_reason(f)
        bullet_points.append(f"{idx}. {reason}")

    explanation_lines = [
        f"Deterministic Risk: {risk_score:.1f}/100 ({risk_band.value}).",
        "Risk increased because:" if bullet_points else "Risk evaluated based on standard factors:",
    ]
    explanation_lines.extend(bullet_points)
    explanation_lines.append("")
    explanation_lines.append(TAXONOMY_DISCLAIMER)

    full_explanation = "\n".join(explanation_lines)
    return full_explanation, bullet_points
