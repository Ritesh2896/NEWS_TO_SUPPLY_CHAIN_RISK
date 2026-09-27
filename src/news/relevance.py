"""Supply-chain disruption relevance detection using rule-based and lexicon matching."""
from __future__ import annotations
import re
from typing import Sequence
from src.news.models import NormalizedArticle

# Core supply-chain entity & logistics indicators
ENTITY_KEYWORDS = [
    "port", "ports", "terminal", "terminals", "harbor", "harbour", "vessel", "vessels",
    "cargo", "freight", "container", "containers", "shipping", "shipment", "shipments",
    "logistics", "supply chain", "supply-chain", "warehouse", "warehouses", "depot",
    "factory", "factories", "plant", "plants", "manufacturing", "supplier", "suppliers",
    "semiconductor", "semiconductors", "chip", "chips", "raw materials", "assembly",
    "trade route", "maritime", "canal", "railway", "freight train"
]

# Disruption event indicators
DISRUPTION_KEYWORDS = [
    "disruption", "disruptions", "strike", "strikes", "protest", "protests", "walkout",
    "shutdown", "shutdowns", "closure", "closed", "halt", "halted", "halts", "suspended",
    "delay", "delays", "delayed", "congestion", "congested", "bottleneck", "bottlenecks",
    "shortage", "shortages", "fire", "explosion", "flood", "floods", "flooding",
    "cyclone", "typhoon", "hurricane", "earthquake", "storm", "tariff", "tariffs",
    "sanction", "sanctions", "embargo", "blockade", "accident", "grounded"
]

# High-priority composite disruption phrases
COMPOSITE_PHRASES = [
    "port disruption", "port strike", "port closure", "factory shutdown", "shipping delay",
    "supply chain disruption", "cargo bottleneck", "container shortage", "freight delay",
    "logistics crisis", "manufacturing halt", "plant closure", "terminal congestion"
]


def check_relevance(text: str) -> tuple[bool, float, list[str]]:
    """Determine whether text describes a supply-chain disruption.

    Returns:
        (is_relevant, relevance_score, matched_terms)
    """
    clean_text = str(text or "").lower()
    if len(clean_text) < 10:
        return False, 0.0, []

    matched_phrases = [p for p in COMPOSITE_PHRASES if p in clean_text]
    matched_entities = [k for k in ENTITY_KEYWORDS if re.search(r"\b" + re.escape(k) + r"\b", clean_text)]
    matched_disruptions = [k for k in DISRUPTION_KEYWORDS if re.search(r"\b" + re.escape(k) + r"\b", clean_text)]

    all_matches = list(set(matched_phrases + matched_entities + matched_disruptions))

    # Scoring heuristic
    score = 0.0
    if matched_phrases:
        score += 0.50 + 0.15 * min(len(matched_phrases), 3)

    if matched_entities and matched_disruptions:
        score += 0.40 + 0.05 * min(len(matched_entities) + len(matched_disruptions), 6)
    elif matched_phrases:
        score += 0.20
    elif len(matched_disruptions) >= 2 or len(matched_entities) >= 3:
        score += 0.35

    score = min(1.0, round(score, 3))
    is_relevant = score >= 0.40 or len(matched_phrases) > 0 or (len(matched_entities) >= 1 and len(matched_disruptions) >= 1)

    return is_relevant, score, all_matches


def filter_relevant_articles(
    articles: Sequence[NormalizedArticle],
) -> tuple[list[NormalizedArticle], int]:
    """Filter out non-supply-chain articles.

    Returns:
        (relevant_articles, irrelevant_removed_count)
    """
    relevant: list[NormalizedArticle] = []
    irrelevant_count = 0

    for a in articles:
        combined_text = f"{a.title} {a.description} {a.content}".strip()
        is_rel, score, _ = check_relevance(combined_text)
        if is_rel:
            relevant.append(a)
        else:
            irrelevant_count += 1

    return relevant, irrelevant_count
