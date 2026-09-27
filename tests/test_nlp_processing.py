"""Unit tests for Phase 3: Local NLP Processing (NEWS -> EVENT -> ENTITY).

Tests:
  - Text preprocessing and sentence segmentation
  - Entity extraction across organizations, suppliers, locations, countries, cities, ports, facilities, products
  - Entity normalization
  - Disruption event extraction across all 10 required types
  - Deterministic severity assignment and modifier logic
  - Provenance and UNKNOWN fallback handling
  - Live enrichment and persistence
"""
from __future__ import annotations
import os
import pytest
import pandas as pd

from src.nlp.preprocessing import clean_text, split_sentences, extract_context_window
from src.nlp.normalization import normalize_entity_name, normalize_entity_type
from src.nlp.entity_extraction import extract_entities, extract_entity_spans
from src.nlp.event_extraction import (
    extract_event,
    calculate_deterministic_severity,
    EVENT_RULES,
)
from src.nlp.live_enrichment import enrich_article, store_events_and_entities


def test_text_preprocessing():
    """Verify HTML stripping, URL removal, and sentence splitting."""
    raw = "<p>Port of Rotterdam reports <b>severe delays</b>. Visit https://news.example.com for live info! Ships are waiting.</p>"
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "https://" not in cleaned
    assert "Port of Rotterdam reports severe delays." in cleaned

    sentences = split_sentences(cleaned)
    assert len(sentences) >= 2
    assert "Port of Rotterdam reports severe delays." in sentences[0]


def test_entity_normalization():
    """Verify canonical entity standardizations, suffix removals, and alias resolution."""
    # Corporate suffix removal
    assert normalize_entity_name("Samsung Electronics Co., Ltd.", "ORG") == "SAMSUNG ELECTRONICS"
    assert normalize_entity_name("Maersk Line Inc.", "ORG") == "MAERSK LINE"

    # Port canonicalization
    assert normalize_entity_name("Port of Rotterdam", "PORT") == "ROTTERDAM PORT"
    assert normalize_entity_name("Shanghai Port", "PORT") == "SHANGHAI PORT"

    # Country aliases
    assert normalize_entity_name("USA", "GPE") == "UNITED STATES"
    assert normalize_entity_name("UK", "GPE") == "UNITED KINGDOM"

    # Empty fallback
    assert normalize_entity_name("", "ORG") == "UNKNOWN"

    # Entity type normalization
    assert normalize_entity_type("ORG") == "ORGANIZATION"
    assert normalize_entity_type("GPE") == "LOCATION"
    assert normalize_entity_type("FAC") == "FACILITY"


def test_entity_extraction_spans():
    """Verify entity extraction across ports, facilities, locations, countries, and products."""
    text = (
        "A major fire broke out at the Foxconn Assembly Plant near Shanghai Port in China. "
        "Shipments of Lithium Batteries and Semiconductors have been halted."
    )
    entities = extract_entities(text)

    # Check categories populated
    assert len(entities["ports"]) > 0 or "Shanghai Port" in str(entities["entities"])
    assert any("China" in c for c in entities["countries"]) or "CHINA" in [e["normalized_entity"] for e in entities["entities"]]
    assert any("Plant" in f or "Assembly" in f for f in entities["facilities"]) or any("FACILITY" == e["entity_type"] for e in entities["entities"])
    assert any("Lithium" in p or "Semiconductor" in p for p in entities["products"])

    # Check structured spans format
    spans = entities["entities"]
    assert len(spans) > 0
    for s in spans:
        assert "entity_text" in s
        assert "entity_type" in s
        assert "normalized_entity" in s
        assert "confidence" in s
        assert 0.0 <= s["confidence"] <= 1.0
        assert s["character_start"] < s["character_end"]


def test_event_extraction_10_types():
    """Verify event extraction identifies all 10 required supply-chain disruption types."""
    test_cases = [
        ("Flash flood waters inundated the container storage yard.", "Flood", "Natural Hazard"),
        ("A magnitude 7.2 earthquake damaged railway bridges.", "Earthquake", "Natural Hazard"),
        ("A warehouse fire destroyed raw material inventory.", "Fire", "Operational"),
        ("Tropical storm winds forced container ships to remain at anchor.", "Storm", "Natural Hazard"),
        ("Severe port disruption and berth congestion delayed twenty vessels.", "Port Disruption", "Logistics"),
        ("Management declared a complete factory shutdown at the semiconductor facility.", "Factory Shutdown", "Operational"),
        ("Union dockworkers voted for a nationwide labor strike halting operations.", "Labor Disruption", "Labor"),
        ("A critical freight corridor highway road closure prevented container delivery.", "Road Closure", "Infrastructure"),
        ("A sudden electrical blackout and power outage stopped the manufacturing lines.", "Power Outage", "Infrastructure"),
        ("A ransomware cyber incident encrypted core logistics scheduling servers.", "Cyber Incident", "Cyber"),
    ]

    for text, expected_type, expected_cat in test_cases:
        evt = extract_event(text, article_id="TEST_ART_01")
        assert evt is not None, f"Failed to extract event for: {text}"
        assert evt["event_type"] == expected_type
        assert evt["event_category"] == expected_cat
        assert 1 <= evt["severity"] <= 5
        assert 0.0 <= evt["extraction_confidence"] <= 1.0
        assert len(evt["evidence_text"]) > 0
        assert evt["article_id"] == "TEST_ART_01"


def test_deterministic_severity_assignment():
    """Verify base severities and modifier adjustments (+boost / -reduction)."""
    # 1. Base severity for earthquake is 4
    base_score, _ = calculate_deterministic_severity(4, "Earthquake detected near the coast.")
    assert base_score == 4

    # 2. Catastrophic modifier boosts to 5
    boosted_score, explanation = calculate_deterministic_severity(
        4, "A catastrophic earthquake destroyed the manufacturing zone."
    )
    assert boosted_score == 5
    assert "+" in explanation

    # 3. Minor / contained modifier reduces severity
    reduced_score, red_explanation = calculate_deterministic_severity(
        3, "A minor flood was quickly contained with minimal damage."
    )
    assert reduced_score == 2
    assert "-" in red_explanation

    # 4. Severity stays strictly within [1, 5] bounds
    extreme_boost, _ = calculate_deterministic_severity(
        4, "Catastrophic, devastating, state of emergency, total shutdown, destroyed!"
    )
    assert extreme_boost == 5

    extreme_low, _ = calculate_deterministic_severity(
        2, "Minor, slight, temporary, contained, minimal, localized issue."
    )
    assert extreme_low >= 1


def test_unknown_facts_not_inferred():
    """Verify unknown location remains UNKNOWN and facts are not fabricated."""
    text = "Power outage affected operations at an undisclosed overseas warehouse."
    evt = extract_event(text)
    assert evt is not None
    assert evt["event_type"] == "Power Outage"
    # Location was not in article, must remain UNKNOWN
    assert evt["location_id"] == "UNKNOWN"

    # Completely irrelevant non-disruption text returns None
    irrelevant = "The new romantic comedy movie will premiere on Friday night in local theaters."
    no_evt = extract_event(irrelevant)
    assert no_evt is None


def test_live_enrichment_and_storage(tmp_path):
    """Verify enrich_article and storage into isolated CSV paths."""
    title = "Port disruption halts container movements in Shanghai"
    content = "Berth congestion and labor strike caused severe bottlenecks at the terminal."
    
    evt, entities = enrich_article(
        title=title,
        content=content,
        article_id="TEST-LIVE-001",
        published_at="2026-09-23T12:00:00Z",
        source_url="https://example.com/live-test",
        data_status="REAL_DATA",
    )

    assert evt is not None
    assert evt["article_id"] == "TEST-LIVE-001"
    assert evt["event_type"] in {"Port Disruption", "Labor Disruption"}
    assert evt["data_status"] == "REAL_DATA"
    assert len(entities) > 0

    test_events_csv = str(tmp_path / "events.csv")
    test_entities_csv = str(tmp_path / "news_entities.csv")

    from src.nlp.live_enrichment import store_events_and_entities
    ev_stored, ent_stored = store_events_and_entities(
        [evt], entities, events_path=test_events_csv, entities_path=test_entities_csv
    )
    assert ev_stored == 1
    assert ent_stored == len(entities)
    assert os.path.exists(test_events_csv)
    assert os.path.exists(test_entities_csv)

    # Re-storing identical items should prevent duplicate records
    ev_again, ent_again = store_events_and_entities(
        [evt], entities, events_path=test_events_csv, entities_path=test_entities_csv
    )
    assert ev_again == 0
    assert ent_again == 0
