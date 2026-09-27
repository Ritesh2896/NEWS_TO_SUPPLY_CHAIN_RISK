"""Deterministic Event Extraction and Documented Severity Assignment Engine.

Identifies 10 disruption event types:
  1. Flood
  2. Earthquake
  3. Fire
  4. Storm
  5. Port Disruption
  6. Factory Shutdown
  7. Labor Disruption
  8. Road Closure
  9. Power Outage
  10. Cyber Incident

Deterministic Severity Rules (Scale 1 - 5):
  - Base Severity by Event Category:
      * Earthquake: Base 4
      * Fire: Base 4
      * Cyber Incident: Base 4
      * Flood: Base 3
      * Storm: Base 3
      * Port Disruption: Base 3
      * Factory Shutdown: Base 3
      * Labor Disruption: Base 3
      * Power Outage: Base 2
      * Road Closure: Base 2
  - Severity Modifiers:
      * High-intensity terms (+1 or +2): 'catastrophic', 'devastating', 'fatal', 'state of emergency',
        'indefinite', 'total shutdown', 'massive', 'severe', 'critical', 'destroyed'
      * Low-intensity terms (-1): 'minor', 'temporary', 'partial', 'slight', 'minimal', 'contained',
        'resumed', 'localized'
      * Final score is clamped to range [1, 5].
"""
from __future__ import annotations
import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Any
import pandas as pd

from src.nlp.preprocessing import clean_text, split_sentences

# Event definitions: (event_type, event_category, base_severity, impact_label, triggers)
EVENT_RULES: list[dict[str, Any]] = [
    {
        "event_type": "Port Disruption",
        "event_category": "Logistics",
        "base_severity": 3,
        "impact": "Cargo Disruption",
        "triggers": [
            "port disruption", "port closure", "port strike", "terminal closure",
            "berth congestion", "port congestion", "container bottleneck",
            "harbor shutdown", "dock congestion", "shipping backlog", "port halted"
        ],
    },
    {
        "event_type": "Factory Shutdown",
        "event_category": "Operational",
        "base_severity": 3,
        "impact": "Factory Shutdown",
        "triggers": [
            "factory shutdown", "plant closure", "production halt", "assembly halted",
            "facility closed", "refinery outage", "foundry shutdown", "manufacturing stopped",
            "operations suspended", "plant fire", "shutdown"
        ],
    },
    {
        "event_type": "Labor Disruption",
        "event_category": "Labor",
        "base_severity": 3,
        "impact": "Production Disruption",
        "triggers": [
            "labor disruption", "labor strike", "dockworker strike", "worker walkout",
            "dockworkers strike", "union strike", "strike action", "picket lines",
            "labor dispute", "trucker strike", "protest blockades", "strike"
        ],
    },
    {
        "event_type": "Cyber Incident",
        "event_category": "Cyber",
        "base_severity": 4,
        "impact": "System Outage",
        "triggers": [
            "cyber incident", "cyberattack", "ransomware attack", "system breach",
            "malware infection", "network compromised", "data outage", "it outage",
            "ddos attack", "it systems down"
        ],
    },
    {
        "event_type": "Power Outage",
        "event_category": "Infrastructure",
        "base_severity": 2,
        "impact": "Power Disruption",
        "triggers": [
            "power outage", "blackout", "grid failure", "electrical blackout",
            "power cut", "substation explosion", "energy rationing", "electricity cut"
        ],
    },
    {
        "event_type": "Road Closure",
        "event_category": "Infrastructure",
        "base_severity": 2,
        "impact": "Transport Disruption",
        "triggers": [
            "road closure", "highway blocked", "bridge collapse", "freight rail blocked",
            "railway disruption", "highway closed", "tunnel closure", "route blocked"
        ],
    },
    {
        "event_type": "Earthquake",
        "event_category": "Natural Hazard",
        "base_severity": 4,
        "impact": "Infrastructure Disruption",
        "triggers": [
            "earthquake", "magnitude tremor", "richter scale", "seismic shock",
            "tsunami warning", "aftershock"
        ],
    },
    {
        "event_type": "Flood",
        "event_category": "Natural Hazard",
        "base_severity": 3,
        "impact": "Cargo Disruption",
        "triggers": [
            "flood", "floods", "flooding", "flash flood", "inundation", "monsoon deluge",
            "submerged"
        ],
    },
    {
        "event_type": "Storm",
        "event_category": "Natural Hazard",
        "base_severity": 3,
        "impact": "Transport Disruption",
        "triggers": [
            "storm", "cyclone", "typhoon", "hurricane", "blizzard", "tornado",
            "gale force winds", "tropical storm", "squall"
        ],
    },
    {
        "event_type": "Fire",
        "event_category": "Operational",
        "base_severity": 4,
        "impact": "Facility Shutdown",
        "triggers": [
            "fire", "warehouse fire", "explosion", "blaze", "wildfire", "chemical explosion"
        ],
    },
]

# Modifiers
HIGH_SEVERITY_MODIFIERS = [
    "catastrophic", "devastating", "fatal", "fatalities", "state of emergency",
    "total shutdown", "indefinite", "massive", "severe", "critical", "destroyed",
    "complete halt", "crippling", "widespread"
]

LOW_SEVERITY_MODIFIERS = [
    "minor", "temporary", "partial", "slight", "minimal", "contained",
    "resumed", "localized", "negligible", "isolated", "brief"
]


def calculate_deterministic_severity(
    base_severity: int, text: str
) -> tuple[int, str]:
    """Calculate deterministic severity score in range [1, 5] and explanation."""
    lower = text.lower()
    score = base_severity
    reasons = [f"Base severity {base_severity} for category"]

    # Check high-severity modifiers
    high_matches = [m for m in HIGH_SEVERITY_MODIFIERS if re.search(r"\b" + re.escape(m) + r"\b", lower)]
    if high_matches:
        boost = 2 if any(m in high_matches for m in ["catastrophic", "total shutdown", "destroyed", "state of emergency"]) else 1
        score += boost
        reasons.append(f"+{boost} for high-impact terms: {', '.join(high_matches[:2])}")

    # Check low-severity modifiers
    low_matches = [m for m in LOW_SEVERITY_MODIFIERS if re.search(r"\b" + re.escape(m) + r"\b", lower)]
    if low_matches:
        score -= 1
        reasons.append(f"-1 for low-impact terms: {', '.join(low_matches[:2])}")

    clamped_score = max(1, min(5, score))
    explanation = "; ".join(reasons)
    return clamped_score, explanation


_CACHED_LOC_PAIRS: list[tuple[str, str]] | None = None


def _get_cached_location_pairs() -> list[tuple[str, str]]:
    global _CACHED_LOC_PAIRS
    if _CACHED_LOC_PAIRS is None:
        pairs = []
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        loc_path = os.path.join(root, "data/master/locations.csv")
        if os.path.exists(loc_path):
            try:
                df = pd.read_csv(loc_path, usecols=["location_id", "location_name", "city"])
                for _, row in df.iterrows():
                    lid = str(row.get("location_id", "UNKNOWN")).strip()
                    lname = str(row.get("location_name", "")).strip().lower()
                    city = str(row.get("city", "")).strip().lower()
                    if lname and len(lname) > 3:
                        pairs.append((lname, lid))
                    if city and len(city) > 3:
                        pairs.append((city, lid))
            except Exception:
                pass
        _CACHED_LOC_PAIRS = pairs
    return _CACHED_LOC_PAIRS


def _match_known_location_id(text: str) -> str:
    """Find matching location_id from data/master/locations.csv if present, else 'UNKNOWN'."""
    lower_text = text.lower()
    pairs = _get_cached_location_pairs()
    for term, loc_id in pairs:
        if term in lower_text:
            return loc_id
    return "UNKNOWN"


def extract_event(
    text: str,
    article_id: str = "UNKNOWN",
    published_at: str | None = None,
    source_url: str = "",
) -> dict[str, Any] | None:
    """Extract disruption event details from article text.
    
    Returns structured event dictionary or None if no disruption event triggers are detected.
    """
    clean_article = clean_text(text)
    if not clean_article or len(clean_article) < 5:
        return None

    lower_text = clean_article.lower()
    sentences = split_sentences(clean_article)

    best_match = None
    best_trigger = ""
    best_sentence = ""
    longest_trigger_len = 0

    # Search longest trigger match across rules for maximal specificity
    for rule in EVENT_RULES:
        for trig in rule["triggers"]:
            pattern = re.compile(r"\b" + re.escape(trig) + r"\b", re.IGNORECASE)
            m = pattern.search(lower_text)
            if m and len(trig) > longest_trigger_len:
                longest_trigger_len = len(trig)
                best_match = rule
                best_trigger = trig
                # Find specific sentence containing the trigger
                for s in sentences:
                    if pattern.search(s):
                        best_sentence = s
                        break
                if not best_sentence:
                    best_sentence = clean_article[:250]

    if not best_match:
        return None

    # Calculate deterministic severity
    severity_int, severity_reason = calculate_deterministic_severity(
        best_match["base_severity"], best_sentence or clean_article
    )

    # Confidence heuristic based on trigger specificity and length
    confidence = min(0.98, round(0.75 + (min(len(best_trigger), 25) / 100.0), 3))

    # Deterministic event ID hash
    hash_seed = f"{article_id}_{best_match['event_type']}_{best_trigger}_{best_sentence[:50]}"
    evt_hash = hashlib.sha256(hash_seed.encode("utf-8")).hexdigest()[:8].upper()
    event_id = f"EVT_{evt_hash}"

    # Location lookup
    location_id = _match_known_location_id(clean_article)

    # Event timestamp
    event_time = published_at or datetime.now(timezone.utc).isoformat()

    # Map severity int to legacy string representation for backwards compatibility
    sev_str_map = {1: "Low", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"}

    return {
        "event_id": event_id,
        "article_id": article_id,
        "event_type": best_match["event_type"],
        "event_category": best_match["event_category"],
        "severity": severity_int,
        "severity_level": sev_str_map.get(severity_int, "Medium"),
        "severity_explanation": severity_reason,
        "location_id": location_id,
        "event_time": event_time,
        "extraction_confidence": confidence,
        "evidence_text": best_sentence.strip(),
        "source_url": source_url or "",
        # Legacy pipeline compatibility keys
        "trigger": best_trigger,
        "impact": best_match["impact"],
    }
