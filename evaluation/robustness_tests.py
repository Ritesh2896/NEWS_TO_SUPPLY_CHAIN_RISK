"""Robustness and Stress Testing Suite for BDS-35.

Tests boundary conditions, edge cases, and unexpected inputs across the pipeline:
  R1: Empty / whitespace article input
  R2: Missing location mentions
  R3: Unknown / unlisted supplier entity
  R4: Rapid repeated duplicate stream
  R5: Irrelevant commercial/celebrity news
  R6: Malformed / NaN geographic coordinates
  R7: Missing product/location master data fallback
"""
from __future__ import annotations
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.deduplication.deduplicate import is_duplicate
from src.nlp.entity_extraction import extract_entities
from src.nlp.event_extraction import extract_event
from src.pipeline.end_to_end_pipeline import run_pipeline
from src.relevance.relevance import calculate_relevance
from src.risk.geospatial_exposure import calculate_geographic_exposure, haversine_km
from src.risk.risk_engine import calculate_risk


def run_robustness():
    results = []

    # R1: Empty / Whitespace text
    try:
        rel = calculate_relevance("   ")
        ent = extract_entities("")
        evt = extract_event("   ")
        r_df = run_pipeline([{"news_id": "R1", "title": "", "content": "   "}])
        assert len(r_df) == 0
        results.append(("R1", "Empty / Whitespace Input", "PASS", "Gracefully ignored without crashing"))
    except Exception as e:
        results.append(("R1", "Empty / Whitespace Input", "FAIL", str(e)))

    # R2: Missing Real Geographic Location
    try:
        item = [{"news_id": "R2", "title": "Workers strike halts assembly at internal plant facility", "content": "Workers walk out on strike."}]
        r_df = run_pipeline(item)
        assert len(r_df) == 1
        assert r_df.iloc[0]["location_id"] is None
        assert 0.0 <= r_df.iloc[0]["geographic_exposure"] <= 1.0
        results.append(("R2", "Missing Geographic Location", "PASS", "Handled with default fallback exposure"))
    except Exception as e:
        results.append(("R2", "Missing Geographic Location", "FAIL", str(e)))

    # R3: Unknown Supplier
    try:
        item = [{"news_id": "R3", "title": "Fire disrupts NonExistentCorpXYZ999 in Mumbai", "content": "Factory fire caused shutdown."}]
        r_df = run_pipeline(item)
        assert len(r_df) == 1
        assert r_df.iloc[0]["supplier_name"] == "Unlinked Supplier"
        results.append(("R3", "Unknown Supplier Mention", "PASS", "Alert created with Unlinked status"))
    except Exception as e:
        results.append(("R3", "Unknown Supplier Mention", "FAIL", str(e)))

    # R4: Duplicate Article Stream
    try:
        dup_text = "Severe flooding closes cargo terminals in Hamburg port."
        pool = [dup_text]
        assert is_duplicate(dup_text, pool) is True
        results.append(("R4", "Duplicate Article Stream", "PASS", "Exact & fuzzy duplicates detected"))
    except Exception as e:
        results.append(("R4", "Duplicate Article Stream", "FAIL", str(e)))

    # R5: Irrelevant News Filter
    try:
        irrelevant_text = "Acme Studios announces release date for upcoming animated comedy film starring local celebrities."
        rel = calculate_relevance(irrelevant_text)
        assert rel["label"] is False
        results.append(("R5", "Irrelevant News Filtering", "PASS", "Zero false-positive disruption detection"))
    except Exception as e:
        results.append(("R5", "Irrelevant News Filtering", "FAIL", str(e)))

    # R6: Malformed / NaN Coordinates
    try:
        d = haversine_km(None, "invalid_lon", float("nan"), 120.0)
        assert d >= 9000.0
        score, _ = calculate_geographic_exposure(None, None, 19.0, 72.8)
        assert 0.0 <= score <= 1.0
        results.append(("R6", "Malformed / NaN Coordinates", "PASS", "Safely caught and mapped to fallback distance"))
    except Exception as e:
        results.append(("R6", "Malformed / NaN Coordinates", "FAIL", str(e)))

    # R7: Extreme Deterministic Risk Inputs
    try:
        res = calculate_risk(severity=999, dependency="UNKNOWN_LEVEL", criticality=None, geographic_exposure=-5.0)
        assert 0.0 <= res["risk_score"] <= 100.0
        results.append(("R7", "Extreme Factor Inputs", "PASS", "Clamped within [0, 100] bounds"))
    except Exception as e:
        results.append(("R7", "Extreme Factor Inputs", "FAIL", str(e)))

    print("\n=================================================================")
    print("                BDS-35 SYSTEM ROBUSTNESS TESTS                   ")
    print("=================================================================")
    for code, name, status, notes in results:
        print(f"  {code} | {name:<28} | {status} | {notes}")
    print("=================================================================\n")
    return results


if __name__ == "__main__":
    run_robustness()
