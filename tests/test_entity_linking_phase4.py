"""Unit tests for Phase 4: Entity Linking (NEWS -> ENTITY -> NORMALIZED ENTITY -> MASTER).

Tests:
  - Exact matching
  - Normalized string matching
  - Alias matching
  - Controlled fuzzy matching and numeric conflict penalty
  - Thresholding & UNKNOWN fallback (no hallucinated real-world entities)
  - Linking-quality report generation
  - Determinism & reproducibility
  - Storage persistence and non-duplication
"""
from __future__ import annotations
import os
import pandas as pd
import pytest

from src.linking.confidence import LinkingThresholds, calibrate_confidence
from src.linking.entity_linker import EntityLinker, save_entity_links
from src.linking.normalization import (
    check_numeric_conflict,
    extract_numeric_tokens,
    normalize_linking_text,
)


@pytest.fixture
def mock_linker():
    """Create an EntityLinker populated with known master test records."""
    sups = pd.DataFrame([
        {"supplier_id": "SUP00001", "supplier_name": "Apex Semiconductor Manufacturing Inc."},
        {"supplier_id": "SUP00002", "supplier_name": "Global Logistics Corp"},
        {"supplier_id": "SUP00003", "supplier_name": "Pacific Battery Systems Ltd."},
    ])
    locs = pd.DataFrame([
        {"location_id": "LOC00001", "location_name": "Port of Rotterdam", "city": "Rotterdam", "country": "Netherlands"},
        {"location_id": "LOC00002", "location_name": "Shanghai Port Facility", "city": "Shanghai", "country": "China"},
        {"location_id": "LOC00003", "location_name": "Houston Logistics Hub", "city": "Houston", "country": "United States"},
    ])
    prods = pd.DataFrame([
        {"product_id": "PROD00001", "product_name": "Lithium-Ion Battery Cell", "category": "Energy Storage"},
        {"product_id": "PROD00002", "product_name": "Microcontroller 32-bit", "category": "Semiconductors"},
    ])
    thresholds = LinkingThresholds(
        supplier_threshold=0.60,
        location_threshold=0.55,
        product_threshold=0.55,
    )
    return EntityLinker(suppliers_df=sups, locations_df=locs, products_df=prods, thresholds=thresholds)


def test_exact_matching(mock_linker):
    """Verify Stage 1: Exact matching identifies exact names or IDs with highest confidence."""
    # Match by exact name
    res = mock_linker.link_supplier("Apex Semiconductor Manufacturing Inc.")
    assert res["linked_id"] == "SUP00001"
    assert res["match_method"] == "exact"
    assert res["match_score"] == 1.0
    assert res["confidence"] >= 0.95

    # Match by exact ID
    res_id = mock_linker.link_supplier("SUP00002")
    assert res_id["linked_id"] == "SUP00002"
    assert res_id["match_method"] == "exact"


def test_normalized_matching(mock_linker):
    """Verify Stage 2: Normalized matching strips corporate suffixes and aligns canonical forms."""
    # Suffix difference: 'Corp' omitted
    res = mock_linker.link_supplier("Global Logistics")
    assert res["linked_id"] == "SUP00002"
    assert res["match_method"] == "normalized"
    assert res["confidence"] >= 0.90

    # Port prefix variation: 'Rotterdam Port' vs 'Port of Rotterdam'
    res_loc = mock_linker.link_location("Rotterdam Port")
    assert res_loc["linked_id"] == "LOC00001"
    assert res_loc["match_method"] == "normalized"


def test_alias_matching(mock_linker):
    """Verify Stage 3: Alias matching maps country and port abbreviations."""
    # 'USA' maps to United States -> Houston Logistics Hub
    res = mock_linker.link_location("USA")
    assert res["linked_id"] == "LOC00003"
    assert res["match_method"] == "alias"
    assert res["confidence"] >= 0.85


def test_controlled_fuzzy_matching_and_numeric_conflict(mock_linker):
    """Verify Stage 4: Fuzzy matching handles minor typos, but strictly penalizes numeric token conflicts."""
    # Minor typo in name
    res = mock_linker.link_product("Lithium-Ion Battery Cel")
    assert res["linked_id"] == "PROD00001"
    assert res["match_method"] == "fuzzy"
    assert res["match_score"] >= 0.85

    # Numeric conflict check: SUP00001 vs SUP00002
    conflict = check_numeric_conflict("Supplier 0001", "Supplier 0002")
    assert conflict is True

    # Attempting to match 'Supplier 0009' against 'Apex Semiconductor Manufacturing Inc.' (no number 9)
    res_unknown = mock_linker.link_supplier("Supplier 0099")
    assert res_unknown["linked_id"] == "UNKNOWN"
    assert res_unknown["match_method"] == "unresolved"
    assert res_unknown["confidence"] == 0.0


def test_below_threshold_unknown(mock_linker):
    """Verify that unlinked or low-confidence mentions remain UNKNOWN without creating fake entities."""
    res = mock_linker.link_supplier("CompletelyNonExistentSupplyChainEntityXYZ")
    assert res["linked_id"] == "UNKNOWN"
    assert res["match_method"] == "unresolved"
    assert res["confidence"] == 0.0

    res_loc = mock_linker.link_location("Atlantis Deep Sea Facility")
    assert res_loc["linked_id"] == "UNKNOWN"
    assert res_loc["match_method"] == "unresolved"


def test_linking_quality_report(mock_linker):
    """Verify generation of linking-quality report covering exact, fuzzy, unresolved, and averages."""
    mentions = [
        ("Apex Semiconductor Manufacturing Inc.", "SUPPLIER"),  # exact
        ("Global Logistics", "SUPPLIER"),                     # normalized
        ("USA", "LOCATION"),                                  # alias
        ("Lithium-Ion Battery Cel", "PRODUCT"),               # fuzzy
        ("Unknown Alien Mining Facility", "LOCATION"),         # unresolved
    ]

    results = [mock_linker.link_entity(text, etype) for text, etype in mentions]
    report = mock_linker.generate_quality_report(results)

    assert report["total_entities"] == 5
    assert report["exact_matches"] == 1
    assert report["normalized_matches"] == 1
    assert report["alias_matches"] == 1
    assert report["fuzzy_matches"] == 1
    assert report["unresolved_entities"] == 1
    assert 0.0 < report["average_confidence"] < 1.0
    assert report["unresolved_percentage"] == "20.0%"


def test_deterministic_reproducibility(mock_linker):
    """Verify that repeated linking runs produce bit-for-bit identical outputs."""
    mention = "Pacific Battery Systems"
    res1 = mock_linker.link_supplier(mention)
    res2 = mock_linker.link_supplier(mention)
    assert res1 == res2
    assert res1["linked_id"] == "SUP00003"


def test_storage_persistence(tmp_path, mock_linker):
    """Verify saving entity links into isolated CSV and duplicate avoidance."""
    links = [
        mock_linker.link_entity("Apex Semiconductor Manufacturing Inc.", "SUPPLIER"),
        mock_linker.link_entity("Unknown Entity", "LOCATION"),
    ]

    test_csv = str(tmp_path / "entity_links.csv")
    stored = save_entity_links(links, csv_path=test_csv)
    assert stored == 2
    assert os.path.exists(test_csv)

    df = pd.read_csv(test_csv)
    assert len(df) == 2
    assert "entity_text" in df.columns
    assert "linked_id" in df.columns
    assert "match_method" in df.columns
    assert "confidence" in df.columns

    # Re-saving identical links should add 0 duplicates
    stored_again = save_entity_links(links, csv_path=test_csv)
    assert stored_again == 0
