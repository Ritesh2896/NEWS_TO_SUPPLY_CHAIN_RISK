"""High-Performance Entity Linking for BDS-35 Supply-Chain Disruption Pipeline.

Resolves extracted text mentions (organizations, locations, products) to canonical
master data IDs using RapidFuzz token-based string similarity, suffix normalization,
and confidence calibration.
"""
from __future__ import annotations
import re
from typing import Any
import pandas as pd

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    _HAS_RAPIDFUZZ = False

# Corporate suffixes to normalize for cleaner matching
CORP_SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|llc|ltd|limited|co|company|enterprises|holdings|group|pvt|distributor|suppliers?|logistics|industries)\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and removing corporate stop-suffixes."""
    text = str(text or "").lower()
    text = CORP_SUFFIXES.sub("", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _similarity(s1: str, s2: str) -> float:
    """Compute normalized string similarity in [0, 1] with numeric identifier awareness."""
    norm1, norm2 = normalize_text(s1), normalize_text(s2)
    if not norm1 or not norm2:
        return 0.0
    if norm1 == norm2:
        return 1.0

    # Extract numeric components (e.g., 0001, 0004)
    nums1 = set(re.findall(r"\b\d+\b", s1))
    nums2 = set(re.findall(r"\b\d+\b", s2))

    # Penalize if numbers conflict
    if nums1 and nums2 and not (nums1 & nums2):
        return 0.0

    if _HAS_RAPIDFUZZ:
        sort_score = fuzz.token_sort_ratio(norm1, norm2) / 100.0
        set_score = fuzz.token_set_ratio(norm1, norm2) / 100.0
        base_score = max(sort_score, set_score)
    else:
        from difflib import SequenceMatcher
        base_score = SequenceMatcher(None, norm1, norm2).ratio()

    # Boost if explicit numeric ID token matches
    if nums1 and nums2 and (nums1 & nums2):
        base_score = max(base_score, 0.92)

    return round(base_score, 4)


def link_supplier(
    organizations: list[str] | str,
    suppliers_df: pd.DataFrame,
    threshold: float = 0.55,
) -> dict[str, Any] | None:
    """Link extracted organization mentions to the best matching supplier in master data."""
    if suppliers_df is None or suppliers_df.empty:
        return None

    if isinstance(organizations, str):
        organizations = [organizations]
    if not organizations:
        return None

    best_match = None
    best_score = 0.0
    best_org = None

    # Pre-extract columns
    supplier_names = suppliers_df["supplier_name"].astype(str).tolist()
    supplier_ids = suppliers_df["supplier_id"].astype(str).tolist()

    for org in organizations:
        org_clean = str(org).strip()
        if len(org_clean) < 2:
            continue

        for sid, sname in zip(supplier_ids, supplier_names):
            score = _similarity(org_clean, sname)
            if score > best_score:
                best_score = score
                best_match = (sid, sname)
                best_org = org_clean
                if score >= 0.98:
                    break

    if best_match is None or best_score < threshold:
        return None

    sid, sname = best_match
    return {
        "entity_text": best_org,
        "supplier_id": sid,
        "supplier_name": sname,
        "linked_id": sid,
        "entity_type": "SUPPLIER",
        "match_score": round(best_score, 3),
        "confidence": round(min(1.0, best_score * 1.02), 3),
        "match_method": "rapidfuzz_token_sort" if _HAS_RAPIDFUZZ else "difflib_ratio",
        "data_status": "SYNTHETIC_DEMO",
    }


def link_location(
    locations: list[str] | str,
    locations_df: pd.DataFrame,
    threshold: float = 0.60,
) -> dict[str, Any] | None:
    """Link extracted location mention to canonical location in master data."""
    if locations_df is None or locations_df.empty:
        return None

    if isinstance(locations, str):
        locations = [locations]
    if not locations:
        return None

    best_match = None
    best_score = 0.0
    best_loc = None

    loc_ids = locations_df["location_id"].astype(str).tolist()
    loc_names = locations_df["location_name"].astype(str).tolist()
    cities = locations_df["city"].astype(str).tolist() if "city" in locations_df else [""] * len(loc_ids)
    countries = locations_df["country"].astype(str).tolist() if "country" in locations_df else [""] * len(loc_ids)

    for mention in locations:
        mention_clean = str(mention).strip()
        if len(mention_clean) < 2:
            continue

        for lid, lname, city, country in zip(loc_ids, loc_names, cities, countries):
            score_name = _similarity(mention_clean, lname)
            score_city = _similarity(mention_clean, city) if city else 0.0
            score_country = _similarity(mention_clean, country) if country else 0.0
            score = max(score_name, score_city, score_country)

            if score > best_score:
                best_score = score
                best_match = (lid, lname, city, country)
                best_loc = mention_clean
                if score >= 0.98:
                    break

    if best_match is None or best_score < threshold:
        return None

    lid, lname, city, country = best_match
    return {
        "entity_text": best_loc,
        "location_id": lid,
        "location_name": lname,
        "city": city,
        "country": country,
        "linked_id": lid,
        "entity_type": "LOCATION",
        "match_score": round(best_score, 3),
        "confidence": round(best_score, 3),
        "match_method": "rapidfuzz_location_multi_field" if _HAS_RAPIDFUZZ else "difflib_ratio",
        "data_status": "SYNTHETIC_DEMO",
    }


def link_product(
    product_mentions: list[str] | str,
    products_df: pd.DataFrame,
    threshold: float = 0.55,
) -> dict[str, Any] | None:
    """Link extracted product mentions to canonical products in master data."""
    if products_df is None or products_df.empty:
        return None

    if isinstance(product_mentions, str):
        product_mentions = [product_mentions]
    if not product_mentions:
        return None

    best_match = None
    best_score = 0.0
    best_mention = None

    prod_ids = products_df["product_id"].astype(str).tolist()
    prod_names = products_df["product_name"].astype(str).tolist()
    categories = products_df["category"].astype(str).tolist() if "category" in products_df else [""] * len(prod_ids)

    for mention in product_mentions:
        m_clean = str(mention).strip()
        if len(m_clean) < 2:
            continue

        for pid, pname, cat in zip(prod_ids, prod_names, categories):
            s_name = _similarity(m_clean, pname)
            s_cat = _similarity(m_clean, cat) if cat else 0.0
            score = max(s_name, s_cat * 0.85)

            if score > best_score:
                best_score = score
                best_match = (pid, pname, cat)
                best_mention = m_clean
                if score >= 0.98:
                    break

    if best_match is None or best_score < threshold:
        return None

    pid, pname, cat = best_match
    return {
        "entity_text": best_mention,
        "product_id": pid,
        "product_name": pname,
        "category": cat,
        "linked_id": pid,
        "entity_type": "PRODUCT",
        "match_score": round(best_score, 3),
        "confidence": round(best_score, 3),
        "match_method": "rapidfuzz_product_match" if _HAS_RAPIDFUZZ else "difflib_ratio",
        "data_status": "SYNTHETIC_DEMO",
    }
