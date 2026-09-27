"""Deterministic, reproducible Entity Linker for BDS-35 Supply-Chain master data.

Implements a 4-stage cascade:
  1. Exact matching
  2. Normalized string matching
  3. Alias matching
  4. Controlled fuzzy matching (with numeric conflict penalty)

Resolves to SUPPLIER, LOCATION, and PRODUCT master records.
If confidence is below threshold, linked_id = UNKNOWN (never hallucinates real entities).
"""
from __future__ import annotations
import os
import re
from typing import Any, Sequence
import pandas as pd

from src.linking.confidence import LinkingThresholds, calibrate_confidence
from src.linking.normalization import (
    COUNTRY_ALIASES,
    check_numeric_conflict,
    extract_numeric_tokens,
    normalize_linking_text,
)

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    _HAS_RAPIDFUZZ = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_ENTITY_LINKS_PATH = os.path.join(ROOT, "data/processed/entity_links.csv")
LIVE_ENTITY_LINKS_PATH = os.path.join(ROOT, "data/processed/entity_links_live.csv")

ENTITY_LINKS_COLUMNS = [
    "entity_text", "linked_id", "entity_type", "match_method",
    "match_score", "confidence", "data_status"
]


class EntityLinker:
    """Deterministic Entity Linker with 4-stage matching cascade and quality metrics."""

    def __init__(
        self,
        suppliers_df: pd.DataFrame | None = None,
        locations_df: pd.DataFrame | None = None,
        products_df: pd.DataFrame | None = None,
        thresholds: LinkingThresholds | None = None,
    ):
        self.thresholds = thresholds or LinkingThresholds()
        self.suppliers_df = suppliers_df
        self.locations_df = locations_df
        self.products_df = products_df

        self._ensure_master_data_loaded()
        self._index_master_data()

    def _ensure_master_data_loaded(self):
        """Load master tables from data/master/ if not provided."""
        if self.suppliers_df is None:
            path = os.path.join(ROOT, "data/master/suppliers.csv")
            self.suppliers_df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

        if self.locations_df is None:
            path = os.path.join(ROOT, "data/master/locations.csv")
            self.locations_df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

        if self.products_df is None:
            path = os.path.join(ROOT, "data/master/products.csv")
            self.products_df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    def _index_master_data(self):
        """Build normalized in-memory index for high-speed, deterministic candidate search."""
        # 1. Suppliers
        self.suppliers_index = []
        if not self.suppliers_df.empty:
            for _, row in self.suppliers_df.iterrows():
                sid = str(row.get("supplier_id", "")).strip()
                sname = str(row.get("supplier_name", "")).strip()
                norm_name = normalize_linking_text(sname, strip_corporate=True)
                raw_lower = sname.lower()
                self.suppliers_index.append({
                    "id": sid,
                    "name": sname,
                    "raw_lower": raw_lower,
                    "norm_name": norm_name,
                    "nums": extract_numeric_tokens(sname),
                })
        # Sort index deterministically by ID
        self.suppliers_index.sort(key=lambda x: x["id"])

        # 2. Locations
        self.locations_index = []
        if not self.locations_df.empty:
            for _, row in self.locations_df.iterrows():
                lid = str(row.get("location_id", "")).strip()
                lname = str(row.get("location_name", "")).strip()
                city = str(row.get("city", "")).strip()
                country = str(row.get("country", "")).strip()
                norm_name = normalize_linking_text(lname, strip_corporate=False)
                norm_city = normalize_linking_text(city, strip_corporate=False)
                norm_country = normalize_linking_text(country, strip_corporate=False)
                self.locations_index.append({
                    "id": lid,
                    "name": lname,
                    "city": city,
                    "country": country,
                    "raw_lower": lname.lower(),
                    "norm_name": norm_name,
                    "norm_city": norm_city,
                    "norm_country": norm_country,
                    "nums": extract_numeric_tokens(lname),
                })
        self.locations_index.sort(key=lambda x: x["id"])

        # 3. Products
        self.products_index = []
        if not self.products_df.empty:
            for _, row in self.products_df.iterrows():
                pid = str(row.get("product_id", "")).strip()
                pname = str(row.get("product_name", "")).strip()
                cat = str(row.get("category", "")).strip()
                norm_name = normalize_linking_text(pname, strip_corporate=False)
                norm_cat = normalize_linking_text(cat, strip_corporate=False)
                self.products_index.append({
                    "id": pid,
                    "name": pname,
                    "category": cat,
                    "raw_lower": pname.lower(),
                    "norm_name": norm_name,
                    "norm_cat": norm_cat,
                    "nums": extract_numeric_tokens(pname),
                })
        self.products_index.sort(key=lambda x: x["id"])

    def _fuzzy_similarity(self, s1_norm: str, s2_norm: str, nums1: set[str], nums2: set[str]) -> float:
        """Compute string similarity with numeric consistency check."""
        if not s1_norm or not s2_norm:
            return 0.0
        if s1_norm == s2_norm:
            return 1.0

        # Numeric conflict penalty
        if nums1 and nums2 and not (nums1 & nums2):
            return 0.0

        if _HAS_RAPIDFUZZ:
            sort_score = fuzz.token_sort_ratio(s1_norm, s2_norm) / 100.0
            set_score = fuzz.token_set_ratio(s1_norm, s2_norm) / 100.0
            base_score = max(sort_score, set_score)
        else:
            from difflib import SequenceMatcher
            base_score = SequenceMatcher(None, s1_norm, s2_norm).ratio()

        # Bonus if explicit numeric ID token matches
        if nums1 and nums2 and (nums1 & nums2):
            base_score = max(base_score, 0.92)

        return round(base_score, 4)

    def link_supplier(self, mention: str, threshold: float | None = None) -> dict[str, Any]:
        """Link an organization/supplier mention using the 4-stage cascade."""
        tau = threshold if threshold is not None else self.thresholds.supplier_threshold
        raw = str(mention or "").strip()
        if not raw or not self.suppliers_index:
            return self._unresolved_result(raw, "SUPPLIER")

        raw_lower = raw.lower()
        norm_mention = normalize_linking_text(raw, strip_corporate=True)
        mention_nums = extract_numeric_tokens(raw)

        # Stage 1: Exact matching
        for cand in self.suppliers_index:
            if raw_lower == cand["raw_lower"] or raw.upper() == cand["id"]:
                return {
                    "entity_text": raw,
                    "linked_id": cand["id"],
                    "entity_type": "SUPPLIER",
                    "match_method": "exact",
                    "match_score": 1.0,
                    "confidence": calibrate_confidence("exact", 1.0, self.thresholds),
                    "target_name": cand["name"],
                }

        # Stage 2: Normalized string matching
        for cand in self.suppliers_index:
            if norm_mention and norm_mention == cand["norm_name"]:
                if not (mention_nums and cand["nums"] and not (mention_nums & cand["nums"])):
                    return {
                        "entity_text": raw,
                        "linked_id": cand["id"],
                        "entity_type": "SUPPLIER",
                        "match_method": "normalized",
                        "match_score": 0.98,
                        "confidence": calibrate_confidence("normalized", 0.98, self.thresholds),
                        "target_name": cand["name"],
                    }

        # Stage 3: Alias matching (corporate acronyms or prefixes)
        # Check if mention matches normalized without suffix or prefix
        for cand in self.suppliers_index:
            if len(norm_mention) > 3 and (norm_mention in cand["norm_name"] or cand["norm_name"] in norm_mention):
                if not (mention_nums and cand["nums"] and not (mention_nums & cand["nums"])):
                    # high substring containment
                    ratio = min(len(norm_mention), len(cand["norm_name"])) / max(len(norm_mention), len(cand["norm_name"]))
                    if ratio >= 0.80:
                        return {
                            "entity_text": raw,
                            "linked_id": cand["id"],
                            "entity_type": "SUPPLIER",
                            "match_method": "alias",
                            "match_score": round(0.90 * ratio, 3),
                            "confidence": calibrate_confidence("alias", round(0.90 * ratio, 3), self.thresholds),
                            "target_name": cand["name"],
                        }

        # Stage 4: Controlled fuzzy matching
        best_cand = None
        best_score = 0.0

        for cand in self.suppliers_index:
            score = self._fuzzy_similarity(norm_mention, cand["norm_name"], mention_nums, cand["nums"])
            if score > best_score:
                best_score = score
                best_cand = cand
                if score >= 0.98:
                    break

        if best_cand is not None and best_score >= tau:
            return {
                "entity_text": raw,
                "linked_id": best_cand["id"],
                "entity_type": "SUPPLIER",
                "match_method": "fuzzy",
                "match_score": round(best_score, 3),
                "confidence": calibrate_confidence("fuzzy", best_score, self.thresholds),
                "target_name": best_cand["name"],
            }

        return self._unresolved_result(raw, "SUPPLIER", best_score)

    def link_location(self, mention: str, threshold: float | None = None) -> dict[str, Any]:
        """Link a geographic mention to master locations using the 4-stage cascade."""
        tau = threshold if threshold is not None else self.thresholds.location_threshold
        raw = str(mention or "").strip()
        if not raw or not self.locations_index:
            return self._unresolved_result(raw, "LOCATION")

        raw_lower = raw.lower()
        norm_mention = normalize_linking_text(raw, strip_corporate=False)
        mention_nums = extract_numeric_tokens(raw)

        # Stage 1: Exact matching
        for cand in self.locations_index:
            if raw_lower == cand["raw_lower"] or raw.upper() == cand["id"]:
                return {
                    "entity_text": raw,
                    "linked_id": cand["id"],
                    "entity_type": "LOCATION",
                    "match_method": "exact",
                    "match_score": 1.0,
                    "confidence": calibrate_confidence("exact", 1.0, self.thresholds),
                    "target_name": cand["name"],
                }

        # Stage 2: Normalized string matching
        for cand in self.locations_index:
            if norm_mention and (norm_mention == cand["norm_name"] or norm_mention == cand["norm_city"]):
                if not (mention_nums and cand["nums"] and not (mention_nums & cand["nums"])):
                    return {
                        "entity_text": raw,
                        "linked_id": cand["id"],
                        "entity_type": "LOCATION",
                        "match_method": "normalized",
                        "match_score": 0.98,
                        "confidence": calibrate_confidence("normalized", 0.98, self.thresholds),
                        "target_name": cand["name"],
                    }

        # Stage 3: Alias matching (Country / City / Port Aliases)
        alias_norm = COUNTRY_ALIASES.get(raw_lower, norm_mention)
        for cand in self.locations_index:
            if cand["norm_country"] and alias_norm == cand["norm_country"]:
                return {
                    "entity_text": raw,
                    "linked_id": cand["id"],
                    "entity_type": "LOCATION",
                    "match_method": "alias",
                    "match_score": 0.92,
                    "confidence": calibrate_confidence("alias", 0.92, self.thresholds),
                    "target_name": cand["name"],
                }

        # Stage 4: Controlled fuzzy matching
        best_cand = None
        best_score = 0.0

        for cand in self.locations_index:
            s_name = self._fuzzy_similarity(norm_mention, cand["norm_name"], mention_nums, cand["nums"])
            s_city = self._fuzzy_similarity(norm_mention, cand["norm_city"], mention_nums, cand["nums"]) if cand["norm_city"] else 0.0
            s_country = self._fuzzy_similarity(norm_mention, cand["norm_country"], mention_nums, set()) if cand["norm_country"] else 0.0
            score = max(s_name, s_city, s_country)

            if score > best_score:
                best_score = score
                best_cand = cand
                if score >= 0.98:
                    break

        if best_cand is not None and best_score >= tau:
            return {
                "entity_text": raw,
                "linked_id": best_cand["id"],
                "entity_type": "LOCATION",
                "match_method": "fuzzy",
                "match_score": round(best_score, 3),
                "confidence": calibrate_confidence("fuzzy", best_score, self.thresholds),
                "target_name": best_cand["name"],
            }

        return self._unresolved_result(raw, "LOCATION", best_score)

    def link_product(self, mention: str, threshold: float | None = None) -> dict[str, Any]:
        """Link a product mention to master products using the 4-stage cascade."""
        tau = threshold if threshold is not None else self.thresholds.product_threshold
        raw = str(mention or "").strip()
        if not raw or not self.products_index:
            return self._unresolved_result(raw, "PRODUCT")

        raw_lower = raw.lower()
        norm_mention = normalize_linking_text(raw, strip_corporate=False)
        mention_nums = extract_numeric_tokens(raw)

        # Stage 1: Exact matching
        for cand in self.products_index:
            if raw_lower == cand["raw_lower"] or raw.upper() == cand["id"]:
                return {
                    "entity_text": raw,
                    "linked_id": cand["id"],
                    "entity_type": "PRODUCT",
                    "match_method": "exact",
                    "match_score": 1.0,
                    "confidence": calibrate_confidence("exact", 1.0, self.thresholds),
                    "target_name": cand["name"],
                }

        # Stage 2: Normalized string matching
        for cand in self.products_index:
            if norm_mention and norm_mention == cand["norm_name"]:
                if not (mention_nums and cand["nums"] and not (mention_nums & cand["nums"])):
                    return {
                        "entity_text": raw,
                        "linked_id": cand["id"],
                        "entity_type": "PRODUCT",
                        "match_method": "normalized",
                        "match_score": 0.98,
                        "confidence": calibrate_confidence("normalized", 0.98, self.thresholds),
                        "target_name": cand["name"],
                    }

        # Stage 3: Alias matching (e.g. Category match)
        for cand in self.products_index:
            if cand["norm_cat"] and norm_mention == cand["norm_cat"]:
                return {
                    "entity_text": raw,
                    "linked_id": cand["id"],
                    "entity_type": "PRODUCT",
                    "match_method": "alias",
                    "match_score": 0.88,
                    "confidence": calibrate_confidence("alias", 0.88, self.thresholds),
                    "target_name": cand["name"],
                }

        # Stage 4: Controlled fuzzy matching
        best_cand = None
        best_score = 0.0

        for cand in self.products_index:
            s_name = self._fuzzy_similarity(norm_mention, cand["norm_name"], mention_nums, cand["nums"])
            s_cat = (self._fuzzy_similarity(norm_mention, cand["norm_cat"], mention_nums, set()) * 0.85) if cand["norm_cat"] else 0.0
            score = max(s_name, s_cat)

            if score > best_score:
                best_score = score
                best_cand = cand
                if score >= 0.98:
                    break

        if best_cand is not None and best_score >= tau:
            return {
                "entity_text": raw,
                "linked_id": best_cand["id"],
                "entity_type": "PRODUCT",
                "match_method": "fuzzy",
                "match_score": round(best_score, 3),
                "confidence": calibrate_confidence("fuzzy", best_score, self.thresholds),
                "target_name": best_cand["name"],
            }

        return self._unresolved_result(raw, "PRODUCT", best_score)

    def link_entity(
        self,
        entity_text: str,
        entity_type: str,
        threshold: float | None = None,
        data_status: str = "REAL_DATA",
    ) -> dict[str, Any]:
        """Dispatch entity linking by type and return standard entity_links record."""
        t_upper = str(entity_type or "").upper()

        if t_upper in {"SUPPLIER", "ORG", "ORGANIZATION"}:
            res = self.link_supplier(entity_text, threshold)
        elif t_upper in {"LOCATION", "GPE", "LOC", "CITY", "COUNTRY", "PORT", "FACILITY"}:
            res = self.link_location(entity_text, threshold)
        elif t_upper in {"PRODUCT"}:
            res = self.link_product(entity_text, threshold)
        else:
            # Fallback attempt across supplier -> location -> product
            res = self.link_supplier(entity_text, threshold)
            if res["linked_id"] == "UNKNOWN":
                res = self.link_location(entity_text, threshold)
            if res["linked_id"] == "UNKNOWN":
                res = self.link_product(entity_text, threshold)

        res["data_status"] = data_status
        return res

    def _unresolved_result(
        self, entity_text: str, entity_type: str, score: float = 0.0
    ) -> dict[str, Any]:
        """Fallback result when confidence is below threshold or entity not found."""
        return {
            "entity_text": entity_text,
            "linked_id": "UNKNOWN",
            "entity_type": entity_type,
            "match_method": "unresolved",
            "match_score": round(score, 3),
            "confidence": 0.0,
            "target_name": "UNKNOWN",
        }

    def generate_quality_report(
        self, linked_records: Sequence[dict[str, Any]] | pd.DataFrame
    ) -> dict[str, Any]:
        """Generate linking-quality report covering match methods and unresolved stats."""
        if isinstance(linked_records, pd.DataFrame):
            records = linked_records.to_dict(orient="records")
        else:
            records = list(linked_records)

        total = len(records)
        if total == 0:
            return {
                "total_entities": 0,
                "exact_matches": 0,
                "normalized_matches": 0,
                "alias_matches": 0,
                "fuzzy_matches": 0,
                "unresolved_entities": 0,
                "average_confidence": 0.0,
                "unresolved_percentage": "0.0%",
            }

        exact_count = sum(1 for r in records if r.get("match_method") == "exact")
        norm_count = sum(1 for r in records if r.get("match_method") == "normalized")
        alias_count = sum(1 for r in records if r.get("match_method") == "alias")
        fuzzy_count = sum(1 for r in records if r.get("match_method") == "fuzzy")
        unresolved_count = sum(
            1 for r in records if r.get("linked_id") == "UNKNOWN" or r.get("match_method") == "unresolved"
        )

        confidences = [float(r.get("confidence", 0.0)) for r in records]
        avg_conf = sum(confidences) / total if total > 0 else 0.0
        unres_pct = (unresolved_count / total) * 100.0 if total > 0 else 0.0

        return {
            "total_entities": total,
            "exact_matches": exact_count,
            "normalized_matches": norm_count,
            "alias_matches": alias_count,
            "fuzzy_matches": fuzzy_count,
            "unresolved_entities": unresolved_count,
            "average_confidence": round(avg_conf, 4),
            "unresolved_percentage": f"{unres_pct:.1f}%",
        }


def save_entity_links(
    links: list[dict[str, Any]],
    csv_path: str = DEFAULT_ENTITY_LINKS_PATH,
) -> int:
    """Append linked entity records into entity_links.csv without duplicates."""
    if not links:
        return 0

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_df = pd.DataFrame(links)
    for col in ENTITY_LINKS_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[ENTITY_LINKS_COLUMNS]

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, dtype=object).fillna("")
        existing_keys = set(
            zip(
                existing_df["entity_text"].str.strip().str.lower(),
                existing_df["linked_id"].str.strip(),
                existing_df["entity_type"].str.strip(),
            )
        )
        filtered_df = new_df[
            ~new_df.apply(
                lambda r: (
                    str(r["entity_text"]).strip().lower(),
                    str(r["linked_id"]).strip(),
                    str(r["entity_type"]).strip(),
                ) in existing_keys,
                axis=1,
            )
        ]
        if not filtered_df.empty:
            combined_df = pd.concat([existing_df, filtered_df], ignore_index=True)
            combined_df.to_csv(csv_path, index=False)
            stored = len(filtered_df)
        else:
            stored = 0
    else:
        new_df.to_csv(csv_path, index=False)
        stored = len(new_df)

    # Also write fresh links to live cache
    os.makedirs(os.path.dirname(LIVE_ENTITY_LINKS_PATH), exist_ok=True)
    new_df.to_csv(LIVE_ENTITY_LINKS_PATH, index=False)

    return stored
