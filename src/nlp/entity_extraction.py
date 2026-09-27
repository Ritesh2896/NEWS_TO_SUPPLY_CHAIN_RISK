"""Local entity extraction using spaCy NER and domain-specific supply chain rule/gazetteer matching.

Extracts:
  - organizations / suppliers
  - locations / countries / cities
  - ports
  - facilities
  - products

Outputs structured spans with character offsets and confidence scores for news_entities.csv.
"""
from __future__ import annotations
import os
import re
from typing import Any, Sequence
import pandas as pd

from src.nlp.normalization import normalize_entity_name, normalize_entity_type
from src.nlp.preprocessing import clean_text

# Lazy load spaCy
_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            try:
                import spacy
                _NLP = spacy.blank("en")
            except Exception:
                _NLP = None
    return _NLP


# Common ports
KNOWN_PORTS = [
    "Port of Shanghai", "Shanghai Port", "Port of Singapore", "Singapore Port",
    "Port of Ningbo", "Ningbo Port", "Port of Rotterdam", "Rotterdam Port",
    "Port of Los Angeles", "Los Angeles Port", "Port of Long Beach", "Long Beach Port",
    "Port of Antwerp", "Antwerp Port", "Port of Busan", "Busan Port",
    "Port of Hamburg", "Hamburg Port", "Port of Kaohsiung", "Kaohsiung Port",
    "Port of Jebel Ali", "Jebel Ali Port", "Jawaharlal Nehru Port", "Nhava Sheva",
    "Mundra Port", "Chennai Port", "Port of Felixstowe", "Port of Santos",
    "Port of Colombo", "Port of Tanjung Pelepas", "Kolkata Port", "Visakhapatnam Port"
]

# Common countries
KNOWN_COUNTRIES = [
    "United States", "USA", "U.S.", "China", "India", "Germany", "Japan",
    "South Korea", "Taiwan", "Vietnam", "Mexico", "United Kingdom", "UK",
    "Netherlands", "Singapore", "Malaysia", "Thailand", "Indonesia", "Brazil",
    "Canada", "Australia", "France", "Italy", "Spain", "Russia", "Turkey",
    "Saudi Arabia", "United Arab Emirates", "UAE", "Egypt", "South Africa"
]

# Common supply-chain cities
KNOWN_CITIES = [
    "Shanghai", "Shenzhen", "Ningbo", "Guangzhou", "Beijing", "Singapore",
    "Tokyo", "Seoul", "Busan", "Taipei", "Mumbai", "Delhi", "Chennai",
    "Bengaluru", "Mundra", "Kolkata", "Rotterdam", "Hamburg", "Antwerp",
    "Frankfurt", "London", "Los Angeles", "Long Beach", "Chicago", "Houston",
    "New York", "Detroit", "Hanoi", "Ho Chi Minh City", "Bangkok", "Jakarta"
]

# Common supply chain products
KNOWN_PRODUCTS = [
    "Semiconductor", "Semiconductors", "Microchip", "Microchips", "Memory Chip",
    "Lithium Battery", "Lithium Batteries", "Automotive Parts", "Steel Coils",
    "Crude Oil", "Refined Petroleum", "Solar Panels", "Pharmaceuticals",
    "Active Pharmaceutical Ingredients", "Electronic Components", "Copper Cathodes",
    "Aluminum Ingots", "Wheat", "Fertilizer", "Natural Gas", "Medical Devices"
]

# Regex patterns for domain-specific entities
_PORT_RE = re.compile(
    r"\b(?:Port\s+of\s+[A-Z][a-zA-Z.-]+(?:\s+[A-Z][a-zA-Z.-]+){0,2}|[A-Z][a-zA-Z.-]+\s+(?:Port|Harbor|Harbour|Container\s+Terminal))\b"
)
_FACILITY_RE = re.compile(
    r"\b[A-Z][a-zA-Z0-9.-]+(?:\s+[A-Z][a-zA-Z0-9.-]+){0,3}\s+(?:Plant|Factory|Refinery|Smelter|Foundry|Complex|Hub|Depot|Assembly\s+Line)\b"
)
_ORG_FALLBACK_RE = re.compile(
    r"\b[A-Z][a-zA-Z0-9&.-]+(?:\s+[A-Z][a-zA-Z0-9&.-]+){1,3}(?:\s+(?:Inc\.?|Corp\.?|Ltd\.?|GmbH|Co\.?|LLC|Enterprises|Technologies|Logistics|Industries|Holdings))\b"
)


def _load_master_catalogs() -> tuple[set[str], set[str], set[str]]:
    """Optionally load supplier names, location names, and products from master CSVs if present."""
    suppliers, locations, products = set(), set(), set()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sup_file = os.path.join(root, "data/master/suppliers.csv")
    loc_file = os.path.join(root, "data/master/locations.csv")
    prod_file = os.path.join(root, "data/master/products.csv")

    if os.path.exists(sup_file):
        try:
            df = pd.read_csv(sup_file, usecols=["supplier_name"])
            suppliers = {str(x).strip() for x in df["supplier_name"].dropna() if len(str(x).strip()) > 2}
        except Exception:
            pass

    if os.path.exists(loc_file):
        try:
            df = pd.read_csv(loc_file, usecols=["location_name", "city", "country"])
            for c in ["location_name", "city", "country"]:
                locations.update({str(x).strip() for x in df[c].dropna() if len(str(x).strip()) > 2})
        except Exception:
            pass

    if os.path.exists(prod_file):
        try:
            df = pd.read_csv(prod_file, usecols=["product_name", "category"])
            for c in ["product_name", "category"]:
                products.update({str(x).strip() for x in df[c].dropna() if len(str(x).strip()) > 2})
        except Exception:
            pass

    return suppliers, locations, products


_CACHED_SUPPLIERS, _CACHED_LOCATIONS, _CACHED_PRODUCTS = None, None, None

def _get_master_catalogs():
    global _CACHED_SUPPLIERS, _CACHED_LOCATIONS, _CACHED_PRODUCTS
    if _CACHED_SUPPLIERS is None:
        _CACHED_SUPPLIERS, _CACHED_LOCATIONS, _CACHED_PRODUCTS = _load_master_catalogs()
    return _CACHED_SUPPLIERS, _CACHED_LOCATIONS, _CACHED_PRODUCTS


def extract_entity_spans(text: str) -> list[dict[str, Any]]:
    """Extract entities with character offsets, normalized forms, and confidence scores.
    
    Returns list of dicts:
      [
        {
          "entity_text": "Port of Rotterdam",
          "entity_type": "PORT",
          "normalized_entity": "ROTTERDAM PORT",
          "confidence": 0.95,
          "character_start": 12,
          "character_end": 29
        },
        ...
      ]
    """
    raw_text = str(text or "")
    if not raw_text.strip():
        return []

    spans: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()

    def add_span(entity_text: str, entity_type: str, start: int, end: int, confidence: float):
        span_key = (start, end)
        if span_key in seen_spans or start >= end:
            return
        # Avoid overlapping substring duplicates
        for s, e in seen_spans:
            if s <= start and end <= e and (start, end) != (s, e):
                return
        seen_spans.add(span_key)
        spans.append({
            "entity_text": entity_text.strip(),
            "entity_type": normalize_entity_type(entity_type),
            "normalized_entity": normalize_entity_name(entity_text, entity_type),
            "confidence": round(float(confidence), 3),
            "character_start": start,
            "character_end": end,
        })

    # 1. Domain Pattern: Ports
    for m in _PORT_RE.finditer(raw_text):
        add_span(m.group(), "PORT", m.start(), m.end(), 0.95)
    for p in KNOWN_PORTS:
        pattern = re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(raw_text):
            add_span(m.group(), "PORT", m.start(), m.end(), 0.96)

    # 2. Domain Pattern: Facilities
    for m in _FACILITY_RE.finditer(raw_text):
        add_span(m.group(), "FACILITY", m.start(), m.end(), 0.92)

    # 3. Domain Pattern: Known Countries
    for c in KNOWN_COUNTRIES:
        pattern = re.compile(r"\b" + re.escape(c) + r"\b")
        for m in pattern.finditer(raw_text):
            add_span(m.group(), "COUNTRY", m.start(), m.end(), 0.94)

    # 4. Domain Pattern: Known Cities
    for city in KNOWN_CITIES:
        pattern = re.compile(r"\b" + re.escape(city) + r"\b")
        for m in pattern.finditer(raw_text):
            add_span(m.group(), "CITY", m.start(), m.end(), 0.93)

    # 5. Domain Pattern: Known Products
    for pr in KNOWN_PRODUCTS:
        pattern = re.compile(r"\b" + re.escape(pr) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(raw_text):
            add_span(m.group(), "PRODUCT", m.start(), m.end(), 0.90)

    # 6. Master Catalog Matching (Suppliers & Products)
    cat_suppliers, cat_locations, cat_products = _get_master_catalogs()
    for s in list(cat_suppliers)[:500]:
        if len(s) > 4:
            pattern = re.compile(r"\b" + re.escape(s) + r"\b", re.IGNORECASE)
            for m in pattern.finditer(raw_text):
                add_span(m.group(), "SUPPLIER", m.start(), m.end(), 0.95)

    for p in list(cat_products)[:300]:
        if len(p) > 4:
            pattern = re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE)
            for m in pattern.finditer(raw_text):
                add_span(m.group(), "PRODUCT", m.start(), m.end(), 0.91)

    # 7. spaCy NER (if available)
    nlp = _get_nlp()
    if nlp is not None:
        try:
            doc = nlp(raw_text)
            for ent in doc.ents:
                start, end = ent.start_char, ent.end_char
                if ent.label_ == "ORG":
                    add_span(ent.text, "ORGANIZATION", start, end, 0.88)
                elif ent.label_ in {"GPE", "LOC"}:
                    add_span(ent.text, "LOCATION", start, end, 0.87)
                elif ent.label_ == "FAC":
                    add_span(ent.text, "FACILITY", start, end, 0.89)
                elif ent.label_ == "PRODUCT":
                    add_span(ent.text, "PRODUCT", start, end, 0.86)
        except Exception:
            pass

    # 8. Regex Corporate Fallback
    for m in _ORG_FALLBACK_RE.finditer(raw_text):
        add_span(m.group(), "ORGANIZATION", m.start(), m.end(), 0.85)

    # Sort spans by character start
    spans.sort(key=lambda x: x["character_start"])
    return spans


def extract_entities(text: str) -> dict[str, Any]:
    """High-level entity extraction compatible with pipeline and evaluation modules.
    
    Returns:
      {
        "organizations": list[str],
        "suppliers": list[str],
        "locations": list[str],
        "countries": list[str],
        "cities": list[str],
        "ports": list[str],
        "facilities": list[str],
        "products": list[str],
        "entities": list[dict]  # full structured records
      }
    """
    spans = extract_entity_spans(text)
    
    def uniq_strs(items: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in items:
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(x.strip())
        return out

    orgs = uniq_strs([s["entity_text"] for s in spans if s["entity_type"] in {"ORGANIZATION", "SUPPLIER"}])
    suppliers = uniq_strs([s["entity_text"] for s in spans if s["entity_type"] == "SUPPLIER"])
    locs = uniq_strs([s["entity_text"] for s in spans if s["entity_type"] in {"LOCATION", "GPE", "CITY", "COUNTRY", "PORT"}])
    countries = uniq_strs([s["entity_text"] for s in spans if s["entity_type"] == "COUNTRY"])
    cities = uniq_strs([s["entity_text"] for s in spans if s["entity_type"] == "CITY"])
    ports = uniq_strs([s["entity_text"] for s in spans if s["entity_type"] == "PORT"])
    facilities = uniq_strs([s["entity_text"] for s in spans if s["entity_type"] == "FACILITY"])
    products = uniq_strs([s["entity_text"] for s in spans if s["entity_type"] == "PRODUCT"])

    # Fallback to locations from prep patterns if none found
    if not locs:
        patterns = [r"(?:in|at|near|around|from|to)\s+([A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+){0,2})"]
        for p in patterns:
            matches = re.findall(p, str(text or ""))
            for m in matches:
                if len(m) > 2 and m not in locs:
                    locs.append(m)

    return {
        "organizations": orgs,
        "suppliers": suppliers if suppliers else orgs,
        "locations": locs,
        "countries": countries,
        "cities": cities,
        "ports": ports,
        "facilities": facilities,
        "products": products,
        "entities": spans,
    }
