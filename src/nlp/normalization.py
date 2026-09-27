"""Entity and location normalization module for local NLP processing."""
from __future__ import annotations
import re

# Corporate suffixes to strip for canonical entity matching
_CORP_SUFFIX_RE = re.compile(
    r"\b(inc\.?|incorporated|corp\.?|corporation|ltd\.?|limited|llc\.?|gmbh|co\.?|company|pvt\.?|s\.?a\.?|plc\.?)\b",
    re.IGNORECASE,
)

# Country aliases
COUNTRY_ALIASES: dict[str, str] = {
    "us": "UNITED STATES",
    "usa": "UNITED STATES",
    "u.s.": "UNITED STATES",
    "u.s.a.": "UNITED STATES",
    "united states of america": "UNITED STATES",
    "uk": "UNITED KINGDOM",
    "u.k.": "UNITED KINGDOM",
    "great britain": "UNITED KINGDOM",
    "prc": "CHINA",
    "people's republic of china": "CHINA",
    "uae": "UNITED ARAB EMIRATES",
    "u.a.e.": "UNITED ARAB EMIRATES",
    "korea": "SOUTH KOREA",
    "rok": "SOUTH KOREA",
    "republic of korea": "SOUTH KOREA",
    "in": "INDIA",
    "de": "GERMANY",
    "jp": "JAPAN",
    "sg": "SINGAPORE",
}

# Standardized entity type mapping
ENTITY_TYPE_MAP: dict[str, str] = {
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "SUPPLIER": "SUPPLIER",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "CITY": "CITY",
    "COUNTRY": "COUNTRY",
    "PORT": "PORT",
    "FACILITY": "FACILITY",
    "PRODUCT": "PRODUCT",
    "FAC": "FACILITY",
    "NORP": "ORGANIZATION",
}


def normalize_entity_name(name: str, entity_type: str = "") -> str:
    """Produce canonical uppercase normalized string representation for an entity.
    
    Examples:
      'Apple Inc.' -> 'APPLE'
      'Port of Rotterdam' -> 'ROTTERDAM PORT'
      'USA' -> 'UNITED STATES'
    """
    if not name:
        return "UNKNOWN"

    raw = str(name).strip()
    lower = raw.lower()

    # 1. Check known country aliases
    if lower in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[lower]

    # 2. Check Port standardization: "Port of X" -> "X PORT"
    port_match = re.match(r"^port\s+(?:of\s+)?(.+)$", raw, re.IGNORECASE)
    if port_match:
        port_core = port_match.group(1).strip()
        return f"{port_core.upper()} PORT"
    if raw.lower().endswith(" port"):
        port_core = raw[:-5].strip()
        return f"{port_core.upper()} PORT"

    # 3. Strip corporate suffixes for organizations/suppliers
    etype_norm = ENTITY_TYPE_MAP.get(entity_type.upper(), entity_type.upper())
    if etype_norm in {"ORGANIZATION", "SUPPLIER", ""}:
        cleaned = raw
        prev = None
        while cleaned != prev:
            prev = cleaned
            cleaned = _CORP_SUFFIX_RE.sub("", cleaned)
            cleaned = re.sub(r"[\s,.;:-]+$", "", cleaned).strip()
        if len(cleaned) >= 2:
            raw = cleaned

    # 4. Collapse whitespace and uppercase
    canonical = re.sub(r"\s+", " ", raw).strip().upper()
    return canonical if canonical else "UNKNOWN"


def normalize_entity_type(raw_type: str) -> str:
    """Normalize raw entity tag into standard BDS-35 entity category."""
    t = str(raw_type or "").strip().upper()
    return ENTITY_TYPE_MAP.get(t, "OTHER")
