"""Normalization and canonicalization utilities for Entity Linking."""
from __future__ import annotations
import re
from typing import Set

# Corporate stop-suffixes to strip for clean canonical matching
_CORP_SUFFIXES = re.compile(
    r"\b(inc\.?|incorporated|corp\.?|corporation|llc\.?|ltd\.?|limited|co\.?|company|enterprises|holdings|group|pvt\.?|s\.?a\.?|plc\.?|technologies|industries|logistics|distributor|suppliers?)\b",
    re.IGNORECASE,
)

# Common geographic / country aliases
COUNTRY_ALIASES: dict[str, str] = {
    "us": "united states",
    "usa": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "great britain": "united kingdom",
    "prc": "china",
    "people's republic of china": "china",
    "uae": "united arab emirates",
    "u.a.e.": "united arab emirates",
    "korea": "south korea",
    "rok": "south korea",
    "republic of korea": "south korea",
    "in": "india",
    "de": "germany",
    "jp": "japan",
    "sg": "singapore",
    "nl": "netherlands",
}

# Common port prefixes/suffixes
_PORT_PREFIX_RE = re.compile(r"^port\s+(?:of\s+)?", re.IGNORECASE)


def normalize_linking_text(text: str, strip_corporate: bool = True) -> str:
    """Normalize entity or master data string for robust matching.
    
    Operations:
      - Lowercase
      - Strip corporate suffixes (optional)
      - Standardize 'Port of X' -> 'X port'
      - Replace non-alphanumeric chars with spaces
      - Collapse whitespace
    """
    if not text:
        return ""
    t = str(text).lower().strip()

    # Normalize port patterns: 'port of rotterdam' -> 'rotterdam port'
    if _PORT_PREFIX_RE.match(t):
        port_core = _PORT_PREFIX_RE.sub("", t).strip()
        t = f"{port_core} port"

    # Corporate suffix stripping
    if strip_corporate:
        prev = None
        while t != prev:
            prev = t
            t = _CORP_SUFFIXES.sub("", t)
            t = re.sub(r"[\s,.;:-]+$", "", t).strip()

    # Check country aliases
    if t in COUNTRY_ALIASES:
        t = COUNTRY_ALIASES[t]

    # Clean punctuation
    t = re.sub(r"[^\w\s]", " ", t)
    t = " ".join(t.split())
    return t


def extract_numeric_tokens(text: str) -> set[str]:
    """Extract all standalone or padded numeric tokens from text."""
    if not text:
        return set()
    nums = re.findall(r"\b\d+\b", str(text))
    # Normalize numbers to unpadded and zero-padded forms for cross-comparison
    res = set()
    for n in nums:
        res.add(n)
        try:
            res.add(str(int(n)))
        except ValueError:
            pass
    return res


def check_numeric_conflict(text1: str, text2: str) -> bool:
    """Return True if both strings contain explicit numeric identifiers and they conflict."""
    n1 = extract_numeric_tokens(text1)
    n2 = extract_numeric_tokens(text2)
    if n1 and n2 and not (n1 & n2):
        return True
    return False
