"""Deterministic deduplication module for incoming news articles.

Removes exact duplicate URLs and filters out near-duplicate article titles using
token-based string similarity.
"""
from __future__ import annotations
import re
from typing import Sequence
from src.news.models import NormalizedArticle

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    _HAS_RAPIDFUZZ = False


def _clean_title(title: str) -> str:
    """Normalize headline for comparison by lowercasing and removing punctuation."""
    t = str(title or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def _title_similarity(t1: str, t2: str) -> float:
    """Calculate similarity ratio between two normalized headlines in [0, 1]."""
    c1, c2 = _clean_title(t1), _clean_title(t2)
    if not c1 or not c2:
        return 0.0
    if c1 == c2:
        return 1.0

    if _HAS_RAPIDFUZZ:
        return fuzz.token_sort_ratio(c1, c2) / 100.0
    else:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, c1, c2).ratio()


def deduplicate_articles(
    articles: Sequence[NormalizedArticle],
    existing_urls: set[str] | None = None,
    existing_titles: list[str] | None = None,
    title_threshold: float = 0.80,
) -> tuple[list[NormalizedArticle], int]:
    """Filter duplicate articles by exact URL and near-duplicate title similarity.

    Returns:
        (unique_articles, duplicates_removed_count)
    """
    seen_urls = set(existing_urls or set())
    seen_titles = list(existing_titles or [])

    unique: list[NormalizedArticle] = []
    dup_count = 0

    for a in articles:
        url_clean = str(a.url or "").strip().lower()
        if not url_clean:
            dup_count += 1
            continue

        # 1. Exact URL deduplication
        if url_clean in seen_urls:
            dup_count += 1
            continue

        # 2. Near-duplicate title detection
        title_clean = str(a.title or "").strip()
        is_dup_title = False
        for old_t in seen_titles:
            if _title_similarity(title_clean, old_t) >= title_threshold:
                is_dup_title = True
                break

        if is_dup_title:
            dup_count += 1
            continue

        seen_urls.add(url_clean)
        seen_titles.append(title_clean)
        unique.append(a)

    return unique, dup_count
