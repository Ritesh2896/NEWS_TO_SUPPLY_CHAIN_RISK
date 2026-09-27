"""Storage layer for raw NewsAPI responses and normalized processed articles.

Requirements:
  - Stores raw JSON under data/raw/newsapi/
  - Stores/appends normalized articles under data/processed/news.csv
  - Strictly preserves data_status and provenance fields
"""
from __future__ import annotations
import json
import os
import re
from datetime import datetime, timezone
from typing import Any
import pandas as pd

from src.news.models import NormalizedArticle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(ROOT, "data/raw/newsapi")
PROCESSED_NEWS_PATH = os.path.join(ROOT, "data/processed/news.csv")
LIVE_CACHE_PATH = os.path.join(ROOT, "data/processed/live_news.csv")

NEWS_COLUMNS = [
    "article_id", "source_name", "author", "title", "description",
    "url", "published_at", "content", "query", "retrieved_at", "data_status"
]


def save_raw_response(
    raw_payload: dict[str, Any], query: str = "", output_dir: str | None = None
) -> str:
    """Save raw NewsAPI response JSON under data/raw/newsapi/ or custom directory."""
    target_dir = output_dir or RAW_DIR
    os.makedirs(target_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_query = re.sub(r"[^\w]+", "_", query.strip().lower())[:30] or "query"
    filename = f"{timestamp}_{clean_query}.json"
    full_path = os.path.join(target_dir, filename)

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(raw_payload, f, indent=2, ensure_ascii=False)

    return full_path


def save_normalized_articles(
    articles: list[NormalizedArticle],
    csv_path: str = PROCESSED_NEWS_PATH,
) -> int:
    """Append normalized articles into processed news.csv without creating duplicates."""
    if not articles:
        return 0

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    new_rows = [a.model_dump() for a in articles]
    new_df = pd.DataFrame(new_rows)
    for col in NEWS_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[NEWS_COLUMNS]

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, dtype=object).fillna("")
        existing_urls = set(existing_df["url"].str.strip().str.lower())
        existing_ids = set(existing_df["article_id"].str.strip())

        # Filter out rows whose URL or article_id already exists in the master news table
        filtered_df = new_df[
            (~new_df["url"].str.strip().str.lower().isin(existing_urls)) &
            (~new_df["article_id"].str.strip().isin(existing_ids))
        ]

        if not filtered_df.empty:
            combined_df = pd.concat([existing_df, filtered_df], ignore_index=True)
            combined_df.to_csv(csv_path, index=False)
            stored_count = len(filtered_df)
        else:
            stored_count = 0
    else:
        new_df.to_csv(csv_path, index=False)
        stored_count = len(new_df)

    # Also write fresh articles to data/processed/live_news.csv for runtime consumption
    os.makedirs(os.path.dirname(LIVE_CACHE_PATH), exist_ok=True)
    new_df.to_csv(LIVE_CACHE_PATH, index=False)

    return stored_count
