"""NewsAPI live news retrieval client with retry, rate-limit handling, and response normalization.

Requirements:
  - Reads NEWSAPI_API_KEY from environment
  - Never hardcodes API credentials
  - Supports configurable topic, hours, language, country, and limit
  - Handles rate limits (HTTP 429) and network timeouts gracefully
  - Normalizes raw payloads into NormalizedArticle models
"""
from __future__ import annotations
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any
import requests
from dotenv import load_dotenv

from src.logging_config import logger
from src.news.models import NormalizedArticle
from src.news.storage import save_raw_response

load_dotenv()

BASE_EVERYTHING_URL = "https://newsapi.org/v2/everything"
BASE_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"


class NewsApiClient:
    """Client for NewsAPI endpoints with error handling and normalization."""

    def __init__(self, api_key: str | None = None, max_retries: int = 2, timeout_seconds: int = 15):
        if api_key is not None:
            self.api_key = str(api_key).strip()
        else:
            self.api_key = (
                os.getenv("NEWSAPI_API_KEY", "").strip()
                or os.getenv("NEWSAPI_KEY", "").strip()
            )
            if not self.api_key:
                try:
                    import streamlit as st
                    if hasattr(st, "secrets"):
                        self.api_key = str(st.secrets.get("NEWSAPI_API_KEY") or st.secrets.get("NEWSAPI_KEY") or "").strip()
                    if not self.api_key and hasattr(st, "session_state") and "custom_newsapi_key" in st.session_state:
                        self.api_key = str(st.session_state.get("custom_newsapi_key") or "").strip()
                except Exception:
                    pass
        self.max_retries = max_retries
        self.timeout = timeout_seconds

    def is_configured(self) -> bool:
        """Check whether a valid API key is present."""
        return bool(self.api_key)

    def fetch_articles(
        self,
        topic: str,
        hours: int = 24,
        language: str = "en",
        country: str | None = None,
        limit: int = 10,
    ) -> tuple[list[NormalizedArticle], dict[str, Any]]:
        """Retrieve live news articles from NewsAPI.

        Returns:
            (normalized_articles, raw_response_dict)
        """
        if not self.api_key:
            logger.warning("NewsAPI key is not configured in environment.")
            return [], {
                "status": "error",
                "code": "apiKeyMissing",
                "message": "NEWSAPI_API_KEY is not configured in environment (.env).",
            }

        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=hours)

        # Decide endpoint: if country is provided, use top-headlines; else everything
        if country:
            url = BASE_HEADLINES_URL
            params: dict[str, Any] = {
                "q": topic,
                "country": country,
                "pageSize": min(limit, 100),
                "apiKey": self.api_key,
            }
        else:
            url = BASE_EVERYTHING_URL
            params = {
                "q": topic,
                "from": start_time.isoformat(timespec="seconds"),
                "to": now.isoformat(timespec="seconds"),
                "language": language,
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),
                "apiKey": self.api_key,
            }

        raw_data: dict[str, Any] = {}
        for attempt in range(1, self.max_retries + 2):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)

                # Graceful handling of rate limits (HTTP 429)
                if response.status_code == 429:
                    logger.warning("NewsAPI rate limit exceeded (HTTP 429).")
                    return [], {
                        "status": "error",
                        "code": "rateLimited",
                        "message": "NewsAPI rate limit reached (HTTP 429). Ingestion skipped gracefully.",
                    }

                # Graceful handling of authentication error (HTTP 401)
                if response.status_code == 401:
                    logger.warning("NewsAPI authentication failed (HTTP 401).")
                    return [], {
                        "status": "error",
                        "code": "apiKeyInvalid",
                        "message": "NewsAPI API key is invalid or unauthorized (HTTP 401).",
                    }

                response.raise_for_status()
                raw_data = response.json()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                logger.warning("NewsAPI attempt %d failed: %s", attempt, exc)
                if attempt <= self.max_retries:
                    time.sleep(1.0 * attempt)
                else:
                    logger.error("NewsAPI request timed out or connection failed after %d attempts.", attempt)
                    return [], {"status": "error", "message": str(exc)}
            except requests.exceptions.RequestException as exc:
                logger.error("NewsAPI request exception: %s", exc)
                return [], {"status": "error", "message": str(exc)}

        if raw_data.get("status") != "ok":
            msg = raw_data.get("message", "NewsAPI request failed.")
            logger.warning("NewsAPI response status: %s — %s", raw_data.get("status"), msg)
            return [], raw_data

        # Save raw JSON under data/raw/newsapi/
        try:
            save_raw_response(raw_data, query=topic)
        except Exception as exc:
            logger.warning("Could not persist raw NewsAPI payload: %s", exc)

        # Normalize articles
        raw_articles = raw_data.get("articles", [])
        normalized = self._normalize_articles(raw_articles, query=topic)
        return normalized[:limit], raw_data

    def _normalize_articles(
        self,
        raw_articles: list[dict[str, Any]],
        query: str,
    ) -> list[NormalizedArticle]:
        """Convert raw NewsAPI article objects into normalized NormalizedArticle models."""
        normalized: list[NormalizedArticle] = []
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        for idx, a in enumerate(raw_articles, 1):
            title = str(a.get("title") or "").strip()
            url = str(a.get("url") or "").strip()
            if not title or not url:
                continue

            source_obj = a.get("source") or {}
            source_name = source_obj.get("name") if isinstance(source_obj, dict) else str(source_obj)

            pub_raw = str(a.get("publishedAt") or "")
            pub_clean = pub_raw if pub_raw else now_iso

            # Deterministic article_id based on published timestamp or retrieval
            compact_time = "".join(c for c in pub_clean[:19] if c.isdigit()) or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            article_id = f"NEWSAPI-{compact_time}-{idx:04d}"

            desc = str(a.get("description") or "").strip()
            content = str(a.get("content") or desc).strip()
            author = str(a.get("author") or "").strip()

            article = NormalizedArticle(
                article_id=article_id,
                source_name=source_name or "NewsAPI Publisher",
                author=author,
                title=title,
                description=desc,
                url=url,
                published_at=pub_clean,
                content=content,
                query=query,
                retrieved_at=now_iso,
                data_status="REAL_DATA",
            )
            normalized.append(article)

        return normalized
