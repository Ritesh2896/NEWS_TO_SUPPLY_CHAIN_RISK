"""Pydantic data models for the BDS-35 Live News Ingestion module."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class NormalizedArticle(BaseModel):
    """Normalized article model matching BDS-35 news.csv schema."""

    article_id: str = Field(description="Unique article identifier, e.g., NEWSAPI-20260923-0001")
    source_name: str = Field(default="Unknown", description="Publisher or news agency name")
    author: Optional[str] = Field(default="", description="Author name if available")
    title: str = Field(description="Headline of the article")
    description: str = Field(default="", description="Short snippet or summary")
    url: str = Field(description="Canonical article URL")
    published_at: str = Field(description="Publication timestamp in ISO 8601")
    content: str = Field(default="", description="Extracted article body or preview text")
    query: str = Field(default="", description="Search query topic used to retrieve article")
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp when article was ingested",
    )
    data_status: str = Field(default="REAL_DATA", description="Provenance marker: REAL_DATA or SYNTHETIC_DEMO")


class IngestRequest(BaseModel):
    """Request payload for live news ingestion."""

    topic: str = Field(
        default="India ports suppliers logistics manufacturing disruptions",
        min_length=3,
        max_length=500,
        description="Search query topic for NewsAPI",
    )
    hours: int = Field(default=24, ge=1, le=168, description="Look-back hours from current time")
    language: str = Field(default="en", description="Two-letter ISO-639-1 language code")
    country: Optional[str] = Field(default=None, description="Optional 2-letter country code for top headlines")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of articles to return")


class IngestResponse(BaseModel):
    """Response payload detailing ingestion statistics and accepted articles."""

    fetched: int = Field(description="Total raw articles returned from NewsAPI")
    accepted: int = Field(description="Articles passing deduplication and relevance filters")
    duplicates_removed: int = Field(description="Number of duplicate articles discarded")
    irrelevant_removed: int = Field(description="Number of non-supply-chain articles filtered out")
    stored: int = Field(description="Number of newly saved articles")
    articles: list[NormalizedArticle] = Field(default_factory=list, description="List of accepted articles")
