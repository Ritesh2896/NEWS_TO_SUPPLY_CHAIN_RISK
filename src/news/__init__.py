"""BDS-35 Live News Ingestion Package.

Integrates with NewsAPI to fetch, normalize, deduplicate, filter, and store
supply-chain disruption articles with complete provenance tracking.
"""
from src.news.client import NewsApiClient
from src.news.deduplication import deduplicate_articles
from src.news.models import IngestRequest, IngestResponse, NormalizedArticle
from src.news.relevance import check_relevance, filter_relevant_articles
from src.news.storage import save_normalized_articles, save_raw_response


def ingest_live_news(
    topic: str = "India ports suppliers logistics manufacturing disruptions",
    hours: int = 24,
    language: str = "en",
    country: str | None = None,
    limit: int = 10,
    client: NewsApiClient | None = None,
) -> IngestResponse:
    """Full live news ingestion pipeline:

    Fetch -> Deduplicate -> Filter Relevance -> Store raw JSON & processed CSV.
    """
    if client is None:
        client = NewsApiClient()

    # 1. Fetch raw articles from NewsAPI
    fetched_articles, _ = client.fetch_articles(
        topic=topic,
        hours=hours,
        language=language,
        country=country,
        limit=max(limit * 2, 20),
    )
    total_fetched = len(fetched_articles)

    # 2. Deterministic Deduplication
    unique_articles, dup_count = deduplicate_articles(fetched_articles)

    # 3. Supply-chain Relevance Filtering
    relevant_articles, irrelev_count = filter_relevant_articles(unique_articles)

    # 4. Limit to requested size
    final_articles = relevant_articles[:limit]

    # 5. Persist to storage
    stored_count = save_normalized_articles(final_articles)

    return IngestResponse(
        fetched=total_fetched,
        accepted=len(final_articles),
        duplicates_removed=dup_count,
        irrelevant_removed=irrelev_count,
        stored=stored_count,
        articles=final_articles,
    )


__all__ = [
    "NewsApiClient",
    "NormalizedArticle",
    "IngestRequest",
    "IngestResponse",
    "deduplicate_articles",
    "check_relevance",
    "filter_relevant_articles",
    "save_raw_response",
    "save_normalized_articles",
    "ingest_live_news",
]
