"""Unit tests for Phase 2: Live News Ingestion module using mocked NewsAPI responses."""
import os
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app
from src.news.client import NewsApiClient
from src.news.deduplication import deduplicate_articles
from src.news.models import IngestRequest, NormalizedArticle
from src.news.relevance import check_relevance, filter_relevant_articles
from src.news.storage import save_normalized_articles, save_raw_response

client = TestClient(app)

MOCK_RAW_NEWSAPI_PAYLOAD = {
    "status": "ok",
    "totalResults": 2,
    "articles": [
        {
            "source": {"id": "reuters", "name": "Reuters"},
            "author": "Maritime Staff",
            "title": "Severe strike halts cargo handling at Port of Mumbai",
            "description": "Port workers initiate strike over wage negotiations, delaying container vessels.",
            "url": "https://example.com/mumbai-port-strike",
            "urlToImage": "https://example.com/img1.jpg",
            "publishedAt": "2026-09-23T10:00:00Z",
            "content": "A sudden labor strike has closed all major terminals in the Port of Mumbai, halting cargo.",
        },
        {
            "source": {"id": "bloomberg", "name": "Bloomberg"},
            "author": "Supply Reporter",
            "title": "Factory shutdown in Shenzhen impacts electronics supplier",
            "description": "Manufacturing plant announces emergency shutdown following fire incident.",
            "url": "https://example.com/shenzhen-factory-fire",
            "urlToImage": "https://example.com/img2.jpg",
            "publishedAt": "2026-09-23T11:00:00Z",
            "content": "Electronics assembly plant shuts down after industrial fire, disrupting supplier deliveries.",
        },
    ],
}


def test_missing_api_key_handling():
    """Verify client does not crash when API key is unconfigured."""
    c = NewsApiClient(api_key="")
    assert c.is_configured() is False
    articles, meta = c.fetch_articles("test query")
    assert articles == []
    assert meta["status"] == "error"
    assert meta["code"] == "apiKeyMissing"


@patch("requests.get")
def test_rate_limit_handling(mock_get):
    """Verify HTTP 429 rate limit is caught gracefully without application crash."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_get.return_value = mock_resp

    c = NewsApiClient(api_key="TEST_KEY")
    articles, meta = c.fetch_articles("supply chain")
    assert articles == []
    assert meta["status"] == "error"
    assert meta["code"] == "rateLimited"


@patch("requests.get")
def test_successful_fetch_and_normalization(mock_get):
    """Verify raw NewsAPI articles are correctly parsed and normalized."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_RAW_NEWSAPI_PAYLOAD
    mock_get.return_value = mock_resp

    c = NewsApiClient(api_key="TEST_KEY")
    articles, meta = c.fetch_articles("ports logistics", hours=24, limit=5)

    assert meta["status"] == "ok"
    assert len(articles) == 2
    a1 = articles[0]
    assert a1.source_name == "Reuters"
    assert a1.author == "Maritime Staff"
    assert "Port of Mumbai" in a1.title
    assert a1.url == "https://example.com/mumbai-port-strike"
    assert a1.data_status == "REAL_DATA"
    assert a1.article_id.startswith("NEWSAPI-")


def test_exact_url_deduplication():
    """Verify duplicate URLs are removed."""
    a1 = NormalizedArticle(
        article_id="A1", source_name="S1", title="Title 1",
        url="https://example.com/repeat-url", published_at="2026-09-23T00:00:00Z"
    )
    a2 = NormalizedArticle(
        article_id="A2", source_name="S2", title="Title 2 Different",
        url="https://example.com/repeat-url", published_at="2026-09-23T01:00:00Z"
    )
    unique, dups = deduplicate_articles([a1, a2])
    assert len(unique) == 1
    assert dups == 1


def test_near_duplicate_title_detection():
    """Verify headlines with high similarity are identified as near-duplicates."""
    a1 = NormalizedArticle(
        article_id="A1", source_name="News1",
        title="Dockworkers launch strike at major container port",
        url="https://example.com/dock-1", published_at="2026-09-23T00:00:00Z"
    )
    a2 = NormalizedArticle(
        article_id="A2", source_name="News2",
        title="Dockworkers launch strike at major container port!",
        url="https://example.com/dock-2", published_at="2026-09-23T00:05:00Z"
    )
    unique, dups = deduplicate_articles([a1, a2], title_threshold=0.85)
    assert len(unique) == 1
    assert dups == 1


def test_relevance_classification():
    """Verify supply chain disruption detection rules."""
    rel_text = "Flooding in Shanghai forces terminal closure and delays container ship departures."
    is_rel, score, keywords = check_relevance(rel_text)
    assert is_rel is True
    assert score >= 0.40

    irrel_text = "Celebrity pop star releases new acoustic album to rave reviews from music critics."
    is_irrel, irrel_score, _ = check_relevance(irrel_text)
    assert is_irrel is False
    assert irrel_score < 0.40


def test_raw_storage_and_processed_storage(tmp_path):
    """Verify raw JSON and normalized CSV persistence."""
    raw_path = save_raw_response(
        {"status": "ok", "articles": []}, query="test_storage", output_dir=str(tmp_path)
    )
    assert os.path.exists(raw_path)
    assert raw_path.endswith(".json")

    test_csv = str(tmp_path / "test_news.csv")
    art = NormalizedArticle(
        article_id="STORE-TEST-1", source_name="TestWire",
        title="Port congestion causes delays", url="https://example.com/store-test",
        published_at="2026-09-23T12:00:00Z"
    )
    stored = save_normalized_articles([art], csv_path=test_csv)
    assert stored == 1
    assert os.path.exists(test_csv)

    # Re-saving same article should avoid duplicates
    stored_again = save_normalized_articles([art], csv_path=test_csv)
    assert stored_again == 0


@patch("src.news.save_normalized_articles", return_value=1)
@patch("src.news.save_raw_response", return_value="data/raw/newsapi/mock.json")
@patch("src.news.client.NewsApiClient.fetch_articles")
def test_api_ingest_live_endpoint(mock_fetch, mock_raw_save, mock_norm_save):
    """Verify POST /ingest/live returns exact required response structure."""
    mock_articles = [
        NormalizedArticle(
            article_id="API-TEST-1",
            source_name="API Wire",
            title="Rail strike causes container freight bottlenecks",
            url="https://example.com/rail-strike-test",
            published_at="2026-09-23T12:00:00Z",
            description="Freight trains halted due to union strike.",
            content="Freight rail shutdown disrupts container traffic.",
            data_status="REAL_DATA",
        ),
        NormalizedArticle(
            article_id="API-TEST-2",
            source_name="Movie Wire",
            title="Hollywood actor wins international award at film festival",
            url="https://example.com/movie-test",
            published_at="2026-09-23T12:00:00Z",
            description="Actor praised for role in romantic drama.",
            data_status="REAL_DATA",
        ),
    ]
    mock_fetch.return_value = (mock_articles, {"status": "ok"})

    payload = {
        "topic": "India ports suppliers logistics manufacturing disruptions",
        "hours": 24,
        "limit": 10,
    }
    response = client.post("/ingest/live", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Check required fields
    assert "fetched" in data
    assert "accepted" in data
    assert "duplicates_removed" in data
    assert "irrelevant_removed" in data
    assert "stored" in data
    assert data["fetched"] == 2
    assert data["irrelevant_removed"] == 1  # Movie article filtered out
    assert data["accepted"] == 1           # Rail strike article accepted
    assert data["stored"] == 1
    assert mock_norm_save.called

