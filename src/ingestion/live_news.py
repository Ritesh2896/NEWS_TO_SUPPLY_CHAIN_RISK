"""Live NewsAPI ingestion for BDS-35.
NewsAPI is the only external news provider in this build; no OpenAI/Gemini keys are used.
"""
from __future__ import annotations
import os, re
from datetime import datetime, timedelta, timezone
from typing import Any
import requests
from dotenv import load_dotenv
load_dotenv()

BASE_URL = "https://newsapi.org/v2/everything"

def _normalize(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out=[]; seen=set(); now=datetime.now(timezone.utc).isoformat()
    for i,a in enumerate(items,1):
        title=str(a.get("title") or "").strip()
        url=str(a.get("url") or "").strip()
        if not title or not url or url in seen: continue
        seen.add(url)
        source=a.get("source") or {}
        out.append({
            "news_id": f"NEWSAPI-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{i}",
            "title": title,
            "content": str(a.get("content") or a.get("description") or "").strip(),
            "description": str(a.get("description") or "").strip(),
            "source": str(source.get("name") if isinstance(source,dict) else source),
            "published_at": str(a.get("publishedAt") or ""),
            "url": url,
            "image_url": str(a.get("urlToImage") or ""),
            "language": "en",
            "provider": "newsapi",
            "retrieved_at": now,
        })
    return out

def search_newsapi(topic: str, hours: int=48, limit: int=20) -> list[dict[str,Any]]:
    key=os.getenv("NEWSAPI_API_KEY")
    if not key: raise RuntimeError("NEWSAPI_API_KEY is not configured in .env")
    now=datetime.now(timezone.utc); start=now-timedelta(hours=hours)
    params={"q":topic,"from":start.isoformat(timespec="seconds"),"to":now.isoformat(timespec="seconds"),"language":"en","sortBy":"publishedAt","pageSize":min(limit,100),"apiKey":key}
    r=requests.get(BASE_URL,params=params,timeout=30)
    r.raise_for_status(); data=r.json()
    if data.get("status") != "ok": raise RuntimeError(data.get("message", "NewsAPI request failed"))
    return _normalize(data.get("articles",[]))[:limit]

def search_live_news(topic: str, provider: str|None=None, hours: int=48, limit: int=20) -> list[dict[str,Any]]:
    # provider is retained for API compatibility but intentionally ignored: NewsAPI only.
    return search_newsapi(topic,hours,limit)
