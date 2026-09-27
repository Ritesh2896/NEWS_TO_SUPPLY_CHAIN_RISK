import pandas as pd
from src.ingestion.live_news import _normalize
from src.gnn.gnn_model import topology_risk

def test_news_normalize():
    x=_normalize([{"title":"Flood disrupts port","url":"https://example.com/a","source":{"name":"Example"},"publishedAt":"2026-09-22T00:00:00Z"},{"title":"dup","url":"https://example.com/a"}])
    assert len(x)==1 and x[0]["provider"]=="newsapi"

def test_topology_fallback():
    scores=topology_risk([[1,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0]],[(0,1)])
    assert len(scores)==2 and 0 <= scores[0] <= 1
