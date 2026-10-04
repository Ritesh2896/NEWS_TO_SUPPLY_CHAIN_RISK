"""Pydantic Schemas for BDS-35 FastAPI Backend (Phase 11).

Defines strongly-typed request and response contracts with input validation,
academic disclaimers, and metadata schemas for:
  - System health and runtime status
  - Live and benchmark pipeline execution
  - Paginated entity queries (News, Events, Alerts, Suppliers, Products, Locations)
  - Graph topology statistics
  - Explainable entity risk assessment
  - Analyst feedback and triage
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field

from src.alerts.schemas import PROJECT_THRESHOLDS_DISCLAIMER, RiskBand

T = TypeVar("T")


# ---------------------------------------------------------
# Health & Status Schemas
# ---------------------------------------------------------
class HealthResponse(BaseModel):
    """System health and diagnostic status."""
    status: str = Field(default="ok", description="Overall service status ('ok', 'degraded')")
    newsapi_configured: bool = Field(description="True if NEWSAPI_API_KEY environment variable is configured")
    graph_loaded: bool = Field(description="True if master supply-chain graph is loaded or master data present")
    trained_models: dict[str, bool] = Field(description="Availability of trained GNN model weights")
    database_status: dict[str, bool] = Field(description="Availability of core master and processed tables")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = Field(default="2.2.0")


# ---------------------------------------------------------
# Generic Pagination Wrapper
# ---------------------------------------------------------
class PaginatedResponse(BaseModel, Generic[T]):
    """Standardized paginated list response."""
    total: int = Field(ge=0, description="Total number of records matching query")
    offset: int = Field(ge=0, description="Number of items skipped")
    limit: int = Field(ge=1, description="Maximum number of items returned")
    items: List[T] = Field(description="List of records for current page")


# ---------------------------------------------------------
# News, Event, and Alert Schemas
# ---------------------------------------------------------
class NewsArticleItem(BaseModel):
    """Normalized news article item."""
    article_id: str
    source_name: Optional[str] = ""
    author: Optional[str] = ""
    title: str
    description: Optional[str] = ""
    url: Optional[str] = ""
    published_at: Optional[str] = ""
    content: Optional[str] = ""
    query: Optional[str] = ""
    retrieved_at: Optional[str] = ""
    data_status: Optional[str] = "REAL_DATA"


class EventItem(BaseModel):
    """Structured disruption event item."""
    event_id: str
    article_id: Optional[str] = ""
    news_id: Optional[str] = ""
    event_type: str
    event_category: Optional[str] = ""
    trigger: Optional[str] = ""
    impact: Optional[str] = ""
    severity: Any
    location_id: Optional[str] = ""
    location_name: Optional[str] = ""
    supplier_id: Optional[str] = ""
    supplier_name: Optional[str] = ""
    event_time: Optional[str] = ""
    extraction_confidence: Optional[float] = None
    evidence_text: Optional[str] = ""
    source_url: Optional[str] = ""
    data_status: Optional[str] = "SYNTHETIC_DEMO"


class AlertItem(BaseModel):
    """Explainable early-warning risk alert."""
    alert_id: str
    supplier_id: str
    supplier_name: Optional[str] = ""
    event_id: Optional[str] = ""
    event_type: Optional[str] = ""
    deterministic_risk: float
    graphsage_risk: Optional[float] = 0.0
    gat_risk: Optional[float] = 0.0
    combined_risk: float
    risk_score_100: float
    risk_band: str
    reasons: Optional[str] = ""
    explanation: Optional[str] = ""
    created_at: Optional[str] = ""
    data_status: Optional[str] = "SYNTHETIC_DEMO"
    threshold_disclaimer: str = PROJECT_THRESHOLDS_DISCLAIMER


# ---------------------------------------------------------
# Master Entities Schemas
# ---------------------------------------------------------
class SupplierItem(BaseModel):
    """Master supplier entity record."""
    supplier_id: str
    supplier_name: str
    country: Optional[str] = ""
    state: Optional[str] = ""
    city: Optional[str] = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    supplier_type: Optional[str] = ""
    industry: Optional[str] = ""
    tier: Optional[str] = ""
    criticality: Optional[str] = ""
    single_source: Optional[Any] = False
    status: Optional[str] = "ACTIVE"
    data_status: Optional[str] = "SYNTHETIC_DEMO"


class ProductItem(BaseModel):
    """Master product entity record."""
    product_id: str
    product_name: str
    category: Optional[str] = ""
    sub_category: Optional[str] = ""
    criticality: Optional[str] = ""
    lead_time_days: Optional[Any] = None
    substitutability: Optional[str] = ""
    data_status: Optional[str] = "SYNTHETIC_DEMO"


class LocationItem(BaseModel):
    """Master location entity record."""
    location_id: str
    location_name: str
    location_type: Optional[str] = ""
    country: Optional[str] = ""
    state: Optional[str] = ""
    city: Optional[str] = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    region: Optional[str] = ""
    data_status: Optional[str] = "SYNTHETIC_DEMO"


# ---------------------------------------------------------
# Graph Topology Statistics Schema
# ---------------------------------------------------------
class GraphStatsResponse(BaseModel):
    """Topological metrics and heterogeneous degree/component stats."""
    num_nodes: int = Field(ge=0, description="Total node count")
    num_edges: int = Field(ge=0, description="Total edge count")
    density: float = Field(ge=0.0, description="Graph network density")
    node_types: dict[str, int] = Field(description="Node count broken down by node type")
    edge_types: Optional[dict[str, int]] = Field(default_factory=dict, description="Edge count by edge type")
    connected_components: Optional[dict[str, Any]] = Field(default_factory=dict, description="Weakly & strongly connected component counts")
    isolated_nodes: Optional[dict[str, Any]] = Field(default_factory=dict, description="Isolated node counts and sample IDs")


# ---------------------------------------------------------
# Single Entity Explainable Risk Schema
# ---------------------------------------------------------
class EntityRiskResponse(BaseModel):
    """Detailed multi-factor risk assessment and explanation for a single entity."""
    entity_id: str
    entity_name: str
    entity_type: str = "SUPPLIER"
    deterministic_risk: float = Field(ge=0.0, le=1.0)
    graphsage_risk: float = Field(ge=0.0, le=1.0)
    gat_risk: float = Field(ge=0.0, le=1.0)
    combined_risk: float = Field(ge=0.0, le=1.0)
    risk_score_100: float = Field(ge=0.0, le=100.0)
    risk_band: str = Field(description="LOW, MEDIUM, HIGH, or CRITICAL")
    reasons: list[str] = Field(description="Bullet-point contributing risk drivers")
    explanation: str = Field(description="Canonical human-readable explanation")
    contributing_factors: dict[str, Any] = Field(description="Normalized individual risk factors")
    active_alerts_count: int = Field(default=0, ge=0)
    threshold_disclaimer: str = PROJECT_THRESHOLDS_DISCLAIMER
    data_status: str = "SYNTHETIC_DEMO"


# ---------------------------------------------------------
# Analyst Feedback & Triage Schemas
# ---------------------------------------------------------
class FeedbackRequest(BaseModel):
    """Analyst review feedback payload."""
    alert_id: str = Field(min_length=1, max_length=100, description="Unique ID of alert being triaged")
    decision: str = Field(min_length=1, max_length=50, description="Decision: ACCEPT, REJECT, ESCALATE, FALSE_POSITIVE")
    analyst_id: Optional[str] = Field(default="analyst_default", max_length=100, description="Analyst identifier")
    comment: Optional[str] = Field(default="", max_length=1000, description="Triage commentary or notes")
    recommended_action: Optional[str] = Field(default="", max_length=500, description="Action recommended")


class FeedbackResponse(BaseModel):
    """Feedback submission confirmation."""
    status: str = "saved"
    feedback_id: str
    alert_id: str
    decision: str
    analyst_id: str
    comment: str
    recommended_action: str
    timestamp: str


# ---------------------------------------------------------
# Live Pipeline Run Schemas
# ---------------------------------------------------------
class LivePipelineRequest(BaseModel):
    """Live pipeline execution request payload."""
    topic: str = Field(
        default="India ports suppliers logistics manufacturing disruptions",
        min_length=3,
        max_length=500,
        description="Search topic query for supply chain news",
    )
    hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Time horizon in hours to query (1-168)",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum articles to fetch and process (1-50)",
    )
    language: str = Field(default="en", max_length=5, description="Article language code")
    country: Optional[str] = Field(default=None, max_length=5, description="Country filter code if supported")
    model_type: str = Field(default="both", description="GNN model for inference: 'graphsage', 'gat', or 'both'")
    api_key: Optional[str] = Field(default=None, max_length=100, description="Optional NewsAPI key (if None, reads from server environment)")
    timeout_seconds: int = Field(default=60, ge=5, le=300, description="Timeout in seconds to prevent blocking indefinitely")


class PipelineStageMetrics(BaseModel):
    """Stage-by-stage counts and metrics."""
    news_ingestion: dict[str, Any]
    nlp_processing: dict[str, Any]
    entity_linking: dict[str, Any]
    graph_update: dict[str, Any]
    geo_exposure: dict[str, Any]
    deterministic_risk: dict[str, Any]
    gnn_inference: dict[str, Any]
    alerts_generated: dict[str, Any]


class PipelineExecutionSummary(BaseModel):
    """Comprehensive pipeline execution summary."""
    status: str = Field(description="'COMPLETED', 'PARTIAL', or 'FALLBACK_DEMO'")
    execution_time_seconds: float = Field(ge=0.0)
    topic: str
    stages: PipelineStageMetrics
    alerts_count: int = Field(ge=0)
    alerts: List[dict[str, Any]] = Field(default_factory=list, description="Top generated alert records")
    data_status: str = Field(default="REAL_DATA", description="REAL_DATA or SYNTHETIC_DEMO")
    threshold_disclaimer: str = PROJECT_THRESHOLDS_DISCLAIMER
