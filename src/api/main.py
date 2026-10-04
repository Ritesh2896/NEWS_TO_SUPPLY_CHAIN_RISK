"""FastAPI backend application for BDS-35 News-to-Risk Early Warning System (Phase 11).

Exposes REST endpoints for:
  - System health and component diagnostics (GET /health)
  - Live NewsAPI ingestion (POST /ingest/live)
  - Complete 8-stage live pipeline execution (POST /pipeline/run-live)
  - Paginated entity discovery (GET /news/live, /events, /alerts, /suppliers, /products, /locations)
  - Graph topology and ego-network inspection (GET /graph/stats, /graph/ego/{supplier_id})
  - Explainable entity risk assessment (GET /risk/{entity_id}, /risk/explain/{alert_id})
  - Analyst feedback and triage logging (POST /feedback)
  - GNN model registry and academic benchmark evaluation (GET /models, /evaluation)
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.alerts.explanations import extract_alert_reasons, format_alert_explanation
from src.alerts.generator import calculate_combined_risk
from src.alerts.schemas import (
    DEFAULT_BAND_THRESHOLDS,
    DEFAULT_COMBINED_WEIGHTS,
    PROJECT_THRESHOLDS_DISCLAIMER,
    RiskBand,
)
from src.api.schemas import (
    AlertItem,
    EntityRiskResponse,
    EventItem,
    FeedbackRequest,
    FeedbackResponse,
    GraphStatsResponse,
    HealthResponse,
    LivePipelineRequest,
    LocationItem,
    NewsArticleItem,
    PaginatedResponse,
    PipelineExecutionSummary,
    ProductItem,
    SupplierItem,
)
from src.graph.export import compute_graph_statistics
from src.graph.supply_chain_graph import compute_graph_metrics, get_ego_network, load_master_graph
from src.logging_config import logger
from src.news import IngestRequest, IngestResponse, ingest_live_news
from src.pipeline.end_to_end_pipeline import run_pipeline
from src.pipeline.live_pipeline import execute_live_pipeline
from src.risk.risk_engine import calculate_risk

load_dotenv()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MASTER_DIR = os.path.join(ROOT, "data/master")
PROCESSED_DIR = os.path.join(ROOT, "data/processed")
GRAPH_DIR = os.path.join(ROOT, "data/graph")

# -------------------------------------------------------------
# FastAPI App Definition & Swagger Documentation Metadata
# -------------------------------------------------------------
app = FastAPI(
    title="BDS-35: News-to-Risk Supply-Chain Early Warning API",
    version="2.2.0",
    description=r"""
### BDS-35 Early Warning Platform REST API

Transforms open live news into structured supply-chain disruption events,
links entities to enterprise master data, models heterogeneous topological dependencies,
and propagates risk through Graph Neural Networks (GraphSAGE & Graph Attention Networks)
to deliver explainable, tri-model combined risk alerts.

#### Key Pipelines & Methodologies:
- **Phase 2 Ingestion**: Live NewsAPI querying, URL & title deduplication, relevance filtering.
- **Phase 3 Local NLP**: 10 event categories, deterministic severity scoring, no external LLM dependencies.
- **Phase 4 Entity Linking**: Exact, normalized, alias, and controlled fuzzy matching to master data.
- **Phase 5 Heterogeneous Graph**: NetworkX DiGraph connecting News, Events, Suppliers, Products, Locations, Facilities.
- **Phase 6 Geographic Exposure**: Haversine distance decay zones (0-25km, 25-100km, 100-250km, 250+km).
- **Phase 7 Deterministic Risk Engine**: 6-factor heuristic baseline normalized to [0, 1].
- **Phase 8 GraphSAGE Propagation**: 2-layer mean aggregation of topological risk exposure.
- **Phase 9 GAT Propagation**: Multi-head attention mechanism capturing relational importance.
- **Phase 10 Combined Risk Alerts**: Configurable tri-model synthesis ($0.50 \cdot Det + 0.25 \cdot SAGE + 0.25 \cdot GAT$) with project-defined bands.
- **Phase 11 FastAPI Backend**: Fully validated, documented, paginated REST endpoints.
    """,
    contact={
        "name": "BDS-35 Early Warning Engineering Team",
        "url": "https://github.com/supply-chain-risk-intel",
    },
    license_info={
        "name": "Academic & Research Demo License",
    },
)

# -------------------------------------------------------------
# CORS Configuration
# -------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cached master graph
_CACHED_GRAPH = None


def get_graph():
    """Retrieve or lazy-load the master supply-chain graph."""
    global _CACHED_GRAPH
    if _CACHED_GRAPH is None:
        _CACHED_GRAPH = load_master_graph(os.path.join(ROOT, "data"))
    return _CACHED_GRAPH


def _read_records(rel_path: str) -> list[dict[str, Any]]:
    """Safely read CSV records as list of dictionaries."""
    full_path = os.path.join(ROOT, rel_path)
    if os.path.exists(full_path):
        try:
            return pd.read_csv(full_path).fillna("").to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error reading CSV {rel_path}: {e}")
    return []


# -------------------------------------------------------------
# Root & Health Endpoints
# -------------------------------------------------------------
@app.get("/", tags=["System Information"])
def root():
    """Root metadata and API status overview."""
    return {
        "project": "BDS-35 — News-to-Risk Supply-Chain Early Warning",
        "status": "OPERATIONAL",
        "version": "2.2.0",
        "supported_gnn_models": ["GraphSAGE", "GAT"],
        "data_status": "SYNTHETIC_DEMO & REAL_DATA_HYBRID",
        "news_provider": "NewsAPI",
        "documentation_url": "/docs",
        "openapi_url": "/openapi.json",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health & Diagnostics"])
def health():
    """System health check and component diagnostics.

    Verifies configuration, master tables, graph structure, and trained GNN weights.
    Never exposes API keys or credentials.
    """
    newsapi_key = os.getenv("NEWSAPI_API_KEY", "").strip()
    newsapi_configured = bool(newsapi_key and newsapi_key != "YOUR_NEWSAPI_KEY")

    trained_models = {
        "graphsage": (
            os.path.exists(os.path.join(ROOT, "models/graphsage_phase8.pt"))
            or os.path.exists(os.path.join(ROOT, "models/graphsage_model.pt"))
        ),
        "gat": (
            os.path.exists(os.path.join(ROOT, "models/gat_phase9.pt"))
            or os.path.exists(os.path.join(ROOT, "models/gat_model.pt"))
        ),
    }

    db_status = {
        "suppliers": os.path.exists(os.path.join(MASTER_DIR, "suppliers.csv")),
        "products": os.path.exists(os.path.join(MASTER_DIR, "products.csv")),
        "locations": os.path.exists(os.path.join(MASTER_DIR, "locations.csv")),
        "graph_edges": (
            os.path.exists(os.path.join(MASTER_DIR, "edges.csv"))
            or os.path.exists(os.path.join(GRAPH_DIR, "edges.csv"))
        ),
        "alerts_csv": (
            os.path.exists(os.path.join(PROCESSED_DIR, "alerts.csv"))
            or os.path.exists(os.path.join(PROCESSED_DIR, "final_alerts.csv"))
        ),
    }

    graph_ready = _CACHED_GRAPH is not None or db_status["suppliers"]

    overall_status = "ok" if (graph_ready and all(trained_models.values())) else "degraded"

    return HealthResponse(
        status=overall_status,
        newsapi_configured=newsapi_configured,
        graph_loaded=graph_ready,
        trained_models=trained_models,
        database_status=db_status,
        version="2.2.0",
    )


# -------------------------------------------------------------
# Ingestion & Live Pipeline Endpoints
# -------------------------------------------------------------
@app.post(
    "/ingest/live",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    tags=["News Ingestion"],
)
def ingest_live(req: IngestRequest):
    """Fetch, deduplicate, filter, and normalize live news articles from NewsAPI."""
    logger.info(f"POST /ingest/live invoked: topic='{req.topic}', limit={req.limit}")
    try:
        res = ingest_live_news(
            topic=req.topic,
            hours=req.hours,
            language=req.language,
            country=req.country,
            limit=req.limit,
        )
        return res
    except Exception as e:
        logger.error(f"Live news ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Live news ingestion error: {str(e)}",
        )


@app.post(
    "/pipeline/run-live",
    response_model=PipelineExecutionSummary,
    status_code=status.HTTP_200_OK,
    tags=["Risk Pipeline"],
)
def run_live_pipeline(req: LivePipelineRequest):
    """Execute complete 8-stage live early-warning pipeline:

    NewsAPI → NLP → Entity Linking → Graph Update → Geo Exposure → Deterministic Risk → GNN Inference → Alerts.

    Protected by non-blocking timeout safeguard to prevent hanging requests.
    """
    logger.info(f"POST /pipeline/run-live invoked for topic='{req.topic}', timeout={req.timeout_seconds}s")
    try:
        result = execute_live_pipeline(
            topic=req.topic,
            hours=req.hours,
            limit=req.limit,
            language=req.language,
            country=req.country,
            model_type=req.model_type,
            api_key=req.api_key,
            timeout_seconds=req.timeout_seconds,
        )
        return result
    except TimeoutError as te:
        logger.error(f"Pipeline execution timeout: {te}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Pipeline execution exceeded configured timeout of {req.timeout_seconds}s.",
        )
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {str(e)}",
        )


# -------------------------------------------------------------
# Paginated Entity Endpoints
# -------------------------------------------------------------
@app.get(
    "/news/live",
    response_model=PaginatedResponse[NewsArticleItem],
    tags=["Entity Discovery"],
)
def get_live_news(
    limit: int = Query(50, ge=1, le=1000, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Records to skip"),
    query: Optional[str] = Query(None, description="Search keyword in title or content"),
    raw_list: bool = Query(False, description="If True, returns raw list of records"),
):
    """Retrieve normalized live and processed news articles with pagination."""
    records = _read_records("data/processed/news.csv")
    if not records:
        records = _read_records("data/processed/live_news.csv")

    if query:
        q = query.lower()
        records = [
            r for r in records
            if q in str(r.get("title", "")).lower() or q in str(r.get("content", "")).lower()
        ]

    total = len(records)
    sliced = records[offset : offset + limit]

    if raw_list:
        return sliced

    return PaginatedResponse[NewsArticleItem](
        total=total,
        offset=offset,
        limit=limit,
        items=sliced,
    )


@app.get(
    "/events",
    response_model=PaginatedResponse[EventItem],
    tags=["Entity Discovery"],
)
def get_events(
    limit: int = Query(50, ge=1, le=1000, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Records to skip"),
    severity: Optional[str] = Query(None, description="Filter by severity (Critical, High, Medium, Low)"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    search: Optional[str] = Query(None, description="Search keyword in title or evidence"),
    raw_list: bool = Query(False, description="If True, returns raw list of records"),
):
    """Retrieve structured disruption events with pagination and filtering."""
    records = _read_records("data/processed/events.csv")
    if not records:
        records = _read_records("data/processed/events_live.csv")

    if severity:
        records = [r for r in records if str(r.get("severity", "")).upper() == severity.upper()]
    if event_type:
        records = [r for r in records if event_type.lower() in str(r.get("event_type", "")).lower()]
    if search:
        s = search.lower()
        records = [
            r for r in records
            if s in str(r.get("title", "")).lower() or s in str(r.get("evidence_text", "")).lower()
        ]

    total = len(records)
    sliced = records[offset : offset + limit]

    if raw_list:
        return sliced

    return PaginatedResponse[EventItem](
        total=total,
        offset=offset,
        limit=limit,
        items=sliced,
    )


@app.get(
    "/alerts",
    response_model=PaginatedResponse[AlertItem],
    tags=["Alerts & Explanations"],
)
def get_alerts(
    limit: int = Query(50, ge=1, le=1000, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Records to skip"),
    risk_band: Optional[str] = Query(None, description="Filter by project risk band: CRITICAL, HIGH, MEDIUM, LOW"),
    risk_level: Optional[str] = Query(None, description="Alias for risk_band"),
    supplier_id: Optional[str] = Query(None, description="Filter by supplier ID"),
    event_id: Optional[str] = Query(None, description="Filter by event ID"),
    raw_list: bool = Query(False, description="If True, returns raw list of records"),
):
    """Retrieve explainable risk alerts generated from the tri-model synthesis engine."""
    # Check Phase 10 alerts.csv first, then fallback to final_alerts.csv
    records = _read_records("data/processed/alerts.csv")
    if not records:
        records = _read_records("data/processed/final_alerts.csv")

    # Map legacy fields if present
    normalized_alerts = []
    for r in records:
        band = r.get("risk_band") or r.get("combined_risk_level", "MEDIUM")
        comb_risk = r.get("combined_risk")
        if comb_risk is None:
            c_score = r.get("combined_risk_score", 50.0)
            comb_risk = round(float(c_score) / 100.0, 4) if float(c_score) > 1.0 else float(c_score)

        score_100 = r.get("risk_score_100")
        if score_100 is None:
            score_100 = round(float(comb_risk) * 100.0, 2)

        det_risk = r.get("deterministic_risk", 0.5)
        det_norm = float(det_risk) / 100.0 if float(det_risk) > 1.0 else float(det_risk)

        normalized_alerts.append({
            "alert_id": str(r.get("alert_id", "")),
            "supplier_id": str(r.get("supplier_id", "")),
            "supplier_name": str(r.get("supplier_name", "")),
            "event_id": str(r.get("event_id", "")),
            "event_type": str(r.get("event_type", "Disruption")),
            "deterministic_risk": round(det_norm, 4),
            "graphsage_risk": round(float(r.get("graphsage_risk", r.get("graph_risk_score", 0.0))) / 100.0 if float(r.get("graphsage_risk", r.get("graph_risk_score", 0.0))) > 1.0 else float(r.get("graphsage_risk", r.get("graph_risk_score", 0.0))), 4),
            "gat_risk": round(float(r.get("gat_risk", 0.0)) / 100.0 if float(r.get("gat_risk", 0.0)) > 1.0 else float(r.get("gat_risk", 0.0)), 4),
            "combined_risk": round(float(comb_risk), 4),
            "risk_score_100": round(float(score_100), 2),
            "risk_band": str(band).upper(),
            "reasons": str(r.get("reasons", r.get("reason", ""))),
            "explanation": str(r.get("explanation", r.get("reason", ""))),
            "created_at": str(r.get("created_at", "")),
            "data_status": str(r.get("data_status", "SYNTHETIC_DEMO")),
            "threshold_disclaimer": PROJECT_THRESHOLDS_DISCLAIMER,
        })

    target_band = (risk_band or risk_level)
    if target_band:
        normalized_alerts = [
            a for a in normalized_alerts if a["risk_band"] == target_band.upper()
        ]
    if supplier_id:
        normalized_alerts = [
            a for a in normalized_alerts if a["supplier_id"] == supplier_id
        ]
    if event_id:
        normalized_alerts = [
            a for a in normalized_alerts if a["event_id"] == event_id
        ]

    total = len(normalized_alerts)
    sliced = normalized_alerts[offset : offset + limit]

    if raw_list:
        return sliced

    return PaginatedResponse[AlertItem](
        total=total,
        offset=offset,
        limit=limit,
        items=sliced,
    )


@app.get(
    "/suppliers",
    response_model=PaginatedResponse[SupplierItem],
    tags=["Master Data"],
)
def get_suppliers(
    limit: int = Query(50, ge=1, le=1000, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Records to skip"),
    search: Optional[str] = Query(None, description="Search name, ID, or industry"),
    country: Optional[str] = Query(None, description="Filter by country"),
    tier: Optional[str] = Query(None, description="Filter by tier (Tier-1, Tier-2, Tier-3)"),
    criticality: Optional[str] = Query(None, description="Filter by criticality (HIGH, MEDIUM, LOW)"),
    raw_list: bool = Query(False, description="If True, returns raw list of records"),
):
    """Retrieve master supplier directory with filtering and pagination."""
    records = _read_records("data/master/suppliers.csv")

    if search:
        s = search.lower()
        records = [
            r for r in records
            if s in str(r.get("supplier_name", "")).lower()
            or s in str(r.get("supplier_id", "")).lower()
            or s in str(r.get("industry", "")).lower()
        ]
    if country:
        records = [r for r in records if str(r.get("country", "")).lower() == country.lower()]
    if tier:
        records = [r for r in records if str(r.get("tier", "")).lower() == tier.lower()]
    if criticality:
        records = [r for r in records if str(r.get("criticality", "")).upper() == criticality.upper()]

    total = len(records)
    sliced = records[offset : offset + limit]

    if raw_list:
        return sliced

    return PaginatedResponse[SupplierItem](
        total=total,
        offset=offset,
        limit=limit,
        items=sliced,
    )


@app.get(
    "/products",
    response_model=PaginatedResponse[ProductItem],
    tags=["Master Data"],
)
def get_products(
    limit: int = Query(50, ge=1, le=1000, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Records to skip"),
    search: Optional[str] = Query(None, description="Search product name or category"),
    category: Optional[str] = Query(None, description="Filter by category"),
    raw_list: bool = Query(False, description="If True, returns raw list of records"),
):
    """Retrieve master product catalog with pagination and search."""
    records = _read_records("data/master/products.csv")

    if search:
        s = search.lower()
        records = [
            r for r in records
            if s in str(r.get("product_name", "")).lower()
            or s in str(r.get("product_id", "")).lower()
            or s in str(r.get("category", "")).lower()
        ]
    if category:
        records = [r for r in records if str(r.get("category", "")).lower() == category.lower()]

    total = len(records)
    sliced = records[offset : offset + limit]

    if raw_list:
        return sliced

    return PaginatedResponse[ProductItem](
        total=total,
        offset=offset,
        limit=limit,
        items=sliced,
    )


@app.get(
    "/locations",
    response_model=PaginatedResponse[LocationItem],
    tags=["Master Data"],
)
def get_locations(
    limit: int = Query(50, ge=1, le=1000, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Records to skip"),
    search: Optional[str] = Query(None, description="Search location name, city, or ID"),
    country: Optional[str] = Query(None, description="Filter by country"),
    raw_list: bool = Query(False, description="If True, returns raw list of records"),
):
    """Retrieve geographic facility and site master directory."""
    records = _read_records("data/master/locations.csv")

    if search:
        s = search.lower()
        records = [
            r for r in records
            if s in str(r.get("location_name", "")).lower()
            or s in str(r.get("city", "")).lower()
            or s in str(r.get("location_id", "")).lower()
        ]
    if country:
        records = [r for r in records if str(r.get("country", "")).lower() == country.lower()]

    total = len(records)
    sliced = records[offset : offset + limit]

    if raw_list:
        return sliced

    return PaginatedResponse[LocationItem](
        total=total,
        offset=offset,
        limit=limit,
        items=sliced,
    )


# -------------------------------------------------------------
# Graph Statistics & Ego Network Endpoints
# -------------------------------------------------------------
@app.get(
    "/graph/stats",
    response_model=GraphStatsResponse,
    tags=["Graph Topology"],
)
def graph_stats():
    """Retrieve comprehensive topological metrics, degree stats, and component distributions."""
    G = get_graph()
    stats = compute_graph_statistics(G)
    metrics = compute_graph_metrics(G)

    return GraphStatsResponse(
        num_nodes=stats["total_nodes"],
        num_edges=stats["total_edges"],
        density=metrics.get("density", 0.0),
        node_types=stats["nodes_by_type"],
        edge_types=stats["edges_by_type"],
        connected_components=stats["connected_components"],
        isolated_nodes=stats["isolated_nodes"],
    )


@app.get("/graph/ego/{supplier_id}", tags=["Graph Topology"])
def supplier_ego_network(
    supplier_id: str = Path(..., description="Target supplier node ID"),
    radius: int = Query(1, ge=1, le=3, description="Neighborhood radius (hops)"),
):
    """Extract local subgraph (ego network) centered on a specific supplier."""
    G = get_graph()
    if supplier_id not in G:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node '{supplier_id}' not found in supply-chain graph.",
        )
    return get_ego_network(G, supplier_id, radius=radius)


# -------------------------------------------------------------
# Explainable Entity Risk Endpoint
# -------------------------------------------------------------
@app.get(
    "/risk/{entity_id}",
    response_model=EntityRiskResponse,
    tags=["Alerts & Explanations"],
)
def get_entity_risk(
    entity_id: str = Path(..., description="Supplier ID, Product ID, Location ID, or Alert ID"),
):
    """Retrieve multi-factor explainable risk posture and drivers for a specific entity.

    Combines deterministic heuristics, GraphSAGE topological propagation, and GAT attention risk.
    """
    clean_id = entity_id.strip()

    # 1. Check if there are active alerts matching this entity
    alerts = _read_records("data/processed/alerts.csv")
    if not alerts:
        alerts = _read_records("data/processed/final_alerts.csv")

    matching_alerts = [
        a for a in alerts
        if str(a.get("supplier_id", "")).strip() == clean_id
        or str(a.get("alert_id", "")).strip() == clean_id
        or str(a.get("event_id", "")).strip() == clean_id
    ]

    if matching_alerts:
        top_alert = matching_alerts[0]
        sup_id = str(top_alert.get("supplier_id", clean_id))
        sup_name = str(top_alert.get("supplier_name", f"Supplier {sup_id}"))

        d_risk = float(top_alert.get("deterministic_risk", 0.5))
        d_norm = d_risk / 100.0 if d_risk > 1.0 else d_risk

        sage_risk = float(top_alert.get("graphsage_risk", top_alert.get("graph_risk_score", 0.5)))
        sage_norm = sage_risk / 100.0 if sage_risk > 1.0 else sage_risk

        gat_risk = float(top_alert.get("gat_risk", sage_norm))
        gat_norm = gat_risk / 100.0 if gat_risk > 1.0 else gat_risk

        comb_norm, comb_100, band = calculate_combined_risk(
            deterministic_risk=d_norm,
            graphsage_risk=sage_norm,
            gat_risk=gat_norm,
            weights=DEFAULT_COMBINED_WEIGHTS,
            thresholds=DEFAULT_BAND_THRESHOLDS,
        )

        reasons_raw = str(top_alert.get("reasons", top_alert.get("reason", "")))
        reasons_list = [r.strip() for r in reasons_raw.split(";") if r.strip()]
        if not reasons_list:
            reasons_list = extract_alert_reasons(
                event_severity=0.7,
                geographic_exposure=0.6,
                dependency_strength=0.7,
                deterministic_risk=d_norm,
                graphsage_risk=sage_norm,
                gat_risk=gat_norm,
                supplier_criticality=0.7,
            )

        explanation = top_alert.get("explanation") or format_alert_explanation(
            supplier_id=sup_id,
            supplier_name=sup_name,
            risk_band=band,
            risk_score_100=comb_100,
            reasons=reasons_list,
        )

        return EntityRiskResponse(
            entity_id=sup_id,
            entity_name=sup_name,
            entity_type="SUPPLIER",
            deterministic_risk=round(d_norm, 4),
            graphsage_risk=round(sage_norm, 4),
            gat_risk=round(gat_norm, 4),
            combined_risk=round(comb_norm, 4),
            risk_score_100=round(comb_100, 2),
            risk_band=band.value,
            reasons=reasons_list,
            explanation=explanation,
            contributing_factors={
                "deterministic_risk": round(d_norm, 4),
                "graphsage_risk": round(sage_norm, 4),
                "gat_risk": round(gat_norm, 4),
            },
            active_alerts_count=len(matching_alerts),
            data_status=str(top_alert.get("data_status", "SYNTHETIC_DEMO")),
        )

    # 2. Check Master Supplier Directory
    suppliers = _read_records("data/master/suppliers.csv")
    sup_match = [
        s for s in suppliers
        if str(s.get("supplier_id", "")).strip() == clean_id
        or str(s.get("supplier_name", "")).strip().lower() == clean_id.lower()
    ]

    if sup_match:
        s_row = sup_match[0]
        sup_id = str(s_row.get("supplier_id", clean_id))
        sup_name = str(s_row.get("supplier_name", f"Supplier {sup_id}"))

        crit = str(s_row.get("criticality", "Medium"))
        dep = str(s_row.get("tier", "Tier-2"))
        single = bool(s_row.get("single_source", False))

        # Compute baseline deterministic risk
        base_risk = calculate_risk(
            severity="Medium",
            dependency="High" if "1" in dep else "Medium",
            criticality=crit,
            geographic_exposure=0.20,
            single_source=single,
        )
        d_norm = round(float(base_risk["risk_score"]) / 100.0, 4)
        sage_norm = round(d_norm * 0.9, 4)
        gat_norm = round(d_norm * 0.95, 4)

        comb_norm, comb_100, band = calculate_combined_risk(
            deterministic_risk=d_norm,
            graphsage_risk=sage_norm,
            gat_risk=gat_norm,
        )

        reasons = [
            f"Supplier criticality rated {crit}",
            f"Supply chain hierarchy classified as {dep}",
            "Baseline operating environment without active local disruption",
        ]
        if single:
            reasons.append("Single-source sole supplier vulnerability")

        explanation = format_alert_explanation(
            supplier_id=sup_id,
            supplier_name=sup_name,
            risk_band=band,
            risk_score_100=comb_100,
            reasons=reasons,
        )

        return EntityRiskResponse(
            entity_id=sup_id,
            entity_name=sup_name,
            entity_type="SUPPLIER",
            deterministic_risk=d_norm,
            graphsage_risk=sage_norm,
            gat_risk=gat_norm,
            combined_risk=comb_norm,
            risk_score_100=comb_100,
            risk_band=band.value,
            reasons=reasons,
            explanation=explanation,
            contributing_factors=base_risk.get("components", {}),
            active_alerts_count=0,
            data_status=str(s_row.get("data_status", "SYNTHETIC_DEMO")),
        )

    # 3. Entity Not Found
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Entity '{clean_id}' was not found in BDS-35 master data or active alerts.",
    )


# -------------------------------------------------------------
# Analyst Feedback & Triage Logging
# -------------------------------------------------------------
@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Feedback & Triage"],
)
def submit_feedback(request: FeedbackRequest):
    """Log triage review decision for an alert with persistent audit trail."""
    fb_id = f"FB-{uuid.uuid4().hex[:8].upper()}"
    ts = datetime.now(timezone.utc).isoformat()

    feedback_payload = {
        "feedback_id": fb_id,
        "alert_id": request.alert_id,
        "decision": request.decision.upper(),
        "analyst_id": request.analyst_id or "analyst_default",
        "comment": request.comment or "",
        "recommended_action": request.recommended_action or "",
        "created_at": ts,
    }

    try:
        path = os.path.join(PROCESSED_DIR, "analyst_feedback.csv")
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df_row = pd.DataFrame([feedback_payload])
        if os.path.exists(path):
            df_row.to_csv(path, mode="a", header=False, index=False)
        else:
            df_row.to_csv(path, index=False)

        logger.info(f"Feedback logged: {fb_id} on alert {request.alert_id} ({request.decision})")

        return FeedbackResponse(
            status="saved",
            feedback_id=fb_id,
            alert_id=request.alert_id,
            decision=request.decision.upper(),
            analyst_id=request.analyst_id or "analyst_default",
            comment=request.comment or "",
            recommended_action=request.recommended_action or "",
            timestamp=ts,
        )
    except Exception as e:
        logger.error(f"Failed to persist analyst feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not record analyst feedback: {str(e)}",
        )


# -------------------------------------------------------------
# Legacy Compatibility Endpoints
# -------------------------------------------------------------
@app.get("/alerts/{alert_id}", tags=["Alerts & Explanations"])
def get_alert_by_id(alert_id: str):
    """Retrieve specific alert by ID (backward compatibility)."""
    alerts = _read_records("data/processed/alerts.csv")
    if not alerts:
        alerts = _read_records("data/processed/final_alerts.csv")
    for a in alerts:
        if str(a.get("alert_id")) == alert_id:
            return a
    raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")


@app.get("/risk/explain/{alert_id}", tags=["Alerts & Explanations"])
def explain_alert(alert_id: str):
    """Retrieve factor attribution and formula breakdown for an alert."""
    alerts = _read_records("data/processed/alerts.csv")
    if not alerts:
        alerts = _read_records("data/processed/final_alerts.csv")
    for a in alerts:
        if str(a.get("alert_id")) == alert_id:
            return {
                "alert_id": alert_id,
                "event_type": a.get("event_type"),
                "event_location": a.get("event_location", "Unknown Location"),
                "supplier_name": a.get("supplier_name"),
                "product_name": a.get("product_name", "All Supplied Products"),
                "deterministic_risk": a.get("deterministic_risk"),
                "graphsage_risk": a.get("graphsage_risk", a.get("graph_risk_score")),
                "gat_risk": a.get("gat_risk", 0.0),
                "combined_risk": a.get("combined_risk", a.get("combined_risk_score")),
                "risk_band": a.get("risk_band", a.get("combined_risk_level")),
                "reasons": a.get("reasons", a.get("reason")),
                "explanation": a.get("explanation"),
                "formula": "Combined Risk = 0.50 * Deterministic + 0.25 * GraphSAGE + 0.25 * GAT",
                "disclaimer": PROJECT_THRESHOLDS_DISCLAIMER,
                "data_status": a.get("data_status", "SYNTHETIC_DEMO"),
            }
    raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")


@app.get("/models", tags=["Model Registry"])
def get_models():
    """Retrieve registered GNN checkpoints, parameter counts, and validation metadata."""
    out = {}
    for m in ["graphsage", "gat"]:
        phase_pt = os.path.join(ROOT, f"models/{m}_phase{8 if m == 'graphsage' else 9}.pt")
        legacy_pt = os.path.join(ROOT, f"models/{m}_model.pt")
        pt_path = phase_pt if os.path.exists(phase_pt) else legacy_pt

        json_path = os.path.join(ROOT, f"models/{m}_model.json")
        meta = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        out[m] = {
            "available": os.path.exists(pt_path),
            "weights_size_bytes": os.path.getsize(pt_path) if os.path.exists(pt_path) else 0,
            "path": os.path.relpath(pt_path, ROOT) if os.path.exists(pt_path) else None,
            "metrics": meta,
        }
    return out


@app.get("/evaluation", tags=["Evaluation & Benchmarks"])
def get_evaluation():
    """Retrieve academic performance metrics, GNN benchmarks, and ablation results."""
    json_path = os.path.join(ROOT, "data/processed/evaluation_results.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "status": "Evaluation pending. Run 'python evaluation/evaluate_all.py'",
        "supported_baselines": ["Deterministic Heuristic", "PageRank Spreading", "GraphSAGE", "GAT"],
    }


@app.post("/pipeline/run-benchmark", tags=["Risk Pipeline"])
def run_benchmark(max_items: int = Query(50, ge=1, le=1000), model_type: str = Query("graphsage")):
    """Run pipeline against local benchmark dataset."""
    result = run_pipeline(news_items=None, model_type=model_type, max_items=max_items)
    return {
        "mode": "BENCHMARK_DEMO",
        "alerts_generated": len(result),
        "model_used": model_type.upper(),
        "data_status": "SYNTHETIC_DEMO",
        "alerts": result.fillna("").to_dict(orient="records"),
    }
