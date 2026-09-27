# BDS-35 FastAPI REST API Documentation

## 1. Overview & Swagger Interface

The BDS-35 backend is built using **FastAPI** (`src/api/main.py`), offering a high-performance, asynchronous RESTful API for supply chain risk analysis, live ingestion, and GNN inference.

- **Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **OpenAPI JSON Schema**: `http://127.0.0.1:8000/openapi.json`
- **CORS Support**: Configured for cross-origin access from Streamlit dashboards and enterprise microservices.

---

## 2. API Endpoints Reference

### 2.1 System & Health

#### `GET /health`
Returns the operational health status of the API service, data directories, and GNN model weights.

- **Response `200 OK`**:
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-09-25T15:20:00Z",
    "models_loaded": {
      "graphsage": true,
      "gat": true
    },
    "database_connected": true,
    "version": "1.0.0"
  }
  ```

---

### 2.2 Live Pipeline & Ingestion

#### `POST /ingest/live`
Triggers immediate real-time news harvesting from NewsAPI, normalizes payloads, and removes duplicates.

- **Request Body**:
  ```json
  {
    "topic": "ports logistics strike disruptions",
    "hours": 24,
    "language": "en",
    "country": null,
    "limit": 10
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "status": "SUCCESS",
    "articles_fetched": 10,
    "duplicates_removed": 2,
    "articles_stored": 8,
    "provenance": "REAL_DATA"
  }
  ```

#### `POST /pipeline/run-live`
Executes the full 8-stage live pipeline end-to-end (NewsAPI → NLP → Entity Linking → Graph Update → Geo Exposure → Deterministic Risk → GNN Inference → Alert Generation). Includes a non-blocking timeout safeguard.

- **Request Body**:
  ```json
  {
    "topic": "India ports suppliers logistics manufacturing disruptions",
    "hours": 48,
    "limit": 10,
    "model_type": "both"
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "status": "COMPLETED",
    "execution_time_seconds": 3.82,
    "topic": "India ports suppliers logistics manufacturing disruptions",
    "stages": {
      "news_ingestion": {"fetched": 10, "accepted": 8, "duplicates_removed": 2},
      "nlp_processing": {"events_extracted": 6, "entities_extracted": 24},
      "entity_linking": {"suppliers_linked": 4, "locations_linked": 5, "products_linked": 3},
      "graph_update": {"nodes_total": 4120, "edges_total": 10840},
      "geo_exposure": {"exposures_calculated": 8},
      "deterministic_risk": {"calculated": 8},
      "gnn_inference": {"models_executed": ["GraphSAGE", "GAT"], "scores_generated": 16},
      "alerts_generated": {"total": 8, "critical": 2, "high": 3, "medium": 2, "low": 1}
    },
    "alerts_count": 8,
    "alerts": [...],
    "data_status": "REAL_DATA",
    "threshold_disclaimer": "PROJECT THRESHOLD DISCLAIMER: Risk bands (LOW, MEDIUM, HIGH, CRITICAL) are project-defined operational prioritization thresholds..."
  }
  ```

---

### 2.3 Master Entities & News

#### `GET /news/live`
Retrieves live or benchmark news records with optional limit and pagination.
- **Query Parameters**:
  - `limit` (int, default: 20)
  - `offset` (int, default: 0)
- **Response `200 OK`**: List of article objects with relevance scores and timestamps.

#### `GET /events`
Retrieves NLP-extracted disruption events with severities and resolved location IDs.
- **Query Parameters**: `event_type` (optional filter), `limit` (default: 50), `offset` (default: 0).
- **Response `200 OK`**: List of event entities with severity and category metadata.

#### `GET /alerts`
Retrieves early-warning risk alerts synthesized by the tri-model engine.
- **Query Parameters**:
  - `risk_band` (optional filter: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
  - `supplier_id` (optional filter)
  - `limit` (int, default: 50)
  - `offset` (int, default: 0)
- **Response `200 OK`**: List of alerts with deterministic, GraphSAGE, GAT, and combined risk scores, plus canonical explanations.

#### `GET /suppliers`
Lists registered enterprise suppliers from master data.
- **Query Parameters**: `tier` (optional: 1, 2, 3), `country` (optional), `limit` (default: 50), `offset` (default: 0).

#### `GET /products`
Lists enterprise products and bill-of-material items with criticality scores.
- **Query Parameters**: `category` (optional), `limit` (default: 50), `offset` (default: 0).

#### `GET /locations`
Lists geocoded logistics nodes, ports, and cities.
- **Query Parameters**: `country` (optional), `limit` (default: 50), `offset` (default: 0).

---

### 2.4 Graph Analytics & Entity Risk

#### `GET /graph/stats`
Computes and returns summary structural metrics for the supply chain graph.
- **Response `200 OK`**:
  ```json
  {
    "node_count": 4120,
    "edge_count": 10840,
    "node_counts_by_type": {
      "SUPPLIER": 1000,
      "PRODUCT": 1000,
      "LOCATION": 1000,
      "FACILITY": 1000,
      "EVENT": 60,
      "NEWS": 60
    },
    "edge_counts_by_type": {
      "SUPPLIER_PROVIDES_PRODUCT": 3000,
      "SUPPLIER_DEPENDS_ON_SUPPLIER": 4000,
      "LOCATION_AFFECTS_FACILITY": 1000,
      "FACILITY_BELONGS_TO_SUPPLIER": 1000,
      "PRODUCT_DEPENDS_ON_PRODUCT": 1800,
      "EVENT_AT_LOCATION": 20,
      "NEWS_HAS_EVENT": 20
    },
    "density": 0.00064,
    "is_connected": false,
    "connected_components": 14
  }
  ```

#### `GET /risk/{entity_id}`
Returns complete risk profile, factor breakdown, and multi-hop neighborhood for a specific supplier.
- **Path Parameter**: `entity_id` (e.g., `SUP00014`)
- **Response `200 OK`**:
  ```json
  {
    "supplier_id": "SUP00014",
    "name": "Shenzhen Semiconductor Components",
    "tier": 1,
    "deterministic_risk": 0.78,
    "graphsage_risk": 0.72,
    "gat_risk": 0.74,
    "combined_risk": 0.755,
    "risk_band": "CRITICAL",
    "contributing_factors": {
      "event_severity": 0.80,
      "geographic_exposure": 1.00,
      "supplier_criticality": 0.85,
      "product_criticality": 0.90,
      "dependency_strength": 0.75,
      "single_source": 1.00
    },
    "explanation": "Risk increased because: 1. Event severity is high...",
    "upstream_suppliers": ["SUP00088", "SUP00112"],
    "downstream_dependents": ["SUP00003"]
  }
  ```

---

### 2.5 Analyst Feedback Loop

#### `POST /feedback`
Allows procurement risk analysts to log qualitative feedback, validate alerts, or report false positives.
- **Request Body**:
  ```json
  {
    "alert_id": "ALT_001",
    "supplier_id": "SUP00014",
    "analyst_id": "ANALYST_42",
    "validation_status": "CONFIRMED_DISRUPTION",
    "notes": "Spoke with logistics liaison; alternative air freight booked."
  }
  ```
- **Response `201 Created`**:
  ```json
  {
    "status": "FEEDBACK_LOGGED",
    "feedback_id": "FBK_00091",
    "recorded_at": "2026-09-25T15:20:00Z"
  }
  ```
