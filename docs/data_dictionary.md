# BDS-35 Data Dictionary

This document provides a comprehensive catalog of all 12 core datasets utilized across the BDS-35 early warning platform, detailing table roles, column schemas, data types, key constraints, and synthetic vs. real-world provenance markers.

---

## 1. Master Entity Datasets (`data/master/`)

### 1.1 `suppliers.csv`
Contains the master registry of tier-1 and sub-tier suppliers.

| Column Name | Data Type | Key / Constraint | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `supplier_id` | String | **Primary Key** (Unique) | Unique supplier identifier | `SUP00001` |
| `name` | String | Not Null | Registered legal corporate name | `Foxconn Precision Tech` |
| `country` | String (ISO2/Name) | Not Null | Country of headquarters | `Taiwan` |
| `criticality_score` | Float | $[0.0, 1.0]$ | Operational importance to the enterprise | `0.85` |
| `financial_stability_score` | Float | $[0.0, 1.0]$ | Financial health indicator (higher = safer) | `0.92` |
| `tier` | Integer | $1, 2, 3$ | Supply chain tier depth | `1` |
| `provenance` | String | Enumerated | Origin of record (`SYNTHETIC_DEMO` or `REAL_DATA`) | `SYNTHETIC_DEMO` |

### 1.2 `products.csv`
Master catalog of assemblies, sub-assemblies, and raw components.

| Column Name | Data Type | Key / Constraint | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `product_id` | String | **Primary Key** (Unique) | Unique product or component identifier | `PRD00042` |
| `name` | String | Not Null | Component or product family name | `Lithium Battery Cell 18650` |
| `category` | String | Not Null | Industry classification | `Electronics` |
| `criticality_score` | Float | $[0.0, 1.0]$ | Impact on final assembly if disrupted | `0.90` |
| `lead_time_days` | Integer | $\ge 0$ | Standard manufacturing & delivery lead time | `45` |
| `provenance` | String | Enumerated | Origin of record | `SYNTHETIC_DEMO` |

### 1.3 `locations.csv`
Geographical reference nodes representing cities, industrial hubs, ports, and administrative regions.

| Column Name | Data Type | Key / Constraint | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `location_id` | String | **Primary Key** (Unique) | Unique geographic node identifier | `LOC00120` |
| `name` | String | Not Null | City or logistics hub name | `Shenzhen Port Hub` |
| `country` | String | Not Null | Sovereign country | `China` |
| `latitude` | Float | $[-90.0, 90.0]$ | WGS84 coordinate latitude | `22.5431` |
| `longitude` | Float | $[-180.0, 180.0]$| WGS84 coordinate longitude | `114.0579` |
| `location_type` | String | Enumerated | Hub type (`PORT`, `CITY`, `AIRPORT`, `LOGISTICS_HUB`) | `PORT` |
| `provenance` | String | Enumerated | Origin of record | `SYNTHETIC_DEMO` |

### 1.4 `facilities.csv`
Physical manufacturing plants, warehouses, and distribution centers operated by suppliers.

| Column Name | Data Type | Key / Constraint | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `facility_id` | String | **Primary Key** (Unique) | Unique physical facility identifier | `FAC00512` |
| `supplier_id` | String | **Foreign Key** (`suppliers.supplier_id`) | Owning supplier | `SUP00001` |
| `location_id` | String | **Foreign Key** (`locations.location_id`) | Geocoded site location | `LOC00120` |
| `facility_name` | String | Not Null | Facility operational designation | `Shenzhen Plant #4` |
| `facility_type` | String | Enumerated | Site type (`MANUFACTURING`, `WAREHOUSE`, `FABRICATION`) | `MANUFACTURING` |
| `provenance` | String | Enumerated | Origin of record | `SYNTHETIC_DEMO` |

---

## 2. Master Relationship Datasets (`data/master/`)

### 2.1 `supplier_products.csv`
Many-to-many bill-of-materials sourcing relationships between suppliers and products.

| Column Name | Data Type | Key / Constraint | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `supplier_id` | String | **Foreign Key** (`suppliers.supplier_id`) | Supplying entity | `SUP00001` |
| `product_id` | String | **Foreign Key** (`products.product_id`) | Sourced product | `PRD00042` |
| `sourcing_share` | Float | $[0.0, 1.0]$ | Fraction of enterprise demand met by this supplier | `0.75` |
| `is_sole_supplier` | Boolean | True / False | Flag indicating single-source vulnerability | `False` |
| `provenance` | String | Enumerated | Origin of record | `SYNTHETIC_DEMO` |

### 2.2 `supplier_locations.csv`
Operational footprint linking suppliers to the geographic regions where they operate.

| Column Name | Data Type | Key / Constraint | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `supplier_id` | String | **Foreign Key** (`suppliers.supplier_id`) | Operating supplier | `SUP00001` |
| `location_id` | String | **Foreign Key** (`locations.location_id`) | Operating location | `LOC00120` |
| `role` | String | Enumerated | Operational role (`HEADQUARTERS`, `OPERATIONAL_HUB`) | `OPERATIONAL_HUB` |
| `provenance` | String | Enumerated | Origin of record | `SYNTHETIC_DEMO` |

---

## 3. Graph Topological Datasets (`data/graph/`)

### 3.1 `nodes.csv`
Exported heterogeneous graph entities formatted for NetworkX and PyG ingestion.

| Column Name | Data Type | Key / Constraint | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `node_id` | String | **Primary Key** (Unique) | Globally unique node identifier | `SUP:SUP00001` |
| `node_type` | String | Enumerated | `NEWS`, `EVENT`, `SUPPLIER`, `PRODUCT`, `LOCATION`, `FACILITY` | `SUPPLIER` |
| `raw_id` | String | Not Null | Entity key stripped of type prefix | `SUP00001` |
| `name` | String | Nullable | Human-readable entity label | `Foxconn Precision Tech` |
| `criticality` | Float | $[0.0, 1.0]$ | Baseline criticality feature | `0.85` |
| `provenance` | String | Enumerated | Origin of node record | `SYNTHETIC_DEMO` |

### 3.2 `edges.csv`
Exported directed relationship graph edges preserving provenance and weight.

| Column Name | Data Type | Key / Constraint | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `source` | String | **Foreign Key** (`nodes.node_id`) | Directed edge source | `SUP:SUP00001` |
| `target` | String | **Foreign Key** (`nodes.node_id`) | Directed edge target | `PRD:PRD00042` |
| `edge_type` | String | Enumerated | One of 7 canonical relationship types | `SUPPLIER_PROVIDES_PRODUCT` |
| `weight` | Float | $> 0.0$ | Dependency or connection strength | `0.75` |
| `provenance` | String | Enumerated | Provenance of edge connection | `SYNTHETIC_DEMO` |

---

## 4. News, NLP & Risk Artifacts (`data/processed/`)

### 4.1 `news.csv` / `live_news.csv`
Aggregated and processed news records from static benchmarks and live NewsAPI polling.

| Column Name | Data Type | Key / Constraint | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `article_id` | String | **Primary Key** (Unique) | Unique SHA-256 hash or sequential ID | `ART_7a8b9c...` |
| `title` | String | Not Null | Headline of news broadcast | `Strike Paralyzes Port Logistics in Rotterdam` |
| `content` | String | Not Null | Full article text or sanitized snippet | `Dockworkers commenced a 48-hour strike...` |
| `published_at` | DateTime (ISO8601) | Not Null | Publication timestamp | `2026-09-24T14:30:00Z` |
| `source` | String | Not Null | Publishing outlet or domain | `Reuters Logistics` |
| `url` | String | URL format | Canonical article permalink | `https://example.com/port-strike` |
| `relevance_score` | Float | $[0.0, 1.0]$ | Domain relevance keyword score | `0.94` |
| `provenance` | String | Enumerated | `REAL_DATA` (NewsAPI) or `SYNTHETIC_DEMO` | `REAL_DATA` |

### 4.2 `events.csv`
Disruption events extracted from news articles by the local NLP pipeline.

| Column Name | Data Type | Key / Constraint | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `event_id` | String | **Primary Key** (Unique) | Unique event identifier | `EVT_00012` |
| `article_id` | String | **Foreign Key** (`news.article_id`) | Source article | `ART_7a8b9c...` |
| `event_type` | String | Enumerated | 1 of 10 disruption categories | `LABOR_STRIKE` |
| `severity` | Float | $[0.0, 1.0]$ | NLP-computed disruption intensity | `0.80` |
| `location_id` | String | Nullable FK (`locations.location_id`) | Resolved epicenter location | `LOC00014` |
| `provenance` | String | Enumerated | Origin of event record | `SYNTHETIC_DEMO` |

### 4.3 `news_entities.csv`
Named entity spans extracted from articles and linked to master entities.

| Column Name | Data Type | Key / Constraint | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `entity_mention_id` | String | **Primary Key** (Unique) | Unique mention record | `ENT_00451` |
| `article_id` | String | **Foreign Key** (`news.article_id`) | Containing article | `ART_7a8b9c...` |
| `entity_text` | String | Not Null | Extracted text span | `Foxconn Tech Ltd` |
| `entity_type` | String | Enumerated | `SUPPLIER`, `PRODUCT`, `LOCATION` | `SUPPLIER` |
| `linked_id` | String | Master ID or `UNKNOWN` | Resolved master data key | `SUP00001` |
| `match_method` | String | Enumerated | `EXACT`, `NORMALIZED`, `ALIAS`, `FUZZY`, `UNKNOWN` | `NORMALIZED` |
| `confidence` | Float | $[0.0, 1.0]$ | Entity linking confidence score | `0.95` |

### 4.4 `alerts.csv` / `final_alerts.csv`
Synthesized operational early-warning risk alerts.

| Column Name | Data Type | Key / Constraint | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `alert_id` | String | **Primary Key** (Unique) | Unique alert identifier | `ALT_001` |
| `supplier_id` | String | **Foreign Key** (`suppliers.supplier_id`) | Vulnerable supplier | `SUP00001` |
| `event_id` | String | **Foreign Key** (`events.event_id`) | Triggering event | `EVT_00012` |
| `deterministic_risk`| Float | $[0.0, 1.0]$ | Heuristic weighted risk score | `0.78` |
| `graphsage_risk` | Float | $[0.0, 1.0]$ | GraphSAGE structural risk | `0.72` |
| `gat_risk` | Float | $[0.0, 1.0]$ | GAT attention propagation risk | `0.74` |
| `combined_risk` | Float | $[0.0, 1.0]$ | Weighted tri-model synthesized score | `0.755` |
| `risk_band` | String | Enumerated | Operational band (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) | `CRITICAL` |
| `explanation` | String | Structured Text | Human-interpretable rationale for score | `Risk increased because: 1. Event severity...` |
| `provenance` | String | Enumerated | `SYNTHETIC_DEMO` or `REAL_DATA` | `REAL_DATA` |
