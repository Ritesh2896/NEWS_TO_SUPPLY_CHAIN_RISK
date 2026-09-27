"""Schema definitions and column specifications for the BDS-35 Dataset Pack.

Defines required columns, primary keys, relationship keys, and foreign-key constraints
across all 12 dataset pack files:
  - master/: suppliers.csv, products.csv, locations.csv, supplier_products.csv, supplier_locations.csv
  - processed/: news.csv, events.csv, news_entities.csv, entity_links.csv, risk_labels.csv
  - graph/: nodes.csv, edges.csv
"""
from __future__ import annotations
from typing import NamedTuple

DATA_STATUS_SYNTHETIC = "SYNTHETIC_DEMO"
DATA_STATUS_REAL = "REAL_DATA"
DATA_STATUS_DERIVED = "DERIVED"


class ForeignKeyRule(NamedTuple):
    child_table: str
    child_col: str
    parent_table: str
    parent_col: str


# Required columns that must be present in each CSV schema
SCHEMAS: dict[str, list[str]] = {
    "suppliers": [
        "supplier_id", "supplier_name", "country", "state", "city",
        "latitude", "longitude", "supplier_type", "industry", "tier",
        "criticality", "single_source", "status", "source", "source_date",
        "evidence", "data_status"
    ],
    "products": [
        "product_id", "product_name", "category", "sub_category",
        "criticality", "lead_time_days", "substitutability",
        "source", "source_date", "evidence", "data_status"
    ],
    "locations": [
        "location_id", "location_name", "location_type", "country",
        "state", "city", "latitude", "longitude", "region",
        "source", "source_date", "evidence", "data_status"
    ],
    "supplier_products": [
        "supplier_id", "product_id", "relationship_type", "dependency_level",
        "share_percent", "lead_time_days", "source", "source_date",
        "evidence", "data_status"
    ],
    "supplier_locations": [
        "supplier_id", "location_id", "facility_type", "ownership",
        "dependency", "source", "source_date", "evidence", "data_status"
    ],
    "news": [
        "article_id", "source_name", "author", "title", "description",
        "url", "published_at", "content", "query", "retrieved_at", "data_status"
    ],
    "events": [
        "event_id", "article_id", "event_type", "event_category",
        "severity", "location_id", "event_time", "extraction_confidence",
        "evidence_text", "source_url", "data_status"
    ],
    "news_entities": [
        "article_id", "entity_text", "entity_type", "normalized_entity",
        "confidence", "character_start", "character_end", "data_status"
    ],
    "entity_links": [
        "entity_text", "linked_id", "entity_type", "match_method",
        "match_score", "confidence", "data_status"
    ],
    "risk_labels": [
        "entity_id", "entity_type", "event_severity", "geographic_exposure",
        "supplier_criticality", "dependency_strength", "deterministic_risk",
        "graph_propagation_risk", "combined_risk", "label_status", "data_status"
    ],
    "nodes": [
        "node_id", "node_type", "label", "source", "data_status"
    ],
    "edges": [
        "source", "target", "edge_type", "weight", "data_status"
    ],
}

# Primary key columns for uniqueness checking
PRIMARY_KEYS: dict[str, str] = {
    "suppliers": "supplier_id",
    "products": "product_id",
    "locations": "location_id",
    "news": "article_id",
    "events": "event_id",
    "nodes": "node_id",
    "risk_labels": "entity_id",
}

# Composite keys for relationship tables
COMPOSITE_RELATIONSHIP_KEYS: dict[str, list[str]] = {
    "supplier_products": ["supplier_id", "product_id"],
    "supplier_locations": ["supplier_id", "location_id"],
    "edges": ["source", "target", "edge_type"],
}

# Foreign key constraints across dataset tables
FOREIGN_KEY_RULES: list[ForeignKeyRule] = [
    ForeignKeyRule("supplier_products", "supplier_id", "suppliers", "supplier_id"),
    ForeignKeyRule("supplier_products", "product_id", "products", "product_id"),
    ForeignKeyRule("supplier_locations", "supplier_id", "suppliers", "supplier_id"),
    ForeignKeyRule("supplier_locations", "location_id", "locations", "location_id"),
    ForeignKeyRule("events", "article_id", "news", "article_id"),
    ForeignKeyRule("events", "location_id", "locations", "location_id"),
    ForeignKeyRule("news_entities", "article_id", "news", "article_id"),
    ForeignKeyRule("risk_labels", "entity_id", "suppliers", "supplier_id"),
    ForeignKeyRule("edges", "source", "nodes", "node_id"),
    ForeignKeyRule("edges", "target", "nodes", "node_id"),
]

# Relative file locations in standard repo layout
STANDARD_FILE_PATHS: dict[str, str] = {
    "suppliers": "data/master/suppliers.csv",
    "products": "data/master/products.csv",
    "locations": "data/master/locations.csv",
    "supplier_products": "data/master/supplier_products.csv",
    "supplier_locations": "data/master/supplier_locations.csv",
    "news": "data/processed/news.csv",
    "events": "data/processed/events.csv",
    "news_entities": "data/processed/news_entities.csv",
    "entity_links": "data/processed/entity_links.csv",
    "risk_labels": "data/processed/risk_labels.csv",
    "nodes": "data/graph/nodes.csv",
    "edges": "data/graph/edges.csv",
}
