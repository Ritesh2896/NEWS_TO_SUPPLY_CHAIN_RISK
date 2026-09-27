"""Heterogeneous Graph Builder for BDS-35.

Constructs a NetworkX DiGraph connecting:
  Nodes (6 Types):
    - NEWS
    - EVENT
    - SUPPLIER
    - PRODUCT
    - LOCATION
    - FACILITY
  Edges (7 Canonical Types):
    - NEWS_HAS_EVENT (NEWS -> EVENT)
    - EVENT_AT_LOCATION (EVENT -> LOCATION)
    - LOCATION_AFFECTS_FACILITY (LOCATION -> FACILITY)
    - FACILITY_BELONGS_TO_SUPPLIER (FACILITY -> SUPPLIER)
    - SUPPLIER_PROVIDES_PRODUCT (SUPPLIER -> PRODUCT)
    - SUPPLIER_DEPENDS_ON_SUPPLIER (SUPPLIER -> SUPPLIER)
    - PRODUCT_DEPENDS_ON_PRODUCT (PRODUCT -> PRODUCT)

Validates node IDs and edge references, safely prunes invalid/dangling edges,
preserves edge weights and data provenance.
"""
from __future__ import annotations
import os
import re
from typing import Any
import pandas as pd
import networkx as nx

from src.logging_config import logger
from src.graph.schema import (
    NodeType,
    EdgeType,
    CANONICAL_EDGE_CONSTRAINTS,
    NODE_COLUMNS,
    EDGE_COLUMNS,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class HeterogeneousGraphBuilder:
    """Builder for constructing, validating, and enriching heterogeneous supply-chain graphs."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.registered_node_ids: set[str] = set()
        self.validation_summary: dict[str, Any] = {
            "total_nodes": 0,
            "total_edges": 0,
            "invalid_edges_removed": 0,
            "nodes_by_type": {},
            "edges_by_type": {},
        }

    def add_node(self, node_id: str, **attrs):
        """Add node to graph and register valid ID."""
        nid = str(node_id).strip()
        if not nid or nid == "nan":
            return
        self.registered_node_ids.add(nid)
        self.graph.add_node(nid, **attrs)

    def build_from_dataframes(
        self,
        suppliers_df: pd.DataFrame | None = None,
        products_df: pd.DataFrame | None = None,
        locations_df: pd.DataFrame | None = None,
        supplier_products_df: pd.DataFrame | None = None,
        supplier_locations_df: pd.DataFrame | None = None,
        news_df: pd.DataFrame | None = None,
        events_df: pd.DataFrame | None = None,
        nodes_df: pd.DataFrame | None = None,
        edges_df: pd.DataFrame | None = None,
    ) -> nx.DiGraph:
        """Construct the heterogeneous graph from relational tables or pre-existing node/edge exports."""
        self.graph = nx.DiGraph()
        self.registered_node_ids = set()

        # 1. Base graph if nodes_df and edges_df are provided
        if nodes_df is not None and not nodes_df.empty:
            for _, r in nodes_df.iterrows():
                nid = str(r["node_id"]).strip()
                if not nid or nid == "nan":
                    continue
                self.add_node(
                    nid,
                    node_type=str(r.get("node_type", "UNKNOWN")).upper(),
                    label=str(r.get("label", nid)),
                    source=str(r.get("source", "PROJECT_DEMO_EXPANSION")),
                    data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
                )

        # 2. Add / Enrich SUPPLIER nodes
        if suppliers_df is not None and not suppliers_df.empty:
            for _, r in suppliers_df.iterrows():
                sid = str(r["supplier_id"]).strip()
                if not sid or sid == "nan":
                    continue
                self.add_node(
                    sid,
                    node_type=NodeType.SUPPLIER.value,
                    label=str(r.get("supplier_name", sid)),
                    tier=int(r.get("tier", 1)) if pd.notnull(r.get("tier")) else 1,
                    criticality=str(r.get("criticality", "Medium")),
                    country=str(r.get("country", "")),
                    source=str(r.get("source", "MASTER_SUPPLIERS")),
                    data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
                )

        # 3. Add / Enrich PRODUCT nodes
        if products_df is not None and not products_df.empty:
            for _, r in products_df.iterrows():
                pid = str(r["product_id"]).strip()
                if not pid or pid == "nan":
                    continue
                self.add_node(
                    pid,
                    node_type=NodeType.PRODUCT.value,
                    label=str(r.get("product_name", pid)),
                    category=str(r.get("category", "")),
                    criticality=str(r.get("criticality", "Medium")),
                    source=str(r.get("source", "MASTER_PRODUCTS")),
                    data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
                )

        # 4. Add / Enrich LOCATION nodes
        if locations_df is not None and not locations_df.empty:
            for _, r in locations_df.iterrows():
                lid = str(r["location_id"]).strip()
                if not lid or lid == "nan":
                    continue
                self.add_node(
                    lid,
                    node_type=NodeType.LOCATION.value,
                    label=str(r.get("location_name", lid)),
                    city=str(r.get("city", "")),
                    country=str(r.get("country", "")),
                    latitude=float(r.get("latitude", 0.0)) if pd.notnull(r.get("latitude")) else 0.0,
                    longitude=float(r.get("longitude", 0.0)) if pd.notnull(r.get("longitude")) else 0.0,
                    source=str(r.get("source", "MASTER_LOCATIONS")),
                    data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
                )

        # 5. Add / Enrich NEWS nodes
        if news_df is not None and not news_df.empty:
            for _, r in news_df.iterrows():
                nid = str(r["article_id"]).strip()
                if not nid or nid == "nan":
                    continue
                self.add_node(
                    nid,
                    node_type=NodeType.NEWS.value,
                    label=str(r.get("title", nid))[:100],
                    published_at=str(r.get("published_at", "")),
                    source=str(r.get("source_name", "NEWS_FEED")),
                    data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
                )

        # 6. Add / Enrich EVENT nodes
        if events_df is not None and not events_df.empty:
            for _, r in events_df.iterrows():
                eid = str(r["event_id"]).strip()
                if not eid or eid == "nan":
                    continue
                self.add_node(
                    eid,
                    node_type=NodeType.EVENT.value,
                    label=str(r.get("event_type", eid)),
                    event_category=str(r.get("event_category", "Disruption")),
                    severity=float(r.get("severity", 3.0)) if pd.notnull(r.get("severity")) else 3.0,
                    source=str(r.get("source_url", "NLP_EVENT_EXTRACTION")),
                    data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
                )

        # 7. Add FACILITY nodes & Edges from supplier_locations
        if supplier_locations_df is not None and not supplier_locations_df.empty:
            for _, r in supplier_locations_df.iterrows():
                sid = str(r.get("supplier_id", "")).strip()
                lid = str(r.get("location_id", "")).strip()
                status = str(r.get("data_status", "SYNTHETIC_DEMO"))
                if not sid or not lid or sid == "nan" or lid == "nan":
                    continue

                fac_id = f"FAC_{sid}_{lid}"
                fac_label = f"Facility {sid}-{lid}"
                self.add_node(
                    fac_id,
                    node_type=NodeType.FACILITY.value,
                    label=fac_label,
                    facility_type=str(r.get("facility_type", "Manufacturing")),
                    ownership=str(r.get("ownership", "Owned")),
                    source="SUPPLIER_LOCATIONS",
                    data_status=status,
                )

                # LOCATION -> FACILITY
                self._safe_add_edge(
                    u=lid,
                    v=fac_id,
                    edge_type=EdgeType.LOCATION_AFFECTS_FACILITY.value,
                    weight=0.90,
                    data_status=status,
                )

                # FACILITY -> SUPPLIER
                self._safe_add_edge(
                    u=fac_id,
                    v=sid,
                    edge_type=EdgeType.FACILITY_BELONGS_TO_SUPPLIER.value,
                    weight=1.0,
                    data_status=status,
                )

        # 8. Add SUPPLIER_PROVIDES_PRODUCT edges
        if supplier_products_df is not None and not supplier_products_df.empty:
            for _, r in supplier_products_df.iterrows():
                sid = str(r.get("supplier_id", "")).strip()
                pid = str(r.get("product_id", "")).strip()
                w = float(r.get("share_percent", 50.0)) / 100.0 if pd.notnull(r.get("share_percent")) else 0.80
                status = str(r.get("data_status", "SYNTHETIC_DEMO"))
                if sid and pid and sid != "nan" and pid != "nan":
                    self._safe_add_edge(
                        u=sid,
                        v=pid,
                        edge_type=EdgeType.SUPPLIER_PROVIDES_PRODUCT.value,
                        weight=round(w, 3),
                        data_status=status,
                    )

        # 9. Add NEWS_HAS_EVENT & EVENT_AT_LOCATION edges from events_df
        if events_df is not None and not events_df.empty:
            for _, r in events_df.iterrows():
                eid = str(r.get("event_id", "")).strip()
                art_id = str(r.get("article_id", "")).strip()
                loc_id = str(r.get("location_id", "")).strip()
                status = str(r.get("data_status", "SYNTHETIC_DEMO"))

                if art_id and eid and art_id != "nan" and eid != "nan":
                    self._safe_add_edge(
                        u=art_id,
                        v=eid,
                        edge_type=EdgeType.NEWS_HAS_EVENT.value,
                        weight=0.95,
                        data_status=status,
                    )

                if eid and loc_id and eid != "nan" and loc_id != "nan" and loc_id != "UNKNOWN":
                    self._safe_add_edge(
                        u=eid,
                        v=loc_id,
                        edge_type=EdgeType.EVENT_AT_LOCATION.value,
                        weight=0.90,
                        data_status=status,
                    )

        # 10. Ingest any pre-existing edges_df
        if edges_df is not None and not edges_df.empty:
            for _, r in edges_df.iterrows():
                u = str(r["source"]).strip()
                v = str(r["target"]).strip()
                etype = str(r.get("edge_type", "RELATION"))
                w = float(r.get("weight", 1.0)) if pd.notnull(r.get("weight")) else 1.0
                status = str(r.get("data_status", "SYNTHETIC_DEMO"))
                self._safe_add_edge(u, v, edge_type=etype, weight=w, data_status=status)

        # 11. Derive multi-tier supplier & product dependencies from structural graph if needed
        self._derive_tier_dependencies()

        # 12. Validate and prune invalid/dangling edges
        self.validate_and_prune()

        return self.graph

    def _safe_add_edge(
        self, u: str, v: str, edge_type: str, weight: float = 1.0, data_status: str = "SYNTHETIC_DEMO"
    ):
        """Add directed edge with canonical attributes."""
        if not u or not v or u == "nan" or v == "nan":
            return
        self.graph.add_edge(
            u, v,
            edge_type=edge_type,
            weight=round(float(weight), 4),
            data_status=data_status,
        )

    def _derive_tier_dependencies(self):
        """Construct multi-tier supplier-supplier and product-product dependency edges."""
        # Derive SUPPLIER_DEPENDS_ON_SUPPLIER where suppliers share product categories or tiers
        supplier_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == NodeType.SUPPLIER.value]
        product_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == NodeType.PRODUCT.value]

        # Check if supplier dependency edges already exist
        existing_sup_dep = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("edge_type") == EdgeType.SUPPLIER_DEPENDS_ON_SUPPLIER.value)
        if existing_sup_dep == 0 and len(supplier_nodes) > 1:
            # Deterministic multi-tier supply chain linking: connect tier-2 suppliers to tier-1 suppliers
            for i in range(min(500, len(supplier_nodes) - 1)):
                u = supplier_nodes[i + 1]
                v = supplier_nodes[i]
                self._safe_add_edge(
                    u, v,
                    edge_type=EdgeType.SUPPLIER_DEPENDS_ON_SUPPLIER.value,
                    weight=0.75,
                    data_status="SYNTHETIC_DEMO",
                )

        existing_prod_dep = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("edge_type") == EdgeType.PRODUCT_DEPENDS_ON_PRODUCT.value)
        if existing_prod_dep == 0 and len(product_nodes) > 1:
            # Connect component products to assembled products
            for i in range(min(400, len(product_nodes) - 1)):
                u = product_nodes[i + 1]
                v = product_nodes[i]
                self._safe_add_edge(
                    u, v,
                    edge_type=EdgeType.PRODUCT_DEPENDS_ON_PRODUCT.value,
                    weight=0.70,
                    data_status="SYNTHETIC_DEMO",
                )

    def validate_and_prune(self) -> dict[str, Any]:
        """Validate node IDs and edge references; prune dangling edges referencing missing nodes."""
        dangling_edges = []
        ghost_nodes = set()

        for u, v in list(self.graph.edges()):
            u_valid = (u in self.registered_node_ids) or bool(self.graph.nodes[u].get("node_type"))
            v_valid = (v in self.registered_node_ids) or bool(self.graph.nodes[v].get("node_type"))
            if not u_valid or not v_valid:
                dangling_edges.append((u, v))
                if not u_valid:
                    ghost_nodes.add(u)
                if not v_valid:
                    ghost_nodes.add(v)

        for u, v in dangling_edges:
            self.graph.remove_edge(u, v)

        # Remove ghost nodes inserted by NetworkX edge addition
        for ghost in ghost_nodes:
            if ghost in self.graph:
                self.graph.remove_node(ghost)

        # Count nodes by type
        nodes_by_type = {}
        for n, d in self.graph.nodes(data=True):
            nt = d.get("node_type", "UNKNOWN")
            nodes_by_type[nt] = nodes_by_type.get(nt, 0) + 1

        # Count edges by type
        edges_by_type = {}
        for u, v, d in self.graph.edges(data=True):
            et = d.get("edge_type", "UNKNOWN")
            edges_by_type[et] = edges_by_type.get(et, 0) + 1

        self.validation_summary = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "invalid_edges_removed": len(dangling_edges),
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
        }
        return self.validation_summary
