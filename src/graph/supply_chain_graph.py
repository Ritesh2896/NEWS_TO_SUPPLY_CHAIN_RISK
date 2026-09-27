"""Supply Chain Graph construction and topological risk propagation for BDS-35.

Constructs a heterogeneous NetworkX supply chain graph connecting suppliers, products,
locations, events, and news articles with edge weights and attributes.
Implements shock propagation, ego-subgraph extraction, and network centrality metrics.
"""
from __future__ import annotations
import os
from typing import Any
import numpy as np
import pandas as pd
import networkx as nx


def build_supply_chain_graph(
    suppliers_df: pd.DataFrame | None = None,
    products_df: pd.DataFrame | None = None,
    locations_df: pd.DataFrame | None = None,
    supplier_products_df: pd.DataFrame | None = None,
    supplier_locations_df: pd.DataFrame | None = None,
    nodes_df: pd.DataFrame | None = None,
    edges_df: pd.DataFrame | None = None,
    relationships_df: pd.DataFrame | None = None,
) -> nx.DiGraph:
    """Build a comprehensive heterogeneous supply chain graph.

    Supports both relational tables (suppliers, products, locations, etc.)
    and graph exports (nodes.csv, edges.csv) for maximum flexibility.
    """
    G = nx.DiGraph()

    # 1. Load from graph export if provided
    if nodes_df is not None and not nodes_df.empty:
        for _, r in nodes_df.iterrows():
            nid = str(r["node_id"])
            G.add_node(
                nid,
                node_type=str(r.get("node_type", "UNKNOWN")).upper(),
                label=str(r.get("label", nid)),
                source=str(r.get("source", "PROJECT_DEMO_EXPANSION")),
                data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
            )

    if edges_df is not None and not edges_df.empty:
        for _, r in edges_df.iterrows():
            u = str(r["source"])
            v = str(r["target"])
            w = float(r.get("weight", 1.0))
            etype = str(r.get("edge_type", "RELATION"))
            status = str(r.get("data_status", "SYNTHETIC_DEMO"))
            G.add_edge(u, v, weight=w, edge_type=etype, data_status=status)

    # 2. Add/Enrich supplier nodes
    if suppliers_df is not None and not suppliers_df.empty:
        for _, r in suppliers_df.iterrows():
            sid = str(r["supplier_id"])
            G.add_node(
                sid,
                node_type="SUPPLIER",
                label=str(r.get("supplier_name", sid)),
                tier=str(r.get("tier", "Tier-1")),
                criticality=str(r.get("criticality", "Medium")),
                single_source=str(r.get("single_source", "UNKNOWN")),
                lat=float(r["latitude"]) if pd.notna(r.get("latitude")) else None,
                lon=float(r["longitude"]) if pd.notna(r.get("longitude")) else None,
                city=str(r.get("city", "")),
                country=str(r.get("country", "")),
                industry=str(r.get("industry", "")),
                data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
            )

    # 3. Add/Enrich product nodes
    if products_df is not None and not products_df.empty:
        for _, r in products_df.iterrows():
            pid = str(r["product_id"])
            G.add_node(
                pid,
                node_type="PRODUCT",
                label=str(r.get("product_name", pid)),
                category=str(r.get("category", "")),
                criticality=str(r.get("criticality", "Medium")),
                lead_time_days=float(r["lead_time_days"]) if pd.notna(r.get("lead_time_days")) else 30.0,
                substitutability=str(r.get("substitutability", "MEDIUM")),
                data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
            )

    # 4. Add/Enrich location nodes
    if locations_df is not None and not locations_df.empty:
        for _, r in locations_df.iterrows():
            lid = str(r["location_id"])
            G.add_node(
                lid,
                node_type="LOCATION",
                label=str(r.get("location_name", lid)),
                city=str(r.get("city", "")),
                country=str(r.get("country", "")),
                lat=float(r["latitude"]) if pd.notna(r.get("latitude")) else None,
                lon=float(r["longitude"]) if pd.notna(r.get("longitude")) else None,
                data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
            )

    # 5. Add supplier -> product edges
    sp_df = supplier_products_df if supplier_products_df is not None else relationships_df
    if sp_df is not None and not sp_df.empty and "supplier_id" in sp_df and "product_id" in sp_df:
        for _, r in sp_df.iterrows():
            s = str(r["supplier_id"])
            p = str(r["product_id"])
            dep = str(r.get("dependency_level", "Medium"))
            dep_w = {"Low": 0.3, "Medium": 0.6, "High": 0.9, "Critical": 1.0}.get(dep, 0.5)
            share = float(r.get("share_percent", 10.0)) / 100.0 if pd.notna(r.get("share_percent")) else 0.1
            weight = round(min(1.0, max(0.05, 0.7 * dep_w + 0.3 * share)), 4)
            G.add_edge(
                s, p,
                edge_type="SUPPLIER_PROVIDES_PRODUCT",
                dependency=dep,
                weight=weight,
                data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
            )

    # 6. Add supplier -> location edges
    sl_df = supplier_locations_df if supplier_locations_df is not None else relationships_df
    if sl_df is not None and not sl_df.empty and "supplier_id" in sl_df and "location_id" in sl_df:
        for _, r in sl_df.iterrows():
            s = str(r["supplier_id"])
            loc = str(r["location_id"])
            dep = str(r.get("dependency", "Medium"))
            dep_w = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9, "CRITICAL": 1.0, "Low": 0.3, "Medium": 0.6, "High": 0.9}.get(dep, 0.5)
            G.add_edge(
                s, loc,
                edge_type="SUPPLIER_HAS_LOCATION",
                weight=dep_w,
                facility_type=str(r.get("facility_type", "Factory")),
                data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
            )
            # Bi-directional impact: disruption at location affects supplier
            G.add_edge(
                loc, s,
                edge_type="LOCATION_AFFECTS_SUPPLIER",
                weight=dep_w,
                data_status=str(r.get("data_status", "SYNTHETIC_DEMO")),
            )

    return G


def propagate_risk(
    G: nx.DiGraph,
    initial_shocks: dict[str, float],
    damping: float = 0.60,
    max_iter: int = 15,
    tol: float = 1e-4,
) -> dict[str, float]:
    """Topological risk diffusion through the supply chain graph.

    Propagates shock from affected locations/suppliers along directed edges:
    R^{(t+1)}(v) = (1 - damping) * R^{(0)}(v) + damping * sum_{u in In(v)} (w_{uv} / sum_k w_{kv}) * R^{(t)}(u)
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}

    # Initialize risk vector
    current_risk = {n: float(initial_shocks.get(n, 0.0)) for n in nodes}
    base_risk = current_risk.copy()

    # Precompute incoming edge weights normalized
    in_neighbors = {}
    for v in nodes:
        preds = list(G.predecessors(v))
        if preds:
            total_w = sum(float(G[u][v].get("weight", 1.0)) for u in preds) or 1.0
            in_neighbors[v] = [(u, float(G[u][v].get("weight", 1.0)) / total_w) for u in preds]
        else:
            in_neighbors[v] = []

    for _ in range(max_iter):
        next_risk = {}
        max_diff = 0.0
        for v in nodes:
            incoming_shock = sum(w_norm * current_risk[u] for u, w_norm in in_neighbors[v]) if in_neighbors[v] else 0.0
            new_val = (1.0 - damping) * base_risk[v] + damping * incoming_shock
            new_val = min(1.0, max(0.0, new_val))
            max_diff = max(max_diff, abs(new_val - current_risk[v]))
            next_risk[v] = new_val

        current_risk = next_risk
        if max_diff < tol:
            break

    return {k: round(v, 4) for k, v in current_risk.items()}


def get_ego_network(G: nx.DiGraph, node_id: str, radius: int = 2) -> dict[str, Any]:
    """Extract an ego-subgraph around a specified node for explanation and visualization."""
    if node_id not in G:
        return {"nodes": [], "edges": []}

    # Convert to undirected view for ego radius search, then extract subgraph
    undirected_G = G.to_undirected(as_view=True)
    ego_nodes = set(nx.ego_graph(undirected_G, node_id, radius=radius).nodes())
    subG = G.subgraph(ego_nodes)

    nodes_list = []
    for n, data in subG.nodes(data=True):
        nodes_list.append({
            "id": n,
            "label": data.get("label", n),
            "type": data.get("node_type", "UNKNOWN"),
            "criticality": data.get("criticality", "Medium"),
            "tier": data.get("tier", ""),
        })

    edges_list = []
    for u, v, data in subG.edges(data=True):
        edges_list.append({
            "source": u,
            "target": v,
            "type": data.get("edge_type", "RELATION"),
            "weight": round(float(data.get("weight", 1.0)), 3),
        })

    return {"nodes": nodes_list, "edges": edges_list}


def compute_graph_metrics(G: nx.DiGraph) -> dict[str, Any]:
    """Compute structural graph statistics and centralities."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0:
        return {"num_nodes": 0, "num_edges": 0, "density": 0.0, "pagerank": {}}

    density = nx.density(G)
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100)
    except Exception:
        pr = {n: 1.0 / n_nodes for n in G.nodes()}

    node_types = {}
    for _, data in G.nodes(data=True):
        t = data.get("node_type", "UNKNOWN")
        node_types[t] = node_types.get(t, 0) + 1

    return {
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "density": round(density, 6),
        "node_types": node_types,
        "pagerank": pr,
    }


def load_master_graph(base_dir: str = "data") -> nx.DiGraph:
    """Convenience loader reading all master tables and graph edge exports."""
    master_dir = os.path.join(base_dir, "master")
    graph_dir = os.path.join(base_dir, "graph")

    def _read_csv(p):
        return pd.read_csv(p) if os.path.exists(p) else None

    suppliers_df = _read_csv(os.path.join(master_dir, "suppliers.csv"))
    products_df = _read_csv(os.path.join(master_dir, "products.csv"))
    locations_df = _read_csv(os.path.join(master_dir, "locations.csv"))
    supplier_products_df = _read_csv(os.path.join(master_dir, "supplier_products.csv"))
    supplier_locations_df = _read_csv(os.path.join(master_dir, "supplier_locations.csv"))

    nodes_df = _read_csv(os.path.join(graph_dir, "nodes.csv"))
    edges_df = _read_csv(os.path.join(graph_dir, "edges.csv"))

    return build_supply_chain_graph(
        suppliers_df=suppliers_df,
        products_df=products_df,
        locations_df=locations_df,
        supplier_products_df=supplier_products_df,
        supplier_locations_df=supplier_locations_df,
        nodes_df=nodes_df,
        edges_df=edges_df,
    )
