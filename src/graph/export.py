"""Graph statistics computation and CSV export for BDS-35 heterogeneous graph."""
from __future__ import annotations
import os
from typing import Any
import numpy as np
import pandas as pd
import networkx as nx

from src.graph.schema import NODE_COLUMNS, EDGE_COLUMNS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_NODES_PATH = os.path.join(ROOT, "data/graph/nodes.csv")
DEFAULT_EDGES_PATH = os.path.join(ROOT, "data/graph/edges.csv")


def compute_graph_statistics(G: nx.DiGraph) -> dict[str, Any]:
    """Compute comprehensive graph topology, degree, component, and isolation statistics.
    
    Generates:
      - node count by type
      - edge count by type
      - degree statistics (min, max, mean, median)
      - connected components (weakly and strongly connected)
      - isolated nodes
    """
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    # 1. Node count by type
    nodes_by_type: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        nt = str(d.get("node_type", "UNKNOWN"))
        nodes_by_type[nt] = nodes_by_type.get(nt, 0) + 1

    # 2. Edge count by type
    edges_by_type: dict[str, int] = {}
    for _, _, d in G.edges(data=True):
        et = str(d.get("edge_type", "UNKNOWN"))
        edges_by_type[et] = edges_by_type.get(et, 0) + 1

    # 3. Degree statistics
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    total_degrees = [in_deg + out_deg for in_deg, out_deg in zip(in_degrees, out_degrees)]

    def calc_stats(deg_list: list[int]) -> dict[str, float]:
        if not deg_list:
            return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
        arr = np.array(deg_list)
        return {
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
        }

    degree_stats = {
        "in_degree": calc_stats(in_degrees),
        "out_degree": calc_stats(out_degrees),
        "total_degree": calc_stats(total_degrees),
    }

    # 4. Connected components
    wcc = list(nx.weakly_connected_components(G))
    scc = list(nx.strongly_connected_components(G))
    wcc_count = len(wcc)
    scc_count = len(scc)
    largest_wcc_size = max(len(c) for c in wcc) if wcc else 0

    # 5. Isolated nodes (degree == 0)
    isolated = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "nodes_by_type": nodes_by_type,
        "edges_by_type": edges_by_type,
        "degree_statistics": degree_stats,
        "connected_components": {
            "weakly_connected": wcc_count,
            "strongly_connected": scc_count,
            "largest_component_size": largest_wcc_size,
        },
        "isolated_nodes": {
            "count": len(isolated),
            "node_ids": isolated[:20],  # preview top 20
        },
    }


def export_graph_to_csv(
    G: nx.DiGraph,
    nodes_path: str = DEFAULT_NODES_PATH,
    edges_path: str = DEFAULT_EDGES_PATH,
) -> tuple[int, int]:
    """Export NetworkX graph nodes and edges to standardized CSV files."""
    os.makedirs(os.path.dirname(nodes_path), exist_ok=True)
    os.makedirs(os.path.dirname(edges_path), exist_ok=True)

    # Export Nodes
    node_rows = []
    for nid, d in G.nodes(data=True):
        row = {
            "node_id": nid,
            "node_type": d.get("node_type", "UNKNOWN"),
            "label": d.get("label", nid),
            "source": d.get("source", "GRAPH_BUILDER"),
            "data_status": d.get("data_status", "SYNTHETIC_DEMO"),
        }
        node_rows.append(row)

    nodes_df = pd.DataFrame(node_rows)
    for col in NODE_COLUMNS:
        if col not in nodes_df.columns:
            nodes_df[col] = ""
    nodes_df = nodes_df[NODE_COLUMNS]
    nodes_df.to_csv(nodes_path, index=False)

    # Export Edges
    edge_rows = []
    for u, v, d in G.edges(data=True):
        row = {
            "source": u,
            "target": v,
            "edge_type": d.get("edge_type", "RELATION"),
            "weight": round(float(d.get("weight", 1.0)), 4),
            "data_status": d.get("data_status", "SYNTHETIC_DEMO"),
        }
        edge_rows.append(row)

    edges_df = pd.DataFrame(edge_rows)
    for col in EDGE_COLUMNS:
        if col not in edges_df.columns:
            edges_df[col] = ""
    edges_df = edges_df[EDGE_COLUMNS]
    edges_df.to_csv(edges_path, index=False)

    return len(nodes_df), len(edges_df)
