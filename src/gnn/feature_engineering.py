"""Feature Engineering for Supply-Chain Graph Neural Networks.

Constructs 12-dimensional node feature vectors for suppliers combining:
  1. Direct News Risk
  2. Media / Alert Volume
  3. Geographic Exposure Volatility
  4. Supplier Criticality
  5. Dependency Level
  6. Single-Source Flag
  7. Supply Chain Tier (Tier-1, Tier-2, Tier-3)
  8. Product Concentration
  9. PageRank Centrality
  10. Degree Centrality
  11. Downstream Products Count
  12. Facility Spread / Redundancy
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx

CRIT_MAP = {"LOW": 0.25, "Low": 0.25, "MEDIUM": 0.50, "Medium": 0.50, "HIGH": 0.75, "High": 0.75, "CRITICAL": 1.0, "Critical": 1.0}
TIER_MAP = {"Tier-1": 1.0, "TIER-1": 1.0, "Tier-2": 0.65, "TIER-2": 0.65, "Tier-3": 0.35, "TIER-3": 0.35}


def build_node_features(
    suppliers: pd.DataFrame | None,
    relationships: pd.DataFrame | None = None,
    alerts: pd.DataFrame | None = None,
    supplier_products: pd.DataFrame | None = None,
    supplier_locations: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Construct a 12-dimensional feature matrix and indexed edge list for suppliers."""
    if suppliers is None or suppliers.empty:
        return pd.DataFrame(), []

    suppliers = suppliers.copy()
    rel = relationships.copy() if relationships is not None else pd.DataFrame()
    alerts = alerts.copy() if alerts is not None else pd.DataFrame()
    sp_df = supplier_products.copy() if supplier_products is not None else pd.DataFrame()
    sl_df = supplier_locations.copy() if supplier_locations is not None else pd.DataFrame()

    ids = suppliers["supplier_id"].astype(str).tolist()
    idx = {x: i for i, x in enumerate(ids)}
    n_suppliers = len(ids)

    # 1. Build supplier-to-supplier graph
    # First: from explicit relationship / edges table
    edges = []
    if not rel.empty:
        # Check source/target or supplier_id / supplier_id_to
        src_col = "source" if "source" in rel.columns else "supplier_id"
        dst_col = "target" if "target" in rel.columns else ("supplier_id_to" if "supplier_id_to" in rel.columns else "product_id")

        for _, r in rel.iterrows():
            s = str(r.get(src_col, ""))
            t = str(r.get(dst_col, ""))
            if s in idx and t in idx and s != t:
                edges.append((idx[s], idx[t]))

    # Second: If no direct supplier-supplier edges, infer supply-chain graph
    # via shared products or consecutive tiers in supplier_products
    if not edges and not sp_df.empty and "supplier_id" in sp_df and "product_id" in sp_df:
        prod_to_sups: dict[str, list[str]] = {}
        for _, r in sp_df.iterrows():
            p = str(r["product_id"])
            s = str(r["supplier_id"])
            if s in idx:
                prod_to_sups.setdefault(p, []).append(s)

        for p, s_list in prod_to_sups.items():
            if len(s_list) > 1:
                # Interconnect suppliers producing same critical component
                for i in range(min(len(s_list), 4)):
                    for j in range(i + 1, min(len(s_list), 4)):
                        u, v = idx[s_list[i]], idx[s_list[j]]
                        edges.append((u, v))
                        edges.append((v, u))

    # Dedup edges
    edges = list(set(edges))

    # 2. Compute Network Centralities
    G = nx.Graph()
    G.add_nodes_from(range(n_suppliers))
    G.add_edges_from(edges)

    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=50)
    except Exception:
        pr = {i: 1.0 / n_suppliers for i in range(n_suppliers)}

    # Normalize pagerank
    max_pr = max(pr.values()) if pr else 1.0
    deg = dict(G.degree())
    max_deg = max(deg.values()) if deg and max(deg.values()) > 0 else 1.0

    # 3. Pre-aggregate products and facilities per supplier
    prod_counts = {}
    if not sp_df.empty and "supplier_id" in sp_df:
        prod_counts = sp_df["supplier_id"].astype(str).value_counts().to_dict()

    fac_counts = {}
    if not sl_df.empty and "supplier_id" in sl_df:
        fac_counts = sl_df["supplier_id"].astype(str).value_counts().to_dict()

    # 4. Assemble 12 features per supplier
    rows = []
    for sid in ids:
        i = idx[sid]
        s_row = suppliers[suppliers["supplier_id"].astype(str) == sid].iloc[0]

        # Feature 1: News risk score
        a = alerts[alerts.get("supplier_id", pd.Series(dtype=str)).astype(str) == sid] if not alerts.empty and "supplier_id" in alerts else pd.DataFrame()
        news_risk = float(a["risk_score"].max() / 100.0) if not a.empty and "risk_score" in a else 0.0

        # Feature 2: Media volume (normalized)
        media_vol = float(min(len(a) / 5.0, 1.0))

        # Feature 3: Geographic exposure shock
        geo_exp = float(a["geographic_exposure"].max()) if not a.empty and "geographic_exposure" in a else 0.20

        # Feature 4: Supplier criticality
        crit_str = str(s_row.get("criticality", "Medium"))
        crit_val = CRIT_MAP.get(crit_str, CRIT_MAP.get(crit_str.upper(), 0.50))

        # Feature 5: Dependency level
        dep_str = str(s_row.get("dependency_level", s_row.get("dependency", "Medium")))
        dep_val = CRIT_MAP.get(dep_str, CRIT_MAP.get(dep_str.upper(), 0.50))

        # Feature 6: Single-source flag
        ss_str = str(s_row.get("single_source", "UNKNOWN")).upper()
        single_source = 1.0 if ss_str in {"YES", "TRUE", "1"} else 0.0

        # Feature 7: Tier level
        tier_str = str(s_row.get("tier", "Tier-1"))
        tier_val = TIER_MAP.get(tier_str, 0.65)

        # Feature 8: Supplier concentration
        n_prods = prod_counts.get(sid, 1)
        concentration = float(min(1.0, 1.0 / max(1, n_prods)))

        # Feature 9: PageRank centrality
        pr_val = float(pr.get(i, 0.0) / (max_pr or 1.0))

        # Feature 10: Degree centrality
        deg_val = float(deg.get(i, 0) / max_deg)

        # Feature 11: Downstream product count
        downstream = float(min(1.0, n_prods / 10.0))

        # Feature 12: Facility spread / redundancy
        n_fac = fac_counts.get(sid, 1)
        facility_spread = float(min(1.0, n_fac / 5.0))

        feat_vector = [
            news_risk,
            media_vol,
            geo_exp,
            crit_val,
            dep_val,
            single_source,
            tier_val,
            concentration,
            pr_val,
            deg_val,
            downstream,
            facility_spread,
        ]
        rows.append([sid, *feat_vector])

    cols = [
        "supplier_id",
        "news_risk",
        "media_volume",
        "geographic_volatility",
        "supplier_criticality",
        "dependency_level",
        "single_source_risk",
        "tier_level",
        "supplier_concentration",
        "pagerank_centrality",
        "degree_centrality",
        "downstream_product_count",
        "facility_spread",
    ]
    return pd.DataFrame(rows, columns=cols), edges
