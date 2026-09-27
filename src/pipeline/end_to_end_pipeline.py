"""End-to-End Pipeline for BDS-35 News-to-Risk Early Warning System.

Executes the complete pipeline:
  Live News / Benchmark Ingestion
  -> Cross-article Deduplication
  -> Disruption Relevance Detection
  -> NLP Event & Entity Extraction (spaCy + Deterministic Rules)
  -> Entity Linking (Suppliers, Products, Locations via RapidFuzz)
  -> Geospatial Haversine Exposure & Distance Decay
  -> Deterministic Risk Engine & Explainable Factor Attribution
  -> Graph Neural Network Risk Propagation (GraphSAGE / GAT)
  -> Combined Risk Synthesis & Alert Persisting
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Any
import pandas as pd
from dotenv import load_dotenv

from src.alerts.alert_engine import create_alert
from src.deduplication.deduplicate import is_duplicate
from src.gnn.inference import score_graph
from src.linking.entity_linking import link_location, link_product, link_supplier
from src.logging_config import logger
from src.nlp.entity_extraction import extract_entities
from src.nlp.event_extraction import extract_event
from src.nlp.live_enrichment import enrich_article
from src.relevance.relevance import calculate_relevance
from src.risk.geospatial_exposure import calculate_supplier_exposure
from src.risk.risk_engine import calculate_risk

load_dotenv()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MASTER = os.path.join(ROOT, "data/master")
PROCESSED = os.path.join(ROOT, "data/processed")
GRAPH_DIR = os.path.join(ROOT, "data/graph")


def _read_master(filename: str) -> pd.DataFrame:
    """Helper to read master or processed CSV files safely."""
    path_m = os.path.join(MASTER, filename)
    if os.path.exists(path_m):
        return pd.read_csv(path_m)
    path_p = os.path.join(PROCESSED, filename)
    if os.path.exists(path_p):
        return pd.read_csv(path_p)
    path_g = os.path.join(GRAPH_DIR, filename)
    if os.path.exists(path_g):
        return pd.read_csv(path_g)
    return pd.DataFrame()


def run_pipeline(
    news_items: list[dict[str, Any]] | None = None,
    model_type: str = "graphsage",
    max_items: int | None = None,
) -> pd.DataFrame:
    """Run the complete BDS-35 pipeline on live news or benchmark dataset."""
    # 1. Determine news inputs
    data_status = "REAL_DATA"
    if news_items is None:
        live_path = os.path.join(PROCESSED, "live_news.csv")
        if os.path.exists(live_path):
            df_live = pd.read_csv(live_path).fillna("")
            if not df_live.empty:
                news_items = df_live.to_dict(orient="records")
                data_status = "REAL_DATA"

        # Fallback to demo/benchmark news if live news is absent or empty
        if not news_items:
            bench_path = os.path.join(PROCESSED, "news.csv")
            if os.path.exists(bench_path):
                df_bench = pd.read_csv(bench_path).fillna("")
                news_items = df_bench.to_dict(orient="records")
                data_status = "SYNTHETIC_DEMO"
            else:
                news_items = []

    if max_items and news_items:
        news_items = news_items[:max_items]

    # 2. Load Master Data Tables
    suppliers_df = _read_master("suppliers.csv")
    products_df = _read_master("products.csv")
    locations_df = _read_master("locations.csv")
    supplier_products_df = _read_master("supplier_products.csv")
    supplier_locations_df = _read_master("supplier_locations.csv")
    edges_df = _read_master("edges.csv")

    raw_alerts: list[dict[str, Any]] = []
    events_live: list[dict[str, Any]] = []
    seen_texts: list[str] = []

    # 3. Process Each News Article
    for i, news in enumerate(news_items, 1):
        news_id = str(news.get("news_id", news.get("article_id", f"NEWS-{i:05d}")))
        title = str(news.get("title", "")).strip()
        content = str(news.get("content", news.get("description", news.get("article_text", "")))).strip()
        text = f"{title} {content}".strip()

        if not text:
            continue

        # Step 3a: Deduplication
        if is_duplicate(text, seen_texts, threshold=0.75):
            continue
        seen_texts.append(text)

        # Step 3b: Disruption Relevance Detection
        rel_res = calculate_relevance(text)
        if not rel_res.get("label", False):
            continue

        # Step 3c: NLP Entity Extraction
        entities = extract_entities(text)
        org_mentions = entities.get("organizations", [])
        loc_mentions = entities.get("locations", [])

        # Step 3d: NLP Event Extraction
        event = extract_event(text)
        if not event:
            # Fallback event structure if relevance passed
            event = {
                "event_type": "Supply Disruption",
                "trigger": "disruption",
                "impact": "Operational Delay",
                "severity": "Medium",
            }

        # Step 3e: Entity Linking
        # Link Location
        loc_link = link_location(loc_mentions, locations_df) if not locations_df.empty else None
        event_loc_name = loc_link["location_name"] if loc_link else (loc_mentions[0] if loc_mentions else "Unknown Location")
        event_loc_id = loc_link["location_id"] if loc_link else None

        # Determine event coordinates
        event_lat, event_lon = None, None
        if loc_link and not locations_df.empty:
            matched_loc = locations_df[locations_df["location_id"].astype(str) == str(event_loc_id)]
            if not matched_loc.empty:
                event_lat = matched_loc.iloc[0].get("latitude")
                event_lon = matched_loc.iloc[0].get("longitude")

        # Link Supplier
        sup_link = link_supplier(org_mentions, suppliers_df) if not suppliers_df.empty else None
        supplier_id = sup_link["supplier_id"] if sup_link else None
        supplier_name = sup_link["supplier_name"] if sup_link else None

        supplier_row = None
        if supplier_id and not suppliers_df.empty:
            match_s = suppliers_df[suppliers_df["supplier_id"].astype(str) == str(supplier_id)]
            if not match_s.empty:
                supplier_row = match_s.iloc[0]

        # Link Product: either through supplier's supply portfolio or text mentions
        product_row = None
        product_id = None
        product_name = None

        if supplier_id and not supplier_products_df.empty and not products_df.empty:
            s_prods = supplier_products_df[supplier_products_df["supplier_id"].astype(str) == str(supplier_id)]
            if not s_prods.empty:
                # Pick product with highest dependency or share
                p_id_match = str(s_prods.iloc[0]["product_id"])
                p_match = products_df[products_df["product_id"].astype(str) == p_id_match]
                if not p_match.empty:
                    product_row = p_match.iloc[0]
                    product_id = p_id_match
                    product_name = str(product_row.get("product_name", ""))

        if not product_id and not products_df.empty:
            prod_link = link_product([title, content], products_df)
            if prod_link:
                product_id = prod_link["product_id"]
                product_name = prod_link["product_name"]
                p_match = products_df[products_df["product_id"].astype(str) == str(product_id)]
                if not p_match.empty:
                    product_row = p_match.iloc[0]

        # Step 3f: Geospatial Exposure
        geo_result = calculate_supplier_exposure(
            event_lat=event_lat,
            event_lon=event_lon,
            supplier_row=supplier_row,
            locations_df=locations_df,
            supplier_locations_df=supplier_locations_df,
        )
        exposure = geo_result["exposure_score"]
        distance_km = geo_result["distance_km"]

        # Step 3g: Deterministic Risk Calculation
        sev = event.get("severity", "Medium")
        dep = supplier_row.get("dependency_level", supplier_row.get("dependency", "Medium")) if supplier_row is not None else "Medium"
        crit = supplier_row.get("criticality", "Medium") if supplier_row is not None else "Medium"
        p_crit = product_row.get("criticality", "Medium") if product_row is not None else None
        single_src = supplier_row.get("single_source", False) if supplier_row is not None else False

        risk = calculate_risk(
            severity=sev,
            dependency=dep,
            criticality=crit,
            geographic_exposure=exposure,
            single_source=single_src,
            product_criticality=p_crit,
        )

        item_status = str(news.get("data_status", data_status))

        # Record structured event
        events_live.append({
            "event_id": f"EVT-{news_id}",
            "news_id": news_id,
            "title": title,
            "event_type": event.get("event_type", "Disruption"),
            "trigger": event.get("trigger", ""),
            "impact": event.get("impact", ""),
            "severity": sev,
            "location_name": event_loc_name,
            "location_id": event_loc_id,
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "source": news.get("source", news.get("source_name", "")),
            "url": news.get("url", ""),
            "published_at": news.get("published_at", ""),
            "data_status": item_status,
        })

        # Record draft alert
        raw_alerts.append({
            "event_id": f"EVT-{news_id}",
            "news_id": news_id,
            "title": title,
            "source": news.get("source", news.get("source_name", "")),
            "url": news.get("url", ""),
            "published_at": news.get("published_at", ""),
            "event_type": event.get("event_type", "Disruption"),
            "event_location": event_loc_name,
            "location_id": event_loc_id,
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "product_id": product_id,
            "product_name": product_name,
            "geographic_exposure": exposure,
            "distance_km": distance_km,
            "risk_result": risk,
            "data_status": item_status,
        })

    # Step 4: Graph Propagation via GraphSAGE / GAT
    alert_draft_df = pd.DataFrame([
        {
            "supplier_id": a["supplier_id"],
            "risk_score": a["risk_result"]["risk_score"],
            "geographic_exposure": a["geographic_exposure"],
        }
        for a in raw_alerts if a["supplier_id"]
    ])

    graph_risk_scores = score_graph(
        suppliers=suppliers_df,
        relationships=edges_df,
        alerts=alert_draft_df,
        model_type=model_type,
        supplier_products=supplier_products_df,
        supplier_locations=supplier_locations_df,
    )

    # Step 5: Synthesize Combined Risk and Construct Final Alerts
    final_alerts: list[dict[str, Any]] = []
    for a in raw_alerts:
        sid = str(a["supplier_id"]) if a["supplier_id"] else ""
        g_risk = graph_risk_scores.get(sid, 0.0)

        alert_obj = create_alert(
            event_id=a["event_id"],
            supplier_id=a["supplier_id"],
            product_id=a["product_id"],
            risk_result=a["risk_result"],
            graph_risk=g_risk,
            news_id=a["news_id"],
            title=a["title"],
            source=a["source"],
            url=a["url"],
            published_at=a["published_at"],
            event_type=a["event_type"],
            event_location=a["event_location"],
            geographic_exposure=a["geographic_exposure"],
            distance_km=a["distance_km"],
            supplier_name=a["supplier_name"],
            product_name=a["product_name"],
            location_id=a["location_id"],
            data_status=a["data_status"],
        )
        final_alerts.append(alert_obj)

    # Step 6: Persist Outputs
    os.makedirs(PROCESSED, exist_ok=True)
    alerts_df = pd.DataFrame(final_alerts)
    events_df = pd.DataFrame(events_live)

    alerts_df.to_csv(os.path.join(PROCESSED, "final_alerts.csv"), index=False)
    events_df.to_csv(os.path.join(PROCESSED, "events_live.csv"), index=False)

    metrics = {
        "articles_retrieved": len(news_items),
        "articles_relevant": len(events_live),
        "alerts_generated": len(final_alerts),
        "critical_alerts": int((alerts_df["combined_risk_level"] == "CRITICAL").sum()) if not alerts_df.empty else 0,
        "high_alerts": int((alerts_df["combined_risk_level"] == "HIGH").sum()) if not alerts_df.empty else 0,
        "model_used": model_type.upper(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_status": data_status,
    }
    with open(os.path.join(PROCESSED, "pipeline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return alerts_df
