"""Live article enrichment module connecting news articles to NLP Event and Entity extraction.

Extracts structured events and entities from incoming news articles and persists
them into data/processed/events.csv and data/processed/news_entities.csv.
"""
from __future__ import annotations
import os
from typing import Any, Sequence
import pandas as pd

from src.nlp.entity_extraction import extract_entity_spans
from src.nlp.event_extraction import extract_event
from src.nlp.preprocessing import clean_text

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROCESSED_DIR = os.path.join(ROOT, "data/processed")
EVENTS_CSV_PATH = os.path.join(PROCESSED_DIR, "events.csv")
ENTITIES_CSV_PATH = os.path.join(PROCESSED_DIR, "news_entities.csv")
LIVE_EVENTS_CSV_PATH = os.path.join(PROCESSED_DIR, "events_live.csv")

EVENT_COLUMNS = [
    "event_id", "article_id", "event_type", "event_category", "severity",
    "location_id", "event_time", "extraction_confidence", "evidence_text",
    "source_url", "data_status"
]

ENTITY_COLUMNS = [
    "article_id", "entity_text", "entity_type", "normalized_entity",
    "confidence", "character_start", "character_end", "data_status"
]


def enrich_article(
    title: str,
    content: str = "",
    article_id: str = "UNKNOWN",
    published_at: str | None = None,
    source_url: str = "",
    data_status: str = "REAL_DATA",
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Extract event and entity records from an article.
    
    Returns:
        (event_record_or_None, list_of_entity_records)
    """
    full_text = f"{clean_text(title)}. {clean_text(content)}".strip()
    if not full_text:
        return None, []

    # 1. Event Extraction
    event_dict = extract_event(
        text=full_text,
        article_id=article_id,
        published_at=published_at,
        source_url=source_url,
    )
    if event_dict:
        event_dict["data_status"] = data_status

    # 2. Entity Spans Extraction
    spans = extract_entity_spans(full_text)
    entity_records: list[dict[str, Any]] = []
    for s in spans:
        rec = {
            "article_id": article_id,
            "entity_text": s["entity_text"],
            "entity_type": s["entity_type"],
            "normalized_entity": s["normalized_entity"],
            "confidence": s["confidence"],
            "character_start": s["character_start"],
            "character_end": s["character_end"],
            "data_status": data_status,
        }
        entity_records.append(rec)

    return event_dict, entity_records


def store_events_and_entities(
    events: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    events_path: str = EVENTS_CSV_PATH,
    entities_path: str = ENTITIES_CSV_PATH,
) -> tuple[int, int]:
    """Store extracted events and entities into processed CSV files without duplicating."""
    events_stored = 0
    entities_stored = 0

    # 1. Store Events
    if events:
        os.makedirs(os.path.dirname(events_path), exist_ok=True)
        new_ev_df = pd.DataFrame(events)
        for col in EVENT_COLUMNS:
            if col not in new_ev_df.columns:
                new_ev_df[col] = "UNKNOWN" if col == "location_id" else ""
        new_ev_df = new_ev_df[EVENT_COLUMNS]

        if os.path.exists(events_path):
            existing_ev = pd.read_csv(events_path, dtype=object).fillna("")
            existing_ids = set(existing_ev["event_id"].str.strip())
            filtered_ev = new_ev_df[~new_ev_df["event_id"].str.strip().isin(existing_ids)]
            if not filtered_ev.empty:
                combined_ev = pd.concat([existing_ev, filtered_ev], ignore_index=True)
                combined_ev.to_csv(events_path, index=False)
                events_stored = len(filtered_ev)
        else:
            new_ev_df.to_csv(events_path, index=False)
            events_stored = len(new_ev_df)

        # Also write fresh events to live cache
        os.makedirs(os.path.dirname(LIVE_EVENTS_CSV_PATH), exist_ok=True)
        new_ev_df.to_csv(LIVE_EVENTS_CSV_PATH, index=False)

    # 2. Store Entities
    if entities:
        os.makedirs(os.path.dirname(entities_path), exist_ok=True)
        new_ent_df = pd.DataFrame(entities)
        for col in ENTITY_COLUMNS:
            if col not in new_ent_df.columns:
                new_ent_df[col] = ""
        new_ent_df = new_ent_df[ENTITY_COLUMNS]

        if os.path.exists(entities_path):
            existing_ent = pd.read_csv(entities_path, dtype=object).fillna("")
            existing_keys = set(
                zip(
                    existing_ent["article_id"].str.strip(),
                    existing_ent["character_start"].astype(str),
                    existing_ent["character_end"].astype(str),
                )
            )
            filtered_ent = new_ent_df[
                ~new_ent_df.apply(
                    lambda r: (
                        str(r["article_id"]).strip(),
                        str(r["character_start"]),
                        str(r["character_end"]),
                    ) in existing_keys,
                    axis=1,
                )
            ]
            if not filtered_ent.empty:
                combined_ent = pd.concat([existing_ent, filtered_ent], ignore_index=True)
                combined_ent.to_csv(entities_path, index=False)
                entities_stored = len(filtered_ent)
        else:
            new_ent_df.to_csv(entities_path, index=False)
            entities_stored = len(new_ent_df)

    return events_stored, entities_stored
