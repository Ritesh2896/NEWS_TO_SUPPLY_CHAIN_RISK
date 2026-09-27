"""Local NLP Processing package for BDS-35.

Provides:
  - Text preprocessing and sentence segmentation
  - Entity extraction (organizations, suppliers, locations, countries, cities, ports, facilities, products)
  - Entity normalization
  - Disruption event extraction (10 event types) and deterministic severity scoring
  - Live article enrichment and CSV storage
"""
from src.nlp.preprocessing import clean_text, split_sentences, extract_context_window
from src.nlp.normalization import normalize_entity_name, normalize_entity_type
from src.nlp.entity_extraction import extract_entities, extract_entity_spans
from src.nlp.event_extraction import extract_event, calculate_deterministic_severity
from src.nlp.live_enrichment import enrich_article, store_events_and_entities

__all__ = [
    "clean_text",
    "split_sentences",
    "extract_context_window",
    "normalize_entity_name",
    "normalize_entity_type",
    "extract_entities",
    "extract_entity_spans",
    "extract_event",
    "calculate_deterministic_severity",
    "enrich_article",
    "store_events_and_entities",
]
