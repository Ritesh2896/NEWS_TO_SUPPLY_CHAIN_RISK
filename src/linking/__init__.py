"""BDS-35 Entity Linking Package.

Resolves extracted NLP mentions to canonical master records (Suppliers, Locations, Products)
using a deterministic 4-stage matching cascade:
  1. Exact matching
  2. Normalized string matching
  3. Alias matching
  4. Controlled fuzzy matching (with numeric conflict penalty)
"""
from src.linking.confidence import LinkingThresholds, calibrate_confidence
from src.linking.entity_linker import EntityLinker, save_entity_links
from src.linking.entity_linking import link_location, link_product, link_supplier
from src.linking.normalization import normalize_linking_text

__all__ = [
    "EntityLinker",
    "LinkingThresholds",
    "calibrate_confidence",
    "normalize_linking_text",
    "save_entity_links",
    "link_supplier",
    "link_location",
    "link_product",
]
