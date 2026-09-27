"""BDS-35 Data Layer.

Provides dataset loading, schema verification, non-destructive normalization,
and data-quality reporting for the 12-file BDS-35 dataset pack.
"""
from src.data.loader import DataLoader
from src.data.normalization import normalize_dataframe, normalize_id, normalize_text
from src.data.schemas import SCHEMAS, PRIMARY_KEYS, FOREIGN_KEY_RULES
from src.data.validators import (
    validate_schema,
    detect_duplicate_ids,
    detect_duplicate_relationships,
    validate_foreign_keys,
    check_missing_values,
    analyze_data_status,
    validate_single_file,
)

__all__ = [
    "DataLoader",
    "normalize_dataframe",
    "normalize_id",
    "normalize_text",
    "validate_schema",
    "detect_duplicate_ids",
    "detect_duplicate_relationships",
    "validate_foreign_keys",
    "check_missing_values",
    "analyze_data_status",
    "validate_single_file",
    "SCHEMAS",
    "PRIMARY_KEYS",
    "FOREIGN_KEY_RULES",
]
