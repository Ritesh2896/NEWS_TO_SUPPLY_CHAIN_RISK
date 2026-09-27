"""Validation and data-quality inspection layer for BDS-35.

Performs strict, honest data verification without silent repair of suspicious records:
  - Validates required columns per schema
  - Detects duplicate primary key identifiers
  - Detects duplicate relationship tuples
  - Quantifies missing/null values
  - Validates cross-table foreign key referential integrity
  - Analyzes synthetic vs real data provenance counts
"""
from __future__ import annotations
from typing import Any
import pandas as pd

from src.data.schemas import (
    COMPOSITE_RELATIONSHIP_KEYS,
    FOREIGN_KEY_RULES,
    ForeignKeyRule,
    PRIMARY_KEYS,
    SCHEMAS,
)


def validate_schema(df: pd.DataFrame, expected_columns: list[str]) -> list[str]:
    """Check for missing required columns in the dataframe."""
    existing = set(df.columns)
    missing = [c for c in expected_columns if c not in existing]
    return missing


def detect_duplicate_ids(df: pd.DataFrame, id_col: str) -> list[str]:
    """Detect and return non-unique identifier values."""
    if id_col not in df.columns or df.empty:
        return []
    duplicates = df[df.duplicated(subset=[id_col], keep=False)][id_col].dropna().unique()
    return [str(x) for x in duplicates]


def detect_duplicate_relationships(df: pd.DataFrame, composite_key: list[str]) -> int:
    """Count duplicate relationship rows for composite keys (e.g., supplier_id + product_id)."""
    if df.empty or not all(k in df.columns for k in composite_key):
        return 0
    return int(df.duplicated(subset=composite_key, keep="first").sum())


def validate_foreign_keys(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    child_col: str,
    parent_col: str,
) -> list[str]:
    """Identify foreign key values in child_df that do not exist in parent_df."""
    if child_df.empty or parent_df.empty:
        return []
    if child_col not in child_df.columns or parent_col not in parent_df.columns:
        return []

    parent_keys = set(parent_df[parent_col].dropna().astype(str).str.strip())
    child_keys = child_df[child_col].dropna().astype(str).str.strip()

    # Find orphan values
    orphans = child_keys[~child_keys.isin(parent_keys)].unique().tolist()
    return orphans


def check_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Return dictionary of column names to missing value counts (only for columns with missing values)."""
    if df.empty:
        return {}
    null_counts = df.isnull().sum()
    return {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}


def analyze_data_status(df: pd.DataFrame) -> dict[str, int]:
    """Quantify synthetic vs real vs unknown records based on data_status field."""
    if "data_status" not in df.columns or df.empty:
        return {"synthetic_rows": 0, "real_rows": 0, "unknown_rows": len(df)}

    val_counts = df["data_status"].astype(str).str.upper().value_counts().to_dict()
    synthetic = val_counts.get("SYNTHETIC_DEMO", 0) + val_counts.get("SYNTHETIC", 0)
    real = val_counts.get("REAL_DATA", 0) + val_counts.get("REAL", 0)
    unknown = sum(v for k, v in val_counts.items() if k not in {"SYNTHETIC_DEMO", "SYNTHETIC", "REAL_DATA", "REAL"})

    return {
        "synthetic_rows": int(synthetic),
        "real_rows": int(real),
        "unknown_rows": int(unknown),
    }


def validate_single_file(
    df: pd.DataFrame,
    file_key: str,
    all_datasets: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Validate a single dataset file against schema, uniqueness, and referential constraints."""
    expected_cols = SCHEMAS.get(file_key, [])
    missing_cols = validate_schema(df, expected_cols)

    pk_col = PRIMARY_KEYS.get(file_key)
    duplicate_ids = detect_duplicate_ids(df, pk_col) if pk_col else []

    composite_key = COMPOSITE_RELATIONSHIP_KEYS.get(file_key)
    duplicate_rels = detect_duplicate_relationships(df, composite_key) if composite_key else 0

    missing_vals = check_missing_values(df)
    status_counts = analyze_data_status(df)

    # Check foreign keys originating from this file
    invalid_refs: dict[str, int] = {}
    if all_datasets:
        for rule in FOREIGN_KEY_RULES:
            if rule.child_table == file_key and rule.parent_table in all_datasets:
                parent_df = all_datasets[rule.parent_table]
                orphans = validate_foreign_keys(df, parent_df, rule.child_col, rule.parent_col)
                if orphans:
                    invalid_refs[f"{rule.child_col}->{rule.parent_table}.{rule.parent_col}"] = len(orphans)

    total_missing_cells = sum(missing_vals.values())
    is_valid = (len(missing_cols) == 0 and len(duplicate_ids) == 0 and sum(invalid_refs.values()) == 0)

    return {
        "file_key": file_key,
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_columns": missing_cols,
        "missing_values_by_col": missing_vals,
        "total_missing_cells": total_missing_cells,
        "duplicate_ids": duplicate_ids,
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_relationships_count": duplicate_rels,
        "invalid_references": invalid_refs,
        "total_invalid_refs": sum(invalid_refs.values()),
        "synthetic_rows": status_counts["synthetic_rows"],
        "real_rows": status_counts["real_rows"],
        "unknown_rows": status_counts["unknown_rows"],
        "is_valid": is_valid,
    }
