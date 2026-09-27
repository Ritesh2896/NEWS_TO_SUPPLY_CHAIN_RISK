"""Data normalization utilities for BDS-35.

Performs non-destructive normalization:
  - Strips leading/trailing whitespace
  - Normalizes identifier tokens
  - Standardizes text whitespace
  - Strictly preserves data_status and provenance/source fields
  - Does NOT silently alter or repair suspicious data
"""
from __future__ import annotations
import re
from typing import Any
import pandas as pd


def normalize_id(val: Any) -> str:
    """Normalize identifier string by stripping whitespace while preserving exact token content."""
    if val is None or pd.isna(val):
        return ""
    val_str = str(val).strip()
    return val_str


def normalize_text(val: Any) -> str:
    """Normalize text field by stripping leading/trailing whitespace and collapsing internal whitespace."""
    if val is None or pd.isna(val):
        return ""
    text = str(val).strip()
    # Collapse multiple consecutive whitespace characters to a single space
    return re.sub(r"\s+", " ", text)


def normalize_dataframe(
    df: pd.DataFrame,
    id_cols: list[str] | None = None,
    text_cols: list[str] | None = None,
    preserve_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply non-destructive text and ID normalization to a dataframe.

    Does not modify numeric, datetime, or provenance fields destructively.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    # Default provenance columns to preserve exactly as-is
    protected = set(preserve_cols or ["data_status", "source", "source_date", "evidence"])

    # Auto-detect ID columns if not provided
    if id_cols is None:
        id_cols = [c for c in out.columns if c.endswith("_id") or c in {"source", "target", "linked_id"}]

    for col in id_cols:
        if col in out.columns and col not in protected:
            out[col] = out[col].astype(str).map(normalize_id)

    # Apply text normalization to string/object columns
    if text_cols is None:
        text_cols = [
            c for c in out.columns
            if c not in id_cols and c not in protected and (
                pd.api.types.is_string_dtype(out[c]) or out[c].dtype == object or str(out[c].dtype) in {"object", "string", "str"}
            )
        ]

    for col in text_cols:
        if col in out.columns and col not in protected:
            # Only normalize string instances, preserving NaNs/sentinels without silent mutation
            out[col] = out[col].map(lambda x: normalize_text(x) if pd.notna(x) else x)

    return out
