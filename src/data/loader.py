"""Robust data-loading layer for the BDS-35 Dataset Pack.

Loads, validates, and normalizes all 12 dataset pack files:
  - master/: suppliers.csv, products.csv, locations.csv, supplier_products.csv, supplier_locations.csv
  - processed/: news.csv, events.csv, news_entities.csv, entity_links.csv, risk_labels.csv
  - graph/: nodes.csv, edges.csv

Features:
  - Non-destructive normalization
  - Schema verification
  - Referential integrity checking
  - Strict preservation of data_status and provenance fields
  - Data-quality summary generation
"""
from __future__ import annotations
import os
from typing import Any
import pandas as pd

from src.data.normalization import normalize_dataframe
from src.data.schemas import STANDARD_FILE_PATHS
from src.data.validators import validate_single_file


class DataLoader:
    """Robust loader and validator for the BDS-35 dataset files."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir

    def resolve_path(self, rel_path: str) -> str:
        """Resolve path relative to root workspace."""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.abspath(rel_path)

    def load_single(
        self,
        file_key: str,
        rel_path: str | None = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Load a single CSV dataset with schema verification and non-destructive normalization."""
        if rel_path is None:
            rel_path = STANDARD_FILE_PATHS.get(file_key, f"{self.base_dir}/{file_key}.csv")

        full_path = self.resolve_path(rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Dataset file not found: {full_path}")

        # Load raw CSV without altering types or repairing suspicious data
        df = pd.read_csv(full_path, dtype=object)

        # Convert numeric columns where appropriate (preserving NaNs)
        for col in df.columns:
            if col in {"latitude", "longitude", "weight", "share_percent", "confidence",
                        "match_score", "lead_time_days", "event_severity", "geographic_exposure",
                        "supplier_criticality", "dependency_strength", "deterministic_risk"}:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if normalize:
            df = normalize_dataframe(df)

        return df

    def load_dataset_pack(
        self,
        normalize: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Load all 12 dataset pack CSVs into a dictionary."""
        pack: dict[str, pd.DataFrame] = {}
        for key, rel_path in STANDARD_FILE_PATHS.items():
            full_path = self.resolve_path(rel_path)
            if os.path.exists(full_path):
                pack[key] = self.load_single(key, rel_path, normalize=normalize)
            else:
                pack[key] = pd.DataFrame()
        return pack

    def get_data_quality_summary(
        self,
        datasets: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, Any]:
        """Generate a data-quality summary covering all loaded files."""
        if datasets is None:
            datasets = self.load_dataset_pack(normalize=True)

        reports: dict[str, Any] = {}
        total_rows = 0
        total_synthetic = 0
        total_real = 0
        total_invalid_fk = 0
        total_missing = 0
        total_duplicates = 0

        for key, df in datasets.items():
            rep = validate_single_file(df, key, all_datasets=datasets)
            reports[key] = rep
            total_rows += rep["row_count"]
            total_synthetic += rep["synthetic_rows"]
            total_real += rep["real_rows"]
            total_invalid_fk += rep["total_invalid_refs"]
            total_missing += rep["total_missing_cells"]
            total_duplicates += rep["duplicate_id_count"]

        return {
            "total_files": len(datasets),
            "total_records": total_rows,
            "synthetic_records": total_synthetic,
            "real_records": total_real,
            "total_missing_cells": total_missing,
            "total_duplicate_ids": total_duplicates,
            "total_invalid_foreign_keys": total_invalid_fk,
            "all_files_valid": all(r["is_valid"] for r in reports.values()),
            "file_reports": reports,
        }
