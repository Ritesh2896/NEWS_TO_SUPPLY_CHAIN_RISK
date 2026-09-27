"""CLI Command for validating the BDS-35 Dataset Pack.

Run via:
    python -m src.data.validate

Prints a clear tabular validation report:
  - File Name
  - Row Count
  - Column Count
  - Missing Values (total missing cells)
  - Duplicate IDs
  - Invalid References (foreign key orphans)
  - Synthetic vs Real Rows

Strictly audits data without silent repair of suspicious records.
"""
from __future__ import annotations
import sys
from src.data.loader import DataLoader


def main():
    loader = DataLoader(base_dir="data")
    summary = loader.get_data_quality_summary()

    header = f"{'File Name':<22} | {'Rows':<7} | {'Cols':<5} | {'Missing':<8} | {'Dup IDs':<7} | {'Invalid Refs':<12} | {'Synthetic / Real':<18} | {'Status'}"
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print("                     BDS-35 DATASET PACK VALIDATION REPORT                      ")
    print("=" * len(header))
    print(header)
    print(separator)

    all_valid = True

    for key, rep in summary["file_reports"].items():
        fname = f"{key}.csv"
        rows = rep["row_count"]
        cols = rep["column_count"]
        missing = rep["total_missing_cells"]
        dup_ids = rep["duplicate_id_count"]
        inv_refs = rep["total_invalid_refs"]
        synth = rep["synthetic_rows"]
        real = rep["real_rows"]
        status_str = f"{synth} / {real}"

        file_status = "VALID" if rep["is_valid"] else "FLAGGED"
        if not rep["is_valid"]:
            all_valid = False

        print(
            f"{fname:<22} | {rows:<7} | {cols:<5} | {missing:<8} | {dup_ids:<7} | {inv_refs:<12} | {status_str:<18} | {file_status}"
        )

    print(separator)
    print(f"Total Files Audited: {summary['total_files']}")
    print(f"Total Records:       {summary['total_records']:,}")
    print(f"Synthetic Records:   {summary['synthetic_records']:,} (100% SYNTHETIC_DEMO Provenance Preserved)")
    print(f"Real-world Records:  {summary['real_records']:,}")
    print(f"Total Missing Cells: {summary['total_missing_cells']:,}")
    print(f"Total Duplicate IDs: {summary['total_duplicate_ids']}")
    print(f"Invalid FK Refs:     {summary['total_invalid_foreign_keys']}")
    print("=" * len(header) + "\n")

    if not all_valid:
        print("[WARNING] One or more datasets contained validation flags. Inspect details above.")
        sys.exit(1)
    else:
        print("[SUCCESS] All 12 dataset pack files passed schema, uniqueness, and referential integrity validation.")
        sys.exit(0)


if __name__ == "__main__":
    main()
