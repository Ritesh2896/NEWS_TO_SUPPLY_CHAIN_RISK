"""Unit and integration tests for the BDS-35 Data Loading & Validation Layer."""
import os
import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.normalization import normalize_dataframe, normalize_id, normalize_text
from src.data.schemas import SCHEMAS, STANDARD_FILE_PATHS
from src.data.validators import (
    analyze_data_status,
    check_missing_values,
    detect_duplicate_ids,
    detect_duplicate_relationships,
    validate_foreign_keys,
    validate_schema,
    validate_single_file,
)


@pytest.fixture
def loader():
    return DataLoader(base_dir="data")


def test_load_all_12_dataset_files(loader):
    """Verify that all 12 dataset pack files load successfully with non-zero rows."""
    pack = loader.load_dataset_pack(normalize=True)
    assert len(pack) == 12

    expected_min_rows = {
        "suppliers": 1000,
        "products": 1000,
        "locations": 1000,
        "supplier_products": 3000,
        "supplier_locations": 2000,
        "news": 1000,
        "events": 1000,
        "news_entities": 3000,
        "entity_links": 2500,
        "risk_labels": 1000,
        "nodes": 5000,
        "edges": 12000,
    }

    for key, expected_count in expected_min_rows.items():
        assert key in pack
        assert len(pack[key]) >= expected_count
        assert "data_status" in pack[key].columns
        # Verify provenance is preserved
        assert (pack[key]["data_status"] == "SYNTHETIC_DEMO").all()


def test_validate_schema_compliance(loader):
    """Verify schema column validator passes on genuine tables and detects missing columns."""
    pack = loader.load_dataset_pack(normalize=False)

    for key, expected_cols in SCHEMAS.items():
        df = pack[key]
        missing = validate_schema(df, expected_cols)
        assert len(missing) == 0, f"Schema mismatch in {key}: missing {missing}"

    # Verify detection of missing columns
    dummy = pd.DataFrame([{"supplier_id": "SUP001"}])
    missing = validate_schema(dummy, SCHEMAS["suppliers"])
    assert len(missing) > 0
    assert "supplier_name" in missing


def test_normalization_behavior():
    """Verify non-destructive normalization on IDs and text fields."""
    assert normalize_id("  SUP00001 \t\n") == "SUP00001"
    assert normalize_id(None) == ""

    assert normalize_text("  Industrial   Chemicals  Supply   Ltd  ") == "Industrial Chemicals Supply Ltd"
    assert normalize_text(None) == ""

    df = pd.DataFrame([
        {
            "supplier_id": "  SUP001  ",
            "supplier_name": "  Apparel   Distributor  ",
            "source": "  PROJECT_DEMO  ",
            "data_status": "SYNTHETIC_DEMO",
        }
    ])
    norm_df = normalize_dataframe(df)
    assert norm_df["supplier_id"].iloc[0] == "SUP001"
    assert norm_df["supplier_name"].iloc[0] == "Apparel Distributor"
    assert norm_df["data_status"].iloc[0] == "SYNTHETIC_DEMO"


def test_detect_duplicate_ids():
    """Verify detection of duplicate primary keys without silent deletion."""
    df_clean = pd.DataFrame([{"supplier_id": "SUP1"}, {"supplier_id": "SUP2"}])
    assert detect_duplicate_ids(df_clean, "supplier_id") == []

    df_dup = pd.DataFrame([{"supplier_id": "SUP1"}, {"supplier_id": "SUP1"}, {"supplier_id": "SUP2"}])
    dups = detect_duplicate_ids(df_dup, "supplier_id")
    assert dups == ["SUP1"]


def test_detect_duplicate_relationships():
    """Verify detection of duplicate tuples on composite keys."""
    df_clean = pd.DataFrame([{"s": "S1", "p": "P1"}, {"s": "S1", "p": "P2"}])
    assert detect_duplicate_relationships(df_clean, ["s", "p"]) == 0

    df_dup = pd.DataFrame([{"s": "S1", "p": "P1"}, {"s": "S1", "p": "P1"}])
    assert detect_duplicate_relationships(df_dup, ["s", "p"]) == 1


def test_foreign_key_validator():
    """Verify referential integrity checking between parent and child tables."""
    parent = pd.DataFrame([{"supplier_id": "SUP1"}, {"supplier_id": "SUP2"}])
    child_valid = pd.DataFrame([{"supplier_id": "SUP1", "product_id": "P1"}])
    child_invalid = pd.DataFrame([{"supplier_id": "SUP_NONEXISTENT", "product_id": "P1"}])

    assert validate_foreign_keys(child_valid, parent, "supplier_id", "supplier_id") == []
    orphans = validate_foreign_keys(child_invalid, parent, "supplier_id", "supplier_id")
    assert orphans == ["SUP_NONEXISTENT"]


def test_missing_values_and_status_counts():
    """Verify detection of missing values and categorization of data status."""
    df = pd.DataFrame([
        {"id": "1", "val": None, "data_status": "SYNTHETIC_DEMO"},
        {"id": "2", "val": "abc", "data_status": "REAL_DATA"},
    ])
    missing = check_missing_values(df)
    assert missing == {"val": 1}

    status = analyze_data_status(df)
    assert status["synthetic_rows"] == 1
    assert status["real_rows"] == 1


def test_data_quality_summary(loader):
    """Verify end-to-end data quality summary on the dataset pack."""
    summary = loader.get_data_quality_summary()
    assert summary["total_files"] == 12
    assert summary["total_records"] == 33500
    assert summary["synthetic_records"] == 33500
    assert summary["total_duplicate_ids"] == 0
    assert summary["total_invalid_foreign_keys"] == 0
    assert summary["all_files_valid"] is True


def test_validate_cli_execution():
    """Verify the CLI entrypoint executes cleanly without raising an unhandled exception."""
    from src.data.validate import main
    # main() calls sys.exit(0) on success
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
