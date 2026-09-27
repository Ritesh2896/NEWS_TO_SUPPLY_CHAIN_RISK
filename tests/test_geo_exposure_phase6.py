"""Tests for Phase 6: Geographic Exposure Engine (Haversine distance, zones, explainability)."""

import math
import pandas as pd
import pytest

from src.risk.geo_exposure import (
    EARTH_RADIUS_KM,
    EXPOSURE_DISCLAIMER,
    ExposureZone,
    ExposureZoneConfig,
    DEFAULT_EXPOSURE_CONFIG,
    SupplierExposureRecord,
    haversine_distance,
    calculate_supplier_event_exposure,
    compute_geographic_exposure,
)
from src.risk.geospatial_exposure import (
    haversine_km,
    exposure_score_from_distance,
    calculate_geographic_exposure,
)


# Known benchmark coordinates: (lat, lon)
COORDINATES = {
    "LONDON": (51.5074, -0.1278),
    "PARIS": (48.8566, 2.3522),
    "NYC": (40.7128, -74.0060),
    "PHILADELPHIA": (39.9526, -75.1652),
    "TOKYO": (35.6762, 139.6503),
    "YOKOHAMA": (35.4437, 139.6380),
    "MUMBAI": (19.0760, 72.8777),
    "PUNE": (18.5204, 73.8567),
}


def test_haversine_distance_known_coordinates():
    """Verify geodesic Haversine distance against standard geodetic benchmarks."""
    # 1. Identical coordinates -> exactly 0.0 km
    lon_lat = COORDINATES["LONDON"]
    assert haversine_distance(lon_lat[0], lon_lat[1], lon_lat[0], lon_lat[1]) == 0.0

    # 2. London to Paris ~ 343 km
    d_london_paris = haversine_distance(
        COORDINATES["LONDON"][0], COORDINATES["LONDON"][1],
        COORDINATES["PARIS"][0], COORDINATES["PARIS"][1],
    )
    assert 340.0 <= d_london_paris <= 346.0

    # 3. New York City to Philadelphia ~ 130 km
    d_nyc_philly = haversine_distance(
        COORDINATES["NYC"][0], COORDINATES["NYC"][1],
        COORDINATES["PHILADELPHIA"][0], COORDINATES["PHILADELPHIA"][1],
    )
    assert 125.0 <= d_nyc_philly <= 135.0

    # 4. Tokyo to Yokohama ~ 26 km
    d_tokyo_yokohama = haversine_distance(
        COORDINATES["TOKYO"][0], COORDINATES["TOKYO"][1],
        COORDINATES["YOKOHAMA"][0], COORDINATES["YOKOHAMA"][1],
    )
    assert 24.0 <= d_tokyo_yokohama <= 28.0

    # 5. Mumbai to Pune ~ 120 km
    d_mumbai_pune = haversine_distance(
        COORDINATES["MUMBAI"][0], COORDINATES["MUMBAI"][1],
        COORDINATES["PUNE"][0], COORDINATES["PUNE"][1],
    )
    assert 115.0 <= d_mumbai_pune <= 125.0


def test_haversine_robust_inputs():
    """Verify string inputs, numeric strings, and out-of-range bounds handling."""
    # String representations of coordinates
    d_str = haversine_distance("51.5074", "-0.1278", "48.8566", "2.3522")
    assert 340.0 <= d_str <= 346.0

    # Missing / None coordinates return inf
    assert haversine_distance(None, 72.8, 19.0, 73.0) == float("inf")
    assert haversine_distance(19.0, 72.8, None, None) == float("inf")
    assert haversine_distance("invalid", 72.8, 19.0, 73.0) == float("inf")

    # Invalid latitude/longitude bounds return inf
    assert haversine_distance(95.0, 0.0, 0.0, 0.0) == float("inf")
    assert haversine_distance(0.0, 190.0, 0.0, 0.0) == float("inf")


def test_default_exposure_zones():
    """Verify standard default exposure zones: 0-25 km, 25-100 km, 100-250 km, 250+ km."""
    cfg = DEFAULT_EXPOSURE_CONFIG

    # 0-25 km zone (Epicenter)
    s1, b1, t1, _ = cfg.get_zone(12.5)
    assert s1 == 1.00
    assert b1 == "0-25 km"
    assert t1 == "CRITICAL_EPICENTER"

    # Boundary at 25.0 km
    s_b25, b_b25, _, _ = cfg.get_zone(25.0)
    assert s_b25 == 1.00
    assert b_b25 == "0-25 km"

    # 25-100 km zone (High Proximity)
    s2, b2, t2, _ = cfg.get_zone(60.0)
    assert s2 == 0.85
    assert b2 == "25-100 km"
    assert t2 == "HIGH_PROXIMITY"

    # 100-250 km zone (Moderate Proximity)
    s3, b3, t3, _ = cfg.get_zone(180.0)
    assert s3 == 0.65
    assert b3 == "100-250 km"
    assert t3 == "MODERATE_PROXIMITY"

    # 250+ km zone (Low Proximity)
    s4, b4, t4, _ = cfg.get_zone(450.0)
    assert s4 == 0.20
    assert b4 == "250+ km"
    assert t4 == "LOW_EXPOSURE"


def test_custom_configurable_zones():
    """Verify custom user-defined exposure zone configurations."""
    custom_cfg = ExposureZoneConfig(zones=[
        ExposureZone(0.0, 10.0, "0-10 km", 1.0, "TIGHT_CORE", "Direct facility premises impact"),
        ExposureZone(10.0, 50.0, "10-50 km", 0.70, "PERIPHERAL", "Metropolitan corridor impact"),
        ExposureZone(50.0, float("inf"), "50+ km", 0.15, "REMOTE", "Negligible impact"),
    ])

    score, band, tier, _ = custom_cfg.get_zone(8.0)
    assert score == 1.0
    assert band == "0-10 km"
    assert tier == "TIGHT_CORE"

    score_med, band_med, _, _ = custom_cfg.get_zone(35.0)
    assert score_med == 0.70
    assert band_med == "10-50 km"

    score_far, band_far, _, _ = custom_cfg.get_zone(120.0)
    assert score_far == 0.15
    assert band_far == "50+ km"


def test_supplier_and_facility_location_resolution():
    """Verify that closest site (HQ or linked physical facility) is selected as critical exposure point."""
    # Disruption event at Yokohama
    e_lat, e_lon = COORDINATES["YOKOHAMA"]

    # Supplier HQ is in London (9,500+ km away)
    supplier_row = {
        "supplier_id": "SUP_GLOBAL_01",
        "supplier_name": "Global Tech Corp",
        "latitude": COORDINATES["LONDON"][0],
        "longitude": COORDINATES["LONDON"][1],
        "city": "London",
    }

    # Linked facility locations: One in Paris, one in Tokyo (~26 km from Yokohama)
    locations_df = pd.DataFrame([
        {
            "location_id": "LOC_PARIS",
            "location_name": "European Logistics Hub",
            "latitude": COORDINATES["PARIS"][0],
            "longitude": COORDINATES["PARIS"][1],
        },
        {
            "location_id": "LOC_TOKYO_PLANT",
            "location_name": "Tokyo Semiconductor Fab",
            "latitude": COORDINATES["TOKYO"][0],
            "longitude": COORDINATES["TOKYO"][1],
        },
    ])

    supplier_locations_df = pd.DataFrame([
        {"supplier_id": "SUP_GLOBAL_01", "location_id": "LOC_PARIS", "facility_type": "Warehouse"},
        {"supplier_id": "SUP_GLOBAL_01", "location_id": "LOC_TOKYO_PLANT", "facility_type": "Factory"},
    ])

    # Calculate exposure
    rec = calculate_supplier_event_exposure(
        event_id="EVT_TYPHOON_01",
        event_lat=e_lat,
        event_lon=e_lon,
        supplier_id="SUP_GLOBAL_01",
        supplier_row=supplier_row,
        locations_df=locations_df,
        supplier_locations_df=supplier_locations_df,
    )

    # The Tokyo facility should be selected (distance ~26 km, not the 9,500 km HQ)
    assert rec.supplier_id == "SUP_GLOBAL_01"
    assert rec.event_id == "EVT_TYPHOON_01"
    assert rec.impacted_location_id == "LOC_TOKYO_PLANT"
    assert rec.impacted_location_type == "Factory"
    assert 24.0 <= rec.distance_km <= 28.0
    assert rec.exposure_band == "25-100 km"
    assert rec.geographic_exposure == 0.85
    assert "Tokyo Semiconductor Fab" in rec.explanation


def test_batch_compute_geographic_exposure_schema():
    """Verify compute_geographic_exposure returns exact required schema."""
    events_df = pd.DataFrame([
        {
            "event_id": "EVT001",
            "location_id": "LOC_MUMBAI",
            "event_type": "Flood",
        }
    ])
    locations_df = pd.DataFrame([
        {
            "location_id": "LOC_MUMBAI",
            "location_name": "Mumbai Port Terminal",
            "latitude": COORDINATES["MUMBAI"][0],
            "longitude": COORDINATES["MUMBAI"][1],
        },
        {
            "location_id": "LOC_PUNE_WH",
            "location_name": "Pune Logistics Facility",
            "latitude": COORDINATES["PUNE"][0],
            "longitude": COORDINATES["PUNE"][1],
        },
    ])
    suppliers_df = pd.DataFrame([
        {
            "supplier_id": "SUP001",
            "supplier_name": "Maharashtra Industrial",
            "latitude": COORDINATES["PUNE"][0],
            "longitude": COORDINATES["PUNE"][1],
        }
    ])
    supplier_locations_df = pd.DataFrame([
        {"supplier_id": "SUP001", "location_id": "LOC_PUNE_WH", "facility_type": "Warehouse"}
    ])

    df = compute_geographic_exposure(
        events_df=events_df,
        suppliers_df=suppliers_df,
        locations_df=locations_df,
        supplier_locations_df=supplier_locations_df,
    )

    # Verify required output columns
    required_cols = [
        "supplier_id",
        "event_id",
        "distance_km",
        "geographic_exposure",
        "exposure_band",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert len(df) == 1
    row = df.iloc[0]
    assert row["supplier_id"] == "SUP001"
    assert row["event_id"] == "EVT001"
    assert 115.0 <= row["distance_km"] <= 125.0
    assert row["geographic_exposure"] == 0.65
    assert row["exposure_band"] == "100-250 km"


def test_academic_disclaimer_and_explainability():
    """Verify the non-claim disclaimer and explainable description are present."""
    rec = calculate_supplier_event_exposure(
        event_id="EVT_DISCLAIMER_TEST",
        event_lat=COORDINATES["LONDON"][0],
        event_lon=COORDINATES["LONDON"][1],
        supplier_id="SUP_TEST",
        supplier_row={
            "supplier_id": "SUP_TEST",
            "latitude": COORDINATES["PARIS"][0],
            "longitude": COORDINATES["PARIS"][1],
        },
    )

    # Must contain the non-claim disclaimer
    assert "Distance alone does not prove operational disruption" in rec.disclaimer
    assert "physical hazard exposure" in rec.disclaimer

    # Explanation must describe physical distance and assigned band
    assert "343" in rec.explanation or "342" in rec.explanation or "344" in rec.explanation
    assert "250+ km" in rec.explanation


def test_backward_compatibility_with_geospatial_exposure():
    """Verify legacy geospatial_exposure functions remain fully operational."""
    # Test haversine_km
    d = haversine_km(COORDINATES["LONDON"][0], COORDINATES["LONDON"][1],
                     COORDINATES["PARIS"][0], COORDINATES["PARIS"][1])
    assert 340.0 <= d <= 346.0

    # Test calculate_geographic_exposure
    score, dist = calculate_geographic_exposure(
        COORDINATES["MUMBAI"][0], COORDINATES["MUMBAI"][1],
        COORDINATES["PUNE"][0], COORDINATES["PUNE"][1]
    )
    assert 0.0 <= score <= 1.0
    assert 115.0 <= dist <= 125.0

    # Test exposure_score_from_distance
    assert exposure_score_from_distance(20.0) == 1.00
    assert exposure_score_from_distance(80.0) == 0.85
    assert exposure_score_from_distance(150.0) == 0.65
