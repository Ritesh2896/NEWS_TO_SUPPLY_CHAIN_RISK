"""Geospatial exposure calculation for BDS-35.

Computes geodesic Haversine distance and calibrated distance-decay exposure scores
between disruption event epicenters and supplier headquarters/facilities.
"""
from __future__ import annotations
from math import exp
from typing import Any
import pandas as pd

from src.risk.geo_exposure import (
    haversine_distance,
    DEFAULT_EXPOSURE_CONFIG,
    calculate_supplier_event_exposure,
    compute_geographic_exposure,
)


def haversine_km(lat1: float | str | None, lon1: float | str | None,
                 lat2: float | str | None, lon2: float | str | None) -> float:
    """Calculate the great-circle distance between two points on the Earth in kilometers."""
    d = haversine_distance(lat1, lon1, lat2, lon2)
    return 9999.0 if d == float("inf") else d


def exposure_score_from_distance(distance_km: float, decay_type: str = "tiered") -> float:
    """Map distance in kilometers to an exposure score in [0.1, 1.0].

    Tiered calibration:
      <= 25 km   : 1.00 (Immediate epicenter / local facility closure)
      <= 100 km  : 0.85 (High proximity shock / regional shutdown)
      <= 250 km  : 0.65 (Moderate proximity / transport corridor disruption)
      <= 500 km  : 0.45 (Regional delay / port feeder congestion)
      <= 1000 km : 0.25 (Sub-continental indirect supply delay)
      > 1000 km  : 0.10 (Negligible direct exposure)
    """
    if distance_km < 0:
        return 0.10

    if decay_type == "exponential":
        # Continuous exponential decay with half-life of ~200 km
        return round(float(max(0.05, min(1.0, exp(-distance_km / 250.0)))), 4)

    # Standard tiered decay
    if distance_km <= 25.0:
        return 1.00
    if distance_km <= 100.0:
        return 0.85
    if distance_km <= 250.0:
        return 0.65
    if distance_km <= 500.0:
        return 0.45
    if distance_km <= 1000.0:
        return 0.25
    return 0.10


def calculate_geographic_exposure(
    event_lat: float | str | None,
    event_lon: float | str | None,
    supplier_lat: float | str | None,
    supplier_lon: float | str | None,
) -> tuple[float, float]:
    """Calculate geographic exposure score and distance in km between event and supplier coordinates."""
    dist = haversine_km(event_lat, event_lon, supplier_lat, supplier_lon)
    score = exposure_score_from_distance(dist)
    return score, dist


def calculate_supplier_exposure(
    event_lat: float | str | None,
    event_lon: float | str | None,
    supplier_row: pd.Series | dict[str, Any] | None,
    locations_df: pd.DataFrame | None = None,
    supplier_locations_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Calculate exposure across primary supplier location and all associated facilities.

    Picks the facility closest to the event to represent worst-case facility shock.
    """
    if event_lat is None or event_lon is None or supplier_row is None:
        return {"exposure_score": 0.20, "distance_km": None, "closest_location": "Unknown"}

    best_distance = 99999.0
    best_loc_name = "Primary Supplier Office"

    # 1. Check primary supplier coordinates
    s_lat = supplier_row.get("latitude") if isinstance(supplier_row, dict) else supplier_row.get("latitude")
    s_lon = supplier_row.get("longitude") if isinstance(supplier_row, dict) else supplier_row.get("longitude")
    if pd.notna(s_lat) and pd.notna(s_lon):
        d0 = haversine_km(event_lat, event_lon, s_lat, s_lon)
        if d0 < best_distance:
            best_distance = d0
            city = supplier_row.get("city", "")
            best_loc_name = f"HQ ({city})" if city else "HQ"

    # 2. Check linked facilities from supplier_locations.csv
    sid = str(supplier_row.get("supplier_id", ""))
    if (
        sid
        and supplier_locations_df is not None
        and not supplier_locations_df.empty
        and locations_df is not None
        and not locations_df.empty
    ):
        sub_sl = supplier_locations_df[supplier_locations_df["supplier_id"].astype(str) == sid]
        for _, sl_row in sub_sl.iterrows():
            lid = str(sl_row.get("location_id", ""))
            loc_matches = locations_df[locations_df["location_id"].astype(str) == lid]
            if not loc_matches.empty:
                loc_r = loc_matches.iloc[0]
                fac_lat, fac_lon = loc_r.get("latitude"), loc_r.get("longitude")
                if pd.notna(fac_lat) and pd.notna(fac_lon):
                    d_fac = haversine_km(event_lat, event_lon, fac_lat, fac_lon)
                    if d_fac < best_distance:
                        best_distance = d_fac
                        best_loc_name = str(loc_r.get("location_name", f"Facility {lid}"))

    if best_distance >= 9999.0:
        return {"exposure_score": 0.20, "distance_km": None, "closest_location": "Default"}

    exposure = exposure_score_from_distance(best_distance)
    return {
        "exposure_score": exposure,
        "distance_km": round(best_distance, 2),
        "closest_location": best_loc_name,
    }
