"""Geographic Exposure Engine for BDS-35 (Phase 6).

Calculates geographic exposure between disruption events and supply-chain locations
(supplier headquarters and linked physical facilities).

Features:
- Geodesic Haversine distance calculation in kilometers.
- Configurable exposure zones (e.g. 0-25 km, 25-100 km, 100-250 km, 250+ km).
- Multi-location resolution: evaluates primary supplier HQ and all associated facilities
  from supplier_locations.csv / locations.csv, identifying the critical exposure point.
- Deterministic exposure score mapping [0.0, 1.0].
- Explicit academic non-claim disclaimer: proximity indicates physical hazard exposure,
  not guaranteed operational disruption.
- Explainable output including human-readable rationales and provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, radians, sin, sqrt
from typing import Any, Sequence
import numpy as np
import pandas as pd

# Earth's volumetric mean radius in kilometers (WGS-84 / IUGG standard)
EARTH_RADIUS_KM: float = 6371.0

# Academic disclaimer mandated by methodology
EXPOSURE_DISCLAIMER: str = (
    "Geographic proximity indicates physical hazard exposure. "
    "Distance alone does not prove operational disruption, which depends on "
    "facility hardening, utility redundancy, emergency inventory, and supplier agility."
)


@dataclass(frozen=True)
class ExposureZone:
    """Defines a geographic exposure zone with distance boundaries and score."""
    min_km: float
    max_km: float
    band_name: str
    exposure_score: float
    tier: str
    description: str


@dataclass
class ExposureZoneConfig:
    """Configurable collection of geographic exposure zones."""
    zones: list[ExposureZone] = field(default_factory=lambda: [
        ExposureZone(
            min_km=0.0,
            max_km=25.0,
            band_name="0-25 km",
            exposure_score=1.00,
            tier="CRITICAL_EPICENTER",
            description="Immediate event epicenter; high acute physical hazard exposure",
        ),
        ExposureZone(
            min_km=25.0,
            max_km=100.0,
            band_name="25-100 km",
            exposure_score=0.85,
            tier="HIGH_PROXIMITY",
            description="High proximity shock zone; substantial regional hazard impact",
        ),
        ExposureZone(
            min_km=100.0,
            max_km=250.0,
            band_name="100-250 km",
            exposure_score=0.65,
            tier="MODERATE_PROXIMITY",
            description="Moderate proximity; secondary logistics or corridor disruption exposure",
        ),
        ExposureZone(
            min_km=250.0,
            max_km=float("inf"),
            band_name="250+ km",
            exposure_score=0.20,
            tier="LOW_EXPOSURE",
            description="Extended distance; low direct physical proximity exposure",
        ),
    ])

    def get_zone(self, distance_km: float | None) -> tuple[float, str, str, str]:
        """Map a distance in km to (exposure_score, band_name, tier, description).

        If distance is None or infinite, returns lowest exposure default.
        """
        if distance_km is None or np.isnan(distance_km) or distance_km < 0:
            return 0.10, "UNKNOWN", "INDETERMINATE", "Distance could not be calculated (missing coordinates)"

        for zone in self.zones:
            if zone.min_km <= distance_km <= zone.max_km:
                return zone.exposure_score, zone.band_name, zone.tier, zone.description

        # Fallback for values beyond configured upper bound
        last_zone = self.zones[-1]
        return last_zone.exposure_score, last_zone.band_name, last_zone.tier, last_zone.description


DEFAULT_EXPOSURE_CONFIG = ExposureZoneConfig()


def haversine_distance(
    lat1: float | str | None,
    lon1: float | str | None,
    lat2: float | str | None,
    lon2: float | str | None,
) -> float:
    """Calculate the great-circle distance between two geographic coordinates in km.

    Uses the Haversine formula on a spherical Earth with radius R = 6371.0 km.

    Args:
        lat1: Latitude of point 1 in degrees (-90.0 to 90.0).
        lon1: Longitude of point 1 in degrees (-180.0 to 180.0).
        lat2: Latitude of point 2 in degrees (-90.0 to 90.0).
        lon2: Longitude of point 2 in degrees (-180.0 to 180.0).

    Returns:
        Great-circle distance in kilometers rounded to 4 decimal places.
        Returns float('inf') if coordinates are missing, invalid, or out of range.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return float("inf")

    try:
        f_lat1, f_lon1 = float(lat1), float(lon1)
        f_lat2, f_lon2 = float(lat2), float(lon2)
    except (ValueError, TypeError):
        return float("inf")

    if np.isnan(f_lat1) or np.isnan(f_lon1) or np.isnan(f_lat2) or np.isnan(f_lon2):
        return float("inf")

    # Validate physical geographic bounds
    if not (-90.0 <= f_lat1 <= 90.0 and -90.0 <= f_lat2 <= 90.0):
        return float("inf")
    if not (-180.0 <= f_lon1 <= 180.0 and -180.0 <= f_lon2 <= 180.0):
        return float("inf")

    # Exact coordinate match optimization
    if f_lat1 == f_lat2 and f_lon1 == f_lon2:
        return 0.0

    phi1, phi2 = radians(f_lat1), radians(f_lat2)
    dphi = radians(f_lat2 - f_lat1)
    dlambda = radians(f_lon2 - f_lon1)

    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    # Guard against float rounding exceeding 1.0
    a = min(1.0, max(0.0, a))
    c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))

    return round(EARTH_RADIUS_KM * c, 4)


@dataclass
class SupplierExposureRecord:
    """Structured representation of a single supplier-event geographic exposure result."""
    supplier_id: str
    event_id: str
    distance_km: float | None
    geographic_exposure: float
    exposure_band: str
    tier: str
    impacted_location_id: str
    impacted_location_name: str
    impacted_location_type: str
    explanation: str
    disclaimer: str = EXPOSURE_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary matching the required schema."""
        return {
            "supplier_id": self.supplier_id,
            "event_id": self.event_id,
            "distance_km": self.distance_km,
            "geographic_exposure": self.geographic_exposure,
            "exposure_band": self.exposure_band,
            "impacted_location_id": self.impacted_location_id,
            "impacted_location_name": self.impacted_location_name,
            "impacted_location_type": self.impacted_location_type,
            "explanation": self.explanation,
            "disclaimer": self.disclaimer,
        }


def calculate_supplier_event_exposure(
    event_id: str,
    event_lat: float | str | None,
    event_lon: float | str | None,
    supplier_id: str,
    supplier_row: pd.Series | dict[str, Any] | None = None,
    locations_df: pd.DataFrame | None = None,
    supplier_locations_df: pd.DataFrame | None = None,
    config: ExposureZoneConfig | None = None,
) -> SupplierExposureRecord:
    """Calculate geographic exposure for a supplier considering HQ and all physical facilities.

    Picks the closest facility/HQ location to the event epicenter (worst-case hazard proximity).

    Args:
        event_id: Unique event identifier.
        event_lat: Latitude of the disruption event.
        event_lon: Longitude of the disruption event.
        supplier_id: Unique supplier identifier.
        supplier_row: Supplier metadata (e.g. latitude, longitude, city, supplier_name).
        locations_df: Master locations DataFrame.
        supplier_locations_df: Relationship table connecting suppliers to facilities.
        config: ExposureZoneConfig defining bands and scores.

    Returns:
        SupplierExposureRecord with exact distance, score, band, and explanation.
    """
    cfg = config or DEFAULT_EXPOSURE_CONFIG

    best_dist = float("inf")
    best_loc_id = "UNKNOWN"
    best_loc_name = "Primary Supplier Office"
    best_loc_type = "HQ"

    # 1. Check primary supplier HQ coordinates
    if supplier_row is not None:
        s_dict = supplier_row.to_dict() if isinstance(supplier_row, pd.Series) else supplier_row
        s_lat = s_dict.get("latitude")
        s_lon = s_dict.get("longitude")
        s_city = s_dict.get("city", "")
        if s_lat is not None and s_lon is not None:
            d_hq = haversine_distance(event_lat, event_lon, s_lat, s_lon)
            if d_hq < best_dist:
                best_dist = d_hq
                best_loc_id = str(s_dict.get("location_id", "HQ"))
                best_loc_name = f"HQ ({s_city})" if s_city else "Supplier Headquarters"
                best_loc_type = "Headquarters"

    # 2. Check all physical facilities linked to this supplier
    if (
        supplier_locations_df is not None
        and not supplier_locations_df.empty
        and locations_df is not None
        and not locations_df.empty
    ):
        sid_str = str(supplier_id).strip()
        # Find facility relations
        sl_sub = supplier_locations_df[supplier_locations_df["supplier_id"].astype(str) == sid_str]
        for _, sl_row in sl_sub.iterrows():
            lid = str(sl_row.get("location_id", "")).strip()
            fac_type = str(sl_row.get("facility_type", "Facility"))
            loc_match = locations_df[locations_df["location_id"].astype(str) == lid]
            if not loc_match.empty:
                loc_r = loc_match.iloc[0]
                fac_lat = loc_r.get("latitude")
                fac_lon = loc_r.get("longitude")
                d_fac = haversine_distance(event_lat, event_lon, fac_lat, fac_lon)
                if d_fac < best_dist:
                    best_dist = d_fac
                    best_loc_id = lid
                    best_loc_name = str(loc_r.get("location_name", f"Facility {lid}"))
                    best_loc_type = fac_type

    # Format result
    if best_dist == float("inf"):
        score, band, tier, desc = cfg.get_zone(None)
        explanation = (
            f"Supplier {supplier_id} could not be geolocated relative to Event {event_id} "
            "due to missing or invalid coordinates. Assigned baseline exposure."
        )
        return SupplierExposureRecord(
            supplier_id=supplier_id,
            event_id=event_id,
            distance_km=None,
            geographic_exposure=score,
            exposure_band=band,
            tier=tier,
            impacted_location_id=best_loc_id,
            impacted_location_name=best_loc_name,
            impacted_location_type=best_loc_type,
            explanation=explanation,
        )

    score, band, tier, desc = cfg.get_zone(best_dist)
    explanation = (
        f"Supplier {supplier_id} closest site '{best_loc_name}' ({best_loc_type}) "
        f"is {best_dist:.2f} km from Event {event_id} epicenter. "
        f"Assigned exposure band '{band}' (score: {score:.2f}, tier: {tier})."
    )

    return SupplierExposureRecord(
        supplier_id=supplier_id,
        event_id=event_id,
        distance_km=round(best_dist, 2),
        geographic_exposure=score,
        exposure_band=band,
        tier=tier,
        impacted_location_id=best_loc_id,
        impacted_location_name=best_loc_name,
        impacted_location_type=best_loc_type,
        explanation=explanation,
    )


def compute_geographic_exposure(
    events_df: pd.DataFrame,
    suppliers_df: pd.DataFrame,
    locations_df: pd.DataFrame,
    supplier_locations_df: pd.DataFrame,
    config: ExposureZoneConfig | None = None,
    max_distance_km: float | None = None,
) -> pd.DataFrame:
    """Compute pairwise geographic exposure between disruption events and suppliers.

    Resolves event epicenter coordinates either from events_df directly or via
    foreign-key location_id mapped to locations_df.

    Args:
        events_df: Disruption events DataFrame (event_id, location_id, [latitude, longitude]).
        suppliers_df: Master suppliers DataFrame (supplier_id, latitude, longitude, etc.).
        locations_df: Master locations DataFrame (location_id, latitude, longitude, etc.).
        supplier_locations_df: Master supplier_locations DataFrame.
        config: Custom exposure zone configuration.
        max_distance_km: Optional filter to exclude distances strictly greater than this value.

    Returns:
        DataFrame with columns:
          supplier_id, event_id, distance_km, geographic_exposure, exposure_band,
          impacted_location_id, impacted_location_name, impacted_location_type, explanation, disclaimer
    """
    if events_df.empty or suppliers_df.empty:
        return pd.DataFrame(columns=[
            "supplier_id", "event_id", "distance_km", "geographic_exposure", "exposure_band",
            "impacted_location_id", "impacted_location_name", "impacted_location_type",
            "explanation", "disclaimer"
        ])

    # Pre-index locations for quick coordinate lookup
    loc_coords: dict[str, tuple[float, float, str]] = {}
    if not locations_df.empty and "location_id" in locations_df.columns:
        for _, row in locations_df.iterrows():
            lid = str(row["location_id"]).strip()
            lat = row.get("latitude")
            lon = row.get("longitude")
            name = str(row.get("location_name", lid))
            if pd.notna(lat) and pd.notna(lon):
                try:
                    loc_coords[lid] = (float(lat), float(lon), name)
                except (ValueError, TypeError):
                    continue

    records: list[dict[str, Any]] = []

    for _, evt_row in events_df.iterrows():
        eid = str(evt_row["event_id"]).strip()
        e_lat = evt_row.get("latitude")
        e_lon = evt_row.get("longitude")

        # Resolve event coordinates from location_id if not present directly
        if (pd.isna(e_lat) or pd.isna(e_lon)) and "location_id" in evt_row:
            lid = str(evt_row.get("location_id", "")).strip()
            if lid in loc_coords:
                e_lat, e_lon, _ = loc_coords[lid]

        for _, sup_row in suppliers_df.iterrows():
            sid = str(sup_row["supplier_id"]).strip()
            rec = calculate_supplier_event_exposure(
                event_id=eid,
                event_lat=e_lat,
                event_lon=e_lon,
                supplier_id=sid,
                supplier_row=sup_row,
                locations_df=locations_df,
                supplier_locations_df=supplier_locations_df,
                config=config,
            )

            # Apply distance filter if specified
            if max_distance_km is not None and rec.distance_km is not None:
                if rec.distance_km > max_distance_km:
                    continue

            records.append(rec.to_dict())

    return pd.DataFrame(records)
