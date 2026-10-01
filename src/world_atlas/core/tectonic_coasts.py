"""Tectonic and relief controlled coastal transgression.

The final shoreline is a contour through a continuous crustal surface.  This
module does not add an independent noise field to that contour.  Instead it
selectively drowns low relative relief beside existing water while preserving
positive local relief as headlands.  The amount of transgression is modulated
by the local plate-velocity strain and nearby plate-boundary setting.

It is deliberately a small landscape-evolution approximation, rather than a
calibrated coastal model.  Its contracts are useful to the generator:

* no source of randomness -- replay is entirely inherited from the plate and
  terrain fields;
* every newly wet cell is connected to a pre-existing water body;
* no cell outside the explicitly measured shore band is modified;
* a given relief field reacts more strongly at an active margin than at a
  quiet intraplate shore.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from scipy import ndimage


_PLANET_RADIUS_KM = 6400.0


def _periodic_gaussian(values: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Blur longitude periodically while retaining physical pole boundaries."""

    source = np.asarray(values, dtype=np.float64)
    if sigma_cells <= 0.0:
        return source.copy()
    radius = max(1, min(source.shape[1], int(math.ceil(3.0 * sigma_cells))))
    padded = np.concatenate(
        (source[:, -radius:], source, source[:, :radius]),
        axis=1,
    )
    return ndimage.gaussian_filter(
        padded,
        sigma=(sigma_cells, sigma_cells),
        mode="nearest",
    )[:, radius:-radius]


def _periodic_water_connection(seed_water: np.ndarray, allowed_water: np.ndarray) -> np.ndarray:
    """Flood through an allowed water mask, including the longitude seam."""

    seed = np.asarray(seed_water, dtype=bool)
    allowed = np.asarray(allowed_water, dtype=bool)
    if seed.shape != allowed.shape:
        raise ValueError("water masks must share one shape")
    padded_seed = np.concatenate((seed[:, -1:], seed, seed[:, :1]), axis=1)
    padded_allowed = np.concatenate((allowed[:, -1:], allowed, allowed[:, :1]), axis=1)
    connected = ndimage.binary_propagation(
        padded_seed,
        mask=padded_allowed,
        structure=ndimage.generate_binary_structure(2, 1),
    )
    return np.asarray(connected[:, 1:-1], dtype=bool)


def _kinematic_activity(
    velocity_east_cm_per_year: np.ndarray,
    velocity_north_cm_per_year: np.ndarray,
    latitude_degrees: np.ndarray,
    shore_band: np.ndarray,
) -> np.ndarray:
    """Return a bounded local strain proxy from the plate velocity field."""

    east = np.asarray(velocity_east_cm_per_year, dtype=np.float64)
    north = np.asarray(velocity_north_cm_per_year, dtype=np.float64)
    height, width = east.shape
    row_km = math.pi * _PLANET_RADIUS_KM / height
    cosine = np.maximum(np.cos(np.radians(latitude_degrees)), 0.12)[:, None]
    column_km = math.tau * _PLANET_RADIUS_KM / width * cosine

    east_x = (np.roll(east, -1, axis=1) - np.roll(east, 1, axis=1)) / (2.0 * column_km)
    north_x = (np.roll(north, -1, axis=1) - np.roll(north, 1, axis=1)) / (2.0 * column_km)
    east_y = np.gradient(east, row_km, axis=0)
    north_y = np.gradient(north, row_km, axis=0)
    dilation = east_x + north_y
    shear = np.hypot(east_x - north_y, east_y + north_x)
    strain = np.hypot(dilation, shear)
    samples = strain[np.asarray(shore_band, dtype=bool)]
    if not len(samples):
        return np.zeros(east.shape, dtype=np.float32)
    scale = float(np.quantile(samples, 0.94))
    if not math.isfinite(scale) or scale <= 1.0e-14:
        return np.zeros(east.shape, dtype=np.float32)
    return np.clip(strain / scale, 0.0, 1.0).astype(np.float32)


def _margin_signal(mask: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Spread a one-cell plate-boundary trace into a normalized margin band."""

    spread = _periodic_gaussian(np.asarray(mask, dtype=np.float32), sigma_cells)
    maximum = float(np.max(spread, initial=0.0))
    if maximum <= 1.0e-12:
        return np.zeros(spread.shape, dtype=np.float32)
    return np.clip(spread / maximum, 0.0, 1.0).astype(np.float32)


def selective_tectonic_transgression(
    relative_height_m: np.ndarray,
    shore_distance_km: np.ndarray,
    boundary_class: np.ndarray,
    velocity_east_cm_per_year: np.ndarray,
    velocity_north_cm_per_year: np.ndarray,
    latitude_degrees: np.ndarray,
    low_relief_support: np.ndarray,
    coastline_detail: float,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Drown coastal lowlands from existing relief and plate kinematics.

    ``relative_height_m`` is measured from the current provisional sea level;
    it is not re-levelled here.  Positive local residuals are resistant
    headlands.  Negative local residuals are valleys or weak lowlands and can
    be transgressed when they sit in a finite shore band.  Plate boundaries
    and velocity strain only alter the response rate -- they do not prescribe
    a coastline shape.
    """

    relative = np.asarray(relative_height_m, dtype=np.float64)
    distance = np.asarray(shore_distance_km, dtype=np.float64)
    boundaries = np.asarray(boundary_class, dtype=np.int8)
    east = np.asarray(velocity_east_cm_per_year, dtype=np.float64)
    north = np.asarray(velocity_north_cm_per_year, dtype=np.float64)
    quiet = np.asarray(low_relief_support, dtype=np.float64)
    latitude = np.asarray(latitude_degrees, dtype=np.float64)
    if relative.ndim != 2:
        raise ValueError("coastal relative height must be two-dimensional")
    if not 0.0 <= float(coastline_detail) <= 1.0:
        raise ValueError("coastline_detail must be between zero and one")
    if any(value.shape != relative.shape for value in (distance, boundaries, east, north, quiet)):
        raise ValueError("coastal fields must share one shape")
    if latitude.shape != (relative.shape[0],):
        raise ValueError("one latitude is required per coastal row")
    if not np.isfinite(relative).all() or not np.isfinite(distance).all():
        raise ValueError("coastal fields must be finite")

    height, width = relative.shape
    land = relative > 0.0
    water = ~land
    row_km = math.pi * _PLANET_RADIUS_KM / height
    shore_width_km = 118.0 + 142.0 * float(coastline_detail)
    shore_band = distance <= 3.0 * shore_width_km
    support = np.where(
        shore_band,
        np.exp(-0.5 * np.square(distance / shore_width_km)),
        0.0,
    )

    # The relief comparison stays in physical scale as resolution changes.
    # It makes broad continental ramps neutral while retaining valleys and
    # block-bounded lowlands from the upstream tectonic surface.
    relief_sigma = max(1.0, 135.0 / row_km)
    # Do not blend an ocean basin into the land reference at a shore.  That
    # makes every ordinary coast read as a positive cliff and inverts the
    # intended valley/headland selection.  A normalized land-only blur is a
    # local bedrock/lowland comparison instead.
    land_weight = _periodic_gaussian(land.astype(np.float64), relief_sigma)
    regional_surface = _periodic_gaussian(
        relative * land.astype(np.float64), relief_sigma
    ) / np.maximum(land_weight, 1.0e-6)
    regional_surface = np.where(land_weight > 1.0e-4, regional_surface, relative)
    residual = relative - regional_surface
    nearshore = land & shore_band
    residual_scale = float(np.quantile(np.abs(residual[nearshore]), 0.90)) if np.any(nearshore) else 0.0
    if not math.isfinite(residual_scale) or residual_scale < 1.5:
        return relative.astype(np.float32), {
            "model": "tectonic-kinematic-selective-transgression-v1",
            "modifiedLandCellCount": 0,
            "waterConnectedNewCellCount": 0,
            "rejectedIsolatedFloodCellCount": 0,
            "outsideCoastalBandChangedCellCount": 0,
            "reason": "insufficient-nearshore-relief",
        }

    # A boundary setting is spatially broad (a margin is not one raster line),
    # whereas strain captures sharp active faults and microplate contacts.
    setting_sigma = max(1.0, 260.0 / row_km)
    convergent = _margin_signal(boundaries == 1, setting_sigma)
    divergent = _margin_signal(boundaries == 2, setting_sigma)
    transform = _margin_signal(boundaries == 3, setting_sigma)
    boundary_activity = np.clip(
        convergent + 0.82 * divergent + 0.64 * transform,
        0.0,
        1.0,
    )
    kinematic = _kinematic_activity(east, north, latitude, shore_band)
    activity = np.maximum(boundary_activity, 0.72 * kinematic)
    setting_multiplier = 1.0 + 0.34 * convergent + 0.26 * divergent + 0.10 * transform

    lowland = np.clip(-residual / residual_scale, 0.0, 1.0)
    quiet = np.clip(quiet, 0.0, 1.0)
    susceptibility = lowland ** 1.18
    susceptibility *= 0.56 + 0.44 * (1.0 - quiet)
    response = 0.40 + 0.60 * activity
    # Barely resolved relief should only receive a barely resolved response;
    # otherwise interpolation noise at a mathematically straight shore would
    # become a row of decorative inlets.
    relief_response = np.clip(residual_scale / 35.0, 0.08, 1.0)
    requested_incision = (
        (96.0 + 178.0 * float(coastline_detail))
        * support
        * susceptibility
        * response
        * setting_multiplier
        * relief_response
    )
    # At this map scale a coastal process may expose a shallow valley but may
    # not tunnel through a high ridge.  The cap preserves mountains that meet
    # the coast and leaves their headlands as a consequence of the source
    # relief, not an independently stamped cape mask.
    maximum_incision = np.where(
        land,
        0.86 * np.maximum(relative, 0.0) + 24.0,
        0.0,
    )
    incision = np.minimum(requested_incision, maximum_incision) * land
    proposed = relative.copy()
    proposed[land] -= incision[land]

    connected = _periodic_water_connection(water, proposed <= 0.0)
    isolated = land & (proposed <= 0.0) & ~connected
    proposed[isolated] = relative[isolated]
    actual_incision = relative - proposed
    modified = actual_incision > 1.0e-8
    new_water = land & (proposed <= 0.0)
    active_band = nearshore & (activity >= 0.20)
    quiet_band = nearshore & (activity < 0.08)
    active_mean = float(np.mean(actual_incision[active_band])) if np.any(active_band) else 0.0
    quiet_mean = float(np.mean(actual_incision[quiet_band])) if np.any(quiet_band) else 0.0
    weighted_lowland_share = (
        float(np.sum(actual_incision * lowland) / max(np.sum(actual_incision), 1.0e-9))
    )
    outside_changed = int(np.count_nonzero(modified & ~shore_band))
    return proposed.astype(np.float32), {
        "model": "tectonic-kinematic-selective-transgression-v1",
        "shoreSupportWidthKm": float(shore_width_km),
        "reliefComparisonScaleKm": float(relief_sigma * row_km),
        "marginInfluenceScaleKm": float(setting_sigma * row_km),
        "modifiedLandCellCount": int(np.count_nonzero(modified)),
        "waterConnectedNewCellCount": int(np.count_nonzero(new_water)),
        "rejectedIsolatedFloodCellCount": int(np.count_nonzero(isolated)),
        "outsideCoastalBandChangedCellCount": outside_changed,
        "maximumIncisionMeters": float(np.max(actual_incision, initial=0.0)),
        "meanIncisionActiveMarginMeters": active_mean,
        "meanIncisionQuietMarginMeters": quiet_mean,
        "lowlandWeightedIncisionShare": weighted_lowland_share,
        "activeShoreCellFraction": float(np.mean(activity[nearshore] >= 0.20)) if np.any(nearshore) else 0.0,
    }


def validate_coastal_surface(
    land_mask: np.ndarray,
    elevation: np.ndarray,
    boundary_class: np.ndarray,
    latitude_degrees: np.ndarray,
) -> Mapping[str, object]:
    """Measure post-synthesis coastal constraints on the actual source field.

    These are diagnostics rather than hard acceptance gates.  A convergent
    margin can legitimately bring high relief to the ocean, but an isolated
    highland terminating at a quiet shore is useful evidence for review.  The
    metrics also make the left/right ocean seam and polar land treatment
    inspectable without inferring them from an exported image.
    """

    land = np.asarray(land_mask, dtype=bool)
    heights = np.asarray(elevation, dtype=np.float64)
    boundaries = np.asarray(boundary_class, dtype=np.int8)
    latitude = np.asarray(latitude_degrees, dtype=np.float64)
    if land.ndim != 2 or heights.shape != land.shape or boundaries.shape != land.shape:
        raise ValueError("coastal validation fields must share one two-dimensional shape")
    if latitude.shape != (land.shape[0],):
        raise ValueError("one latitude is required per coastal validation row")

    water = ~land
    north_water = np.zeros_like(water)
    south_water = np.zeros_like(water)
    north_water[1:] = water[:-1]
    south_water[:-1] = water[1:]
    water_neighbour = (
        np.roll(water, 1, axis=1)
        | np.roll(water, -1, axis=1)
        | north_water
        | south_water
    )
    coast = land & water_neighbour
    highland = land & (heights >= 0.72)
    coastal_highland = highland & coast
    moderate_footslope = land & (heights >= 0.12) & (heights < 0.72)
    padded_footslope = np.concatenate(
        (moderate_footslope[:, -2:], moderate_footslope, moderate_footslope[:, :2]),
        axis=1,
    )
    foothill_nearby = ndimage.maximum_filter(
        padded_footslope,
        size=(5, 5),
        mode="nearest",
    )[:, 2:-2].astype(bool)
    active_trace = _margin_signal(boundaries != 0, max(1.0, 180.0 / (math.pi * _PLANET_RADIUS_KM / land.shape[0])))
    active_margin = active_trace >= 0.16
    unsupported_terminal = coastal_highland & ~foothill_nearby & ~active_margin
    polar = np.abs(latitude) >= 72.0
    north_polar = latitude >= 72.0
    south_polar = latitude <= -72.0
    return {
        "model": "post-synthesis-coastal-source-validation-v1",
        "leftMapEdgeOceanFraction": float(np.mean(~land[:, 0])),
        "rightMapEdgeOceanFraction": float(np.mean(~land[:, -1])),
        "northPolarLandFraction": float(np.mean(land[north_polar])) if np.any(north_polar) else 0.0,
        "southPolarLandFraction": float(np.mean(land[south_polar])) if np.any(south_polar) else 0.0,
        "polarFullWidthLandRowCount": int(np.count_nonzero(np.all(land[polar], axis=1))) if np.any(polar) else 0,
        "coastalHighlandCellCount": int(np.count_nonzero(coastal_highland)),
        "tectonicallySupportedCoastalHighlandFraction": float(
            np.mean(active_margin[coastal_highland])
        ) if np.any(coastal_highland) else 1.0,
        "unsupportedCoastalHighlandTerminalFraction": float(
            np.mean(unsupported_terminal[coastal_highland])
        ) if np.any(coastal_highland) else 0.0,
    }
