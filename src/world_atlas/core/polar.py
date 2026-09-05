"""Polar surface metadata and ocean ice derived from generated continents."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class PolarRegions:
    """Canonical polar continent and surrounding sea-ice masks."""

    north_land: np.ndarray
    south_land: np.ndarray
    sea_ice: np.ndarray
    north_coast_latitude: np.ndarray
    south_coast_latitude: np.ndarray
    north_ice_edge_latitude: np.ndarray
    south_ice_edge_latitude: np.ndarray


def polar_continent_mask(grid: Any) -> np.ndarray:
    """Return canonical land belonging to generated polar continents."""

    polar = grid.metadata.get("polarRegions")
    if not isinstance(polar, Mapping):
        return np.zeros(grid.shape, dtype=bool)
    arctic = polar.get("arctic", {})
    antarctic = polar.get("antarctic", {})
    north_continent = isinstance(arctic, Mapping) and arctic.get("surface") == "continental-land"
    south_continent = isinstance(antarctic, Mapping) and antarctic.get("surface") == "continental-land"
    if not north_continent and not south_continent:
        return np.zeros(grid.shape, dtype=bool)
    planet = grid.metadata.get("planet", {})
    tilt = float(planet.get("axialTiltDegrees", 23.44))
    polar_circle = 90.0 - float(np.clip(tilt, 0.0, 45.0))
    extents = grid.metadata["extents"]
    north = float(extents["north"])
    south = float(extents["south"])
    latitude = north - (np.arange(grid.shape[0], dtype=np.float64) + 0.5) * (
        north - south
    ) / grid.shape[0]
    return (grid.water == 0) & (
        ((latitude[:, None] >= polar_circle) & north_continent)
        | ((latitude[:, None] <= -polar_circle) & south_continent)
    )


def _hash_unit(index: int, seed: int) -> float:
    value = (index + seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return (value >> 11) / float(1 << 53)


def _periodic_noise(width: int, anchors: int, seed: int) -> np.ndarray:
    values = np.fromiter(
        (_hash_unit(index, seed) for index in range(anchors)),
        dtype=np.float64,
        count=anchors,
    )
    position = np.arange(width, dtype=np.float64) * anchors / width
    left = np.floor(position).astype(np.int64) % anchors
    right = (left + 1) % anchors
    fraction = position - np.floor(position)
    weight = fraction * fraction * (3.0 - 2.0 * fraction)
    return values[left] * (1.0 - weight) + values[right] * weight


def _cell_coordinates(
    shape: tuple[int, int], extents: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    west = float(extents["west"])
    south = float(extents["south"])
    east = float(extents["east"])
    north = float(extents["north"])
    if (west, south, east, north) != (-180.0, -90.0, 180.0, 90.0):
        raise ValueError("polar surface requires the global EIR-GEOG-1 extent")
    height, width = shape
    latitude = north - (np.arange(height, dtype=np.float64) + 0.5) * (
        north - south
    ) / height
    longitude = west + (np.arange(width, dtype=np.float64) + 0.5) * (
        east - west
    ) / width
    return latitude[:, None], longitude[None, :]


def _coast_latitudes(
    latitude: np.ndarray,
    north_land: np.ndarray,
    south_land: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    latitude_grid = np.broadcast_to(latitude, north_land.shape)
    if not bool(north_land.any()) and not bool(south_land.any()):
        width = north_land.shape[1]
        return (
            np.full(width, 90.0, dtype=np.float64),
            np.full(width, -90.0, dtype=np.float64),
        )
    north = (np.min(np.where(north_land, latitude_grid, np.inf), axis=0)
             if north_land.any() else np.full(north_land.shape[1], 90.0))
    south = (np.max(np.where(south_land, latitude_grid, -np.inf), axis=0)
             if south_land.any() else np.full(south_land.shape[1], -90.0))
    if not np.all(np.isfinite(north)) or not np.all(np.isfinite(south)):
        raise ValueError("each polar continent must cross every longitude at the pole")
    return north, south


def derive_polar_regions(
    shape: tuple[int, int],
    extents: Mapping[str, Any],
    water: np.ndarray,
    north_land: np.ndarray,
    south_land: np.ndarray,
) -> PolarRegions:
    """Derive pack ice around, never instead of, both polar continents."""

    for name, value in (
        ("water", water),
        ("north_land", north_land),
        ("south_land", south_land),
    ):
        if np.asarray(value).shape != shape:
            raise ValueError(f"{name} and polar-region shape must match")
    north_land = np.asarray(north_land, dtype=bool)
    south_land = np.asarray(south_land, dtype=bool)
    if np.any(north_land & south_land):
        raise ValueError("north and south polar continents must be disjoint")
    if np.any((north_land | south_land) & (water != 0)):
        raise ValueError("generated polar continents must be canonical land")

    latitude, _longitude = _cell_coordinates(shape, extents)
    width = shape[1]
    north_edge = 70.4 + 1.9 * (_periodic_noise(width, 23, 1709) - 0.5)
    north_edge += 0.9 * (_periodic_noise(width, 79, 1777) - 0.5)
    south_edge = -70.2 - 2.0 * (_periodic_noise(width, 19, 1871) - 0.5)
    south_edge -= 0.8 * (_periodic_noise(width, 71, 1931) - 0.5)
    ocean = water == 1
    sea_ice = ocean & (
        (latitude >= north_edge[None, :])
        | (latitude <= south_edge[None, :])
    )
    north_coast, south_coast = _coast_latitudes(
        latitude, north_land, south_land
    )
    return PolarRegions(
        north_land=north_land,
        south_land=south_land,
        sea_ice=sea_ice.astype(bool),
        north_coast_latitude=north_coast,
        south_coast_latitude=south_coast,
        north_ice_edge_latitude=north_edge.astype(np.float64),
        south_ice_edge_latitude=south_edge.astype(np.float64),
    )


def apply_polar_sea_ice(
    arrays: MutableMapping[str, np.ndarray],
    extents: Mapping[str, Any],
    north_land: np.ndarray,
    south_land: np.ndarray,
) -> PolarRegions:
    """Attach derived ocean ice without mutating land, relief or hydrology."""

    regions = derive_polar_regions(
        arrays["water"].shape,
        extents,
        arrays["water"],
        north_land,
        south_land,
    )
    arrays["sea_ice"] = regions.sea_ice
    return regions


__all__ = [
    "PolarRegions",
    "apply_polar_sea_ice",
    "derive_polar_regions",
    "polar_continent_mask",
]
