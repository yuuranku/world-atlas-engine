"""Deterministic, longitude-periodic completion of omitted ocean relief."""

from __future__ import annotations

from collections import deque

import numpy as np


def _distance_from_land(land: np.ndarray, *, maximum_distance: int) -> np.ndarray:
    """Four-neighbour distance with wrapped longitude and finite poles."""

    source = np.asarray(land, dtype=bool)
    height, width = source.shape
    distance = np.full(source.shape, maximum_distance + 1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for raw_row, raw_column in np.argwhere(source):
        row = int(raw_row)
        column = int(raw_column)
        distance[row, column] = 0
        queue.append((row, column))
    while queue:
        row, column = queue.popleft()
        next_distance = int(distance[row, column]) + 1
        if next_distance > maximum_distance:
            continue
        for delta_row, delta_column in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            next_row = row + delta_row
            next_column = (column + delta_column) % width
            if (
                0 <= next_row < height
                and next_distance < distance[next_row, next_column]
            ):
                distance[next_row, next_column] = next_distance
                queue.append((next_row, next_column))
    return distance


def complete_ocean_bathymetry(
    ocean_mask: np.ndarray,
    land_mask: np.ndarray,
    authored_bathymetry: np.ndarray,
) -> np.ndarray:
    """Fill omitted deep-ocean relief without overwriting authored shelves.

    The source artwork has broad uniform strips around the antimeridian.  This
    function supplies a low-frequency abyssal-ridge and fracture-zone field in
    those deep areas.  Every longitudinal harmonic is periodic, so the left and
    right edges are the same ocean basin rather than two clipped margins.
    """

    ocean = np.asarray(ocean_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    authored = np.asarray(authored_bathymetry, dtype=np.float64)
    if ocean.ndim != 2 or land.shape != ocean.shape or authored.shape != ocean.shape:
        raise ValueError("ocean-floor fields must be two-dimensional and share a shape")
    if bool(np.any(ocean & land)):
        raise ValueError("ocean and land masks must be disjoint")
    if not bool(np.all(np.isfinite(authored))):
        raise ValueError("authored bathymetry must be finite")

    height, width = ocean.shape
    rows, columns = np.indices(ocean.shape, dtype=np.float64)
    longitude = columns / max(1.0, float(width)) * np.pi * 2.0
    latitude = (rows / max(1.0, float(height - 1)) - 0.5) * np.pi

    # Broad ridges, offset fracture zones and smaller abyssal swells.  The
    # oblique latitude terms avoid decorative horizontal stripes while every
    # longitude frequency still closes exactly at the antimeridian.
    warp = (
        0.24 * np.sin(3.0 * longitude - 1.70 * latitude + 0.37)
        + 0.11 * np.sin(7.0 * longitude + 2.40 * latitude - 1.18)
    )
    basin_a = np.sin(2.0 * (longitude + warp) + 1.25 * latitude + 0.61)
    basin_b = np.cos(
        3.0 * longitude
        - 2.35 * latitude
        + 0.31 * np.sin(4.0 * longitude + latitude)
        + 2.37
    )
    knotted_swells = basin_a * basin_b
    structural = (
        0.81
        + 0.052 * basin_a
        + 0.048 * basin_b
        + 0.052 * knotted_swells
        + 0.032 * np.cos(8.0 * longitude + 4.30 * latitude - 0.84)
        + 0.022
        * np.sin(13.0 * longitude - 7.20 * latitude + 1.19)
        * np.cos(5.0 * longitude + 3.10 * latitude)
    )
    structural = np.clip(structural, 0.58, 0.99)

    distance = _distance_from_land(land, maximum_distance=max(24, width // 5))
    far_ocean = ocean & (distance >= max(10, int(round(height * 0.035))))
    authored_shelf = ocean & (authored <= 0.58)
    blend = np.clip(
        (distance.astype(np.float64) - 8.0) / max(12.0, width * 0.055),
        0.0,
        0.78,
    )
    completed = authored.copy()
    completed[far_ocean] = (
        authored[far_ocean] * (1.0 - blend[far_ocean])
        + structural[far_ocean] * blend[far_ocean]
    )
    # Authored continental shelves and nearshore bathymetry are evidence and
    # may only become shallower, never be deepened by the completion field.
    completed[authored_shelf] = np.minimum(
        completed[authored_shelf], authored[authored_shelf]
    )
    completed[land] = 0.0
    completed[ocean] = np.clip(completed[ocean], 0.0, 1.0)
    return completed


__all__ = ["complete_ocean_bathymetry"]
