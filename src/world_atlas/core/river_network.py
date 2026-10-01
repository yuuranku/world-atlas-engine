"""Resolve physical ocean outlet edges omitted by the canonical D8 storage.

The physical generator deliberately stores an ocean mouth at its final land
cell with flow_to=-1. Lake inflows retain an explicit water-cell target. The
accepted metre surface chooses the steepest D8 descent from a declared ocean
terminal, so all source geometry consumers share one physical outlet edge.
"""
from __future__ import annotations

import numpy as np

from ..physical.hydrology import D8_OFFSETS


def hydrologic_outlet_targets(grid, raw_elevation_m) -> dict[int, int]:
    """Return terminal land-index -> adjacent ocean-index without changing truth.

    Only represented flow_to=-1 channels touching explicit ocean qualify.
    Freshwater lakes, inland seas and inland sinks receive no inferred edge.
    Ocean candidates require positive-to-negative accepted metre descent;
    ties retain the physical generator's clockwise D8 ordering.
    """
    raw = np.asarray(raw_elevation_m, dtype=np.float64)
    if raw.shape != grid.shape or not np.all(np.isfinite(raw)):
        raise ValueError("hydrologic outlets require finite accepted metres aligned with the native grid")
    if not np.array_equal(raw > 0, np.asarray(grid.water) == 0):
        raise ValueError("hydrologic outlet metres must share the native physical land sign")
    height, width = grid.shape
    terminals = np.flatnonzero((grid.river_order > 0) & (grid.flow_to == -1) & (grid.water == 0))
    candidates = []
    for source in terminals:
        row, column = divmod(int(source), width)
        wet = [(direction, row + dy, column + dx, dy, dx)
               for direction, (dy, dx) in enumerate(D8_OFFSETS)
               if 0 <= row + dy < height and 0 <= column + dx < width
               and grid.water[row + dy, column + dx] == 1]
        if wet:
            candidates.append((int(source), row, column, wet))
    if not candidates:
        return {}
    extents = grid.metadata["extents"]
    latitude_step = float(extents["north"] - extents["south"]) / height
    longitude_step = float(extents["east"] - extents["west"]) / width
    radius_km = float(grid.metadata["planet"]["radiusKm"])
    if (not np.all(np.isfinite((latitude_step, longitude_step, radius_km)))
            or latitude_step <= 0 or longitude_step <= 0 or radius_km <= 0):
        raise ValueError("hydrologic outlets require positive native extents and planet radius")
    result = {}
    for source, row, column, neighbors in candidates:
        chosen, best_slope = None, -np.inf
        for _direction, target_row, target_column, dy, dx in neighbors:
            if raw[target_row, target_column] >= 0:
                continue
            latitude = np.radians(float(extents["north"]) - (row + .5) * latitude_step)
            target_latitude = np.radians(float(extents["north"]) - (target_row + .5) * latitude_step)
            haversine = (np.sin((target_latitude - latitude) * .5)**2
                         + np.cos(latitude) * np.cos(target_latitude)
                         * np.sin(np.radians(longitude_step * dx) * .5)**2)
            distance_km = 2 * radius_km * np.arcsin(np.sqrt(np.clip(haversine, 0., 1.)))
            if distance_km <= 0:
                raise ValueError("hydrologic outlet neighbors must have positive physical separation")
            slope = (raw[row, column] - raw[target_row, target_column]) / distance_km
            if slope > best_slope:
                chosen, best_slope = target_row * width + target_column, slope
        if chosen is None:
            raise ValueError(f"ocean terminal {row},{column} has no negative accepted-ground outlet")
        result[source] = chosen
    return result
