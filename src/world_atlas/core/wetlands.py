"""Regional freshwater wetland support from the accepted drainage graph.

This is a topo-hydrologic estimate, not a hydric-soil survey or a flood map.
The grid contains relative heights, so drainage-relative height and slope
retain those units. The model never mistakes a lake's zero display height
for its water level, nor a globally ranked discharge cell for standing water.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

from .model import WorldGrid
from .suitability import relative_land_slope, smooth_transition

if TYPE_CHECKING:
    from .thematic import ClimateDrivers


WETLAND_SUPPORT_THRESHOLD = 0.5


def _lake_bank_level(elevation: np.ndarray, land: np.ndarray,
                     lake: np.ndarray) -> np.ndarray:
    """Use the lowest adjacent land bank as each lake's relative stage proxy."""
    bank = ndimage.minimum_filter(np.where(land, elevation, np.inf), size=3,
                                  mode=("constant", "wrap"), cval=np.inf)
    labels, count = ndimage.label(lake, structure=np.ones((3, 3), dtype=np.uint8))
    # Merge the same lake across the storage meridian before computing stage.
    parents = np.arange(count + 1)
    for row in range(lake.shape[0]):
        for other in range(max(0, row - 1), min(lake.shape[0], row + 2)):
            first, second = int(labels[row, 0]), int(labels[other, -1])
            if not (first and second):
                continue
            while parents[first] != first:
                first = int(parents[first])
            while parents[second] != second:
                second = int(parents[second])
            parents[max(first, second)] = min(first, second)
    while np.any(parents != parents[parents]):
        parents = parents[parents]
    labels = parents[labels]
    levels = np.r_[np.inf, ndimage.minimum(bank, labels,
                                         index=np.arange(1, count + 1))]
    return np.asarray(levels[labels], dtype=np.float64)


def _drainage_reference(grid: WorldGrid, sources: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Find the first downstream water source and path length in kilometres.

    Longitude is periodic and latitude bounded in the existing D8 graph.
    Absorbing source cells stop pointer jumping at the nearest channel,
    rather than at an outlet or a geographically close unrelated river.
    """
    height, width = grid.shape
    count = height * width
    indices = np.arange(count, dtype=np.int32)
    flow = np.asarray(grid.flow_to, dtype=np.int32).ravel()
    if flow.shape != (count,) or np.any((flow < -1) | (flow >= count)):
        raise ValueError("wetland drainage requires valid flattened flow targets")
    source = np.asarray(sources, dtype=bool).ravel()
    land = np.asarray(grid.water).ravel() == 0
    valid = land & (flow >= 0) & ~source
    parent = np.where(valid, flow, indices).astype(np.int32)
    extents = grid.metadata["extents"]
    north, south = float(extents["north"]), float(extents["south"])
    east, west = float(extents["east"]), float(extents["west"])
    latitude = north - (indices // width + .5) * (north - south) / height
    radius = float(grid.metadata["planet"]["radiusKm"])
    target = np.maximum(flow, 0)
    row_delta = (target // width - indices // width).astype(np.float64)
    col_delta = np.abs(target % width - indices % width).astype(np.float64)
    col_delta = np.minimum(col_delta, width - col_delta)
    dx = radius * np.radians((east - west) / width) * np.cos(np.radians(latitude))
    dy = radius * np.radians((north - south) / height)
    distance = np.where(valid, np.hypot(col_delta * dx, row_delta * dy), 0.0)
    for _ in range(int(np.ceil(np.log2(max(count, 2)))) + 2):
        next_parent = parent[parent]
        if np.array_equal(next_parent, parent):
            break
        distance = distance + distance[parent]
        parent = next_parent
    if np.any(valid & (parent != parent[target])):
        raise ValueError("wetland drainage contains a cycle")
    reachable = source[parent]
    return np.where(reachable, parent, -1).reshape(grid.shape), np.where(
        reachable, distance, np.inf).reshape(grid.shape)


def derive_wetland_support(grid: WorldGrid, climate: ClimateDrivers) -> np.ndarray:
    """Return continuous evidence for supplied, low, poorly drained floodplains.

    HAND-style flow connectivity and vertical distance establish drainage
    position. Flatness, local topographic position and lowland relief limit
    retention; river seasonality and climatic water balance limit supply.
    No world-area quota, global quantile, mask expansion or noise is used.
    The score is deliberately kept continuous for cartographic isolines.
    """
    land = np.asarray(grid.water) == 0
    elevation = np.asarray(grid.elevation, dtype=np.float64)
    lake = np.asarray(grid.water) == 2
    sources = lake | (land & (grid.river_order >= 2))
    if not np.any(land) or not np.any(sources):
        return np.zeros(grid.shape, dtype=np.float32)
    reference, distance = _drainage_reference(grid, sources)
    reachable = reference >= 0
    target = np.maximum(reference, 0)
    level = np.where(lake, _lake_bank_level(elevation, land, lake), elevation)
    drainage_height = np.maximum(elevation - level.ravel()[target], 0.0)
    drainage_position = 1.0 - smooth_transition(drainage_height, .002, .018)
    channel_access = 1.0 - smooth_transition(distance, 0.0, 100.0)

    slope = relative_land_slope(elevation, land)
    flatness = 1.0 - smooth_transition(slope, .002, .018)
    lowland = 1.0 - smooth_transition(elevation, .20, .45)
    support = ndimage.uniform_filter(land.astype(np.float64), size=5,
                                     mode=("constant", "wrap"))
    total = ndimage.uniform_filter(np.where(land, elevation, 0.0), size=5,
                                   mode=("constant", "wrap"))
    local_mean = np.divide(total, support, out=np.zeros(grid.shape), where=support > 0)
    # A flat valley floor can retain water; convex shoulders cannot.
    retention = 1.0 - smooth_transition(elevation - local_mean, .001, .012)

    seasonal = np.asarray(grid.seasonal_river_strength, dtype=np.float64)
    annual_strength = seasonal.mean(axis=0)
    permanence = np.divide(seasonal.min(axis=0), annual_strength,
                           out=np.zeros(grid.shape), where=annual_strength > 0)
    river_supply = smooth_transition(permanence, .25, .75)
    supplied = np.where(lake, 1.0, river_supply).ravel()[target]
    temperature = np.asarray(climate.mean_annual_temperature_c, dtype=np.float64)
    demand = 200.0 + 45.0 * np.maximum(temperature + 5.0, 0.0)
    rain_balance = np.maximum(climate.annual_precipitation_mm, 0.0) / demand
    rain_supply = smooth_transition(rain_balance, .65, 1.15)
    water_supply = np.maximum(supplied, rain_supply)
    growing = smooth_transition(climate.warmest_month_temperature_c, 0.0, 5.0)
    score = (drainage_position * channel_access * flatness * lowland
             * retention * water_supply * growing)
    score[~land | grid.snow | ~reachable] = 0.0
    return np.clip(score, 0.0, 1.0).astype(np.float32)
