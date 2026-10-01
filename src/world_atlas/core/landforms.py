"""Evidence-based regional landforms on the accepted native physical grid.

Two- and six-cell topographic position and local relief describe regional
forms at this grid's resolution. This is not the paired line-of-sight
geomorphon algorithm, nor a survey of metre-scale cliffs or flood hazards.
No volcanic, glacial-motion, substrate or sediment-grain class is inferred.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .model import WorldGrid
from .thematic import ThematicLayers
from .society.spatial import connected_components


LANDFORM_TYPES = (
    "ridge", "hills", "valley", "gorge", "foothills", "steep-slope",
    "lowland-valley", "snow-mountain", "cape", "peninsula", "isthmus", "arid-upland",
)


@dataclass(frozen=True, slots=True)
class LandformInventory:
    masks: dict[str, np.ndarray]
    wetland_support: np.ndarray
    direction_east: np.ndarray
    direction_north: np.ndarray
    diagnostics: dict


def _shift(values: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """Look toward a neighbor with periodic longitude and bounded latitude."""
    result = np.roll(values, -dx, axis=1)
    if dy > 0:
        result = np.concatenate((result[dy:], np.full_like(result[:dy], fill)), axis=0)
    elif dy < 0:
        result = np.concatenate((np.full_like(result[dy:], fill), result[:dy]), axis=0)
    return result


def _land_neighborhood(raw: np.ndarray, land: np.ndarray, radius: int):
    size = 2 * radius + 1
    support = ndimage.uniform_filter(land.astype(float), size=size, mode=("constant", "wrap"))
    total = ndimage.uniform_filter(np.where(land, raw, 0.0), size=size, mode=("constant", "wrap"))
    mean = np.divide(total, support, out=np.zeros_like(raw), where=support > 0)
    high = ndimage.maximum_filter(np.where(land, raw, -np.inf), size=size, mode=("constant", "wrap"), cval=-np.inf)
    low = ndimage.minimum_filter(np.where(land, raw, np.inf), size=size, mode=("constant", "wrap"), cval=np.inf)
    relief = np.where(land, high - low, 0.0)
    return np.where(land, raw - mean, 0.0), relief, support


def _physical_downslope(grid: WorldGrid, raw: np.ndarray, land: np.ndarray):
    """Land-only differences in metres per physical metre, including latitude."""
    extents = grid.metadata["extents"]
    west, south, east, north = (float(extents[key]) for key in ("west", "south", "east", "north"))
    radius_m = float(grid.metadata["planet"]["radiusKm"]) * 1000.0
    height, width = grid.shape
    latitude = north - (np.arange(height) + .5) * (north - south) / height
    dx_m = radius_m * np.radians((east - west) / width) * np.cos(np.radians(latitude))[:, None]
    dy_m = radius_m * np.radians((north - south) / height)
    derivatives = []
    for dy, dx, spacing in ((0, 1, dx_m), (1, 0, dy_m)):
        ahead, behind = _shift(raw, dy, dx, 0), _shift(raw, -dy, -dx, 0)
        ahead_land, behind_land = _shift(land, dy, dx, False), _shift(land, -dy, -dx, False)
        both = land & ahead_land & behind_land
        delta = np.where(both, (ahead - behind) / 2,
                         np.where(land & ahead_land, ahead - raw,
                                  np.where(land & behind_land, raw - behind, 0.0)))
        derivatives.append(np.divide(delta, spacing, out=np.zeros_like(raw), where=np.asarray(spacing) > 0))
    east_gradient, south_gradient = derivatives
    magnitude = np.hypot(east_gradient, south_gradient)
    east = np.divide(-east_gradient, magnitude, out=np.zeros_like(raw), where=magnitude > 0)
    north = np.divide(south_gradient, magnitude, out=np.zeros_like(raw), where=magnitude > 0)
    return magnitude, east.astype(np.float32), north.astype(np.float32)


def _isthmus_support(land: np.ndarray, maritime: np.ndarray) -> np.ndarray:
    """Verify narrow land necks by a local cut separating substantial land arms.

Only resolvable cardinal necks are published. The cut is a test, never a
modification of the grid or coastline; a narrow isolated island fails it.
"""
    result = np.zeros_like(land)
    for dy, dx in ((1, 0), (0, 1)):
        across_y, across_x = dx, dy
        water_first = np.zeros_like(land)
        water_second = np.zeros_like(land)
        for distance in range(1, 4):
            water_first |= _shift(maritime, across_y * distance, across_x * distance, False)
            water_second |= _shift(maritime, -across_y * distance, -across_x * distance, False)
        arms = land.copy()
        for distance in (4, 6):
            arms &= _shift(land, dy * distance, dx * distance, False)
            arms &= _shift(land, -dy * distance, -dx * distance, False)
        rows, columns = np.nonzero(arms & water_first & water_second)
        for row, column in zip(rows, columns):
            if row < 8 or row >= land.shape[0] - 8:
                continue
            patch = land[row - 8:row + 9, (np.arange(column - 8, column + 9) % land.shape[1])].copy()
            original, _ = ndimage.label(patch)
            first, second = (8 + dy * 6, 8 + dx * 6), (8 - dy * 6, 8 - dx * 6)
            if original[first] == 0 or original[first] != original[second]:
                continue
            if dy:
                patch[8, 5:12] = False
            else:
                patch[5:12, 8] = False
            labels, _ = ndimage.label(patch)
            sizes = np.bincount(labels.ravel())
            a, b = labels[first], labels[second]
            if a and b and a != b and min(sizes[a], sizes[b]) >= 12:
                result[row, column] = True
    return result


def derive_landform_inventory(
    grid: WorldGrid, thematic: ThematicLayers, *, raw_elevation_m: np.ndarray,
) -> LandformInventory:
    raw = np.asarray(raw_elevation_m, dtype=np.float64)
    land = grid.water == 0
    maritime = np.isin(grid.water, (1, 3))
    if raw.shape != grid.shape or np.any(~np.isfinite(raw)) or np.any(raw[land] <= 0):
        raise ValueError("landforms require the accepted finite, positive land DEM in actual metres")
    if thematic.land_potential.shape != grid.shape:
        raise ValueError("landform thematic support must align with the native grid")
    tpi_small, relief_small, _ = _land_neighborhood(raw, land, 2)
    tpi_large, relief_large, fraction_large = _land_neighborhood(raw, land, 6)
    slope, east, north = _physical_downslope(grid, raw, land)
    _, _, fraction_small = _land_neighborhood(raw, land, 2)
    highland = land & (grid.elevation >= .56)
    upland_access = ndimage.maximum_filter(highland, size=9, mode=("constant", "wrap"))
    river = land & (grid.river_order >= 2)
    river_access = ndimage.maximum_filter(river, size=5, mode=("constant", "wrap"))
    valley = land & river_access & (tpi_small <= -np.maximum(65.0, .10 * relief_small))
    side_slope = ndimage.maximum_filter(slope, size=5, mode=("constant", "wrap"))
    gorge = valley & (relief_small >= 800.0) & (side_slope >= .008) & (tpi_small <= -180.0)
    lowland = land & river_access & (raw <= 500.0) & (relief_small <= 250.0) & (slope <= .004)
    ridge = land & (relief_small >= 350.0) & (tpi_small >= np.maximum(75.0, .12 * relief_small)) & (tpi_large > 0)
    hills = land & ~highland & (relief_small >= 180.0) & (relief_small < 700.0) & ~valley & ~ridge
    foothills = land & ~highland & upland_access & (relief_large >= 700.0) & (relief_small >= 200.0) & ~valley & ~ridge
    steep = land & (relief_small >= 650.0) & (slope >= .012)
    snow = highland & grid.snow & (relief_small >= 250.0)
    arid = land & thematic.physiography.desert & (relief_small >= 180.0)
    labels, sizes = connected_components(land)
    substantial = land & (np.r_[0, sizes][labels] >= 128)
    coast = land & (_shift(maritime, 0, 1, False) | _shift(maritime, 0, -1, False)
                    | _shift(maritime, 1, 0, False) | _shift(maritime, -1, 0, False))
    water_rays = np.zeros(grid.shape, dtype=np.uint8)
    inland_rays = np.zeros_like(water_rays)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            water_rays += _shift(maritime, 4 * dy, 4 * dx, False)
            inland_continuation = land.copy()
            for distance in range(1, 9):
                inland_continuation &= _shift(land, distance * dy, distance * dx, False)
            inland_rays += inland_continuation
    cape = coast & substantial & (water_rays >= 5) & (inland_rays >= 1)
    maritime_fraction = ndimage.uniform_filter(maritime.astype(float), size=13, mode=("constant", "wrap"))
    peninsula = substantial & (fraction_large >= .16) & (fraction_large <= .42) & (fraction_small >= .35) & (inland_rays >= 1) & (maritime_fraction >= .58)
    isthmus = _isthmus_support(land, maritime)
    masks = dict(zip(LANDFORM_TYPES, (ridge, hills, valley & ~gorge & ~lowland, gorge,
        foothills, steep, lowland, snow, cape, peninsula, isthmus, arid)))
    masks["wetland"] = land & thematic.physiography.wetland
    masks["desert"] = land & thematic.physiography.desert
    diagnostics = {
        "schema": "native-landform-support-v1", "method": "land-only two-scale topographic position, local relief, physical slope and supported hydro/coastal morphology",
        "tpiRadiiNativeCells": [2, 6], "reliefUnits": "metres", "slopeUnits": "metres per physical metre",
        "slopeLandOnly": True, "longitudePeriodic": True,
        "cellCounts": {kind: int(mask.sum()) for kind, mask in masks.items()},
        "criteria": {
            "ridge": "2-cell TPI >= max(75m, 12% local relief), relief >=350m; positive 6-cell TPI",
            "hills": "non-highland, local relief 180-700m; excluding ridge and river-valley cores",
            "valley": "2-cell TPI <= -max(65m,10% local relief), within 2 native cells of river order>=2",
            "gorge": "supported river valley, local relief>=800m, TPI<=-180m, nearby regional side slope>=.008",
            "foothills": "lower land within 4 cells of highland, 6-cell relief>=700m and 2-cell relief>=200m",
            "steep-slope": "2-cell relief>=650m and regional physical slope>=.012; not a local surveyed cliff",
            "lowland-valley": "within 2 cells of order>=2 river, actual elevation<=500m, relief<=250m and slope<=.004; no flood frequency claim",
            "snow-mountain": "native snow and elevation>=.56 highland with local relief>=250m; no glacier claim",
            "cape": "substantial connected land, actual coast, >=5 water-facing 4-cell rays and >=1 connected inland 4/8-cell ray",
            "peninsula": "substantial connected land, 6-cell land fraction .16-.42, 2-cell fraction>=.35, inland continuation",
            "isthmus": "water on opposite cross-sections within 3 cells, two substantial connected 6-cell land arms separated by a tested local cut",
            "arid-upland": "actual arid climate support plus local relief>=180m; neither lithology nor sand grain inferred",
            "wetland": "existing soil/hydroclimate freshwater and low-relief wetland support",
        },
        "unsupported": ["volcanic-crater", "glacier", "sand-dune", "rock-substrate", "sedimentary-delta"],
    }
    return LandformInventory(masks, thematic.physiography.wetland_support, east, north, diagnostics)
