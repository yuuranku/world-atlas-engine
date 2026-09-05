"""Infer broad tectonic, coastal, and basin provinces from shared terrain fields."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class PhysiographicProvinceResult:
    elevation: np.ndarray
    mountain_belt_mask: np.ndarray
    foothill_mask: np.ndarray
    active_margin_mask: np.ndarray
    basin_rim_mask: np.ndarray
    diagnostics: dict[str, int | float]


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    padded = np.pad(array, ((radius, radius), (radius, radius)), mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    size = radius * 2 + 1
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    ) / float(size * size)


def _shift(values: np.ndarray, delta_row: int, delta_column: int) -> np.ndarray:
    result = np.zeros_like(values)
    height, width = values.shape
    if abs(delta_row) >= height or abs(delta_column) >= width:
        return result
    source_row_start = max(0, -delta_row)
    source_row_stop = min(height, height - delta_row)
    source_column_start = max(0, -delta_column)
    source_column_stop = min(width, width - delta_column)
    target_row_start = source_row_start + delta_row
    target_row_stop = source_row_stop + delta_row
    target_column_start = source_column_start + delta_column
    target_column_stop = source_column_stop + delta_column
    result[target_row_start:target_row_stop, target_column_start:target_column_stop] = values[
        source_row_start:source_row_stop,
        source_column_start:source_column_stop,
    ]
    return result


def _directional_support(seed: np.ndarray) -> np.ndarray:
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    offsets = (-30, -20, -10, 0, 10, 20, 30)
    supports: list[np.ndarray] = []
    for delta_row, delta_column in directions:
        total = np.zeros(seed.shape, dtype=np.float64)
        for offset in offsets:
            total += _shift(seed, delta_row * offset, delta_column * offset)
        supports.append(total / float(len(offsets)))
    return np.maximum.reduce(supports)


def _hierarchical_orogenic_detail(shape: tuple[int, int]) -> np.ndarray:
    """Return deterministic, non-repeating regional-to-local ridge modulation."""

    rows, columns = np.indices(shape, dtype=np.float64)
    warp = (
        0.62 * np.sin(columns * 0.009 - rows * 0.017)
        + 0.38 * np.sin(columns * 0.015 + rows * 0.021 + 1.7)
    )
    regional = np.sin(columns * 0.041 + rows * 0.017 + 0.78 * warp)
    subrange = np.sin(columns * 0.073 - rows * 0.031 - 0.52 * warp + 0.9)
    intermediate = np.sin(columns * 0.093 + rows * 0.026 + 0.61 * warp + 0.35)
    local = np.sin(columns * 0.119 + rows * 0.047 + 0.36 * warp + 2.1)
    cross = np.sin(columns * 0.029 - rows * 0.087 + 0.44 * np.sin(columns * 0.013))
    detail = (
        0.32 * regional
        + 0.23 * subrange
        + 0.18 * intermediate
        + 0.19 * local
        + 0.08 * cross
    )
    maximum = float(np.max(np.abs(detail), initial=1.0))
    return detail / maximum


def _distance_from_mask(
    source_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    maximum_distance: int,
) -> np.ndarray:
    source = np.asarray(source_mask, dtype=bool)
    active = np.asarray(active_mask, dtype=bool)
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
            next_column = column + delta_column
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and active[next_row, next_column]
                and next_distance < distance[next_row, next_column]
            ):
                distance[next_row, next_column] = next_distance
                queue.append((next_row, next_column))
    return distance


def shape_physiographic_provinces(
    elevation: np.ndarray,
    land_mask: np.ndarray,
    ocean_mask: np.ndarray,
    inland_sea_mask: np.ndarray,
    bathymetry: np.ndarray,
    *,
    levels: int,
) -> PhysiographicProvinceResult:
    """Add bounded relief whose form follows mountain, margin, and basin evidence."""

    authored = np.asarray(elevation, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    ocean = np.asarray(ocean_mask, dtype=bool)
    inland_sea = np.asarray(inland_sea_mask, dtype=bool)
    depth = np.asarray(bathymetry, dtype=np.float64)
    if authored.ndim != 2 or authored.size == 0:
        raise ValueError("physiographic elevation must be a non-empty two-dimensional array")
    if any(mask.shape != authored.shape for mask in (land, ocean, inland_sea)) or depth.shape != authored.shape:
        raise ValueError("physiographic fields must share a shape")
    if bool(np.any(land & (ocean | inland_sea))) or bool(np.any(ocean & inland_sea)):
        raise ValueError("physiographic surface masks must be disjoint")
    if not bool(np.all(np.isfinite(authored))) or not bool(np.all(np.isfinite(depth))):
        raise ValueError("physiographic fields must contain finite values")
    if levels < 2:
        raise ValueError("physiographic shaping requires at least two elevation levels")

    band_width = 1.0 / float(levels)
    clipped = np.clip(authored, 0.0, 1.0)
    ridge_seed = np.zeros(authored.shape, dtype=np.float64)
    ridge_seed[land] = np.clip((clipped[land] - 0.38) / 0.42, 0.0, 1.0)
    axial = _directional_support(ridge_seed)
    belt_signal = _box_mean(np.maximum(axial, ridge_seed), radius=11)
    mountain_belt = land & (belt_signal >= 0.055)
    foothill_signal = _box_mean(mountain_belt.astype(np.float64), radius=12)
    foothill = land & ~mountain_belt & (foothill_signal >= 0.035)
    orogenic_uplift = band_width * (
        1.55 * np.clip(belt_signal, 0.0, 1.0)
        + 0.48 * np.clip(foothill_signal, 0.0, 0.45)
    )

    deep_ocean = ocean & (depth >= 0.65)
    coast_distance = _distance_from_mask(ocean, land, maximum_distance=48)
    # Deep water may be separated from land by a narrow shallow shelf.  The
    # distance therefore crosses that shelf before entering the land domain;
    # restricting traversal to land would make every real shelf an infinite
    # barrier and erase the active-margin signal.
    deep_distance = _distance_from_mask(
        deep_ocean,
        np.ones(land.shape, dtype=bool),
        maximum_distance=48,
    )
    coast_boundary = land & (coast_distance == 1)
    boundary_density = _box_mean(coast_boundary.astype(np.float64), radius=9) * 19.0
    ruggedness = np.clip((boundary_density - 0.55) / 1.25, 0.0, 1.0)
    deep_proximity = np.clip(1.0 - (deep_distance.astype(np.float64) - 1.0) / 28.0, 0.0, 1.0)
    coast_position = np.clip(coast_distance.astype(np.float64) / 36.0, 0.0, 1.0)
    inland_hump = np.sin(math.pi * coast_position)
    active_margin_signal = ruggedness * deep_proximity * inland_hump
    active_margin = land & (active_margin_signal >= 0.055)
    active_margin_uplift = band_width * 1.35 * active_margin_signal

    sea_distance = _distance_from_mask(inland_sea, land, maximum_distance=72)
    sea_distance_float = sea_distance.astype(np.float64)
    basin_rim_signal = np.exp(-0.5 * ((sea_distance_float - 25.0) / 11.0) ** 2)
    basin_rim_signal[(sea_distance <= 6) | (sea_distance > 64) | ~land] = 0.0
    basin_rim = land & (basin_rim_signal >= 0.18)
    basin_rim_uplift = band_width * 1.05 * basin_rim_signal

    hierarchy_support = np.clip(
        1.20 * belt_signal
        + 0.65 * foothill_signal
        + 0.85 * active_margin_signal
        + 0.42 * basin_rim_signal,
        0.0,
        1.0,
    )
    hierarchical_detail = _hierarchical_orogenic_detail(authored.shape)
    subrange_signal = np.clip((hierarchical_detail - 0.02) / 0.98, 0.0, 1.0)
    # A secondary range must survive elevation quantisation at world scale.
    # Keep the modulation inside supported orogenic provinces, but allow its
    # relief to span a little more than one rendered elevation band.
    subrange_uplift = band_width * 1.80 * subrange_signal * hierarchy_support

    candidate = (
        clipped
        + orogenic_uplift
        + active_margin_uplift
        + basin_rim_uplift
        + subrange_uplift
    )
    authored_band = np.floor(np.clip(clipped, 0.0, 1.0 - 1.0e-12) * levels)
    upper = np.minimum(1.0, (authored_band + 3.0 - 1.0e-6) / levels)
    result = np.zeros(authored.shape, dtype=np.float64)
    result[land] = np.clip(candidate[land], clipped[land], upper[land])
    result_band = np.floor(np.clip(result, 0.0, 1.0 - 1.0e-12) * levels)
    maximum_band_displacement = float(
        np.max(np.abs(result_band[land] - authored_band[land]), initial=0.0)
    )
    return PhysiographicProvinceResult(
        elevation=result,
        mountain_belt_mask=mountain_belt,
        foothill_mask=foothill,
        active_margin_mask=active_margin,
        basin_rim_mask=basin_rim,
        diagnostics={
            "mountainBeltCellCount": int(np.count_nonzero(mountain_belt)),
            "foothillCellCount": int(np.count_nonzero(foothill)),
            "activeMarginCellCount": int(np.count_nonzero(active_margin)),
            "basinRimCellCount": int(np.count_nonzero(basin_rim)),
            "maximumBandDisplacement": maximum_band_displacement,
            "maximumOrogenicUplift": float(np.max(orogenic_uplift[land], initial=0.0)),
            "maximumActiveMarginUplift": float(np.max(active_margin_uplift[land], initial=0.0)),
            "maximumBasinRimUplift": float(np.max(basin_rim_uplift[land], initial=0.0)),
            "subrangeCellCount": int(np.count_nonzero(land & (subrange_uplift > 0.0))),
            "maximumSubrangeUplift": float(np.max(subrange_uplift[land], initial=0.0)),
        },
    )


__all__ = ["PhysiographicProvinceResult", "shape_physiographic_provinces"]
