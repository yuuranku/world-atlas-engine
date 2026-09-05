"""Deterministic continuous terrain refinement and bounded valley shaping."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TerrainRefinementResult:
    """A refined relative elevation field with reproducibility diagnostics."""

    elevation: np.ndarray
    land_mask: np.ndarray
    diagnostics: dict[str, int | float]


@dataclass(frozen=True, slots=True)
class DrainageIncisionResult:
    """A terrain field lowered only inside selected drainage corridors."""

    elevation: np.ndarray
    diagnostics: dict[str, int | float]


def _validated_inputs(
    elevation: np.ndarray,
    land_mask: np.ndarray,
    ocean_mask: np.ndarray,
    lake_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(elevation, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    ocean = np.asarray(ocean_mask, dtype=bool)
    lake = np.asarray(lake_mask, dtype=bool)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("terrain elevation must be a non-empty two-dimensional array")
    if land.shape != values.shape or ocean.shape != values.shape or lake.shape != values.shape:
        raise ValueError("terrain refinement masks must share the elevation shape")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("terrain elevation must contain only finite values")
    if bool(np.any(land & (ocean | lake))) or bool(np.any(ocean & lake)):
        raise ValueError("land, ocean, and lake masks must be disjoint")
    return values, land, ocean, lake


def _distance_from_water(
    land: np.ndarray,
    water: np.ndarray,
) -> np.ndarray:
    """Return four-neighbour distance from every land cell to a water boundary."""

    height, width = land.shape
    distance = np.zeros(land.shape, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for row, column in np.argwhere(land):
        row = int(row)
        column = int(column)
        touches_water = False
        for delta_row, delta_column in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            next_row = row + delta_row
            next_column = column + delta_column
            if (
                next_row < 0
                or next_row >= height
                or next_column < 0
                or next_column >= width
                or water[next_row, next_column]
            ):
                touches_water = True
                break
        if touches_water:
            distance[row, column] = 1
            queue.append((row, column))
    while queue:
        row, column = queue.popleft()
        next_distance = int(distance[row, column]) + 1
        for delta_row, delta_column in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            next_row = row + delta_row
            next_column = column + delta_column
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and land[next_row, next_column]
                and distance[next_row, next_column] == 0
            ):
                distance[next_row, next_column] = next_distance
                queue.append((next_row, next_column))
    disconnected = land & (distance == 0)
    if bool(disconnected.any()):
        distance[disconnected] = 1
    return distance


def _structured_detail(elevation: np.ndarray) -> np.ndarray:
    rows, columns = np.indices(elevation.shape, dtype=np.float64)
    gradient_row, gradient_column = np.gradient(elevation)
    orientation = np.arctan2(gradient_row, gradient_column + 1.0e-12)
    broad = np.sin(columns * 0.061 + rows * 0.029 + 0.45 * np.sin(orientation))
    secondary = np.sin(columns * 0.023 - rows * 0.079 - 0.35 * np.cos(orientation))
    regional = np.sin(columns * 0.011 + rows * 0.013)
    detail = broad + 0.55 * secondary + 0.35 * regional
    maximum = float(np.max(np.abs(detail), initial=1.0))
    return detail / maximum


def _rolling_hill_detail(shape: tuple[int, int]) -> np.ndarray:
    """Return broad, warped hill groups instead of cell-scale terrain noise."""

    rows, columns = np.indices(shape, dtype=np.float64)
    warp = (
        0.72 * np.sin(columns * 0.012 - rows * 0.017 + 0.4)
        + 0.28 * np.sin(columns * 0.021 + rows * 0.009 + 2.0)
    )
    broad = np.sin(columns * 0.034 + rows * 0.021 + 0.78 * warp)
    oblique = np.sin(columns * 0.057 - rows * 0.039 - 0.46 * warp + 1.2)
    local = np.cos(columns * 0.083 + rows * 0.052 + 0.31 * warp + 2.4)
    detail = 0.58 * broad + 0.27 * oblique + 0.15 * local
    maximum = float(np.max(np.abs(detail), initial=1.0))
    return detail / maximum


def refine_continuous_terrain(
    elevation: np.ndarray,
    land_mask: np.ndarray,
    ocean_mask: np.ndarray,
    lake_mask: np.ndarray,
    *,
    levels: int,
) -> TerrainRefinementResult:
    """Reconstruct sub-band relief without changing authored surface masks."""

    values, land, ocean, lake = _validated_inputs(
        elevation,
        land_mask,
        ocean_mask,
        lake_mask,
    )
    if levels < 2:
        raise ValueError("terrain refinement requires at least two elevation levels")

    authored = np.clip(values, 0.0, 1.0)
    band_width = 1.0 / float(levels)
    authored_band = np.floor(
        np.clip(authored, 0.0, 1.0 - 1.0e-12) * levels
    )
    water = ocean | lake
    shore_distance = _distance_from_water(land, water).astype(np.float64)
    lowland = land & (authored < band_width)
    lowland_maximum = float(np.max(shore_distance[lowland], initial=1.0))
    lowland_position = np.zeros(authored.shape, dtype=np.float64)
    lowland_position[lowland] = shore_distance[lowland] / max(1.0, lowland_maximum)

    candidate = authored.copy()
    candidate[lowland] += band_width * (
        0.03 + 1.12 * lowland_position[lowland]
    )
    detail = _structured_detail(authored)
    detail_amplitude = band_width * 0.24
    candidate[land] += detail[land] * detail_amplitude

    # Broad, visually flat provinces need a second relief scale that survives
    # the sixteen-band SVG quantisation.  Suppress it on steep terrain and in
    # immediate coastal/floodplain margins; grouped positive lobes then read
    # as rolling hills and eroded upland remnants, not random global noise.
    gradient_row, gradient_column = np.gradient(authored)
    gradient = np.hypot(gradient_row, gradient_column)
    flat_support = np.clip(
        1.0 - gradient / max(1.0e-12, band_width * 0.32),
        0.0,
        1.0,
    ) ** 2
    inland_support = np.clip((shore_distance - 6.0) / 30.0, 0.0, 1.0)
    low_relief_support = np.clip((0.56 - authored) / 0.34, 0.0, 1.0)
    rolling_hill_signal = np.clip(
        (_rolling_hill_detail(authored.shape) - 0.06) / 0.94,
        0.0,
        1.0,
    ) ** 1.25
    rolling_hill_uplift = (
        band_width
        * 1.95
        * rolling_hill_signal
        * flat_support
        * inland_support
        * low_relief_support
    )
    rolling_hill_uplift[~land] = 0.0
    candidate[land] += rolling_hill_uplift[land]
    candidate[lowland] = np.maximum(candidate[lowland], band_width * 0.005)

    lower = np.maximum(0.0, (authored_band - 1.0) / levels)
    upper = np.minimum(1.0, (authored_band + 2.0 - 1.0e-6) / levels)
    result = np.zeros(authored.shape, dtype=np.float64)
    result[land] = np.clip(candidate[land], lower[land], upper[land])
    exact_zero_before = int(np.count_nonzero(land & (authored == 0.0)))
    exact_zero_after = int(np.count_nonzero(land & (result == 0.0)))
    refined_band = np.floor(
        np.clip(result, 0.0, 1.0 - 1.0e-12) * levels
    )
    maximum_band_displacement = float(
        np.max(np.abs(refined_band[land] - authored_band[land]), initial=0.0)
    )
    return TerrainRefinementResult(
        elevation=result,
        land_mask=land.copy(),
        diagnostics={
            "exactZeroLandCellsBefore": exact_zero_before,
            "exactZeroLandCellsAfter": exact_zero_after,
            "maximumBandDisplacement": maximum_band_displacement,
            "lowlandCellCount": int(np.count_nonzero(lowland)),
            "detailAmplitude": float(detail_amplitude),
            "rollingHillCellCount": int(
                np.count_nonzero(rolling_hill_uplift >= band_width * 0.08)
            ),
            "maximumRollingHillUplift": float(
                np.max(rolling_hill_uplift, initial=0.0)
            ),
        },
    )


def incise_drainage(
    elevation: np.ndarray,
    land_mask: np.ndarray,
    stream_mask: np.ndarray,
    downstream_index: np.ndarray,
    accumulation: np.ndarray,
    *,
    maximum_depth: float = 0.035,
) -> DrainageIncisionResult:
    """Lower selected valleys within a one-cell corridor and a finite depth."""

    values = np.asarray(elevation, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    stream = np.asarray(stream_mask, dtype=bool)
    downstream = np.asarray(downstream_index, dtype=np.int64)
    flow = np.asarray(accumulation, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("drainage incision elevation must be a non-empty 2-D array")
    if (
        land.shape != values.shape
        or stream.shape != values.shape
        or downstream.shape != values.shape
        or flow.shape != values.shape
    ):
        raise ValueError("drainage incision arrays must share one shape")
    if not bool(np.all(np.isfinite(values))) or not bool(np.all(np.isfinite(flow))):
        raise ValueError("drainage incision arrays must contain finite values")
    if maximum_depth <= 0.0:
        raise ValueError("maximum drainage incision depth must be positive")
    if bool(np.any(stream & ~land)):
        raise ValueError("drainage incision stream cells must be land")

    relative_flow = np.zeros(values.shape, dtype=np.float64)
    if bool(stream.any()):
        logarithmic_maximum = max(
            1.0,
            float(np.log1p(np.max(flow[stream], initial=1.0))),
        )
        relative_flow[stream] = np.log1p(np.maximum(0.0, flow[stream])) / logarithmic_maximum
    stream_depth = np.zeros(values.shape, dtype=np.float64)
    stream_depth[stream] = np.minimum(
        maximum_depth,
        0.006 + (maximum_depth - 0.006) * relative_flow[stream],
    )
    influence = stream_depth.copy()
    height, width = values.shape
    for delta_row in (-1, 0, 1):
        for delta_column in (-1, 0, 1):
            if delta_row == 0 and delta_column == 0:
                continue
            source_row_start = max(0, -delta_row)
            source_row_stop = min(height, height - delta_row)
            source_column_start = max(0, -delta_column)
            source_column_stop = min(width, width - delta_column)
            target_row_start = source_row_start + delta_row
            target_row_stop = source_row_stop + delta_row
            target_column_start = source_column_start + delta_column
            target_column_stop = source_column_stop + delta_column
            shifted = stream_depth[
                source_row_start:source_row_stop,
                source_column_start:source_column_stop,
            ] * 0.45
            influence[
                target_row_start:target_row_stop,
                target_column_start:target_column_stop,
            ] = np.maximum(
                influence[
                    target_row_start:target_row_stop,
                    target_column_start:target_column_stop,
                ],
                shifted,
            )
    influence[~land] = 0.0
    incised = values.copy()
    incised[land] = np.maximum(0.0, values[land] - influence[land])

    stream_indices = np.flatnonzero(stream.ravel())
    order = sorted(stream_indices, key=lambda index: float(flow.ravel()[index]), reverse=True)
    for source in order:
        target = int(downstream.ravel()[source])
        if target < 0 or not stream.ravel()[target]:
            continue
        if incised.ravel()[source] < incised.ravel()[target]:
            incised.ravel()[source] = min(
                values.ravel()[source],
                incised.ravel()[target],
            )

    actual_depth = values - incised
    return DrainageIncisionResult(
        elevation=incised,
        diagnostics={
            "incisedCellCount": int(np.count_nonzero(actual_depth > 0.0)),
            "streamCellCount": int(np.count_nonzero(stream)),
            "maximumIncisionDepth": float(np.max(actual_depth, initial=0.0)),
        },
    )
