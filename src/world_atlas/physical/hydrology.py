"""Deterministic, grid-based elevation hydrology for the physical atlas.

The module intentionally stops at a hydrologic network of grid cells.  It does
not know about SVG paths, map styling, or authored display geometry.  A coloured
elevation raster can therefore be converted to a stable drainage graph before
any cartographic simplification happens.

The implementation uses four stages:

* finite-depth stream burning on the supplied relative elevations;
* a deterministic Priority-Flood depression correction, with explicit lakes
  kept out of the fill domain;
* D8 routing with a flood-order tie break on flats;
* accumulation-threshold stream extraction and directed network compression.

All neighbourhood scans are row-major and all ties use explicit integer keys,
so repeated calls with the same arrays produce identical arrays and network
records.  Standard library and NumPy are sufficient.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


Cell = tuple[int, int]
NetworkRecord = dict[str, Any]


# Clockwise D8 order, starting at north.  Keeping this order public through the
# result's integer direction array makes the tie break reproducible for callers.
D8_OFFSETS: tuple[Cell, ...] = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


# These defaults are deliberately named and surfaced through
# ``HydrologyConfig`` below.  Keeping the scale factors together makes the
# selection policy auditable: none of the stream/valley gates depend on a
# source-map coordinate or an authored pixel count.
DEFAULT_MFD_EXPONENT = 1.1
# Keep the raw uphill gate on the same relative scale as the independent
# hydrography audit.  It filters authored quantisation noise without allowing
# a material ridge crossing to become a published route.
DEFAULT_RAW_UPHILL_TOLERANCE_FRACTION = 0.02
DEFAULT_VALLEY_WINDOW_FRACTION = 0.02
DEFAULT_VALLEY_DEPTH_FRACTION = 0.01
DEFAULT_VALLEY_SUPPORT_THRESHOLD = 0.2
DEFAULT_VALLEY_SUPPORT_DISTANCE_FACTOR = 1.0
DEFAULT_PARALLEL_SEARCH_RADIUS_FACTOR = 1.0
DEFAULT_PARALLEL_PRIORITY_WEIGHT = 0.35
DEFAULT_MINIMUM_STREAM_SPAN_FACTOR = 0.25
DEFAULT_CLOSURE_BUDGET_FRACTION = 0.05
DEFAULT_CLOSURE_MAX_CHAIN_FRACTION = 0.012
DEFAULT_RUNOFF_WEIGHT_FLOOR = 0.35
DEFAULT_SNOWMELT_RUNOFF_BONUS = 0.75
DEFAULT_HEADWATER_UPLAND_QUANTILE = 0.65
DEFAULT_LOWLAND_HEADWATER_FLOW_MULTIPLIER = 3.0
DEFAULT_MAJOR_RIVERS_PER_CONTINENT = 2
DEFAULT_MAJOR_RIVER_TRIBUTARIES = 3
DEFAULT_CONTINENT_MINIMUM_LAND_FRACTION = 0.02
DEFAULT_MAJOR_RIVER_MINIMUM_SPAN_FACTOR = 0.35


@dataclass(frozen=True, slots=True)
class HydrologyConfig:
    """Tunable, display-independent hydrology parameters.

    Fractions are relative to the number of ``True`` cells in ``land_mask``.
    ``stream_threshold_fraction`` controls which cells become stream cells;
    the larger fractions only classify extracted network segments.  A finite
    ``stream_burn_depth`` lowers authored constraint cells before depression
    correction; it never forces a constraint cell into the final network.

    ``lake_outlets`` optionally maps a caller-selected stable anchor cell from
    each lake component to a cell inside that same lake.  The anchor may be
    any deterministic lake cell; component ordinal numbers are deliberately
    not accepted because they can change when unrelated lakes are added or
    removed.  Without an entry, every lake cell is a terminal water body.  With
    an entry, an acyclic D8 tree routes lake cells toward that outlet while the
    extracted river still terminates at the first lake cell it reaches.
    """

    stream_burn_depth: float = 0.05
    stream_threshold_fraction: float = 0.005
    tributary_threshold_fraction: float = 0.02
    mainstem_threshold_fraction: float = 0.08
    minimum_headwater_length: int = 3
    lake_outlets: Mapping[Cell, Cell] | None = None
    inland_sink_cells: tuple[Cell, ...] = ()
    tie_epsilon: float = 1.0e-12
    # A relative floor prevents a stream threshold of zero/small fraction from
    # turning every one-cell source into a channel on a large grid.
    minimum_stream_span_factor: float = DEFAULT_MINIMUM_STREAM_SPAN_FACTOR
    # Maximum number of unselected terrain cells that flow-path closure may
    # add, normalized by the sampled land-cell count.  A zero value disables
    # closure additions; it never removes selected evidence cells.
    closure_budget_fraction: float = DEFAULT_CLOSURE_BUDGET_FRACTION
    # A closure chain longer than this fraction of the raster short side is
    # treated as an unobserved gap.  It is intentionally separate from the
    # total closure budget: a large budget must not rebuild one long straight
    # channel from a single selected cell.
    closure_max_chain_fraction: float = DEFAULT_CLOSURE_MAX_CHAIN_FRACTION
    # The following factors are dimensionless and scale with the sampled grid
    # or with valley depth; they are not geographic/map-coordinate constants.
    mfd_exponent: float = DEFAULT_MFD_EXPONENT
    valley_window_fraction: float = DEFAULT_VALLEY_WINDOW_FRACTION
    valley_depth_fraction: float = DEFAULT_VALLEY_DEPTH_FRACTION
    valley_support_threshold: float = DEFAULT_VALLEY_SUPPORT_THRESHOLD
    valley_support_distance_factor: float = DEFAULT_VALLEY_SUPPORT_DISTANCE_FACTOR
    parallel_search_radius_factor: float = DEFAULT_PARALLEL_SEARCH_RADIUS_FACTOR
    parallel_priority_weight: float = DEFAULT_PARALLEL_PRIORITY_WEIGHT
    # Climate yield is combined with a bounded terrain/background term and
    # perennial snowmelt, then normalized to one land-cell equivalent on
    # average so area-fraction thresholds remain resolution independent.
    runoff_weight_floor: float = DEFAULT_RUNOFF_WEIGHT_FLOOR
    snowmelt_runoff_bonus: float = DEFAULT_SNOWMELT_RUNOFF_BONUS
    # A candidate headwater is retained when its upstream catchment reaches
    # this land-elevation quantile, contains perennial snow, or carries the
    # larger lowland flow multiple below.  This allows rain/groundwater-fed
    # lowland rivers without letting every broad plain become a comb.
    headwater_upland_quantile: float = DEFAULT_HEADWATER_UPLAND_QUANTILE
    lowland_headwater_flow_multiplier: float = (
        DEFAULT_LOWLAND_HEADWATER_FLOW_MULTIPLIER
    )
    # Large land components retain a small number of continent-scale river
    # spines.  Their length scales with sqrt(component area), so the contract
    # survives different source maps and raster resolutions.
    major_rivers_per_continent: int = DEFAULT_MAJOR_RIVERS_PER_CONTINENT
    major_river_tributaries: int = DEFAULT_MAJOR_RIVER_TRIBUTARIES
    continent_minimum_land_fraction: float = (
        DEFAULT_CONTINENT_MINIMUM_LAND_FRACTION
    )
    major_river_minimum_span_factor: float = (
        DEFAULT_MAJOR_RIVER_MINIMUM_SPAN_FACTOR
    )


@dataclass(frozen=True, slots=True)
class HydrologyResult:
    """Outputs of :func:`compute_hydrology`.

    ``flow_direction`` stores a D8 code from :data:`D8_OFFSETS`, or ``-1`` for
    an in-grid terminal/outlet.  ``downstream_index`` stores a flattened target
    cell index, or ``-1`` for an ocean, lake terminal, inland sink, or inactive
    cell.  Lake cells are included in the latter graph so optional lake outlet
    routing can be inspected, but they are not counted as land streams.

    ``mfd_accumulation`` is a fractional, multiple-flow accumulation computed
    from the same corrected surface.  It is used only to select representative
    stream corridors; ``accumulation`` remains the single-target D8 discharge
    used for directed network order and reach metadata.

    ``valley_score`` and ``valley_support`` retain the normalized local
    concavity field for map-lab audits; ``valley_supported`` is the propagated
    boolean gate used for stream selection.  ``constraint_strength`` and
    ``flat_distance`` are retained for the same purpose.  These arrays are not
    serialized into the physical source.  ``diagnostics`` holds scalar
    selection/pruning counts.

    ``network`` is a tuple of JSON-friendly dictionaries.  ``from`` and ``to``
    are ``[row, column]`` cells; a coast endpoint is represented by its final
    in-grid land cell and a lake endpoint by the first lake cell reached.
    ``outlet_type`` is one of ``ocean``, ``lake``, or ``inland_sink``.
    """

    burned_elevation: np.ndarray
    corrected_elevation: np.ndarray
    flow_direction: np.ndarray
    accumulation: np.ndarray
    mfd_accumulation: np.ndarray
    valley_score: np.ndarray
    valley_support: np.ndarray
    valley_supported: np.ndarray
    constraint_strength: np.ndarray
    flat_distance: np.ndarray
    stream_mask: np.ndarray
    stream_order: np.ndarray
    outlet_type: np.ndarray
    downstream_index: np.ndarray
    land_mask: np.ndarray
    lake_mask: np.ndarray
    network: tuple[NetworkRecord, ...]
    diagnostics: dict[str, Any]

class HydrologyError(ValueError):
    """Raised when raster inputs or hydrologic invariants are invalid."""


def _validate_shape(array: np.ndarray, shape: tuple[int, int], name: str) -> None:
    if array.ndim != 2 or array.shape != shape:
        raise HydrologyError(f"{name} must be a 2-D array with shape {shape}")


def _normalise_constraint_mask(
    value: np.ndarray | Sequence[Sequence[float]] | None,
    shape: tuple[int, int],
) -> np.ndarray:
    """Convert boolean or non-negative authored strengths to ``[0, 1]``."""

    if value is None:
        return np.zeros(shape, dtype=np.float64)
    raw = np.asarray(value)
    _validate_shape(raw, shape, "river_constraint_mask")
    if raw.dtype == np.bool_:
        return raw.astype(np.float64, copy=False)
    try:
        strengths = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise HydrologyError("river_constraint_mask must be numeric or boolean") from error
    if not np.all(np.isfinite(strengths)):
        raise HydrologyError("river_constraint_mask contains non-finite values")
    if np.any(strengths < 0):
        raise HydrologyError("river_constraint_mask cannot contain negative values")
    maximum = float(np.max(strengths, initial=0.0))
    if maximum > 1.0:
        strengths = strengths / maximum
    return strengths


def _component_cells(mask: np.ndarray) -> list[list[Cell]]:
    """Return deterministic 8-connected components of a boolean raster."""

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[Cell]] = []
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or visited[row, column]:
                continue
            queue: deque[Cell] = deque([(row, column)])
            visited[row, column] = True
            cells: list[Cell] = []
            while queue:
                current_row, current_column = queue.popleft()
                cells.append((current_row, current_column))
                for row_delta, column_delta in D8_OFFSETS:
                    next_row = current_row + row_delta
                    next_column = current_column + column_delta
                    if not (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and mask[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        continue
                    visited[next_row, next_column] = True
                    queue.append((next_row, next_column))
            components.append(sorted(cells))
    return components


def _neighbours(cell: Cell, shape: tuple[int, int]) -> list[tuple[int, Cell]]:
    """Return in-bounds D8 neighbours in explicit direction order."""

    row, column = cell
    height, width = shape
    result: list[tuple[int, Cell]] = []
    for direction, (row_delta, column_delta) in enumerate(D8_OFFSETS):
        next_row = row + row_delta
        next_column = column + column_delta
        if 0 <= next_row < height and 0 <= next_column < width:
            result.append((direction, (next_row, next_column)))
    return result


def _direction_between(source: Cell, target: Cell) -> int:
    row_delta = target[0] - source[0]
    column_delta = target[1] - source[1]
    try:
        return D8_OFFSETS.index((row_delta, column_delta))
    except ValueError as error:  # pragma: no cover - internal invariant guard
        raise HydrologyError(f"non-D8 target {source} -> {target}") from error


def _validate_config(config: HydrologyConfig) -> None:
    values = (
        ("stream_burn_depth", config.stream_burn_depth),
        ("stream_threshold_fraction", config.stream_threshold_fraction),
        ("tributary_threshold_fraction", config.tributary_threshold_fraction),
        ("mainstem_threshold_fraction", config.mainstem_threshold_fraction),
        ("tie_epsilon", config.tie_epsilon),
        ("minimum_stream_span_factor", config.minimum_stream_span_factor),
        ("closure_budget_fraction", config.closure_budget_fraction),
        ("closure_max_chain_fraction", config.closure_max_chain_fraction),
        ("mfd_exponent", config.mfd_exponent),
        ("valley_window_fraction", config.valley_window_fraction),
        ("valley_depth_fraction", config.valley_depth_fraction),
        ("valley_support_threshold", config.valley_support_threshold),
        (
            "valley_support_distance_factor",
            config.valley_support_distance_factor,
        ),
        ("parallel_search_radius_factor", config.parallel_search_radius_factor),
        ("parallel_priority_weight", config.parallel_priority_weight),
        ("runoff_weight_floor", config.runoff_weight_floor),
        ("snowmelt_runoff_bonus", config.snowmelt_runoff_bonus),
        ("headwater_upland_quantile", config.headwater_upland_quantile),
        (
            "lowland_headwater_flow_multiplier",
            config.lowland_headwater_flow_multiplier,
        ),
        (
            "continent_minimum_land_fraction",
            config.continent_minimum_land_fraction,
        ),
        (
            "major_river_minimum_span_factor",
            config.major_river_minimum_span_factor,
        ),
    )
    for name, value in values:
        if not math.isfinite(float(value)):
            raise HydrologyError(f"{name} must be finite")
    if config.stream_burn_depth < 0:
        raise HydrologyError("stream_burn_depth cannot be negative")
    if config.tie_epsilon <= 0:
        raise HydrologyError("tie_epsilon must be positive")
    if config.minimum_stream_span_factor < 0:
        raise HydrologyError("minimum_stream_span_factor cannot be negative")
    if not 0 <= config.closure_budget_fraction <= 1:
        raise HydrologyError("closure_budget_fraction must be within [0, 1]")
    if not 0 <= config.closure_max_chain_fraction <= 1:
        raise HydrologyError("closure_max_chain_fraction must be within [0, 1]")
    if config.mfd_exponent <= 0:
        raise HydrologyError("mfd_exponent must be positive")
    if not 0 <= config.valley_window_fraction <= 1:
        raise HydrologyError("valley_window_fraction must be within [0, 1]")
    if not 0 <= config.valley_depth_fraction <= 1:
        raise HydrologyError("valley_depth_fraction must be within [0, 1]")
    if not 0 <= config.valley_support_threshold <= 1:
        raise HydrologyError("valley_support_threshold must be within [0, 1]")
    if config.valley_support_distance_factor < 0:
        raise HydrologyError("valley_support_distance_factor cannot be negative")
    if config.parallel_search_radius_factor < 0:
        raise HydrologyError("parallel_search_radius_factor cannot be negative")
    if config.parallel_priority_weight < 0:
        raise HydrologyError("parallel_priority_weight cannot be negative")
    if not 0 < config.runoff_weight_floor <= 1:
        raise HydrologyError("runoff_weight_floor must be within (0, 1]")
    if config.snowmelt_runoff_bonus < 0:
        raise HydrologyError("snowmelt_runoff_bonus cannot be negative")
    if not 0 <= config.headwater_upland_quantile <= 1:
        raise HydrologyError("headwater_upland_quantile must be within [0, 1]")
    if config.lowland_headwater_flow_multiplier < 1:
        raise HydrologyError(
            "lowland_headwater_flow_multiplier must be at least one"
        )
    if config.major_rivers_per_continent < 0:
        raise HydrologyError("major_rivers_per_continent cannot be negative")
    if config.major_river_tributaries < 0:
        raise HydrologyError("major_river_tributaries cannot be negative")
    if not 0 <= config.continent_minimum_land_fraction <= 1:
        raise HydrologyError(
            "continent_minimum_land_fraction must be within [0, 1]"
        )
    if config.major_river_minimum_span_factor < 0:
        raise HydrologyError("major_river_minimum_span_factor cannot be negative")
    if not 0 <= config.stream_threshold_fraction:
        raise HydrologyError("stream_threshold_fraction must be non-negative")
    if not (
        config.stream_threshold_fraction
        <= config.tributary_threshold_fraction
        <= config.mainstem_threshold_fraction
    ):
        raise HydrologyError(
            "stream, tributary, and mainstem fractions must be non-decreasing"
        )
    if config.minimum_headwater_length < 1:
        raise HydrologyError("minimum_headwater_length must be at least one")


def _prepare_lakes(
    lake_mask: np.ndarray,
    config: HydrologyConfig,
) -> tuple[
    list[list[Cell]],
    np.ndarray,
    dict[Cell, Cell],
    dict[int, tuple[Cell, Cell]],
]:
    """Label lakes and construct optional acyclic outlet parent links."""

    components = _component_cells(lake_mask)
    labels = np.full(lake_mask.shape, -1, dtype=np.int32)
    for component_index, cells in enumerate(components):
        for row, column in cells:
            labels[row, column] = component_index

    outlet_parents: dict[Cell, Cell] = {}
    configured_by_component: dict[int, tuple[Cell, Cell]] = {}
    for raw_anchor, raw_outlet in (config.lake_outlets or {}).items():
        try:
            anchor_values = tuple(raw_anchor)
        except (TypeError, ValueError) as error:
            raise HydrologyError(
                "lake_outlets keys must be stable anchor cells"
            ) from error
        if len(anchor_values) != 2:
            raise HydrologyError("lake_outlets keys must be stable anchor cells")
        try:
            anchor = (int(anchor_values[0]), int(anchor_values[1]))
        except (TypeError, ValueError) as error:
            raise HydrologyError(
                "lake_outlets keys must be stable anchor cells"
            ) from error
        if not (
            0 <= anchor[0] < lake_mask.shape[0]
            and 0 <= anchor[1] < lake_mask.shape[1]
            and lake_mask[anchor]
        ):
            raise HydrologyError(
                f"lake outlet anchor must be inside a lake: {anchor}"
            )
        component_index = int(labels[anchor])
        if component_index in configured_by_component:
            previous_anchor = configured_by_component[component_index][0]
            raise HydrologyError(
                "lake_outlets contains duplicate anchors for lake component: "
                f"{previous_anchor}, {anchor}"
            )
        try:
            outlet_values = tuple(raw_outlet)
        except (TypeError, ValueError) as error:
            raise HydrologyError(f"lake outlet {anchor} is not a cell") from error
        if len(outlet_values) != 2:
            raise HydrologyError(f"lake outlet {anchor} is not a cell")
        try:
            outlet = (int(outlet_values[0]), int(outlet_values[1]))
        except (TypeError, ValueError) as error:
            raise HydrologyError(f"lake outlet {anchor} is not a cell") from error
        if not (
            0 <= outlet[0] < lake_mask.shape[0]
            and 0 <= outlet[1] < lake_mask.shape[1]
        ):
            raise HydrologyError(
                f"lake outlet {anchor} is outside the raster: {outlet}"
            )
        if labels[outlet] != component_index:
            raise HydrologyError(
                f"lake outlet {anchor} must be inside its lake component"
            )
        configured_by_component[component_index] = (anchor, outlet)

    for component_index, cells in enumerate(components):
        configured = configured_by_component.get(component_index)
        if configured is None:
            continue
        outlet_cell = configured[1]
        queue: deque[Cell] = deque([outlet_cell])
        visited = {outlet_cell}
        while queue:
            current = queue.popleft()
            for _, neighbour in _neighbours(current, lake_mask.shape):
                if (
                    neighbour in visited
                    or labels[neighbour] != component_index
                ):
                    continue
                visited.add(neighbour)
                outlet_parents[neighbour] = current
                queue.append(neighbour)
        if len(visited) != len(cells):  # pragma: no cover - labels are components
            raise HydrologyError(f"lake component {component_index} could not reach outlet")
    return components, labels, outlet_parents, configured_by_component


def _touches_ocean(
    cell: Cell,
    ocean_mask: np.ndarray,
) -> bool:
    """Whether a terrain cell sees an explicitly classified ocean cell.

    Raster edges are intentionally not implicit outlets.  A caller that wants
    an edge to be coastal must mark its adjacent ocean cells in ``ocean_mask``;
    clipped/nodata cells can remain inactive without becoming ocean.
    """

    for _, neighbour in _neighbours(cell, ocean_mask.shape):
        row, column = neighbour
        if ocean_mask[row, column]:
            return True
    return False


def _lake_neighbours(cell: Cell, lake_mask: np.ndarray) -> list[tuple[int, Cell]]:
    return [
        (direction, neighbour)
        for direction, neighbour in _neighbours(cell, lake_mask.shape)
        if lake_mask[neighbour]
    ]


def _choose_lake_neighbour(
    cell: Cell,
    lake_mask: np.ndarray,
    lake_labels: np.ndarray,
    lake_outlet_parents: Mapping[Cell, Cell],
    lake_outlet_config: Mapping[int, tuple[Cell, Cell]],
) -> Cell | None:
    """Choose a deterministic neighbouring lake cell for an outlet seed."""

    candidates = _lake_neighbours(cell, lake_mask)
    if not candidates:
        return None
    component_index = int(lake_labels[candidates[0][1]])
    configured = lake_outlet_config.get(component_index)
    if configured is None:
        return min(candidates, key=lambda item: item[0])[1]
    outlet_cell = configured[1]
    # Distance to the configured outlet is monotone along the BFS parent tree;
    # the row/column and direction keys resolve equal-distance ties.
    def distance(candidate: Cell) -> int:
        current = candidate
        distance_value = 0
        while current != outlet_cell:
            current = lake_outlet_parents.get(current, outlet_cell)
            distance_value += 1
            if distance_value > lake_mask.size:
                break
        return distance_value

    return min(candidates, key=lambda item: (distance(item[1]), item[0]))[1]


def _relative_class(
    accumulation: float,
    land_count: int,
    config: HydrologyConfig,
) -> str:
    fraction = accumulation / max(1, land_count)
    if fraction >= config.mainstem_threshold_fraction:
        return "mainstem"
    if fraction >= config.tributary_threshold_fraction:
        return "tributary"
    return "headwater"


def _ground_distance(
    source: Cell,
    target: Cell,
    latitude_degrees: np.ndarray | None,
) -> float:
    """Return a relative ground distance for one D8 step.

    The raster is authored in a plate-carree board, so a column step is not
    the same physical distance at every latitude.  Callers that do not have
    geographic row coordinates retain the unit-grid distance used by the
    synthetic/core API.  The small cosine floor avoids a singular pole while
    keeping the direction deterministic there.
    """

    row_delta = abs(target[0] - source[0])
    column_delta = abs(target[1] - source[1])
    if latitude_degrees is None or column_delta == 0:
        horizontal = float(column_delta)
    else:
        source_latitude = float(latitude_degrees[source[0]])
        target_latitude = float(latitude_degrees[target[0]])
        mean_latitude = math.radians(0.5 * (source_latitude + target_latitude))
        horizontal = float(column_delta) * max(0.05, abs(math.cos(mean_latitude)))
    return math.hypot(float(row_delta), horizontal)


def _masked_box_mean(
    values: np.ndarray,
    valid: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Compute a deterministic local mean without a SciPy dependency."""

    if radius < 1:
        return values.copy()
    weighted = np.where(valid, values, 0.0).astype(np.float64, copy=False)
    weights = valid.astype(np.float64, copy=False)
    padded_weighted = np.pad(
        weighted,
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=0.0,
    )
    padded_weights = np.pad(
        weights,
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=0.0,
    )
    size = 2 * radius + 1

    def integral(array: np.ndarray) -> np.ndarray:
        return np.pad(
            array.cumsum(axis=0).cumsum(axis=1),
            ((1, 0), (1, 0)),
            mode="constant",
            constant_values=0.0,
        )

    weighted_integral = integral(padded_weighted)
    weight_integral = integral(padded_weights)
    weighted_sum = (
        weighted_integral[size:, size:]
        - weighted_integral[:-size, size:]
        - weighted_integral[size:, :-size]
        + weighted_integral[:-size, :-size]
    )
    weight_sum = (
        weight_integral[size:, size:]
        - weight_integral[:-size, size:]
        - weight_integral[size:, :-size]
        + weight_integral[:-size, :-size]
    )
    result = np.zeros_like(values, dtype=np.float64)
    np.divide(weighted_sum, weight_sum, out=result, where=weight_sum > 0)
    result[weight_sum <= 0] = values[weight_sum <= 0]
    return result


def _valley_score(
    elevation: np.ndarray,
    terrain: np.ndarray,
    tie_epsilon: float,
    window_fraction: float,
    depth_fraction: float,
) -> tuple[np.ndarray, int]:
    """Measure concavity on an automatically scaled, non-metric valley field.

    A cell is valley-supported when it is lower than the surrounding local
    mean.  The window is a small fraction of the raster's short dimension,
    which keeps the test and production behavior scale-aware without tying it
    to a particular map or pixel coordinate.  The score is normalized by the
    90th percentile of positive depths so authored-relative elevation units do
    not become a hidden absolute threshold.
    """

    short_dimension = min(elevation.shape)
    radius = max(1, int(round(short_dimension * window_fraction)))
    local_mean = _masked_box_mean(elevation, terrain, radius)
    depth = np.maximum(0.0, local_mean - elevation)
    depth[~terrain] = 0.0
    # Compare depth with the local sampled relief rather than the global
    # range.  A large mountain should not make a genuine valley disappear,
    # while tiny colour/noise ripples on a broad plane should not become
    # valley support merely because the positive depths were normalized.
    neighbour_differences: list[np.ndarray] = []
    for row_delta, column_delta in ((1, 0), (0, 1)):
        source = elevation[:-row_delta or None, :-column_delta or None]
        target = elevation[row_delta:, column_delta:]
        source_valid = terrain[:-row_delta or None, :-column_delta or None]
        target_valid = terrain[row_delta:, column_delta:]
        valid_differences = np.abs(source - target)[source_valid & target_valid]
        if valid_differences.size:
            neighbour_differences.append(valid_differences)
    if neighbour_differences:
        local_relief = float(
            np.quantile(np.concatenate(neighbour_differences), 0.9)
        )
    else:
        local_relief = 0.0
    minimum_depth = max(tie_epsilon, local_relief * depth_fraction)
    depth[depth < minimum_depth] = 0.0
    positive = depth[terrain & (depth > tie_epsilon)]
    score = np.zeros_like(elevation, dtype=np.float64)
    if positive.size == 0:
        return score, radius
    scale = float(np.quantile(positive, 0.9))
    if not math.isfinite(scale) or scale <= tie_epsilon:
        scale = float(np.max(positive))
    if scale > tie_epsilon:
        score[terrain] = np.clip(depth[terrain] / scale, 0.0, 1.0)
    return score, radius


def _propagate_valley_support(
    valley_score: np.ndarray,
    downstream_index: np.ndarray,
    topological: Sequence[int],
    maximum_distance: int,
    support_threshold: float,
) -> np.ndarray:
    """Allow tributaries to reach a supported valley within a local window."""

    supported = valley_score >= support_threshold
    distance = np.full(valley_score.shape, -1, dtype=np.int32)
    distance[supported] = 0
    width = valley_score.shape[1]
    for index in reversed(topological):
        index = int(index)
        target = int(downstream_index.flat[index])
        if target < 0:
            continue
        target_distance = int(distance.flat[target])
        if 0 <= target_distance < maximum_distance:
            distance.flat[index] = target_distance + 1
            supported.flat[index] = True
    return supported


def _flat_routing_distance(
    corrected_elevation: np.ndarray,
    terrain: np.ndarray,
    ocean_mask: np.ndarray,
    lake_mask: np.ndarray,
    seed_kind: np.ndarray,
    tie_epsilon: float,
) -> np.ndarray:
    """Assign a cardinal distance-to-spill value to each corrected flat.

    Priority-Flood raises closed depressions to their spill elevation.  The
    resulting plateau has no physical gradient, so following the heap's
    parent pointer makes its route depend on insertion order and can produce
    long diagonal combs.  A multi-source, weighted 8-neighbour distance
    supplies an explicit, acyclic pseudo-gradient toward a real lower cell,
    coast, lake, or inland-sink seed.  Cardinal moves cost 10 and diagonals
    cost 14, so a diagonal is considered only when it is genuinely needed to
    cross an 8-connected narrow land bridge.
    """

    distance = np.full(terrain.shape, -1, dtype=np.int32)
    seeds: list[Cell] = []
    height, width = terrain.shape
    cardinal_directions = (0, 2, 4, 6)
    for row, column in zip(*np.where(terrain), strict=True):
        cell = (int(row), int(column))
        has_lower = False
        for _, neighbour in _neighbours(cell, terrain.shape):
            if terrain[neighbour] and (
                float(corrected_elevation[cell])
                - float(corrected_elevation[neighbour])
                > tie_epsilon
            ):
                has_lower = True
                break
        if (
            has_lower
            or bool(seed_kind[cell])
            or _touches_ocean(cell, ocean_mask)
            or bool(_lake_neighbours(cell, lake_mask))
        ):
            distance[cell] = 0
            seeds.append(cell)

    queue: list[tuple[int, int, int]] = [
        (0, cell[0], cell[1]) for cell in sorted(seeds)
    ]
    heapq.heapify(queue)
    while queue:
        queued_distance, row, column = heapq.heappop(queue)
        current = (row, column)
        current_distance = int(distance[current])
        if queued_distance != current_distance:
            continue
        current_elevation = float(corrected_elevation[current])
        for direction in range(len(D8_OFFSETS)):
            row_delta, column_delta = D8_OFFSETS[direction]
            neighbour = (
                current[0] + row_delta,
                current[1] + column_delta,
            )
            if not (
                0 <= neighbour[0] < height
                and 0 <= neighbour[1] < width
                and terrain[neighbour]
                and abs(float(corrected_elevation[neighbour]) - current_elevation)
                <= tie_epsilon
            ):
                continue
            step_cost = 10 if direction in cardinal_directions else 14
            next_distance = current_distance + step_cost
            if distance[neighbour] >= 0 and next_distance >= int(distance[neighbour]):
                continue
            distance[neighbour] = next_distance
            heapq.heappush(queue, (next_distance, neighbour[0], neighbour[1]))
    return distance


def _unresolved_alluvial_fields(
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic low-frequency relief for featureless flats.

    The painted source DEM has broad single-colour lowlands where the resolved
    elevation is exactly constant for tens of native cells.  A shortest-path
    spill route across such a surface produces ruler-straight rivers even
    though real floodplains contain bars, abandoned channels and shallow
    depressions below the map's elevation resolution.  This field represents
    only that unresolved ordering signal: it is consulted after ridge, raw
    ascent, valley and authored-channel evidence, and never changes the public
    elevation surface or permits uphill flow.

    Several incommensurate waves avoid a repeated grid or checkerboard motif.
    Their wavelengths scale with the raster short side so the rule remains
    useful in synthetic tests and at the native 2176-by-1088 atlas resolution.
    """

    height, width = shape
    rows, columns = np.indices(shape, dtype=np.float64)
    base = max(5.0, min(height, width) * 0.018)
    tau = 2.0 * math.pi
    signal = (
        0.52 * np.sin(tau * (columns + 0.41 * rows) / base + 0.37)
        + 0.31
        * np.sin(tau * (-0.36 * columns + rows) / (base * 1.91) + 1.73)
        + 0.17
        * np.sin(tau * (0.68 * columns + 0.73 * rows) / (base * 3.47) + 2.41)
    )
    minimum = float(np.min(signal))
    span = float(np.max(signal) - minimum)
    relief = (
        np.zeros(shape, dtype=np.float64)
        if span <= np.finfo(np.float64).eps
        else (signal - minimum) / span
    )
    orientation = math.pi * (
        0.5
        + 0.32
        * np.sin(tau * (columns - 0.23 * rows) / (base * 2.27) + 0.91)
        + 0.18
        * np.sin(tau * (0.31 * columns + rows) / (base * 4.13) + 2.07)
    )
    return relief, np.cos(orientation), np.sin(orientation)


def _alluvial_step_cost(
    current: Cell,
    neighbour: Cell,
    relief: np.ndarray,
    axis_east: np.ndarray,
    axis_south: np.ndarray,
) -> int:
    """Cost one flat edge against shallow, slowly rotating swale axes."""

    row_delta = neighbour[0] - current[0]
    column_delta = neighbour[1] - current[1]
    distance = math.hypot(row_delta, column_delta)
    # Use an intermediate D8 premium.  The former 2.4x diagonal cost pinned
    # long north/south or east/west routes even inside a broad flat valley;
    # the rotating swale orientation below now carries part of the burden of
    # preventing repeated 45-degree combs.
    base_cost = 100 if distance <= 1.0 else 205
    east = 0.5 * (float(axis_east[current]) + float(axis_east[neighbour]))
    south = 0.5 * (float(axis_south[current]) + float(axis_south[neighbour]))
    axis_length = max(math.hypot(east, south), 1.0e-12)
    alignment = abs(
        (column_delta * east + row_delta * south) / (distance * axis_length)
    )
    orientation_cost = int(round(260.0 * (1.0 - min(1.0, alignment)) ** 2))
    relief_cost = int(
        round(
            110.0
            * 0.5
            * (float(relief[current]) + float(relief[neighbour]))
        )
    )
    return base_cost + orientation_cost + relief_cost


def _flat_routing_cost(
    corrected_elevation: np.ndarray,
    raw_elevation: np.ndarray,
    terrain: np.ndarray,
    ocean_mask: np.ndarray,
    lake_mask: np.ndarray,
    seed_kind: np.ndarray,
    valley_score: np.ndarray,
    constraint_strength: np.ndarray,
    unresolved_relief: np.ndarray,
    unresolved_axis_east: np.ndarray,
    unresolved_axis_south: np.ndarray,
    tie_epsilon: float,
    raw_uphill_tolerance: float,
    support_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a lexicographic cost-to-spill field for corrected flats.

    ``_flat_routing_distance`` is intentionally retained as the geometric
    field used by MFD.  It is not sufficient as a D8 direction field,
    however: the nearest spill can be reached by climbing an authored raw
    ridge while a slightly longer equal-height path follows a raw descent.
    This reverse Dijkstra field makes that choice explicit.  Its ordered
    cost is, in order, unsupported positive raw ascent count, total positive
    raw ascent, unsupported-edge count, and a positive travel distance through
    unresolved alluvial relief.  The relief term breaks ruler-straight routes
    only where every higher-priority physical signal is tied.
    Every edge has a positive final step cost, so following an equal-height
    route strictly decreases this field and cannot introduce a cycle.

    The first two terms are deliberately about the *raw* surface while the
    graph itself remains the corrected surface.  Valley and authored river
    evidence only remove an edge's unsupported penalty; they never create a
    route through an ocean or lake cell.
    """

    shape = terrain.shape
    unsupported_count = np.full(shape, np.iinfo(np.int32).max, dtype=np.int32)
    raw_ascent = np.full(shape, np.inf, dtype=np.float64)
    unsupported_edge_count = np.full(
        shape,
        np.iinfo(np.int32).max,
        dtype=np.int32,
    )
    path_distance = np.full(shape, np.iinfo(np.int32).max, dtype=np.int32)
    seeds: list[Cell] = []
    for row, column in zip(*np.where(terrain), strict=True):
        cell = (int(row), int(column))
        has_lower = False
        for _, neighbour in _neighbours(cell, shape):
            if terrain[neighbour] and (
                float(corrected_elevation[cell])
                - float(corrected_elevation[neighbour])
                > tie_epsilon
            ):
                has_lower = True
                break
        if (
            has_lower
            or bool(seed_kind[cell])
            or _touches_ocean(cell, ocean_mask)
            or bool(_lake_neighbours(cell, lake_mask))
        ):
            unsupported_count[cell] = 0
            raw_ascent[cell] = 0.0
            unsupported_edge_count[cell] = 0
            path_distance[cell] = 0
            seeds.append(cell)

    # The heap stores the complete ordered cost followed by row/column for a
    # stable tie break.  Raw ascent is a sum of positive gaps; all gaps below
    # the routing epsilon are treated as authored ties.
    queue: list[tuple[int, float, int, int, int, int]] = [
        (0, 0.0, 0, 0, cell[0], cell[1]) for cell in sorted(seeds)
    ]
    heapq.heapify(queue)
    while queue:
        (
            queued_unsupported,
            queued_raw_ascent,
            queued_unsupported_edges,
            queued_distance,
            row,
            column,
        ) = heapq.heappop(queue)
        current = (row, column)
        current_cost = (
            int(unsupported_count[current]),
            float(raw_ascent[current]),
            int(unsupported_edge_count[current]),
            int(path_distance[current]),
        )
        queued_cost = (
            queued_unsupported,
            queued_raw_ascent,
            queued_unsupported_edges,
            queued_distance,
        )
        if queued_cost != current_cost:
            continue
        current_elevation = float(corrected_elevation[current])
        for direction, neighbour in _neighbours(current, shape):
            if not terrain[neighbour] or abs(
                float(corrected_elevation[neighbour]) - current_elevation
            ) > tie_epsilon:
                continue

            # Reverse traversal: the eventual route goes neighbour -> current.
            raw_gap = max(
                0.0,
                float(raw_elevation[current]) - float(raw_elevation[neighbour]),
            )
            authored_supported = max(
                float(constraint_strength[current]),
                float(constraint_strength[neighbour]),
            ) > tie_epsilon
            raw_tie_tolerance = (
                raw_uphill_tolerance * 0.25
                if authored_supported
                else tie_epsilon
            )
            if raw_gap <= raw_tie_tolerance:
                raw_gap = 0.0
            supported = (
                max(float(valley_score[current]), float(valley_score[neighbour]))
                >= support_threshold
                or authored_supported
            )
            edge_unsupported = int(
                raw_gap > raw_uphill_tolerance and not supported
            )
            edge_support_gap = int(not supported)
            step_cost = _alluvial_step_cost(
                current,
                neighbour,
                unresolved_relief,
                unresolved_axis_east,
                unresolved_axis_south,
            )
            candidate = (
                queued_unsupported + edge_unsupported,
                queued_raw_ascent + raw_gap,
                queued_unsupported_edges + edge_support_gap,
                queued_distance + step_cost,
            )
            existing = (
                int(unsupported_count[neighbour]),
                float(raw_ascent[neighbour]),
                int(unsupported_edge_count[neighbour]),
                int(path_distance[neighbour]),
            )
            if candidate >= existing:
                continue
            unsupported_count[neighbour] = candidate[0]
            raw_ascent[neighbour] = candidate[1]
            unsupported_edge_count[neighbour] = candidate[2]
            path_distance[neighbour] = candidate[3]
            heapq.heappush(
                queue,
                (*candidate, neighbour[0], neighbour[1]),
            )

    return (
        unsupported_count,
        raw_ascent,
        unsupported_edge_count,
        path_distance,
    )


def _stream_incoming(
    stream_mask: np.ndarray,
    downstream_index: np.ndarray,
) -> np.ndarray:
    """Count stream predecessors using one row-major pass."""

    incoming = np.zeros(stream_mask.shape, dtype=np.int16)
    width = stream_mask.shape[1]
    for index in np.flatnonzero(stream_mask.ravel()):
        target = int(downstream_index.flat[int(index)])
        if target < 0:
            continue
        target_row, target_column = divmod(target, width)
        if stream_mask[target_row, target_column]:
            incoming[target_row, target_column] += 1
    return incoming


def _runoff_source_weights(
    elevation: np.ndarray,
    terrain: np.ndarray,
    perennial_snow: np.ndarray,
    runoff_floor: float,
    snowmelt_bonus: float,
    runoff_source_yield: np.ndarray | None = None,
) -> np.ndarray:
    """Return normalized potential runoff supplied by every land cell.

    Upper terrain receives progressively more potential yield and perennial
    snow adds a meltwater contribution.  When supplied, the deterministic
    climate yield reduces dry-season contribution and concentrates perennial
    channels in wetter catchments.  Normalizing the terrain mean to one keeps
    area-fraction thresholds comparable across worlds and resolutions.
    """

    weights = np.zeros(elevation.shape, dtype=np.float64)
    terrain_values = np.asarray(elevation[terrain], dtype=np.float64)
    if terrain_values.size == 0:
        return weights
    lower = float(np.quantile(terrain_values, 0.10))
    upper = float(np.quantile(terrain_values, 0.90))
    if upper - lower <= 1.0e-12:
        upland_position = np.zeros(elevation.shape, dtype=np.float64)
    else:
        upland_position = np.clip((elevation - lower) / (upper - lower), 0.0, 1.0)
    weights[terrain] = runoff_floor + (1.0 - runoff_floor) * (
        upland_position[terrain] ** 1.5
    )
    if runoff_source_yield is not None:
        weights[terrain] *= runoff_source_yield[terrain]
    weights[perennial_snow & terrain] += snowmelt_bonus
    total = float(weights[terrain].sum())
    if total <= 0.0:  # pragma: no cover - guarded by positive runoff floor
        raise HydrologyError("runoff source weights have no positive terrain supply")
    weights[terrain] *= float(np.count_nonzero(terrain)) / total
    return weights


def _multiple_flow_accumulation(
    corrected_elevation: np.ndarray,
    terrain: np.ndarray,
    downstream_index: np.ndarray,
    flat_distance: np.ndarray,
    latitude_degrees: np.ndarray | None,
    tie_epsilon: float,
    exponent: float,
    source_weight: np.ndarray,
) -> np.ndarray:
    """Accumulate fractional discharge over every valid downslope neighbour.

    D8 remains the authoritative directed graph, but using its single target
    for stream *selection* gives every broad planar row the same full catchment
    and recreates a comb.  MFD distributes a cell's unit discharge according
    to physical slope, with D8's chosen target as the deterministic fallback
    for flats and lake sinks.  It is computed from the same corrected surface,
    masks, and latitude contract as the published D8 field.
    """

    size = int(terrain.size)
    width = terrain.shape[1]
    edge_targets: list[list[tuple[int, float]]] = [[] for _ in range(size)]
    node_mask = terrain.copy()

    # Build the complete MFD edge set first.  The D8 topological order is not
    # sufficient here: a cell can send flow to a lower neighbour that was not
    # its single D8 target.  Strictly lower edges and flat-distance-decreasing
    # edges form a DAG; lake outlet parents reuse the already acyclic D8 tree.
    for index in range(size):
        if not terrain.flat[index] and int(downstream_index.flat[index]) < 0:
            continue
        if terrain.flat[index]:
            row, column = divmod(index, width)
            source = (row, column)
            current_elevation = float(corrected_elevation[source])
            targets: list[int] = []
            weights: list[float] = []
            for _, neighbour in _neighbours(source, terrain.shape):
                if not terrain[neighbour]:
                    continue
                drop = current_elevation - float(corrected_elevation[neighbour])
                if drop <= tie_epsilon:
                    continue
                distance = _ground_distance(source, neighbour, latitude_degrees)
                slope = drop / max(distance, tie_epsilon)
                targets.append(neighbour[0] * width + neighbour[1])
                weights.append(slope**exponent)
            if targets:
                weight_sum = float(sum(weights))
                edge_targets[index] = [
                    (target, weight / weight_sum)
                    for target, weight in zip(targets, weights)
                ]
            else:
                # Priority-Flood plateaus have no strict downslope neighbour.
                # Share discharge over every lower-distance equal-height
                # neighbour instead of sending it through one D8 tie parent.
                current_distance = int(flat_distance.flat[index])
                equal_targets: list[int] = []
                cardinal_targets: list[int] = []
                if current_distance >= 0:
                    for direction, neighbour in _neighbours(source, terrain.shape):
                        if not terrain[neighbour] or (
                            abs(
                                float(corrected_elevation[neighbour])
                                - current_elevation
                            )
                            > tie_epsilon
                        ):
                            continue
                        neighbour_index = neighbour[0] * width + neighbour[1]
                        if int(flat_distance.flat[neighbour_index]) >= current_distance:
                            continue
                        equal_targets.append(neighbour_index)
                        if direction % 2 == 0:
                            cardinal_targets.append(neighbour_index)
                if cardinal_targets:
                    equal_targets = cardinal_targets
                if equal_targets:
                    share = 1.0 / len(equal_targets)
                    edge_targets[index] = [
                        (target, share) for target in equal_targets
                    ]
        if not edge_targets[index]:
            # Lake cells and corrected-flat terminals are represented by the
            # D8 target, so optional lake outlet trees also receive discharge.
            target = int(downstream_index.flat[index])
            if target >= 0:
                target_row, target_column = divmod(target, width)
                current_elevation = float(corrected_elevation.ravel()[index])
                target_elevation = float(corrected_elevation[target_row, target_column])
                target_is_lake = not bool(terrain[target_row, target_column])
                target_is_downhill = target_elevation < current_elevation - tie_epsilon
                target_is_flat_gradient = (
                    abs(target_elevation - current_elevation) <= tie_epsilon
                    and int(flat_distance.flat[target]) < int(flat_distance.flat[index])
                )
                # A D8 fallback on a terrain plateau is only admissible when
                # it still follows the corrected flat-distance gradient.  The
                # previous unconditional fallback could point two near-equal
                # terrain cells at one another on a fine raster, creating an
                # MFD cycle even though the D8 graph itself was repaired.
                if target_is_lake or target_is_downhill or target_is_flat_gradient:
                    edge_targets[index] = [(target, 1.0)]
        for target, _ in edge_targets[index]:
            node_mask.flat[target] = True

    def topological_order() -> tuple[np.ndarray, list[int]]:
        indegree = np.zeros(size, dtype=np.int64)
        for node in np.flatnonzero(node_mask.ravel()):
            for target, _ in edge_targets[int(node)]:
                indegree[target] += 1
        ready = [
            int(node)
            for node in np.flatnonzero(node_mask.ravel())
            if indegree[int(node)] == 0
        ]
        heapq.heapify(ready)
        order: list[int] = []
        while ready:
            node = heapq.heappop(ready)
            order.append(node)
            for target, _ in edge_targets[node]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(ready, target)
        return indegree, order

    indegree, mfd_topological = topological_order()
    node_count = int(np.count_nonzero(node_mask))
    if len(mfd_topological) != node_count:
        # A very fine, exactly flat plateau can make a custom fractional edge
        # set mutually referential despite the D8 graph being repaired.  Cut
        # one lowest node per residual cycle, then rerun Kahn's pass.  Strict
        # downslope and flat-distance edges remain untouched in the normal
        # case.  The residual is tiny compared with the raster, so a bounded
        # DFS is preferable to repeatedly walking all million cells.
        residual = {
            int(index)
            for index in np.flatnonzero(node_mask.ravel() & (indegree > 0))
        }
        cycle_breaks: set[int] = set()
        state: dict[int, int] = {}
        stack: list[int] = []
        positions: dict[int, int] = {}

        def visit(node: int) -> None:
            state[node] = 1
            positions[node] = len(stack)
            stack.append(node)
            for target, _ in edge_targets[node]:
                target_index = int(target)
                if target_index not in residual:
                    continue
                target_state = state.get(target_index, 0)
                if target_state == 0:
                    visit(target_index)
                elif target_state == 1:
                    cycle = stack[positions[target_index] :]
                    cycle_breaks.add(
                        min(
                            cycle,
                            key=lambda index: (
                                float(corrected_elevation.ravel()[index]),
                                index,
                            ),
                        )
                    )
            stack.pop()
            positions.pop(node, None)
            state[node] = 2

        for node in sorted(residual):
            if state.get(node, 0) == 0:
                visit(node)
        for break_index in sorted(cycle_breaks):
            edge_targets[break_index] = []
        indegree, mfd_topological = topological_order()
        if len(mfd_topological) != node_count:
            # A multi-edge strongly connected plateau can contain a cycle
            # that is not exposed by the first DFS back-edge.  Residual nodes
            # are precisely the still-unordered SCC remainder; ending their
            # fractional shares is bounded, deterministic, and keeps all
            # published flow edges acyclic.
            for residual_index in residual:
                edge_targets[residual_index] = []
            indegree, mfd_topological = topological_order()
        if len(mfd_topological) != node_count:
            raise HydrologyError(
                "MFD flow graph contains a cycle after repair "
                f"({node_count - len(mfd_topological)} residual nodes)"
            )

    result = np.asarray(source_weight, dtype=np.float64).copy()
    for index in mfd_topological:
        for target, fraction in edge_targets[index]:
            result.flat[target] += result.flat[index] * fraction
    return result


def _strahler_order(
    stream_mask: np.ndarray,
    downstream_index: np.ndarray,
    topological: Sequence[int],
) -> np.ndarray:
    """Accumulate tributary orders without one Python list per terrain cell.

    D8 gives each cell one outgoing edge. Processing that edge in topological
    order needs only the greatest incoming order and its multiplicity, cutting
    millions of empty predecessor containers from the full-resolution build.
    """
    order = np.zeros(stream_mask.shape, dtype=np.int16)
    highest = np.zeros(stream_mask.size, dtype=np.int16)
    equal_highest = np.zeros(stream_mask.size, dtype=np.uint8)
    selected = stream_mask.ravel()
    downstream = downstream_index.ravel()
    values = order.ravel()
    for index in topological:
        if not selected[index]:
            continue
        value = max(1, int(highest[index]) + int(equal_highest[index] >= 2))
        values[index] = value
        target = int(downstream[index])
        if target < 0 or not selected[target]:
            continue
        if value > highest[target]:
            highest[target] = value
            equal_highest[target] = 1
        elif value == highest[target]:
            equal_highest[target] += 1

    return order


def _suppress_parallel_streams(
    stream_mask: np.ndarray,
    flow_direction: np.ndarray,
    downstream_index: np.ndarray,
    accumulation: np.ndarray,
    valley_score: np.ndarray,
    constraint_strength: np.ndarray,
    outlet_index: np.ndarray,
    valley_radius: int,
    search_radius_factor: float,
    priority_weight: float,
) -> np.ndarray:
    """Keep one supported corridor when adjacent D8 channels run in parallel.

    The operation is a deterministic, local non-maximum suppression.  Valley
    support and finite authored evidence contribute to corridor priority; they
    are not absolute exemptions because a broad valley or a thin authored
    stroke can otherwise select one parallel copy per row.  Candidate cells
    must still drain to a nearby terminal outlet, so distinct basins are not
    collapsed merely because their local slopes are parallel.  A later
    flow-path closure reconnects any one-cell gaps.
    """

    result = stream_mask.copy()
    incoming = _stream_incoming(stream_mask, downstream_index)
    # A half-window is useful for small synthetic rasters, but it is too short
    # to see the adjacent corridors produced by a broad sampled valley.  Keep
    # the configured factor effective while ensuring the derived valley window
    # is the minimum comparison horizon.
    radius = max(
        1,
        int(round(valley_radius * max(1.0, search_radius_factor))),
    )
    maximum_accumulation = max(1.0, float(np.max(accumulation, initial=1.0)))
    log_scale = math.log1p(maximum_accumulation)
    shape = stream_mask.shape
    # Protect only the immediate junction neighbourhood, not a valley-sized
    # section of every incoming branch.  The former radius scaled with
    # ``valley_radius`` and consequently exempted whole short tributaries from
    # competition, producing the characteristic parallel comb on rolling
    # slopes.  Two cells retain the distinct valley mouths and their turns;
    # the longer, still-parallel upstream portion has to win on evidence.
    confluence_buffer = 2
    distance_to_confluence = np.full(shape, -1, dtype=np.int32)
    for source_index in np.flatnonzero(stream_mask.ravel()):
        source_index = int(source_index)
        source = divmod(source_index, shape[1])
        current = source
        distance = 0
        seen: set[Cell] = set()
        while current not in seen and stream_mask[current]:
            seen.add(current)
            if incoming[current] >= 2:
                distance_to_confluence[source] = distance
                break
            target = int(downstream_index[current])
            if target < 0:
                break
            target_cell = divmod(target, shape[1])
            if not stream_mask[target_cell]:
                break
            current = target_cell
            distance += 1

    def priority(cell: Cell) -> float:
        row, column = cell
        flow_component = math.log1p(float(accumulation[row, column])) / log_scale
        return (
            float(valley_score[row, column])
            + float(constraint_strength[row, column])
            + priority_weight * flow_component
        )

    ordered_indices = [int(index) for index in np.flatnonzero(stream_mask.ravel())]
    # Two opposite sweeps reduce parallel cells on both sides of a sampled
    # junction.  ``incoming`` is intentionally held from the evidence mask:
    # once a real D8 confluence is recognized, pruning one of its tributaries
    # must not erase the junction merely because row-major processing visited
    # that tributary first.
    for index in (*ordered_indices, *reversed(ordered_indices)):
        index = int(index)
        row, column = divmod(index, shape[1])
        cell = (row, column)
        if not result[cell]:
            continue
        if incoming[cell] >= 2:
            continue
        if (
            0 <= distance_to_confluence[cell] <= confluence_buffer
        ):
            # Keep a short terminal section on every true incoming branch so
            # corridor thinning cannot turn a real confluence into a single
            # orphan line.  Longer upstream portions remain eligible for
            # competition with adjacent copies.
            continue
        direction = int(flow_direction[cell])
        if direction < 0:
            continue
        source_priority = priority(cell)
        basin = int(outlet_index[cell])
        basin_row, basin_column = divmod(basin, shape[1]) if basin >= 0 else (-1, -1)
        # The two D8 directions two steps clockwise/counter-clockwise are the
        # local cross-flow axes for this cell's direction.
        cross_directions = (
            (direction - 2) % len(D8_OFFSETS),
            (direction + 2) % len(D8_OFFSETS),
        )
        for step in range(1, radius + 1):
            competing: list[Cell] = []
            for cross_direction in cross_directions:
                row_delta, column_delta = D8_OFFSETS[cross_direction]
                candidate = (
                    row + step * row_delta,
                    column + step * column_delta,
                )
                if not (
                    0 <= candidate[0] < shape[0]
                    and 0 <= candidate[1] < shape[1]
                ):
                    continue
                if not result[candidate]:
                    continue
                candidate_basin = int(outlet_index[candidate])
                if candidate_basin < 0 or basin < 0:
                    continue
                candidate_basin_row, candidate_basin_column = divmod(
                    candidate_basin, shape[1]
                )
                if math.hypot(
                    candidate_basin_row - basin_row,
                    candidate_basin_column - basin_column,
                ) > radius:
                    # Nearby outlets along one coast can belong to different
                    # D8 terminal cells while still representing one sampled
                    # corridor.  Keep genuinely separated basins distinct.
                    continue
                candidate_direction = int(flow_direction[candidate])
                if candidate_direction < 0:
                    continue
                direction_vector = D8_OFFSETS[direction]
                candidate_vector = D8_OFFSETS[candidate_direction]
                direction_length = math.hypot(*direction_vector)
                candidate_length = math.hypot(*candidate_vector)
                alignment = (
                    direction_vector[0] * candidate_vector[0]
                    + direction_vector[1] * candidate_vector[1]
                ) / max(1.0e-12, direction_length * candidate_length)
                if alignment < 0.9:
                    # A tributary turning into this corridor is not a parallel
                    # copy; retain it so the confluence remains explicit.
                    continue
                if (
                    0 <= distance_to_confluence[candidate] <= confluence_buffer
                ):
                    continue
                competing.append(candidate)
            if not result[cell]:
                break
            for candidate in competing:
                candidate_priority = priority(candidate)
                if (
                    candidate_priority > source_priority + 1.0e-12
                    or (
                        abs(candidate_priority - source_priority) <= 1.0e-12
                        and candidate < cell
                    )
                ):
                    result[cell] = False
                    break
            if not result[cell]:
                break
    return result


def _close_stream_paths(
    stream_mask: np.ndarray,
    downstream_index: np.ndarray,
    terrain: np.ndarray,
    maximum_added_cells: int,
    maximum_chain_length: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Reconnect selected cells within an explicit normalized cell budget.

    The closure follows only the existing D8 graph and never invents a
    lateral route.  A candidate chain is added atomically; if it would exceed
    the remaining budget, it is left unselected so an interior truncation is
    not published as an ocean/lake outlet.
    """

    maximum_added_cells = max(0, int(maximum_added_cells))
    chain_limit = max(0, int(maximum_chain_length))

    result = stream_mask.copy()
    width = stream_mask.shape[1]
    before_count = int(np.count_nonzero(stream_mask))
    added_count = 0
    longest_chain_length = 0
    over_budget_chain_count = 0
    for index in np.flatnonzero(stream_mask.ravel()):
        row, column = divmod(int(index), width)
        current = (row, column)
        seen: set[Cell] = set()
        chain: list[Cell] = []
        while True:
            target = int(downstream_index[current])
            if target < 0:
                break
            target_cell = divmod(target, width)
            if not terrain[target_cell] or result[target_cell] or target_cell in seen:
                break
            seen.add(target_cell)
            chain.append(target_cell)
            current = target_cell
        if not chain:
            continue
        if (
            chain_limit == 0
            or len(chain) > chain_limit
            or added_count + len(chain) > maximum_added_cells
        ):
            over_budget_chain_count += 1
            continue
        for cell in chain:
            result[cell] = True
        added_count += len(chain)
        longest_chain_length = max(longest_chain_length, len(chain))
    # A skipped chain must not leave its selected upstream cell claiming the
    # eventual D8 outlet through an unselected interior endpoint.  Remove
    # dangling selected cells backwards until every retained stream cell either
    # reaches another retained stream cell or a real non-terrain terminal.
    discarded_count = 0
    while True:
        remove: list[Cell] = []
        for index in np.flatnonzero(result.ravel()):
            row, column = divmod(int(index), width)
            target = int(downstream_index[row, column])
            if target < 0:
                continue
            target_cell = divmod(target, width)
            if terrain[target_cell] and not result[target_cell]:
                remove.append((row, column))
        if not remove:
            break
        for cell in remove:
            result[cell] = False
        discarded_count += len(remove)
    return result, {
        "beforeCells": before_count,
        "budgetCells": maximum_added_cells,
        "maxAllowedChainLength": chain_limit,
        "addedCells": added_count,
        "maxChainLength": longest_chain_length,
        "addedRatio": added_count / max(1, before_count),
        "overBudgetChainCount": over_budget_chain_count,
        "discardedUnclosedCells": discarded_count,
    }


def _prune_short_headwaters(
    stream_mask: np.ndarray,
    downstream_index: np.ndarray,
    minimum_length: int,
) -> tuple[np.ndarray, int, int]:
    """Iteratively remove short source-to-junction stream branches.

    The minimum applies equally to branches ending at a lake.  A lake is a
    valid terminal, but it is not evidence that a one-cell source is a
    publishable reach; retaining that exception made the final network violate
    its own headwater contract after closure and compression.
    """

    result = stream_mask.copy()
    width = stream_mask.shape[1]
    removed_branches = 0
    removed_cells = 0
    while True:
        incoming = _stream_incoming(result, downstream_index)
        remove: set[Cell] = set()
        short_branch_count = 0
        for index in np.flatnonzero(result.ravel()):
            index = int(index)
            row, column = divmod(index, width)
            if incoming[row, column] != 0:
                continue
            path: list[Cell] = []
            current = (row, column)
            seen: set[Cell] = set()
            while current not in seen and result[current]:
                seen.add(current)
                path.append(current)
                target = int(downstream_index[current])
                if target < 0:
                    break
                target_cell = divmod(target, width)
                if not result[target_cell] or incoming[target_cell] != 1:
                    break
                current = target_cell
            if len(path) < minimum_length:
                remove.update(path)
                short_branch_count += 1
        if not remove:
            return result, removed_branches, removed_cells
        removed_branches += short_branch_count
        removed_cells += len(remove)
        for cell in remove:
            result[cell] = False


def _upslope_source_provenance(
    elevation: np.ndarray,
    perennial_snow: np.ndarray,
    terrain: np.ndarray,
    downstream_index: np.ndarray,
    topological: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate maximum source elevation and snow evidence downstream."""

    maximum_elevation = np.full(elevation.shape, -np.inf, dtype=np.float64)
    maximum_elevation[terrain] = elevation[terrain]
    contains_snow = np.asarray(perennial_snow & terrain, dtype=bool).copy()
    for index in topological:
        target = int(downstream_index.flat[int(index)])
        if target < 0:
            continue
        maximum_elevation.flat[target] = max(
            float(maximum_elevation.flat[target]),
            float(maximum_elevation.flat[int(index)]),
        )
        contains_snow.flat[target] = bool(
            contains_snow.flat[target] or contains_snow.flat[int(index)]
        )
    return maximum_elevation, contains_snow


def _prune_unqualified_headwaters(
    stream_mask: np.ndarray,
    downstream_index: np.ndarray,
    mfd_accumulation: np.ndarray,
    valley_supported: np.ndarray,
    upslope_maximum_elevation: np.ndarray,
    upslope_contains_snow: np.ndarray,
    upland_threshold: float,
    stream_threshold: np.ndarray,
    lowland_flow_multiplier: float,
    maximum_prunable_length: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove source branches lacking an upland, snow, or large-basin origin.

    Qualification is evaluated from the full upstream catchment, not from the
    displayed first channel cell.  A river may therefore emerge in a foothill
    after subsurface or diffuse hillslope flow, while a lowland source must
    carry substantially more accumulated runoff and occupy a supported valley.
    """

    result = stream_mask.copy()
    width = result.shape[1]
    initial_incoming = _stream_incoming(result, downstream_index)
    candidate_count = int(np.count_nonzero(result & (initial_incoming == 0)))
    removed_branches = 0
    removed_cells = 0

    def qualified(cell: Cell) -> bool:
        return bool(
            upslope_contains_snow[cell]
            or float(upslope_maximum_elevation[cell]) >= upland_threshold
            or (
                valley_supported[cell]
                and float(mfd_accumulation[cell])
                >= float(stream_threshold[cell]) * lowland_flow_multiplier
            )
        )

    while True:
        incoming = _stream_incoming(result, downstream_index)
        remove: set[Cell] = set()
        branch_count = 0
        for index in np.flatnonzero(result.ravel()):
            source = divmod(int(index), width)
            if incoming[source] != 0 or qualified(source):
                continue
            path: list[Cell] = []
            current = source
            seen: set[Cell] = set()
            while current not in seen and result[current]:
                seen.add(current)
                path.append(current)
                target = int(downstream_index[current])
                if target < 0:
                    break
                target_cell = divmod(target, width)
                if not result[target_cell] or incoming[target_cell] != 1:
                    break
                current = target_cell
            # Source provenance on the published D8 tree is intentionally a
            # conservative test.  A long established reach may be supplied by
            # diffuse rain or groundwater that this pre-climate model cannot
            # yet resolve, so only short comb-like teeth are removed here.
            if len(path) <= maximum_prunable_length:
                remove.update(path)
                branch_count += 1
        if not remove:
            break
        for cell in remove:
            result[cell] = False
        removed_branches += branch_count
        removed_cells += len(remove)

    final_incoming = _stream_incoming(result, downstream_index)
    remaining_sources = [
        divmod(int(index), width)
        for index in np.flatnonzero(result.ravel())
        if final_incoming.flat[int(index)] == 0
    ]
    return result, {
        "candidateHeadwaterCount": candidate_count,
        "prunedBranchCount": removed_branches,
        "prunedCellCount": removed_cells,
        "maximumPrunableLength": int(maximum_prunable_length),
        "remainingHeadwaterCount": len(remaining_sources),
        "uplandQualifiedCount": int(sum(
            float(upslope_maximum_elevation[cell]) >= upland_threshold
            for cell in remaining_sources
        )),
        "snowQualifiedCount": int(sum(
            bool(upslope_contains_snow[cell]) for cell in remaining_sources
        )),
        "largeLowlandQualifiedCount": int(sum(
            bool(
                float(upslope_maximum_elevation[cell]) < upland_threshold
                and not upslope_contains_snow[cell]
                and valley_supported[cell]
                and float(mfd_accumulation[cell])
                >= float(stream_threshold[cell]) * lowland_flow_multiplier
            )
            for cell in remaining_sources
        )),
    }


def _prune_competing_headwater_branches(
    stream_mask: np.ndarray,
    downstream_index: np.ndarray,
    outlet_index: np.ndarray,
    mfd_accumulation: np.ndarray,
    valley_score: np.ndarray,
    valley_radius: int,
    minimum_headwater_length: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Remove nearby parallel short branches that express one hillslope flow.

    Cell-level corridor suppression cannot distinguish a long river in a
    neighbouring valley from a short comb tooth next to a junction.  This pass
    works on complete source-to-confluence branches instead.  Only short
    branches in the same terminal basin compete, and only when both their
    source and confluence neighbourhoods are close and their net directions
    are almost parallel.  The winning branch is chosen by contributing flow,
    valley evidence, and length in that order.
    """

    result = stream_mask.copy()
    incoming = _stream_incoming(result, downstream_index)
    height, width = result.shape
    comparison_radius = max(2.0, float(valley_radius) * 1.5)
    junction_radius = max(3.0, float(valley_radius) * 2.0)
    maximum_branch_length = max(
        int(minimum_headwater_length) * 4,
        int(round(valley_radius * 2.0)),
    )

    branches: list[dict[str, Any]] = []
    for source_index in np.flatnonzero(result.ravel() & (incoming.ravel() == 0)):
        source_index = int(source_index)
        source = divmod(source_index, width)
        path: list[Cell] = []
        current = source
        seen: set[Cell] = set()
        junction: Cell | None = None
        while current not in seen and result[current]:
            seen.add(current)
            path.append(current)
            target_index = int(downstream_index[current])
            if target_index < 0:
                break
            target = divmod(target_index, width)
            if not result[target]:
                break
            if incoming[target] >= 2:
                junction = target
                break
            current = target
        if junction is None or not path or len(path) > maximum_branch_length:
            continue
        vector = (
            float(junction[0] - source[0]),
            float(junction[1] - source[1]),
        )
        vector_length = math.hypot(*vector)
        if vector_length <= 1.0e-12:
            continue
        terminal = path[-1]
        branches.append(
            {
                "source": source,
                "junction": junction,
                "path": path,
                "basin": int(outlet_index[source]),
                "vector": vector,
                "vectorLength": vector_length,
                "flow": float(mfd_accumulation[terminal]),
                "valley": float(np.mean([valley_score[cell] for cell in path])),
                "length": len(path),
            }
        )

    def priority(branch: dict[str, Any]) -> tuple[float, float, int, int, int]:
        source = branch["source"]
        return (
            float(branch["flow"]),
            float(branch["valley"]),
            int(branch["length"]),
            -int(source[0]),
            -int(source[1]),
        )

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for branch in sorted(branches, key=priority, reverse=True):
        source = branch["source"]
        junction = branch["junction"]
        vector = branch["vector"]
        vector_length = float(branch["vectorLength"])
        competes = False
        for winner in kept:
            if int(branch["basin"]) != int(winner["basin"]):
                continue
            winner_source = winner["source"]
            if math.hypot(
                source[0] - winner_source[0],
                source[1] - winner_source[1],
            ) > comparison_radius:
                continue
            winner_junction = winner["junction"]
            if math.hypot(
                junction[0] - winner_junction[0],
                junction[1] - winner_junction[1],
            ) > junction_radius:
                continue
            winner_vector = winner["vector"]
            alignment = (
                vector[0] * winner_vector[0]
                + vector[1] * winner_vector[1]
            ) / max(
                1.0e-12,
                vector_length * float(winner["vectorLength"]),
            )
            if alignment >= 0.85:
                competes = True
                break
        if competes:
            removed.append(branch)
        else:
            kept.append(branch)

    removed_cells: set[Cell] = set()
    for branch in removed:
        removed_cells.update(branch["path"])
    for cell in removed_cells:
        result[cell] = False
    return result, {
        "candidateBranchCount": len(branches),
        "removedBranchCount": len(removed),
        "removedCellCount": len(removed_cells),
        "maximumBranchLength": int(maximum_branch_length),
        "comparisonRadius": float(comparison_radius),
        "junctionRadius": float(junction_radius),
    }


def _ensure_continental_major_rivers(
    stream_mask: np.ndarray,
    candidate_stream_mask: np.ndarray,
    terrain: np.ndarray,
    downstream_index: np.ndarray,
    outlet_index: np.ndarray,
    outlet_type: np.ndarray,
    elevation: np.ndarray,
    mfd_accumulation: np.ndarray,
    valley_score: np.ndarray,
    topological: Sequence[int],
    blocked_route: np.ndarray,
    config: HydrologyConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Retain two independent highland-to-water river spines per continent.

    A seasonal channel threshold is appropriate for minor rivers, but by
    itself it can expose only the lowland tail of every drainage path.  Large
    land components therefore retain a small number of their strongest full
    D8 paths.  Selection is coordinate-free: component area defines both the
    continent and length scales, source elevation is relative to that
    component, and different outlet indices guarantee independent basins.
    """

    result = stream_mask.copy()
    requested = int(config.major_rivers_per_continent)
    land_count = int(np.count_nonzero(terrain))
    minimum_component_area = max(
        config.minimum_headwater_length * 4,
        int(math.ceil(land_count * config.continent_minimum_land_fraction)),
    )
    components = [
        cells
        for cells in _component_cells(terrain)
        if len(cells) >= minimum_component_area
    ]
    distance_to_terminal = np.zeros(terrain.shape, dtype=np.int32)
    path_allowed = np.asarray(terrain & ~blocked_route, dtype=bool).copy()
    for index in reversed(topological):
        index = int(index)
        if not terrain.flat[index]:
            continue
        target = int(downstream_index.flat[index])
        if target >= 0 and terrain.flat[target]:
            distance_to_terminal.flat[index] = (
                int(distance_to_terminal.flat[target]) + 1
            )
            path_allowed.flat[index] = bool(
                path_allowed.flat[index] and path_allowed.flat[target]
            )
        else:
            distance_to_terminal.flat[index] = 1
        if str(outlet_type.flat[index]) not in {"ocean", "lake"}:
            path_allowed.flat[index] = False

    continent_stats: list[dict[str, Any]] = []
    total_added_cells = 0
    width = terrain.shape[1]
    candidate_incoming = _stream_incoming(
        candidate_stream_mask,
        downstream_index,
    )
    candidate_sources = [
        int(index)
        for index in np.flatnonzero(
            candidate_stream_mask.ravel() & (candidate_incoming.ravel() == 0)
        )
    ]
    for component_number, cells in enumerate(components, start=1):
        component_mask = np.zeros(terrain.shape, dtype=bool)
        component_rows, component_columns = zip(*cells)
        component_mask[component_rows, component_columns] = True
        component_elevation = np.asarray(
            [float(elevation[cell]) for cell in cells],
            dtype=np.float64,
        )
        highland_threshold = float(
            np.quantile(component_elevation, config.headwater_upland_quantile)
        )
        minimum_length = max(
            config.minimum_headwater_length,
            int(
                math.ceil(
                    math.sqrt(len(cells))
                    * config.major_river_minimum_span_factor
                )
            ),
        )

        # Keep the furthest highland source for each independent terminal
        # basin.  This avoids tracing every cell on a million-cell atlas.
        best_source_by_outlet: dict[int, Cell] = {}
        for cell in cells:
            if (
                not path_allowed[cell]
                or float(elevation[cell]) < highland_threshold
                or int(distance_to_terminal[cell]) < minimum_length
            ):
                continue
            terminal = int(outlet_index[cell])
            if terminal < 0:
                continue
            current = best_source_by_outlet.get(terminal)
            if current is None or (
                int(distance_to_terminal[cell]),
                float(elevation[cell]),
                -cell[0],
                -cell[1],
            ) > (
                int(distance_to_terminal[current]),
                float(elevation[current]),
                -current[0],
                -current[1],
            ):
                best_source_by_outlet[terminal] = cell

        candidates: list[dict[str, Any]] = []
        for terminal, source in best_source_by_outlet.items():
            path: list[Cell] = []
            current = source
            seen: set[Cell] = set()
            while current not in seen and terrain[current]:
                seen.add(current)
                path.append(current)
                target = int(downstream_index[current])
                if target < 0 or not terrain.flat[target]:
                    break
                current = divmod(target, width)
            if len(path) < minimum_length:
                continue
            mouth = path[-1]
            mouth_flow = float(mfd_accumulation[mouth])
            candidates.append(
                {
                    "source": source,
                    "outletIndex": int(terminal),
                    "path": path,
                    "lengthCells": len(path),
                    "sourceElevation": float(elevation[source]),
                    "mouthFlow": mouth_flow,
                    "score": math.log1p(max(0.0, mouth_flow)) * len(path),
                }
            )

        candidates.sort(
            key=lambda item: (
                float(item["score"]),
                int(item["lengthCells"]),
                float(item["sourceElevation"]),
                -int(item["source"][0]),
                -int(item["source"][1]),
            ),
            reverse=True,
        )
        selected = candidates[:requested] if requested > 0 else []
        rivers: list[dict[str, Any]] = []
        for candidate in selected:
            added = 0
            for cell in candidate["path"]:
                if not result[cell]:
                    result[cell] = True
                    added += 1
            total_added_cells += added
            source = candidate["source"]
            rivers.append(
                {
                    "source": [int(source[0]), int(source[1])],
                    "outletIndex": int(candidate["outletIndex"]),
                    "lengthCells": int(candidate["lengthCells"]),
                    "sourceElevation": float(candidate["sourceElevation"]),
                    "mouthFlow": float(candidate["mouthFlow"]),
                    "addedCells": int(added),
                    "tributaryCount": 0,
                    "tributaryAddedCells": 0,
                    "tributaries": [],
                }
            )

        all_major_cells = {
            cell
            for candidate in selected
            for cell in candidate["path"]
        }
        desired_tributaries = int(config.major_river_tributaries)
        for candidate, river in zip(selected, rivers):
            major_path = candidate["path"]
            major_cells = set(major_path)
            major_positions = {
                cell: position for position, cell in enumerate(major_path)
            }
            feeder_candidates: list[dict[str, Any]] = []
            for source_index in candidate_sources:
                source = divmod(source_index, width)
                if not component_mask[source] or source in all_major_cells:
                    continue
                path: list[Cell] = []
                current = source
                seen: set[Cell] = set()
                junction: Cell | None = None
                while current not in seen and terrain[current]:
                    seen.add(current)
                    if current in major_cells:
                        junction = current
                        break
                    if current in all_major_cells:
                        break
                    if blocked_route[current]:
                        break
                    path.append(current)
                    target = int(downstream_index[current])
                    if target < 0 or not terrain.flat[target]:
                        break
                    current = divmod(target, width)
                if (
                    junction is None
                    or len(path) < config.minimum_headwater_length
                ):
                    continue
                terminal = path[-1]
                mean_valley = float(
                    np.mean([valley_score[cell] for cell in path])
                )
                terminal_flow = float(mfd_accumulation[terminal])
                feeder_candidates.append(
                    {
                        "source": source,
                        "junction": junction,
                        "junctionPosition": int(major_positions[junction]),
                        "path": path,
                        "lengthCells": len(path),
                        "flow": terminal_flow,
                        "valley": mean_valley,
                        "score": (
                            math.log1p(max(0.0, terminal_flow))
                            * math.sqrt(len(path))
                            * (0.75 + mean_valley)
                        ),
                    }
                )
            feeder_candidates.sort(
                key=lambda item: (
                    float(item["score"]),
                    int(item["lengthCells"]),
                    -int(item["source"][0]),
                    -int(item["source"][1]),
                ),
                reverse=True,
            )
            retained_feeders: list[dict[str, Any]] = []
            retained_cells: list[set[Cell]] = []
            for feeder in feeder_candidates:
                feeder_cells = set(feeder["path"])
                if any(
                    len(feeder_cells & other_cells)
                    / max(1, min(len(feeder_cells), len(other_cells)))
                    > 0.25
                    for other_cells in retained_cells
                ):
                    continue
                source = feeder["source"]
                junction = feeder["junction"]
                vector = (
                    float(junction[0] - source[0]),
                    float(junction[1] - source[1]),
                )
                vector_length = math.hypot(*vector)
                too_similar = False
                for other in retained_feeders:
                    other_source = other["source"]
                    other_junction = other["junction"]
                    if math.hypot(
                        source[0] - other_source[0],
                        source[1] - other_source[1],
                    ) > config.minimum_headwater_length * 2:
                        continue
                    other_vector = (
                        float(other_junction[0] - other_source[0]),
                        float(other_junction[1] - other_source[1]),
                    )
                    alignment = (
                        vector[0] * other_vector[0]
                        + vector[1] * other_vector[1]
                    ) / max(1.0e-12, vector_length * math.hypot(*other_vector))
                    if alignment >= 0.85:
                        too_similar = True
                        break
                if too_similar:
                    continue
                retained_feeders.append(feeder)
                retained_cells.append(feeder_cells)
                if len(retained_feeders) >= desired_tributaries:
                    break

            feeder_added_cells = 0
            feeder_records: list[dict[str, Any]] = []
            for feeder in retained_feeders:
                added = 0
                for cell in feeder["path"]:
                    if not result[cell]:
                        result[cell] = True
                        added += 1
                feeder_added_cells += added
                source = feeder["source"]
                junction = feeder["junction"]
                feeder_records.append(
                    {
                        "source": [int(source[0]), int(source[1])],
                        "junction": [int(junction[0]), int(junction[1])],
                        "lengthCells": int(feeder["lengthCells"]),
                        "flow": float(feeder["flow"]),
                        "addedCells": int(added),
                    }
                )
            total_added_cells += feeder_added_cells
            river["tributaryCount"] = len(feeder_records)
            river["tributaryAddedCells"] = int(feeder_added_cells)
            river["tributaries"] = feeder_records
        continent_stats.append(
            {
                "component": component_number,
                "areaCells": len(cells),
                "highlandThreshold": highland_threshold,
                "minimumLengthCells": int(minimum_length),
                "candidateBasinCount": len(candidates),
                "requestedRiverCount": requested,
                "retainedRiverCount": len(rivers),
                "rivers": rivers,
            }
        )

    return result, {
        "minimumComponentAreaCells": int(minimum_component_area),
        "requestedRiversPerContinent": requested,
        "continentCount": len(components),
        "addedCellCount": int(total_added_cells),
        "continents": continent_stats,
    }


def _extend_lake_headwaters(
    stream_mask: np.ndarray,
    downstream_index: np.ndarray,
    terrain: np.ndarray,
    lake_mask: np.ndarray,
    accumulation: np.ndarray,
    valley_score: np.ndarray,
    minimum_length: int,
    maximum_added_cells: int,
) -> tuple[np.ndarray, int, int]:
    """Extend selected lake-bound sources along the existing D8 graph.

    A lake mouth can be a valid outlet while its nearby headwater evidence is
    shorter than the published minimum.  Before pruning, extend only such a
    source by choosing an unselected reverse-D8 predecessor with the greatest
    D8 accumulation (then valley score and cell key).  This keeps the added
    cells on the same corrected surface and consumes a bounded share of the
    normalized closure budget; it never draws a new lateral line to a lake.
    """

    result = stream_mask.copy()
    if minimum_length <= 1 or maximum_added_cells <= 0:
        return result, 0, 0
    height, width = result.shape
    added_cells = 0
    extended_branches = 0

    def reaches_lake(start: Cell) -> bool:
        current = start
        seen: set[Cell] = set()
        while current not in seen:
            seen.add(current)
            target = int(downstream_index[current])
            if target < 0:
                return False
            target_cell = divmod(target, width)
            if lake_mask[target_cell]:
                return True
            current = target_cell
        return False

    while added_cells < maximum_added_cells:
        incoming = _stream_incoming(result, downstream_index)
        changed = False
        for index in np.flatnonzero(result.ravel()):
            if added_cells >= maximum_added_cells:
                break
            source = divmod(int(index), width)
            if incoming[source] != 0:
                continue
            path: list[Cell] = []
            current = source
            seen: set[Cell] = set()
            lake_bound = False
            while current not in seen and result[current]:
                seen.add(current)
                path.append(current)
                target = int(downstream_index[current])
                if target < 0:
                    break
                target_cell = divmod(target, width)
                if lake_mask[target_cell]:
                    lake_bound = True
                    break
                if not result[target_cell] or incoming[target_cell] != 1:
                    lake_bound = reaches_lake(target_cell)
                    break
                current = target_cell
            if not lake_bound or len(path) >= minimum_length:
                continue

            extension_source = source
            branch_added = 0
            while len(path) + branch_added < minimum_length:
                predecessor_candidates: list[Cell] = []
                target_index = extension_source[0] * width + extension_source[1]
                for _, candidate in _neighbours(extension_source, result.shape):
                    if not terrain[candidate] or result[candidate]:
                        continue
                    if int(downstream_index[candidate]) == target_index:
                        predecessor_candidates.append(candidate)
                if not predecessor_candidates or added_cells >= maximum_added_cells:
                    break
                predecessor = max(
                    predecessor_candidates,
                    key=lambda candidate: (
                        float(accumulation[candidate]),
                        float(valley_score[candidate]),
                        -candidate[0],
                        -candidate[1],
                    ),
                )
                result[predecessor] = True
                extension_source = predecessor
                branch_added += 1
                added_cells += 1
                changed = True
            if branch_added:
                extended_branches += 1
        if not changed:
            break
    return result, extended_branches, added_cells


def _extract_network(
    stream_mask: np.ndarray,
    stream_order: np.ndarray,
    accumulation: np.ndarray,
    outlet_type: np.ndarray,
    downstream_index: np.ndarray,
    land_count: int,
    config: HydrologyConfig,
) -> tuple[NetworkRecord, ...]:
    """Compress the directed stream-cell graph at sources and confluences."""

    height, width = stream_mask.shape
    stream_indices = [int(index) for index in np.flatnonzero(stream_mask.ravel())]
    if not stream_indices:
        return ()

    incoming = np.zeros(stream_mask.shape, dtype=np.int16)
    for index in stream_indices:
        downstream = int(downstream_index.flat[index])
        if downstream < 0:
            continue
        row, column = divmod(downstream, width)
        if stream_mask[row, column]:
            incoming[row, column] += 1

    starts = [
        index
        for index in stream_indices
        if incoming.flat[index] != 1
    ]
    starts.sort()
    visited_edges: set[tuple[int, int]] = set()
    records: list[NetworkRecord] = []

    for start in starts:
        current = start
        path = [start]
        endpoint_index = -1
        endpoint_is_confluence = False
        while True:
            terminal_target = int(downstream_index.flat[current])
            if terminal_target < 0:
                break
            target_row, target_column = divmod(terminal_target, width)
            if not stream_mask[target_row, target_column]:
                endpoint_index = terminal_target
                break
            edge = (current, terminal_target)
            if edge in visited_edges:
                endpoint_index = terminal_target
                endpoint_is_confluence = (
                    incoming.flat[terminal_target] >= 2
                    and int(downstream_index.flat[terminal_target]) >= 0
                    and stream_mask.flat[int(downstream_index.flat[terminal_target])]
                )
                break
            visited_edges.add(edge)
            path.append(terminal_target)
            current = terminal_target
            if incoming.flat[current] != 1:
                # A stream cell at the coast can have multiple incoming
                # tributaries while still being the terminal outlet.  It is
                # only a confluence reach endpoint when the junction has a
                # downstream stream cell of its own.
                next_target = int(downstream_index.flat[current])
                endpoint_is_confluence = (
                    incoming.flat[current] >= 2
                    and next_target >= 0
                    and stream_mask.flat[next_target]
                )
                if endpoint_is_confluence:
                    endpoint_index = current
                elif next_target >= 0:
                    endpoint_index = next_target
                else:
                    endpoint_index = current
                break

        final_row, final_column = divmod(path[-1], width)
        if endpoint_index >= 0:
            endpoint = divmod(endpoint_index, width)
        else:
            endpoint = (final_row, final_column)

        path_cells = [list(divmod(index, width)) for index in path]
        # The confluence cell already contains both incoming discharges.  Keep
        # it as the legal segment endpoint, but measure an incoming segment's
        # order/class from its own cells so two tributaries are not promoted to
        # the downstream mainstem merely by touching the junction.
        measurement_path = (
            path[:-1] if endpoint_is_confluence and len(path) > 1 else path
        )
        if len(path) < 2:
            continue
        segment_order = max(int(stream_order.flat[index]) for index in measurement_path)
        segment_accumulation = max(
            float(accumulation.flat[index]) for index in measurement_path
        )
        if (
            segment_order <= 1
            and len(path) < config.minimum_headwater_length
        ):
            continue

        outlet = str(outlet_type.flat[start])
        if outlet not in {"ocean", "lake", "inland_sink"}:
            outlet = "inland_sink"
        records.append(
            {
                "from": list(divmod(start, width)),
                "to": [int(endpoint[0]), int(endpoint[1])],
                "cells": path_cells,
                "length_cells": len(path),
                "order": segment_order,
                "accumulation": segment_accumulation,
                "accumulation_class": _relative_class(
                    segment_accumulation,
                    land_count,
                    config,
                ),
                "outlet_type": outlet,
                "end_type": "confluence" if endpoint_is_confluence else "outlet",
            }
        )

    # A directed acyclic graph should make every stream edge reachable from a
    # source or a confluence.  Keep this guard deterministic if a future routing
    # change introduces an orphan edge.
    for index in stream_indices:
        downstream = int(downstream_index.flat[index])
        if downstream >= 0 and stream_mask.flat[downstream] and (
            index,
            downstream,
        ) not in visited_edges:
            raise HydrologyError("stream graph contains an uncompressed edge")

    records.sort(key=lambda record: (tuple(record["from"]), tuple(record["to"])))
    for sequence, record in enumerate(records, start=1):
        record["id"] = f"river-{sequence:04d}"
    return tuple(records)


def _topological_accumulation(
    active_indices: Sequence[int],
    downstream: np.ndarray,
    shape: tuple[int, int],
    active: np.ndarray,
    terrain: np.ndarray,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Return indegrees, deterministic topological order, and accumulation."""

    indegree = np.zeros(shape, dtype=np.int64)
    for index in active_indices:
        target_index = int(downstream.flat[index])
        if target_index < 0:
            continue
        target_row, target_column = divmod(target_index, shape[1])
        if active[target_row, target_column]:
            indegree[target_row, target_column] += 1
        else:  # pragma: no cover - guarded by caller's active mask
            raise HydrologyError("flow target is not active")

    accumulation = np.zeros(shape, dtype=np.float64)
    accumulation[terrain] = 1.0
    ready = [index for index in active_indices if indegree.flat[index] == 0]
    heapq.heapify(ready)
    topological: list[int] = []
    while ready:
        index = heapq.heappop(ready)
        topological.append(index)
        target_index = int(downstream.flat[index])
        if target_index < 0:
            continue
        accumulation.flat[target_index] += accumulation.flat[index]
        target_row, target_column = divmod(target_index, shape[1])
        indegree[target_row, target_column] -= 1
        if indegree[target_row, target_column] == 0:
            heapq.heappush(ready, target_index)
    return indegree, topological, accumulation


def accumulate_d8_runoff(
    runoff_source: np.ndarray,
    terrain_mask: np.ndarray,
    downstream_index: np.ndarray,
) -> np.ndarray:
    """Accumulate one runoff field on an existing acyclic D8 graph.

    This function never chooses or repairs a route.  It only transports the
    supplied non-negative seasonal runoff along ``downstream_index`` where
    both cells are part of ``terrain_mask``.  The resulting field therefore
    cannot invent a seasonal channel outside the canonical hydrology graph.
    """

    source = np.asarray(runoff_source, dtype=np.float64)
    terrain = np.asarray(terrain_mask)
    downstream = np.asarray(downstream_index)
    if source.ndim != 2 or source.size == 0:
        raise HydrologyError("runoff_source must be a non-empty 2-D array")
    if terrain.shape != source.shape or downstream.shape != source.shape:
        raise HydrologyError(
            "terrain_mask and downstream_index must match runoff_source shape"
        )
    if terrain.dtype != np.dtype(bool):
        raise HydrologyError("terrain_mask must have boolean dtype")
    if downstream.dtype.kind not in "iu":
        raise HydrologyError("downstream_index must have integer dtype")
    if not np.all(np.isfinite(source)) or np.any(source < 0.0):
        raise HydrologyError("runoff_source must be finite and non-negative")
    cell_count = source.size
    if np.any((downstream < -1) | (downstream >= cell_count)):
        raise HydrologyError("downstream_index contains an out-of-range target")
    if np.any(source[~terrain] != 0.0):
        raise HydrologyError("runoff_source must be zero outside terrain_mask")

    active_indices = np.flatnonzero(terrain.ravel())
    indegree = np.zeros(source.shape, dtype=np.int64)
    for index in active_indices:
        target = int(downstream.flat[int(index)])
        if target >= 0 and terrain.flat[target]:
            indegree.flat[target] += 1
    ready = [int(index) for index in active_indices if indegree.flat[index] == 0]
    heapq.heapify(ready)
    accumulation = source.copy()
    visited = 0
    while ready:
        index = heapq.heappop(ready)
        visited += 1
        target = int(downstream.flat[index])
        if target < 0 or not terrain.flat[target]:
            continue
        accumulation.flat[target] += accumulation.flat[index]
        indegree.flat[target] -= 1
        if indegree.flat[target] == 0:
            heapq.heappush(ready, target)
    if visited != len(active_indices):
        raise HydrologyError("downstream_index contains a terrain cycle")
    accumulation[~terrain] = 0.0
    return accumulation


def _cycle_break_cells(
    active_indices: Sequence[int],
    downstream: np.ndarray,
    indegree: np.ndarray,
    raw_elevation: np.ndarray,
) -> tuple[int, ...]:
    """Find one stable, low-elevation outlet per residual directed cycle."""

    flat_indegree = indegree.ravel()
    remaining = {int(index) for index in active_indices if flat_indegree[index] > 0}
    visited: set[int] = set()
    breaks: list[int] = []
    width = downstream.shape[1]
    flat_downstream = downstream.ravel()
    flat_raw = raw_elevation.ravel()
    while remaining:
        start = min(remaining)
        path: list[int] = []
        positions: dict[int, int] = {}
        current = start
        while current in remaining and current not in visited:
            visited.add(current)
            positions[current] = len(path)
            path.append(current)
            target = int(flat_downstream[current])
            if target < 0 or target not in remaining:
                break
            current = target
        if current in positions:
            cycle = path[positions[current] :]
            breaks.append(
                min(
                    cycle,
                    key=lambda index: (
                        float(flat_raw[index]),
                        divmod(index, width)[0],
                        divmod(index, width)[1],
                    ),
                )
            )
        remaining.difference_update(path)
    return tuple(sorted(set(breaks)))


def compute_hydrology(
    elevation: np.ndarray | Sequence[Sequence[float]],
    land_mask: np.ndarray | Sequence[Sequence[bool]],
    ocean_mask: np.ndarray | Sequence[Sequence[bool]],
    lake_mask: np.ndarray | Sequence[Sequence[bool]] | None = None,
    river_constraint_mask: np.ndarray | Sequence[Sequence[float]] | None = None,
    perennial_snow_mask: np.ndarray | Sequence[Sequence[bool]] | None = None,
    runoff_source_yield: np.ndarray | Sequence[Sequence[float]] | None = None,
    seasonality_index: np.ndarray | Sequence[Sequence[float]] | None = None,
    config: HydrologyConfig | None = None,
    latitude_degrees: np.ndarray | Sequence[float] | None = None,
) -> HydrologyResult:
    """Build a deterministic D8 hydrologic network from a relative raster.

    Parameters
    ----------
    elevation:
        Finite 2-D float-like relative elevations.  Units are intentionally
        unspecified; only ordering and finite burn depth matter.
    land_mask:
        2-D mask of cells considered land.  Cells that are neither land nor
        explicitly classified ocean/lake remain inactive and seed an inland
        sink only when they disconnect a terrain component.
    ocean_mask:
        Required 2-D mask of explicit ocean cells.  Raster edges are not
        implicit outlets; clipped/nodata cells must not be marked ocean unless
        the caller has evidence that they are ocean.
    lake_mask:
        Optional explicit water mask.  Lake cells are excluded from depression
        filling and are terminal by default.  They may be routed to configured
        outlets through :class:`HydrologyConfig`.
    river_constraint_mask:
        Optional boolean or non-negative numeric authored strength mask.  It is
        converted to ``[0, 1]`` and lowered by finite ``stream_burn_depth``;
        connected authored pixels are never copied into the network directly.
    perennial_snow_mask:
        Optional perennial snow/ice cells derived for the same land surface.
        They increase potential runoff supply and qualify an upstream
        catchment as a plausible meltwater-fed headwater.  Snow outside
        non-lake land is rejected.
    runoff_source_yield:
        Optional non-negative per-cell dry-season climate yield.  It scales
        local rain and baseflow before snowmelt is added, then is normalized
        internally so only its spatial contrast changes channel selection.
    seasonality_index:
        Optional ``[0, 1]`` monsoon seasonality.  It is validated alongside
        the physical inputs, but does not create a second topology gate.
        ``runoff_source_yield`` already expresses dry-season baseflow in the
        accumulated catchment evidence.  A structural tributary is therefore
        not removed merely because it is seasonal; consumers may use the
        four seasonal runoff fields to draw an intermittent reach differently.
    config:
        Deterministic thresholds, finite burn depth, optional lake outlets, and
        optional explicit inland sink seeds.
    latitude_degrees:
        Optional geographic latitude for each raster row.  When supplied,
        east/west D8 steps use the local cosine scale; omitted values use the
        unit-grid distance expected by synthetic relative rasters.  The
        authoring-grid caller supplies this from its plate-carree registration.

    Returns
    -------
    HydrologyResult
        Corrected surface, D8 direction/accumulation rasters, stream mask,
        outlet labels, and a direction-oriented compressed network.

    Complexity is ``O(N log N)`` time and ``O(N)`` memory for ``N`` raster
    cells, dominated by the Priority-Flood heap.  Every other stage is linear.
    """

    active_config = config or HydrologyConfig()
    _validate_config(active_config)

    try:
        raw_elevation = np.asarray(elevation, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise HydrologyError("elevation must be a numeric 2-D array") from error
    if raw_elevation.ndim != 2 or raw_elevation.size == 0:
        raise HydrologyError("elevation must be a non-empty 2-D array")
    if not np.all(np.isfinite(raw_elevation)):
        raise HydrologyError("elevation contains non-finite values")
    shape = raw_elevation.shape

    if latitude_degrees is None:
        row_latitudes = None
    else:
        try:
            row_latitudes = np.asarray(latitude_degrees, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise HydrologyError("latitude_degrees must be a finite row vector") from error
        if row_latitudes.ndim != 1 or row_latitudes.shape[0] != shape[0]:
            raise HydrologyError(
                f"latitude_degrees must contain one value per raster row ({shape[0]})"
            )
        if not np.all(np.isfinite(row_latitudes)) or np.any(np.abs(row_latitudes) > 90.0):
            raise HydrologyError("latitude_degrees must be finite and within [-90, 90]")

    try:
        raw_land = np.asarray(land_mask)
    except (TypeError, ValueError) as error:
        raise HydrologyError("land_mask must be a 2-D array") from error
    _validate_shape(raw_land, shape, "land_mask")
    land = raw_land.astype(bool, copy=True)
    try:
        raw_ocean = np.asarray(ocean_mask)
    except (TypeError, ValueError) as error:
        raise HydrologyError("ocean_mask must be a 2-D boolean-like array") from error
    _validate_shape(raw_ocean, shape, "ocean_mask")
    ocean = raw_ocean.astype(bool, copy=True)
    if np.any(land & ocean):
        raise HydrologyError("land_mask and ocean_mask must be disjoint")
    if lake_mask is None:
        raw_lake = np.zeros(shape, dtype=bool)
    else:
        try:
            raw_lake = np.asarray(lake_mask).astype(bool, copy=False)
        except (TypeError, ValueError) as error:
            raise HydrologyError("lake_mask must be a 2-D boolean-like array") from error
    _validate_shape(raw_lake, shape, "lake_mask")
    lake = raw_lake.copy()
    if np.any(ocean & lake):
        raise HydrologyError("ocean_mask and lake_mask must be disjoint")
    if perennial_snow_mask is None:
        perennial_snow = np.zeros(shape, dtype=bool)
    else:
        try:
            perennial_snow = np.asarray(perennial_snow_mask).astype(
                bool, copy=False
            )
        except (TypeError, ValueError) as error:
            raise HydrologyError(
                "perennial_snow_mask must be a 2-D boolean-like array"
            ) from error
        _validate_shape(perennial_snow, shape, "perennial_snow_mask")
        perennial_snow = perennial_snow.copy()
    if np.any(perennial_snow & ~(land & ~lake)):
        raise HydrologyError(
            "perennial_snow_mask must contain only non-lake land cells"
        )
    climate_yield: np.ndarray | None
    if runoff_source_yield is None:
        climate_yield = None
    else:
        try:
            climate_yield = np.asarray(runoff_source_yield, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise HydrologyError("runoff_source_yield must be numeric") from error
        if climate_yield.ndim != 2 or climate_yield.shape != shape:
            raise HydrologyError(f"runoff_source_yield must have shape {shape}")
        if not np.all(np.isfinite(climate_yield)):
            raise HydrologyError("runoff_source_yield contains non-finite values")
        if np.any(climate_yield < 0.0):
            raise HydrologyError("runoff_source_yield cannot contain negative values")
        climate_yield = climate_yield.copy()
        climate_yield[~(land & ~lake)] = 0.0
        if not np.any(climate_yield[land & ~lake] > 0.0):
            raise HydrologyError("runoff_source_yield has no positive terrain supply")
    seasonality: np.ndarray | None
    if seasonality_index is None:
        seasonality = None
    else:
        try:
            seasonality = np.asarray(seasonality_index, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise HydrologyError("seasonality_index must be numeric") from error
        if seasonality.ndim != 2 or seasonality.shape != shape:
            raise HydrologyError(f"seasonality_index must have shape {shape}")
        if not np.all(np.isfinite(seasonality)):
            raise HydrologyError("seasonality_index contains non-finite values")
        if np.any((seasonality < 0.0) | (seasonality > 1.0)):
            raise HydrologyError("seasonality_index must stay within [0, 1]")
        seasonality = seasonality.copy()
        seasonality[~(land & ~lake)] = 0.0
    constraint_strength = _normalise_constraint_mask(river_constraint_mask, shape)

    components, lake_labels, lake_outlet_parents, lake_outlet_config = _prepare_lakes(
        lake, active_config
    )
    terrain = land & ~lake
    burned = raw_elevation.copy()
    burned[terrain] -= active_config.stream_burn_depth * constraint_strength[terrain]
    corrected = burned.copy()

    # Priority-Flood seeds are all terrain cells that touch ocean, explicit
    # lakes, or a configured inland sink.  Components without any seed get a
    # deterministic lowest-cell inland sink so non-lake pseudo-pits cannot
    # remain unresolved.
    seeds: dict[Cell, tuple[Cell | None, str]] = {}
    explicit_sinks: set[Cell] = set()
    for sink in active_config.inland_sink_cells:
        try:
            sink_values = tuple(sink)
        except (TypeError, ValueError) as error:
            raise HydrologyError(f"invalid inland sink cell: {sink}") from error
        if len(sink_values) != 2:
            raise HydrologyError(f"invalid inland sink cell: {sink}")
        try:
            sink_cell = (int(sink_values[0]), int(sink_values[1]))
        except (TypeError, ValueError) as error:
            raise HydrologyError(f"invalid inland sink cell: {sink}") from error
        row, column = sink_cell
        if not (0 <= row < shape[0] and 0 <= column < shape[1]) or not terrain[sink_cell]:
            raise HydrologyError(f"inland sink must be a non-lake land cell: {sink}")
        explicit_sinks.add(sink_cell)

    for row, column in zip(*np.where(terrain), strict=True):
        cell = (int(row), int(column))
        if _touches_ocean(cell, ocean):
            seeds[cell] = (None, "ocean")
            continue
        lake_neighbour = _choose_lake_neighbour(
            cell,
            lake,
            lake_labels,
            lake_outlet_parents,
            lake_outlet_config,
        )
        if lake_neighbour is not None:
            seeds[cell] = (lake_neighbour, "lake")
            continue
        if cell in explicit_sinks:
            seeds[cell] = (None, "inland_sink")

    for component in _component_cells(terrain):
        if any(cell in seeds for cell in component):
            continue
        fallback = min(component, key=lambda cell: (burned[cell], cell[0], cell[1]))
        seeds[fallback] = (None, "inland_sink")

    visited = np.zeros(shape, dtype=bool)
    flood_parent = np.full(shape, None, dtype=object)
    seed_kind = np.full(shape, "", dtype=object)
    flood_rank = np.full(shape, -1, dtype=np.int64)
    priority: list[tuple[float, int, int]] = []
    for cell in sorted(
        seeds,
        key=lambda candidate: (
            float(burned[candidate]),
            candidate[0],
            candidate[1],
            0 if seeds[candidate][1] == "ocean" else 1,
        ),
    ):
        row, column = cell
        visited[cell] = True
        flood_parent[cell] = seeds[cell][0]
        seed_kind[cell] = seeds[cell][1]
        heapq.heappush(priority, (float(corrected[cell]), row, column))

    rank = 0
    while priority:
        _, row, column = heapq.heappop(priority)
        if flood_rank[row, column] >= 0:
            continue
        flood_rank[row, column] = rank
        rank += 1
        current = (row, column)
        for _, neighbour in _neighbours(current, shape):
            next_row, next_column = neighbour
            if not terrain[neighbour] or visited[neighbour]:
                continue
            visited[neighbour] = True
            flood_parent[neighbour] = current
            corrected[neighbour] = max(float(burned[neighbour]), float(corrected[current]))
            heapq.heappush(priority, (float(corrected[neighbour]), next_row, next_column))

    # Every terrain cell should belong to a seeded component.  This also turns
    # accidental future changes to seed logic into a clear error instead of an
    # array of silently un-routed cells.
    if np.any(terrain & ~visited):  # pragma: no cover - guarded by component seeds
        raise HydrologyError("Priority-Flood left terrain cells unvisited")

    valley_score, valley_radius = _valley_score(
        corrected,
        terrain,
        active_config.tie_epsilon,
        active_config.valley_window_fraction,
        active_config.valley_depth_fraction,
    )
    flat_distance = _flat_routing_distance(
        corrected,
        terrain,
        ocean,
        lake,
        seed_kind,
        active_config.tie_epsilon,
    )
    raw_terrain_values = raw_elevation[terrain]
    raw_range = (
        float(np.max(raw_terrain_values)) - float(np.min(raw_terrain_values))
        if raw_terrain_values.size
        else 0.0
    )
    raw_uphill_tolerance = max(
        active_config.tie_epsilon,
        raw_range * DEFAULT_RAW_UPHILL_TOLERANCE_FRACTION,
    )
    (
        unresolved_relief,
        unresolved_axis_east,
        unresolved_axis_south,
    ) = _unresolved_alluvial_fields(shape)
    (
        flat_unsupported_count,
        flat_raw_ascent,
        flat_unsupported_edge_count,
        flat_cost_distance,
    ) = _flat_routing_cost(
        corrected,
        raw_elevation,
        terrain,
        ocean,
        lake,
        seed_kind,
        valley_score,
        constraint_strength,
        unresolved_relief,
        unresolved_axis_east,
        unresolved_axis_south,
        active_config.tie_epsilon,
        raw_uphill_tolerance,
        active_config.valley_support_threshold,
    )
    flow_direction = np.full(shape, -1, dtype=np.int8)
    downstream = np.full(shape, -1, dtype=np.int64)
    terminal_kind = np.full(shape, "", dtype=object)
    unsupported_flat_route = np.zeros(shape, dtype=bool)

    for row, column in zip(*np.where(terrain), strict=True):
        cell = (int(row), int(column))
        current_elevation = float(corrected[cell])
        lower: list[tuple[float, float, float, float, float, int, Cell]] = []
        for direction, neighbour in _neighbours(cell, shape):
            if not terrain[neighbour]:
                continue
            drop = current_elevation - float(corrected[neighbour])
            if drop <= active_config.tie_epsilon:
                continue
            distance = _ground_distance(cell, neighbour, row_latitudes)
            slope = drop / max(distance, active_config.tie_epsilon)
            lower.append(
                (
                    slope,
                    drop,
                    float(raw_elevation[cell]) - float(raw_elevation[neighbour]),
                    float(valley_score[neighbour]),
                    float(constraint_strength[neighbour]),
                    direction,
                    neighbour,
                )
            )
        target: Cell | None = None
        if lower:
            highest_slope = max(item[0] for item in lower)
            slope_ties = [
                item
                for item in lower
                if item[0] >= highest_slope - active_config.tie_epsilon
            ]
            _, _, _, _, _, _, target = max(
                slope_ties,
                key=lambda item: (
                    item[2],
                    item[3],
                    item[4],
                    item[1],
                    -int(item[5] % 2 == 1),
                    -item[5],
                    -item[6][0],
                    -item[6][1],
                ),
            )
        else:
            lake_candidates = _lake_neighbours(cell, lake)
            if lake_candidates:
                # Explicit lakes are terminal sinks even when authored relative
                # elevations put their water above a neighbouring flat cell.
                target = min(lake_candidates, key=lambda item: item[0])[1]
            else:
                # Descend the corrected-flat cost field.  Its legal candidates
                # are exactly the neighbours on an optimal path to a real
                # spill/terminal.  Raw descent is the first local tie-break,
                # followed by valley/constraint evidence; cardinality and the
                # geometric fields remain deterministic final tie-breaks.
                flat_candidates: list[
                    tuple[float, float, float, int, int, int, Cell]
                ] = []
                current_cost = (
                    int(flat_unsupported_count[cell]),
                    float(flat_raw_ascent[cell]),
                    int(flat_unsupported_edge_count[cell]),
                    int(flat_cost_distance[cell]),
                )
                for direction, neighbour in _neighbours(cell, shape):
                    if not terrain[neighbour]:
                        continue
                    if abs(float(corrected[neighbour]) - current_elevation) > (
                        active_config.tie_epsilon
                    ):
                        continue
                    neighbour_cost = (
                        int(flat_unsupported_count[neighbour]),
                        float(flat_raw_ascent[neighbour]),
                        int(flat_unsupported_edge_count[neighbour]),
                        int(flat_cost_distance[neighbour]),
                    )
                    raw_gap = max(
                        0.0,
                        float(raw_elevation[neighbour])
                        - float(raw_elevation[cell]),
                    )
                    authored_supported = max(
                        float(constraint_strength[cell]),
                        float(constraint_strength[neighbour]),
                    ) > active_config.tie_epsilon
                    raw_tie_tolerance = (
                        raw_uphill_tolerance * 0.25
                        if authored_supported
                        else active_config.tie_epsilon
                    )
                    if raw_gap <= raw_tie_tolerance:
                        raw_gap = 0.0
                    supported = (
                        max(
                            float(valley_score[cell]),
                            float(valley_score[neighbour]),
                        )
                        >= active_config.valley_support_threshold
                        or authored_supported
                    )
                    edge_unsupported = int(
                        raw_gap > raw_uphill_tolerance and not supported
                    )
                    edge_support_gap = int(not supported)
                    is_diagonal = abs(neighbour[0] - row) == 1 and abs(
                        neighbour[1] - column
                    ) == 1
                    step_cost = _alluvial_step_cost(
                        cell,
                        neighbour,
                        unresolved_relief,
                        unresolved_axis_east,
                        unresolved_axis_south,
                    )
                    candidate_cost = (
                        edge_unsupported + neighbour_cost[0],
                        raw_gap + neighbour_cost[1],
                        edge_support_gap + neighbour_cost[2],
                        step_cost + neighbour_cost[3],
                    )
                    if current_cost[0] != candidate_cost[0] or (
                        abs(current_cost[1] - candidate_cost[1])
                        > active_config.tie_epsilon
                        or current_cost[2] != candidate_cost[2]
                        or current_cost[3] != candidate_cost[3]
                    ):
                        continue
                    flat_candidates.append(
                        (
                            float(raw_elevation[cell]) - float(raw_elevation[neighbour]),
                            float(valley_score[neighbour]),
                            float(constraint_strength[neighbour]),
                            int(is_diagonal),
                            direction,
                            int(neighbour_cost[3]),
                            neighbour,
                        )
                    )
                if flat_candidates:
                    _, _, _, _, _, _, target = max(
                        flat_candidates,
                        key=lambda item: (
                            item[0],  # raw drop (minimum raw ascent)
                            item[1],  # local valley support
                            item[2],  # authored constraint strength
                            -item[3],  # cardinal before diagonal
                            -item[5],  # shorter global cost-to-spill path
                            -item[4],  # stable D8 direction
                            -item[6][0],
                            -item[6][1],
                        ),
                    )
                else:
                    # A corrected component without a cardinal BFS seed is a
                    # guarded fallback for future mask/raster variants.  Use
                    # the Priority-Flood parent only when it is an equal or
                    # lower terrain cell, preserving the old acyclic proof.
                    parent = flood_parent[cell]
                    if parent is not None:
                        parent_cell = tuple(parent)
                        if lake[parent_cell] or (
                            terrain[parent_cell]
                            and float(corrected[parent_cell])
                            <= current_elevation + active_config.tie_epsilon
                        ):
                            target = parent_cell

        if target is None:
            if _touches_ocean(cell, ocean) or seed_kind[cell] == "ocean":
                terminal_kind[cell] = "ocean"
            elif seed_kind[cell] == "lake":
                terminal_kind[cell] = "lake"
            else:
                terminal_kind[cell] = "inland_sink"
            continue

        target_row, target_column = target
        direction = _direction_between(cell, target)
        flow_direction[cell] = direction
        downstream[cell] = target_row * shape[1] + target_column
        corrected_delta = float(corrected[target]) - current_elevation
        raw_delta = float(raw_elevation[target]) - float(raw_elevation[cell])
        supported_edge = (
            max(float(valley_score[cell]), float(valley_score[target]))
            >= active_config.valley_support_threshold
            or max(
                float(constraint_strength[cell]),
                float(constraint_strength[target]),
            )
            > active_config.tie_epsilon
        )
        if (
            abs(corrected_delta) <= active_config.tie_epsilon
            and raw_delta > raw_uphill_tolerance
            and not supported_edge
        ):
            unsupported_flat_route[cell] = True

    # Explicit lake outlet trees are D8-directed toward their configured target.
    for cell, parent in sorted(lake_outlet_parents.items()):
        flow_direction[cell] = _direction_between(cell, parent)
        downstream[cell] = parent[0] * shape[1] + parent[1]

    active = terrain | lake
    active_indices = [int(index) for index in np.flatnonzero(active.ravel())]
    indegree, topological, accumulation = _topological_accumulation(
        active_indices,
        downstream,
        shape,
        active,
        terrain,
    )
    if len(topological) != len(active_indices):
        # Very fine rasters expose large, numerically flat plateaus.  The
        # corrected-flat cost field normally orients them, but a future tie or
        # a custom source can still leave a tiny directed cycle.  Cut one
        # lowest cell per cycle as an explicit endorheic sink, then recompute
        # the same topological accumulation.  This is preferable to rejecting
        # an otherwise usable world and never invents an uphill edge.
        cycle_breaks = _cycle_break_cells(
            active_indices,
            downstream,
            indegree,
            raw_elevation,
        )
        if not cycle_breaks:  # pragma: no cover - defensive graph guard
            raise HydrologyError("D8 flow graph contains a cycle")
        for index in cycle_breaks:
            downstream.flat[index] = -1
            flow_direction.flat[index] = -1
            row, column = divmod(index, shape[1])
            terminal_kind[row, column] = "inland_sink"
        indegree, topological, accumulation = _topological_accumulation(
            active_indices,
            downstream,
            shape,
            active,
            terrain,
        )
        if len(topological) != len(active_indices):  # pragma: no cover
            raise HydrologyError("D8 flow graph contains a cycle after repair")

    outlet_type = np.full(shape, "", dtype=object)
    for index in active_indices:
        row, column = divmod(index, shape[1])
        target_index = int(downstream.flat[index])
        if target_index >= 0:
            continue
        if lake[row, column]:
            outlet_type[row, column] = "lake"
        elif terminal_kind[row, column]:
            outlet_type[row, column] = terminal_kind[row, column]
        elif _touches_ocean((row, column), ocean):
            outlet_type[row, column] = "ocean"
        else:
            outlet_type[row, column] = "inland_sink"
    for index in reversed(topological):
        target_index = int(downstream.flat[index])
        if target_index >= 0:
            outlet_type.flat[index] = outlet_type.flat[target_index]

    outlet_index = np.full(shape, -1, dtype=np.int64)
    for index in active_indices:
        if int(downstream.flat[index]) < 0:
            outlet_index.flat[index] = index
    for index in reversed(topological):
        target_index = int(downstream.flat[index])
        if target_index >= 0:
            outlet_index.flat[index] = outlet_index.flat[target_index]

    runoff_source_weight = _runoff_source_weights(
        raw_elevation,
        terrain,
        perennial_snow,
        active_config.runoff_weight_floor,
        active_config.snowmelt_runoff_bonus,
        climate_yield,
    )
    mfd_accumulation = _multiple_flow_accumulation(
        corrected,
        terrain,
        downstream,
        flat_distance,
        row_latitudes,
        active_config.tie_epsilon,
        active_config.mfd_exponent,
        runoff_source_weight,
    )
    mfd_input = float(runoff_source_weight[terrain].sum())
    mfd_terminal_cells = (terrain | lake) & (downstream < 0)
    mfd_terminal = float(mfd_accumulation[mfd_terminal_cells].sum())
    mfd_finite = bool(np.all(np.isfinite(mfd_accumulation)))
    mfd_nonnegative = bool(np.all(mfd_accumulation >= 0.0))
    if not mfd_finite:
        raise HydrologyError("MFD accumulation contains non-finite values")
    if not mfd_nonnegative:
        raise HydrologyError("MFD accumulation contains negative values")
    mfd_conservation_error = mfd_terminal - mfd_input

    land_count = max(1, int(np.count_nonzero(land)))
    # This explicit reach-length threshold must not increase when unrelated
    # land is added elsewhere. Catchment/valley/source evidence determines
    # channel eligibility; global land area cannot erase a qualified local
    # tributary that already meets the configured minimum length.
    effective_minimum_headwater_length = active_config.minimum_headwater_length
    selection_config = active_config
    threshold = max(
        1,
        int(math.ceil(land_count * active_config.stream_threshold_fraction)),
        int(math.ceil(math.sqrt(land_count) * active_config.minimum_stream_span_factor)),
    )
    mainstem_threshold = max(
        threshold,
        int(math.ceil(land_count * active_config.mainstem_threshold_fraction)),
    )
    # An accumulation threshold alone treats every row of a broad plane as a
    # river.  Require local valley support for tributaries; only the largest
    # relative-flow class is allowed to cross a broad, unsupported slope.
    valley_supported = _propagate_valley_support(
        valley_score,
        downstream,
        topological,
        maximum_distance=max(
            2,
            int(round(valley_radius * active_config.valley_support_distance_factor)),
        ),
        support_threshold=active_config.valley_support_threshold,
    )
    upland_threshold = float(
        np.quantile(
            raw_elevation[terrain],
            active_config.headwater_upland_quantile,
        )
    )
    upland = terrain & (raw_elevation >= upland_threshold)
    valley_threshold = max(1, int(math.ceil(threshold * 0.78)))
    upland_valley_threshold = max(1, int(math.ceil(threshold * 0.68)))
    snow_threshold = max(1, int(math.ceil(threshold * 0.60)))
    local_stream_threshold = np.full(shape, threshold, dtype=np.int64)
    local_stream_threshold[valley_supported] = valley_threshold
    local_stream_threshold[valley_supported & upland] = upland_valley_threshold
    # Canonical ``stream_order`` is a physical drainage topology, not a
    # four-season cartographic snapshot.  The climate source yield has already
    # reduced persistent catchment supply before MFD accumulation; multiplying
    # this threshold a second time used to erase whole monsoon and high-latitude
    # tributary trees.  Keep every reach that clears the shared accumulation,
    # valley/source, length, and anti-comb evidence gates below.  Seasonal map
    # views can still show zero strength when their actual runoff is zero.
    #
    # Keep a ones field for scalar diagnostics and to make the selection rule
    # explicit in the serialized hydrology audit.
    seasonal_multiplier = np.ones(shape, dtype=np.float64)
    local_mainstem_threshold = np.full(shape, mainstem_threshold, dtype=np.int64)
    local_stream_threshold[perennial_snow] = snow_threshold
    lake_draining = terrain & (outlet_type == "lake")
    lake_drainage_cell_count = int(np.count_nonzero(lake_draining))
    lake_basin_threshold = (
        min(
            threshold,
            max(
                1,
                selection_config.minimum_headwater_length * 2,
                int(
                    math.ceil(
                        math.sqrt(lake_drainage_cell_count)
                        * active_config.minimum_stream_span_factor
                    )
                ),
            ),
        )
        if lake_drainage_cell_count
        else threshold
    )
    local_stream_threshold[lake_draining] = np.minimum(
        local_stream_threshold[lake_draining],
        lake_basin_threshold,
    )
    stream_mask = terrain & (mfd_accumulation >= local_stream_threshold) & (
        valley_supported | (mfd_accumulation >= local_mainstem_threshold)
        | perennial_snow | lake_draining
    )
    raw_stream_mask = stream_mask.copy()
    stream_mask = _suppress_parallel_streams(
        stream_mask,
        flow_direction,
        downstream,
        mfd_accumulation,
        valley_score,
        constraint_strength,
        outlet_index,
        valley_radius,
        active_config.parallel_search_radius_factor,
        active_config.parallel_priority_weight,
    )
    suppressed_stream_mask = stream_mask.copy()
    closure_budget_cells = int(
        math.ceil(land_count * active_config.closure_budget_fraction)
    )
    closure_max_chain_length = (
        max(
            int(selection_config.minimum_headwater_length),
            int(
                math.ceil(
                    min(shape) * active_config.closure_max_chain_fraction
                )
            ),
        )
        if active_config.closure_max_chain_fraction > 0.0
        else 0
    )
    stream_mask, closure_stats = _close_stream_paths(
        stream_mask,
        downstream,
        terrain,
        closure_budget_cells,
        closure_max_chain_length,
    )
    lake_extension_budget = max(
        0,
        closure_budget_cells - int(closure_stats["addedCells"]),
    )
    stream_mask, extended_lake_headwater_count, extended_lake_headwater_cells = (
        _extend_lake_headwaters(
            stream_mask,
            downstream,
            terrain,
            lake,
            accumulation,
            valley_score,
            selection_config.minimum_headwater_length,
            lake_extension_budget,
        )
    )
    closed_stream_mask = stream_mask.copy()
    # Do not run cell-level corridor suppression after closure.  Removing one
    # interior cell at that point severs a valid drainage tree and forces every
    # upstream reach to be discarded.  Parallel low-order tributaries are
    # compared later as complete source-to-confluence branches, which keeps the
    # topology intact while still preventing comb-shaped river fans.
    stream_mask, pruned_headwater_count, pruned_headwater_cells = _prune_short_headwaters(
        stream_mask,
        downstream,
        selection_config.minimum_headwater_length,
    )
    upslope_maximum_elevation, upslope_contains_snow = _upslope_source_provenance(
        raw_elevation,
        perennial_snow,
        terrain,
        downstream,
        topological,
    )
    stream_mask, headwater_source_stats = _prune_unqualified_headwaters(
        stream_mask,
        downstream,
        mfd_accumulation,
        valley_supported | lake_draining,
        upslope_maximum_elevation,
        upslope_contains_snow,
        upland_threshold,
        local_stream_threshold,
        active_config.lowland_headwater_flow_multiplier,
        max(
            selection_config.minimum_headwater_length * 3,
            int(math.ceil(min(shape) * 0.01)),
        ),
    )
    stream_mask, competing_headwater_stats = _prune_competing_headwater_branches(
        stream_mask,
        downstream,
        outlet_index,
        mfd_accumulation,
        valley_score,
        valley_radius,
        selection_config.minimum_headwater_length,
    )
    # A corrected-flat route with a significant raw uphill transition and no
    # local valley/constraint evidence is not a publishable river corridor.
    # Remove only cells whose selected D8 edge has that exact evidence shape,
    # then perform the same zero-budget closure cleanup used for suppressed
    # gaps.  This keeps the rule generic and lets a genuinely supported or
    # raw-descending route survive without deleting authored reaches by name.
    unsupported_flat_stream_cells = int(
        np.count_nonzero(stream_mask & unsupported_flat_route)
    )
    unsupported_flat_pruned_cells = 0
    unsupported_flat_cleanup: dict[str, int | float] = {
        "beforeCells": int(np.count_nonzero(stream_mask)),
        "budgetCells": 0,
        "maxAllowedChainLength": 0,
        "addedCells": 0,
        "maxChainLength": 0,
        "addedRatio": 0.0,
        "overBudgetChainCount": 0,
        "discardedUnclosedCells": 0,
    }
    if unsupported_flat_stream_cells:
        stream_mask &= ~unsupported_flat_route
        before_cleanup = int(np.count_nonzero(stream_mask))
        stream_mask, unsupported_flat_cleanup = _close_stream_paths(
            stream_mask,
            downstream,
            terrain,
            0,
            0,
        )
        unsupported_flat_pruned_cells = before_cleanup - int(
            np.count_nonzero(stream_mask)
        )
    # Evidence and unsupported-flat pruning both change confluence indegrees.
    # A retained sibling can consequently become a new source even though it
    # was not a headwater during the earlier length pass.  Reapply the public
    # minimum to the final topology so the renderer never receives isolated
    # one- or two-cell blue stubs.
    (
        stream_mask,
        final_pruned_headwater_count,
        final_pruned_headwater_cells,
    ) = _prune_short_headwaters(
        stream_mask,
        downstream,
        selection_config.minimum_headwater_length,
    )
    stream_mask, continental_major_river_stats = _ensure_continental_major_rivers(
        stream_mask,
        raw_stream_mask,
        terrain,
        downstream,
        outlet_index,
        outlet_type,
        raw_elevation,
        mfd_accumulation,
        valley_score,
        topological,
        unsupported_flat_route,
        selection_config,
    )
    stream_order = _strahler_order(stream_mask, downstream, topological)

    network = _extract_network(
        stream_mask,
        stream_order,
        accumulation,
        outlet_type,
        downstream,
        land_count,
        selection_config,
    )
    diagnostics: dict[str, Any] = {
        "effectiveStreamThreshold": int(threshold),
        "effectiveMinimumHeadwaterLength": int(
            effective_minimum_headwater_length
        ),
        "localStreamThreshold": {
            "global": int(threshold),
            "valley": int(valley_threshold),
            "uplandValley": int(upland_valley_threshold),
            "snow": int(snow_threshold),
            "lakeBasin": int(lake_basin_threshold),
            "lakeDrainingCellCount": lake_drainage_cell_count,
            "maximumSeasonalMultiplier": float(np.max(seasonal_multiplier[terrain])),
            "seasonalityTopologyGate": "none; climate runoff yield remains applied",
            "seasonalityInputMaximum": (
                float(np.max(seasonality[terrain]))
                if seasonality is not None
                else None
            ),
        },
        "mainstemThreshold": int(mainstem_threshold),
        "rawSelectedStreamCellCount": int(np.count_nonzero(raw_stream_mask)),
        "parallelSuppressedCellCount": int(
            np.count_nonzero(raw_stream_mask)
            - np.count_nonzero(suppressed_stream_mask)
        ),
        "closedStreamCellCount": int(np.count_nonzero(closed_stream_mask)),
        "selectedStreamCellCount": int(np.count_nonzero(stream_mask)),
        "valleySupportedCellCount": int(np.count_nonzero(valley_supported)),
        "flatRoutingCellCount": int(np.count_nonzero(terrain & (flat_distance >= 0))),
        "flatRoutingCostUnsupportedCellCount": int(
            np.count_nonzero(terrain & (flat_unsupported_count > 0))
        ),
        "rawElevationRange": raw_range,
        "rawUphillTolerance": raw_uphill_tolerance,
        "unsupportedFlatRouteCellCount": int(
            np.count_nonzero(unsupported_flat_route)
        ),
        "unsupportedFlatStreamCellCount": unsupported_flat_stream_cells,
        "unsupportedFlatPrunedCellCount": unsupported_flat_pruned_cells,
        "unsupportedFlatCleanup": unsupported_flat_cleanup,
        "prunedHeadwaterCount": int(
            pruned_headwater_count + final_pruned_headwater_count
        ),
        "competingHeadwaters": competing_headwater_stats,
        "continentalMajorRivers": continental_major_river_stats,
        "prunedHeadwaterCellCount": int(
            pruned_headwater_cells + final_pruned_headwater_cells
        ),
        "finalTopologyPrunedHeadwaterCount": int(final_pruned_headwater_count),
        "finalTopologyPrunedHeadwaterCellCount": int(final_pruned_headwater_cells),
        "extendedLakeHeadwaterCount": int(extended_lake_headwater_count),
        "extendedLakeHeadwaterCellCount": int(extended_lake_headwater_cells),
        "runoffSource": {
            "model": (
                "normalized-climate-upland-plus-perennial-snowmelt"
                if climate_yield is not None
                else "normalized-upland-background-plus-perennial-snowmelt"
            ),
            "climateYieldApplied": climate_yield is not None,
            "climateYieldMinimum": (
                float(np.min(climate_yield[terrain]))
                if climate_yield is not None
                else None
            ),
            "climateYieldMaximum": (
                float(np.max(climate_yield[terrain]))
                if climate_yield is not None
                else None
            ),
            "snowCellCount": int(np.count_nonzero(perennial_snow)),
            "minimumWeight": float(np.min(runoff_source_weight[terrain])),
            "maximumWeight": float(np.max(runoff_source_weight[terrain])),
            "meanWeight": float(np.mean(runoff_source_weight[terrain])),
        },
        "headwaterSource": {
            "uplandElevationThreshold": upland_threshold,
            "uplandQuantile": active_config.headwater_upland_quantile,
            "lowlandFlowMultiplier": (
                active_config.lowland_headwater_flow_multiplier
            ),
            **headwater_source_stats,
        },
        "mfdMaximumAccumulation": float(np.max(mfd_accumulation, initial=0.0)),
        "mfdTerminalFlow": mfd_terminal,
        "mfd": {
            "input": mfd_input,
            "terminal": mfd_terminal,
            "conservationError": mfd_conservation_error,
            "finite": mfd_finite,
            "nonnegative": mfd_nonnegative,
        },
        "streamNetworkReachCount": int(len(network)),
        "streamClosure": closure_stats,
        "lakeOutletAnchors": [
            {
                "anchor": list(anchor),
                "outlet": list(outlet),
            }
            for _, (anchor, outlet) in sorted(
                lake_outlet_config.items(), key=lambda item: item[0]
            )
        ],
    }
    return HydrologyResult(
        burned_elevation=burned,
        corrected_elevation=corrected,
        flow_direction=flow_direction,
        accumulation=accumulation,
        mfd_accumulation=mfd_accumulation,
        valley_score=valley_score,
        valley_support=valley_score,
        valley_supported=valley_supported,
        constraint_strength=constraint_strength,
        flat_distance=flat_distance,
        stream_mask=stream_mask,
        stream_order=stream_order,
        outlet_type=outlet_type,
        downstream_index=downstream,
        land_mask=land,
        lake_mask=lake,
        network=network,
        diagnostics=diagnostics,
    )


__all__ = [
    "Cell",
    "D8_OFFSETS",
    "HydrologyConfig",
    "HydrologyError",
    "HydrologyResult",
    "accumulate_d8_runoff",
    "compute_hydrology",
]
