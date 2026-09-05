"""Deterministic island groups derived from inland-water and shelf geometry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from world_atlas.tectonic_foundation import (
    BOUNDARY_CONVERGENT,
    derive_tectonic_foundation,
)

from world_atlas.physical.procedural_landmass import (
    ellipse_margin,
    fractal_plane_noise,
    lattice_noise,
)


@dataclass(frozen=True, slots=True)
class LakeIslandField:
    island_mask: np.ndarray
    relative_elevation: np.ndarray
    island_count: int
    modified_lake_count: int


@dataclass(frozen=True, slots=True)
class ShelfArchipelagoField:
    island_mask: np.ndarray
    large_island_mask: np.ndarray
    shelf_island_mask: np.ndarray
    island_arc_mask: np.ndarray
    hotspot_island_mask: np.ndarray
    seamount_mask: np.ndarray
    shelf_mask: np.ndarray
    shelf_bathymetry: np.ndarray
    trench_mask: np.ndarray
    relative_elevation: np.ndarray
    island_count: int
    group_count: int


@dataclass(frozen=True, slots=True)
class _FragmentAnchor:
    """One rifted continental fragment and the margin it came from."""

    centre: tuple[int, int]
    source: tuple[int, int]
    radius: float
    orientation_degrees: float
    orogenic: bool


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    pending = np.asarray(mask, dtype=bool).copy()
    height, width = pending.shape
    result: list[list[tuple[int, int]]] = []
    while bool(pending.any()):
        raw_row, raw_column = np.argwhere(pending)[0]
        start = (int(raw_row), int(raw_column))
        pending[start] = False
        queue: deque[tuple[int, int]] = deque((start,))
        cells: list[tuple[int, int]] = []
        while queue:
            row, column = queue.popleft()
            cells.append((row, column))
            for delta_row, delta_column in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                next_row = row + delta_row
                next_column = column + delta_column
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and pending[next_row, next_column]
                ):
                    pending[next_row, next_column] = False
                    queue.append((next_row, next_column))
        result.append(cells)
    return result


def _interior_distance(mask: np.ndarray) -> np.ndarray:
    active = np.asarray(mask, dtype=bool)
    height, width = active.shape
    distance = np.zeros(active.shape, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for raw_row, raw_column in np.argwhere(active):
        row = int(raw_row)
        column = int(raw_column)
        if (
            row == 0
            or row == height - 1
            or column == 0
            or column == width - 1
            or not active[row - 1, column]
            or not active[row + 1, column]
            or not active[row, column - 1]
            or not active[row, column + 1]
        ):
            distance[row, column] = 1
            queue.append((row, column))
    while queue:
        row, column = queue.popleft()
        next_distance = int(distance[row, column]) + 1
        for delta_row, delta_column in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            next_row = row + delta_row
            next_column = (column + delta_column) % width
            if (
                0 <= next_row < height
                and active[next_row, next_column]
                and distance[next_row, next_column] == 0
            ):
                distance[next_row, next_column] = next_distance
                queue.append((next_row, next_column))
    return distance


def _distance_from_mask(
    source_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    maximum_distance: int,
) -> np.ndarray:
    source = np.asarray(source_mask, dtype=bool)
    active = np.asarray(active_mask, dtype=bool)
    if source.shape != active.shape:
        raise ValueError("distance masks must share a shape")
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
                and active[next_row, next_column]
                and next_distance < distance[next_row, next_column]
            ):
                distance[next_row, next_column] = next_distance
                queue.append((next_row, next_column))
    return distance


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    # Longitude is periodic.  Pad columns first with wrapping, then keep the
    # latitude boundaries finite instead of accidentally joining the poles.
    padded = np.pad(array, ((0, 0), (radius, radius)), mode="wrap")
    padded = np.pad(padded, ((radius, radius), (0, 0)), mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    size = radius * 2 + 1
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / float(size * size)


def _stable_phase(shape: tuple[int, int], ordinal: int, salt: int = 0) -> float:
    value = (
        shape[0] * 2_654_435_761
        + shape[1] * 2_246_822_519
        + ordinal * 3_266_489_917
        + salt * 668_265_263
    ) & 0xFFFFFFFF
    return value / 0xFFFFFFFF * math.tau


def classify_inland_seas(lake_mask: np.ndarray, *, minimum_area: int) -> np.ndarray:
    """Return the subset of closed inland water large enough to behave as a sea."""

    lake = np.asarray(lake_mask, dtype=bool)
    if lake.ndim != 2:
        raise ValueError("lake mask must be two-dimensional")
    if minimum_area < 1:
        raise ValueError("minimum inland-sea area must be positive")
    result = np.zeros(lake.shape, dtype=bool)
    for component in _components(lake):
        if len(component) < minimum_area:
            continue
        rows, columns = zip(*component)
        result[np.asarray(rows), np.asarray(columns)] = True
    return result


def derive_inland_sea_bathymetry(
    inland_sea_mask: np.ndarray,
    *,
    maximum_depth: float = 0.78,
) -> np.ndarray:
    """Build shallow shelves and deep closed basins for inland seas."""

    inland_sea = np.asarray(inland_sea_mask, dtype=bool)
    if inland_sea.ndim != 2:
        raise ValueError("inland-sea mask must be two-dimensional")
    if not 0.0 < maximum_depth <= 1.0:
        raise ValueError("inland-sea maximum depth must be in (0, 1]")
    result = np.zeros(inland_sea.shape, dtype=np.float64)
    for ordinal, component in enumerate(_components(inland_sea)):
        component_mask = np.zeros(inland_sea.shape, dtype=bool)
        rows, columns = zip(*component)
        component_mask[np.asarray(rows), np.asarray(columns)] = True
        distance = _interior_distance(component_mask).astype(np.float64)
        deepest = float(np.max(distance[component_mask], initial=1.0))
        normalized = np.zeros(inland_sea.shape, dtype=np.float64)
        if deepest > 1.0:
            normalized[component_mask] = (
                (distance[component_mask] - 1.0) / (deepest - 1.0)
            )
        grid_rows, grid_columns = np.indices(inland_sea.shape, dtype=np.float64)
        basin_texture = lattice_noise(
            grid_columns,
            grid_rows,
            spacing=max(7.0, math.sqrt(len(component)) * 0.09),
            seed=1291 + ordinal * 97,
        )
        shaped = np.clip(
            0.035
            + (maximum_depth - 0.035)
            * np.power(normalized, 0.68)
            * (0.92 + 0.08 * basin_texture),
            0.0,
            maximum_depth,
        )
        result[component_mask] = shaped[component_mask]
    return result


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, iterations)):
        north = np.zeros_like(result)
        south = np.zeros_like(result)
        north[1:] = result[:-1]
        south[:-1] = result[1:]
        result |= north | south | np.roll(result, 1, axis=1) | np.roll(result, -1, axis=1)
    return result


def _largest_component(mask: np.ndarray) -> np.ndarray:
    result = np.zeros(np.asarray(mask).shape, dtype=bool)
    components = _components(mask)
    if not components:
        return result
    rows, columns = zip(*max(components, key=len))
    result[np.asarray(rows), np.asarray(columns)] = True
    return result


def _wrapped_column_delta(target: float, origin: float, width: int) -> float:
    """Return the shortest signed longitudinal displacement on a periodic map."""

    return (target - origin + width / 2.0) % width - width / 2.0


def _coast_axis_orientation(
    coast: np.ndarray,
    source: tuple[int, int],
    *,
    window_radius: float,
) -> float:
    """Measure the local continental-margin tangent at a rift source."""

    rows, columns = np.nonzero(coast)
    if rows.size < 3:
        return 90.0
    delta_row = rows.astype(np.float64) - float(source[0])
    delta_column = columns.astype(np.float64) - float(source[1])
    width = coast.shape[1]
    delta_column = (delta_column + width / 2.0) % width - width / 2.0
    nearby = np.hypot(delta_row, delta_column) <= window_radius
    if int(np.count_nonzero(nearby)) < 3:
        return 90.0
    covariance = np.cov(np.vstack((delta_column[nearby], delta_row[nearby])))
    axis = np.linalg.eigh(covariance)[1][:, -1]
    return math.degrees(math.atan2(float(axis[1]), float(axis[0])))


def _orogenic_axis_orientation(
    relief: np.ndarray,
    land: np.ndarray,
    source: tuple[int, int],
    *,
    window_radius: float,
    fallback_degrees: float,
) -> float:
    """Infer the mountain-belt axis reaching one continental margin."""

    terrain = np.asarray(relief, dtype=np.float64)
    solid = np.asarray(land, dtype=bool)
    rows, columns = np.indices(solid.shape, dtype=np.float64)
    delta_row = rows - float(source[0])
    delta_column = columns - float(source[1])
    width = solid.shape[1]
    delta_column = (delta_column + width / 2.0) % width - width / 2.0
    local = solid & (np.hypot(delta_row, delta_column) <= window_radius)
    values = terrain[local]
    if values.size < 12 or float(np.ptp(values)) < 0.08:
        return fallback_degrees
    threshold = float(np.quantile(values, 0.62))
    highland = local & (terrain >= threshold)
    if int(np.count_nonzero(highland)) < 7:
        return fallback_degrees
    x = delta_column[highland]
    y = delta_row[highland]
    weights = np.square(
        np.clip(
            (terrain[highland] - threshold)
            / max(1.0e-9, float(values.max()) - threshold),
            0.10,
            1.0,
        )
    )
    x -= float(np.average(x, weights=weights))
    y -= float(np.average(y, weights=weights))
    covariance = np.asarray(
        (
            (np.average(x * x, weights=weights), np.average(x * y, weights=weights)),
            (np.average(x * y, weights=weights), np.average(y * y, weights=weights)),
        ),
        dtype=np.float64,
    )
    axis = np.linalg.eigh(covariance)[1][:, -1]
    return math.degrees(math.atan2(float(axis[1]), float(axis[0])))


def _continental_fragment_anchors(
    ocean: np.ndarray,
    land: np.ndarray,
    depth: np.ndarray,
    relief: np.ndarray,
    temperate_rows: np.ndarray,
    permitted: np.ndarray,
    *,
    base_radius: float,
    maximum_count: int,
) -> list[_FragmentAnchor]:
    """Derive microcontinents from real outward-facing continental margins.

    Empty water is deliberately not evidence.  Each returned centre is projected
    seaward from a measured coast tangent, retaining a parent-margin source that
    can later be expressed as a submerged rifted platform.
    """

    if maximum_count <= 0:
        return []
    height, width = ocean.shape
    rows, columns = np.indices(ocean.shape, dtype=np.float64)
    terrain = np.asarray(relief, dtype=np.float64)
    midlatitude_land = land & temperate_rows
    relief_values = terrain[midlatitude_land]
    if relief_values.size and float(np.ptp(relief_values)) >= 0.08:
        # Quantiles alone collapse to the lowland baseline when a narrow range
        # crosses a broad plain.  Require genuine relief above that baseline so
        # the selected outlet is the mountain belt, not an arbitrary coast cell.
        relief_floor = float(np.min(relief_values))
        highland_threshold = max(
            float(np.quantile(relief_values, 0.68)),
            relief_floor + 0.30 * float(np.ptp(relief_values)),
        )
        highland = midlatitude_land & (terrain >= highland_threshold)
        highland_distance = _distance_from_mask(
            highland,
            land,
            maximum_distance=max(8, int(round(base_radius * 2.4))),
        )
        orogenic_evidence = np.exp(
            -highland_distance.astype(np.float64) / max(3.0, base_radius * 0.52)
        )
    else:
        orogenic_evidence = np.zeros(ocean.shape, dtype=np.float64)
    west_facing = (
        land
        & np.roll(ocean, 1, axis=1)
        & temperate_rows
    )
    east_facing = (
        land
        & np.roll(ocean, -1, axis=1)
        & temperate_rows
    )
    # One tectonic sector may shed one coherent fragment family.  A wider
    # source spacing prevents a single coast from emitting several parallel
    # look-alike chains merely because the global group budget is available.
    source_spacing = max(base_radius * 3.20, height * 0.18)
    centre_spacing = base_radius * 2.35
    probe_offset = max(3, int(round(base_radius * 1.15)))
    column_indices = np.arange(width)

    def margin_score(coast: np.ndarray, outward_column: float, seed: int) -> np.ndarray:
        coast_density = _box_mean(
            coast, radius=max(2, int(round(base_radius * 0.20)))
        )
        probe_distances = (
            probe_offset,
            max(probe_offset + 2, int(round(base_radius * 1.85))),
            max(probe_offset + 4, int(round(base_radius * 2.55))),
        )
        probe_column_sets = [
            (
                column_indices
                + int(outward_column) * probe_distance
            )
            % width
            for probe_distance in probe_distances
        ]
        seaward_depth = np.mean(
            np.stack([depth[:, item] for item in probe_column_sets]),
            axis=0,
        )
        seaward_openness = np.mean(
            np.stack(
                [ocean[:, item].astype(np.float64) for item in probe_column_sets]
            ),
            axis=0,
        )
        tectonic_texture = lattice_noise(
            columns,
            rows,
            spacing=max(17.0, base_radius * 3.1),
            seed=seed,
        )
        score = (
            0.22
            + 0.20 * seaward_openness
            + 0.17 * np.clip(1.0 - seaward_depth, 0.0, 1.0)
            + 0.07 * np.clip(coast_density * 12.0, 0.0, 1.0)
            + 0.36 * orogenic_evidence
        )
        score *= 0.90 + 0.20 * (0.5 + 0.5 * tectonic_texture)
        score[~coast] = -np.inf
        return score

    west_score = margin_score(west_facing, -1.0, 1013)
    east_score = margin_score(east_facing, 1.0, 1559)
    global_score = np.maximum(west_score, east_score)
    global_coast = west_facing | east_facing
    available_coast = global_coast.copy()
    selected_centres: list[tuple[int, int]] = []
    selected_sources: list[tuple[int, int]] = []
    anchors: list[_FragmentAnchor] = []
    orogenic_extension_claimed = False
    for ordinal in range(maximum_count):
        source = _best_target(
            available_coast,
            global_score,
            selected_sources,
            minimum_spacing=source_spacing,
        )
        if source is None:
            break
        selected_sources.append(source)
        outward_column = (
            -1.0 if west_score[source] >= east_score[source] else 1.0
        )
        coast = west_facing if outward_column < 0.0 else east_facing
        # Treat a long, continuous side of one continental margin as one
        # tectonic sector.  Without this suppression the group budget creates
        # several near-parallel copies along the same coast.
        source_row_delta = rows - float(source[0])
        source_column_delta = columns - float(source[1])
        source_column_delta = (
            source_column_delta + width / 2.0
        ) % width - width / 2.0
        sector_radius = max(base_radius * 4.4, height * 0.46)
        available_coast &= ~(
            coast
            & (
                np.hypot(source_row_delta, source_column_delta)
                <= sector_radius
            )
        )
        phase = _stable_phase(ocean.shape, ordinal, 463 + int(outward_column > 0.0))
        radius_scale = float(
            np.clip(
                0.55
                + 0.58 * (0.5 + 0.5 * math.sin(phase * 1.37 + ordinal * 2.11)),
                0.55,
                1.13,
            )
        )
        effective_radius = base_radius * radius_scale
        coast_orientation = _coast_axis_orientation(
            coast,
            source,
            window_radius=max(12.0, base_radius * 2.2),
        )
        margin_orientation = _orogenic_axis_orientation(
            terrain,
            land,
            source,
            window_radius=max(18.0, base_radius * 2.9),
            fallback_degrees=coast_orientation,
        )
        axis_radians = math.radians(margin_orientation)
        direction_column = math.cos(axis_radians)
        direction_row = math.sin(axis_radians)
        if direction_column * outward_column < 0.0:
            direction_column *= -1.0
            direction_row *= -1.0
        # A world may contain many mountain belts, but only the strongest
        # outward-running sector becomes the dominant offshore continuation
        # in this scale band.  Treating every high coast as the same long arc
        # is what produced paired, copy-like tails on opposite continents.
        source_has_orogen = (
            float(orogenic_evidence[source]) >= 0.35
            and not orogenic_extension_claimed
        )
        if source_has_orogen:
            if abs(direction_column) < 0.28:
                direction_column = outward_column * 0.28
        else:
            direction_column = outward_column
            direction_row *= 0.34
        direction_norm = math.hypot(direction_row, direction_column)
        direction_row /= direction_norm
        direction_column /= direction_norm
        orientation = math.degrees(math.atan2(direction_row, direction_column))
        orientation += 11.0 * math.sin(phase)
        drift_distance = base_radius * (
            2.35 + 0.72 * radius_scale + 0.22 * math.sin(phase * 1.31)
        )
        oblique_offset = base_radius * (
            0.28 + 0.38 * (0.5 + 0.5 * math.sin(phase * 1.73))
        ) * (-1.0 if math.sin(phase * 0.91) < 0.0 else 1.0)
        target_row = (
            source[0]
            + direction_row * drift_distance
            - direction_column * oblique_offset
        )
        target_column = (
            source[1]
            + direction_column * drift_distance
            + direction_row * oblique_offset
        ) % width
        column_delta = np.minimum(
            np.abs(columns - target_column),
            width - np.abs(columns - target_column),
        )
        centre_score = (
            np.exp(
                -np.square(
                    (rows - target_row) / max(7.0, effective_radius * 0.78)
                )
            )
            * np.exp(
                -np.square(
                    column_delta / max(7.0, effective_radius * 0.88)
                )
            )
            * (0.76 + 0.24 * np.clip(1.0 - depth, 0.0, 1.0))
        )
        centre = _best_target(
            permitted,
            centre_score,
            selected_centres,
            minimum_spacing=centre_spacing,
        )
        if centre is None:
            continue
        selected_centres.append(centre)
        anchors.append(
            _FragmentAnchor(
                centre=centre,
                source=source,
                radius=effective_radius,
                orientation_degrees=orientation,
                orogenic=source_has_orogen,
            )
        )
        orogenic_extension_claimed |= source_has_orogen
    return anchors


def _rifted_platform(
    shape: tuple[int, int],
    anchor: _FragmentAnchor,
    ocean: np.ndarray,
    *,
    ordinal: int,
) -> np.ndarray:
    """Rasterize the submerged, slightly sinuous crustal trace to its parent."""

    height, width = shape
    source_row, source_column = anchor.source
    centre_row, centre_column = anchor.centre
    delta_row = float(centre_row - source_row)
    delta_column = _wrapped_column_delta(float(centre_column), float(source_column), width)
    distance = math.hypot(delta_row, delta_column)
    steps = max(3, int(math.ceil(distance * 1.35)))
    tangent = math.radians(anchor.orientation_degrees)
    phase = _stable_phase(shape, ordinal, 557)
    bend_sign = -1.0 if math.sin(phase * 1.17) < 0.0 else 1.0
    bend = anchor.radius * (
        0.44 + 0.34 * abs(math.sin(phase * 0.83))
    ) * bend_sign
    seed = np.zeros(shape, dtype=bool)
    for step in range(steps + 1):
        position = step / steps
        lateral = (
            math.sin(math.pi * position) * bend
            + math.sin(math.tau * position + phase) * anchor.radius * 0.18
        )
        row = int(
            round(
                source_row
                + delta_row * position
                - math.cos(tangent) * lateral
            )
        )
        column = int(
            round(
                source_column
                + delta_column * position
                + math.sin(tangent) * lateral
            )
        ) % width
        if 0 <= row < height:
            seed[row, column] = True
    platform_width = max(5, int(round(anchor.radius * 0.30)))
    distance_to_trace = _distance_from_mask(
        seed,
        ocean,
        maximum_distance=platform_width + 3,
    )
    rows, columns = np.indices(shape, dtype=np.float64)
    texture = lattice_noise(
        columns,
        rows,
        spacing=max(4.0, platform_width * 1.35),
        seed=2141 + ordinal * 149,
    )
    delta_rows = rows - float(source_row)
    delta_columns = columns - float(source_column)
    delta_columns = (delta_columns + width / 2.0) % width - width / 2.0
    along = np.clip(
        (delta_rows * delta_row + delta_columns * delta_column)
        / max(distance * distance, 1.0),
        0.0,
        1.0,
    )
    # Foundered crust alternates between exposed banks and deeper saddles.
    # A one-cell spine keeps the tectonic connection continuous below sea
    # level, while broad shallow water only appears around irregular blocks.
    bank_signal = 0.5 + 0.5 * np.sin(
        along * math.tau * (2.1 + 0.37 * (ordinal % 3))
        + phase
        + 0.72 * texture
    )
    bank_signal = np.power(np.clip(bank_signal, 0.0, 1.0), 1.55)
    endpoint_lobes = np.maximum(
        np.exp(-np.square(along / 0.17)),
        np.exp(-np.square((1.0 - along) / 0.22)),
    )
    local_width = platform_width * (
        0.18
        + 0.18 * texture
        + 0.74 * np.maximum(bank_signal, endpoint_lobes)
    )
    platform = ocean & (distance_to_trace <= local_width)
    # Preserve a narrow continuous crustal spine while the broader platform
    # breaks into banks, embayments, and drowned fault blocks around it.
    platform |= _dilate(seed, 1) & ocean
    return platform


def _structural_ridge(
    delta_column: np.ndarray,
    delta_row: np.ndarray,
    *,
    radius: float,
    orientation_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one curved orogenic axis used by both landform and relief."""

    angle = math.radians(orientation_degrees)
    major = delta_column * math.cos(angle) + delta_row * math.sin(angle)
    minor = -delta_column * math.sin(angle) + delta_row * math.cos(angle)
    curved_axis = minor - 0.09 * radius * np.sin(
        major / max(radius, 1.0) * 2.4
    )
    ridge = np.exp(
        -np.square(curved_axis / max(1.2, radius * 0.12))
    ) * np.exp(-np.power(major / max(radius * 1.2, 1.0), 4))
    return major, minor, curved_axis, ridge


def _island_axis_orientation(
    island: np.ndarray,
    centre: tuple[int, int],
    *,
    fallback_degrees: float,
) -> float:
    """Measure the surviving island grain after bays and shelf clipping."""

    rows, columns = np.nonzero(island)
    if rows.size < 3:
        return fallback_degrees
    delta_row = rows.astype(np.float64) - float(centre[0])
    delta_column = columns.astype(np.float64) - float(centre[1])
    width = island.shape[1]
    delta_column = (delta_column + width / 2.0) % width - width / 2.0
    covariance = np.cov(np.vstack((delta_column, delta_row)))
    axis = np.linalg.eigh(covariance)[1][:, -1]
    return math.degrees(math.atan2(float(axis[1]), float(axis[0])))


def _orthogonal_neighbour_count(mask: np.ndarray) -> np.ndarray:
    active = np.asarray(mask, dtype=bool)
    north = np.zeros_like(active)
    south = np.zeros_like(active)
    north[1:] = active[:-1]
    south[:-1] = active[1:]
    return (
        north.astype(np.int8)
        + south.astype(np.int8)
        + np.roll(active, 1, axis=1).astype(np.int8)
        + np.roll(active, -1, axis=1).astype(np.int8)
    )


def _clean_coast_cells(mask: np.ndarray, permitted: np.ndarray) -> np.ndarray:
    """Remove pixel noise while retaining narrow capes and embayments."""

    result = np.asarray(mask, dtype=bool).copy()
    allowed = np.asarray(permitted, dtype=bool)
    for _ in range(2):
        neighbours = _orthogonal_neighbour_count(result)
        result &= neighbours > 0
        result |= allowed & ~result & (neighbours == 4)
    return _largest_component(result & allowed)


def _break_straight_coast_runs(
    mask: np.ndarray,
    *,
    maximum_run: int = 10,
) -> np.ndarray:
    """Cut small deterministic coves into ruler-straight raster shore runs."""

    result = np.asarray(mask, dtype=bool).copy()
    if maximum_run < 4:
        raise ValueError("maximum coast run must be at least four cells")
    for pass_ordinal in range(3):
        neighbours = _orthogonal_neighbour_count(result)
        coast = result & (neighbours < 4)
        remove = np.zeros(result.shape, dtype=bool)
        for transposed in (False, True):
            lines = coast.T if transposed else coast
            neighbour_lines = neighbours.T if transposed else neighbours
            for line_index, line in enumerate(lines):
                padded = np.pad(line.astype(np.int8), (1, 1))
                changes = np.flatnonzero(np.diff(padded))
                for start, end in zip(changes[::2], changes[1::2]):
                    length = int(end - start)
                    if length <= maximum_run:
                        continue
                    first = int(start + maximum_run - 2 + (line_index + pass_ordinal) % 3)
                    for position in range(first, int(end) - 1, maximum_run - 2):
                        local = slice(max(int(start) + 1, position - 1), min(int(end) - 1, position + 2))
                        candidates = np.flatnonzero(
                            line[local] & (neighbour_lines[line_index, local] >= 2)
                        )
                        if candidates.size == 0:
                            continue
                        chosen = int(local.start + candidates[candidates.size // 2])
                        if transposed:
                            remove[chosen, line_index] = True
                        else:
                            remove[line_index, chosen] = True
        if not bool(remove.any()):
            break
        result[remove] = False
        result = _largest_component(result)
    return result


def _longest_true_run(values: np.ndarray) -> int:
    line = np.asarray(values, dtype=bool).reshape(-1)
    if not bool(line.any()):
        return 0
    padded = np.pad(line.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return int(np.max(changes[1::2] - changes[::2], initial=0))


def _refine_island_coast(
    island: np.ndarray,
    permitted: np.ndarray,
    centre: tuple[float, float],
    *,
    radius: float,
    orientation_radians: float,
    phase: float,
    roughness: float,
    embayment: float,
) -> np.ndarray:
    """Add deterministic 1–5 cell coastal detail without redrawing the island."""

    base = np.asarray(island, dtype=bool)
    allowed = np.asarray(permitted, dtype=bool)
    if base.shape != allowed.shape:
        raise ValueError("coast masks must share a shape")
    if radius <= 0.0 or roughness < 0.0 or embayment < 0.0:
        raise ValueError("coast-detail parameters must be non-negative")
    if not bool(base.any()):
        return base.copy()
    if radius < 3.2 or int(np.count_nonzero(base)) < 12:
        return base.copy()

    inside = _interior_distance(base)
    outside = _distance_from_mask(
        base,
        allowed & ~base,
        maximum_distance=5,
    )
    rows, columns = np.indices(base.shape, dtype=np.float64)
    delta_row = rows - centre[0]
    delta_column = columns - centre[1]
    delta_column = (
        (delta_column + base.shape[1] / 2.0) % base.shape[1]
        - base.shape[1] / 2.0
    )
    cosine = math.cos(orientation_radians)
    sine = math.sin(orientation_radians)
    major = delta_column * cosine + delta_row * sine
    minor = -delta_column * sine + delta_row * cosine
    seed = 2309 + int(phase * 1009.0) % 100_003
    medium = lattice_noise(major, minor, spacing=3.6, seed=seed)
    fine = lattice_noise(major, minor, spacing=1.65, seed=seed + 53)
    valleys = np.square(
        np.sin(major / max(radius, 1.0) * 7.0 + phase)
    )
    signed = np.where(base, inside - 0.5, -(outside - 0.5)).astype(
        np.float64
    )
    perturbation = (
        roughness * (1.08 * medium + 0.52 * fine)
        - embayment * valleys * 0.62
    )
    coastal_band = (
        (base & (inside <= 5)) | (~base & (outside <= 4))
    ) & allowed
    refined = base.copy()
    refined[coastal_band] = (
        signed[coastal_band] + perturbation[coastal_band] >= 0.0
    )
    refined[inside > 5] = True
    cleaned = _clean_coast_cells(refined & allowed, allowed)
    intersection = int(np.count_nonzero(base & cleaned))
    union = int(np.count_nonzero(base | cleaned))
    # The refinement changes only a five-cell coastal band.  Requiring a 0.90
    # whole-mask Jaccard nevertheless rejected every useful inlet on medium
    # islands; 0.84 retains the continental silhouette while allowing visible
    # headlands, coves, and fault-cut shore detail to survive.
    required_similarity = 0.84 if roughness >= 1.5 else 0.90
    if union == 0 or intersection / union < required_similarity:
        return (
            _break_straight_coast_runs(base)
            if roughness >= 1.5
            else base.copy()
        )

    def coast_density(mask: np.ndarray) -> float:
        area = int(np.count_nonzero(mask))
        if area == 0:
            return 0.0
        neighbours = _orthogonal_neighbour_count(mask)
        perimeter = int(np.sum(4 - neighbours[mask]))
        return perimeter / math.sqrt(float(area))

    if coast_density(cleaned) < coast_density(base) * 1.03:
        return (
            _break_straight_coast_runs(base)
            if roughness >= 1.5
            else base.copy()
        )
    return _break_straight_coast_runs(cleaned) if roughness >= 1.5 else cleaned


def _large_island_field(
    permitted: np.ndarray,
    centre: tuple[int, int],
    *,
    radius: float,
    ordinal: int,
    orientation_degrees: float,
) -> np.ndarray:
    """Generate one microcontinent with the same coast grammar as a polar cap."""

    rows, columns = np.indices(permitted.shape, dtype=np.float64)
    delta_row = rows - float(centre[0])
    delta_column = columns - float(centre[1])
    width = permitted.shape[1]
    delta_column = (delta_column + width / 2.0) % width - width / 2.0
    orientation = orientation_degrees
    radius_x = radius * (1.18 + 0.11 * (ordinal % 3))
    radius_y = radius * (0.72 + 0.08 * ((ordinal + 1) % 3))
    field = ellipse_margin(
        delta_column,
        delta_row,
        0.0,
        0.0,
        radius_x,
        radius_y,
        orientation,
    )
    orientation_radians = math.radians(orientation)
    along_x = math.cos(orientation_radians)
    along_y = math.sin(orientation_radians)
    cross_x = -along_y
    cross_y = along_x
    forward_lobe = ellipse_margin(
        delta_column,
        delta_row,
        along_x * radius * (0.48 + 0.08 * (ordinal % 2)),
        along_y * radius * (0.48 + 0.08 * (ordinal % 2)),
        radius_x * (0.58 + 0.05 * (ordinal % 3)),
        radius_y * 0.62,
        orientation + 12.0 - 7.0 * (ordinal % 3),
    ) - 0.07
    side_lobe = ellipse_margin(
        delta_column,
        delta_row,
        -along_x * radius * 0.28 + cross_x * radius * (0.25 - 0.12 * (ordinal % 3)),
        -along_y * radius * 0.28 + cross_y * radius * (0.25 - 0.12 * (ordinal % 3)),
        radius_x * (0.43 + 0.07 * ((ordinal + 1) % 3)),
        radius_y * 0.48,
        orientation - 24.0 + 11.0 * ordinal,
    ) - 0.12
    field = np.maximum(field, np.maximum(forward_lobe, side_lobe))
    coordinate_scale = 10.0 / max(radius, 1.0)
    local_x = delta_column * coordinate_scale
    local_y = delta_row * coordinate_scale
    broad = lattice_noise(local_x, local_y, spacing=7.8, seed=1601 + ordinal * 113)
    medium = lattice_noise(local_x, local_y, spacing=3.7, seed=1663 + ordinal * 113)
    fine = fractal_plane_noise(local_x, local_y, seed=1721 + ordinal * 113)
    coast_envelope = 0.24 + 0.76 * np.exp(-np.square(field / 0.64))
    _major, _minor, _curved_axis, tectonic_ridge = _structural_ridge(
        delta_column,
        delta_row,
        radius=radius,
        orientation_degrees=orientation,
    )
    strength = (
        field
        + (0.34 * broad + 0.24 * medium + 0.18 * fine) * coast_envelope
        + 0.34 * tectonic_ridge * coast_envelope
    )

    # Cut a small number of asymmetrical drowned valleys through the coast.
    # Their centres straddle the zero contour, producing bays and headlands
    # while retaining one coherent microcontinent rather than random holes.
    for bay_ordinal in range(2 + ordinal % 2):
        bay_angle = _stable_phase(
            permitted.shape, ordinal * 5 + bay_ordinal, 181 + ordinal
        )
        bay_x = math.cos(bay_angle) * radius_x * (0.70 + 0.05 * bay_ordinal)
        bay_y = math.sin(bay_angle) * radius_y * (0.70 + 0.04 * bay_ordinal)
        bay = ellipse_margin(
            delta_column,
            delta_row,
            bay_x,
            bay_y,
            radius * (0.38 + 0.04 * (bay_ordinal % 2)),
            radius * (0.22 + 0.025 * ((ordinal + bay_ordinal) % 2)),
            math.degrees(bay_angle) + 24.0,
        )
        ridge_protection = 1.0 - 0.72 * tectonic_ridge
        strength -= (
            (0.72 + 0.08 * bay_ordinal)
            * np.clip(bay, 0.0, 1.0)
            * ridge_protection
        )
    return _largest_component(permitted & (strength >= 0.0))


def _write_large_island_relief(
    island: np.ndarray,
    relative_elevation: np.ndarray,
    centre: tuple[int, int],
    *,
    radius: float,
    ordinal: int,
    orientation_degrees: float,
) -> None:
    """Give a microcontinent textured ridges instead of a concentric mound."""

    interior = _interior_distance(island).astype(np.float64)
    peak = float(np.max(interior[island], initial=1.0))
    normalized = np.zeros(island.shape, dtype=np.float64)
    if peak > 1.0:
        normalized[island] = (interior[island] - 1.0) / (peak - 1.0)
    rows, columns = np.indices(island.shape, dtype=np.float64)
    delta_row = rows - float(centre[0])
    delta_column = columns - float(centre[1])
    width = island.shape[1]
    delta_column = (delta_column + width / 2.0) % width - width / 2.0
    surviving_orientation = _island_axis_orientation(
        island,
        centre,
        fallback_degrees=orientation_degrees,
    )
    major, minor, _curved_axis, _ridge = _structural_ridge(
        delta_column,
        delta_row,
        radius=radius,
        orientation_degrees=surviving_orientation,
    )
    phase = _stable_phase(island.shape, ordinal, 719)
    coordinate_scale = 10.0 / max(radius, 1.0)
    texture = fractal_plane_noise(
        delta_column * coordinate_scale,
        delta_row * coordinate_scale,
        seed=1871 + ordinal * 131,
    )
    broad_texture = lattice_noise(
        major,
        minor,
        spacing=max(3.0, radius * 0.34),
        seed=1931 + ordinal * 137,
    )
    along_texture = lattice_noise(
        major,
        minor * 0.16,
        spacing=max(2.5, radius * 0.29),
        seed=1993 + ordinal * 139,
    )
    axis_wander = radius * (
        0.095 * np.sin(major / max(radius, 1.0) * 2.1 + phase)
        + 0.045 * np.sin(major / max(radius, 1.0) * 5.3 - phase * 0.61)
    )
    local_width = radius * (
        0.095 + 0.055 * (0.5 + 0.5 * broad_texture)
    )
    main_range = np.exp(
        -np.square((minor - axis_wander) / np.maximum(local_width, 1.0))
    ) * np.exp(-np.power(major / max(radius * 1.18, 1.0), 4))
    erosion = np.clip(
        0.34
        + 0.44 * (0.5 + 0.5 * along_texture)
        + 0.22 * (0.5 + 0.5 * texture),
        0.20,
        1.0,
    )
    main_range *= erosion

    branch_ranges = np.zeros(island.shape, dtype=np.float64)
    branch_count = 2 + ordinal % 2
    for branch_ordinal in range(branch_count):
        fraction = (branch_ordinal + 1.0) / (branch_count + 1.0)
        origin_major = radius * (
            -0.66 + 1.32 * fraction + 0.10 * math.sin(phase + branch_ordinal)
        )
        origin_cross = radius * 0.08 * math.sin(
            origin_major / max(radius, 1.0) * 2.1 + phase
        )
        turn = (0.34 + 0.16 * math.sin(phase * 0.73 + branch_ordinal * 1.91)) * (
            -1.0 if branch_ordinal % 2 else 1.0
        )
        local_major = major - origin_major
        local_minor = minor - origin_cross
        branch_along = (
            local_major * math.cos(turn) + local_minor * math.sin(turn)
        )
        branch_cross = (
            -local_major * math.sin(turn) + local_minor * math.cos(turn)
        )
        branch_length = radius * (
            0.42 + 0.11 * (0.5 + 0.5 * math.sin(phase + branch_ordinal * 2.3))
        )
        branch_width = radius * (
            0.072 + 0.025 * (0.5 + 0.5 * math.cos(phase * 1.2 + branch_ordinal))
        )
        branch = np.exp(
            -np.square(branch_cross / max(branch_width, 0.9))
        ) * np.exp(
            -np.power(
                (branch_along - radius * 0.12) / max(branch_length, 1.0),
                4,
            )
        )
        branch *= np.clip(
            0.30
            + 0.46 * (0.5 + 0.5 * broad_texture)
            + 0.24 * (0.5 + 0.5 * texture),
            0.18,
            1.0,
        )
        branch_ranges = np.maximum(branch_ranges, branch)

    parallel_axis = minor - axis_wander - radius * (
        0.28 + 0.04 * math.sin(phase)
    )
    parallel_range = np.exp(
        -np.square(parallel_axis / max(1.2, radius * 0.105))
    ) * np.exp(
        -np.power((major + radius * 0.14) / max(radius * 0.68, 1.0), 4)
    )
    parallel_range *= np.clip(0.24 + 0.62 * (0.5 + 0.5 * texture), 0.12, 0.86)
    mountain_field = np.maximum.reduce(
        (0.88 * main_range, 0.88 * branch_ranges, 0.66 * parallel_range)
    )
    summit_texture = lattice_noise(
        major,
        minor,
        spacing=max(2.2, radius * 0.19),
        seed=2053 + ordinal * 149,
    )
    summit_texture = np.clip(
        0.50 + 0.50 * summit_texture + 0.16 * texture,
        0.0,
        1.0,
    )
    mountain_field *= 0.16 + 0.84 * np.power(summit_texture, 1.28)
    mountain_envelope = np.clip(normalized * 3.4, 0.0, 1.0)
    mountain_gain, ceiling = (
        (0.40, 0.57),
        (0.44, 0.57),
        (0.37, 0.52),
        (0.46, 0.64),
    )[ordinal % 4]
    relief = (
        0.025
        + 0.095 * np.power(normalized, 0.68)
        + 0.065 * texture * mountain_envelope
        + mountain_gain * mountain_field * mountain_envelope
    )
    relative_elevation[island] = np.clip(relief[island], 0.025, ceiling)


def _best_target(
    candidate: np.ndarray,
    score: np.ndarray,
    occupied_centres: list[tuple[int, int]],
    *,
    minimum_spacing: float,
) -> tuple[int, int] | None:
    ranked = np.asarray(score, dtype=np.float64).copy()
    ranked[~candidate] = -np.inf
    rows, columns = np.indices(candidate.shape, dtype=np.float64)
    for other_row, other_column in occupied_centres:
        column_delta = np.minimum(
            np.abs(columns - other_column),
            candidate.shape[1] - np.abs(columns - other_column),
        )
        ranked[np.hypot(rows - other_row, column_delta) < minimum_spacing] = -np.inf
    flat = int(np.argmax(ranked))
    if not np.isfinite(ranked.reshape(-1)[flat]):
        return None
    row, column = np.unravel_index(flat, ranked.shape)
    return int(row), int(column)


def _choose_lake_centres(
    component_mask: np.ndarray,
    shore_distance: np.ndarray,
    *,
    count: int,
    radius: float,
    shore_clearance: int,
) -> list[tuple[int, int]]:
    candidates = np.argwhere(
        component_mask & (shore_distance >= int(math.ceil(radius)) + shore_clearance)
    )
    centres: list[tuple[int, int]] = []
    minimum_spacing_squared = (radius * 3.2) ** 2
    for ordinal in range(count):
        best: tuple[float, int, int] | None = None
        phase = _stable_phase(component_mask.shape, ordinal, int(np.count_nonzero(component_mask)))
        for raw_row, raw_column in candidates:
            row = int(raw_row)
            column = int(raw_column)
            separation_squared = (
                min((row - other_row) ** 2 + (column - other_column) ** 2 for other_row, other_column in centres)
                if centres
                else 1.0
            )
            if centres and separation_squared < minimum_spacing_squared:
                continue
            score = float(shore_distance[row, column]) * math.sqrt(separation_squared)
            score *= 1.0 + 0.1 * math.sin(row * 0.754877666 + column * 0.569840296 + phase)
            candidate = (score, -row, -column)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            break
        centres.append((-best[1], -best[2]))
    return centres


def _organic_ellipse(
    permitted: np.ndarray,
    centre: tuple[float, float],
    *,
    radius: float,
    orientation: float,
    phase: float,
    wrap_columns: bool = False,
) -> np.ndarray:
    rows, columns = np.indices(permitted.shape, dtype=np.float64)
    delta_row = rows - centre[0]
    delta_column = columns - centre[1]
    if wrap_columns:
        width = permitted.shape[1]
        delta_column = (delta_column + width / 2.0) % width - width / 2.0
    cosine = math.cos(orientation)
    sine = math.sin(orientation)
    major = delta_column * cosine + delta_row * sine
    minor = -delta_column * sine + delta_row * cosine
    shape_phase = 0.5 + 0.5 * math.sin(phase * 1.61803398875 + 0.7)
    aspect = 1.02 + 0.68 * shape_phase
    normalized_major = major / (radius * aspect)
    normalized_minor = minor / (radius / aspect)
    normalized_radius = np.hypot(normalized_major, normalized_minor)
    angle = np.arctan2(normalized_minor, normalized_major)
    primary_lobes = 2.0 + float(int(phase / math.tau * 5.0) % 3)
    primary_amplitude = 0.07 + 0.09 * (0.5 + 0.5 * math.cos(phase * 1.37))
    secondary_amplitude = 0.035 + 0.055 * shape_phase
    edge = (
        1.0
        + primary_amplitude * np.sin(primary_lobes * angle + phase)
        + secondary_amplitude * np.sin((primary_lobes + 3.0) * angle - phase * 0.73)
        + 0.025 * np.sin(7.0 * angle + phase * 1.91)
    )
    coastal_noise = lattice_noise(
        delta_column,
        delta_row,
        spacing=max(1.15, radius * 0.30),
        seed=1901 + int(phase * 997.0) % 100_003,
    )
    edge += (0.07 + 0.045 * shape_phase) * coastal_noise
    notch_angle = phase * 0.73 + 1.1
    angular_delta = np.arctan2(
        np.sin(angle - notch_angle), np.cos(angle - notch_angle)
    )
    notch_strength = 0.16 * float(np.clip((radius - 1.5) / 3.5, 0.0, 1.0))
    edge -= notch_strength * np.exp(-np.square(angular_delta / 0.24))
    return permitted & (normalized_radius <= edge)


def _write_relief(
    island: np.ndarray,
    relative_elevation: np.ndarray,
    *,
    base: float,
    amplitude: float,
) -> None:
    interior = _interior_distance(island).astype(np.float64)
    peak = float(np.max(interior[island], initial=1.0))
    normalized = np.zeros(island.shape, dtype=np.float64)
    if peak > 1.0:
        normalized[island] = (interior[island] - 1.0) / (peak - 1.0)
    relative_elevation[island] = base + amplitude * normalized[island]


def _write_structural_relief(
    island: np.ndarray,
    relative_elevation: np.ndarray,
    centre: tuple[float, float],
    *,
    radius: float,
    orientation_radians: float,
    phase: float,
    base: float,
    amplitude: float,
) -> None:
    """Raise an island around the same tectonic axis that shaped its coast."""

    interior = _interior_distance(island).astype(np.float64)
    peak = float(np.max(interior[island], initial=1.0))
    normalized = np.zeros(island.shape, dtype=np.float64)
    if peak > 1.0:
        normalized[island] = (interior[island] - 1.0) / (peak - 1.0)
    rows, columns = np.indices(island.shape, dtype=np.float64)
    delta_row = rows - centre[0]
    delta_column = columns - centre[1]
    width = island.shape[1]
    delta_column = (delta_column + width / 2.0) % width - width / 2.0
    major = (
        delta_column * math.cos(orientation_radians)
        + delta_row * math.sin(orientation_radians)
    )
    minor = (
        -delta_column * math.sin(orientation_radians)
        + delta_row * math.cos(orientation_radians)
    )
    curved_axis = minor - 0.12 * radius * np.sin(
        major / max(radius, 1.0) * 2.1 + phase * 0.17
    )
    ridge = np.exp(
        -np.square(curved_axis / max(0.8, radius * 0.20))
    ) * np.exp(-np.power(major / max(radius * 1.35, 1.0), 4))
    along_variation = 0.78 + 0.22 * np.sin(
        major / max(radius, 1.0) * 3.2 + phase
    )
    texture = lattice_noise(
        delta_column,
        delta_row,
        spacing=max(1.1, radius * 0.38),
        seed=2027 + int(phase * 991.0) % 100_003,
    )
    mountain_envelope = np.clip(normalized * 4.2, 0.0, 1.0)
    relief = base + amplitude * (
        0.22 * np.power(normalized, 0.72)
        + 0.66 * ridge * along_variation * mountain_envelope
        + 0.12 * (0.5 + 0.5 * texture) * mountain_envelope
    )
    relative_elevation[island] = np.clip(
        relief[island], base, base + amplitude
    )


def _authored_island_shelves(
    land: np.ndarray,
    ocean: np.ndarray,
    temperate_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Give inherited non-continental islands their missing depth apron."""

    shelf = np.zeros(land.shape, dtype=bool)
    shelf_depth = np.ones(land.shape, dtype=np.float64)
    maximum_island_area = max(64, int(round(land.size * 0.045)))
    for component in _components(land & temperate_rows):
        area = len(component)
        if area < 10 or area > maximum_island_area:
            continue
        component_mask = np.zeros(land.shape, dtype=bool)
        rows, columns = zip(*component)
        component_mask[np.asarray(rows), np.asarray(columns)] = True
        width = int(np.clip(round(math.sqrt(area) * 0.22), 5, 18))
        distance = _distance_from_mask(
            component_mask,
            ocean,
            maximum_distance=width,
        )
        local = ocean & (distance >= 1) & (distance <= width)
        graded = 0.075 + 0.50 * np.power(
            np.clip(distance.astype(np.float64) / width, 0.0, 1.0),
            0.78,
        )
        shelf |= local
        shelf_depth[local] = np.minimum(shelf_depth[local], graded[local])
    return shelf, shelf_depth


def _generate_drowned_margin_archipelagos(
    ocean: np.ndarray,
    land: np.ndarray,
    depth: np.ndarray,
    ordinary_candidate: np.ndarray,
    shore_distance: np.ndarray,
    occupied: np.ndarray,
    relative_elevation: np.ndarray,
    *,
    maximum_groups: int,
    maximum_islands_per_group: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Raise residual highs inside broad, shallow, embayed continental shelves."""

    if maximum_groups <= 0:
        return (
            np.zeros(ocean.shape, dtype=bool),
            np.zeros(ocean.shape, dtype=bool),
            [],
        )
    height, width = ocean.shape
    radius = max(10, int(round(min(ocean.shape) * 0.065)))
    context_radius = max(radius, int(round(min(ocean.shape) * 0.145)))
    surrounding_land = _box_mean(land, radius=context_radius)
    local_shelf = _box_mean(ordinary_candidate, radius=max(5, radius // 2))
    directional_land = _box_mean(land, radius=max(3, radius // 3))
    directional_offset = max(10, int(round(radius * 2.05)))
    north_land = np.zeros(ocean.shape, dtype=np.float64)
    south_land = np.zeros(ocean.shape, dtype=np.float64)
    north_land[directional_offset:] = directional_land[:-directional_offset]
    south_land[:-directional_offset] = directional_land[directional_offset:]
    west_land = np.roll(directional_land, directional_offset, axis=1)
    east_land = np.roll(directional_land, -directional_offset, axis=1)
    opposed_land = np.maximum(
        np.minimum(north_land, south_land),
        np.minimum(west_land, east_land),
    )
    bay_candidate = (
        ordinary_candidate
        & (surrounding_land >= 0.075)
        & (surrounding_land <= 0.52)
        & (local_shelf >= 0.18)
        & (opposed_land >= 0.035)
    )
    rows, columns = np.indices(ocean.shape, dtype=np.float64)
    bay_score = (
        surrounding_land
        * (0.48 + local_shelf)
        * (0.52 + np.clip(1.0 - depth, 0.0, 1.0))
        * (0.20 + 2.10 * opposed_land)
        * (1.0 + 0.035 * np.sin(rows * 0.21 + columns * 0.17))
    )
    available_distances = shore_distance[bay_candidate]
    if available_distances.size:
        target_distance = float(
            np.clip(np.quantile(available_distances, 0.62), 10.0, 28.0)
        )
        distance_preference = np.exp(
            -np.square(
                (shore_distance.astype(np.float64) - target_distance)
                / max(5.0, target_distance * 0.48)
            )
        )
        bay_score *= 0.18 + 0.82 * distance_preference
    gradient_rows = np.gradient(shore_distance.astype(np.float64), axis=0)
    gradient_columns = (
        np.roll(shore_distance.astype(np.float64), -1, axis=1)
        - np.roll(shore_distance.astype(np.float64), 1, axis=1)
    ) * 0.5
    group_centres: list[tuple[int, int]] = []
    island_mask = np.zeros(ocean.shape, dtype=bool)
    shelf_mask = np.zeros(ocean.shape, dtype=bool)
    working_occupied = np.asarray(occupied, dtype=bool).copy()
    group_limit = min(3, maximum_groups)
    for group_ordinal in range(group_limit):
        centre = _best_target(
            bay_candidate & ~_dilate(working_occupied, 4),
            bay_score,
            group_centres,
            minimum_spacing=max(24.0, min(ocean.shape) * 0.12),
        )
        if centre is None:
            break
        group_centres.append(centre)
        gradient = np.asarray(
            (gradient_rows[centre], gradient_columns[centre]), dtype=np.float64
        )
        norm = float(np.linalg.norm(gradient))
        if norm <= 1.0e-9:
            phase = _stable_phase(ocean.shape, group_ordinal, 811)
            tangent = np.asarray((math.sin(phase), math.cos(phase)), dtype=np.float64)
        else:
            gradient /= norm
            tangent = np.asarray((-gradient[1], gradient[0]), dtype=np.float64)
        normal = np.asarray((tangent[1], -tangent[0]), dtype=np.float64)
        desired = min(maximum_islands_per_group, 7 + group_ordinal)
        group_islands = np.zeros(ocean.shape, dtype=bool)
        for island_ordinal in range(desired):
            phase = _stable_phase(
                ocean.shape, island_ordinal, 839 + group_ordinal
            )
            position = island_ordinal - (desired - 1) / 2.0
            along = position * (5.2 + 0.35 * group_ordinal)
            across = math.sin(island_ordinal * 2.19 + phase) * (
                4.0 + 1.2 * (island_ordinal % 3)
            )
            target_row = centre[0] + tangent[0] * along + normal[0] * across
            target_column = (
                centre[1] + tangent[1] * along + normal[1] * across
            ) % width
            row = int(round(float(target_row)))
            column = int(round(float(target_column))) % width
            if not 0 <= row < height or not ordinary_candidate[row, column]:
                continue
            # Shelf/depth/latitude masks select a geologically plausible
            # centre; they must never act as cookie cutters on the coastline.
            permitted = ocean & ~_dilate(working_occupied, 2)
            radius_scale = (4.8, 3.4, 2.6, 3.9, 2.3, 3.0, 2.1, 2.7, 1.9)[
                island_ordinal % 9
            ]
            orientation = math.atan2(float(tangent[0]), float(tangent[1]))
            orientation += 0.24 * math.sin(phase)
            island = _organic_ellipse(
                permitted,
                (float(row), float(column)),
                radius=radius_scale,
                orientation=orientation,
                phase=phase,
                wrap_columns=True,
            )
            island = _refine_island_coast(
                island,
                permitted,
                (float(row), float(column)),
                radius=radius_scale,
                orientation_radians=orientation,
                phase=phase,
                roughness=0.88,
                embayment=0.38,
            )
            if int(np.count_nonzero(island)) < 6:
                continue
            island_mask |= island
            group_islands |= island
            working_occupied |= island
            _write_structural_relief(
                island,
                relative_elevation,
                (float(row), float(column)),
                radius=radius_scale,
                orientation_radians=orientation,
                phase=phase,
                base=0.016,
                amplitude=0.18,
            )
        if bool(group_islands.any()):
            distance = _distance_from_mask(
                group_islands,
                ocean,
                maximum_distance=7,
            )
            shelf_mask |= ocean & (distance <= 7)
    return island_mask, shelf_mask, group_centres


def generate_inland_water_islands(
    lake_mask: np.ndarray,
    *,
    minimum_lake_area: int,
    maximum_islands: int = 4,
    shore_clearance: int = 3,
) -> LakeIslandField:
    lake = np.asarray(lake_mask, dtype=bool)
    if lake.ndim != 2:
        raise ValueError("lake mask must be two-dimensional")
    if minimum_lake_area < 1 or maximum_islands < 1 or shore_clearance < 1:
        raise ValueError("inland island parameters must be positive")
    island_mask = np.zeros(lake.shape, dtype=bool)
    relative_elevation = np.zeros(lake.shape, dtype=np.float64)
    island_count = 0
    modified_lake_count = 0
    for component in _components(lake):
        area = len(component)
        if area < minimum_lake_area:
            continue
        component_mask = np.zeros(lake.shape, dtype=bool)
        rows, columns = zip(*component)
        component_mask[np.asarray(rows), np.asarray(columns)] = True
        shore_distance = _interior_distance(component_mask)
        radius = float(np.clip(math.sqrt(area) * 0.05, 8.0, 14.0))
        desired_count = int(np.clip(round(math.sqrt(area) / 48.0), 1, maximum_islands))
        centres = _choose_lake_centres(
            component_mask,
            shore_distance,
            count=desired_count,
            radius=radius,
            shore_clearance=shore_clearance,
        )
        added = 0
        for ordinal, centre in enumerate(centres):
            phase = _stable_phase(lake.shape, ordinal, area)
            island = _organic_ellipse(
                component_mask & (shore_distance > shore_clearance),
                centre,
                radius=radius * (0.88 + 0.08 * ordinal),
                orientation=phase * 0.5,
                phase=phase,
            ) & ~island_mask
            if int(np.count_nonzero(island)) < 12:
                continue
            _write_relief(island, relative_elevation, base=0.015, amplitude=0.265)
            island_mask |= island
            island_count += 1
            added += 1
        if added:
            modified_lake_count += 1
    return LakeIslandField(
        island_mask=island_mask,
        relative_elevation=relative_elevation,
        island_count=island_count,
        modified_lake_count=modified_lake_count,
    )


def generate_shelf_archipelagos(
    ocean_mask: np.ndarray,
    land_mask: np.ndarray,
    bathymetry: np.ndarray,
    land_relief: np.ndarray,
    *,
    maximum_groups: int = 8,
    maximum_islands_per_group: int = 6,
    minimum_shore_distance: int = 4,
    maximum_shore_distance: int = 34,
    shallow_limit: float = 0.68,
) -> ShelfArchipelagoField:
    """Generate islands as four distinct geologic families.

    Microcontinents and their shelf islands share a broad platform.  Volcanic
    island arcs follow a curved ridge landward of a parallel trench.  Hotspot
    trails cross deep ocean with an explicit young-to-old size gradient.
    """

    ocean = np.asarray(ocean_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    depth = np.asarray(bathymetry, dtype=np.float64)
    relief = np.asarray(land_relief, dtype=np.float64)
    if (
        ocean.ndim != 2
        or land.shape != ocean.shape
        or depth.shape != ocean.shape
        or relief.shape != ocean.shape
    ):
        raise ValueError("archipelago fields must be two-dimensional and share a shape")
    if bool(np.any(ocean & land)):
        raise ValueError("archipelago ocean and land masks must be disjoint")
    if not bool(np.all(np.isfinite(depth))) or not bool(np.all(np.isfinite(relief))):
        raise ValueError("archipelago bathymetry and relief must be finite")
    if maximum_groups < 1 or maximum_islands_per_group < 2:
        raise ValueError("archipelago group limits are invalid")
    if minimum_shore_distance < 2 or maximum_shore_distance <= minimum_shore_distance:
        raise ValueError("archipelago shore-distance limits are invalid")

    height, width = ocean.shape
    tectonic_foundation = derive_tectonic_foundation(land, relief, depth)
    tectonic_plate_id = tectonic_foundation.plate_id
    tectonic_boundary_class = tectonic_foundation.boundary_class
    convergent_boundary = tectonic_boundary_class == BOUNDARY_CONVERGENT
    convergent_distance = _distance_from_mask(
        convergent_boundary,
        np.ones(ocean.shape, dtype=bool),
        maximum_distance=max(maximum_shore_distance + 18, 52),
    ).astype(np.float64)
    convergent_signal = np.exp(
        -convergent_distance / max(5.0, maximum_shore_distance * 0.30)
    )
    plate_velocity_east = np.asarray(
        [plate.velocity_east_cm_per_year for plate in tectonic_foundation.plates],
        dtype=np.float64,
    )[tectonic_plate_id]
    plate_velocity_north = np.asarray(
        [plate.velocity_north_cm_per_year for plate in tectonic_foundation.plates],
        dtype=np.float64,
    )[tectonic_plate_id]
    plate_speed = np.hypot(plate_velocity_east, plate_velocity_north)
    plate_motion_direction = np.arctan2(
        -plate_velocity_north,
        plate_velocity_east,
    )
    extension_distance = max(maximum_shore_distance + 8, int(round(width * 0.22)))
    shore_distance = _distance_from_mask(
        land, ocean, maximum_distance=extension_distance
    )
    shelf = ocean & (depth <= shallow_limit)
    ordinary_candidate = (
        shelf
        & (shore_distance > minimum_shore_distance)
        & (shore_distance <= maximum_shore_distance)
    )
    edge_clearance = max(8, minimum_shore_distance)
    edge_band_width = max(28, int(round(width * 0.10)))
    edge_columns = np.zeros(ocean.shape, dtype=bool)
    edge_columns[:, :edge_band_width] = True
    edge_columns[:, -edge_band_width:] = True
    temperate_rows = np.zeros(ocean.shape, dtype=bool)
    latitude_margin = max(edge_clearance, int(round(height * 0.16)))
    temperate_rows[latitude_margin : height - latitude_margin, :] = True
    ordinary_candidate &= temperate_rows
    extension_candidate = (
        ocean
        & edge_columns
        & temperate_rows
        & (shore_distance > minimum_shore_distance)
        & (shore_distance <= extension_distance)
    )
    candidate = ordinary_candidate | extension_candidate
    density = _box_mean(candidate, radius=7)
    candidate &= density >= 0.14

    large_island_mask = np.zeros(ocean.shape, dtype=bool)
    shelf_island_mask = np.zeros(ocean.shape, dtype=bool)
    island_arc_mask = np.zeros(ocean.shape, dtype=bool)
    hotspot_island_mask = np.zeros(ocean.shape, dtype=bool)
    seamount_mask = np.zeros(ocean.shape, dtype=bool)
    generated_shelf_mask = np.zeros(ocean.shape, dtype=bool)
    shelf_bathymetry = np.ones(ocean.shape, dtype=np.float64)
    trench_mask = np.zeros(ocean.shape, dtype=bool)
    relative_elevation = np.zeros(ocean.shape, dtype=np.float64)
    occupied_centres: list[tuple[int, int]] = []
    large_records: list[_FragmentAnchor] = []
    large_components: list[np.ndarray] = []
    row_grid, column_grid = np.indices(ocean.shape, dtype=np.float64)

    # Resolve drowned continental bays before placing offshore fragments.  The
    # bay floor is already tectonic evidence for foundered crust and therefore
    # has priority over a generic passive-margin microcontinent.
    bay_islands, bay_shelf, bay_centres = _generate_drowned_margin_archipelagos(
        ocean,
        land,
        depth,
        ordinary_candidate,
        shore_distance,
        land,
        relative_elevation,
        maximum_groups=max(1, maximum_groups // 3),
        maximum_islands_per_group=maximum_islands_per_group,
    )
    shelf_island_mask |= bay_islands
    generated_shelf_mask |= bay_shelf

    # First rift continental fragments from measured outward-facing margins.
    # Their centres, long axes, and submarine platforms all retain the same
    # parent-coast evidence; empty ocean by itself never creates a microcontinent.
    large_radius = float(np.clip(min(ocean.shape) * 0.064, 14.0, 48.0))
    # Keep the *whole* natural microcontinent inside the temperate placement
    # domain by moving its centre equatorward.  Cropping the finished shape at
    # the domain edge created ruler-flat coasts; allowing it to cross the polar
    # habitat limit created settlements and routes where none may exist.
    large_centre_margin = min(
        height // 2 - 1,
        latitude_margin + int(math.ceil(large_radius * 1.08)),
    )
    large_centre_rows = np.zeros(ocean.shape, dtype=bool)
    large_centre_rows[
        large_centre_margin : height - large_centre_margin,
        :,
    ] = True
    large_centre_candidate = (
        ocean
        & large_centre_rows
        & (shore_distance > max(minimum_shore_distance + 2, int(large_radius * 0.72)))
        & ~_dilate(bay_islands, max(5, int(round(large_radius * 0.45))))
    )
    large_shape_permitted = ocean & ~_dilate(
        bay_islands,
        max(5, int(round(large_radius * 0.45))),
    )
    fragment_anchors = _continental_fragment_anchors(
        ocean,
        land,
        depth,
        relief,
        temperate_rows,
        large_centre_candidate,
        base_radius=large_radius,
        maximum_count=min(6, maximum_groups),
    )
    for ordinal, anchor in enumerate(fragment_anchors):
        centre = anchor.centre
        effective_radius = anchor.radius
        island = _large_island_field(
            large_shape_permitted,
            centre,
            radius=effective_radius,
            ordinal=ordinal,
            orientation_degrees=anchor.orientation_degrees,
        )
        island = _refine_island_coast(
            island,
            large_shape_permitted,
            (float(centre[0]), float(centre[1])),
            radius=effective_radius,
            orientation_radians=math.radians(anchor.orientation_degrees),
            phase=_stable_phase(ocean.shape, ordinal, 521),
            roughness=2.18,
            embayment=1.42,
        )
        if int(np.count_nonzero(island)) < int(effective_radius * effective_radius * 0.58):
            continue
        large_island_mask |= island
        occupied_centres.append(centre)
        large_records.append(anchor)
        large_components.append(island)
        generated_shelf_mask |= _rifted_platform(
            ocean.shape,
            anchor,
            ocean,
            ordinal=ordinal,
        )
        _write_large_island_relief(
            island,
            relative_elevation,
            centre,
            radius=effective_radius,
            ordinal=ordinal,
            orientation_degrees=anchor.orientation_degrees,
        )

    # Rift satellites are subordinate to both the microcontinent and any
    # pre-existing drowned-margin archipelago.
    occupied = land | large_island_mask | bay_islands

    # Satellite islands follow the long axes of each large island and occupy
    # the same continental shelf.  They intentionally vary in size and leave
    # navigable straits between their parent and one another.
    for group_ordinal, (anchor, parent_island) in enumerate(
        zip(large_records, large_components)
    ):
        centre = anchor.centre
        parent_radius = anchor.radius
        desired = min(
            maximum_islands_per_group,
            (7 + group_ordinal % 4) if anchor.orogenic else (7 + group_ordinal % 2),
        )
        added = 0
        group_satellites = np.zeros(ocean.shape, dtype=bool)
        axis_heading = math.radians(anchor.orientation_degrees)
        axis = np.asarray(
            (math.sin(axis_heading), math.cos(axis_heading)), dtype=np.float64
        )
        source_vector = np.asarray(
            (
                float(centre[0] - anchor.source[0]),
                _wrapped_column_delta(
                    float(centre[1]), float(anchor.source[1]), width
                ),
            ),
            dtype=np.float64,
        )
        if float(np.dot(axis, source_vector)) < 0.0:
            axis *= -1.0
        normal = np.asarray((-axis[1], axis[0]), dtype=np.float64)
        for island_ordinal in range(desired):
            phase = _stable_phase(ocean.shape, island_ordinal, 211 + group_ordinal)
            if anchor.orogenic:
                spacing = 0.34 + 0.13 * (0.5 + 0.5 * math.sin(phase * 1.17))
                distance_along = parent_radius * (
                    1.52 + spacing * island_ordinal
                )
                lateral = parent_radius * (
                    0.16 * math.sin(island_ordinal * 1.31 + phase)
                    + 0.055 * island_ordinal * math.sin(phase * 0.73)
                )
                offset = axis * distance_along + normal * lateral
            else:
                # Passive-margin fragments occupy a compact, asymmetric fault
                # mosaic around and landward of the microcontinent.  They must
                # not become the same one-directional tail as an island arc.
                # A passive fragment field fans across the foundered corridor
                # toward its parent margin.  It is neither a radial halo nor a
                # one-way volcanic tail.  The phase terms give every rift a
                # different fault mosaic without mirroring another margin.
                fan_position = (
                    island_ordinal / max(1, desired - 1) * 2.0 - 1.0
                )
                fan_width = 1.02 + 0.24 * (0.5 + 0.5 * math.sin(phase * 0.73))
                azimuth = (
                    math.pi
                    + fan_position * fan_width
                    + 0.18 * math.sin(phase * 1.19 + island_ordinal * 1.73)
                )
                radial_distance = parent_radius * (
                    1.58
                    + 0.32 * (island_ordinal % 2)
                    + 0.16 * (0.5 + 0.5 * math.sin(phase * 1.41))
                )
                offset = (
                    axis * math.cos(azimuth) + normal * math.sin(azimuth)
                ) * radial_distance
            target_row = centre[0] + float(offset[0])
            target_column = (
                centre[1] + float(offset[1])
            ) % width
            row = int(round(target_row))
            column = int(round(target_column)) % width
            if (
                not 0 <= row < height
                or not ocean[row, column]
                or not temperate_rows[row, column]
            ):
                continue
            # Continental shelves have a heavy-tailed island hierarchy: one
            # or two secondary landmasses, then many much smaller fragments.
            hierarchy = (0.34, 0.25, 0.19, 0.15, 0.13, 0.115, 0.10)
            satellite_radius = max(
                1.7,
                parent_radius
                * hierarchy[min(island_ordinal, len(hierarchy) - 1)]
                * (0.90 + 0.18 * (0.5 + 0.5 * math.sin(phase * 1.73))),
            )
            island_orientation = math.atan2(float(axis[0]), float(axis[1]))
            island_orientation += (
                0.22 * math.sin(phase)
                if anchor.orogenic
                else 0.58 * math.sin(phase * 1.37 + island_ordinal)
            )
            permitted = ocean & ~_dilate(occupied, 2)
            island = _organic_ellipse(
                permitted,
                (float(row), float(column)),
                radius=satellite_radius,
                orientation=island_orientation,
                phase=phase,
                wrap_columns=True,
            )
            island = _refine_island_coast(
                island,
                permitted,
                (float(row), float(column)),
                radius=satellite_radius,
                orientation_radians=island_orientation,
                phase=phase,
                roughness=0.82,
                embayment=0.46,
            )
            if int(np.count_nonzero(island)) < 7:
                continue
            shelf_island_mask |= island
            group_satellites |= island
            occupied |= island
            _write_structural_relief(
                island,
                relative_elevation,
                (float(row), float(column)),
                radius=satellite_radius,
                orientation_radians=island_orientation,
                phase=phase,
                base=0.018,
                amplitude=0.21,
            )
            added += 1
        if added:
            plate_seed = parent_island | group_satellites
            shelf_width = max(
                5,
                int(
                    round(
                        parent_radius * (0.30 if anchor.orogenic else 0.38)
                    )
                ),
            )
            plate_distance = _distance_from_mask(
                plate_seed, ocean, maximum_distance=shelf_width
            )
            generated_shelf_mask |= ocean & (plate_distance <= shelf_width)

    # Add a few smaller subduction-style arcs on authored shallow shelves.
    gradient_rows = np.gradient(shore_distance.astype(np.float64), axis=0)
    gradient_columns = (
        np.roll(shore_distance.astype(np.float64), -1, axis=1)
        - np.roll(shore_distance.astype(np.float64), 1, axis=1)
    ) * 0.5
    boundary_gradient_rows = np.gradient(convergent_distance, axis=0)
    boundary_gradient_columns = (
        np.roll(convergent_distance, -1, axis=1)
        - np.roll(convergent_distance, 1, axis=1)
    ) * 0.5
    arc_group_limit = min(maximum_groups, 3)
    arc_centres: list[tuple[int, int]] = []
    continental_coast = land & (
        np.roll(ocean, 1, axis=1)
        | np.roll(ocean, -1, axis=1)
        | np.pad(ocean[:-1], ((1, 0), (0, 0)))
        | np.pad(ocean[1:], ((0, 1), (0, 0)))
    )
    terrain_values = relief[land & temperate_rows]
    if terrain_values.size and float(np.ptp(terrain_values)) >= 0.08:
        arc_highland = land & temperate_rows & (
            relief >= float(np.quantile(terrain_values, 0.68))
        )
        arc_highland_distance = _distance_from_mask(
            arc_highland,
            land | ordinary_candidate,
            maximum_distance=max(18, maximum_shore_distance + 10),
        )
        orogenic_signal = np.exp(
            -arc_highland_distance.astype(np.float64)
            / max(5.0, maximum_shore_distance * 0.42)
        )
    else:
        orogenic_signal = np.zeros(ocean.shape, dtype=np.float64)
    arc_score = (
        density
        * (0.35 + np.clip(1.0 - depth, 0.0, 1.0))
        * (0.62 + 0.38 * orogenic_signal)
        * convergent_signal
    )
    arc_domain = (
        ordinary_candidate
        & (convergent_distance <= maximum_shore_distance + 12)
    )
    for arc_ordinal in range(arc_group_limit):
        centre = _best_target(
            arc_domain & ~_dilate(occupied, 5),
            arc_score * (1.0 + 0.04 * np.sin(row_grid * 0.27 + column_grid * 0.19 + arc_ordinal)),
            occupied_centres + arc_centres,
            minimum_spacing=max(24.0, min(ocean.shape) * 0.13),
        )
        if centre is None:
            break
        arc_centres.append(centre)
        gradient = np.asarray(
            (
                boundary_gradient_rows[centre],
                boundary_gradient_columns[centre],
            ),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(gradient))
        if norm <= 1.0e-9:
            gradient = np.asarray(
                (gradient_rows[centre], gradient_columns[centre]),
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(gradient))
        if norm <= 1.0e-9:
            continue
        else:
            gradient /= norm
            tangent = np.asarray((-gradient[1], gradient[0]), dtype=np.float64)
        coast_points = np.argwhere(continental_coast)
        if coast_points.size and convergent_distance[centre] > maximum_shore_distance:
            coast_row_delta = coast_points[:, 0].astype(np.float64) - centre[0]
            coast_column_delta = coast_points[:, 1].astype(np.float64) - centre[1]
            coast_column_delta = (
                coast_column_delta + width / 2.0
            ) % width - width / 2.0
            source_index = int(
                np.argmin(np.hypot(coast_row_delta, coast_column_delta))
            )
            source = (
                int(coast_points[source_index, 0]),
                int(coast_points[source_index, 1]),
            )
            fallback = math.degrees(
                math.atan2(float(tangent[0]), float(tangent[1]))
            )
            mountain_axis = _orogenic_axis_orientation(
                relief,
                land,
                source,
                window_radius=max(18.0, maximum_shore_distance * 1.2),
                fallback_degrees=fallback,
            )
            axis_radians = math.radians(mountain_axis)
            tangent = np.asarray(
                (math.sin(axis_radians), math.cos(axis_radians)),
                dtype=np.float64,
            )
        normal = np.asarray((tangent[1], -tangent[0]), dtype=np.float64)
        desired = min(maximum_islands_per_group, 7 + arc_ordinal)
        added = 0
        for island_ordinal in range(desired):
            normalized_position = (
                island_ordinal / max(1, desired - 1) * 2.0 - 1.0
            )
            position = normalized_position * 0.5 * (desired - 1)
            phase = _stable_phase(ocean.shape, island_ordinal, 331 + arc_ordinal)
            target = (
                np.asarray(centre, dtype=np.float64)
                + tangent * position * 12.0
                + normal
                * (
                    9.0 * (1.0 - normalized_position * normalized_position)
                    + math.sin(position * 0.9 + phase) * 2.8
                )
            )
            row = int(round(float(target[0])))
            column = int(round(float(target[1]))) % width
            if not 0 <= row < height or not temperate_rows[row, column]:
                continue
            # The shelf candidate controls where the arc is anchored.  Its
            # contour is not a shoreline and therefore cannot clip island
            # lobes into straight threshold-shaped edges.
            permitted = ocean & ~_dilate(occupied, 2)
            arc_radius = (2.6, 4.2, 3.1, 5.8, 3.4, 4.8, 2.8, 3.8, 2.5)[
                island_ordinal % 9
            ] * (0.92 + 0.16 * (0.5 + 0.5 * math.sin(phase * 1.67)))
            arc_orientation = math.atan2(float(tangent[0]), float(tangent[1]))
            island = _organic_ellipse(
                permitted,
                (float(row), float(column)),
                radius=arc_radius,
                orientation=arc_orientation,
                phase=phase,
                wrap_columns=True,
            )
            if island_ordinal in (1, 3, 5) and bool(island.any()):
                lobe_centre = (
                    float(target[0] + tangent[0] * 3.5 + normal[0] * 1.8),
                    float(target[1] + tangent[1] * 3.5 + normal[1] * 1.8) % width,
                )
                island |= _organic_ellipse(
                    permitted,
                    lobe_centre,
                    radius=(2.1, 3.4, 2.7)[island_ordinal // 2],
                    orientation=math.atan2(float(tangent[0]), float(tangent[1])) + 0.42,
                    phase=phase + 1.37,
                    wrap_columns=True,
                )
            island = _refine_island_coast(
                island,
                permitted,
                (float(row), float(column)),
                radius=arc_radius,
                orientation_radians=arc_orientation,
                phase=phase,
                roughness=0.94,
                embayment=0.34,
            )
            if int(np.count_nonzero(island)) < 7:
                continue
            island_arc_mask |= island
            occupied |= island
            _write_structural_relief(
                island,
                relative_elevation,
                (float(row), float(column)),
                radius=arc_radius,
                orientation_radians=arc_orientation,
                phase=phase,
                base=0.018,
                amplitude=0.26,
            )
            added += 1
        if added >= 2:
            shelf_distance = _distance_from_mask(
                island_arc_mask, ocean, maximum_distance=7
            )
            generated_shelf_mask |= ocean & (shelf_distance <= 7)

            # The trench runs along the seaward (increasing shore-distance)
            # side of the arc.  Overlapping organic segments form one uneven
            # trough instead of a ruler-straight decorative stroke.
            trench_offset = 13.0 + 2.0 * arc_ordinal
            for trench_ordinal in range(desired * 2 - 1):
                normalized_position = (
                    trench_ordinal / max(1, desired * 2 - 2) * 2.0 - 1.0
                )
                position = normalized_position * 0.5 * (desired - 1)
                trench_phase = _stable_phase(
                    ocean.shape, trench_ordinal, 367 + arc_ordinal
                )
                target = (
                    np.asarray(centre, dtype=np.float64)
                    + tangent * position * 12.0
                    + normal
                    * (
                        trench_offset
                        + 9.0 * (1.0 - normalized_position * normalized_position)
                        + math.sin(position * 0.9 + trench_phase) * 2.4
                    )
                )
                trench_segment = _organic_ellipse(
                    ocean & ~_dilate(occupied, 2),
                    (float(target[0]), float(target[1]) % width),
                    radius=3.0,
                    orientation=math.atan2(float(tangent[0]), float(tangent[1])),
                    phase=trench_phase,
                    wrap_columns=True,
                )
                trench_mask |= trench_segment

    # Finally place sparse hotspot trails in deep ocean.  Their islands are
    # much smaller, linear, and carry only a narrow volcanic apron rather than
    # a continental shelf.
    deep_candidate = (
        ocean
        & temperate_rows
        & (depth >= max(0.78, shallow_limit + 0.08))
        & (shore_distance > max(minimum_shore_distance + 10, int(height * 0.08)))
        & ~_dilate(occupied, 8)
        & ~trench_mask
    )
    # A visible hotspot trail must fit wholly inside one mapped ocean basin.
    # The antimeridian is not a placement target: otherwise one trail is split
    # into two meaningless edge fragments that read as mirrored decoration.
    seam_clearance = max(26, int(round(width * 0.095)))
    seam_safe = np.zeros(ocean.shape, dtype=bool)
    seam_safe[:, seam_clearance : width - seam_clearance] = True
    deep_candidate &= seam_safe
    hotspot_centres: list[tuple[int, int]] = []
    hotspot_group_limit = 1
    for chain_ordinal in range(hotspot_group_limit):
        desired = max(8, min(10, maximum_islands_per_group + 2))
        maximum_track_length = (desired - 1) * (
            6.35 + 0.43 * (desired - 1)
        )
        # A hotspot trail records motion of the plate over a nearly fixed
        # mantle source.  Its local track therefore follows the velocity of
        # the inferred plate instead of choosing a decorative random angle.
        best_track_candidate = deep_candidate & (plate_speed >= 0.08)
        for fraction in (0.50, 1.0):
            projected_rows = row_grid + (
                np.sin(plate_motion_direction)
                * maximum_track_length
                * fraction
            )
            projected_columns = column_grid + (
                np.cos(plate_motion_direction)
                * maximum_track_length
                * fraction
            )
            inside = (
                (projected_rows >= latitude_margin)
                & (projected_rows < height - latitude_margin)
                & (projected_columns >= seam_clearance)
                & (projected_columns < width - seam_clearance)
            )
            sample_rows = np.clip(
                np.rint(projected_rows).astype(np.int32), 0, height - 1
            )
            sample_columns = np.clip(
                np.rint(projected_columns).astype(np.int32), 0, width - 1
            )
            best_track_candidate &= (
                inside
                & deep_candidate[sample_rows, sample_columns]
                & (tectonic_plate_id[sample_rows, sample_columns] == tectonic_plate_id)
            )
        if not np.any(best_track_candidate):
            continue
        tectonic_texture = lattice_noise(
            column_grid,
            row_grid,
            spacing=max(19.0, min(ocean.shape) * 0.15),
            seed=2609 + chain_ordinal * 173,
        )
        score = (
            (0.52 + 0.48 * np.clip(depth, 0.0, 1.0))
            * (0.78 + 0.22 * tectonic_texture)
            * (0.72 + 0.28 * np.clip(shore_distance / max(1.0, height * 0.22), 0.0, 1.0))
        )
        centre = _best_target(
            best_track_candidate,
            score,
            hotspot_centres,
            minimum_spacing=max(32.0, width * 0.18),
        )
        if centre is None:
            continue
        hotspot_centres.append(centre)
        direction = float(plate_motion_direction[centre])
        chain_emergent: list[np.ndarray] = []
        chain_has_seamount = False
        for island_ordinal in range(desired):
            age = island_ordinal / max(1, desired - 1)
            distance_along = island_ordinal * (6.7 + 0.48 * island_ordinal)
            bend = math.sin(age * math.pi * 1.15) * (2.0 + chain_ordinal)
            row = int(
                round(
                    centre[0]
                    + math.sin(direction) * distance_along
                    + math.cos(direction) * bend
                )
            )
            column = int(
                round(
                    centre[1]
                    + math.cos(direction) * distance_along
                    - math.sin(direction) * bend
                )
            )
            if (
                not 0 <= row < height
                or not seam_clearance <= column < width - seam_clearance
            ):
                continue
            local_phase = _stable_phase(ocean.shape, island_ordinal, 443 + chain_ordinal)
            # Deep-water suitability chooses the hotspot track.  Emergent
            # volcanic coast geometry is allowed to occupy adjacent ocean.
            permitted = (
                ocean
                & seam_safe
                & (depth >= max(0.78, shallow_limit + 0.08))
                & ~_dilate(occupied, 1)
                & ~trench_mask
            )
            hotspot_radius = (4.4 - 3.0 * age) * (
                0.94
                + 0.12 * (0.5 + 0.5 * math.sin(local_phase * 1.91))
            )
            island = _organic_ellipse(
                permitted,
                (float(row), float(column)),
                radius=hotspot_radius,
                orientation=direction,
                phase=local_phase,
                wrap_columns=True,
            )
            island = _refine_island_coast(
                island,
                permitted,
                (float(row), float(column)),
                radius=hotspot_radius,
                orientation_radians=direction,
                phase=local_phase,
                roughness=0.42,
                embayment=0.12,
            )
            if int(np.count_nonzero(island)) < 4:
                continue
            if age > 0.78:
                seamount_mask |= island
                occupied |= island
                chain_has_seamount = True
                continue
            hotspot_island_mask |= island
            chain_emergent.append(island)
            occupied |= island
            _write_relief(
                island,
                relative_elevation,
                base=0.022,
                amplitude=0.40 - 0.22 * age,
            )
        if not chain_has_seamount and len(chain_emergent) >= 2:
            oldest = chain_emergent[-1]
            hotspot_island_mask &= ~oldest
            seamount_mask |= oldest
            relative_elevation[oldest] = 0.0

    if not bool(seamount_mask.any()) and len(_components(hotspot_island_mask)) > 1:
        smallest_component = min(_components(hotspot_island_mask), key=len)
        rows, columns = zip(*smallest_component)
        fallback_seamount = np.zeros(ocean.shape, dtype=bool)
        fallback_seamount[np.asarray(rows), np.asarray(columns)] = True
        hotspot_island_mask &= ~fallback_seamount
        seamount_mask |= fallback_seamount
        relative_elevation[fallback_seamount] = 0.0

    if bool(hotspot_island_mask.any() | seamount_mask.any()):
        volcanic_apron = _distance_from_mask(
            hotspot_island_mask | seamount_mask, ocean, maximum_distance=3
        )
        generated_shelf_mask |= ocean & (volcanic_apron <= 3)

    island_mask = (
        large_island_mask
        | shelf_island_mask
        | island_arc_mask
        | hotspot_island_mask
    )
    # Placement latitude is a candidate-domain constraint, not a geological
    # boundary.  A long exposed run exactly on that row proves that a caller
    # has accidentally used the domain mask to crop island geometry.
    north_clip = island_mask[latitude_margin] & ~island_mask[latitude_margin - 1]
    south_row = height - latitude_margin - 1
    south_clip = island_mask[south_row] & ~island_mask[south_row + 1]
    latitude_clip_run = max(
        _longest_true_run(north_clip),
        _longest_true_run(south_clip),
    )
    if latitude_clip_run > 12:
        raise ValueError(
            "island coastline was clipped by the latitude placement domain"
        )
    authored_shelf, authored_shelf_depth = _authored_island_shelves(
        land,
        ocean,
        temperate_rows,
    )
    generated_shelf_mask |= authored_shelf
    trench_mask &= ocean & ~island_mask
    generated_shelf_mask &= ocean & ~island_mask & ~trench_mask
    shelf_distance = _distance_from_mask(
        land | island_mask,
        ocean,
        maximum_distance=max(24, int(round(min(ocean.shape) * 0.11))),
    )
    shelf_width = max(14.0, min(ocean.shape) * 0.075)
    graded_shelf_depth = 0.075 + 0.51 * np.power(
        np.clip(shelf_distance.astype(np.float64) / shelf_width, 0.0, 1.0),
        0.76,
    )
    target_shelf_depth = np.minimum(depth, np.minimum(0.58, graded_shelf_depth))
    target_shelf_depth[authored_shelf] = np.minimum(
        target_shelf_depth[authored_shelf],
        authored_shelf_depth[authored_shelf],
    )
    # Match the inherited ocean floor at the *outer* shelf edge.  Measuring
    # from exterior water (rather than every mask boundary) keeps the coast
    # shallow while feathering the distal platform into existing contours.
    water_after_islands = ocean & ~island_mask
    outer_ocean = water_after_islands & ~generated_shelf_mask
    shelf_blend_width = max(4, int(round(shelf_width * 0.42)))
    distance_from_outer_ocean = _distance_from_mask(
        outer_ocean,
        water_after_islands,
        maximum_distance=shelf_blend_width,
    ).astype(np.float64)
    shelf_blend = np.clip(
        distance_from_outer_ocean / float(shelf_blend_width),
        0.0,
        1.0,
    )
    shelf_blend = shelf_blend * shelf_blend * (3.0 - 2.0 * shelf_blend)
    blended_shelf_depth = depth + shelf_blend * (target_shelf_depth - depth)
    shelf_bathymetry[generated_shelf_mask] = blended_shelf_depth[
        generated_shelf_mask
    ]
    shelf_bathymetry[~generated_shelf_mask] = 1.0
    return ShelfArchipelagoField(
        island_mask=island_mask,
        large_island_mask=large_island_mask,
        shelf_island_mask=shelf_island_mask,
        island_arc_mask=island_arc_mask,
        hotspot_island_mask=hotspot_island_mask,
        seamount_mask=seamount_mask,
        shelf_mask=generated_shelf_mask,
        shelf_bathymetry=shelf_bathymetry,
        trench_mask=trench_mask,
        relative_elevation=relative_elevation,
        island_count=len(_components(island_mask)),
        group_count=(
            len(large_records)
            + len(bay_centres)
            + len(arc_centres)
            + len(hotspot_centres)
        ),
    )


__all__ = [
    "LakeIslandField",
    "ShelfArchipelagoField",
    "classify_inland_seas",
    "derive_inland_sea_bathymetry",
    "generate_inland_water_islands",
    "generate_shelf_archipelagos",
]
