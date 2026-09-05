"""Infer one tectonic foundation from existing physical geography.

The authored continents, mountain belts, continental shelves, trenches and
ocean-floor highs are evidence.  This module explains that evidence with one
deterministic plate partition.  It is deliberately independent of rendering
and island geometry so both stages can consume the same causal model.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
import math
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np


BOUNDARY_INTERIOR = 0
BOUNDARY_CONVERGENT = 1
BOUNDARY_DIVERGENT = 2
BOUNDARY_TRANSFORM = 3

_BOUNDARY_CODES = {
    "convergent": BOUNDARY_CONVERGENT,
    "divergent": BOUNDARY_DIVERGENT,
    "transform": BOUNDARY_TRANSFORM,
}

_BOUNDARY_PRIORITY = {
    BOUNDARY_INTERIOR: 0,
    BOUNDARY_TRANSFORM: 1,
    BOUNDARY_DIVERGENT: 2,
    BOUNDARY_CONVERGENT: 3,
}


@dataclass(frozen=True, slots=True)
class FoundationPlate:
    identifier: int
    kind: str
    row: int
    column: int
    area_cells: int
    velocity_east_cm_per_year: float
    velocity_north_cm_per_year: float


@dataclass(frozen=True, slots=True)
class FoundationBoundary:
    plate_ids: tuple[int, int]
    classification: str
    paths: tuple[tuple[tuple[float, float], ...], ...]
    evidence: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True, slots=True)
class TectonicFoundation:
    plate_id: np.ndarray
    boundary_class: np.ndarray
    plates: tuple[FoundationPlate, ...]
    boundaries: tuple[FoundationBoundary, ...]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        labels = np.array(self.plate_id, dtype=np.int16, copy=True)
        boundary_class = np.array(self.boundary_class, dtype=np.int8, copy=True)
        if labels.ndim != 2 or boundary_class.shape != labels.shape:
            raise ValueError("tectonic foundation arrays must share a 2-D shape")
        if np.any(labels < 0):
            raise ValueError("every tectonic cell must belong to a plate")
        if np.any((boundary_class < 0) | (boundary_class > BOUNDARY_TRANSFORM)):
            raise ValueError("tectonic boundary classes are invalid")
        labels.setflags(write=False)
        boundary_class.setflags(write=False)
        object.__setattr__(self, "plate_id", labels)
        object.__setattr__(self, "boundary_class", boundary_class)
        object.__setattr__(self, "plates", tuple(self.plates))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

def _normalise(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    sample = values if mask is None else values[mask]
    if sample.size == 0:
        return np.zeros(values.shape, dtype=np.float64)
    low, high = np.quantile(sample.astype(np.float64), (0.08, 0.94))
    if high <= low + 1.0e-12:
        return np.zeros(values.shape, dtype=np.float64)
    return np.clip((values.astype(np.float64) - low) / (high - low), 0.0, 1.0)


def _neighbours(
    row: int,
    column: int,
    height: int,
    width: int,
) -> Iterable[tuple[int, int]]:
    yield row, (column - 1) % width
    yield row, (column + 1) % width
    if row:
        yield row - 1, column
    if row + 1 < height:
        yield row + 1, column


def _distance_from(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    distance = np.full(mask.shape, height + width + 1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for raw_row, raw_column in np.argwhere(mask):
        row, column = int(raw_row), int(raw_column)
        distance[row, column] = 0
        queue.append((row, column))
    while queue:
        row, column = queue.popleft()
        candidate = int(distance[row, column]) + 1
        for next_row, next_column in _neighbours(row, column, height, width):
            if candidate < distance[next_row, next_column]:
                distance[next_row, next_column] = candidate
                queue.append((next_row, next_column))
    return distance


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    height, _width = values.shape
    rows = np.arange(height)
    total = np.zeros(values.shape, dtype=np.float64)
    count = 0
    for row_offset in range(-radius, radius + 1):
        row_sample = values[np.clip(rows + row_offset, 0, height - 1)]
        for column_offset in range(-radius, radius + 1):
            total += np.roll(row_sample, column_offset, axis=1)
            count += 1
    return total / count


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    for raw_row, raw_column in np.argwhere(mask):
        start = (int(raw_row), int(raw_column))
        if visited[start]:
            continue
        visited[start] = True
        queue = deque((start,))
        component: list[tuple[int, int]] = []
        while queue:
            row, column = queue.popleft()
            component.append((row, column))
            for next_row, next_column in _neighbours(row, column, height, width):
                if mask[next_row, next_column] and not visited[next_row, next_column]:
                    visited[next_row, next_column] = True
                    queue.append((next_row, next_column))
        result.append(component)
    return sorted(result, key=lambda item: (-len(item), min(item)))


def _wrapped_distance_sq(
    row: np.ndarray,
    column: np.ndarray,
    seed_row: int,
    seed_column: int,
    width: int,
) -> np.ndarray:
    delta_column = np.abs(column - seed_column)
    delta_column = np.minimum(delta_column, width - delta_column)
    return np.square(row - seed_row) + np.square(delta_column)


def _select_seed(
    candidates: np.ndarray,
    score: np.ndarray,
    existing: list[tuple[int, int, str]],
) -> tuple[int, int]:
    if not np.any(candidates):
        candidates = np.ones(candidates.shape, dtype=bool)
    combined = np.asarray(score, dtype=np.float64).copy()
    if existing:
        rows, columns = np.indices(candidates.shape)
        nearest = np.full(candidates.shape, np.inf, dtype=np.float64)
        for seed_row, seed_column, _kind in existing:
            nearest = np.minimum(
                nearest,
                _wrapped_distance_sq(
                    rows,
                    columns,
                    seed_row,
                    seed_column,
                    candidates.shape[1],
                ),
            )
        combined += np.sqrt(nearest) * 1.08
    combined[~candidates] = -np.inf
    return divmod(int(np.argmax(combined)), candidates.shape[1])


def _plate_seeds(
    land: np.ndarray,
    elevation: np.ndarray,
    bathymetry: np.ndarray,
    barrier: np.ndarray,
    *,
    target_count: int | None = None,
) -> list[tuple[int, int, str]]:
    height, width = land.shape
    rows = np.arange(height)[:, None]
    latitude = 90.0 - (rows + 0.5) * 180.0 / height
    polar = np.abs(latitude) >= 66.56
    distance_ocean = _distance_from(~land)
    distance_land = _distance_from(land)
    seeds: list[tuple[int, int, str]] = []

    for north in (True, False):
        domain = polar & land & ((latitude > 0) if north else (latitude < 0))
        if np.any(domain):
            score = distance_ocean * 1.8 + elevation * 1.4 - barrier * 16.0
            row, column = _select_seed(domain, score, seeds)
        else:
            row = 0 if north else height - 1
            column = width // (3 if north else 2)
        seeds.append((row, column, "polar"))

    temperate_land = land & ~polar
    minimum_component = max(4, land.size // 1800)
    land_components = [
        item for item in _components(temperate_land) if len(item) >= minimum_component
    ]
    # Enough continental interiors are required for opposing wavefronts to
    # meet on the major mountain belts instead of across arbitrary lowlands.
    if target_count is None:
        target_count = min(13, max(9, land.size // 4500 + 9))
    land_budget = min(
        max(0, target_count - len(seeds)),
        min(7, max(3, len(land_components) + 4)),
    )
    allocations = [1 for _ in land_components[:land_budget]]
    while sum(allocations) < land_budget and allocations:
        index = max(
            range(len(allocations)),
            key=lambda value: len(land_components[value]) / (allocations[value] + 0.55),
        )
        allocations[index] += 1
    for component, count in zip(land_components, allocations, strict=False):
        domain = np.zeros(land.shape, dtype=bool)
        component_rows, component_columns = zip(*component, strict=True)
        domain[np.asarray(component_rows), np.asarray(component_columns)] = True
        for _ in range(count):
            score = distance_ocean * 1.35 + (1.0 - elevation) * 1.8 - barrier * 20.0
            row, column = _select_seed(domain, score, seeds)
            seeds.append((row, column, "continental"))

    ocean_domain = ~land & ~polar
    while len(seeds) < target_count:
        deep = np.clip(bathymetry / 7.0, 0.0, 1.0)
        score = distance_land * 1.45 + deep * 7.0 - barrier * 15.0
        row, column = _select_seed(ocean_domain, score, seeds)
        seeds.append((row, column, "oceanic"))
        rr, cc = np.indices(land.shape)
        ocean_domain &= _wrapped_distance_sq(rr, cc, row, column, width) > max(
            9,
            (min(height, width) / 10.0) ** 2,
        )
        if not np.any(ocean_domain):
            ocean_domain = ~land & ~polar
    return seeds


def _grow_plates(
    land: np.ndarray,
    bathymetry: np.ndarray,
    barrier: np.ndarray,
    seeds: list[tuple[int, int, str]],
) -> np.ndarray:
    height, width = land.shape
    cost = np.full(land.shape, np.inf, dtype=np.float64)
    owner = np.full(land.shape, -1, dtype=np.int16)
    heap: list[tuple[float, int, int, int]] = []
    for identifier, (row, column, _kind) in enumerate(seeds):
        cost[row, column] = 0.0
        owner[row, column] = identifier
        heapq.heappush(heap, (0.0, identifier, row, column))
    while heap:
        current, identifier, row, column = heapq.heappop(heap)
        if identifier != int(owner[row, column]) or current > cost[row, column] + 1.0e-10:
            continue
        kind = seeds[identifier][2]
        latitude = 90.0 - (row + 0.5) * 180.0 / height
        for next_row, next_column in _neighbours(row, column, height, width):
            horizontal = next_row == row
            metric = max(0.22, math.cos(math.radians(latitude))) if horizontal else 1.0
            local_barrier = float((barrier[row, column] + barrier[next_row, next_column]) * 0.5)
            # A super-linear barrier makes ridge axes and trenches preferred
            # meeting lines while broad low-relief interiors remain cheap.
            step = metric * (1.0 + 27.0 * local_barrier**1.45)
            target_land = bool(land[next_row, next_column])
            if kind == "oceanic" and target_land:
                step += 0.42
            elif kind == "continental" and not target_land:
                step += 0.07 + 0.025 * float(bathymetry[next_row, next_column])
            elif kind == "polar" and abs(
                90.0 - (next_row + 0.5) * 180.0 / height
            ) < 48.0:
                step += 0.42
            candidate = current + step
            previous_owner = int(owner[next_row, next_column])
            if candidate < cost[next_row, next_column] - 1.0e-10 or (
                math.isclose(candidate, cost[next_row, next_column], abs_tol=1.0e-10)
                and (previous_owner < 0 or identifier < previous_owner)
            ):
                cost[next_row, next_column] = candidate
                owner[next_row, next_column] = identifier
                heapq.heappush(heap, (candidate, identifier, next_row, next_column))
    return owner


def _simplify(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    if len(points) <= 2:
        return tuple(points)
    simple = [points[0]]
    for index in range(1, len(points) - 1):
        previous, current, following = simple[-1], points[index], points[index + 1]
        first = (current[0] - previous[0], current[1] - previous[1])
        second = (following[0] - current[0], following[1] - current[1])
        if abs(first[0] * second[1] - first[1] * second[0]) > 1.0e-9:
            simple.append(current)
    simple.append(points[-1])
    if len(simple) < 4:
        return tuple(simple)
    smooth = [simple[0]]
    for left, right in zip(simple, simple[1:]):
        smooth.append((0.72 * left[0] + 0.28 * right[0], 0.72 * left[1] + 0.28 * right[1]))
        smooth.append((0.28 * left[0] + 0.72 * right[0], 0.28 * left[1] + 0.72 * right[1]))
    smooth.append(simple[-1])
    return tuple(smooth)


def _trace_edges(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    adjacency: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (first, second) in enumerate(edges):
        adjacency[first].append(index)
        adjacency[second].append(index)
    used: set[int] = set()
    paths: list[tuple[tuple[float, float], ...]] = []
    starts = sorted(node for node, incident in adjacency.items() if len(incident) != 2)
    starts.extend(node for node in sorted(adjacency) if node not in starts)
    for start in starts:
        for first_edge in sorted(adjacency[start]):
            if first_edge in used:
                continue
            points = [start]
            node = start
            edge_index = first_edge
            while edge_index not in used:
                used.add(edge_index)
                first, second = edges[edge_index]
                node = second if first == node else first
                points.append(node)
                candidates = [value for value in adjacency[node] if value not in used]
                if len(adjacency[node]) != 2 or not candidates:
                    break
                edge_index = min(candidates)
            path = _simplify([(float(x), float(y)) for x, y in points])
            if len(path) >= 2:
                paths.append(path)
    return tuple(paths)


def _local_boundary_class(
    *,
    land_fraction: float,
    mountain: float,
    trench: float,
    ridge: float,
) -> str:
    # Classify each connected stretch from its local physical evidence.  One
    # plate pair may converge along one arc and transform elsewhere.
    if mountain >= 0.25 or trench >= 0.46:
        return "convergent"
    if land_fraction <= 0.10 and ridge >= 0.20:
        return "divergent"
    return "transform"


def _boundary_inputs(
    labels: np.ndarray,
    land: np.ndarray,
    mountain: np.ndarray,
    trench: np.ndarray,
    ridge: np.ndarray,
) -> tuple[tuple[FoundationBoundary, ...], np.ndarray]:
    height, width = labels.shape
    grouped_edges: dict[
        tuple[tuple[int, int], str],
        list[tuple[tuple[int, int], tuple[int, int]]],
    ] = defaultdict(list)
    grouped_samples: dict[tuple[tuple[int, int], str], list[tuple[float, ...]]] = defaultdict(list)
    boundary_class = np.zeros(labels.shape, dtype=np.int8)

    def add_edge(
        first: tuple[int, int],
        second: tuple[int, int],
        edge: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        first_id, second_id = int(labels[first]), int(labels[second])
        if first_id == second_id:
            return
        pair = tuple(sorted((first_id, second_id)))
        land_fraction = float(bool(land[first]) or bool(land[second]))
        mountain_value = float(max(mountain[first], mountain[second]))
        trench_value = float(max(trench[first], trench[second]))
        ridge_value = float(max(ridge[first], ridge[second]))
        classification = _local_boundary_class(
            land_fraction=land_fraction,
            mountain=mountain_value,
            trench=trench_value,
            ridge=ridge_value,
        )
        key = (pair, classification)
        grouped_edges[key].append(edge)
        grouped_samples[key].append(
            (
                land_fraction,
                mountain_value,
                trench_value,
                ridge_value,
                float(bool(land[first]) != bool(land[second])),
            )
        )
        code = _BOUNDARY_CODES[classification]
        for row, column in (first, second):
            current = int(boundary_class[row, column])
            if _BOUNDARY_PRIORITY[code] > _BOUNDARY_PRIORITY[current]:
                boundary_class[row, column] = code

    for row in range(height):
        for column in range(width):
            if column + 1 < width:
                add_edge(
                    (row, column),
                    (row, column + 1),
                    ((column + 1, row), (column + 1, row + 1)),
                )
            elif int(labels[row, 0]) != int(labels[row, column]):
                add_edge(
                    (row, column),
                    (row, 0),
                    ((0, row), (0, row + 1)),
                )
                add_edge(
                    (row, column),
                    (row, 0),
                    ((width, row), (width, row + 1)),
                )
            if row + 1 < height:
                add_edge(
                    (row, column),
                    (row + 1, column),
                    ((column, row + 1), (column + 1, row + 1)),
                )

    boundaries: list[FoundationBoundary] = []
    for key, edges in sorted(grouped_edges.items()):
        pair, classification = key
        values = np.asarray(grouped_samples[key], dtype=np.float64)
        boundaries.append(
            FoundationBoundary(
                pair,
                classification,
                _trace_edges(edges),
                {
                    "landFraction": float(np.mean(values[:, 0])),
                    "mountainEvidence": float(np.quantile(values[:, 1], 0.65)),
                    "trenchEvidence": float(np.quantile(values[:, 2], 0.65)),
                    "ridgeEvidence": float(np.quantile(values[:, 3], 0.65)),
                    "coastCoincidenceFraction": float(np.mean(values[:, 4])),
                    "edgeCount": float(len(values)),
                },
            )
        )
    return tuple(boundaries), boundary_class


def _pair_normal(
    first: tuple[int, int, str],
    second: tuple[int, int, str],
    width: int,
) -> tuple[float, float]:
    dx = float(second[1] - first[1])
    if dx > width / 2:
        dx -= width
    elif dx < -width / 2:
        dx += width
    dy = float(second[0] - first[0])
    length = max(1.0e-9, math.hypot(dx, dy))
    return dx / length, dy / length


def _solve_velocities(
    seeds: list[tuple[int, int, str]],
    boundaries: tuple[FoundationBoundary, ...],
    width: int,
) -> np.ndarray:
    count = len(seeds)
    equations: list[np.ndarray] = []
    targets: list[float] = []
    for boundary in boundaries:
        first, second = boundary.plate_ids
        nx, ny = _pair_normal(seeds[first], seeds[second], width)
        weight = float(np.clip(math.sqrt(boundary.evidence["edgeCount"] / 24.0), 0.7, 3.0))
        normal = np.zeros(count * 2, dtype=np.float64)
        normal[first * 2 : first * 2 + 2] = (-nx, -ny)
        normal[second * 2 : second * 2 + 2] = (nx, ny)
        equations.append(normal * weight)
        targets.append(
            weight
            * (
                -2.8
                if boundary.classification == "convergent"
                else 2.5
                if boundary.classification == "divergent"
                else 0.0
            )
        )
        if boundary.classification == "transform":
            tangent = np.zeros(count * 2, dtype=np.float64)
            tx, ty = -ny, nx
            tangent[first * 2 : first * 2 + 2] = (-tx, -ty)
            tangent[second * 2 : second * 2 + 2] = (tx, ty)
            equations.append(tangent * weight)
            targets.append(weight * (1.8 if (first + second) % 2 == 0 else -1.8))
    for index in range(count * 2):
        regularizer = np.zeros(count * 2, dtype=np.float64)
        regularizer[index] = 0.12
        equations.append(regularizer)
        targets.append(0.0)
    for component in range(2):
        mean = np.zeros(count * 2, dtype=np.float64)
        mean[component::2] = 0.7
        equations.append(mean)
        targets.append(0.0)
    solution, *_ = np.linalg.lstsq(np.vstack(equations), np.asarray(targets), rcond=None)
    velocity = solution.reshape((count, 2))
    maximum = float(np.max(np.linalg.norm(velocity, axis=1), initial=0.0))
    if maximum > 4.8:
        velocity *= 4.8 / maximum
    return velocity


def derive_tectonic_foundation(
    land_mask: np.ndarray,
    elevation: np.ndarray,
    bathymetry: np.ndarray,
    *,
    plate_count: int | None = None,
) -> TectonicFoundation:
    """Infer plates from geography before generated ocean islands are added."""

    land_source = np.asarray(land_mask, dtype=bool)
    elevation_source = np.asarray(elevation, dtype=np.float64)
    bathymetry_source = np.asarray(bathymetry, dtype=np.float64)
    if (
        land_source.ndim != 2
        or elevation_source.shape != land_source.shape
        or bathymetry_source.shape != land_source.shape
    ):
        raise ValueError("tectonic source fields must be 2-D and share a shape")
    if not np.all(np.isfinite(elevation_source)) or not np.all(np.isfinite(bathymetry_source)):
        raise ValueError("tectonic source fields must be finite")
    if plate_count is not None and (
        isinstance(plate_count, bool) or not 3 <= int(plate_count) <= 24
    ):
        raise ValueError("plate_count must be between three and twenty-four")

    land = land_source
    elevation = elevation_source
    bathymetry = bathymetry_source
    elevation_norm = _normalise(elevation, land)
    vertical = np.abs(np.diff(elevation_norm, axis=0, prepend=elevation_norm[:1]))
    horizontal = np.abs(elevation_norm - np.roll(elevation_norm, 1, axis=1))
    relief_gradient = np.clip((vertical + horizontal) * 1.7, 0.0, 1.0)
    mountain = np.where(
        land,
        np.clip((elevation_norm - 0.43) / 0.43, 0.0, 1.0),
        0.0,
    )
    distance_land = _distance_from(land)
    bathy_norm = np.clip(
        bathymetry / max(1.0, float(np.max(bathymetry))),
        0.0,
        1.0,
    )
    orogen = land & ((mountain >= 0.18) | (relief_gradient >= 0.48))
    distance_orogen = _distance_from(orogen)
    local_bathy = _box_mean(bathy_norm, 4)
    margin_band = np.exp(-np.square((distance_land - 3.8) / 3.1))
    orogen_support = np.exp(-distance_orogen / 7.0)
    relative_deep = np.clip((bathy_norm - local_bathy - 0.03) / 0.28, 0.0, 1.0)
    trench = np.where(
        ~land,
        margin_band * orogen_support * (0.62 * bathy_norm + 0.38 * relative_deep),
        0.0,
    )
    relative_high = np.clip((local_bathy - bathy_norm - 0.025) / 0.24, 0.0, 1.0)
    shallow_high = np.clip((0.68 - bathy_norm) / 0.52, 0.0, 1.0)
    ridge = np.where(
        (~land) & (distance_land >= 8),
        0.68 * relative_high + 0.32 * shallow_high,
        0.0,
    )
    barrier = np.clip(
        0.82 * mountain
        + 0.34 * relief_gradient
        + 0.76 * trench
        + 0.48 * ridge,
        0.0,
        1.0,
    )
    seeds = _plate_seeds(
        land,
        elevation_norm,
        bathymetry,
        barrier,
        target_count=None if plate_count is None else int(plate_count),
    )
    labels = _grow_plates(land, bathymetry, barrier, seeds)
    boundaries, boundary_class = _boundary_inputs(
        labels,
        land,
        mountain,
        trench,
        ridge,
    )
    velocities = _solve_velocities(seeds, boundaries, labels.shape[1])
    areas = np.bincount(labels.reshape(-1), minlength=len(seeds))
    plates = tuple(
        FoundationPlate(
            identifier,
            kind,
            row,
            column,
            int(areas[identifier]),
            float(velocities[identifier, 0]),
            float(-velocities[identifier, 1]),
        )
        for identifier, (row, column, kind) in enumerate(seeds)
    )

    total_edges = sum(boundary.evidence["edgeCount"] for boundary in boundaries)
    convergent = [
        boundary for boundary in boundaries
        if boundary.classification == "convergent"
    ]
    divergent = [
        boundary for boundary in boundaries
        if boundary.classification == "divergent"
    ]
    convergent_edges = sum(boundary.evidence["edgeCount"] for boundary in convergent)
    divergent_edges = sum(boundary.evidence["edgeCount"] for boundary in divergent)
    coast_edges = sum(
        boundary.evidence["edgeCount"]
        * boundary.evidence["coastCoincidenceFraction"]
        for boundary in boundaries
    )
    diagnostics = {
        "derivation": "geography-first-shared-tectonic-foundation",
        "shape": {"height": labels.shape[0], "width": labels.shape[1]},
        "plateCount": len(plates),
        "boundaryPathCounts": {
            kind: sum(len(boundary.paths) for boundary in boundaries if boundary.classification == kind)
            for kind in _BOUNDARY_CODES
        },
        "plateCoverageFraction": float(np.mean(labels >= 0)),
        "coastBoundaryFraction": float(coast_edges / max(1.0, total_edges)),
        "convergentEvidenceFraction": float(
            sum(
                boundary.evidence["edgeCount"]
                for boundary in convergent
                if boundary.evidence["mountainEvidence"] >= 0.25
                or boundary.evidence["trenchEvidence"] >= 0.46
            )
            / max(1.0, convergent_edges)
        ),
        "divergentOceanSupportFraction": float(
            sum(
                boundary.evidence["edgeCount"]
                for boundary in divergent
                if boundary.evidence["landFraction"] <= 0.10
                and boundary.evidence["ridgeEvidence"] >= 0.20
            )
            / max(1.0, divergent_edges)
        ),
        "mountainBarrierMean": float(np.mean(barrier[orogen])) if np.any(orogen) else 0.0,
    }
    return TectonicFoundation(
        labels,
        boundary_class,
        plates,
        boundaries,
        diagnostics,
    )


def foundation_from_causal_fields(
    plate_id: np.ndarray,
    boundary_class: np.ndarray,
    land_mask: np.ndarray,
    elevation: np.ndarray,
    bathymetry: np.ndarray,
    velocity_east_cm_per_year: np.ndarray,
    velocity_north_cm_per_year: np.ndarray,
) -> TectonicFoundation:
    """Publish an already-generated plate system without inferring a new one."""

    labels = np.asarray(plate_id, dtype=np.int16)
    classes = np.asarray(boundary_class, dtype=np.int8)
    land = np.asarray(land_mask, dtype=bool)
    elevation_values = np.asarray(elevation, dtype=np.float64)
    bathymetry_values = np.asarray(bathymetry, dtype=np.float64)
    east_velocity = np.asarray(velocity_east_cm_per_year, dtype=np.float64)
    north_velocity = np.asarray(velocity_north_cm_per_year, dtype=np.float64)
    shape = labels.shape
    if len(shape) != 2 or any(
        value.shape != shape
        for value in (
            classes,
            land,
            elevation_values,
            bathymetry_values,
            east_velocity,
            north_velocity,
        )
    ):
        raise ValueError("causal tectonic fields must share one two-dimensional shape")
    if (
        np.any(labels < 0)
        or np.any((classes < BOUNDARY_INTERIOR) | (classes > BOUNDARY_TRANSFORM))
        or not all(
            np.all(np.isfinite(value))
            for value in (
                elevation_values,
                bathymetry_values,
                east_velocity,
                north_velocity,
            )
        )
    ):
        raise ValueError("causal tectonic fields contain invalid values")
    identifiers = np.unique(labels)
    if not np.array_equal(identifiers, np.arange(len(identifiers))):
        raise ValueError("causal plate identifiers must be contiguous from zero")

    height, width = shape
    class_names = {
        BOUNDARY_CONVERGENT: "convergent",
        BOUNDARY_DIVERGENT: "divergent",
        BOUNDARY_TRANSFORM: "transform",
    }
    grouped_edges: dict[
        tuple[tuple[int, int], str],
        list[tuple[tuple[int, int], tuple[int, int]]],
    ] = defaultdict(list)
    grouped_samples: dict[
        tuple[tuple[int, int], str], list[tuple[float, ...]]
    ] = defaultdict(list)

    def add_edge(
        first: tuple[int, int],
        second: tuple[int, int],
        edge: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        first_id, second_id = int(labels[first]), int(labels[second])
        if first_id == second_id:
            return
        code = max(
            (int(classes[first]), int(classes[second])),
            key=lambda value: _BOUNDARY_PRIORITY[value],
        )
        if code == BOUNDARY_INTERIOR:
            code = BOUNDARY_TRANSFORM
        classification = class_names[code]
        key = (tuple(sorted((first_id, second_id))), classification)
        grouped_edges[key].append(edge)
        grouped_samples[key].append(
            (
                float(bool(land[first]) or bool(land[second])),
                float(max(elevation_values[first], elevation_values[second])),
                float(max(bathymetry_values[first], bathymetry_values[second])),
                float(1.0 - min(bathymetry_values[first], bathymetry_values[second])),
                float(bool(land[first]) != bool(land[second])),
            )
        )

    for row in range(height):
        for column in range(width):
            if column + 1 < width:
                add_edge(
                    (row, column),
                    (row, column + 1),
                    ((column + 1, row), (column + 1, row + 1)),
                )
            elif int(labels[row, 0]) != int(labels[row, column]):
                add_edge(
                    (row, column),
                    (row, 0),
                    ((0, row), (0, row + 1)),
                )
                add_edge(
                    (row, column),
                    (row, 0),
                    ((width, row), (width, row + 1)),
                )
            if row + 1 < height:
                add_edge(
                    (row, column),
                    (row + 1, column),
                    ((column, row + 1), (column + 1, row + 1)),
                )

    boundaries: list[FoundationBoundary] = []
    for key, edges in sorted(grouped_edges.items()):
        pair, classification = key
        samples = np.asarray(grouped_samples[key], dtype=np.float64)
        boundaries.append(
            FoundationBoundary(
                pair,
                classification,
                _trace_edges(edges),
                {
                    "landFraction": float(np.mean(samples[:, 0])),
                    "mountainEvidence": float(np.quantile(samples[:, 1], 0.65)),
                    "trenchEvidence": float(np.quantile(samples[:, 2], 0.65)),
                    "ridgeEvidence": float(np.quantile(samples[:, 3], 0.65)),
                    "coastCoincidenceFraction": float(np.mean(samples[:, 4])),
                    "edgeCount": float(len(samples)),
                },
            )
        )

    plates: list[FoundationPlate] = []
    for identifier in identifiers:
        mask = labels == identifier
        rows, columns = np.where(mask)
        angles = (columns.astype(np.float64) + 0.5) * math.tau / width
        mean_angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
        if mean_angle < 0.0:
            mean_angle += math.tau
        column = int(np.clip(math.floor(mean_angle / math.tau * width), 0, width - 1))
        row = int(np.clip(round(float(np.mean(rows))), 0, height - 1))
        kind = "continental" if float(np.mean(land[mask])) >= 0.12 else "oceanic"
        plates.append(
            FoundationPlate(
                int(identifier),
                kind,
                row,
                column,
                int(np.count_nonzero(mask)),
                float(np.mean(east_velocity[mask])),
                float(np.mean(north_velocity[mask])),
            )
        )

    return TectonicFoundation(
        labels,
        classes,
        tuple(plates),
        tuple(boundaries),
        {
            "derivation": "direct-causal-procedural-plate-fields",
            "shape": {"height": height, "width": width},
            "plateCount": len(plates),
            "boundaryPathCounts": {
                name: sum(
                    len(boundary.paths)
                    for boundary in boundaries
                    if boundary.classification == name
                )
                for name in class_names.values()
            },
            "plateCoverageFraction": float(np.mean(labels >= 0)),
            "coastBoundaryFraction": float(
                sum(
                    boundary.evidence["edgeCount"]
                    * boundary.evidence["coastCoincidenceFraction"]
                    for boundary in boundaries
                )
                / max(
                    1.0,
                    sum(boundary.evidence["edgeCount"] for boundary in boundaries),
                )
            ),
        },
    )


__all__ = [
    "BOUNDARY_CONVERGENT",
    "BOUNDARY_DIVERGENT",
    "BOUNDARY_INTERIOR",
    "BOUNDARY_TRANSFORM",
    "FoundationBoundary",
    "FoundationPlate",
    "TectonicFoundation",
    "derive_tectonic_foundation",
    "foundation_from_causal_fields",
]
