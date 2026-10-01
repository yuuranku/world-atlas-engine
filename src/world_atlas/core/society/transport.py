"""Deterministic roads, navigable rivers, sea lanes, and accessibility."""

from __future__ import annotations

import heapq
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt

from ..model import WorldGrid
from ..suitability import relative_land_slope
from ..thematic import ThematicLayers, smooth_field
from ..road_engineering import RoadEngineering
from .model import (
    Bridge,
    PoliticalLayers,
    PopulationLayers,
    Settlement,
    TransportLayers,
    TransportRoute,
)
from .spatial import connected_components, reduce_field


_NEIGHBORS = (
    (-1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)),
    (1, 0, 1.0),
    (1, 1, math.sqrt(2.0)),
)
_TIER_RANK = {"metropolis": 2, "city": 1, "town": 0, "site": 0}
_ROAD_LOCAL_DEGREE = {"metropolis": 3, "city": 2, "town": 1, "site": 1}
_TECHNOLOGY_ERAS = frozenset(
    {"tribal", "ancient", "medieval", "early-modern", "preindustrial", "industrial", "contemporary"}
)
_RAIL_ERAS = frozenset({"industrial", "contemporary"})


def _technology_era(grid: WorldGrid) -> str:
    """Return the strict transport era carried by the canonical world profile."""

    profile = grid.metadata.get("worldProfile", {})
    era = profile.get("technologyEra") if isinstance(profile, Mapping) else None
    if era not in _TECHNOLOGY_ERAS:
        raise ValueError(
            "transport requires a strict worldProfile.technologyEra; choose one of: "
            + ", ".join(sorted(_TECHNOLOGY_ERAS))
        )
    return era


@dataclass
class RoutingCache:
    """Reuse exact routes within one human-world simulation, never across worlds."""

    paths: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @staticmethod
    def signature(*arrays: np.ndarray) -> bytes:
        digest = hashlib.sha256()
        for array in arrays:
            values = np.ascontiguousarray(array)
            digest.update(str((values.shape, values.dtype.str)).encode())
            digest.update(memoryview(values).cast("B"))
        return digest.digest()

    def route(self, signature, friction, valid, start, goal, refinement, *, simplify: bool = True,
              blocked_edges=None, river_barrier=None, edge_costs=None):
        key = (signature, start, goal, simplify)
        if key in self.paths:
            self.hits += 1
            return self.paths[key]
        self.misses += 1
        path = _least_cost_path(friction, valid, start, goal,blocked_edges=blocked_edges,edge_costs=edge_costs)
        result = _terrain_safe_simplification(path, refinement, valid,river_barrier=river_barrier,edge_costs=edge_costs) if path and simplify else path
        self.paths[key] = result
        return result


def _wrapped_distance(
    first: tuple[int, int],
    second: tuple[int, int],
    width: int,
) -> float:
    dy = first[0] - second[0]
    dx = abs(first[1] - second[1])
    return math.hypot(dy, min(dx, width - dx))


def _least_cost_path(
    friction: np.ndarray,
    valid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *, blocked_edges=None, edge_costs=None,
) -> tuple[tuple[int, int], ...]:
    """A* path on a wrapped raster; an empty tuple means disconnected."""

    if start == goal:
        return (start, goal)
    height, width = friction.shape
    valid_friction = friction[valid]
    heuristic_scale = max(
        1.0e-9,
        float(valid_friction.min(initial=1.0)),
    )
    if edge_costs is not None:
        heuristic_scale = edge_costs.heuristic_scale
        if not math.isfinite(heuristic_scale):
            return ()
        heuristic_cost = edge_costs.heuristic(goal)
        edge_costs = memoryview(edge_costs.values)
    else:
        heuristic_cost = lambda point: _wrapped_distance(point,goal,width)*heuristic_scale
    # Buffer access returns native scalars; avoid millions of NumPy scalar
    # allocations while keeping costs, heap ordering and ties identical.
    friction = memoryview(np.ascontiguousarray(friction, dtype=np.float64))
    valid = memoryview(np.ascontiguousarray(valid, dtype=bool))
    costs = {start: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue: list[tuple[float, float, int, int]] = [
        (
            heuristic_cost(start),
            0.0,
            start[0],
            start[1],
        )
    ]
    while queue:
        _priority, cost, row, column = heapq.heappop(queue)
        current = (row, column)
        if cost != costs.get(current):
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(previous[path[-1]])
            path.reverse()
            return tuple(path)
        for direction, (dy, dx, distance) in enumerate(_NEIGHBORS):
            if blocked_edges is not None and blocked_edges[row,column] & (1 << direction):
                continue
            next_row = row + dy
            if next_row < 0 or next_row >= height:
                continue
            next_column = (column + dx) % width
            if not valid[next_row, next_column]:
                continue
            if dy and dx:
                # A diagonal may not jump across the corner of a mountain,
                # coast, or island channel.  Both orthogonal cells must be
                # traversable, otherwise the visible route leaves its real
                # physical corridor.
                if not (
                    valid[row, next_column]
                    and valid[next_row, column]
                ):
                    continue
            edge = float(edge_costs[row,column,direction]) if edge_costs is not None else 0.5 * (
                float(friction[row, column]) + float(friction[next_row, next_column])
            ) * distance
            if not math.isfinite(edge):
                continue
            next_cost = cost + edge
            point = (next_row, next_column)
            if next_cost >= costs.get(point, math.inf) - 1.0e-12:
                continue
            costs[point] = next_cost
            previous[point] = current
            heuristic = heuristic_cost(point)
            heapq.heappush(
                queue,
                (next_cost + heuristic, next_cost, next_row, next_column),
            )
    return ()


def _simplify_lattice_path(
    path: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    if len(path) <= 2:
        return path
    result = [path[0]]
    previous_direction: tuple[int, int] | None = None
    for index in range(1, len(path)):
        dy = int(np.sign(path[index][0] - path[index - 1][0]))
        dx = int(np.sign(path[index][1] - path[index - 1][1]))
        direction = (dy, dx)
        if previous_direction is not None and direction != previous_direction:
            result.append(path[index - 1])
        previous_direction = direction
    result.append(path[-1])
    return tuple(result)


def _world_path(
    path: tuple[tuple[int, int], ...],
    step: int,
    shape: tuple[int, int],
    source: Settlement,
    target: Settlement | None,
    *,
    simplify: bool = True,
    snap_to_settlements: bool = True,
) -> tuple[tuple[float, float], ...]:
    lattice_path = _simplify_lattice_path(path) if simplify else path
    points = [
        (
            min(shape[1] - 0.5, column * step + step * 0.5),
            min(shape[0] - 0.5, row * step + step * 0.5),
        )
        for row, column in lattice_path
    ]
    if snap_to_settlements:
        points[0] = (source.column + 0.5, source.row + 0.5)
        if target is not None:
            points[-1] = (target.column + 0.5, target.row + 0.5)
    return tuple(points)


def _terrain_safe_simplification(
    path: tuple[tuple[int, int], ...],
    friction: np.ndarray,
    valid: np.ndarray,
    *,
    maximum_span: int = 6,
    river_barrier=None,
    edge_costs=None,
) -> tuple[tuple[int, int], ...]:
    """Remove flat-land lattice noise without cutting across real relief."""

    if len(path) <= 2:
        return path
    if edge_costs is not None:
        # Grade-aware road stations are authoritative. Retain turns instead of
        # replacing them with chords whose construction cost was never routed.
        return _simplify_lattice_path(path)
    height, width = valid.shape

    def line_cells(
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> tuple[tuple[int, int], ...]:
        dy = end[0] - start[0]
        dx = end[1] - start[1]
        if dx > width * 0.5:
            dx -= width
        elif dx < -width * 0.5:
            dx += width
        # Four samples per crossed cell catch thin capes and island corners
        # that a two-sample visual chord can otherwise cross between checks.
        steps = max(1, int(math.ceil(max(abs(dy), abs(dx)) * 4.0)))
        result: list[tuple[int, int]] = []
        for offset in range(steps + 1):
            fraction = offset / steps
            row = min(
                height - 1,
                max(0, int(math.floor(start[0] + 0.5 + dy * fraction))),
            )
            column = int(math.floor(start[1] + 0.5 + dx * fraction)) % width
            cell = (row, column)
            if not result or result[-1] != cell:
                result.append(cell)
        return tuple(result)

    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        selected = anchor + 1
        furthest = min(len(path) - 1, anchor + maximum_span)
        original_values: list[float] = []
        for candidate in range(anchor + 1, furthest + 1):
            original_values.append(float(friction[path[candidate]]))
            # Preserve the actual valley route. Similar endpoint friction
            # does not justify replacing a winding corridor with a long chord.
            source = np.asarray(path[anchor:candidate + 1], dtype=np.float64)
            source[:, 1] = source[0, 1] + (source[:, 1] - source[0, 1] + width / 2) % width - width / 2
            delta = source[-1] - source[0]
            length_squared = float(np.dot(delta, delta))
            if length_squared > 0:
                projection = np.clip((source - source[0]) @ delta / length_squared, 0, 1)
                deviation = np.linalg.norm(source - (source[0] + projection[:, None] * delta), axis=1)
                if float(deviation.max()) > 0.35:
                    continue
            segment = line_cells(path[anchor], path[candidate])
            if river_barrier is not None:
                chord = shapely.LineString([(point[1]+.5,point[0]+.5) for point in source[[0,-1]]])
                if shapely.crosses(chord,river_barrier):
                    continue
            rows, columns = zip(*segment, strict=True)
            if not np.all(valid[rows, columns]):
                continue
            segment_values = friction[rows, columns].astype(np.float64)
            reference = np.asarray(
                [float(friction[path[anchor]]), *original_values],
                dtype=np.float64,
            )
            if (
                float(segment_values.max(initial=0.0))
                <= float(reference.max(initial=0.0)) * 1.025 + 0.04
                and float(segment_values.mean())
                <= float(reference.mean()) * 1.035 + 0.03
            ):
                selected = candidate
        simplified.append(path[selected])
        anchor = selected
    return tuple(simplified)


def _refine_road_path(
    coarse_path: tuple[tuple[int, int], ...],
    *,
    step: int,
    friction: np.ndarray,
    land: np.ndarray,
    source: Settlement,
    target: Settlement,
) -> tuple[tuple[float, float], ...]:
    """Resolve a coarse route onto exact land cells inside its local corridor."""

    height, width = land.shape
    corridor = np.zeros(land.shape, dtype=bool)
    # The fine route needs room to leave the coarse cell centre and find the
    # actual valley floor or low pass.  A four-pixel tube merely reproduced a
    # straight coarse link across nearby relief.
    padding = max(5, step * 3)
    for row, column in coarse_path:
        center_row = min(height - 1, row * step + step // 2)
        center_column = (column * step + step // 2) % width
        row_start = max(0, center_row - padding)
        row_stop = min(height, center_row + padding + 1)
        columns = np.arange(
            center_column - padding,
            center_column + padding + 1,
            dtype=np.int64,
        ) % width
        corridor[row_start:row_stop, columns] = True
    start = (source.row, source.column)
    goal = (target.row, target.column)
    corridor[start] = True
    corridor[goal] = True
    exact_path = _least_cost_path(friction, corridor & land, start, goal)
    if not exact_path:
        return ()
    exact_path = _terrain_safe_simplification(
        exact_path,
        friction,
        corridor & land,
    )
    return _world_path(
        exact_path,
        1,
        land.shape,
        source,
        target,
        simplify=False,
    )


def _nearest_valid(
    valid: np.ndarray,
    origin: tuple[int, int],
    *,
    radius: int = 3,
) -> tuple[int, int] | None:
    height, width = valid.shape
    row, column = origin
    if 0 <= row < height and valid[row, column % width]:
        return row, column % width
    candidates: list[tuple[float, int, int]] = []
    for dy in range(-radius, radius + 1):
        next_row = row + dy
        if next_row < 0 or next_row >= height:
            continue
        for dx in range(-radius, radius + 1):
            next_column = (column + dx) % width
            if valid[next_row, next_column]:
                candidates.append((math.hypot(dy, dx), next_row, next_column))
    if not candidates:
        return None
    _distance, result_row, result_column = min(candidates)
    return result_row, result_column


def _route_importance(source: Settlement, target: Settlement | None) -> str:
    if target is not None and source.tier == target.tier == "metropolis":
        return "trunk"
    if source.tier != "town" or (target is not None and target.tier != "town"):
        return "regional"
    return "local"


def _connect_pair_components(
    settlements: tuple[Settlement, ...],
    cells: dict[str, tuple[int, int]],
    components: np.ndarray,
    pairs: set[tuple[str, str]],
) -> None:
    """Add the shortest missing links until each physical component is connected."""

    available = tuple(
        sorted(
            (item for item in settlements if item.identifier in cells),
            key=lambda item: item.identifier,
        )
    )
    parent = {item.identifier: item.identifier for item in available}

    def find(identifier: str) -> str:
        root = identifier
        while parent[root] != root:
            root = parent[root]
        while parent[identifier] != identifier:
            next_identifier = parent[identifier]
            parent[identifier] = root
            identifier = next_identifier
        return root

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, second in sorted(pairs):
        if first in parent and second in parent:
            union(first, second)

    width = components.shape[1]
    component_ids = sorted({int(components[cells[item.identifier]]) for item in available})
    for component_id in component_ids:
        members = tuple(
            item
            for item in available
            if int(components[cells[item.identifier]]) == component_id
        )
        candidates: list[tuple[float, str, str]] = []
        for index, source in enumerate(members):
            for target in members[index + 1 :]:
                candidates.append(
                    (
                        _wrapped_distance(
                            cells[source.identifier],
                            cells[target.identifier],
                            width,
                        ),
                        source.identifier,
                        target.identifier,
                    )
                )
        for _distance, first, second in sorted(candidates):
            if find(first) == find(second):
                continue
            pairs.add(tuple(sorted((first, second))))
            union(first, second)


def _road_pairs(
    settlements: tuple[Settlement, ...],
    cells: dict[str, tuple[int, int]],
    components: np.ndarray,
) -> tuple[tuple[Settlement, Settlement], ...]:
    width = components.shape[1]
    pairs: set[tuple[str, str]] = set()
    by_identifier = {item.identifier: item for item in settlements}
    for source in settlements:
        source_cell = cells[source.identifier]
        component = int(components[source_cell])
        candidates = [
            target
            for target in settlements
            if target.identifier != source.identifier
            and int(components[cells[target.identifier]]) == component
            and (
                _TIER_RANK[target.tier] >= _TIER_RANK[source.tier]
                or source.tier == "metropolis"
            )
        ]
        if not candidates:
            candidates = [
                target
                for target in settlements
                if target.identifier != source.identifier
                and int(components[cells[target.identifier]]) == component
            ]
        candidates.sort(
            key=lambda target: (
                _wrapped_distance(source_cell, cells[target.identifier], width)
                * (0.72 if target.holy_religion_identifier is not None else 1.0),
                target.holy_religion_identifier is None,
                -_TIER_RANK[target.tier],
                target.identifier,
            )
        )
        degree = _ROAD_LOCAL_DEGREE[source.tier] + int(
            source.holy_religion_identifier is not None
        )
        for target in candidates[:degree]:
            pairs.add(tuple(sorted((source.identifier, target.identifier))))
    _connect_pair_components(settlements, cells, components, pairs)
    return tuple(
        (by_identifier[first], by_identifier[second])
        for first, second in sorted(pairs)
    )


def _river_routes(
    grid: WorldGrid,
    settlements: tuple[Settlement, ...],
) -> list[TransportRoute]:
    routes: list[TransportRoute] = []
    river_sites = {
        (item.row, item.column): item
        for item in settlements
        if item.site_type == "river-city"
    }
    for source in sorted(river_sites.values(), key=lambda item: item.identifier):
        flat = source.row * grid.shape[1] + source.column
        seen: set[int] = set()
        points: list[tuple[float, float]] = [(source.column + 0.5, source.row + 0.5)]
        target: Settlement | None = None
        while flat >= 0 and flat not in seen and len(points) < grid.shape[0] * grid.shape[1]:
            seen.add(flat)
            row, column = divmod(flat, grid.shape[1])
            next_flat = int(grid.flow_to[row, column])
            if next_flat < 0:
                break
            next_row, next_column = divmod(next_flat, grid.shape[1])
            points.append((next_column + 0.5, next_row + 0.5))
            candidate = river_sites.get((next_row, next_column))
            if candidate is not None and candidate.identifier != source.identifier:
                target = candidate
                break
            flat = next_flat
            if grid.water[next_row, next_column] != 0:
                break
        if len(points) < 5:
            continue
        # River navigation is already expressed on the canonical native-cell
        # flow graph.  Retain every point: interval subsampling drew straight
        # chords across bends and visually detached the route from its river.
        exact_path = tuple(points)
        routes.append(
            TransportRoute(
                identifier=f"river-{len(routes) + 1:04d}",
                mode="river",
                importance="trunk" if grid.river_order[source.row, source.column] >= 3 else "regional",
                source_settlement_id=source.identifier,
                target_settlement_id=None if target is None else target.identifier,
                path=exact_path,
            )
        )
    return routes


def _mark_segment(field: np.ndarray, start: tuple[float, float], end: tuple[float, float], value: float) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    count = max(1, int(math.ceil(max(abs(dx), abs(dy)))))
    for index in range(count + 1):
        fraction = index / count
        column = int(round(start[0] + dx * fraction - 0.5)) % field.shape[1]
        row = int(round(start[1] + dy * fraction - 0.5))
        if 0 <= row < field.shape[0]:
            field[row, column] = max(float(field[row, column]), value)


def _accessibility_field(
    shape: tuple[int, int],
    settlements: tuple[Settlement, ...],
    routes: tuple[TransportRoute, ...],
) -> np.ndarray:
    """Map-plane proximity to transport, with periodic longitude.

    This measures local access, not terrain travel time. Each source strength
    competes by exact Euclidean distance; four-neighbour propagation would
    instead impose Manhattan diamonds on every downstream control field.
    """
    field = np.zeros(shape, dtype=np.float64)
    route_value = {"trunk": 0.82, "regional": 0.66, "local": 0.48}
    for route in routes:
        value = route_value[route.importance]
        for start, end in zip(route.path, route.path[1:]):
            _mark_segment(field, start, end, value)
    tier_value = {"metropolis": 1.0, "city": 0.84, "town": 0.62, "site": 0.58}
    for settlement in settlements:
        field[settlement.row, settlement.column] = max(
            field[settlement.row, settlement.column], tier_value[settlement.tier]
        )
    radius = max(2, min(18, min(shape) // 42))
    current = np.zeros(shape, dtype=np.float64)
    halo = radius + 1
    for value in np.unique(field[field > 0]):
        longitude = np.pad(field != value, ((0, 0), (halo, halo)), mode="wrap")
        padded = np.pad(longitude, ((halo, halo), (0, 0)),
                        mode="constant", constant_values=True)
        distance = distance_transform_edt(padded)[halo:-halo, halo:-halo]
        current = np.maximum(current, np.where(
            distance <= radius, value * np.power(0.88, distance), 0.0))
    return np.clip(current, 0.0, 1.0).astype(np.float32)


def _rail_engineering_fields(
    grid: WorldGrid,
    thematic: ThematicLayers,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native railway construction friction and traversable ground.

    Railway engineering is deliberately separate from roads: grades and local
    relief dominate its cost, water is never traversable, and a represented
    river channel has a large penalty so a line follows neither its thalweg nor
    a decorative road corridor.  Crossing a river remains possible only when
    the later bridge validation accepts a short transverse span.
    """

    elevation = np.asarray(grid.elevation, dtype=np.float64)
    water = np.asarray(grid.water) != 0
    snow = np.asarray(grid.snow, dtype=bool)
    raw_grade = relative_land_slope(elevation, ~water)
    ordinary_land = ~water & ~snow & (elevation < 0.90)
    reference = float(np.percentile(raw_grade[ordinary_land], 92.0)) if np.any(ordinary_land) else 0.0
    relative_grade = raw_grade / max(reference, 1.0e-9)
    broad = smooth_field(
        elevation,
        radius=max(4, min(12, min(grid.shape) // 86)),
        passes=2,
    )
    relief = np.clip(elevation - broad, 0.0, 1.0)
    river_order = np.asarray(grid.river_order, dtype=np.float64)
    # A route can use a shallow bridge cell when necessary, but cannot use a
    # summit face or an enduring snowfield as an engineering shortcut.
    valid = ordinary_land & (relative_grade <= 3.7)
    friction = (
        1.0
        + 16.0 * np.square(np.clip(relative_grade, 0.0, 3.7))
        + 92.0 * np.square(np.clip(relative_grade - 1.15, 0.0, 2.55))
        + 44.0 * relief
        + 18.0 * np.square(np.clip(elevation - 0.42, 0.0, 0.58))
        + 2.8 * (1.0 - np.clip(thematic.land_potential, 0.0, 1.0))
        # A bridge is costly, but a continental rail should not take a
        # planet-scale detour merely to avoid one crossable river cell.  The
        # subsequent transverse-span gate rejects any path that tries to use a
        # river as a longitudinal corridor.
        + np.where(river_order > 0.0, 8.0 + river_order * 3.5, 0.0)
    )
    return np.clip(friction, 0.82, None), valid


def _rail_terminals(
    settlements: tuple[Settlement, ...],
    components: np.ndarray,
    valid: np.ndarray,
    *,
    era: str,
) -> tuple[Settlement, ...]:
    """Select a bounded, high-demand terminal set before expensive routing."""

    caps = {"industrial": 12, "contemporary": 24}
    grouped: dict[int, list[Settlement]] = {}
    for settlement in settlements:
        if settlement.tier not in {"city", "metropolis"}:
            continue
        cell = (settlement.row, settlement.column)
        if not valid[cell] or int(components[cell]) <= 0:
            continue
        grouped.setdefault(int(components[cell]), []).append(settlement)
    selected: list[Settlement] = []
    for component in sorted(grouped):
        candidates = grouped[component]
        candidates.sort(
            key=lambda item: (
                -_TIER_RANK[item.tier],
                -item.population_max,
                -item.population_min,
                item.identifier,
            )
        )
        cap = min(caps[era], max(2, int(math.ceil(math.sqrt(len(candidates)) * 3.2))))
        selected.extend(candidates[:cap])
    return tuple(sorted(selected, key=lambda item: item.identifier))


def _rail_pairs(
    terminals: tuple[Settlement, ...],
    components: np.ndarray,
    *,
    era: str,
) -> tuple[tuple[Settlement, Settlement], ...]:
    """Form a sparse terminal forest without borrowing the road graph."""

    by_identifier = {item.identifier: item for item in terminals}
    cells = {item.identifier: (item.row, item.column) for item in terminals}
    width = components.shape[1]
    pairs: set[tuple[str, str]] = set()
    degree_bonus = 1 if era == "contemporary" else 0
    for source in terminals:
        component = int(components[cells[source.identifier]])
        candidates = [
            target
            for target in terminals
            if target.identifier != source.identifier
            and int(components[cells[target.identifier]]) == component
        ]
        candidates.sort(
            key=lambda target: (
                _wrapped_distance(cells[source.identifier], cells[target.identifier], width)
                / (1.0 + 0.16 * _TIER_RANK[target.tier]),
                -_TIER_RANK[target.tier],
                -target.population_max,
                target.identifier,
            )
        )
        degree = 1 + degree_bonus + int(source.tier == "metropolis")
        for target in candidates[:degree]:
            pairs.add(tuple(sorted((source.identifier, target.identifier))))
    _connect_pair_components(terminals, cells, components, pairs)
    return tuple((by_identifier[first], by_identifier[second]) for first, second in sorted(pairs))


def _refine_rail_path(
    coarse_path: tuple[tuple[int, int], ...],
    *,
    step: int,
    friction: np.ndarray,
    valid: np.ndarray,
    source: Settlement,
    target: Settlement,
) -> tuple[tuple[int, int], ...]:
    """Resolve a coarse engineering corridor without straight-line shortcuts."""

    height, width = valid.shape
    corridor = np.zeros(valid.shape, dtype=bool)
    padding = max(6, step * 4)
    for row, column in coarse_path:
        center_row = min(height - 1, row * step + step // 2)
        center_column = (column * step + step // 2) % width
        row_start = max(0, center_row - padding)
        row_stop = min(height, center_row + padding + 1)
        columns = np.arange(center_column - padding, center_column + padding + 1, dtype=np.int64) % width
        corridor[row_start:row_stop, columns] = True
    start = (source.row, source.column)
    goal = (target.row, target.column)
    corridor[start] = True
    corridor[goal] = True
    return _least_cost_path(friction, corridor & valid, start, goal)


def _rail_bridge_permitted(era: str, importance: str, river_order: int, run_length: int) -> bool:
    """Keep rail bridges shorter and more selective than ordinary road spans."""

    maximum_order = {
        "industrial": {"local": 1, "regional": 3, "trunk": 5},
        "contemporary": {"local": 2, "regional": 4, "trunk": 6},
    }[era][importance]
    maximum_run = {
        "industrial": {"local": 1, "regional": 2, "trunk": 3},
        "contemporary": {"local": 1, "regional": 3, "trunk": 4},
    }[era][importance]
    return river_order <= maximum_order and run_length <= maximum_run


def _rail_path_is_valid(
    grid: WorldGrid,
    path: tuple[tuple[int, int], ...],
    *,
    era: str,
    importance: str,
) -> bool:
    if len(path) < 2 or any(int(grid.water[row, column]) != 0 for row, column in path):
        return False
    for run in _river_runs(path, grid.river_order):
        crossing = _classify_transverse_crossing(grid, path, run)
        if crossing is None:
            return False
        if not _rail_bridge_permitted(era, importance, crossing[2], run[1] - run[0] + 1):
            return False
    return True


def _rail_routes(
    grid: WorldGrid,
    thematic: ThematicLayers,
    settlements: tuple[Settlement, ...],
    *,
    routing_cache: RoutingCache,
) -> tuple[TransportRoute, ...]:
    """Generate independent, terrain-engineered rail corridors by era."""

    era = _technology_era(grid)
    if era not in _RAIL_ERAS:
        return ()
    friction, valid = _rail_engineering_fields(grid, thematic)
    # City centres on water, permanent snow, a represented river channel, or
    # an engineering-grade cliff are not silently snapped across terrain.
    valid = valid.copy()
    for settlement in settlements:
        if int(grid.river_order[settlement.row, settlement.column]) > 0:
            valid[settlement.row, settlement.column] = False
    components, _sizes = connected_components(valid)
    terminals = _rail_terminals(settlements, components, valid, era=era)
    if len(terminals) < 2:
        return ()
    pairs = _rail_pairs(terminals, components, era=era)
    if not pairs:
        return ()
    step = max(2, min(8, min(grid.shape) // 160))
    coarse_friction = reduce_field(friction, step=step, mode="mean")
    coarse_valid = reduce_field(valid, step=step, mode="mean") >= 0.62
    for terminal in terminals:
        coarse_valid[terminal.row // step, terminal.column // step] = True
    signature = routing_cache.signature(coarse_friction, coarse_valid, friction, valid)
    routes: list[TransportRoute] = []
    for source, target in pairs:
        source_cell = _nearest_valid(
            coarse_valid,
            (source.row // step, source.column // step),
            radius=2,
        )
        target_cell = _nearest_valid(
            coarse_valid,
            (target.row // step, target.column // step),
            radius=2,
        )
        if source_cell is None or target_cell is None:
            continue
        coarse_path = routing_cache.route(
            signature,
            coarse_friction,
            coarse_valid,
            source_cell,
            target_cell,
            coarse_friction,
            simplify=False,
        )
        if not coarse_path:
            continue
        importance = _route_importance(source, target)
        native_path = _refine_rail_path(
            coarse_path,
            step=step,
            friction=friction,
            valid=valid,
            source=source,
            target=target,
        )
        if not _rail_path_is_valid(grid, native_path, era=era, importance=importance):
            continue
        routes.append(
            TransportRoute(
                identifier=f"rail-{len(routes) + 1:04d}",
                mode="rail",
                importance=importance,
                source_settlement_id=source.identifier,
                target_settlement_id=target.identifier,
                path=_world_path(
                    native_path,
                    1,
                    grid.shape,
                    source,
                    target,
                    simplify=False,
                ),
            )
        )
    return tuple(routes)


def derive_transport(
    grid: WorldGrid,
    thematic: ThematicLayers,
    population: PopulationLayers,
    settlements: tuple[Settlement, ...],
    *,
    raw_elevation_m: np.ndarray,
    routing_cache: RoutingCache | None = None,
) -> TransportLayers:
    """Build roads and water routes before any state border is generated."""

    if thematic.land_potential.shape != grid.shape or population.population_weight.shape != grid.shape:
        raise ValueError("transport inputs must match the WorldGrid shape")
    if not settlements:
        return TransportLayers(
            accessibility=np.zeros(grid.shape, dtype=np.float32),
            routes=(),
        )
    routing_cache = routing_cache if routing_cache is not None else RoutingCache()
    # The old 360-cell analysis lattice was adequate for accessibility but
    # visibly bridged bays and rounded straight across mountain spurs.  Keep
    # roughly one route node per two authored map pixels instead.
    step = 1
    road_land = (
        (grid.water == 0)
        & ~grid.snow
        & (grid.elevation < 0.88)
    )
    # A mountain settlement may itself sit just above the normal through-road
    # ceiling; keep the node valid while forbidding a route from using an
    # extended snowfield or summit ridge as a shortcut.
    for settlement in settlements:
        if grid.water[settlement.row, settlement.column] == 0:
            road_land[settlement.row, settlement.column] = True
    land_fraction = reduce_field(road_land, step=step, mode="mean")
    # Coastlines and inhabited small islands occupy only part of a transport
    # analysis cell.  A majority-land threshold erased those legitimate
    # gateways before routing even began.
    coarse_land = land_fraction >= 0.68
    elevation = reduce_field(grid.elevation, step=step, mode="mean")
    summit = reduce_field(grid.elevation, step=step, mode="max")
    potential = reduce_field(thematic.land_potential, step=step, mode="mean")
    river = reduce_field(grid.river_order, step=step, mode="max")
    full_slope = relative_land_slope(grid.elevation, grid.water == 0)
    slope = reduce_field(full_slope, step=step, mode="max") * step
    broad_elevation = smooth_field(elevation, radius=4, passes=2)
    ridge_excess = np.clip(elevation - broad_elevation, 0.0, 1.0)
    friction = (
        1.0
        + 7.5 * np.clip(elevation - 0.32, 0.0, 0.68)
        + 24.0 * np.square(np.clip(summit - 0.48, 0.0, 0.52))
        + 76.0 * slope
        + 18.0 * ridge_excess
        + 1.6 * (1.0 - np.clip(potential, 0.0, 1.0))
        - 0.62 * np.clip(river.astype(np.float64) / 3.0, 0.0, 1.0)
    )
    friction = np.clip(friction, 0.72, None)
    relief_radius = max(4, min(11, min(grid.shape) // 90))
    broad_full_elevation = smooth_field(
        grid.elevation.astype(np.float64),
        radius=relief_radius,
        passes=2,
    )
    full_ridge_excess = np.clip(
        grid.elevation.astype(np.float64) - broad_full_elevation,
        0.0,
        1.0,
    )
    river_corridor = smooth_field(
        (grid.river_order > 0).astype(np.float64),
        radius=max(2, min(7, min(grid.shape) // 150)),
        passes=1,
    )
    full_friction = (
        1.0
        + 7.5 * np.clip(grid.elevation - 0.32, 0.0, 0.68)
        + 24.0 * np.square(np.clip(grid.elevation - 0.48, 0.0, 0.52))
        + 76.0 * full_slope
        + 18.0 * full_ridge_excess
        + 1.6 * (1.0 - np.clip(thematic.land_potential, 0.0, 1.0))
        - 0.48 * np.clip(river_corridor, 0.0, 1.0)
        - 0.42 * np.clip(grid.river_order.astype(np.float64) / 3.0, 0.0, 1.0)
    )
    full_friction = np.clip(full_friction, 0.72, None)
    engineering = RoadEngineering(raw_elevation_m, float(grid.metadata['planet']['radiusKm']), _technology_era(grid))
    road_edges = engineering.edge_costs(full_friction)
    cells: dict[str, tuple[int, int]] = {}
    for settlement in settlements:
        origin = (settlement.row // step, settlement.column // step)
        cell = _nearest_valid(coarse_land, origin, radius=2)
        if cell is None:
            # The source settlement is already guaranteed to lie on a full-
            # resolution land cell. Preserve that tiny island/coastal cell in
            # the coarse graph instead of silently deleting the settlement.
            coarse_land[origin] = True
            cell = origin
        cells[settlement.identifier] = cell
    components, _sizes = connected_components(coarse_land)
    road_constraints = {importance:_road_river_constraints(grid,importance,raw_elevation_m=raw_elevation_m)
                        for importance in ("local","regional","trunk")}
    road_signatures = {importance:routing_cache.signature(friction,coarse_land&constraint[0],full_friction,constraint[1],road_edges.values)
                       for importance,constraint in road_constraints.items()}
    routes: list[TransportRoute] = []
    for source, target in _road_pairs(settlements, cells, components):
        importance = _route_importance(source,target)
        river_valid,blocked_edges,river_barrier = road_constraints[importance]
        valid=coarse_land&river_valid
        valid[cells[source.identifier]]=True
        valid[cells[target.identifier]]=True
        path = routing_cache.route(road_signatures[importance], friction, valid,
            cells[source.identifier], cells[target.identifier], full_friction,
            blocked_edges=blocked_edges,river_barrier=river_barrier,edge_costs=road_edges)
        if not path:
            continue
        if step == 1:
            exact_path = path
            world_path = _world_path(
                exact_path,
                1,
                grid.shape,
                source,
                target,
                simplify=False,
            )
        else:
            world_path = _refine_road_path(
                path,
                step=step,
                friction=full_friction,
                land=road_land,
                source=source,
                target=target,
            )
        if not world_path:
            continue
        routes.append(
            TransportRoute(
                identifier=f"road-{len(routes) + 1:04d}",
                mode="road",
                importance=importance,
                source_settlement_id=source.identifier,
                target_settlement_id=target.identifier,
                path=world_path,
            )
        )
    routes.extend(_river_routes(grid, settlements))

    # Ports on the same water body receive a sparse two-neighbour sea network.
    for water_values, site_type in (((1, 3), "port"), ((2,), "lake-port")):
        coarse_water = reduce_field(np.isin(grid.water, water_values), step=step, mode="mean") >= 0.34
        water_components, _water_sizes = connected_components(coarse_water)
        ports = [
            item
            for item in settlements
            if item.site_type == site_type
            or (site_type == "port" and item.site_type == "island-port")
        ]
        water_cell: dict[str, tuple[int, int]] = {}
        for port in ports:
            cell = _nearest_valid(
                coarse_water,
                (port.row // step, port.column // step),
                radius=3,
            )
            if cell is not None:
                water_cell[port.identifier] = cell
        pairs: set[tuple[str, str]] = set()
        for source in ports:
            if source.identifier not in water_cell:
                continue
            source_cell = water_cell[source.identifier]
            candidates = [
                target
                for target in ports
                if target.identifier != source.identifier
                and target.identifier in water_cell
                and water_components[water_cell[target.identifier]] == water_components[source_cell]
            ]
            candidates.sort(
                key=lambda target: (
                    _wrapped_distance(source_cell, water_cell[target.identifier], coarse_water.shape[1]),
                    target.identifier,
                )
            )
            for target in candidates[:2]:
                pairs.add(tuple(sorted((source.identifier, target.identifier))))
        routed_ports = tuple(
            item for item in ports if item.identifier in water_cell
        )
        _connect_pair_components(
            routed_ports,
            water_cell,
            water_components,
            pairs,
        )
        by_identifier = {item.identifier: item for item in ports}
        water_friction = np.ones(coarse_water.shape, dtype=np.float64)
        water_signature = routing_cache.signature(water_friction, coarse_water, water_friction)
        for first, second in sorted(pairs):
            source = by_identifier[first]
            target = by_identifier[second]
            path = routing_cache.route(
                water_signature,
                water_friction,
                coarse_water,
                water_cell[first],
                water_cell[second],
                water_friction,
            )
            if not path:
                continue
            routes.append(
                TransportRoute(
                    identifier=f"sea-{len([route for route in routes if route.mode == 'sea']) + 1:04d}",
                    mode="sea",
                    importance=_route_importance(source, target),
                    source_settlement_id=source.identifier,
                    target_settlement_id=target.identifier,
                    # Sea lanes terminate at the nearest navigable berth rather
                    # than drawing inland to the centre of the port cell.  The
                    # terrain-safe simplifier also prevents a long visual chord
                    # from cutting across a cape or island.
                    path=_world_path(
                        path,
                        step,
                        grid.shape,
                        source,
                        target,
                        simplify=False,
                        snap_to_settlements=False,
                    ),
                )
            )
    # Rails are an era-specific engineering layer. They never alter the road
    # corridor, settlement placement, or political simulation that already
    # passed physical routing checks.
    routes.extend(
        _rail_routes(
            grid,
            thematic,
            settlements,
            routing_cache=routing_cache,
        )
    )
    finalized = tuple(routes)
    return TransportLayers(
        accessibility=_accessibility_field(grid.shape, settlements, finalized),
        routes=finalized,
        bridges=derive_bridges(grid, finalized,raw_elevation_m=raw_elevation_m),
    )


def _path_cells(
    path: tuple[tuple[float, float], ...],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    height, width = shape
    cells: list[tuple[int, int]] = []
    for (start_x, start_y), (end_x, end_y) in zip(path, path[1:]):
        dx = end_x - start_x
        if dx > width * 0.5:
            dx -= width
        elif dx < -width * 0.5:
            dx += width
        dy = end_y - start_y
        steps = max(1, int(math.ceil(max(abs(dx), abs(dy)) * 4.0)))
        for offset in range(steps + 1):
            fraction = offset / steps
            row = min(height - 1, max(0, int(math.floor(start_y + dy * fraction))))
            column = int(math.floor(start_x + dx * fraction)) % width
            cell = (row, column)
            if not cells or cells[-1] != cell:
                cells.append(cell)
    return tuple(cells)


def _river_runs(
    cells: tuple[tuple[int, int], ...],
    river_order: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, (row, column) in enumerate(cells):
        if int(river_order[row, column]) > 0:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(cells) - 1))
    return tuple(runs)


def _wrapped_vector(
    first: tuple[int, int],
    second: tuple[int, int],
    width: int,
) -> tuple[float, float]:
    dy = float(second[0] - first[0])
    dx = float(second[1] - first[1])
    if dx > width * 0.5:
        dx -= width
    elif dx < -width * 0.5:
        dx += width
    return dy, dx


def _river_tangent(
    grid: WorldGrid,
    row: int,
    column: int,
) -> tuple[float, float] | None:
    downstream = int(grid.flow_to[row, column])
    if 0 <= downstream < grid.river_order.size:
        next_row, next_column = divmod(downstream, grid.shape[1])
        if int(grid.river_order[next_row, next_column]) > 0:
            vector = _wrapped_vector((row, column), (next_row, next_column), grid.shape[1])
            if vector != (0.0, 0.0):
                return vector

    points: list[tuple[float, float]] = []
    for dy in range(-2, 3):
        next_row = row + dy
        if next_row < 0 or next_row >= grid.shape[0]:
            continue
        for dx in range(-2, 3):
            next_column = (column + dx) % grid.shape[1]
            if (dy or dx) and int(grid.river_order[next_row, next_column]) > 0:
                points.append((float(dy), float(dx)))
    if not points:
        return None
    coordinates = np.asarray(points, dtype=np.float64)
    covariance = coordinates.T @ coordinates
    values, vectors = np.linalg.eigh(covariance)
    tangent = vectors[:, int(np.argmax(values))]
    return float(tangent[0]), float(tangent[1])


def _classify_transverse_crossing(
    grid: WorldGrid,
    cells: tuple[tuple[int, int], ...],
    run: tuple[int, int],
) -> tuple[int, int, int] | None:
    start, end = run
    if start == 0 or end >= len(cells) - 1:
        return None
    first_bank = cells[start - 1]
    second_bank = cells[end + 1]
    if (
        int(grid.water[first_bank]) != 0
        or int(grid.water[second_bank]) != 0
        or int(grid.river_order[first_bank]) > 0
        or int(grid.river_order[second_bank]) > 0
    ):
        return None
    road_vector = _wrapped_vector(first_bank, second_bank, grid.shape[1])
    road_length = math.hypot(*road_vector)
    if road_length <= 0.0:
        return None
    run_cells = cells[start : end + 1]
    maximum_order = max(int(grid.river_order[cell]) for cell in run_cells)
    ordered = [cell for cell in run_cells if int(grid.river_order[cell]) == maximum_order]
    row, column = ordered[len(ordered) // 2]
    river_vector = _river_tangent(grid, row, column)
    if river_vector is None:
        return None
    river_length = math.hypot(*river_vector)
    alignment = abs(
        (road_vector[0] * river_vector[0] + road_vector[1] * river_vector[1])
        / (road_length * river_length)
    )
    if alignment > 0.65:
        return None
    return row, column, maximum_order


def _road_river_constraints(grid, importance, *, raw_elevation_m):
    """Native flow edges are crossing barriers beyond this road's capacity.

    River support cells block crossing through a channel vertex. Opposing D8
    diagonals can cross between four cell centres without visiting a river
    cell; those exact flow intersections are also forbidden routing edges.
    """
    maximum_order = {"local":2,"regional":3,"trunk":255}[importance]
    from ..river_network import hydrologic_outlet_targets
    outlets=hydrologic_outlet_targets(grid,raw_elevation_m)
    major = np.asarray(grid.river_order) > maximum_order
    valid = ~major
    blocked = np.zeros(grid.shape,dtype=np.uint8)
    if not np.any(major):
        return valid,blocked,shapely.GeometryCollection()
    # Selecting segments, rather than whole merged reaches, avoids blocking
    # an upstream small stream merely because its downstream reach is large.
    source = np.flatnonzero(major.ravel())
    target = np.asarray(grid.flow_to).ravel()[source].copy()
    for index,cell in enumerate(source):
        if int(cell) in outlets:
            target[index]=outlets[int(cell)]
    present = (target>=0)&(target<major.size)
    source,target=source[present],target[present]
    width = grid.shape[1]
    first=np.column_stack((source%width+.5,source//width+.5))
    last=np.column_stack((target%width+.5,target//width+.5))
    ordinary=np.abs(first[:,0]-last[:,0])<=width/2
    barrier=shapely.multilinestrings(shapely.linestrings(np.stack((first[ordinary],last[ordinary]),axis=1)))
    # There are only four diagonal lattice edges adjacent to each channel
    # cell. Vectorized GEOS predicates keep this sparse scan independent of
    # the number of requested city routes.
    rows,columns=np.nonzero(major)
    candidates=set()
    for row,column in zip(rows,columns,strict=True):
        candidates.update((int(y),int(x)%width)for y in range(max(0,row-1),min(grid.shape[0],row+2))for x in range(column-1,column+2))
    cells=np.asarray(sorted(candidates),dtype=np.int32)
    barrier_index=shapely.STRtree(shapely.get_parts(barrier))
    for direction,(dy,dx,_distance) in enumerate(_NEIGHBORS):
        if not (dy and dx):
            continue
        next_rows=cells[:,0]+dy
        next_columns=(cells[:,1]+dx)%width
        present=(next_rows>=0)&(next_rows<grid.shape[0])
        starts=cells[present];ends=np.column_stack((next_rows[present],next_columns[present]))
        present=valid[starts[:,0],starts[:,1]]&valid[ends[:,0],ends[:,1]]
        starts,ends=starts[present],ends[present]
        ordinary=np.abs(starts[:,1]-ends[:,1])<=width/2
        starts,ends=starts[ordinary],ends[ordinary]
        lines=shapely.linestrings(np.stack((starts[:,::-1]+.5,ends[:,::-1]+.5),axis=1))
        crossed=np.unique(barrier_index.query(lines,predicate="crosses")[0])
        blocked[starts[crossed,0],starts[crossed,1]] |= np.uint8(1<<direction)
    return valid,blocked,barrier


def _route_can_bridge(importance: str, river_order: int) -> bool:
    maximum_order = {"local": 2, "regional": 3, "trunk": 255}[importance]
    return river_order <= maximum_order


def derive_bridges(
    grid: WorldGrid,
    routes: tuple[TransportRoute, ...],
    *, raw_elevation_m: np.ndarray,
) -> tuple[Bridge, ...]:
    """Resolve each actual road/flow crossing without erasing nearby facilities."""

    from ..transport_geometry import native_river_geometry, source_transport_crossings

    candidates = []
    for route, point, (row, column), order in source_transport_crossings(
            grid, routes, native_river_geometry(grid,raw_elevation_m=raw_elevation_m),raw_elevation_m=raw_elevation_m):
        if _route_can_bridge(route.importance, order):
            candidates.append((row, column, route.identifier, order, route.importance,
                               (round(point.x, 7), round(point.y, 7))))

    rank = {"local": 0, "regional": 1, "trunk": 2}
    selected: list[tuple[int, int, str, int, str]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-item[3], -rank[item[4]], item[2], item[0], item[1]),
    ):
        if any(candidate[5] == existing[5] for existing in selected):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(
        Bridge(
            identifier=f"bridge-{index:04d}",
            row=row,
            column=column,
            route_identifier=route_identifier,
            river_order=order,
            importance=importance,
        )
        for index, (row, column, route_identifier, order, importance, _point) in enumerate(
            selected,
            start=1,
        )
    )


def road_network_fields(
    shape: tuple[int, int],
    routes: tuple[TransportRoute, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize overland corridors and distinguish true route junctions.

    Accessibility mixes roads, rivers, sea lanes and existing cities.  Urban
    feedback needs a narrower signal: a road can support a relay or market,
    while overlapping important roads identify a commercial hub.  The small
    shoulder around each line lets a settlement sit beside rather than on top
    of the rendered road stroke.
    """

    corridor = np.zeros(shape, dtype=np.float64)
    load = np.zeros(shape, dtype=np.float64)
    route_value = {"trunk": 0.86, "regional": 0.68, "local": 0.46}
    for route in routes:
        if route.mode not in {"road", "rail"} or len(route.path) < 2:
            continue
        value = route_value[route.importance]
        cells = set(_path_cells(route.path, shape))
        if not cells:
            continue
        rows, columns = zip(*cells, strict=True)
        corridor[rows, columns] = np.maximum(corridor[rows, columns], value)
        load[rows, columns] += value

    junction = np.clip((load - 0.82) / 1.36, 0.0, 1.0)
    for _distance in range(2):
        padded_corridor = np.pad(corridor, ((1, 1), (0, 0)), mode="constant")
        padded_junction = np.pad(junction, ((1, 1), (0, 0)), mode="constant")
        corridor_neighbor = np.maximum.reduce(
            (
                np.roll(corridor, 1, axis=1),
                np.roll(corridor, -1, axis=1),
                padded_corridor[:-2],
                padded_corridor[2:],
            )
        )
        junction_neighbor = np.maximum.reduce(
            (
                np.roll(junction, 1, axis=1),
                np.roll(junction, -1, axis=1),
                padded_junction[:-2],
                padded_junction[2:],
            )
        )
        corridor = np.maximum(corridor, corridor_neighbor * 0.86)
        junction = np.maximum(junction, junction_neighbor * 0.82)
    return corridor.astype(np.float32), junction.astype(np.float32)


def conform_transport_to_politics(
    grid: WorldGrid,
    thematic: ThematicLayers,
    settlements: tuple[Settlement, ...],
    transport: TransportLayers,
    politics: PoliticalLayers,
    *, raw_elevation_m: np.ndarray,
) -> TransportLayers:
    """Keep overland transport inside its legally usable political corridor.

    Roads retain their historical same-state correction.  Railways have a
    stricter rule: a domestic line stays inside its state, while an
    international line may use only the two endpoint states.  A railway that
    cannot satisfy that rule is omitted instead of being left as a visible
    illegal shortcut through a third country.
    """

    by_identifier = {item.identifier: item for item in settlements}
    offending_roads: list[tuple[int, int]] = []
    offending_rails: list[tuple[int, frozenset[int]]] = []
    for index, route in enumerate(transport.routes):
        if route.target_settlement_id is None:
            continue
        source = by_identifier[route.source_settlement_id]
        target = by_identifier[route.target_settlement_id]
        source_state = int(politics.state_id[source.row, source.column])
        target_state = int(politics.state_id[target.row, target.column])
        route_cells = _path_cells(route.path, grid.shape)
        if route.mode == "road":
            if source_state <= 0 or source_state != target_state:
                continue
            if any(
                int(politics.state_id[row, column]) != source_state
                for row, column in route_cells
            ):
                offending_roads.append((index, source_state))
            continue
        if route.mode != "rail":
            continue
        permitted_states = frozenset((source_state, target_state))
        if source_state <= 0 or target_state <= 0 or any(
            int(politics.state_id[row, column]) not in permitted_states
            for row, column in route_cells
        ):
            offending_rails.append((index, permitted_states))
    if not offending_roads and not offending_rails:
        return transport

    routes: list[TransportRoute | None] = list(transport.routes)
    if offending_roads:
        elevation = grid.elevation.astype(np.float64)
        slope = relative_land_slope(elevation, grid.water == 0)
        broad = smooth_field(
            elevation,
            radius=max(4, min(11, min(grid.shape) // 90)),
            passes=2,
        )
        river_corridor = smooth_field(
            (grid.river_order > 0).astype(np.float64),
            radius=max(2, min(7, min(grid.shape) // 150)),
            passes=1,
        )
        road_friction = np.clip(
            1.0
            + 7.5 * np.clip(elevation - 0.32, 0.0, 0.68)
            + 24.0 * np.square(np.clip(elevation - 0.48, 0.0, 0.52))
            + 76.0 * slope
            + 18.0 * np.clip(elevation - broad, 0.0, 1.0)
            + 1.6 * (1.0 - np.clip(thematic.land_potential, 0.0, 1.0))
            - 0.48 * np.clip(river_corridor, 0.0, 1.0)
            - 0.42 * np.clip(grid.river_order.astype(np.float64) / 3.0, 0.0, 1.0),
            0.72,
            None,
        )
        road_constraints = {importance:_road_river_constraints(grid,importance,raw_elevation_m=raw_elevation_m)
                            for importance in ("local","regional","trunk")}
        engineering = RoadEngineering(raw_elevation_m, float(grid.metadata['planet']['radiusKm']), _technology_era(grid))
        road_edges = engineering.edge_costs(road_friction)
        for index, state_identifier in offending_roads:
            route = routes[index]
            if route is None:
                continue
            source = by_identifier[route.source_settlement_id]
            target = by_identifier[route.target_settlement_id or ""]
            river_valid,blocked_edges,river_barrier = road_constraints[route.importance]
            valid = (politics.state_id == state_identifier)&river_valid
            valid[source.row,source.column]=True
            valid[target.row,target.column]=True
            path = _least_cost_path(
                road_friction,
                valid,
                (source.row, source.column),
                (target.row, target.column),
                blocked_edges=blocked_edges,
                edge_costs=road_edges,
            )
            if not path:
                continue
            path = _terrain_safe_simplification(path, road_friction, valid,river_barrier=river_barrier,edge_costs=road_edges)
            world_path = _world_path(
                path,
                1,
                grid.shape,
                source,
                target,
                simplify=False,
            )
            routes[index] = replace(route, path=world_path)

    if offending_rails:
        era = _technology_era(grid)
        rail_friction, rail_ground = _rail_engineering_fields(grid, thematic)
        rail_ground = rail_ground.copy()
        for settlement in settlements:
            if int(grid.river_order[settlement.row, settlement.column]) > 0:
                rail_ground[settlement.row, settlement.column] = False
        step = max(2, min(8, min(grid.shape) // 160))
        coarse_friction = reduce_field(rail_friction, step=step, mode="mean")
        rail_cache = RoutingCache()
        for index, permitted_states in offending_rails:
            route = routes[index]
            if route is None:
                continue
            source = by_identifier[route.source_settlement_id]
            target = by_identifier[route.target_settlement_id or ""]
            if era not in _RAIL_ERAS or min(permitted_states) <= 0:
                routes[index] = None
                continue
            political_ground = rail_ground & np.isin(
                politics.state_id,
                tuple(sorted(permitted_states)),
            )
            start = (source.row, source.column)
            goal = (target.row, target.column)
            if not political_ground[start] or not political_ground[goal]:
                routes[index] = None
                continue
            coarse_ground = reduce_field(political_ground, step=step, mode="mean") >= 0.62
            coarse_ground[start[0] // step, start[1] // step] = True
            coarse_ground[goal[0] // step, goal[1] // step] = True
            source_cell = _nearest_valid(
                coarse_ground,
                (start[0] // step, start[1] // step),
                radius=2,
            )
            target_cell = _nearest_valid(
                coarse_ground,
                (goal[0] // step, goal[1] // step),
                radius=2,
            )
            if source_cell is None or target_cell is None:
                routes[index] = None
                continue
            signature = rail_cache.signature(
                coarse_friction,
                coarse_ground,
                rail_friction,
                political_ground,
            )
            coarse_path = rail_cache.route(
                signature,
                coarse_friction,
                coarse_ground,
                source_cell,
                target_cell,
                coarse_friction,
                simplify=False,
            )
            if not coarse_path:
                routes[index] = None
                continue
            native_path = _refine_rail_path(
                coarse_path,
                step=step,
                friction=rail_friction,
                valid=political_ground,
                source=source,
                target=target,
            )
            if not _rail_path_is_valid(
                grid,
                native_path,
                era=era,
                importance=route.importance,
            ):
                routes[index] = None
                continue
            routes[index] = replace(
                route,
                path=_world_path(
                    native_path,
                    1,
                    grid.shape,
                    source,
                    target,
                    simplify=False,
                ),
            )

    finalized = tuple(route for route in routes if route is not None)
    return TransportLayers(
        accessibility=_accessibility_field(grid.shape, settlements, finalized),
        routes=finalized,
        bridges=derive_bridges(grid, finalized,raw_elevation_m=raw_elevation_m),
    )


__all__ = [
    "conform_transport_to_politics",
    "derive_bridges",
    "derive_transport",
    "road_network_fields",
]
