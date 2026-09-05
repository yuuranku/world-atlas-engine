"""Deterministic roads, navigable rivers, sea lanes, and accessibility."""

from __future__ import annotations

import heapq
import math
from dataclasses import replace

import numpy as np

from ..model import WorldGrid
from ..thematic import ThematicLayers, smooth_field
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
    costs = {start: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue: list[tuple[float, float, int, int]] = [
        (
            _wrapped_distance(start, goal, width) * heuristic_scale,
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
        for dy, dx, distance in _NEIGHBORS:
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
            edge = 0.5 * (
                float(friction[row, column]) + float(friction[next_row, next_column])
            ) * distance
            next_cost = cost + edge
            point = (next_row, next_column)
            if next_cost >= costs.get(point, math.inf) - 1.0e-12:
                continue
            costs[point] = next_cost
            previous[point] = current
            heuristic = _wrapped_distance(point, goal, width) * heuristic_scale
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
    maximum_span: int = 18,
) -> tuple[tuple[int, int], ...]:
    """Remove flat-land lattice noise without cutting across real relief."""

    if len(path) <= 2:
        return path
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
            segment = line_cells(path[anchor], path[candidate])
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
    current = field
    for _distance in range(radius):
        padded = np.pad(current, ((1, 1), (0, 0)), mode="constant")
        neighbor = np.maximum.reduce(
            (
                np.roll(current, 1, axis=1),
                np.roll(current, -1, axis=1),
                padded[:-2],
                padded[2:],
            )
        )
        current = np.maximum(current, neighbor * 0.88)
    return np.clip(current, 0.0, 1.0).astype(np.float32)


def derive_transport(
    grid: WorldGrid,
    thematic: ThematicLayers,
    population: PopulationLayers,
    settlements: tuple[Settlement, ...],
) -> TransportLayers:
    """Build roads and water routes before any state border is generated."""

    if thematic.land_potential.shape != grid.shape or population.population_weight.shape != grid.shape:
        raise ValueError("transport inputs must match the WorldGrid shape")
    if not settlements:
        return TransportLayers(
            accessibility=np.zeros(grid.shape, dtype=np.float32),
            routes=(),
        )
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
    vertical = np.gradient(elevation, axis=0)
    horizontal = np.gradient(elevation, axis=1)
    slope = np.hypot(vertical, horizontal)
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
    full_vertical = np.gradient(grid.elevation.astype(np.float64), axis=0)
    full_horizontal = (
        np.roll(grid.elevation.astype(np.float64), -1, axis=1)
        - np.roll(grid.elevation.astype(np.float64), 1, axis=1)
    ) * 0.5
    full_slope = np.hypot(full_vertical, full_horizontal)
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
    routes: list[TransportRoute] = []
    for source, target in _road_pairs(settlements, cells, components):
        path = _least_cost_path(friction, coarse_land, cells[source.identifier], cells[target.identifier])
        if not path:
            continue
        if step == 1:
            exact_path = _terrain_safe_simplification(
                path,
                full_friction,
                road_land,
            )
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
                importance=_route_importance(source, target),
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
        for first, second in sorted(pairs):
            source = by_identifier[first]
            target = by_identifier[second]
            path = _least_cost_path(
                water_friction,
                coarse_water,
                water_cell[first],
                water_cell[second],
            )
            if not path:
                continue
            path = _terrain_safe_simplification(
                path,
                water_friction,
                coarse_water,
            )
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
    finalized = tuple(routes)
    return TransportLayers(
        accessibility=_accessibility_field(grid.shape, settlements, finalized),
        routes=finalized,
        bridges=derive_bridges(grid, finalized),
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


def _route_can_bridge(importance: str, river_order: int) -> bool:
    maximum_order = {"local": 2, "regional": 3, "trunk": 255}[importance]
    return river_order <= maximum_order


def derive_bridges(
    grid: WorldGrid,
    routes: tuple[TransportRoute, ...],
) -> tuple[Bridge, ...]:
    """Resolve engineered crossings from transverse road/river intersections."""

    candidates: list[tuple[int, int, str, int, str]] = []
    for route in routes:
        if route.mode != "road":
            continue
        cells = _path_cells(route.path, grid.shape)
        for run in _river_runs(cells, grid.river_order):
            crossing = _classify_transverse_crossing(grid, cells, run)
            if crossing is None or not _route_can_bridge(route.importance, crossing[2]):
                continue
            row, column, order = crossing
            candidates.append((row, column, route.identifier, order, route.importance))

    rank = {"local": 0, "regional": 1, "trunk": 2}
    selected: list[tuple[int, int, str, int, str]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-item[3], -rank[item[4]], item[2], item[0], item[1]),
    ):
        row, column = candidate[:2]
        if any(
            _wrapped_distance((row, column), existing[:2], grid.shape[1]) < 5.0
            for existing in selected
        ):
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
        for index, (row, column, route_identifier, order, importance) in enumerate(
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
        if route.mode != "road" or len(route.path) < 2:
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
) -> TransportLayers:
    """Reroute same-state roads that would otherwise enter another state."""

    by_identifier = {item.identifier: item for item in settlements}
    offending: list[tuple[int, int]] = []
    for index, route in enumerate(transport.routes):
        if route.mode != "road" or route.target_settlement_id is None:
            continue
        source = by_identifier[route.source_settlement_id]
        target = by_identifier[route.target_settlement_id]
        source_state = int(politics.state_id[source.row, source.column])
        target_state = int(politics.state_id[target.row, target.column])
        if source_state <= 0 or source_state != target_state:
            continue
        if any(
            int(politics.state_id[row, column]) != source_state
            for row, column in _path_cells(route.path, grid.shape)
        ):
            offending.append((index, source_state))
    if not offending:
        return transport

    elevation = grid.elevation.astype(np.float64)
    vertical = np.gradient(elevation, axis=0)
    horizontal = (np.roll(elevation, -1, axis=1) - np.roll(elevation, 1, axis=1)) * 0.5
    slope = np.hypot(vertical, horizontal)
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
    friction = np.clip(
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
    routes = list(transport.routes)
    for index, state_identifier in offending:
        route = routes[index]
        source = by_identifier[route.source_settlement_id]
        target = by_identifier[route.target_settlement_id or ""]
        valid = politics.state_id == state_identifier
        path = _least_cost_path(
            friction,
            valid,
            (source.row, source.column),
            (target.row, target.column),
        )
        if not path:
            continue
        path = _terrain_safe_simplification(path, friction, valid)
        world_path = _world_path(
            path,
            1,
            grid.shape,
            source,
            target,
            simplify=False,
        )
        routes[index] = replace(route, path=world_path)
    finalized = tuple(routes)
    return TransportLayers(
        accessibility=_accessibility_field(grid.shape, settlements, finalized),
        routes=finalized,
        bridges=derive_bridges(grid, finalized),
    )


__all__ = [
    "conform_transport_to_politics",
    "derive_bridges",
    "derive_transport",
    "road_network_fields",
]
