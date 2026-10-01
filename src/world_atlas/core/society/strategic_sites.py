"""Post-political sites that make borders and sparse frontiers legible."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from ..model import WorldGrid
from ..suitability import relative_land_slope
from ..thematic import (
    BIOME_SAVANNA_DRY_GRASSLAND,
    BIOME_TEMPERATE_GRASSLAND,
    ThematicLayers,
)
from .model import (
    CultureLayers,
    PoliticalLayers,
    PopulationLayers,
    Settlement,
    TransportLayers,
    TransportRoute,
)
from .spatial import reduce_field, select_spaced_seeds
from .population import population_density


def _adjacent(mask: np.ndarray, *, radius: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    frontier = result.copy()
    for _distance in range(max(0, int(radius))):
        padded = np.pad(frontier, ((1, 1), (0, 0)), mode="constant")
        frontier = (
            frontier
            | np.roll(frontier, 1, axis=1)
            | np.roll(frontier, -1, axis=1)
            | padded[:-2]
            | padded[2:]
        )
        result |= frontier
    return result


def _route_cells(
    path: tuple[tuple[float, float], ...],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    height, width = shape
    cells: list[tuple[int, int]] = []
    for (start_column, start_row), (end_column, end_row) in zip(path, path[1:]):
        dx = end_column - start_column
        if dx > width * 0.5:
            dx -= width
        elif dx < -width * 0.5:
            dx += width
        dy = end_row - start_row
        steps = max(1, int(math.ceil(max(abs(dx), abs(dy)))))
        for offset in range(steps + 1):
            fraction = offset / steps
            row = int(round(start_row + dy * fraction - 0.5))
            column = int(round(start_column + dx * fraction - 0.5)) % width
            cell = (min(height - 1, max(0, row)), column)
            if not cells or cells[-1] != cell:
                cells.append(cell)
    return tuple(cells)


def _wrapped_distance(
    first: tuple[int, int],
    second: tuple[int, int],
    width: int,
) -> float:
    dx = abs(first[1] - second[1])
    return math.hypot(first[0] - second[0], min(dx, width - dx))


def _strategic_road_demand(
    route: TransportRoute,
    settlements: dict[str, Settlement],
    politics: PoliticalLayers,
) -> float:
    """A minor road or a road created solely for a site cannot justify a gate."""

    if route.mode != "road" or route.importance == "local":
        return 0.0
    source = settlements.get(route.source_settlement_id)
    target = settlements.get(route.target_settlement_id)
    if source is None or target is None or source.tier == "site" or target.tier == "site":
        return 0.0
    if route.importance == "trunk":
        return 0.80
    major = {"metropolis", "city"}
    if source.tier in major and target.tier in major:
        return 0.52
    source_state = int(politics.state_id[source.row, source.column])
    target_state = int(politics.state_id[target.row, target.column])
    if (
        source_state > 0
        and target_state > 0
        and source_state != target_state
        and (source.tier in major or target.tier in major)
    ):
        return 0.42
    return 0.0


def _road_bottleneck_scores(
    grid: WorldGrid,
    cells: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Find a traversable saddle or local pinch, with raised terrain on both sides.

    The cross section is perpendicular to the actual road. A steep hillside,
    broad plateau, coastline, or long uniform valley is insufficient; the
    approaches must open out or descend on both sides of the choke point.
    """

    scores = np.zeros(len(cells), dtype=np.float64)
    radius = max(3, min(12, int(round(grid.shape[1] / 200.0))))
    approach = radius * 2
    if len(cells) <= approach * 2:
        return scores
    locations = np.asarray(cells, dtype=np.int32)
    indices = np.arange(len(cells))
    before = np.maximum(0, indices - approach)
    after = np.minimum(len(cells) - 1, indices + approach)
    tangent_span = max(2, radius // 2)
    tangent_start = locations[np.maximum(0, indices - tangent_span)]
    tangent_end = locations[np.minimum(len(cells) - 1, indices + tangent_span)]
    dy = (tangent_end[:, 0] - tangent_start[:, 0]).astype(np.float64)
    width = grid.shape[1]
    dx = (tangent_end[:, 1] - tangent_start[:, 1] + width * 0.5) % width - width * 0.5
    length = np.hypot(dx, dy)
    normal_row = np.divide(dx, length, out=np.zeros_like(dx), where=length > 0)
    normal_column = np.divide(-dy, length, out=np.zeros_like(dy), where=length > 0)
    offsets = np.arange(1, radius + 1, dtype=np.float64)
    elevation = grid.elevation[locations[:, 0], locations[:, 1]].astype(np.float64)
    flank_relief = []
    flank_valid = []
    for side in (-1.0, 1.0):
        rows = np.rint(locations[:, 0, None] + side * normal_row[:, None] * offsets).astype(np.int32)
        columns = np.rint(locations[:, 1, None] + side * normal_column[:, None] * offsets).astype(np.int32) % width
        inside = (rows >= 0) & (rows < grid.shape[0])
        rows = np.clip(rows, 0, grid.shape[0] - 1)
        side_elevation = grid.elevation[rows, columns].astype(np.float64)
        land = inside & (grid.water[rows, columns] == 0)
        raised = land & (side_elevation >= elevation[:, None] + 0.05)
        side_peak = np.max(np.where(land, side_elevation, elevation[:, None]), axis=1)
        flank_relief.append(side_peak - elevation)
        flank_valid.append((raised.sum(axis=1) >= 2) & (side_peak >= 0.35))
    relief = np.minimum(flank_relief[0], flank_relief[1])
    saddle = elevation - np.maximum(elevation[before], elevation[after])
    pinch = relief - np.maximum(relief[before], relief[after])
    traversable = (
        (grid.water[locations[:, 0], locations[:, 1]] == 0)
        & ~grid.snow[locations[:, 0], locations[:, 1]]
        & (grid.river_order[locations[:, 0], locations[:, 1]] == 0)
        & (elevation < 0.72)
        & (elevation[before] <= elevation + 0.05)
        & (elevation[after] <= elevation + 0.05)
    )
    valid = (
        traversable
        & flank_valid[0]
        & flank_valid[1]
        & ((saddle >= 0.025) | (pinch >= 0.025))
        & (indices >= approach)
        & (indices < len(cells) - approach)
    )
    scores[valid] = (
        np.clip(relief[valid] / 0.15, 0.0, 1.0)
        + np.clip(np.maximum(saddle[valid], pinch[valid]) / 0.12, 0.0, 1.0)
    )
    return scores


def _border_candidates(
    grid: WorldGrid,
    thematic: ThematicLayers,
    cultures: CultureLayers,
    politics: PoliticalLayers,
    transport: TransportLayers,
    settlements: tuple[Settlement, ...],
) -> tuple[tuple[float, int, int, str], ...]:
    """Return important border crossings and real choke points on major roads."""

    elevation = grid.elevation.astype(np.float64)
    slope = relative_land_slope(elevation, grid.water == 0)
    by_identifier = {item.identifier: item for item in settlements}
    candidates: dict[tuple[int, int], tuple[float, str]] = {}
    for route in transport.routes:
        demand = _strategic_road_demand(route, by_identifier, politics)
        if demand <= 0.0:
            continue
        cells = _route_cells(route.path, grid.shape)
        if len(cells) < 2:
            continue
        bottleneck = _road_bottleneck_scores(grid, cells)
        states = [int(politics.state_id[row, column]) for row, column in cells]
        civilizations = [int(cultures.civilization_id[row, column]) for row, column in cells]
        crossings: list[tuple[int, float]] = []
        previous_positive: tuple[int, int] | None = None
        for index, state in enumerate(states):
            if state <= 0:
                continue
            if previous_positive is not None:
                previous_index, previous_state = previous_positive
                if state != previous_state and index - previous_index <= 72:
                    crossings.append(((previous_index + index) // 2, 1.0))
            previous_positive = (index, state)
        for index in range(1, len(cells)):
            first = civilizations[index - 1]
            second = civilizations[index]
            if first > 0 and second > 0 and first != second:
                crossings.append((index, 0.72))
        for index, boundary_weight in crossings:
            row, column = cells[min(len(cells) - 1, max(0, index))]
            if (
                grid.water[row, column] != 0
                or grid.snow[row, column]
                or grid.river_order[row, column] > 0
            ):
                continue
            relief = float(np.clip((elevation[row, column] - 0.30) / 0.38, 0.0, 1.0))
            constriction = float(np.clip(slope[row, column] / 0.055, 0.0, 1.0))
            score = (
                1.20
                + demand
                + boundary_weight
                + 0.30 * relief
                + 0.26 * constriction
                + 0.16 * float(transport.accessibility[row, column])
                + 0.10 * float(thematic.land_potential[row, column])
            )
            site_type = "pass" if bottleneck[index] > 0.0 else "fortress"
            previous = candidates.get((row, column))
            if previous is None or score > previous[0]:
                candidates[(row, column)] = (score, site_type)
        # A strategically used pass can lie inside one country. Its physical
        # choke point, rather than a nearby border or a mountain quota, earns
        # the role. Keep only the strongest choke point on each road.
        if np.any(bottleneck > 0.0):
            index = int(np.argmax(bottleneck))
            row, column = cells[index]
            score = 2.10 + demand + float(bottleneck[index])
            previous = candidates.get((row, column))
            if previous is None or score > previous[0]:
                candidates[(row, column)] = (score, "pass")
    return tuple(
        sorted(
            ((score, row, column, site_type) for (row, column), (score, site_type) in candidates.items()),
            key=lambda item: (-item[0], item[1], item[2]),
        )
    )


def _population_interval(
    grid: WorldGrid,
    population: PopulationLayers,
    row: int,
    column: int,
    *,
    frontier: bool,
    density: float,
) -> tuple[int, int]:
    support_level = 6.0 * density / (density + 1.0)
    if frontier:
        low = int(300 + support_level * 280)
        high = int(low + 900 + support_level * 520)
        # Density is not a headcount. Bound new frontier
        # sites by the population of a compact, land-connected catchment;
        # nearby mainland across water cannot supply an isolated island.
        reached = {(row, column)}
        boundary = [(row, column)]
        for _distance in range(3):
            following = []
            for current_row, current_column in boundary:
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    target = (current_row + dy, (current_column + dx) % grid.shape[1])
                    if (0 <= target[0] < grid.shape[0] and target not in reached
                            and int(grid.water[target]) == 0):
                        reached.add(target)
                        following.append(target)
            boundary = following
        support = sum(float(population.population_weight[cell]) for cell in reached)
        return min(low, int(support * population.population_min)), min(high, int(support * population.population_max))
    low = int(1_200 + support_level * 550)
    return low, int(low + 2_200 + support_level * 900)


def _promote_nearby_border_settlements(
    settlements: tuple[Settlement, ...],
    candidates: tuple[tuple[float, int, int, str], ...],
    *,
    width: int,
    maximum_distance: float = 6.0,
) -> tuple[Settlement, ...]:
    """Give an existing road-crossing town a border role before adding a site."""

    excluded = {"port", "island-port", "lake-port", "oasis"}
    promotion: dict[str, tuple[float, str]] = {}
    eligible = [
        item
        for item in settlements
        if item.site_type not in excluded
    ]
    for score, row, column, site_type in candidates:
        nearby = [
            item
            for item in eligible
            if _wrapped_distance(
                (row, column),
                (item.row, item.column),
                width,
            )
            <= (min(2.0, maximum_distance) if site_type == "pass" else maximum_distance)
        ]
        if not nearby:
            continue
        chosen = min(
            nearby,
            key=lambda item: (
                _wrapped_distance(
                    (row, column),
                    (item.row, item.column),
                    width,
                ),
                -item.score,
                item.identifier,
            ),
        )
        previous = promotion.get(chosen.identifier)
        if previous is None or (score, site_type == "pass") > (
            previous[0], previous[1] == "pass"
        ):
            promotion[chosen.identifier] = (score, site_type)
    return tuple(
        replace(item, site_type=promotion[item.identifier][1])
        if item.identifier in promotion
        else item
        for item in settlements
    )


def promote_border_settlements(
    grid: WorldGrid,
    thematic: ThematicLayers,
    cultures: CultureLayers,
    politics: PoliticalLayers,
    transport: TransportLayers,
    settlements: tuple[Settlement, ...],
) -> tuple[Settlement, ...]:
    """Resolve strategic roles from the final urban road network and terrain.

    Re-evaluating old strategic roles also permits a saved world to retain all its
    identities, occupied locations and administrative cores while correcting
    mountain-only gates. Physical place names remain stable.
    """

    river_bank = _adjacent(grid.river_order > 0, radius=2)
    settlements = tuple(
        replace(item, site_type="river-city" if river_bank[item.row, item.column] else "market")
        if item.site_type in {"pass", "fortress"}
        else item
        for item in settlements
    )

    candidates = _border_candidates(
        grid,
        thematic,
        cultures,
        politics,
        transport,
        settlements,
    )
    return _promote_nearby_border_settlements(
        settlements,
        candidates,
        width=grid.shape[1],
    )


def derive_strategic_sites(
    grid: WorldGrid,
    thematic: ThematicLayers,
    population: PopulationLayers,
    settlements: tuple[Settlement, ...],
    cultures: CultureLayers,
    transport: TransportLayers,
    politics: PoliticalLayers,
    *,
    border_count: int | None = None,
    frontier_count: int | None = None,
) -> tuple[Settlement, ...]:
    """Add sparse sites at real border crossings and inhabited frontiers."""

    if politics.state_id.shape != grid.shape or cultures.civilization_id.shape != grid.shape:
        raise ValueError("strategic-site layers must match WorldGrid shape")
    if border_count is None:
        border_count = max(14, min(38, len(transport.routes) // 20))
    if frontier_count is None:
        frontier_count = max(20, min(48, len(settlements) // 10))
    occupied = [(item.row, item.column) for item in settlements]
    selected: list[Settlement] = []
    width = grid.shape[1]
    density = population_density(grid, population)

    for score, row, column, site_type in _border_candidates(
        grid,
        thematic,
        cultures,
        politics,
        transport,
        settlements,
    ):
        if len(selected) >= border_count:
            break
        location = (row, column)
        if any(_wrapped_distance(location, other, width) < 18.0 for other in (*occupied, *((item.row, item.column) for item in selected))):
            continue
        population_min, population_max = _population_interval(
            grid, population, row, column, frontier=False,
            density=float(density[row, column]),
        )
        selected.append(
            Settlement(
                identifier=f"border-site-{len(selected) + 1:03d}",
                name=f"边关聚落{len(selected) + 1:03d}",
                row=row,
                column=column,
                tier="site",
                site_type=site_type,
                score=score,
                population_min=population_min,
                population_max=population_max,
            )
        )

    if frontier_count <= 0:
        return tuple(selected)

    land = grid.water == 0
    slope = relative_land_slope(grid.elevation, land)
    coast = land & _adjacent(np.isin(grid.water, (1, 3)), radius=1)
    lake = land & _adjacent(grid.water == 2, radius=2)
    # Frontier settlements can use a mapped tributary as a water source even
    # when it is not a navigable mainstem.  ``channel`` below still keeps the
    # actual river cell unbuildable.
    river = grid.river_order > 0
    channel = grid.river_order > 0
    freshwater = _adjacent(river | (grid.water == 2), radius=4) & land
    population_relative = density.astype(np.float64)
    maximum = float(population_relative.max(initial=0.0))
    if maximum > 0.0:
        population_relative /= maximum
    frontier_valid = (
        politics.frontier
        & land
        & ~channel
        & ~grid.snow
        & (grid.elevation < 0.73)
        & (slope < 0.115)
        & (density > 0.0)
        & ((thematic.land_potential >= 0.10) | freshwater | coast)
    )
    frontier_score = np.where(
        frontier_valid,
        0.42 * population_relative
        + 0.22 * thematic.land_potential.astype(np.float64)
        + 0.16 * freshwater
        + 0.13 * coast
        + 0.10 * np.clip(transport.accessibility.astype(np.float64), 0.0, 1.0)
        - 0.12 * np.clip(slope / 0.10, 0.0, 1.0),
        -np.inf,
    )
    step = 1
    coarse_score = reduce_field(frontier_score, step=step, mode="max")
    coarse_valid = reduce_field(frontier_valid, step=step, mode="max").astype(bool)
    seeds = select_spaced_seeds(
        coarse_score,
        coarse_valid,
        count=max(1, frontier_count * 4),
        minimum_distance=max(3.0, 18.0 / step),
    )
    frontier_added = 0
    for coarse_row, coarse_column in seeds:
        row_start = coarse_row * step
        column_start = coarse_column * step
        block = frontier_score[
            row_start : min(row_start + step, grid.shape[0]),
            column_start : min(column_start + step, grid.shape[1]),
        ]
        flat = int(np.argmax(block))
        local_row, local_column = np.unravel_index(flat, block.shape)
        row = row_start + int(local_row)
        column = column_start + int(local_column)
        location = (row, column)
        all_occupied = (*occupied, *((item.row, item.column) for item in selected))
        if any(_wrapped_distance(location, other, width) < 22.0 for other in all_occupied):
            continue
        biome = int(thematic.biome_zone[row, column])
        if coast[row, column]:
            site_type = "port"
        elif lake[row, column]:
            site_type = "lake-port"
        elif freshwater[row, column]:
            site_type = "river-city"
        elif biome in {BIOME_TEMPERATE_GRASSLAND, BIOME_SAVANNA_DRY_GRASSLAND}:
            site_type = "market"
        elif thematic.climate.annual_precipitation[row, column] < 0.055 and freshwater[row, column]:
            site_type = "oasis"
        else:
            site_type = "market"
        population_min, population_max = _population_interval(
            grid, population, row, column, frontier=True,
            density=float(density[row, column]),
        )
        if population_max <= 0:
            continue
        tier = (
            "town"
            if site_type == "port"
            and thematic.land_potential[row, column] >= 0.20
            and density[row, column] >= 0.5
            and population_max >= 2_500
            else "site"
        )
        frontier_added += 1
        selected.append(
            Settlement(
                identifier=f"frontier-site-{frontier_added:03d}",
                name=f"边地聚落{frontier_added:03d}",
                row=row,
                column=column,
                tier=tier,
                site_type=site_type,
                score=float(frontier_score[row, column]),
                population_min=population_min,
                population_max=population_max,
            )
        )
        if frontier_added >= frontier_count:
            break
    return tuple(selected)


__all__ = ["derive_strategic_sites", "promote_border_settlements"]
