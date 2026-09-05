"""Post-political sites that make borders and sparse frontiers legible."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from ..model import WorldGrid
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
)
from .spatial import reduce_field, select_spaced_seeds


def _gradient_magnitude(values: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    if any(size < 2 for size in field.shape):
        return np.zeros(field.shape, dtype=np.float64)
    vertical, horizontal = np.gradient(field)
    return np.hypot(vertical, horizontal)


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


def _border_candidates(
    grid: WorldGrid,
    thematic: ThematicLayers,
    cultures: CultureLayers,
    politics: PoliticalLayers,
    transport: TransportLayers,
) -> tuple[tuple[float, int, int, str], ...]:
    """Return actual road crossings, including short unruled border gaps."""

    elevation = grid.elevation.astype(np.float64)
    slope = _gradient_magnitude(elevation)
    route_rank = {"trunk": 0.52, "regional": 0.34, "local": 0.18}
    candidates: dict[tuple[int, int], tuple[float, str]] = {}
    for route in transport.routes:
        if route.mode != "road":
            continue
        cells = _route_cells(route.path, grid.shape)
        if len(cells) < 2:
            continue
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
            if grid.water[row, column] != 0 or grid.snow[row, column]:
                continue
            relief = float(np.clip((elevation[row, column] - 0.30) / 0.38, 0.0, 1.0))
            constriction = float(np.clip(slope[row, column] / 0.055, 0.0, 1.0))
            score = (
                1.20
                + route_rank[route.importance]
                + boundary_weight
                + 0.30 * relief
                + 0.26 * constriction
                + 0.16 * float(transport.accessibility[row, column])
                + 0.10 * float(thematic.land_potential[row, column])
            )
            site_type = "pass" if relief >= 0.34 or constriction >= 0.40 else "fortress"
            previous = candidates.get((row, column))
            if previous is None or score > previous[0]:
                candidates[(row, column)] = (score, site_type)
    return tuple(
        sorted(
            ((score, row, column, site_type) for (row, column), (score, site_type) in candidates.items()),
            key=lambda item: (-item[0], item[1], item[2]),
        )
    )


def _population_interval(
    population: PopulationLayers,
    row: int,
    column: int,
    *,
    frontier: bool,
) -> tuple[int, int]:
    band = int(population.population_band[row, column])
    if frontier:
        low = 300 + band * 280
        return low, low + 900 + band * 520
    low = 1_200 + band * 550
    return low, low + 2_200 + band * 900


def _promote_nearby_border_settlements(
    settlements: tuple[Settlement, ...],
    candidates: tuple[tuple[float, int, int, str], ...],
    *,
    width: int,
    maximum_distance: float = 14.0,
) -> tuple[Settlement, ...]:
    """Give an existing road-crossing town a border role before adding a site."""

    excluded = {"port", "island-port", "lake-port", "oasis"}
    promotion: dict[str, tuple[float, str]] = {}
    urban = [
        item
        for item in settlements
        if item.tier != "site" and item.site_type not in excluded
    ]
    for score, row, column, site_type in candidates:
        nearby = [
            item
            for item in urban
            if _wrapped_distance(
                (row, column),
                (item.row, item.column),
                width,
            )
            <= maximum_distance
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
    """Fortify towns that already occupy an actual international road crossing."""

    candidates = _border_candidates(
        grid,
        thematic,
        cultures,
        politics,
        transport,
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

    for score, row, column, site_type in _border_candidates(
        grid,
        thematic,
        cultures,
        politics,
        transport,
    ):
        if len(selected) >= border_count:
            break
        location = (row, column)
        if any(_wrapped_distance(location, other, width) < 18.0 for other in (*occupied, *((item.row, item.column) for item in selected))):
            continue
        population_min, population_max = _population_interval(
            population, row, column, frontier=False
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

    land = grid.water == 0
    slope = _gradient_magnitude(grid.elevation)
    coast = land & _adjacent(np.isin(grid.water, (1, 3)), radius=1)
    lake = land & _adjacent(grid.water == 2, radius=2)
    river = grid.river_order >= 2
    freshwater = _adjacent(river | (grid.water == 2), radius=4) & land
    population_relative = population.population_weight.astype(np.float64)
    maximum = float(population_relative.max(initial=0.0))
    if maximum > 0.0:
        population_relative /= maximum
    frontier_valid = (
        politics.frontier
        & land
        & ~grid.snow
        & (grid.elevation < 0.73)
        & (slope < 0.115)
        & (population.population_band >= 1)
        & ((thematic.land_potential >= 0.10) | freshwater | coast)
    )
    frontier_score = np.where(
        frontier_valid,
        0.34 * population_relative
        + 0.22 * thematic.land_potential.astype(np.float64)
        + 0.16 * freshwater
        + 0.13 * coast
        + 0.10 * np.clip(transport.accessibility.astype(np.float64), 0.0, 1.0)
        + 0.08 * (population.population_band.astype(np.float64) / 6.0)
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
        elif river[row, column]:
            site_type = "river-city"
        elif biome in {BIOME_TEMPERATE_GRASSLAND, BIOME_SAVANNA_DRY_GRASSLAND}:
            site_type = "market"
        elif thematic.climate.annual_precipitation[row, column] < 0.055 and freshwater[row, column]:
            site_type = "oasis"
        else:
            site_type = "market"
        population_min, population_max = _population_interval(
            population, row, column, frontier=True
        )
        tier = (
            "town"
            if site_type == "port"
            and thematic.land_potential[row, column] >= 0.20
            and population.population_band[row, column] >= 2
            else "site"
        )
        if tier == "town":
            population_min = max(population_min, 2_500)
            population_max = max(population_max, 8_000)
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
