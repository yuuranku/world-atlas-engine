"""One feedback pass linking country extent to the urban hierarchy."""

from __future__ import annotations

import math

import numpy as np

from ..model import WorldGrid
from ..suitability import relative_land_slope
from ..thematic import (
    BIOME_SAVANNA_DRY_GRASSLAND,
    BIOME_TEMPERATE_GRASSLAND,
    ThematicLayers,
)
from .model import PoliticalLayers, PopulationLayers, Settlement, TransportLayers
from .population import _river_bank_location, cell_areas_km2, population_density
from .transport import road_network_fields


def _adjacent(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    result = np.roll(source, 1, axis=1) | np.roll(source, -1, axis=1)
    if source.shape[0] > 1:
        result[1:] |= source[:-1]
        result[:-1] |= source[1:]
    return result


def _wrapped_distances(
    rows: np.ndarray,
    columns: np.ndarray,
    centres: list[tuple[int, int]],
    width: int,
) -> np.ndarray:
    distance = np.full(rows.shape, np.inf, dtype=np.float64)
    for centre_row, centre_column in centres:
        column_delta = np.abs(columns - centre_column)
        distance = np.minimum(
            distance,
            np.hypot(
                rows - centre_row,
                np.minimum(column_delta, width - column_delta),
            ),
        )
    return distance


def derive_administrative_centres(
    grid: WorldGrid,
    thematic: ThematicLayers,
    population: PopulationLayers,
    settlements: tuple[Settlement, ...],
    transport: TransportLayers,
    politics: PoliticalLayers,
    *,
    maximum_additions: int | None = None,
) -> tuple[Settlement, ...]:
    """Add only the urban nodes required by an initial country's service gaps."""

    if politics.state_id.shape != grid.shape:
        raise ValueError("political layers must match WorldGrid")
    urban = [item for item in settlements if item.tier != "site"]
    controlled = politics.state_id > 0
    if not urban or not bool(np.any(controlled)):
        return ()
    if maximum_additions is None:
        maximum_additions = min(72, max(12, len(urban) // 6))
    maximum_additions = max(0, int(maximum_additions))
    if maximum_additions == 0:
        return ()

    controlled_area = int(np.count_nonzero(controlled))
    mean_service_area = controlled_area / len(urban)
    cell_areas = cell_areas_km2(grid)
    physical_service_area = float(np.sum(cell_areas[controlled])) / len(urban)
    density_field = population_density(grid, population)
    controlled_population = float(
        np.sum(population.population_weight[controlled], dtype=np.float64)
    )
    midpoint_population = (population.population_min + population.population_max) / 2.0
    world_density = controlled_population * midpoint_population / float(np.sum(cell_areas[controlled]))
    ocean_coast = (grid.water == 0) & _adjacent(np.isin(grid.water, (1, 3)))
    lake_coast = (grid.water == 0) & _adjacent(grid.water == 2)
    slope = relative_land_slope(grid.elevation, grid.water == 0)
    positive_population = density_field[density_field > 0.0]
    population_scale = (
        float(np.quantile(positive_population, 0.90))
        if positive_population.size
        else 1.0
    )
    road_access, road_junction = road_network_fields(
        grid.shape,
        transport.routes,
    )
    state_settlements: dict[int, list[Settlement]] = {}
    for settlement in urban:
        identifier = int(politics.state_id[settlement.row, settlement.column])
        if identifier > 0:
            state_settlements.setdefault(identifier, []).append(settlement)

    deficits: list[tuple[int, int]] = []
    for state in politics.states:
        region = politics.state_id == state.identifier
        area = int(np.count_nonzero(region))
        local = state_settlements.get(state.identifier, [])
        if area <= 0 or not local:
            continue
        physical_area = float(np.sum(cell_areas[region]))
        density = float(np.sum(population.population_weight[region])) * midpoint_population / physical_area
        density_factor = float(
            np.clip(max(density / max(world_density, 1.0e-12), 1.0e-6) ** 0.34, 0.68, 1.72)
        )
        potential = float(np.mean(thematic.land_potential[region]))
        access = float(np.mean(transport.accessibility[region]))
        steppe = float(
            np.mean(
                np.isin(
                    thematic.biome_zone[region],
                    (BIOME_TEMPERATE_GRASSLAND, BIOME_SAVANNA_DRY_GRASSLAND),
                )
            )
        )
        coast = float(np.mean(ocean_coast[region]))
        sparse_mobility = steppe * (1.0 - potential)
        geographic_capacity = float(
            np.clip(
                1.0 + 2.4 * sparse_mobility + 0.28 * coast,
                1.0,
                3.4,
            )
        )
        route_share = float(np.mean(road_access[region] >= 0.24))
        commercial_support = float(
            np.clip(0.42 * access + 0.58 * route_share / 0.075, 0.0, 1.0)
        )
        desired = int(
            math.ceil(
                physical_area
                / max(physical_service_area, 1.0)
                * density_factor
                * (1.0 + 0.48 * commercial_support)
                / geographic_capacity
            )
        )
        missing = min(3, max(0, desired - len(local)))
        if missing:
            deficits.append((missing, state.identifier))

    additions: list[Settlement] = []
    identifier_offset = sum(
        item.identifier.startswith("administrative-centre-")
        for item in settlements
    )
    service_spacing = max(10.0, math.sqrt(mean_service_area / math.pi) * 0.58)
    for missing, state_identifier in sorted(deficits, key=lambda item: (-item[0], item[1])):
        if len(additions) >= maximum_additions:
            break
        region = politics.state_id == state_identifier
        centres = [
            (item.row, item.column)
            for item in state_settlements.get(state_identifier, [])
        ]
        for _index in range(missing):
            if len(additions) >= maximum_additions:
                break
            valid = (
                region
                & (grid.water == 0)
                & (grid.river_order == 0)
                & ~grid.snow
                & (grid.elevation < 0.72)
                & (slope < 0.11)
                & (
                    (density_field > 0.0)
                    | (thematic.land_potential >= 0.30)
                    | (road_access >= 0.24)
                    | ocean_coast
                    | lake_coast
                )
            )
            rows, columns = np.nonzero(valid)
            if not rows.size:
                break
            distance = _wrapped_distances(rows, columns, centres, grid.shape[1])
            spaced = distance >= service_spacing
            if not bool(np.any(spaced)):
                break
            rows = rows[spaced]
            columns = columns[spaced]
            distance = distance[spaced]
            score = (
                0.24 * np.clip(distance / (service_spacing * 2.8), 0.0, 1.0)
                + 0.24
                * np.clip(
                    density_field[rows, columns]
                    / max(population_scale, 1.0e-12),
                    0.0,
                    1.0,
                )
                + 0.18 * thematic.land_potential[rows, columns]
                + 0.16 * road_access[rows, columns]
                + 0.18 * road_junction[rows, columns]
                + 0.05 * ocean_coast[rows, columns]
                + 0.03 * lake_coast[rows, columns]
                + 0.09 * transport.accessibility[rows, columns]
                - 0.12 * np.clip(slope[rows, columns] / 0.11, 0.0, 1.0)
            )
            order = np.lexsort(
                (columns, rows, -score)
            )
            choice = int(order[0])
            row = int(rows[choice])
            column = int(columns[choice])
            chosen_score = float(score[choice])
            river_bank_city = False
            if int(grid.river_order[row, column]) > 0:
                bank_score = np.where(
                    region & (grid.water == 0) & ~grid.snow,
                    0.46 * np.clip(
                        density_field / max(population_scale, 1.0e-12),
                        0.0,
                        1.0,
                    )
                    + 0.34 * thematic.land_potential
                    + 0.20 * transport.accessibility,
                    -np.inf,
                )
                bank_row, bank_column = _river_bank_location(
                    row,
                    column,
                    region & (grid.water == 0),
                    grid.river_order,
                    bank_score,
                    occupied=set(centres),
                )
                if (bank_row, bank_column) != (row, column):
                    row, column = bank_row, bank_column
                    river_bank_city = True
            centres.append((row, column))
            if ocean_coast[row, column]:
                site_type = "port"
            elif lake_coast[row, column]:
                site_type = "lake-port"
            elif river_bank_city or grid.river_order[row, column] > 0:
                site_type = "river-city"
            elif thematic.climate.annual_precipitation[row, column] < 0.065:
                site_type = "oasis"
            else:
                site_type = "market"
            radius = 4
            local_rows = slice(max(0, row - radius), min(grid.shape[0], row + radius + 1))
            local_columns = (
                np.arange(column - radius, column + radius + 1) % grid.shape[1]
            ).astype(int)
            local_weight = float(
                population.population_weight[local_rows][:, local_columns].sum()
            )
            population_min = max(
                1_000,
                int(round(local_weight * population.population_min / 1000.0)) * 1000,
            )
            population_max = max(
                population_min + 500,
                int(round(local_weight * population.population_max / 1000.0)) * 1000,
            )
            tier = (
                "city"
                if density_field[row, column] >= 1.0
                or density_field[row, column] >= population_scale
                or road_junction[row, column] >= 0.48
                else "town"
            )
            additions.append(
                Settlement(
                    identifier=(
                        f"administrative-centre-"
                        f"{identifier_offset + len(additions) + 1:03d}"
                    ),
                    name=f"行政聚落{len(additions) + 1:03d}",
                    row=row,
                    column=column,
                    tier=tier,
                    site_type=site_type,
                    score=chosen_score,
                    population_min=population_min,
                    population_max=population_max,
                )
            )
    return tuple(additions)


__all__ = ["derive_administrative_centres"]
