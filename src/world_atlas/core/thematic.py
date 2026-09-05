"""Deterministic thematic fields derived from the canonical WorldGrid.

These arrays are review products, not a second world model.  Climate, biomes,
drainage basins and land potential therefore share the same elevation,
hydrology and four-season precipitation inputs as the physical map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import WorldGrid
from .polar import polar_continent_mask


BIOME_ICE = 0
BIOME_TUNDRA_ALPINE = 1
BIOME_BOREAL_FOREST = 2
BIOME_TEMPERATE_MIXED_FOREST = 3
BIOME_TEMPERATE_GRASSLAND = 4
BIOME_MEDITERRANEAN_SCRUB = 5
BIOME_HUMID_SUBTROPICAL = 6
BIOME_TROPICAL_RAINFOREST = 7
BIOME_TROPICAL_SEASONAL_FOREST = 8
BIOME_SAVANNA_DRY_GRASSLAND = 9
BIOME_DESERT_SCRUB = 10
BIOME_POLAR_ICE_DESERT = 11
BIOME_POLAR_COASTAL_TUNDRA = 12
BIOME_COUNT = 13


@dataclass(frozen=True, slots=True)
class ClimateDrivers:
    """Continuous climate drivers and their broad categorical partition."""

    temperature: np.ndarray
    seasonal_precipitation: np.ndarray
    annual_precipitation: np.ndarray
    precipitation_range: np.ndarray
    climate_zone: np.ndarray

    def __post_init__(self) -> None:
        shape = np.asarray(self.temperature).shape
        expected = {
            "temperature": (shape, np.dtype(np.float64)),
            "seasonal_precipitation": ((4, *shape), np.dtype(np.float64)),
            "annual_precipitation": (shape, np.dtype(np.float64)),
            "precipitation_range": (shape, np.dtype(np.float64)),
            "climate_zone": (shape, np.dtype(np.int8)),
        }
        for name, (expected_shape, expected_dtype) in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != expected_shape or value.dtype != expected_dtype:
                raise ValueError(
                    f"{name} must have shape {expected_shape} and dtype {expected_dtype}"
                )
            object.__setattr__(self, name, _immutable(value))


@dataclass(frozen=True, slots=True)
class ThematicLayers:
    """All derived thematic fields needed by the three review maps."""

    climate: ClimateDrivers
    biome_zone: np.ndarray
    drainage_basin: np.ndarray
    major_basin_rank: np.ndarray
    land_potential: np.ndarray
    land_potential_band: np.ndarray

    def __post_init__(self) -> None:
        shape = self.climate.temperature.shape
        expected = {
            "biome_zone": np.dtype(np.int8),
            "drainage_basin": np.dtype(np.int32),
            "major_basin_rank": np.dtype(np.int16),
            "land_potential": np.dtype(np.float32),
            "land_potential_band": np.dtype(np.uint8),
        }
        for name, dtype in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
            object.__setattr__(self, name, _immutable(value))


def _immutable(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value).copy()
    result.setflags(write=False)
    return result


def smooth_field(field: np.ndarray, *, radius: int, passes: int = 2) -> np.ndarray:
    """Box-smooth a world field with wrapped longitude and clamped poles."""

    result = np.asarray(field, dtype=np.float64)
    radius = int(radius)
    if radius <= 0 or passes <= 0:
        return result.copy()
    kernel_width = radius * 2 + 1
    height = result.shape[0]
    for _ in range(passes):
        wrapped = np.concatenate(
            (result[:, -radius:], result, result[:, :radius]),
            axis=1,
        )
        horizontal_sum = np.pad(
            np.cumsum(wrapped, axis=1, dtype=np.float64),
            ((0, 0), (1, 0)),
        )
        horizontal = (
            horizontal_sum[:, kernel_width:] - horizontal_sum[:, :-kernel_width]
        ) / kernel_width
        padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
        vertical_sum = np.pad(
            np.cumsum(padded, axis=0, dtype=np.float64),
            ((1, 0), (0, 0)),
        )
        result = (
            vertical_sum[kernel_width:] - vertical_sum[:-kernel_width]
        ) / kernel_width
        if result.shape[0] != height:
            raise RuntimeError("smoothed climate field changed shape")
    return result


def regularize_partition(
    zones: np.ndarray,
    *,
    category_count: int,
    radius: int,
    protected_categories: tuple[int, ...] = (-1, 0),
) -> np.ndarray:
    """Remove categorical slivers without changing protected water/ice cells."""

    values = np.asarray(zones)
    if radius <= 0:
        return values.copy()
    best_zone = values.copy()
    best_support = np.full(values.shape, -1.0, dtype=np.float64)
    original_support = np.zeros(values.shape, dtype=np.float64)
    protected = np.isin(values, protected_categories)
    for zone in range(category_count):
        if zone in protected_categories:
            continue
        support = smooth_field(values == zone, radius=radius, passes=1)
        original_support[values == zone] = support[values == zone]
        stronger = support > best_support
        best_zone[stronger] = zone
        best_support[stronger] = support[stronger]
    cleaned = values.copy()
    replace = ~protected & (best_support > original_support + 1.0e-12)
    cleaned[replace] = best_zone[replace]
    return cleaned


def derive_climate_drivers(grid: WorldGrid) -> ClimateDrivers:
    """Derive the established broad climate field from the four seasons."""

    height, _width = grid.shape
    extents = grid.metadata["extents"]
    north = float(extents["north"])
    south = float(extents["south"])
    latitude = north - (np.arange(height, dtype=np.float64) + 0.5) * (
        (north - south) / height
    )
    latitude_field = latitude[:, None]
    climate_scale = min(grid.shape)
    relief_radius = min(12, climate_scale // 36)
    rain_radius = min(16, climate_scale // 28)
    temperature = np.clip(
        np.cos(np.radians(np.abs(latitude_field))) ** 1.25
        - 0.48 * smooth_field(grid.elevation, radius=relief_radius),
        0.0,
        1.0,
    )
    seasonal_rain = np.stack(
        [
            smooth_field(season.astype(np.float64) / 255.0, radius=rain_radius)
            for season in grid.seasonal_precipitation
        ]
    )
    annual_rain = seasonal_rain.mean(axis=0)
    rain_range = seasonal_rain.max(axis=0) - seasonal_rain.min(axis=0)
    northern = latitude_field >= 0.0
    summer_rain = np.where(northern, seasonal_rain[1], seasonal_rain[3])
    winter_rain = np.where(northern, seasonal_rain[3], seasonal_rain[1])
    dry_limit = 0.035 + 0.025 * temperature

    zones = np.full(grid.shape, 4, dtype=np.int8)
    zones[temperature < 0.38] = 2
    zones[temperature < 0.22] = 1
    zones[
        (temperature >= 0.38)
        & (temperature < 0.62)
        & (annual_rain >= 0.17)
        & (rain_range < 0.20)
    ] = 3
    zones[
        (temperature >= 0.42)
        & (summer_rain + 0.025 < winter_rain * 0.62)
        & (annual_rain >= dry_limit * 1.25)
    ] = 5
    zones[temperature >= 0.62] = 6
    zones[temperature >= 0.72] = 8
    zones[
        (temperature >= 0.72)
        & (annual_rain >= 0.24)
        & (seasonal_rain.min(axis=0) >= 0.12)
    ] = 7
    dry_eligible = temperature >= 0.22
    zones[dry_eligible & (annual_rain < dry_limit * 1.25)] = 9
    zones[dry_eligible & (annual_rain < dry_limit * 0.55)] = 10
    zones[grid.snow & (grid.water == 0)] = 0
    zones[grid.water != 0] = -1
    zones = regularize_partition(
        zones,
        category_count=11,
        radius=min(5, climate_scale // 180),
    ).astype(np.int8)
    return ClimateDrivers(
        temperature=temperature.astype(np.float64),
        seasonal_precipitation=seasonal_rain.astype(np.float64),
        annual_precipitation=annual_rain.astype(np.float64),
        precipitation_range=rain_range.astype(np.float64),
        climate_zone=zones,
    )


def derive_biome_zones(grid: WorldGrid, climate: ClimateDrivers) -> np.ndarray:
    """Classify adjacent, non-overlapping vegetation regions from climate."""

    temperature = climate.temperature
    annual = climate.annual_precipitation
    seasonal_min = climate.seasonal_precipitation.min(axis=0)
    zones = np.full(grid.shape, BIOME_TEMPERATE_MIXED_FOREST, dtype=np.int8)

    zones[temperature < 0.38] = BIOME_BOREAL_FOREST
    zones[temperature < 0.22] = BIOME_TUNDRA_ALPINE
    zones[(temperature >= 0.38) & (annual < 0.085)] = BIOME_TEMPERATE_GRASSLAND
    zones[(temperature >= 0.62) & (annual >= 0.10)] = BIOME_HUMID_SUBTROPICAL
    zones[temperature >= 0.76] = BIOME_TROPICAL_SEASONAL_FOREST
    zones[(temperature >= 0.76) & (annual < 0.10)] = BIOME_SAVANNA_DRY_GRASSLAND
    zones[
        (temperature >= 0.76)
        & (annual >= 0.16)
        & (seasonal_min >= 0.055)
    ] = BIOME_TROPICAL_RAINFOREST
    zones[(temperature >= 0.64) & (annual < 0.05)] = BIOME_SAVANNA_DRY_GRASSLAND
    zones[
        (temperature >= 0.22) & (temperature < 0.64) & (annual < 0.055)
    ] = BIOME_TEMPERATE_GRASSLAND
    zones[annual < 0.032] = BIOME_DESERT_SCRUB

    height = grid.shape[0]
    extents = grid.metadata["extents"]
    north = float(extents["north"])
    south = float(extents["south"])
    latitude = north - (np.arange(height, dtype=np.float64) + 0.5) * (
        (north - south) / height
    )
    northern = latitude[:, None] >= 0.0
    summer_rain = np.where(
        northern,
        climate.seasonal_precipitation[1],
        climate.seasonal_precipitation[3],
    )
    winter_rain = np.where(
        northern,
        climate.seasonal_precipitation[3],
        climate.seasonal_precipitation[1],
    )
    mediterranean = (
        (temperature >= 0.45)
        & (temperature < 0.76)
        & (annual >= 0.065)
        & (summer_rain + 0.005 < winter_rain * 0.90)
    )
    zones[mediterranean] = BIOME_MEDITERRANEAN_SCRUB
    zones[grid.snow & (grid.water == 0)] = BIOME_ICE
    zones[grid.water != 0] = -1
    cleaned = regularize_partition(
        zones,
        category_count=BIOME_COUNT,
        radius=min(5, min(grid.shape) // 180),
    )
    polar_land = polar_continent_mask(grid)
    if bool(polar_land.any()):
        ocean_nearby = grid.water == 1
        for _ in range(max(2, min(8, min(grid.shape) // 120))):
            vertical = np.pad(ocean_nearby, ((1, 1), (0, 0)), mode="constant")
            ocean_nearby = (
                ocean_nearby
                | vertical[:-2]
                | vertical[2:]
                | np.roll(ocean_nearby, 1, axis=1)
                | np.roll(ocean_nearby, -1, axis=1)
            )
        coastal_tundra = (
            polar_land
            & ocean_nearby
            & ~grid.snow
            & (temperature >= 0.08)
            & (annual >= 0.035)
            & (grid.elevation <= 0.30)
        )
        cleaned[polar_land] = BIOME_POLAR_ICE_DESERT
        cleaned[coastal_tundra] = BIOME_POLAR_COASTAL_TUNDRA
    return _immutable(cleaned.astype(np.int8))


def derive_drainage_basins(grid: WorldGrid) -> np.ndarray:
    """Assign every land cell to its terminal D8 outlet using pointer jumping."""

    cell_count = int(np.prod(grid.shape, dtype=np.int64))
    indices = np.arange(cell_count, dtype=np.int64)
    flow = grid.flow_to.reshape(-1).astype(np.int64, copy=False)
    water = grid.water.reshape(-1) != 0
    parent = indices.copy()
    valid = (flow >= 0) & ~water
    parent[valid] = flow[valid]
    parent[water] = indices[water]

    maximum_steps = max(2, int(np.ceil(np.log2(max(2, cell_count)))) + 2)
    for _ in range(maximum_steps):
        next_parent = parent[parent]
        if np.array_equal(next_parent, parent):
            parent = next_parent
            break
        parent = next_parent

    # A functional-graph cycle does not converge to one shared terminal under
    # pointer jumping.  Validate the original edge relation after compression.
    if np.any(valid & (parent != parent[np.maximum(flow, 0)])):
        raise ValueError("flow_to contains a drainage cycle")

    basins = parent.astype(np.int32)
    basins[water] = -1
    return _immutable(basins.reshape(grid.shape))


def rank_major_basins(
    grid: WorldGrid,
    basins: np.ndarray,
    *,
    maximum: int = 24,
) -> np.ndarray:
    """Rank the largest outlet basins; zero denotes all minor basins."""

    if isinstance(maximum, bool) or int(maximum) < 1:
        raise ValueError("maximum must be a positive integer")
    basin_values = np.asarray(basins, dtype=np.int32)
    if basin_values.shape != grid.shape:
        raise ValueError("basins must match the WorldGrid shape")
    land_values = basin_values[grid.water == 0]
    identifiers, counts = np.unique(land_values[land_values >= 0], return_counts=True)
    order = np.lexsort((identifiers, -counts))[: int(maximum)]
    ranked = np.zeros(grid.shape, dtype=np.int16)
    ranked[grid.water != 0] = -1
    for rank, index in enumerate(order, start=1):
        ranked[basin_values == identifiers[index]] = rank
    return _immutable(ranked)


def _distance_influence(mask: np.ndarray, radius: int) -> np.ndarray:
    """Return a deterministic finite-radius proximity score with wrapped longitude."""

    active = np.asarray(mask, dtype=bool)
    score = active.astype(np.float64)
    frontier = active.copy()
    for distance in range(1, radius + 1):
        padded = np.pad(frontier, ((1, 1), (0, 0)), mode="constant")
        expanded = (
            np.roll(frontier, 1, axis=1)
            | np.roll(frontier, -1, axis=1)
            | padded[:-2]
            | padded[2:]
        )
        frontier = expanded
        score = np.maximum(score, expanded * (1.0 - distance / (radius + 1.0)))
    return score


def _relief_score(elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(elevation, dtype=np.float64)
    east_west = (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1)) * 0.5
    north_south = np.empty_like(values)
    if values.shape[0] == 1:
        north_south.fill(0.0)
    else:
        north_south[1:-1] = (values[2:] - values[:-2]) * 0.5
        north_south[0] = values[1] - values[0]
        north_south[-1] = values[-1] - values[-2]
    gradient = np.hypot(east_west, north_south)
    score = (1.0 - np.clip(gradient / 0.08, 0.0, 1.0)) * (
        1.0 - np.clip((values - 0.62) / 0.30, 0.0, 1.0)
    )
    return score, gradient


def derive_land_potential(
    grid: WorldGrid,
    climate: ClimateDrivers,
) -> tuple[np.ndarray, np.ndarray]:
    """Score agricultural capacity before the population model is applied.

    Climate and water determine the resource ceiling.  Terrain is a carrying-
    capacity constraint rather than another interchangeable bonus: steep or
    high ground remains marginal even when it is wet, while flat lower river
    valleys and lake shores receive a modest alluvial-access advantage.
    """

    heat = np.clip(1.0 - np.abs(climate.temperature - 0.62) / 0.55, 0.0, 1.0)
    moisture = np.clip(climate.annual_precipitation / 0.18, 0.0, 1.0) * np.clip(
        (0.50 - climate.annual_precipitation) / 0.20,
        0.0,
        1.0,
    )
    reliability = 1.0 - np.clip(climate.precipitation_range / 0.45, 0.0, 1.0)
    _relief, gradient = _relief_score(grid.elevation)
    scale = min(grid.shape)
    freshwater_radius = max(2, min(16, scale // 64))
    freshwater = _distance_influence(
        (grid.water == 2) | (grid.river_order > 0),
        freshwater_radius,
    )
    maritime = _distance_influence(
        (grid.water == 1) | (grid.water == 3),
        max(2, freshwater_radius // 2),
    )
    climate_water = (
        0.34 * heat
        + 0.29 * moisture * reliability
        + 0.19 * freshwater
        + 0.10 * maritime
        + 0.08 * reliability
    )

    land = grid.water == 0
    smoothing_radius = min(6, scale // 180)
    if smoothing_radius > 0:
        support = smooth_field(land, radius=smoothing_radius, passes=1)
        smoothed = smooth_field(climate_water * land, radius=smoothing_radius, passes=1)
        climate_water = np.divide(
            smoothed,
            support,
            out=climate_water.copy(),
            where=support > 1.0e-9,
        )

    # Relative slope breakpoints mirror the ordering of established land-
    # evaluation classes without pretending that this normalized elevation
    # grid is a metre-based DEM.
    slope_capacity = np.interp(
        gradient,
        (0.0, 0.018, 0.035, 0.060, 0.095, 0.14),
        (1.0, 0.98, 0.84, 0.58, 0.28, 0.08),
    )
    highland = np.clip((grid.elevation - 0.42) / 0.43, 0.0, 1.0)
    elevation_capacity = 1.0 - 0.72 * np.power(highland, 1.25)
    terrain_capacity = slope_capacity * elevation_capacity

    lower_land = np.clip((0.66 - grid.elevation) / 0.28, 0.0, 1.0)
    flat_land = np.clip((slope_capacity - 0.25) / 0.75, 0.0, 1.0)
    major_river_access = _distance_influence(
        grid.river_order >= 2,
        max(2, freshwater_radius // 2),
    )
    lake_access = _distance_influence(
        grid.water == 2,
        max(2, freshwater_radius // 2),
    )
    alluvial_access = major_river_access * flat_land * lower_land
    lake_shore_access = lake_access * flat_land * lower_land

    value = climate_water * (0.22 + 0.78 * terrain_capacity)
    value += 0.12 * alluvial_access + 0.06 * lake_shore_access
    value += 0.05 * flat_land * lower_land
    value = np.clip(value, 0.0, 1.0)
    value[climate.temperature < 0.16] = np.minimum(value[climate.temperature < 0.16], 0.12)
    value[climate.annual_precipitation < 0.035] = np.minimum(
        value[climate.annual_precipitation < 0.035], 0.15
    )
    value[grid.elevation > 0.85] = np.minimum(value[grid.elevation > 0.85], 0.15)
    value[gradient > 0.12] = np.minimum(value[gradient > 0.12], 0.20)
    severe_terrain = terrain_capacity < 0.22
    value[severe_terrain] = np.minimum(value[severe_terrain], 0.24)
    value[grid.snow | ~land] = 0.0

    potential = value.astype(np.float32)
    bands = np.digitize(potential, (0.34, 0.44, 0.50, 0.58, 0.67)).astype(np.uint8)
    bands[~land] = 0
    return _immutable(potential), _immutable(bands)


def derive_thematic_layers(grid: WorldGrid) -> ThematicLayers:
    """Derive all review themes once from one immutable WorldGrid."""

    climate = derive_climate_drivers(grid)
    biomes = derive_biome_zones(grid, climate)
    basins = derive_drainage_basins(grid)
    ranks = rank_major_basins(grid, basins)
    potential, potential_bands = derive_land_potential(grid, climate)
    return ThematicLayers(
        climate=climate,
        biome_zone=biomes,
        drainage_basin=basins,
        major_basin_rank=ranks,
        land_potential=potential,
        land_potential_band=potential_bands,
    )
