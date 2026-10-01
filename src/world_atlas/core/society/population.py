"""Late-medieval population support derived from physical opportunity."""

from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy import ndimage

from ..model import WorldGrid
from ..thematic import ThematicLayers, smooth_field
from ..suitability import continuous_proximity, relative_land_slope
from .model import PopulationLayers, Settlement
from .spatial import (
    connected_components,
    reduce_field,
    select_spaced_seeds,
    select_spaced_candidates,
    society_domain_mask,
)


POPULATION_DENSITY_THRESHOLDS = (0.0,.1,.5,1.0,2.0,5.0)


def _settlement_layout_variation(
    shape: tuple[int, int],
    world_identity: str,
) -> np.ndarray:
    """Return a stable, low-amplitude field for equally plausible town sites.

    Geography remains the dominant settlement cause.  This field only breaks
    broad plateaus of near-equal physical opportunity, so a regenerated human
    world does not repeatedly pin every town to the same scan-order cell.
    """

    if len(shape) != 2 or any(isinstance(size, bool) or int(size) < 1 for size in shape):
        raise ValueError("shape must contain two positive dimensions")
    if not isinstance(world_identity, str) or not world_identity:
        raise ValueError("world_identity must be a non-empty string")
    seed = np.uint64(
        int.from_bytes(
            hashlib.sha256(
                f"society-layout-v2:{world_identity}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
    )
    rows, columns = np.indices(tuple(int(size) for size in shape), dtype=np.uint64)
    values = (
        seed
        + rows * np.uint64(0x9E3779B97F4A7C15)
        + columns * np.uint64(0xBF58476D1CE4E5B9)
    )
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    unit = (values >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    field = smooth_field(unit * 2.0 - 1.0, radius=2, passes=2)
    maximum = float(np.max(np.abs(field), initial=0.0))
    if maximum > 0.0:
        field /= maximum
    return np.clip(field, -1.0, 1.0)


def _proximity(mask: np.ndarray, radius: int) -> np.ndarray:
    active = np.asarray(mask, dtype=bool)
    if radius <= 0 or not active.any():
        return active.astype(np.float64)
    # The former radius rounds of four-neighbour dilation are exactly the
    # taxicab distance transform. Only the queried longitude halo is needed.
    padded = np.pad(~active, ((0, 0), (radius, radius)), mode="wrap")
    distance = ndimage.distance_transform_cdt(padded, metric="taxicab")[:, radius:-radius]
    scores = np.maximum(0.0, 1.0 - np.arange(radius + 2) / (radius + 1.0))
    distance = np.where(distance < 0, radius + 1, np.minimum(distance, radius + 1))
    return scores[distance]


def _latitude_area_weights(grid: WorldGrid) -> np.ndarray:
    extents = grid.metadata["extents"]
    north = float(extents["north"])
    south = float(extents["south"])
    latitudes = north - (np.arange(grid.shape[0], dtype=np.float64) + 0.5) * (
        (north - south) / grid.shape[0]
    )
    return np.clip(np.cos(np.radians(latitudes)), 0.0, 1.0)[:, None]


def _population_bands(density: np.ndarray, land: np.ndarray) -> np.ndarray:
    result = np.zeros(density.shape, dtype=np.uint8)
    positive=land&(density>0.0)
    result[positive]=np.digitize(density[positive],POPULATION_DENSITY_THRESHOLDS).astype(np.uint8)
    return result


def cell_areas_km2(grid: WorldGrid) -> np.ndarray:
    """Return exact spherical native-cell areas as a read-only broadcast view."""
    extents=grid.metadata['extents']
    radius=float(grid.metadata['planet']['radiusKm'])
    latitude=np.linspace(float(extents['north']),float(extents['south']),grid.shape[0]+1)
    longitude_width=math.radians(float(extents['east'])-float(extents['west']))/grid.shape[1]
    areas=radius*radius*longitude_width*np.abs(np.diff(np.sin(np.radians(latitude))))
    if np.any(areas<=0.0) or np.any(~np.isfinite(areas)):
        raise ValueError('Population density requires positive spherical cell areas')
    return np.broadcast_to(areas[:,None], grid.shape)


def _density_from_weights(grid: WorldGrid, weights: np.ndarray, minimum: int, maximum: int) -> np.ndarray:
    if weights.shape!=grid.shape:
        raise ValueError('Population weights must match the WorldGrid shape')
    density=(np.asarray(weights,dtype=np.float64)*((minimum+maximum)/2.0)/cell_areas_km2(grid)).astype(np.float32)
    density.setflags(write=False)
    return density


def population_density(grid: WorldGrid, population: PopulationLayers) -> np.ndarray:
    """Estimate persons/km² from midpoint population and actual spherical area.

    Population weights remain cell headcount fractions. Displaying those
    fractions directly would confuse smaller high-latitude cells with sparse
    settlement and change colours despite equal density on the ground.
    """
    return _density_from_weights(grid,population.population_weight,
                                 population.population_min,population.population_max)


def _latent_drainage_access(
    discharge: np.ndarray,
    rainfall_reliability: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Estimate local streams, springs and shallow-water access below map scale.

    ``river_order`` is a cartographic selection, while runoff is accumulated
    for every drainage cell.  Log-scaled discharge therefore supplies a useful
    proxy for tributaries and local watercourses that are real to settlement
    geography but too small to draw on the world map.
    """

    runoff = np.maximum(np.asarray(discharge, dtype=np.float64), 0.0)
    reliability = np.clip(
        np.asarray(rainfall_reliability, dtype=np.float64), 0.0, 1.0
    )
    allowed = np.asarray(valid, dtype=bool)
    if runoff.shape != reliability.shape or allowed.shape != runoff.shape:
        raise ValueError("latent drainage fields must align")
    positive = runoff[allowed & (runoff > 0.0)]
    if positive.size == 0:
        return np.zeros(runoff.shape, dtype=np.float64)
    reference = max(1.0, float(np.quantile(positive, 0.96)))
    drainage = np.clip(
        np.log1p(runoff) / max(math.log1p(reference), 1.0e-12),
        0.0,
        1.0,
    )
    drainage *= 0.52 + 0.48 * reliability
    drainage[~allowed] = 0.0
    return drainage


def _settlement_freshwater_access(
    grid: WorldGrid,
    rainfall_reliability: np.ndarray,
    land: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mapped-stream and sub-map water access for settlement choice.

    ``river_order`` is the complete, evidence-gated canonical network.  A
    settlement may benefit from every stream class, but never occupies a
    channel cell itself.  Major rivers retain a separate corridor influence;
    accumulated discharge provides the distinct groundwater/spring proxy for
    habitable interior land that is not beside any mapped line.
    """

    allowed = np.asarray(land, dtype=bool)
    if allowed.shape != grid.shape:
        raise ValueError("settlement freshwater land mask must match the WorldGrid")
    scale = min(grid.shape)
    streams = grid.river_order > 0
    major_streams = grid.river_order >= 2
    stream_access = continuous_proximity(
        streams,
        max(2, min(12, scale // 96)),
    )
    major_stream_access = continuous_proximity(
        major_streams,
        max(2, min(10, scale // 110)),
    )
    lake_access = continuous_proximity(
        grid.water == 2,
        max(1, min(5, scale // 180)),
    )
    stream_bank = allowed & (_proximity(streams, 1) > 0.0)
    groundwater_access = smooth_field(
        _latent_drainage_access(
            grid.discharge,
            rainfall_reliability,
            allowed,
        ),
        radius=min(grid.shape[1], max(2, min(6, scale // 180))),
        passes=2,
    )
    groundwater_access[~allowed] = 0.0
    freshwater = np.maximum.reduce(
        (stream_access, lake_access, groundwater_access)
    )
    freshwater[~allowed] = 0.0
    return (
        stream_access,
        major_stream_access,
        stream_bank,
        groundwater_access,
        freshwater,
    )


def derive_population(
    grid: WorldGrid,
    thematic: ThematicLayers,
    *,
    population_min: int = 450_000_000,
    population_max: int = 600_000_000,
) -> PopulationLayers:
    """Derive normalized population weights without inventing local headcounts."""

    if thematic.land_potential.shape != grid.shape or thematic.habitability.shape != grid.shape:
        raise ValueError("thematic layers must match the WorldGrid shape")
    # Population, culture and states share one explicit human domain.  Polar
    # continents deliberately stop at physical climate/biome generation.
    land = society_domain_mask(grid)
    reliability = 1.0 - np.clip(
        thematic.climate.precipitation_range / 0.45,
        0.0,
        1.0,
    )
    potential = np.asarray(thematic.land_potential, dtype=np.float64)
    habitability=np.asarray(thematic.habitability,dtype=np.float64)
    support = (
        np.power(np.clip(potential, 0.0, 1.0), 1.55)
        * np.power(np.clip(habitability,0.0,1.0),1.10)
        * (0.62 + 0.38 * reliability)
        * _latitude_area_weights(grid)
    )
    inhabitable = land & ~grid.snow
    support[~inhabitable] = 0.0
    total = float(support.sum(dtype=np.float64))
    if total <= 0.0:
        return PopulationLayers(
            population_weight=np.zeros(grid.shape, dtype=np.float32),
            population_band=np.zeros(grid.shape, dtype=np.uint8),
            population_min=population_min,
            population_max=population_max,
        )
    weights = (support / total).astype(np.float32)
    # Float32 normalization can drift by a few ulps on global grids.  Correct
    # one stable cell so downstream population intervals preserve their total.
    anchor = int(np.argmax(weights.reshape(-1)))
    correction = np.float32(1.0 - float(weights.sum(dtype=np.float64)))
    weights.reshape(-1)[anchor] += correction
    density=_density_from_weights(grid,weights,population_min,population_max)
    bands = _population_bands(density, land)
    return PopulationLayers(
        population_weight=weights,
        population_band=bands,
        population_min=population_min,
        population_max=population_max,
    )


def _adjacent_water(grid: WorldGrid, water_values: tuple[int, ...]) -> np.ndarray:
    water = np.isin(grid.water, water_values)
    padded = np.pad(water, ((1, 1), (0, 0)), mode="constant")
    return (
        water
        | np.roll(water, 1, axis=1)
        | np.roll(water, -1, axis=1)
        | padded[:-2]
        | padded[2:]
    )


def _select_spaced_additions(
    score: np.ndarray,
    valid: np.ndarray,
    *,
    count: int,
    minimum_distance: float,
    occupied: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    if count <= 0:
        return ()
    values = np.asarray(score, dtype=np.float64)
    allowed = np.asarray(valid, dtype=bool)
    flat = np.flatnonzero(allowed & np.isfinite(values))
    rows, columns = np.unravel_index(flat, values.shape)
    # Stable spatial hash breaks equal-score plateaus without the old
    # row-major bias that packed every plain town into one side of a uniform
    # fertile region.
    tie_break = (
        (rows.astype(np.uint64) * np.uint64(73856093))
        ^ (columns.astype(np.uint64) * np.uint64(19349663))
    ) & np.uint64(0xFFFFFFFF)
    order = flat[
        np.lexsort(
            (
                flat,
                -tie_break.astype(np.float64),
                -values.reshape(-1)[flat],
            )
        )
    ]
    chosen = select_spaced_candidates(order, values.shape, count=count,
        minimum_distance=minimum_distance, occupied=occupied)
    occupied.extend(chosen)
    return tuple(chosen)


def _resolve_seed(
    seed: tuple[int, int],
    score: np.ndarray,
    *,
    step: int,
    shape: tuple[int, int],
) -> tuple[int, int, float]:
    coarse_row, coarse_column = seed
    row_start = coarse_row * step
    column_start = coarse_column * step
    block = score[
        row_start : min(row_start + step, shape[0]),
        column_start : min(column_start + step, shape[1]),
    ]
    flat_index = int(np.argmax(block))
    local_row, local_column = np.unravel_index(flat_index, block.shape)
    row = row_start + int(local_row)
    column = column_start + int(local_column)
    return row, column, float(score[row, column])


def _river_bank_location(
    row: int,
    column: int,
    land: np.ndarray,
    river_order: np.ndarray,
    site_score: np.ndarray,
    *,
    occupied: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Place a river settlement on one bank, never in the channel cell.

    The selected land cell lies on the nearest dry raster outside the rendered
    channel envelope.  Its exact side then becomes available to later culture
    and state allocation; a city no longer acts as a seed planted
    simultaneously on both banks.
    """

    allowed = np.asarray(land, dtype=bool)
    rivers = np.asarray(river_order)
    scores = np.asarray(site_score, dtype=np.float64)
    if allowed.shape != rivers.shape or scores.shape != allowed.shape:
        raise ValueError("river-bank settlement fields must align")
    if int(rivers[row, column]) <= 0:
        return int(row), int(column)
    used = set() if occupied is None else occupied
    height, width = allowed.shape
    candidates: list[tuple[float, int, int]] = []
    for radius in range(1, 5):
        for dy in range(-radius, radius + 1):
            next_row = row + dy
            if next_row < 0 or next_row >= height:
                continue
            for dx in range(-radius, radius + 1):
                if max(abs(dy), abs(dx)) != radius:
                    continue
                next_column = (column + dx) % width
                location = (next_row, next_column)
                if (
                    location in used
                    or not allowed[location]
                    or int(rivers[location]) != 0
                    or not np.isfinite(scores[location])
                ):
                    continue
                candidates.append(
                    (
                        float(scores[location]) - 0.015 * math.hypot(dy, dx),
                        next_row,
                        next_column,
                    )
                )
        if candidates:
            break
    if not candidates:
        return int(row), int(column)
    _score, bank_row, bank_column = max(
        candidates,
        key=lambda item: (item[0], -item[1], -item[2]),
    )
    return bank_row, bank_column


def derive_settlements(
    grid: WorldGrid,
    thematic: ThematicLayers,
    population: PopulationLayers,
    *,
    target_count: int | None = None,
) -> tuple[Settlement, ...]:
    """Resolve a deterministic network of supplied, watered settlement sites."""

    if population.population_weight.shape != grid.shape:
        raise ValueError("population layers must match the WorldGrid shape")
    if not np.any(population.population_weight > 0.0):
        return ()
    # Do not re-open polar continents merely because they are ordinary land in
    # the physical grid.  Settlement selection inherits the population domain.
    land = society_domain_mask(grid)
    ocean_coast = land & _adjacent_water(grid, (1, 3))
    lake_coast = land & _adjacent_water(grid, (2,))
    river = grid.river_order > 0
    # A channel is hydrology, not buildable ground.  Select banks and nearby
    # terraces directly rather than choosing a line cell and repairing it
    # later; the final bank relocation remains a defensive invariant.
    settlement_land = land & ~river
    annual = thematic.climate.annual_precipitation
    population_relative = population.population_weight.astype(np.float64)
    maximum = float(population_relative.max(initial=0.0))
    if maximum > 0.0:
        population_relative /= maximum
    river_permanence = np.min(
        grid.seasonal_river_strength.astype(np.float64), axis=0
    ) / 255.0
    incoming = np.zeros(grid.shape, dtype=np.float64)
    routed = river & (grid.flow_to >= 0)
    np.add.at(incoming.reshape(-1), grid.flow_to[routed], 1.0)
    confluence_score = np.clip((incoming - 1.0) / 2.0, 0.0, 1.0)
    rainfall_reliability = 1.0 - np.clip(
        thematic.climate.precipitation_range.astype(np.float64) / 0.45,
        0.0,
        1.0,
    )
    (
        river_access,
        major_river_access,
        river_bank,
        _groundwater_access,
        freshwater_access,
    ) = _settlement_freshwater_access(grid, rainfall_reliability, land)
    river_bank &= settlement_land
    river_corridor = settlement_land & (major_river_access > 0.0)
    flowing_stream_access = _proximity(
        river & (river_permanence > 0.0),
        max(2, min(12, min(grid.shape) // 96)),
    )
    confluence_access = _proximity(
        confluence_score > 0.0,
        max(2, min(8, min(grid.shape) // 144)),
    )
    river_score = (
        0.58 * river_access * (0.72 + 0.28 * flowing_stream_access)
        + 0.42 * major_river_access
    )
    dry_water_need = np.clip((0.16 - annual) / 0.16, 0.0, 1.0)
    elevation = grid.elevation.astype(np.float64)
    slope = relative_land_slope(elevation, grid.water == 0)
    climate_support=thematic.habitability.astype(np.float64)
    settlement_terrain_penalty = (
        0.62 * np.clip(slope / 0.060, 0.0, 1.0)
        + 0.38 * np.clip((elevation - 0.48) / 0.30, 0.0, 1.0)
    )
    site_score = (
        0.40 * population_relative
        + 0.15 * thematic.land_potential.astype(np.float64)
        + 0.14 * thematic.habitability.astype(np.float64)
        + 0.10 * freshwater_access
        + 0.03 * river_score
        + 0.03 * confluence_access
        + 0.05 * ocean_coast
        + 0.06 * lake_coast
        + 0.10 * climate_support
        - 0.18 * dry_water_need * (1.0 - freshwater_access)
        - 0.22 * settlement_terrain_penalty
    )
    site_score += 0.035 * _settlement_layout_variation(
        grid.shape,
        # File locations and build provenance belong in the integrity digest,
        # not in random world decisions. Moving the replay bundle must not
        # move its cities and consequently change every human layer.
        "world-seed:"
        + str(grid.metadata.get("societyGeneration", {}).get("humanSeed", 0)),
    )
    site_score[~settlement_land | grid.snow] = -np.inf
    step = 1
    coarse_score = reduce_field(site_score, step=step, mode="max")
    positive_population = population_relative[population_relative > 0.0]
    population_floor = (
        float(np.quantile(positive_population, 0.22))
        if positive_population.size
        else 0.0
    )
    watered_exception = (
        ocean_coast | lake_coast | (freshwater_access >= 0.26)
    ) & (population_relative >= population_floor * 0.45)
    urban_valid = (
        settlement_land
        & ~grid.snow
        & (elevation < 0.72)
        & (slope < 0.105)
        & (
            (population_relative >= population_floor)
            | watered_exception
        )
    )
    coarse_valid = reduce_field(urban_valid, step=step, mode="max").astype(bool)
    coarse_land_valid = reduce_field(land & ~grid.snow, step=step, mode="max").astype(bool)
    if not np.any(coarse_valid):
        return ()
    if target_count is None:
        # A late-medieval atlas needs a fairly dense mesh of market towns, not
        # only a sparse set of capitals.  The former divisor left broad fertile
        # plains and temperate river valleys visually empty; keep more local
        # nodes while retaining spacing and physical-site validation below.
        target_count = int(np.clip(np.count_nonzero(coarse_valid) / 154.0, 300, 480))
    target_count = max(1, min(int(target_count), int(np.count_nonzero(coarse_valid))))
    spacing = max(
        1.0,
        np.sqrt(np.count_nonzero(coarse_valid) / (target_count * np.pi)) * 0.58,
    )
    strategic_enabled = target_count >= 40
    gateway_count = int(round(target_count * 0.025)) if strategic_enabled else 0
    coast_count = int(round(target_count * 0.075)) if strategic_enabled else 0
    river_count = int(round(target_count * 0.035)) if strategic_enabled else 0
    lake_count = int(round(target_count * 0.04)) if strategic_enabled else 0
    valley_count = int(round(target_count * 0.065)) if strategic_enabled else 0
    plain_count = int(round(target_count * 0.39)) if strategic_enabled else 0
    base_count = (
        target_count
        - gateway_count
        - coast_count
        - river_count
        - lake_count
        - valley_count
        - plain_count
    )
    base_seeds = select_spaced_seeds(
        coarse_score,
        coarse_valid,
        count=base_count,
        minimum_distance=float(spacing),
    )
    occupied = list(base_seeds)
    coarse_coast = reduce_field(
        ocean_coast & settlement_land,
        step=step,
        mode="max",
    ).astype(bool)
    coarse_river = reduce_field(river_corridor, step=step, mode="max").astype(bool)

    # Maritime gateways are not ordinary agricultural coast.  A narrow strait
    # or the transition between an inland sea and the open ocean can support a
    # major port through tolls, trans-shipment and naval control even when the
    # immediate hinterland is only moderately fertile.
    ocean_water = np.isin(grid.water, (1, 3))
    padded_land = np.pad(land, ((1, 1), (0, 0)), mode="constant")
    north_land = padded_land[:-2]
    south_land = padded_land[2:]
    west_land = np.roll(land, 1, axis=1)
    east_land = np.roll(land, -1, axis=1)
    diagonal_opposition = (
        (np.roll(north_land, 1, axis=1) & np.roll(south_land, -1, axis=1))
        | (np.roll(north_land, -1, axis=1) & np.roll(south_land, 1, axis=1))
    )
    narrow_water = ocean_water & (
        (north_land & south_land) | (west_land & east_land) | diagonal_opposition
    )
    inland_edge = _adjacent_water(grid, (3,))
    ocean_edge = _adjacent_water(grid, (1,))
    sea_transition = ocean_water & inland_edge & ocean_edge
    gateway_water = narrow_water | sea_transition
    gateway_reach = _proximity(
        gateway_water,
        max(3, min(12, min(grid.shape) // 120)),
    )
    gateway_valid = (
        ocean_coast
        & settlement_land
        & ~grid.snow
        & (gateway_reach >= 0.32)
        & (elevation < 0.64)
        & (slope < 0.095)
    )
    gateway_site_score = np.where(
        gateway_valid,
        site_score
        + 0.30 * gateway_reach
        + 0.12 * climate_support
        + 0.08 * freshwater_access,
        -np.inf,
    )
    coarse_gateway = reduce_field(gateway_valid, step=step, mode="max").astype(bool)
    coarse_gateway_score = reduce_field(gateway_site_score, step=step, mode="max")
    lake_components, lake_sizes = connected_components(grid.water == 2)
    lake_importance = np.zeros(grid.shape, dtype=np.float64)
    if lake_sizes:
        size_reference = max(1.0, float(np.quantile(lake_sizes, 0.82)))
        for identifier, size in enumerate(lake_sizes, start=1):
            lake_importance[lake_components == identifier] = np.clip(
                math.log1p(size) / math.log1p(size_reference),
                0.0,
                1.0,
            )
    adjacent_lake_importance = np.maximum.reduce(
        (
            lake_importance,
            np.roll(lake_importance, 1, axis=1),
            np.roll(lake_importance, -1, axis=1),
            np.pad(lake_importance, ((1, 1), (0, 0)), mode="constant")[:-2],
            np.pad(lake_importance, ((1, 1), (0, 0)), mode="constant")[2:],
        )
    )
    coarse_lake = reduce_field(
        lake_coast & settlement_land,
        step=step,
        mode="max",
    ).astype(bool)
    coast_site_score = np.where(ocean_coast & settlement_land, site_score, -np.inf)
    river_site_score = np.where(
        river_bank | river_corridor,
        site_score + 0.20 * confluence_access + 0.08 * flowing_stream_access,
        -np.inf,
    )
    coarse_coast_score = reduce_field(coast_site_score, step=step, mode="max")
    coarse_river_score = reduce_field(river_site_score, step=step, mode="max")
    lake_site_score = np.where(
        lake_coast & settlement_land,
        site_score + 0.24 * adjacent_lake_importance,
        -np.inf,
    )
    coarse_lake_score = reduce_field(lake_site_score, step=step, mode="max")

    # Warm, reliable, low-relief river valleys are the main agricultural and
    # transport corridors of the late-medieval settlement system.  They may
    # support cities even when the exact urban cell is beside, rather than on,
    # the rasterized river line.
    valley_valid = (
        settlement_land
        & ~grid.snow
        & (river_access >= 0.34)
        & (thematic.land_potential >= 0.40)
        & (population_relative >= population_floor * 0.42)
        & (thematic.habitability >= 0.32)
        & ((annual >= 0.070) | (freshwater_access >= 0.55))
        & (elevation < 0.54)
        & (slope < 0.058)
    )
    valley_site_score = np.where(
        valley_valid,
        site_score
        + 0.22 * climate_support
        + 0.18 * river_access
        + 0.12 * confluence_access
        + 0.08 * (1.0 - np.clip(slope / 0.058, 0.0, 1.0)),
        -np.inf,
    )
    coarse_valley_score = reduce_field(valley_site_score, step=step, mode="max")
    coarse_valley_valid = reduce_field(valley_valid, step=step, mode="max").astype(bool)

    # A separate quota prevents broad fertile plains from losing every town to
    # their coast and river edges.  These are deliberately towns rather than
    # automatic great cities; hierarchy is still earned by population and
    # corridor centrality.
    plain_valid = (
        settlement_land
        & ~grid.snow
        & ~ocean_coast
        & ~lake_coast
        & ~river
        & (thematic.land_potential >= 0.52)
        & (population_relative >= population_floor * 0.24)
        & (climate_support >= 0.48)
        & (elevation < 0.43)
        & (slope < 0.036)
    )
    plain_site_score = np.where(
        plain_valid,
        site_score
        + 0.24 * thematic.land_potential.astype(np.float64)
        + 0.14 * climate_support
        + 0.10 * population_relative
        + 0.06 * freshwater_access,
        -np.inf,
    )
    coarse_plain_score = reduce_field(plain_site_score, step=step, mode="max")
    coarse_plain_valid = reduce_field(plain_valid, step=step, mode="max").astype(bool)

    # Mountain relief alone cannot create a defended pass. Strategic gates
    # are selected after the real urban road network and its demand exist.
    seed_groups: list[tuple[tuple[tuple[int, int], ...], np.ndarray, str | None, str]] = [
        (base_seeds, site_score, None, "urban"),
        (
            _select_spaced_additions(
                coarse_gateway_score,
                coarse_gateway,
                count=gateway_count,
                minimum_distance=max(1.5, spacing * 0.58),
                occupied=occupied,
            ),
            gateway_site_score,
            "port",
            "urban",
        ),
        (
            _select_spaced_additions(
                coarse_coast_score,
                coarse_coast,
                count=coast_count,
                minimum_distance=max(2.0, spacing * 0.78),
                occupied=occupied,
            ),
            coast_site_score,
            None,
            "urban",
        ),
        (
            _select_spaced_additions(
                coarse_river_score,
                coarse_river,
                count=river_count,
                minimum_distance=max(2.0, spacing * 0.72),
                occupied=occupied,
            ),
            river_site_score,
            None,
            "urban",
        ),
        (
            _select_spaced_additions(
                coarse_lake_score,
                coarse_lake,
                count=lake_count,
                minimum_distance=max(2.0, spacing * 0.72),
                occupied=occupied,
            ),
            lake_site_score,
            None,
            "urban",
        ),
        (
            _select_spaced_additions(
                coarse_valley_score,
                coarse_valley_valid,
                count=valley_count,
                minimum_distance=max(2.0, spacing * 0.58),
                occupied=occupied,
            ),
            valley_site_score,
            None,
            "urban",
        ),
        (
            _select_spaced_additions(
                coarse_plain_score,
                coarse_plain_valid,
                count=plain_count,
                minimum_distance=max(1.2, spacing * 0.52),
                occupied=occupied,
            ),
            plain_site_score,
            "market",
            "plain-town",
        ),
    ]
    missing = target_count - sum(len(seeds) for seeds, _score, _type, _tier in seed_groups)
    if missing > 0:
        seed_groups.append(
            (
                _select_spaced_additions(
                    coarse_score,
                    coarse_valid,
                    count=missing,
                    minimum_distance=1.0,
                    occupied=occupied,
                ),
                site_score,
                None,
                "urban",
            )
        )

    resolved: list[tuple[int, int, float, str | None, str]] = []
    strategic_river_sites: set[tuple[int, int]] = set()
    strategic_coastal_sites: set[tuple[int, int]] = set()
    administrative_centres: set[tuple[int, int]] = set()
    for seeds, active_score, forced_type, category in seed_groups:
        for seed in seeds:
            row, column, score = _resolve_seed(
                seed,
                active_score,
                step=step,
                shape=grid.shape,
            )
            resolved.append((row, column, score, forced_type, category))

    # Important island components receive a small port even when their local
    # farming potential is below the global urban cut-off.
    if strategic_enabled:
        components, component_sizes = connected_components(coarse_land_valid)
        occupied_components = {
            int(components[row, column]) for row, column in occupied if components[row, column] > 0
        }
        maximum_island_size = max(8, int(coarse_valid.size * 0.018))
        island_candidates = sorted(
            (
                (size, identifier)
                for identifier, size in enumerate(component_sizes, start=1)
                if 3 <= size <= maximum_island_size and identifier not in occupied_components
            ),
            reverse=True,
        )[: min(42, max(16, target_count // 9))]
        island_score = (
            0.58 * population_relative
            + 0.42 * thematic.land_potential.astype(np.float64)
            + 0.18 * ocean_coast
        )
        coarse_island_score = reduce_field(
            np.where(ocean_coast & settlement_land, island_score, -np.inf),
            step=step,
            mode="max",
        )
        for _size, component_identifier in island_candidates:
            valid = (components == component_identifier) & coarse_coast & np.isfinite(coarse_island_score)
            if not np.any(valid):
                continue
            flat = np.flatnonzero(valid)
            best = flat[np.argmax(coarse_island_score.reshape(-1)[flat])]
            seed = tuple(int(value) for value in np.unravel_index(int(best), valid.shape))
            row, column, score = _resolve_seed(
                seed,
                np.where(ocean_coast & settlement_land, island_score, -np.inf),
                step=step,
                shape=grid.shape,
            )
            resolved.append((row, column, score, "island-port", "urban"))

        # Quotas favor the best gateways globally, but must not leave a second
        # major sea outlet completely unserved. Cover every materially narrow
        # gateway component with a coastal settlement unless one already lies
        # close enough to control the passage.
        gateway_components, gateway_sizes = connected_components(gateway_water)
        coverage_radius = max(8.0, step * 5.0)
        for component_identifier, component_size in enumerate(gateway_sizes, start=1):
            if component_size < 2:
                continue
            component_reach = _proximity(
                gateway_components == component_identifier,
                max(3, step * 2),
            )
            candidates = gateway_valid & (component_reach > 0.0)
            if not np.any(candidates):
                continue
            existing_ports = [
                (row, column)
                for row, column, _score, forced_type, category in resolved
                if category != "site"
                and (forced_type in {"port", "island-port"} or ocean_coast[row, column])
            ]
            candidate_rows, candidate_columns = np.nonzero(candidates)
            if existing_ports:
                existing = np.asarray(existing_ports, dtype=np.float64)
                row_delta = candidate_rows[:, None] - existing[None, :, 0]
                column_delta = np.abs(candidate_columns[:, None] - existing[None, :, 1])
                column_delta = np.minimum(column_delta, grid.shape[1] - column_delta)
                if float(np.hypot(row_delta, column_delta).min()) <= coverage_radius:
                    continue
            candidate_score = np.where(
                candidates,
                site_score + 0.26 * component_reach,
                -np.inf,
            )
            flat = int(np.argmax(candidate_score))
            row, column = np.unravel_index(flat, grid.shape)
            resolved.append(
                (int(row), int(column), float(candidate_score[row, column]), "port", "urban")
            )

        # Global coastal quotas can cluster in a handful of exceptional bays.
        # Audit the remaining physically usable shoreline so a long temperate
        # coast with freshwater and a viable hinterland cannot stay empty for
        # hundreds of kilometres.
        coastal_gap_budget = max(10, min(36, int(round(target_count * 0.06))))
        coastal_coverage_radius = max(18, min(34, min(grid.shape) // 28))
        minimum_coastal_gap = max(12, coastal_coverage_radius // 2)
        usable_coast = (
            ocean_coast
            & settlement_land
            & ~grid.snow
            & (elevation < 0.66)
            & (slope < 0.10)
            & (
                (thematic.land_potential >= 0.18)
                | (population_relative >= population_floor * 0.38)
                | (freshwater_access >= 0.26)
            )
        )
        for _addition in range(coastal_gap_budget):
            occupied_mask = np.zeros(grid.shape, dtype=bool)
            for row, column, _score, forced_type, category in resolved:
                if category != "site" and (
                    forced_type in {"port", "island-port"}
                    or ocean_coast[row, column]
                ):
                    occupied_mask[row, column] = True
            covered = _proximity(occupied_mask, coastal_coverage_radius) > 0.0
            uncovered = usable_coast & ~covered
            gap_components, gap_sizes = connected_components(uncovered)
            eligible_gaps = sorted(
                (
                    (size, identifier)
                    for identifier, size in enumerate(gap_sizes, start=1)
                    if size >= minimum_coastal_gap
                ),
                reverse=True,
            )
            if not eligible_gaps:
                break
            placed = False
            for _size, gap_identifier in eligible_gaps:
                gap_reach = _proximity(
                    gap_components == gap_identifier,
                    max(5, coastal_coverage_radius // 3),
                )
                candidates = usable_coast & ~covered & (gap_reach > 0.0)
                if not np.any(candidates):
                    continue
                candidate_score = np.where(
                    candidates,
                    site_score
                    + 0.24 * gap_reach
                    + 0.12 * freshwater_access
                    + 0.08 * climate_support,
                    -np.inf,
                )
                flat = int(np.argmax(candidate_score))
                row, column = np.unravel_index(flat, grid.shape)
                location = (int(row), int(column))
                resolved.append(
                    (
                        location[0],
                        location[1],
                        float(candidate_score[row, column]),
                        "port",
                        "urban",
                    )
                )
                strategic_coastal_sites.add(location)
                placed = True
                break
            if not placed:
                break

        # A quota can still concentrate on the globally best deltas and leave
        # hundreds of kilometres of another major river without a market node.
        # Audit the actual river network after all ordinary seeds are resolved
        # and insert one river town into every materially long uncovered reach.
        river_gap_budget = max(4, min(14, int(round(target_count * 0.025))))
        river_coverage_radius = max(24, min(44, min(grid.shape) // 24))
        minimum_gap_size = max(10, river_coverage_radius // 2)
        major_river = (grid.river_order >= 3) & land & ~grid.snow
        for _addition in range(river_gap_budget):
            occupied_mask = np.zeros(grid.shape, dtype=bool)
            for row, column, _score, _forced_type, category in resolved:
                if category != "site" and river_access[row, column] >= 0.56:
                    occupied_mask[row, column] = True
            covered = _proximity(occupied_mask, river_coverage_radius) > 0.0
            uncovered = major_river & ~covered
            gap_components, gap_sizes = connected_components(uncovered)
            eligible_gaps = sorted(
                (
                    (size, identifier)
                    for identifier, size in enumerate(gap_sizes, start=1)
                    if size >= minimum_gap_size
                ),
                reverse=True,
            )
            if not eligible_gaps:
                break
            placed = False
            for _size, gap_identifier in eligible_gaps:
                gap_reach = _proximity(
                    gap_components == gap_identifier,
                    max(5, river_coverage_radius // 3),
                )
                candidates = (
                    urban_valid
                    & river_corridor
                    & ~covered
                    & (gap_reach > 0.0)
                )
                if not np.any(candidates):
                    continue
                candidate_score = np.where(
                    candidates,
                    river_site_score
                    + 0.22 * gap_reach
                    + 0.06 * major_river_access,
                    -np.inf,
                )
                flat = int(np.argmax(candidate_score))
                row, column = np.unravel_index(flat, grid.shape)
                location = (int(row), int(column))
                resolved.append(
                    (
                        location[0],
                        location[1],
                        float(candidate_score[row, column]),
                        "river-city",
                        "urban",
                    )
                )
                strategic_river_sites.add(location)
                placed = True
                break
            if not placed:
                break

        # Province formation needs a mesh of genuine urban centres, not only
        # a large global total of settlements.  The ordinary quotas above can
        # still leave a broad, modestly productive interior without any city
        # because exceptional deltas and coasts win the world-wide ranking.
        # Audit the inhabitable land itself and add a city only where a
        # province-scale gap remains.  Ice, barren highlands and land with no
        # population support are deliberately outside this administrative
        # domain.
        administrative_domain = (
            settlement_land
            & ~grid.snow
            & (population.population_weight > 0.0)
            & (thematic.land_potential >= 0.08)
            & (thematic.habitability >= 0.14)
            & (elevation < 0.68)
            & (slope < 0.090)
        )
        administrative_valid = administrative_domain & (
            urban_valid
            | (freshwater_access >= 0.18)
            | ((annual >= 0.10) & (climate_support >= 0.34))
        )
        provisional_urban = sorted(
            (item for item in resolved if item[4] == "urban"),
            key=lambda item: (-item[2], item[0] * grid.shape[1] + item[1]),
        )
        provisional_metropolises = min(
            28,
            max(8, int(round(math.sqrt(len(provisional_urban)) * 1.45))),
        )
        provisional_city_limit = min(
            len(provisional_urban),
            provisional_metropolises
            + max(12, int(round(len(provisional_urban) * 0.30))),
        )
        administrative_centres.update(
            (row, column)
            for row, column, _score, _forced_type, _category
            in provisional_urban[:provisional_city_limit]
        )
        administrative_centres.update(strategic_river_sites)
        administrative_centres.update(strategic_coastal_sites)
        administrative_gap_budget = max(
            16,
            min(72, int(round(target_count * 0.14))),
        )
        administrative_coverage_radius = max(
            24,
            min(58, min(grid.shape) // 18),
        )
        minimum_administrative_gap = max(
            48,
            int(round(administrative_coverage_radius**2 * 0.16)),
        )
        all_settlement_mask = np.zeros(grid.shape, dtype=bool)
        for row, column, _score, _forced_type, category in resolved:
            if category != "site":
                all_settlement_mask[row, column] = True
        for _addition in range(administrative_gap_budget):
            centre_mask = np.zeros(grid.shape, dtype=bool)
            for row, column in administrative_centres:
                centre_mask[row, column] = True
            covered = _proximity(
                centre_mask,
                administrative_coverage_radius,
            ) > 0.0
            uncovered = administrative_domain & ~covered
            gap_components, gap_sizes = connected_components(uncovered)
            eligible_gaps = sorted(
                (
                    (size, identifier)
                    for identifier, size in enumerate(gap_sizes, start=1)
                    if size >= minimum_administrative_gap
                ),
                reverse=True,
            )
            if not eligible_gaps:
                break
            nearby_settlement = _proximity(all_settlement_mask, max(3, step * 2)) > 0.0
            placed = False
            for _size, gap_identifier in eligible_gaps:
                gap_reach = _proximity(
                    gap_components == gap_identifier,
                    max(6, administrative_coverage_radius // 3),
                )
                candidates = (
                    administrative_valid
                    & ~covered
                    & ~nearby_settlement
                    & (gap_reach > 0.0)
                )
                if not np.any(candidates):
                    candidates = (
                        administrative_valid
                        & ~covered
                        & (gap_reach > 0.0)
                    )
                if not np.any(candidates):
                    continue
                candidate_score = np.where(
                    candidates,
                    site_score
                    + 0.30 * gap_reach
                    + 0.14 * freshwater_access
                    + 0.10 * climate_support
                    + 0.08 * thematic.land_potential.astype(np.float64),
                    -np.inf,
                )
                flat = int(np.argmax(candidate_score))
                row, column = np.unravel_index(flat, grid.shape)
                location = (int(row), int(column))
                resolved.append(
                    (
                        location[0],
                        location[1],
                        float(candidate_score[row, column]),
                        None,
                        "urban",
                    )
                )
                administrative_centres.add(location)
                all_settlement_mask[location] = True
                placed = True
                break
            if not placed:
                break

    bank_resolved: list[tuple[int, int, float, str | None, str]] = []
    occupied_resolved: set[tuple[int, int]] = set()
    for row, column, score, forced_type, category in sorted(
        resolved,
        key=lambda item: (-item[2], item[0] * grid.shape[1] + item[1]),
    ):
        if int(grid.river_order[row, column]) > 0:
            bank_row, bank_column = _river_bank_location(
                row,
                column,
                land,
                grid.river_order,
                site_score,
                occupied=occupied_resolved,
            )
            if (bank_row, bank_column) != (row, column):
                row, column = bank_row, bank_column
                score = float(site_score[row, column])
                if forced_type is None:
                    forced_type = "river-city"
        if (row, column) in occupied_resolved:
            continue
        occupied_resolved.add((row, column))
        bank_resolved.append((row, column, score, forced_type, category))
    resolved = bank_resolved

    urban_resolved = sorted(
        (item for item in resolved if item[4] == "urban"),
        key=lambda item: (-item[2], item[0] * grid.shape[1] + item[1]),
    )
    plain_resolved = sorted(
        (item for item in resolved if item[4] == "plain-town"),
        key=lambda item: (-item[2], item[0] * grid.shape[1] + item[1]),
    )
    site_resolved = sorted(
        (item for item in resolved if item[4] == "site"),
        key=lambda item: (-item[2], item[0] * grid.shape[1] + item[1]),
    )
    resolved = urban_resolved + plain_resolved + site_resolved
    settlements: list[Settlement] = []
    metropolis_count = min(
        28,
        max(8, int(round(math.sqrt(len(urban_resolved)) * 1.45))),
    )
    city_limit = min(
        len(urban_resolved),
        metropolis_count + max(12, int(round(len(urban_resolved) * 0.30))),
    )
    for rank, (row, column, score, forced_type, category) in enumerate(resolved):
        if forced_type is not None:
            site_type = forced_type
        elif ocean_coast[row, column]:
            site_type = "port"
        elif lake_coast[row, column]:
            site_type = "lake-port"
        elif river_bank[row, column]:
            site_type = "river-city"
        elif annual[row, column] < 0.065:
            site_type = "oasis"
        else:
            site_type = "market"
        tier = (
            "site"
            if category == "site"
            else "town"
            if category == "plain-town"
            else "metropolis"
            if rank < metropolis_count
            else "city"
            if rank < city_limit
            else "town"
        )
        # A port controlling a strait or an inland-sea outlet is a regional
        # exchange and defence node, so it must rank above an ordinary town.
        if (
            tier == "town"
            and site_type in {"port", "island-port"}
            and gateway_reach[row, column] >= 0.32
        ):
            tier = "city"
        if tier == "town" and (row, column) in strategic_river_sites:
            tier = "city"
        if tier == "town" and (row, column) in strategic_coastal_sites:
            tier = "city"
        if tier == "town" and (row, column) in administrative_centres:
            tier = "city"
        radius = max(1, step * 2)
        rows = slice(max(0, row - radius), min(grid.shape[0], row + radius + 1))
        columns = (np.arange(column - radius, column + radius + 1) % grid.shape[1]).astype(int)
        local_weight = float(population.population_weight[rows][:, columns].sum())
        population_min = max(500, int(round(local_weight * population.population_min / 1000.0)) * 1000)
        population_max = max(
            population_min + 500,
            int(round(local_weight * population.population_max / 1000.0)) * 1000,
        )
        if tier == "site":
            population_min = min(population_min, 2_000)
            population_max = min(max(population_max, population_min + 500), 12_000)
        settlements.append(
            Settlement(
                identifier=f"settlement-{rank + 1:04d}",
                name=f"聚落{rank + 1:03d}",
                row=row,
                column=column,
                tier=tier,
                site_type=site_type,
                score=score,
                population_min=population_min,
                population_max=population_max,
            )
        )
    return tuple(settlements)
