"""Environment-led peoples for land outside stable state administration."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from ..model import WorldGrid
from ..thematic import (
    BIOME_BOREAL_FOREST,
    BIOME_DESERT_SCRUB,
    BIOME_SAVANNA_DRY_GRASSLAND,
    BIOME_TEMPERATE_GRASSLAND,
    BIOME_TEMPERATE_MIXED_FOREST,
    BIOME_TROPICAL_RAINFOREST,
    BIOME_TROPICAL_SEASONAL_FOREST,
    BIOME_TUNDRA_ALPINE,
    ThematicLayers,
    smooth_field,
)
from .model import CultureLayers, FrontierGroup, PopulationLayers
from .population import population_density
from .onomastics import lineage_roots, lineage_style_index
from .spatial import (
    connected_components,
    reduce_field,
    refine_partition_boundaries,
    select_spaced_seeds,
)


_SINITIC_EXONYM_ROOTS = {
    "north": ("北狄", "狄", "胡", "北胡"),
    "south": ("南蛮", "蛮", "百越", "南越"),
    "east": ("东夷", "夷", "海夷", "东夷"),
    "west": ("西羌", "西戎", "羌", "戎"),
}


def derive_stateless_frontier(
    land: np.ndarray,
    civilization_id: np.ndarray,
    biome_zone: np.ndarray,
    snow: np.ndarray,
    elevation: np.ndarray,
    annual_precipitation: np.ndarray,
    land_potential: np.ndarray,
    population_density: np.ndarray,
    cell_areas: np.ndarray,
    accessibility: np.ndarray,
    river_order: np.ndarray,
    transition_penalty: np.ndarray,
    protected: np.ndarray,
    *,
    target_share: float = 0.35,
) -> np.ndarray:
    """Reserve broad low-state-formation belts before countries expand.

    The result represents inhabited pastoral, tribal and otherwise weakly
    administered country, not an empty population mask.  A low-frequency
    state-formation score chooses the broad extent, while a narrow native-
    resolution regrowth pass places its edge on major rivers, drainage
    divides and ridges.  Dense settled belts and urban/transport anchors are
    never reserved as stateless land. ``target_share`` is a soft upper budget:
    geography may support less frontier, never forced clearance of farmland.
    """

    domain = np.asarray(land, dtype=bool)
    civilizations = np.asarray(civilization_id)
    biomes = np.asarray(biome_zone)
    snow_cover = np.asarray(snow, dtype=bool)
    height = np.asarray(elevation, dtype=np.float64)
    precipitation = np.asarray(annual_precipitation, dtype=np.float64)
    potential = np.asarray(land_potential, dtype=np.float64)
    density = np.asarray(population_density, dtype=np.float64)
    areas = np.asarray(cell_areas, dtype=np.float64)
    access = np.asarray(accessibility, dtype=np.float64)
    rivers = np.asarray(river_order)
    transitions = np.asarray(transition_penalty, dtype=np.float32)
    anchors = np.asarray(protected, dtype=bool)
    shape = domain.shape
    fields = (
        civilizations,
        biomes,
        snow_cover,
        height,
        precipitation,
        potential,
        density,
        areas,
        access,
        rivers,
        anchors,
    )
    if domain.ndim != 2 or any(field.shape != shape for field in fields):
        raise ValueError("stateless-frontier cell fields must align")
    if transitions.shape != (8, *shape):
        raise ValueError("stateless-frontier transitions must have shape (8, H, W)")
    if not np.isfinite(target_share) or not 0.0 <= target_share < 1.0:
        raise ValueError("target_share must lie in [0, 1)")
    land_count = int(np.count_nonzero(domain))
    target_count = int(round(land_count * float(target_share)))
    if land_count == 0 or target_count == 0:
        return np.zeros(shape, dtype=bool)

    if np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("frontier population density must be finite and non-negative")
    if np.any(~np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("frontier cell areas must be finite and positive")
    population_scale = density / max(float(density.max(initial=0.0)), 1.0e-12)
    access_scale = np.clip(access, 0.0, 1.0)
    potential_scale = np.clip(potential, 0.0, 1.0)
    relief = np.clip((height - 0.42) / 0.34, 0.0, 1.0)
    aridity = np.clip((0.10 - precipitation) / 0.10, 0.0, 1.0)
    steppe = np.isin(
        biomes,
        (BIOME_TEMPERATE_GRASSLAND, BIOME_SAVANNA_DRY_GRASSLAND),
    )
    cold_forest = biomes == BIOME_BOREAL_FOREST
    desert = biomes == BIOME_DESERT_SCRUB
    tundra = biomes == BIOME_TUNDRA_ALPINE
    riverine_support = np.clip(rivers.astype(np.float64) / 5.0, 0.0, 1.0)

    local_propensity = (
        0.94 * (1.0 - population_scale)
        + 0.76 * (1.0 - access_scale)
        + 0.56 * (1.0 - potential_scale)
        + 0.48 * steppe
        + 0.42 * desert
        + 0.32 * cold_forest
        + 0.58 * tundra
        + 0.42 * relief
        + 0.30 * aridity
        + 0.32 * snow_cover
        + 0.28 * (civilizations <= 0)
        - 0.16 * riverine_support
    )
    # Smooth numerator and support separately so seas do not depress coastal
    # hinterlands.  The scale is deliberately regional: it produces steppe,
    # plateau and forest belts rather than scattered low-population pixels.
    macro_radius = max(8, int(round(min(shape) / 54.0)))
    support = smooth_field(domain.astype(np.float64), radius=macro_radius, passes=2)
    macro = smooth_field(
        np.where(domain, local_propensity, 0.0),
        radius=macro_radius,
        passes=2,
    )
    macro = np.divide(macro, support, out=np.zeros_like(macro), where=support > 1.0e-6)
    frontier_score = 0.82 * macro + 0.18 * local_propensity
    frontier_score += 0.035 * np.max(transitions, axis=0)

    # A regional preference score cannot turn an inhabited agricultural belt
    # into stateless land merely because a requested area quota is still
    # unfilled. Compare density within this world, not against its single
    # busiest pixel, which can make ordinary productive countryside look empty.
    mean_density = float(np.average(density[domain], weights=areas[domain]))
    relative_density = density / max(mean_density, 1.0e-12)
    settled_belt = (relative_density >= 0.75) | (
        (relative_density >= 0.45)
        & ((access_scale >= 0.25) | (potential_scale >= 0.45))
    )
    eligible = domain & ~anchors & ~settled_belt
    target_count = min(target_count, int(np.count_nonzero(eligible)))
    frontier = np.zeros(shape, dtype=bool)
    remaining = target_count
    candidates = np.flatnonzero(eligible)
    if remaining and candidates.size:
        remaining = min(remaining, int(candidates.size))
        candidate_scores = frontier_score.reshape(-1)[candidates]
        if remaining == candidates.size:
            selected = candidates
        else:
            selected = candidates[
                np.argpartition(candidate_scores, candidate_scores.size - remaining)[
                    -remaining:
                ]
            ]
        frontier.reshape(-1)[selected] = True

    # Re-grow only the seam, then reapply the settlement constraints. Natural
    # boundary refinement must not expand a belt into protected countryside.
    labels = np.zeros(shape, dtype=np.int8)
    labels[domain] = 1
    labels[frontier] = 2
    labels = refine_partition_boundaries(
        labels,
        domain,
        transitions,
        {},
        friction=np.maximum(0.25, 1.0 + 0.18 * (1.0 - potential_scale)).astype(
            np.float32
        ),
        band_radius=9,
    )
    result = eligible & (labels == 2)

    # A low-frequency threshold can still leave tiny hilltop or coastal
    # islands of stateless color.  They reproduce the unexplained holes the
    # frontier model is meant to remove, so retain only regional-scale belts.
    # Removing specks is allowed to reduce the requested share.
    components, sizes = connected_components(result)
    minimum_component = max(
        32,
        min(513, max(32, target_count // 8)),
        int(round(land_count * 0.0004)),
    )
    for identifier, size in enumerate(sizes, start=1):
        if int(size) < minimum_component:
            result[components == identifier] = False

    return result.astype(bool)


def frontier_exonym_candidates(
    observer_family: str,
    *,
    direction: str,
    livelihood: str,
    organization: str,
    seed: int,
) -> tuple[str, ...]:
    """Return atlas exonyms from the nearest high civilization's lexicon.

    Frontier peoples still exist independently; this is the label a literate
    neighbouring atlas tradition applies to them.  The first naming style uses
    the directional 蛮／夷／羌／戎／狄 family, while every other generated
    lineage uses its own phonology and collective terms.
    """

    if direction not in {"north", "south", "east", "west"}:
        raise ValueError("frontier exonym direction is invalid")
    roots = lineage_roots(observer_family, count=64)
    style = lineage_style_index(observer_family)
    collectives = (
        ("汗", "汗庭")
        if organization == "khan-court"
        else ("部盟", "部落", "诸部")
        if organization == "confederacy"
        else ("部落", "诸部", "部")
    )
    result: list[str] = []
    if style == 0:
        exonyms = list(_SINITIC_EXONYM_ROOTS[direction])
        if livelihood == "highland":
            exonyms.sort(key=lambda item: ("羌" not in item, item))
        elif livelihood == "pastoral":
            exonyms.sort(key=lambda item: (not any(token in item for token in "狄胡戎"), item))
        elif livelihood == "maritime":
            exonyms.sort(key=lambda item: (not any(token in item for token in "夷越"), item))
        for offset in range(64):
            exonym = exonyms[(seed + offset) % len(exonyms)]
            root = roots[(seed * 7 + offset * 11) % len(roots)]
            if offset < len(exonyms):
                candidate = exonym
            else:
                candidate = root + collectives[offset % len(collectives)]
            if candidate not in result:
                result.append(candidate)
        return tuple(result)

    # Non-Sinitic observers describe outsiders with their own generated sound
    # inventory.  The atlas translates only the political collective (tribe,
    # league or khan), never the group's biome or livelihood into its name.
    for offset in range(48):
        root = roots[(seed + offset * 13) % len(roots)]
        candidate = root + collectives[offset % len(collectives)]
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


def _ocean_adjacent(grid: WorldGrid) -> np.ndarray:
    ocean = np.isin(grid.water, (1, 3))
    adjacent = np.roll(ocean, 1, axis=1) | np.roll(ocean, -1, axis=1)
    if grid.shape[0] > 1:
        adjacent[1:] |= ocean[:-1]
        adjacent[:-1] |= ocean[1:]
    return adjacent


def _window(values: np.ndarray, row: int, column: int, radius: int) -> np.ndarray:
    rows = np.arange(max(0, row - radius), min(values.shape[0], row + radius + 1))
    columns = np.arange(column - radius, column + radius + 1) % values.shape[1]
    return values[np.ix_(rows, columns)]


def _dominant_positive(values: np.ndarray) -> int:
    positive = np.asarray(values)[np.asarray(values) > 0]
    if positive.size == 0:
        return 0
    return int(np.bincount(positive.astype(np.int64)).argmax())


def _nearby_language(
    cultures: CultureLayers,
    row: int,
    column: int,
) -> int:
    direct = int(cultures.language_id[row, column])
    if direct > 0:
        return direct
    for radius in (12, 24, 48, 96, 192):
        identifier = _dominant_positive(
            _window(cultures.language_id, row, column, radius)
        )
        if identifier > 0:
            return identifier
    if not cultures.languages:
        raise ValueError("frontier-group generation requires at least one language")
    return cultures.languages[0].identifier


def _nearby_observer_civilization(
    cultures: CultureLayers,
    row: int,
    column: int,
) -> int:
    direct = int(cultures.civilization_id[row, column])
    if direct > 0 and int(cultures.civilization_influence[row, column]) >= 2:
        return direct
    for radius in (12, 24, 48, 96, 192):
        identifiers = _window(cultures.civilization_id, row, column, radius)
        influence = _window(cultures.civilization_influence, row, column, radius)
        strong = identifiers[(identifiers > 0) & (influence >= 2)]
        if strong.size:
            return int(np.bincount(strong.astype(np.int64)).argmax())
    positive = cultures.civilization_id[cultures.civilization_id > 0]
    if positive.size:
        return int(np.bincount(positive.astype(np.int64)).argmax())
    raise ValueError("frontier-group generation requires an observing civilization")


def _observer_language(
    cultures: CultureLayers,
    civilization_identifier: int,
    row: int,
    column: int,
) -> int:
    for radius in (12, 24, 48, 96, 192):
        civilization = _window(cultures.civilization_id, row, column, radius)
        languages = _window(cultures.language_id, row, column, radius)
        matching = languages[(civilization == civilization_identifier) & (languages > 0)]
        if matching.size:
            return int(np.bincount(matching.astype(np.int64)).argmax())
    for language in cultures.languages:
        if language.family_identifier == civilization_identifier:
            return language.identifier
    return _nearby_language(cultures, row, column)


def _observer_direction(
    cultures: CultureLayers,
    civilization_identifier: int,
    row: int,
    column: int,
) -> str:
    support = (
        (cultures.civilization_id == civilization_identifier)
        & (cultures.civilization_influence >= 2)
    )
    rows, columns = np.nonzero(support)
    if rows.size == 0:
        return "west"
    centre_row = float(np.mean(rows))
    angles = columns.astype(np.float64) / cultures.civilization_id.shape[1] * math.tau
    centre_angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    centre_column = centre_angle / math.tau * cultures.civilization_id.shape[1]
    column_delta = (
        column - centre_column + cultures.civilization_id.shape[1] / 2.0
    ) % cultures.civilization_id.shape[1] - cultures.civilization_id.shape[1] / 2.0
    row_delta = row - centre_row
    if abs(column_delta) >= abs(row_delta):
        return "east" if column_delta >= 0.0 else "west"
    return "south" if row_delta >= 0.0 else "north"


def _livelihood(
    grid: WorldGrid,
    thematic: ThematicLayers,
    ocean_adjacent: np.ndarray,
    row: int,
    column: int,
    radius: int,
) -> str:
    biome = _window(thematic.biome_zone, row, column, radius)
    elevation = _window(grid.elevation, row, column, radius)
    potential = _window(thematic.land_potential, row, column, radius)
    precipitation = _window(
        thematic.climate.annual_precipitation,
        row,
        column,
        radius,
    )
    river = _window(grid.river_order > 0, row, column, radius)
    coast = _window(ocean_adjacent, row, column, radius)
    steppe_share = float(
        np.mean(
            np.isin(
                biome,
                (BIOME_TEMPERATE_GRASSLAND, BIOME_SAVANNA_DRY_GRASSLAND),
            )
        )
    )
    desert_share = float(np.mean(biome == BIOME_DESERT_SCRUB))
    forest_share = float(
        np.mean(
            np.isin(
                biome,
                (
                    BIOME_BOREAL_FOREST,
                    BIOME_TEMPERATE_MIXED_FOREST,
                    BIOME_TROPICAL_RAINFOREST,
                    BIOME_TROPICAL_SEASONAL_FOREST,
                ),
            )
        )
    )
    cold_share = float(np.mean(biome == BIOME_TUNDRA_ALPINE))
    high_share = float(np.mean(elevation > 0.58))
    mean_potential = float(np.mean(potential))
    mean_precipitation = float(np.mean(precipitation))
    river_share = float(np.mean(river))
    if high_share >= 0.34 or float(grid.elevation[row, column]) > 0.67:
        return "highland"
    if (steppe_share + desert_share * 0.75) >= 0.40 and mean_precipitation < 0.16:
        return "pastoral"
    # A large sampling window can touch a distant coast even when its anchor
    # is an inland steppe or forest.  Maritime peoples require an actually
    # coastal anchor or a substantial coastal share, not a single coast cell.
    coast_share = float(np.mean(coast))
    if (
        bool(ocean_adjacent[row, column]) or coast_share >= 0.08
    ) and mean_potential < 0.50:
        return "maritime"
    if (forest_share + cold_share * 0.80) >= 0.34 and mean_potential < 0.50:
        return "foraging"
    if river_share >= 0.018 and mean_potential < 0.50:
        return "riverine"
    return "agropastoral"


def _organization(livelihood: str, *, major: bool, seed: int) -> str:
    choices = {
        "pastoral": (
            ("khan-court", "confederacy", "tribes")
            if major
            else ("confederacy", "tribes")
        ),
        "maritime": ("sea-clans", "boat-people", "island-clans"),
        "foraging": ("hunters", "forest-clans"),
        "agropastoral": ("tribes", "village-league"),
        "highland": ("hill-tribes", "clan-league", "village-league"),
        "riverine": ("river-clans", "fishers"),
    }[livelihood]
    return choices[seed % len(choices)]


def _anchor_in_block(
    frontier: np.ndarray,
    score: np.ndarray,
    coarse_row: int,
    coarse_column: int,
    step: int,
) -> tuple[int, int]:
    center_row = min(frontier.shape[0] - 1, coarse_row * step + step // 2)
    center_column = (coarse_column * step + step // 2) % frontier.shape[1]
    radius = max(2, step * 2)
    rows = np.arange(max(0, center_row - radius), min(frontier.shape[0], center_row + radius + 1))
    columns = np.arange(center_column - radius, center_column + radius + 1) % frontier.shape[1]
    local_frontier = frontier[np.ix_(rows, columns)]
    if not np.any(local_frontier):
        raise ValueError("coarse frontier seed has no full-resolution support")
    local_score = score[np.ix_(rows, columns)].copy()
    row_distance = rows[:, None] - center_row
    column_distance = np.abs(columns[None, :] - center_column)
    column_distance = np.minimum(column_distance, frontier.shape[1] - column_distance)
    local_score -= 0.002 * np.hypot(row_distance, column_distance)
    local_score[~local_frontier] = -np.inf
    local_index = int(np.argmax(local_score))
    local_row, local_column = np.unravel_index(local_index, local_score.shape)
    return int(rows[local_row]), int(columns[local_column])


def derive_frontier_groups(
    grid: WorldGrid,
    thematic: ThematicLayers,
    population: PopulationLayers,
    cultures: CultureLayers,
    frontier: np.ndarray,
) -> tuple[FrontierGroup, ...]:
    """Place named peoples in frontier land without drawing claimed ranges."""

    support = np.asarray(frontier, dtype=bool) & (grid.water == 0)
    if not np.any(support) or not cultures.languages:
        return ()
    step = 1
    coarse_support = reduce_field(support, step=step, mode="mean") >= 0.35
    density = smooth_field(coarse_support.astype(np.float64), radius=3, passes=1)
    ocean_adjacent = _ocean_adjacent(grid)
    coarse_potential = reduce_field(thematic.land_potential, step=step, mode="mean")
    density_field = population_density(grid, population)
    coarse_population = reduce_field(
        density_field,
        step=step,
        mode="mean",
    )
    maximum_population = float(np.max(coarse_population, initial=0.0))
    if maximum_population > 0.0:
        coarse_population = coarse_population / maximum_population
    coarse_river = reduce_field(grid.river_order > 0, step=step, mode="mean")
    coarse_coast = reduce_field(ocean_adjacent, step=step, mode="mean")
    coarse_steppe = reduce_field(
        np.isin(
            thematic.biome_zone,
            (BIOME_TEMPERATE_GRASSLAND, BIOME_SAVANNA_DRY_GRASSLAND),
        ),
        step=step,
        mode="mean",
    )
    coarse_snow = reduce_field(grid.snow, step=step, mode="mean")
    coarse_score = (
        1.35 * density
        + 0.46 * coarse_potential
        + 0.22 * coarse_population
        + 0.24 * coarse_river
        + 0.42 * coarse_steppe
        - 0.08 * coarse_coast
        - 0.28 * coarse_snow
    )
    valid = coarse_support & (density >= 0.30)
    target_count = int(np.clip(round(np.count_nonzero(support) / 7_000), 12, 36))
    target_count = min(target_count, int(np.count_nonzero(valid)))
    if target_count < 1:
        return ()
    minimum_distance = max(
        4.0,
        math.sqrt(float(np.count_nonzero(valid)) / target_count) * 0.52,
    )
    coarse_seeds = select_spaced_seeds(
        coarse_score,
        valid,
        count=target_count,
        minimum_distance=minimum_distance,
    )
    full_score = (
        thematic.land_potential.astype(np.float64)
        + 0.20 * (grid.river_order > 0)
        + 0.28
        * np.isin(
            thematic.biome_zone,
            (BIOME_TEMPERATE_GRASSLAND, BIOME_SAVANNA_DRY_GRASSLAND),
        )
        - 0.08 * ocean_adjacent
        + 0.10 * density_field / max(float(density_field.max(initial=0.0)), 1.0e-12)
        - 0.20 * grid.snow
    )
    anchors = [
        _anchor_in_block(support, full_score, row, column, step)
        for row, column in coarse_seeds
    ]
    ranked = sorted(
        range(len(anchors)),
        key=lambda index: (
            -float(coarse_score[coarse_seeds[index]]),
            anchors[index],
        ),
    )
    major_indices = set(ranked[: max(1, math.ceil(len(anchors) * 0.55))])
    language_by_identifier = {item.identifier: item for item in cultures.languages}
    civilization_by_identifier = {
        item.identifier: item for item in cultures.civilizations
    }
    used_names: set[str] = set()
    result: list[FrontierGroup] = []
    profile_radius = max(2, round(min(grid.shape) / 36))
    for index, (row, column) in enumerate(anchors):
        civilization_identifier = _nearby_observer_civilization(
            cultures,
            row,
            column,
        )
        language_identifier = _observer_language(
            cultures,
            civilization_identifier,
            row,
            column,
        )
        language = language_by_identifier[language_identifier]
        livelihood = _livelihood(
            grid,
            thematic,
            ocean_adjacent,
            row,
            column,
            profile_radius,
        )
        digest = hashlib.sha256(
            f"{language.name_family}:{row}:{column}:{livelihood}".encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:4], "big")
        major = index in major_indices
        organization = _organization(livelihood, major=major, seed=seed)
        observer = civilization_by_identifier[civilization_identifier]
        candidates = frontier_exonym_candidates(
            observer.name_family,
            direction=_observer_direction(
                cultures,
                civilization_identifier,
                row,
                column,
            ),
            livelihood=livelihood,
            organization=organization,
            seed=seed,
        )
        for candidate in candidates:
            if candidate not in used_names:
                used_names.add(candidate)
                break
        else:
            raise ValueError("frontier naming lineage cannot provide a unique name")
        result.append(
            FrontierGroup(
                identifier=f"frontier-group-{index + 1:02d}",
                name=candidate,
                row=row,
                column=column,
                livelihood=livelihood,
                organization=organization,
                civilization_identifier=civilization_identifier,
                language_identifier=language_identifier,
                tier="major" if major else "secondary",
            )
        )
    return tuple(result)
