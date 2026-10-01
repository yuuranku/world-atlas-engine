"""Civilization and language geography derived from population connectivity."""

from __future__ import annotations

import math

import numpy as np

from ..model import WorldGrid
from ..thematic import (
    BIOME_SAVANNA_DRY_GRASSLAND,
    BIOME_TEMPERATE_GRASSLAND,
    ThematicLayers,
)
from .model import (
    Civilization,
    CultureLayers,
    Language,
    NameLexicon,
    PopulationLayers,
    Settlement,
    TransportLayers,
)
from .onomastics import (
    lineage_civilization_candidates,
    lineage_key,
    lineage_language_candidates,
)
from .population import population_density
from .spatial import (
    allocate_regions_by_proximity,
    connected_components,
    fill_unreachable_components,
    physical_transition_penalties,
    reduce_field,
    refine_partition_boundaries,
    society_domain_mask,
)
from .territorial_simulation import (
    TerritorySeed,
    TerritorySimulation,
    bridge_transition_discounts,
    simulate_territories,
)


_EASTERN_COURT_EXPANSION_BONUS = 1.24


def _prosperous_eastern_index(
    coordinates: tuple[tuple[int, int], ...],
    prosperity: np.ndarray,
    *,
    width: int,
    landmass_sizes: tuple[int, ...] | None = None,
) -> int:
    """Choose the eastern naming hearth from generated economic geography.

    "Eastern" only restricts the candidate half of the projected world.  The
    actual hearth is the most fertile, populated and connected candidate, so
    the naming register follows this world's settlement ecology rather than a
    fixed coordinate or a bound political system.
    """

    values = np.asarray(prosperity, dtype=np.float64)
    if values.shape != (len(coordinates),):
        raise ValueError("prosperity must correspond to civilization coordinates")
    if not np.all(np.isfinite(values)):
        raise ValueError("civilization prosperity must be finite")
    if landmass_sizes is None:
        sizes = np.ones(len(coordinates), dtype=np.int64)
    else:
        sizes = np.asarray(landmass_sizes, dtype=np.int64)
        if sizes.shape != (len(coordinates),) or np.any(sizes < 1):
            raise ValueError("landmass_sizes must be positive and correspond to coordinates")
    eastern = [
        index
        for index, (_row, column) in enumerate(coordinates)
        if column >= width * 0.50
    ]
    candidates = eastern or list(range(len(coordinates)))
    if not candidates:
        raise ValueError("at least one civilization coordinate is required")
    # A prosperous entrepot on a tiny island may borrow the eastern naming
    # register, but it must not become the world's only mainland court culture.
    # Keep candidates on a substantial landmass whenever one is available.
    largest = max(int(sizes[index]) for index in candidates)
    continental = [
        index for index in candidates if int(sizes[index]) >= largest * 0.08
    ]
    candidates = continental or candidates
    return max(
        candidates,
        key=lambda index: (
            float(values[index]),
            -coordinates[index][0],
            -coordinates[index][1],
            -index,
        ),
    )


def _coarse_settlement_field(
    settlements: tuple[Settlement, ...],
    shape: tuple[int, int],
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    height = int(math.ceil(shape[0] / step))
    width = int(math.ceil(shape[1] / step))
    scores = np.full((height, width), -np.inf, dtype=np.float64)
    valid = np.zeros((height, width), dtype=bool)
    for settlement in settlements:
        row = min(height - 1, settlement.row // step)
        column = min(width - 1, settlement.column // step)
        tier_bonus = {
            "metropolis": 0.12,
            "city": 0.06,
            "town": 0.02,
            "site": -0.18,
        }.get(settlement.tier, 0.0)
        score = settlement.score + tier_bonus
        if not valid[row, column] or score > scores[row, column]:
            scores[row, column] = score
            valid[row, column] = True
    return scores, valid


def _civilization_seed_candidate_mask(
    land: np.ndarray,
    candidates: np.ndarray,
    *,
    civilization_count: int,
) -> np.ndarray:
    """Keep civilization cores off islands too small to hold a macro-culture."""

    coarse_land = np.asarray(land, dtype=bool)
    available = np.asarray(candidates, dtype=bool)
    if coarse_land.shape != available.shape:
        raise ValueError("land and civilization candidates must align")
    component, sizes = connected_components(coarse_land)
    minimum_size = max(
        8,
        int(math.ceil(np.count_nonzero(coarse_land) / max(1, civilization_count * 80))),
    )
    component_candidates = {
        identifier: int(np.count_nonzero(available & (component == identifier)))
        for identifier in range(1, len(sizes) + 1)
    }
    largest_size = max(sizes, default=0)
    substantial_size = max(minimum_size * 3, int(math.ceil(largest_size * 0.18)))
    substantial_components = {
        identifier
        for identifier, size in enumerate(sizes, start=1)
        if size >= substantial_size
    }
    nearshore_components: set[int] = set()
    if substantial_components:
        distance_to_mainland = _distance_from_mask(
            np.isin(component, tuple(sorted(substantial_components))),
            np.ones(coarse_land.shape, dtype=bool),
            maximum_distance=5,
        )
        for identifier, size in enumerate(sizes, start=1):
            if identifier in substantial_components or size >= substantial_size:
                continue
            region = component == identifier
            if region.any() and float(np.min(distance_to_mainland[region])) <= 5.0:
                nearshore_components.add(identifier)
    eligible = {
        identifier
        for identifier, size in enumerate(sizes, start=1)
        if size >= minimum_size
        and component_candidates[identifier] > 0
        and identifier not in nearshore_components
    }
    for identifier in sorted(
        range(1, len(sizes) + 1),
        key=lambda item: (
            item in nearshore_components,
            -sizes[item - 1],
            item,
        ),
    ):
        if sum(component_candidates[item] for item in eligible) >= civilization_count:
            break
        if component_candidates[identifier] > 0:
            eligible.add(identifier)
    return available & np.isin(component, tuple(sorted(eligible)))


def _consolidate_unjustified_mainland_enclaves(
    labels: np.ndarray,
    population_support: np.ndarray,
    transition_penalty: np.ndarray,
    *,
    protected_identifiers: set[int] | None = None,
    minimum_identifiers: int = 1,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Absorb a small same-land culture that has no defensible separation.

    Remote islands never touch another label and therefore remain eligible for
    their own civilization.  A small mainland pocket may touch several larger
    cultures; it survives only when none has a decisive, cheaply crossed shared
    frontier.  This removes seed-shaped enclaves without erasing defensible
    peninsulas or distinct island hearths.
    """

    source = np.asarray(labels, dtype=np.int32)
    support = np.asarray(population_support, dtype=np.float64)
    penalties = np.asarray(transition_penalty, dtype=np.float64)
    if source.shape != support.shape or penalties.shape != (8, *source.shape):
        raise ValueError("civilization consolidation fields must align")
    protected = set() if protected_identifiers is None else set(protected_identifiers)
    identifiers = [int(value) for value in np.unique(source) if value > 0]
    if isinstance(minimum_identifiers, bool) or int(minimum_identifiers) < 1:
        raise ValueError("minimum_identifiers must be a positive integer")
    minimum_identifiers = min(int(minimum_identifiers), len(identifiers))
    if len(identifiers) <= 1:
        return source.copy(), tuple(identifiers)
    areas = {
        identifier: int(np.count_nonzero(source == identifier))
        for identifier in identifiers
    }
    populations = {
        identifier: float(np.sum(support[source == identifier]))
        for identifier in identifiers
    }
    area_limit = max(10.0, float(np.median(tuple(areas.values()))) * 0.34)
    positive_populations = [value for value in populations.values() if value > 0.0]
    population_limit = (
        float(np.median(positive_populations)) * 0.55
        if positive_populations
        else 0.0
    )
    result = source.copy()
    surviving_count = len(identifiers)
    cardinal_directions = (
        (1, -1, 0),
        (3, 0, -1),
        (4, 0, 1),
        (6, 1, 0),
    )
    for identifier in sorted(identifiers, key=lambda item: (areas[item], item)):
        if surviving_count <= minimum_identifiers:
            break
        if identifier in protected:
            continue
        region = result == identifier
        area = int(np.count_nonzero(region))
        if area == 0 or area > area_limit:
            continue
        if populations[identifier] > population_limit:
            continue
        contacts: dict[int, int] = {}
        boundary_costs: dict[int, list[float]] = {}
        for direction, dy, dx in cardinal_directions:
            neighbor = np.roll(result, shift=(-dy, -dx), axis=(0, 1))
            boundary = region & (neighbor > 0) & (neighbor != identifier)
            if dy < 0:
                boundary[0, :] = False
            elif dy > 0:
                boundary[-1, :] = False
            for neighbor_identifier in np.unique(neighbor[boundary]):
                target = int(neighbor_identifier)
                target_boundary = boundary & (neighbor == target)
                contacts[target] = contacts.get(target, 0) + int(
                    np.count_nonzero(target_boundary)
                )
                boundary_costs.setdefault(target, []).extend(
                    penalties[direction][target_boundary].tolist()
                )
        if not contacts:
            continue
        total_contact = sum(contacts.values())
        target = min(
            contacts,
            key=lambda neighbor: (-contacts[neighbor], neighbor),
        )
        if contacts[target] / max(1, total_contact) < 0.55:
            continue
        costs = np.asarray(boundary_costs[target], dtype=np.float64)
        if costs.size == 0 or float(np.quantile(costs, 0.65)) >= 8.0:
            continue
        result[region] = target
        surviving_count -= 1

    kept = tuple(int(value) for value in np.unique(result) if value > 0)
    remap = np.zeros(max(kept, default=0) + 1, dtype=np.int32)
    for new_identifier, old_identifier in enumerate(kept, start=1):
        remap[old_identifier] = new_identifier
    positive = result > 0
    result[positive] = remap[result[positive]]
    return result, kept


def _civilization_component_quotas(
    land: np.ndarray,
    candidates: np.ndarray,
    population_support: np.ndarray,
    *,
    civilization_count: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Reserve civilization seeds for every materially inhabited landmass.

    Environmental specialization decides *which* family occupies a landmass,
    but it must not decide whether an entire populated continent receives a
    civilization at all. Quotas blend population, urban cores, and land area;
    every macro-landmass receives one seed before surplus seeds are assigned.
    """

    coarse_land = np.asarray(land, dtype=bool)
    available = np.asarray(candidates, dtype=bool)
    support = np.asarray(population_support, dtype=np.float64)
    if coarse_land.shape != available.shape or support.shape != coarse_land.shape:
        raise ValueError("civilization component fields must align")
    component, sizes = connected_components(coarse_land)
    identifiers = [
        identifier
        for identifier in range(1, len(sizes) + 1)
        if np.any(available & (component == identifier))
    ]
    if not identifiers:
        raise ValueError("civilization generation requires inhabited land components")
    candidate_counts = {
        identifier: int(np.count_nonzero(available & (component == identifier)))
        for identifier in identifiers
    }
    population_mass = {
        identifier: float(np.sum(support[component == identifier], dtype=np.float64))
        for identifier in identifiers
    }
    total_candidates = max(1, sum(candidate_counts.values()))
    total_population = max(1.0e-15, sum(population_mass.values()))
    total_area = max(1, sum(sizes[identifier - 1] for identifier in identifiers))
    weights = {
        identifier: (
            0.52 * population_mass[identifier] / total_population
            + 0.30 * candidate_counts[identifier] / total_candidates
            + 0.18 * sizes[identifier - 1] / total_area
        )
        for identifier in identifiers
    }
    ranked = sorted(identifiers, key=lambda item: (-weights[item], item))
    target = min(
        int(civilization_count),
        sum(candidate_counts[identifier] for identifier in identifiers),
    )
    # Each disconnected inhabited network needs its own hearth; otherwise one
    # civilization can appear on both sides of an uninhabited void. If callers
    # explicitly request fewer civilizations than networks, keep the strongest
    # networks and leave the remaining ones outside organized civilization.
    covered = set(ranked[: min(len(identifiers), target)])
    quotas = {identifier: 1 for identifier in covered}
    ideals = {identifier: target * weights[identifier] for identifier in identifiers}
    while sum(quotas.values()) < target:
        available_components = [
            identifier
            for identifier in identifiers
            if quotas.get(identifier, 0) < candidate_counts[identifier]
        ]
        if not available_components:
            break
        selected = max(
            available_components,
            key=lambda identifier: (
                ideals[identifier] - quotas.get(identifier, 0),
                weights[identifier],
                -identifier,
            ),
        )
        quotas[selected] = quotas.get(selected, 0) + 1
    return component, quotas


def _civilization_transmission_mask(
    land: np.ndarray,
    candidates: np.ndarray,
    population_support: np.ndarray,
    accessibility: np.ndarray,
    potential: np.ndarray,
    elevation: np.ndarray,
    river_order: np.ndarray,
    steppe: np.ndarray,
    coast: np.ndarray,
    snow: np.ndarray,
    habitability: np.ndarray,
) -> np.ndarray:
    """Return land reachable through continuous human and physical corridors.

    Civilizations radiate from settlement systems, not across every cell of a
    continent. Roads, populated plains, navigable river valleys and productive
    coasts form the transmission spine. Snowfields, high ridges and empty dry
    interiors cut that spine, leaving real frontier between distant cultures.
    """

    coarse_land = np.asarray(land, dtype=bool)
    hearths = np.asarray(candidates, dtype=bool)
    population = np.asarray(population_support, dtype=np.float64)
    access = np.asarray(accessibility, dtype=np.float64)
    suitability = np.asarray(potential, dtype=np.float64)
    heights = np.asarray(elevation, dtype=np.float64)
    rivers = np.asarray(river_order)
    grassland = np.asarray(steppe, dtype=np.float64)
    coastal = np.asarray(coast, dtype=np.float64)
    permanent_snow = np.asarray(snow, dtype=bool)
    residential_capacity = np.asarray(habitability, dtype=np.float64)
    fields = (
        hearths,
        population,
        access,
        suitability,
        heights,
        rivers,
        grassland,
        coastal,
        permanent_snow,
        residential_capacity,
    )
    if any(field.shape != coarse_land.shape for field in fields):
        raise ValueError("civilization transmission fields must align")
    if not np.any(hearths & coarse_land):
        raise ValueError("civilization transmission requires inhabited hearths")

    population_scale = population / max(float(population.max(initial=0.0)), 1.0e-15)
    access_scale = np.clip(access, 0.0, 1.0)
    navigable_valley = (
        (rivers >= 2)
        & (suitability >= 0.15)
        & ((population_scale >= 0.16) | (access_scale >= 0.035))
    )
    productive_belt = (suitability >= 0.30) & (population_scale >= 0.22)
    transmission_spine = coarse_land & (
        hearths
        | (population_scale >= 0.34)
        | (access_scale >= 0.10)
        | navigable_valley
        | productive_belt
    )

    # Ridges and true empty deserts are barriers, rather than merely expensive
    # cells that a continent-filling Voronoi pass will eventually cross.
    hard_barrier = permanent_snow | (heights >= 0.88) | (
        (heights >= 0.76)
        & (population_scale < 0.12)
        & (access_scale < 0.06)
    ) | (
        (residential_capacity < 0.20)
        & (suitability < 0.11)
        & (population_scale < 0.15)
        & (access_scale < 0.06)
    )
    traversable = coarse_land & ~hard_barrier
    traversable[hearths & coarse_land] = True
    transmission_spine &= traversable
    transmission_spine |= hearths & coarse_land

    # The fringe is deliberately finite. On the production lattice this is
    # roughly a few hundred kilometres: enough for rural hinterlands and local
    # diffusion, but too short to leap a broad uninhabited interior.
    fringe_radius = max(3, int(round(min(coarse_land.shape) / 30.0)))
    distance = _distance_from_mask(
        transmission_spine,
        traversable,
        maximum_distance=fringe_radius,
    )
    result = traversable & (distance <= fringe_radius)
    hearth_distance = _distance_from_mask(
        hearths,
        traversable,
        maximum_distance=int(math.ceil(fringe_radius * 1.9)),
    )
    # Sparse subsistence systems need more room than intensive agriculture.
    # They still require continuity back to a real hearth, so a remote biome
    # patch cannot become a cultural exclave by resemblance alone.
    result |= traversable & (grassland >= 0.46) & (
        hearth_distance <= fringe_radius * 1.9
    )
    result |= traversable & (coastal > 0.0) & (
        hearth_distance <= fringe_radius * 1.55
    )
    result |= hearths & coarse_land
    return result


def _wrapped_box_mean(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """Average a macroregion without wrapping across the poles."""

    source = np.asarray(values, dtype=np.float64)
    allowed = np.asarray(valid, dtype=bool)
    if source.shape != allowed.shape:
        raise ValueError("macroregion values and mask must align")
    radius = max(0, int(radius))
    if radius == 0:
        return np.where(allowed, source, 0.0)

    def window_sum(field: np.ndarray) -> np.ndarray:
        horizontal = np.pad(field, ((0, 0), (radius, radius)), mode="wrap")
        padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="constant")
        integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
        size = radius * 2 + 1
        return (
            integral[size:, size:]
            - integral[:-size, size:]
            - integral[size:, :-size]
            + integral[:-size, :-size]
        )

    numerator = window_sum(np.where(allowed, source, 0.0))
    denominator = window_sum(allowed.astype(np.float64))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def _distance_from_mask(
    source: np.ndarray,
    allowed: np.ndarray,
    *,
    maximum_distance: int,
) -> np.ndarray:
    """Limited four-neighbour distance on native cells, with wrapped longitude."""

    starts = np.asarray(source, dtype=bool)
    valid = np.asarray(allowed, dtype=bool)
    if starts.shape != valid.shape:
        raise ValueError("distance source and allowed mask must align")
    maximum_distance = int(maximum_distance)
    if maximum_distance < 0:
        raise ValueError("maximum_distance must be non-negative")
    distance = np.full(valid.shape, np.inf, dtype=np.float32)
    reached = starts & valid
    distance[reached] = 0.0
    frontier = reached.copy()
    for active_distance in range(1, maximum_distance + 1):
        padded = np.pad(frontier, ((1, 1), (0, 0)), mode="constant")
        adjacent = (
            padded[:-2]
            | padded[2:]
            | np.roll(frontier, 1, axis=1)
            | np.roll(frontier, -1, axis=1)
        )
        frontier = adjacent & valid & ~reached
        if not np.any(frontier):
            break
        distance[frontier] = float(active_distance)
        reached |= frontier
    return distance


def _select_civilization_seed_cells(
    candidate_score: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    minimum_distance: float,
    component: np.ndarray,
    component_quotas: dict[int, int],
) -> tuple[tuple[int, int], ...]:
    """Choose demographic-transport hearths without preassigned culture types."""

    scores = np.asarray(candidate_score, dtype=np.float64)
    valid = np.asarray(candidate_mask, dtype=bool)
    if scores.shape != valid.shape:
        raise ValueError("candidate scores and mask must align")
    component_labels = np.asarray(component, dtype=np.int32)
    quotas = {int(key): int(value) for key, value in component_quotas.items()}
    if component_labels.shape != scores.shape:
        raise ValueError("civilization component labels must align")
    candidate_values = scores[valid]
    if candidate_values.size < sum(quotas.values()):
        raise ValueError("not enough settlement candidates for civilization cores")
    width = scores.shape[1]
    selected: list[tuple[int, int]] = []
    for component_identifier in sorted(quotas):
        component_valid = valid & (component_labels == component_identifier)
        flat = np.flatnonzero(component_valid)
        order = flat[np.lexsort((flat, -scores.reshape(-1)[flat]))]
        local: list[tuple[int, int]] = []
        for _slot in range(quotas[component_identifier]):
            chosen: tuple[int, int] | None = None
            best_rank: tuple[float, float, int] | None = None
            for index in order:
                row, column = np.unravel_index(int(index), scores.shape)
                cell = (int(row), int(column))
                if cell in local:
                    continue
                if local:
                    separation = min(
                        math.hypot(
                            row - other_row,
                            min(abs(column - other_column), width - abs(column - other_column)),
                        )
                        for other_row, other_column in local
                    )
                else:
                    separation = minimum_distance
                spread = min(1.0, separation / max(minimum_distance, 1.0e-9))
                rank = (float(scores[row, column]) + 0.34 * spread, separation, -int(index))
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    chosen = cell
            if chosen is None:
                raise ValueError("cannot place every civilization core")
            local.append(chosen)
        selected.extend(local)
    return tuple(selected)


def _select_language_cores(
    candidates: list[Settlement],
    *,
    quota: int,
    region_cell_count: int,
    step: int,
    width: int,
) -> list[Settlement]:
    """Choose strong but geographically distinct language hearths."""

    if quota <= 0:
        return []
    coarse_width = int(math.ceil(width / step))
    minimum_distance = max(
        1.0,
        math.sqrt(max(region_cell_count, 1) / max(quota, 1)) * 0.36,
    )
    selected: list[Settlement] = []
    remaining = list(candidates)
    while remaining and len(selected) < quota:

        def rank(item: Settlement) -> tuple[float, float, str]:
            if not selected:
                separation = minimum_distance
            else:
                row = item.row // step
                column = item.column // step
                separation = min(
                    math.hypot(
                        row - other.row // step,
                        min(
                            abs(column - other.column // step),
                            coarse_width - abs(column - other.column // step),
                        ),
                    )
                    for other in selected
                )
            spread = min(1.0, separation / minimum_distance)
            return (0.72 * item.score + 0.28 * spread, separation, item.identifier)

        selected_item = max(remaining, key=rank)
        selected.append(selected_item)
        remaining.remove(selected_item)
    return selected


def _contact_mask(language_id: np.ndarray) -> np.ndarray:
    values = np.asarray(language_id)
    contact = np.zeros(values.shape, dtype=bool)
    east = np.roll(values, -1, axis=1)
    contact |= (values > 0) & (east > 0) & (values != east)
    contact |= np.roll(contact, 1, axis=1)
    if values.shape[0] > 1:
        south = values[1:]
        north = values[:-1]
        difference = (south > 0) & (north > 0) & (south != north)
        contact[1:] |= difference
        contact[:-1] |= difference
    return contact


def _core_settlement(
    region: np.ndarray,
    population_weight: np.ndarray,
    settlements: tuple[Settlement, ...],
) -> Settlement:
    """Choose a civilization core from its best central population hub.

    Family expansion seeds define where a civilization starts, but they are not
    necessarily where its mature demographic and transport core develops.  The
    capital-grade core is therefore selected after territorial allocation from
    the best available settlement tier near the population-weighted centroid.
    """

    regional = tuple(
        settlement
        for settlement in settlements
        if region[settlement.row, settlement.column]
    )
    if not regional:
        regional = tuple(settlements)
    if not regional:
        raise ValueError("culture generation requires at least one urban settlement")

    urban = tuple(item for item in regional if item.tier != "site")
    if not urban:
        urban = regional

    tier_rank = {"town": 1, "city": 2, "metropolis": 3}
    best_tier = max(tier_rank.get(item.tier, 0) for item in urban)
    candidates = tuple(item for item in urban if tier_rank.get(item.tier, 0) == best_tier)

    rows, columns = np.nonzero(region)
    weights = population_weight[region].astype(np.float64)
    if rows.size == 0:
        return min(candidates, key=lambda item: item.identifier)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(rows.shape, dtype=np.float64)
    weight_sum = float(weights.sum())
    row_center = float(np.dot(rows, weights) / weight_sum)
    angles = columns.astype(np.float64) * (2.0 * np.pi / region.shape[1])
    sine = float(np.dot(np.sin(angles), weights) / weight_sum)
    cosine = float(np.dot(np.cos(angles), weights) / weight_sum)
    column_center = (
        math.atan2(sine, cosine) % (2.0 * np.pi)
    ) * region.shape[1] / (2.0 * np.pi)
    region_scale = max(1.0, math.sqrt(rows.size / np.pi))

    scores = np.asarray([item.score for item in candidates], dtype=np.float64)
    populations = np.asarray([item.population_max for item in candidates], dtype=np.float64)

    def normalized(values: np.ndarray, index: int) -> float:
        minimum = float(values.min())
        maximum = float(values.max())
        if maximum <= minimum:
            return 1.0
        return float((values[index] - minimum) / (maximum - minimum))

    ranked: list[tuple[float, Settlement]] = []
    for index, settlement in enumerate(candidates):
        column_delta = abs(settlement.column - column_center)
        column_delta = min(column_delta, region.shape[1] - column_delta)
        distance = math.hypot(
            (settlement.row - row_center) / region_scale,
            column_delta / region_scale,
        )
        centrality = 1.0 / (1.0 + distance)
        utility = (
            0.70 * centrality
            + 0.16 * normalized(scores, index)
            + 0.14 * normalized(populations, index)
        )
        ranked.append((utility, settlement))
    return min(ranked, key=lambda item: (-item[0], item[1].identifier))[1]


def _quotas(counts: dict[int, int], target: int) -> dict[int, int]:
    identifiers = sorted(identifier for identifier, count in counts.items() if count > 0)
    target = min(target, sum(counts[identifier] for identifier in identifiers))
    result = {identifier: 1 for identifier in identifiers}
    total = max(1, sum(counts[identifier] for identifier in identifiers))
    ideals = {
        identifier: target * counts[identifier] / total for identifier in identifiers
    }
    while sum(result.values()) < target:
        available = [
            identifier
            for identifier in identifiers
            if result[identifier] < counts[identifier]
        ]
        if not available:
            break
        selected = max(
            available,
            key=lambda identifier: (
                ideals[identifier] - result[identifier],
                counts[identifier] - result[identifier],
                -identifier,
            ),
        )
        result[selected] += 1
    return result


def _language_name(
    lexicon: NameLexicon | None,
    family: str,
    identifier: int,
    used: set[str],
) -> str:
    del lexicon
    for candidate in lineage_language_candidates(family, identifier):
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError(f"name family cannot provide a unique language name: {family}")


def _subsistence_regime(profile: dict[str, float]) -> str:
    """Infer subsistence from evidence rather than a fixed culture template."""

    if profile.get("coast_reach", profile.get("coast_share", 0.0)) >= 0.58 and profile[
        "access"
    ] >= 0.11:
        return "maritime"
    if profile["steppe_share"] >= 0.52 and profile["precipitation"] <= 0.16:
        return "pastoral"
    water_reach = max(
        profile.get("river_reach", profile.get("river_share", 0.0)),
        profile.get("lake_share", 0.0) * 25.0,
    )
    if profile["potential"] <= 0.42 and water_reach >= 0.14:
        return "foraging"
    if profile["potential"] >= 0.34:
        return "agricultural"
    return "mixed"


def _civilization_archetype(profile: dict[str, float]) -> str:
    """Describe a generated region from its dominant material basis."""

    if profile.get("lake_share", 0.0) >= 0.006 and profile["potential"] < 0.45:
        return "河湖渔猎文明"
    regime = _subsistence_regime(profile)
    if regime == "maritime":
        return "海洋航贸文明"
    if regime == "pastoral":
        return "草原游牧文明"
    if regime == "foraging":
        return "河湖渔猎文明"

    if profile["lake_share"] >= 0.008 and profile["potential"] >= 0.30:
        return "湖盆农商文明"
    if profile["high_share"] >= 0.085:
        return "山地关隘文明"
    if (
        0.28 <= profile["steppe_share"] < 0.52
        and 0.30 <= profile["potential"] < 0.58
    ):
        return "半农半牧文明"
    if profile["river_share"] >= 0.045 and profile["potential"] >= 0.34:
        return "河谷农耕文明"
    if profile["potential"] >= 0.55:
        return "平原农耕文明"
    if profile["precipitation"] >= 0.15:
        return "森林河网文明"
    if profile["access"] >= 0.14:
        return "内陆商路文明"
    return "半农半牧文明" if regime == "mixed" else "丘陵农林文明"


def _civilization_name(
    lexicon: NameLexicon | None,
    family: str,
    identifier: int,
    used: set[str],
) -> str:
    del lexicon
    for candidate in lineage_civilization_candidates(family, identifier):
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError(f"name lineage cannot provide a unique civilization name: {family}")


def align_culture_names(
    cultures: CultureLayers,
    settlements: tuple[Settlement, ...],
) -> CultureLayers:
    """Name languages and civilizations from their actual core settlements.

    The lineage still controls local phonology, while the visible map labels
    share the same geographic root as the states grown from those settlements.
    """

    settlement_by_id = {item.identifier: item for item in settlements}

    def core_name(identifier: str) -> str:
        try:
            return settlement_by_id[identifier].name
        except KeyError as error:
            raise ValueError(
                f"culture core settlement is missing: {identifier}"
            ) from error

    civilizations = tuple(
        Civilization(
            identifier=item.identifier,
            name=f"{core_name(item.core_settlement_id)}文明圈",
            name_family=item.name_family,
            core_settlement_id=item.core_settlement_id,
            internal_diversity=item.internal_diversity,
        )
        for item in cultures.civilizations
    )
    languages = tuple(
        Language(
            identifier=item.identifier,
            name=f"{core_name(item.core_settlement_id)}语",
            name_family=item.name_family,
            family_identifier=item.family_identifier,
            core_settlement_id=item.core_settlement_id,
        )
        for item in cultures.languages
    )
    return CultureLayers(
        civilization_id=cultures.civilization_id,
        civilization_influence=cultures.civilization_influence,
        language_family_id=cultures.language_family_id,
        language_id=cultures.language_id,
        language_contact=cultures.language_contact,
        civilizations=civilizations,
        languages=languages,
    )


def assign_culture_lineages(
    cultures: CultureLayers,
    settlements: tuple[Settlement, ...],
    grid: WorldGrid,
    thematic: ThematicLayers,
) -> CultureLayers:
    """Assign naming grammars from each generated civilization's geography.

    These are language-shape choices, not civilization templates.  The
    retained eastern court register is reserved for the strongest eastern
    demographic core; other historic registers are matched to coast, steppe
    and highland evidence.  Every visible name inside one civilization then
    inherits the same lineage.
    """

    settlement_by_id = {item.identifier: item for item in settlements}
    civilization_by_id = {
        item.identifier: item for item in cultures.civilizations
    }
    core_by_civilization = {
        identifier: settlement_by_id[item.core_settlement_id]
        for identifier, item in civilization_by_id.items()
    }
    ocean = np.isin(grid.water, (1, 3))
    ocean_adjacent = np.roll(ocean, 1, axis=1) | np.roll(ocean, -1, axis=1)
    if grid.shape[0] > 1:
        ocean_adjacent[1:] |= ocean[:-1]
        ocean_adjacent[:-1] |= ocean[1:]
    profiles: dict[int, dict[str, float]] = {}
    land_components, land_component_sizes = connected_components(grid.water == 0)
    for identifier in civilization_by_id:
        region = cultures.civilization_id == identifier
        if not np.any(region):
            raise ValueError(f"civilization has no territory: {identifier}")
        profiles[identifier] = {
            "coast": float(np.mean(ocean_adjacent[region])),
            "steppe": float(
                np.mean(
                    np.isin(
                        thematic.biome_zone[region],
                        (BIOME_TEMPERATE_GRASSLAND, BIOME_SAVANNA_DRY_GRASSLAND),
                    )
                )
            ),
            "highland": float(np.mean((grid.elevation > 0.54)[region])),
            "wet": float(np.mean(thematic.climate.annual_precipitation[region])),
        }

    available = set(civilization_by_id)
    style_by_civilization: dict[int, int] = {}
    retained_eastern = [
        item.identifier
        for item in cultures.civilizations
        if item.name_family.startswith("lineage-s00-")
    ]
    if len(retained_eastern) == 1:
        eastern = retained_eastern[0]
    else:
        identifiers = tuple(sorted(civilization_by_id))
        population_values = np.asarray(
            [core_by_civilization[item].population_max for item in identifiers],
            dtype=np.float64,
        )
        score_values = np.asarray(
            [core_by_civilization[item].score for item in identifiers],
            dtype=np.float64,
        )

        def unit(values: np.ndarray) -> np.ndarray:
            low = float(values.min(initial=0.0))
            span = max(float(values.max(initial=1.0)) - low, 1.0e-12)
            return np.clip((values - low) / span, 0.0, 1.0)

        prosperity = np.asarray(
            [
                0.52
                * float(
                    thematic.land_potential[
                        core_by_civilization[item].row,
                        core_by_civilization[item].column,
                    ]
                )
                for item in identifiers
            ],
            dtype=np.float64,
        )
        prosperity += 0.30 * unit(population_values) + 0.18 * unit(score_values)
        eastern = identifiers[
            _prosperous_eastern_index(
                tuple(
                    (
                        core_by_civilization[item].row,
                        core_by_civilization[item].column,
                    )
                    for item in identifiers
                ),
                prosperity,
                width=grid.shape[1],
                landmass_sizes=tuple(
                    land_component_sizes[
                        int(
                            land_components[
                                core_by_civilization[item].row,
                                core_by_civilization[item].column,
                            ]
                        )
                        - 1
                    ]
                    for item in identifiers
                ),
            )
        ]
    style_by_civilization[eastern] = 0
    available.remove(eastern)

    register_roles = (
        (4, lambda identifier: profiles[identifier]["steppe"]),
        (2, lambda identifier: profiles[identifier]["coast"]),
        (5, lambda identifier: profiles[identifier]["highland"]),
        (
            9,
            lambda identifier: (
                0.46 * profiles[identifier]["coast"]
                + 0.34 * profiles[identifier]["highland"]
                + 0.20 * profiles[identifier]["wet"]
            ),
        ),
    )
    for style, score in register_roles:
        if not available:
            break
        selected = max(
            available,
            key=lambda identifier: (
                score(identifier),
                core_by_civilization[identifier].population_max,
                -identifier,
            ),
        )
        style_by_civilization[selected] = style
        available.remove(selected)

    remaining_styles = (1, 3, 6, 7, 8, 10, 11)
    for ordinal, identifier in enumerate(sorted(available)):
        style_by_civilization[identifier] = remaining_styles[
            ordinal % len(remaining_styles)
        ]
    family_by_civilization = {
        identifier: lineage_key(
            identifier,
            style_index=style_by_civilization[identifier],
        )
        for identifier in civilization_by_id
    }
    civilizations = tuple(
        Civilization(
            identifier=item.identifier,
            name=item.name,
            name_family=family_by_civilization[item.identifier],
            core_settlement_id=item.core_settlement_id,
            internal_diversity=item.internal_diversity,
        )
        for item in cultures.civilizations
    )
    languages = tuple(
        Language(
            identifier=item.identifier,
            name=item.name,
            name_family=family_by_civilization[item.family_identifier],
            family_identifier=item.family_identifier,
            core_settlement_id=item.core_settlement_id,
        )
        for item in cultures.languages
    )
    return CultureLayers(
        civilization_id=cultures.civilization_id,
        civilization_influence=cultures.civilization_influence,
        language_family_id=cultures.language_family_id,
        language_id=cultures.language_id,
        language_contact=cultures.language_contact,
        civilizations=civilizations,
        languages=languages,
    )


def derive_cultures(
    grid: WorldGrid,
    thematic: ThematicLayers,
    population: PopulationLayers,
    settlements: tuple[Settlement, ...],
    *,
    transport: TransportLayers | None = None,
    civilization_count: int | None = None,
    minimum_civilization_count: int | None = None,
    language_count: int | None = None,
    lexicon: NameLexicon | None = None,
) -> CultureLayers:
    """Derive distinct civilization and language partitions from shared access."""

    if not settlements:
        raise ValueError("culture generation requires settlements")
    step = 1
    physical_land = grid.water == 0
    land = society_domain_mask(grid)
    coarse_land = reduce_field(land, step=step, mode="max").astype(bool)
    elevation = reduce_field(grid.elevation, step=step, mode="max")
    potential = reduce_field(thematic.land_potential, step=step, mode="mean")
    population_support = reduce_field(population.population_weight, step=step, mode="max")
    density_support = reduce_field(population_density(grid, population), step=step, mode="max")
    river = reduce_field(grid.river_order > 0, step=step, mode="max").astype(bool)
    river_order = reduce_field(grid.river_order, step=step, mode="max")
    transition_penalty = physical_transition_penalties(
        elevation,
        river_order,
        land_mask=grid.water == 0,
    ) * 1.45
    snow = reduce_field(grid.snow, step=step, mode="max").astype(bool)
    habitability = reduce_field(
        thematic.habitability,
        step=step,
        mode="mean",
    )
    residential_constraint = 1.0 - np.clip(habitability, 0.0, 1.0)
    ocean = np.isin(grid.water, (1, 3))
    ocean_adjacent = np.roll(ocean, 1, axis=1) | np.roll(ocean, -1, axis=1)
    if grid.shape[0] > 1:
        ocean_adjacent[1:] |= ocean[:-1]
        ocean_adjacent[:-1] |= ocean[1:]
    candidate_score, candidate_mask = _coarse_settlement_field(
        settlements,
        grid.shape,
        step,
    )
    precipitation = reduce_field(
        thematic.climate.annual_precipitation,
        step=step,
        mode="mean",
    )
    temperature = reduce_field(
        thematic.climate.temperature,
        step=step,
        mode="mean",
    )
    steppe = reduce_field(
        np.isin(
            thematic.biome_zone,
            (BIOME_TEMPERATE_GRASSLAND, BIOME_SAVANNA_DRY_GRASSLAND),
        ),
        step=step,
        mode="mean",
    )
    coast = reduce_field(ocean_adjacent, step=step, mode="mean")
    river_density = reduce_field(grid.river_order > 0, step=step, mode="mean")
    accessibility = (
        reduce_field(transport.accessibility, step=step, mode="mean")
        if transport is not None
        else np.zeros(coarse_land.shape, dtype=np.float64)
    )
    urban_count = sum(settlement.tier != "site" for settlement in settlements)
    automatic_count = civilization_count is None
    if automatic_count:
        civilization_count = int(np.clip(round(math.sqrt(urban_count) * 0.98), 8, 22))
    civilization_count = int(civilization_count)
    if civilization_count < 1 or civilization_count > 24:
        raise ValueError("civilization_count must be between one and twenty-four")
    civilization_count = min(civilization_count, int(np.count_nonzero(candidate_mask)))
    if minimum_civilization_count is None:
        minimum_civilization_count = 1
    if (
        isinstance(minimum_civilization_count, bool)
        or int(minimum_civilization_count) < 1
        or int(minimum_civilization_count) > civilization_count
    ):
        raise ValueError(
            "minimum_civilization_count must be between one and civilization_count"
        )
    minimum_civilization_count = int(minimum_civilization_count)
    inhabited_candidate_mask = candidate_mask.copy()
    candidate_mask = _civilization_seed_candidate_mask(
        coarse_land,
        candidate_mask,
        civilization_count=civilization_count,
    )
    transmission_support = _civilization_transmission_mask(
        coarse_land,
        inhabited_candidate_mask,
        density_support,
        accessibility,
        potential,
        elevation,
        river_order,
        steppe,
        coast,
        snow,
        habitability,
    )
    transmission_component, _transmission_sizes = connected_components(
        transmission_support
    )
    inhabited_network_count = len(
        {
            int(identifier)
            for identifier in np.unique(transmission_component[candidate_mask])
            if identifier > 0
        }
    )
    if automatic_count:
        civilization_count = min(
            24,
            int(np.count_nonzero(candidate_mask)),
            max(civilization_count, inhabited_network_count),
        )
    component, component_quotas = _civilization_component_quotas(
        transmission_support,
        candidate_mask,
        population_support,
        civilization_count=civilization_count,
    )
    macro_radius = max(2, int(round(min(coarse_land.shape) / 28.0)))
    population_scale = density_support / max(
        float(density_support.max(initial=0.0)), 1.0e-15
    )
    candidate_values = candidate_score[candidate_mask]
    candidate_low = float(candidate_values.min())
    candidate_span = max(float(candidate_values.max()) - candidate_low, 1.0e-12)
    candidate_score = np.where(
        candidate_mask,
        (candidate_score - candidate_low) / candidate_span
        + 0.46 * population_scale
        + 0.38 * np.clip(accessibility, 0.0, 1.0),
        -np.inf,
    )
    spacing = max(
        1.0,
        math.sqrt(max(1, np.count_nonzero(candidate_mask)) / civilization_count) * 0.45,
    )
    civilization_seeds = _select_civilization_seed_cells(
        candidate_score,
        candidate_mask,
        minimum_distance=spacing,
        component=component,
        component_quotas=component_quotas,
    )
    coast_distance = _distance_from_mask(
        coast > 0.0,
        coarse_land,
        maximum_distance=80,
    )
    river_distance = _distance_from_mask(
        river_density > 0.0,
        coarse_land,
        maximum_distance=80,
    )
    coast_reach = np.exp(-np.minimum(coast_distance, 80.0) / 7.5)
    river_reach = np.exp(-np.minimum(river_distance, 80.0) / 6.0)
    coast_reach[~coarse_land] = 0.0
    river_reach[~coarse_land] = 0.0

    def normalized(field: np.ndarray) -> np.ndarray:
        values = np.asarray(field, dtype=np.float64)
        lower, upper = np.quantile(values[coarse_land], (0.04, 0.96))
        return np.clip((values - float(lower)) / max(float(upper - lower), 1.0e-12), 0.0, 1.0)

    environment = (
        normalized(potential),
        normalized(elevation),
        normalized(precipitation),
        normalized(temperature),
        np.clip(steppe, 0.0, 1.0),
        coast_reach,
        river_reach,
    )
    macro_environment = tuple(
        _wrapped_box_mean(field, coarse_land, radius=macro_radius)
        for field in environment
    )
    civilization_friction = (
        1.0
        + 6.2 * np.square(np.clip(elevation, 0.0, 1.0))
        + 2.6 * (1.0 - np.clip(potential, 0.0, 1.0))
        + 3.8 * residential_constraint
        + 30.0 * snow
        - 0.38 * river
    )
    seed_regimes: list[str] = []
    for seed in civilization_seeds:
        seed_profile = {
            "potential": float(macro_environment[0][seed]),
            "precipitation": float(macro_environment[2][seed]),
            "steppe_share": float(macro_environment[4][seed]),
            "coast_reach": float(macro_environment[5][seed]),
            "river_reach": float(macro_environment[6][seed]),
            "access": float(np.clip(accessibility[seed], 0.0, 1.0)),
        }
        regime = _subsistence_regime(seed_profile)
        seed_regimes.append(regime)
    seed_quality = np.asarray([candidate_score[seed] for seed in civilization_seeds])
    seed_low = float(seed_quality.min())
    seed_span = max(float(seed_quality.max()) - seed_low, 1.0e-12)
    regime_strength = {
        "agricultural": 1.06,
        "pastoral": 1.02,
        "maritime": 0.96,
        "mixed": 0.84,
        "foraging": 0.72,
    }
    seed_strengths = [
        float(
            np.clip(
                (0.68 + 0.82 * float((value - seed_low) / seed_span))
                * regime_strength[regime],
                0.62,
                1.55,
            )
        )
        for value, regime in zip(seed_quality, seed_regimes, strict=True)
    ]
    seed_prosperity = np.asarray(
        [
            0.42 * float(macro_environment[0][seed])
            + 0.24 * float(population_scale[seed])
            + 0.20 * float(np.clip(accessibility[seed], 0.0, 1.0))
            + 0.09 * float(macro_environment[6][seed])
            + 0.05 * float(macro_environment[2][seed])
            for seed in civilization_seeds
        ],
        dtype=np.float64,
    )
    coarse_land_components, coarse_land_sizes = connected_components(coarse_land)
    eastern_seed_index = _prosperous_eastern_index(
        tuple(civilization_seeds),
        seed_prosperity,
        width=coarse_land.shape[1],
        landmass_sizes=tuple(
            coarse_land_sizes[int(coarse_land_components[seed]) - 1]
            for seed in civilization_seeds
        ),
    )
    seed_strengths[eastern_seed_index] = float(
        min(
            1.82,
            seed_strengths[eastern_seed_index] * _EASTERN_COURT_EXPANSION_BONUS,
        )
    )
    bridge_edges = (
        bridge_transition_discounts(
            np.asarray(grid.river_order),
            transport.routes,
            transport.bridges,
        )
        if transport is not None
        else np.zeros((8, *coarse_land.shape), dtype=np.float32)
    )
    simulated_civilizations = simulate_territories(
        TerritorySimulation(
            valid=coarse_land,
            friction=civilization_friction.astype(np.float32),
            transition_penalty=transition_penalty.astype(np.float32),
            road_access=np.clip(accessibility, 0.0, 1.0).astype(np.float32),
            bridge_edges=bridge_edges,
        ),
        tuple(
            TerritorySeed(
                row=seed[0],
                column=seed[1],
                owner=index,
                strength=seed_strengths[index - 1],
            )
            for index, seed in enumerate(civilization_seeds, start=1)
        ),
    )
    civilization_labels = simulated_civilizations.owner
    civilization_costs = simulated_civilizations.cost
    eastern_identifier = eastern_seed_index + 1
    civilization_labels, kept_civilizations = (
        _consolidate_unjustified_mainland_enclaves(
            civilization_labels,
            population_support,
            transition_penalty,
            protected_identifiers={eastern_identifier},
            minimum_identifiers=minimum_civilization_count,
        )
    )
    civilization_seeds = tuple(
        civilization_seeds[identifier - 1]
        for identifier in kept_civilizations
    )
    eastern_seed_index = kept_civilizations.index(eastern_identifier)
    # Every surviving civilization owns a distinct language lineage. Style
    # zero marks only the prosperous eastern hearth selected above.
    family_by_civilization = {
        identifier: lineage_key(
            identifier,
            style_index=0 if identifier == eastern_seed_index + 1 else None,
        )
        for identifier in range(1, len(civilization_seeds) + 1)
    }
    civilization = civilization_labels.astype(np.int16)
    civilization[~physical_land] = -1
    civilization[physical_land & ~land] = 0
    civilization[land & (civilization < 0)] = 0
    settlement_cells = np.zeros(grid.shape, dtype=bool)
    for settlement in settlements:
        settlement_cells[settlement.row, settlement.column] = True
    civilization = refine_partition_boundaries(
        civilization,
        civilization > 0,
        transition_penalty * 2.2,
        {
            (settlement.row, settlement.column): int(
                civilization[settlement.row, settlement.column]
            )
            for settlement in settlements
            if civilization[settlement.row, settlement.column] > 0
        },
        band_radius=28,
    )
    cultural_frontier = land & grid.snow
    civilization[cultural_frontier & ~settlement_cells] = 0
    civilization_labels = civilization.astype(np.int32)
    coarse_influence = np.zeros(civilization_labels.shape, dtype=np.uint8)
    for identifier in range(1, len(civilization_seeds) + 1):
        region = civilization_labels == identifier
        finite_costs = civilization_costs[region & np.isfinite(civilization_costs)]
        if finite_costs.size == 0:
            continue
        inner, outer = np.quantile(finite_costs, (0.34, 0.72))
        coarse_influence[region] = 1
        coarse_influence[region & (civilization_costs <= outer)] = 2
        coarse_influence[region & (civilization_costs <= inner)] = 3
    civilization_influence = coarse_influence.astype(np.uint8)
    civilization_influence[(civilization <= 0) | ~land] = 0

    if language_count is None:
        language_count = int(
            np.clip(max(len(civilization_seeds) * 3, len(settlements) // 5), 24, 56)
        )
    language_friction = (
        1.0
        + 8.5 * np.square(np.clip(elevation, 0.0, 1.0))
        + 1.2 * (1.0 - np.clip(potential, 0.0, 1.0))
        + 2.8 * residential_constraint
        + 40.0 * snow
        - 0.42 * river
        - 0.30 * np.clip(accessibility, 0.0, 1.0)
        - 0.18 * np.clip(
            population_scale,
            0.0,
            1.0,
        )
    )
    settlement_groups: dict[int, list[Settlement]] = {}
    occupied_by_civilization: dict[int, set[tuple[int, int]]] = {}
    for settlement in sorted(settlements, key=lambda item: (-item.score, item.identifier)):
        identifier = int(civilization[settlement.row, settlement.column])
        if identifier <= 0:
            continue
        cell = (settlement.row // step, settlement.column // step)
        occupied = occupied_by_civilization.setdefault(identifier, set())
        if cell in occupied:
            continue
        occupied.add(cell)
        settlement_groups.setdefault(identifier, []).append(settlement)
    language_count = min(int(language_count), sum(map(len, settlement_groups.values())))
    language_quotas = _quotas(
        {identifier: len(values) for identifier, values in settlement_groups.items()},
        language_count,
    )
    language_labels = np.full(coarse_land.shape, -1, dtype=np.int32)
    language_core_by_identifier: dict[int, Settlement] = {}
    civilization_by_language: dict[int, int] = {}
    next_language = 1
    for civilization_identifier in sorted(language_quotas):
        cores = _select_language_cores(
            settlement_groups[civilization_identifier],
            quota=language_quotas[civilization_identifier],
            region_cell_count=int(
                np.count_nonzero(civilization_labels == civilization_identifier)
            ),
            step=step,
            width=grid.shape[1],
        )
        seeds = tuple((item.row // step, item.column // step) for item in cores)
        valid = coarse_land & (civilization_labels == civilization_identifier)
        local_labels, _costs = allocate_regions_by_proximity(
            language_friction,
            valid,
            seeds=seeds,
        )
        local_labels = fill_unreachable_components(local_labels, valid, seeds)
        for local_identifier, core in enumerate(cores, start=1):
            language_labels[local_labels == local_identifier] = next_language
            language_core_by_identifier[next_language] = core
            civilization_by_language[next_language] = civilization_identifier
            next_language += 1
    language = language_labels.astype(np.int16)
    language[~physical_land] = -1
    language[physical_land & ~land] = 0
    language[(civilization <= 0) & land] = 0
    language_parent = np.zeros(next_language, dtype=np.int16)
    for identifier, civilization_identifier in civilization_by_language.items():
        language_parent[identifier] = civilization_identifier
    language = refine_partition_boundaries(
        language,
        language > 0,
        transition_penalty,
        {
            (core.row, core.column): identifier
            for identifier, core in language_core_by_identifier.items()
        },
        owner_field=civilization,
        label_owner=language_parent,
        friction=language_friction,
        band_radius=14,
    )
    family_for_language = {
        identifier: civilization_by_language[identifier]
        for identifier in range(1, next_language)
    }
    family = np.zeros(grid.shape, dtype=np.int16)
    family[~physical_land] = -1
    family[physical_land & ~land] = 0
    for identifier, family_identifier in family_for_language.items():
        family[language == identifier] = family_identifier
    contact = _contact_mask(language) & land

    lake = grid.water == 2
    lake_adjacent = np.roll(lake, 1, axis=1) | np.roll(lake, -1, axis=1)
    if grid.shape[0] > 1:
        lake_adjacent[1:] |= lake[:-1]
        lake_adjacent[:-1] |= lake[1:]
    coast_reach_full = coast_reach
    river_reach_full = river_reach
    full_accessibility = (
        transport.accessibility
        if transport is not None
        else np.zeros(grid.shape, dtype=np.float32)
    )
    used_civilization_names: set[str] = set()
    civilizations: list[Civilization] = []
    for identifier, _seed in enumerate(civilization_seeds, start=1):
        region = civilization == identifier
        coast_share = float(np.mean(ocean_adjacent[region])) if np.any(region) else 0.0
        river_share = float(np.mean((grid.river_order > 0)[region])) if np.any(region) else 0.0
        high_share = float(np.mean((grid.elevation > 0.58)[region])) if np.any(region) else 0.0
        profile = {
            "coast_share": coast_share,
            "coast_reach": float(np.mean(coast_reach_full[region])) if np.any(region) else 0.0,
            "river_reach": float(np.mean(river_reach_full[region])) if np.any(region) else 0.0,
            "lake_share": float(np.mean(lake_adjacent[region])) if np.any(region) else 0.0,
            "river_share": river_share,
            "high_share": high_share,
            "steppe_share": float(
                np.mean(
                    np.isin(
                        thematic.biome_zone[region],
                        (BIOME_TEMPERATE_GRASSLAND, BIOME_SAVANNA_DRY_GRASSLAND),
                    )
                )
            ) if np.any(region) else 0.0,
            "potential": float(np.mean(thematic.land_potential[region])) if np.any(region) else 0.0,
            "precipitation": float(
                np.mean(thematic.climate.annual_precipitation[region])
            ) if np.any(region) else 0.0,
            "access": float(np.mean(full_accessibility[region])) if np.any(region) else 0.0,
        }
        archetype = _civilization_archetype(profile)
        diversity = (
            f"主导物质形态：{archetype}",
            "上游聚落与下游市场竞争" if river_share > 0.01 else "中心平原与边缘聚落竞争",
            "沿海港社与内陆腹地并存" if coast_share > 0.02 else (
                "高地传统与低地农耕并存" if high_share > 0.08 else "城市网络与乡村共同体并存"
            ),
        )
        core = _core_settlement(region, population.population_weight, settlements)
        name_family = family_by_civilization[identifier]
        civilizations.append(
            Civilization(
                identifier=identifier,
                name=_civilization_name(
                    lexicon,
                    name_family,
                    identifier,
                    used_civilization_names,
                ),
                name_family=name_family,
                core_settlement_id=core.identifier,
                internal_diversity=diversity,
            )
        )
        civilization_influence[core.row, core.column] = 3
    languages: list[Language] = []
    used_language_names: set[str] = set()
    for identifier in range(1, next_language):
        core = language_core_by_identifier[identifier]
        civilization_identifier = civilization_by_language[identifier]
        name_family = family_by_civilization[civilization_identifier]
        languages.append(
            Language(
                identifier=identifier,
                name=_language_name(
                    lexicon,
                    name_family,
                    identifier,
                    used_language_names,
                ),
                name_family=name_family,
                family_identifier=family_for_language[identifier],
                core_settlement_id=core.identifier,
            )
        )
    return CultureLayers(
        civilization_id=civilization,
        civilization_influence=civilization_influence,
        language_family_id=family,
        language_id=language,
        language_contact=contact.astype(bool),
        civilizations=tuple(civilizations),
        languages=tuple(languages),
    )
