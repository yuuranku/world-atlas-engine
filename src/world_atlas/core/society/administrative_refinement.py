"""Recompute existing administrative seams without regenerating identities.

Travel uses the current native terrain/route solver.  This is deliberately
not a claim that eight-direction Dijkstra is a continuous geodesic solver.
The correction removes invented basin walls and coordinate-wave borders,
then lets the saved cities compete within a finite band of existing claims.
Every accepted change is a digital-topology simple point for both the old
and new province, and, at a national border, both countries.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import ndimage

from ..model import WorldGrid
from ..suitability import relative_land_slope
from ..thematic import ThematicLayers
from .model import SocietyLayers
from .politics import (
    _settlement_protection_mask, _state_transition_penalties,
)
from .population import cell_areas_km2, population_density
from .provinces import _province_transition_penalties
from .spatial import categorical_components, physical_transition_penalties
from .territorial_simulation import (
    TerritorySeed, TerritorySimulation, bridge_transition_discounts, simulate_territories,
)
from .transport import road_network_fields


_RING = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _simple_point_table() -> np.ndarray:
    """4-connected foreground / 8-connected background simple-point test."""
    result = np.zeros(256, dtype=bool)
    for pattern in range(256):
        foreground = np.zeros((3, 3), dtype=bool)
        for bit, (dy, dx) in enumerate(_RING):
            foreground[dy + 1, dx + 1] = bool(pattern & (1 << bit))
        labels, _ = ndimage.label(foreground)
        adjacent = np.unique(labels[[0, 1, 1, 2], [1, 0, 2, 1]])
        adjacent = adjacent[adjacent > 0]
        background = ~foreground
        background[1, 1] = False
        _, background_count = ndimage.label(background, structure=np.ones((3, 3), dtype=bool))
        result[pattern] = len(adjacent) == 1 and background_count == 1
    result.setflags(write=False)
    return result


_SIMPLE = _simple_point_table()


def _pattern(labels: np.ndarray, row: int, column: int, owner: int) -> int:
    height, width = labels.shape
    pattern = 0
    for bit, (dy, dx) in enumerate(_RING):
        next_row = row + dy
        if 0 <= next_row < height and int(labels[next_row, (column + dx) % width]) == owner:
            pattern |= 1 << bit
    return pattern


def _boundary_band(labels: np.ndarray, radius: int) -> np.ndarray:
    positive = labels > 0
    boundary = positive & (np.roll(labels, -1, axis=1) > 0) & (labels != np.roll(labels, -1, axis=1))
    boundary |= np.roll(boundary, 1, axis=1)
    south = positive[:-1] & positive[1:] & (labels[:-1] != labels[1:])
    boundary[:-1] |= south
    boundary[1:] |= south
    return positive & ndimage.maximum_filter(boundary, size=2 * radius + 1, mode=("constant", "wrap"))


def _component_counts(labels: np.ndarray, maximum: int) -> list[int]:
    nodes, components, owners = categorical_components(labels)
    if not nodes.size:
        return [0] * (maximum + 1)
    _, first = np.unique(components, return_index=True)
    return np.bincount(owners[first], minlength=maximum + 1).astype(int).tolist()


def _supported_boundary_cells(labels: np.ndarray, physical: np.ndarray, river_order: np.ndarray) -> np.ndarray:
    """Hold existing borders where a real river bank or measured crest exists."""
    positive = labels > 0
    neighbor = np.roll(labels, -1, axis=1)
    east = positive & (neighbor > 0) & (labels != neighbor)
    east_support = np.maximum(physical[4], np.roll(physical[3], -1, axis=1)) >= 5.0
    east_support |= (river_order >= 2) ^ (np.roll(river_order, -1, axis=1) >= 2)
    east &= east_support
    result = east | np.roll(east, 1, axis=1)
    south = positive[:-1] & positive[1:] & (labels[:-1] != labels[1:])
    south_support = np.maximum(physical[6, :-1], physical[1, 1:]) >= 5.0
    south_support |= (river_order[:-1] >= 2) ^ (river_order[1:] >= 2)
    south &= south_support
    result[:-1] |= south
    result[1:] |= south
    return result


def _accept_topological_changes(
    provinces: np.ndarray,
    states: np.ndarray,
    candidate: np.ndarray,
    parent: np.ndarray,
    eligible: np.ndarray,
    travel_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply only source-directed moves with existing recipient contact.

    A source component can never lose its last cell.  The four-neighbour
    contact tests exclude diagonal-only possessions, cut bridges, merged
    exclaves and newly enclosed administrative holes.
    """
    result_provinces = provinces.copy()
    result_states = states.copy()
    indices = np.flatnonzero(eligible.ravel())
    order = np.lexsort((indices, travel_cost.ravel()[indices]))
    pending = indices[order].tolist()
    width = states.shape[1]
    accepted = 0
    while pending:
        deferred = []
        changed = False
        for index in pending:
            row, column = divmod(index, width)
            old = int(result_provinces[row, column])
            new = int(candidate[row, column])
            if old == new:
                continue
            if not (_SIMPLE[_pattern(result_provinces, row, column, old)]
                    and _SIMPLE[_pattern(result_provinces, row, column, new)]):
                deferred.append(index)
                continue
            old_state = int(result_states[row, column])
            new_state = int(parent[new])
            if old_state != new_state and not (
                _SIMPLE[_pattern(result_states, row, column, old_state)]
                and _SIMPLE[_pattern(result_states, row, column, new_state)]
            ):
                deferred.append(index)
                continue
            result_provinces[row, column] = new
            result_states[row, column] = new_state
            accepted += 1
            changed = True
        if not changed:
            break
        pending = deferred
    return result_provinces, result_states, accepted


def refine_saved_administrations(
    grid: WorldGrid,
    society: SocietyLayers,
    *,
    thematic: ThematicLayers,
) -> tuple[SocietyLayers, dict]:
    """Return a corrected saved society and audit; never write a checkpoint.

    Native physical fields, settlements, identities and frontier claims are
    unchanged.  Only cells within eight native cells of an existing national
    or provincial seam may change administrator.  This finite band bounds
    changes to the accepted world; it does not impose a target border length
    or manufacture bends on featureless plains.
    """
    original_states = np.asarray(society.politics.state_id)
    original_provinces = np.asarray(society.provinces.province_id)
    if grid.shape != original_states.shape:
        raise ValueError("saved society and terrain must align")
    land = np.asarray(grid.water) == 0
    controlled = original_states > 0
    parent = np.zeros(len(society.provinces.provinces) + 1, dtype=np.int16)
    for province in society.provinces.provinces:
        parent[province.identifier] = province.state_identifier
    cities = tuple(city for city in society.settlements if controlled[city.row, city.column])
    seeds = tuple(
        TerritorySeed(
            city.row, city.column, int(original_provinces[city.row, city.column]),
            domain=int(society.cultures.civilization_id[city.row, city.column]),
        )
        for city in cities
    )
    if not seeds:
        raise ValueError("saved administrations require existing governed settlements")
    roads, _ = road_network_fields(grid.shape, society.transport.routes)
    bridges = bridge_transition_discounts(grid.river_order, society.transport.routes, society.transport.bridges)
    elevation = np.asarray(grid.elevation, dtype=np.float64)
    density = population_density(grid, society.population)
    friction = np.maximum(
        0.25,
        1.0 + 7.2 * elevation ** 2
        + 18.0 * relative_land_slope(elevation, land)
        + 1.9 * (1.0 - np.clip(thematic.land_potential, 0.0, 1.0))
        + 2.6 * (1.0 - np.clip(thematic.habitability, 0.0, 1.0))
        - 0.08 * (grid.river_order > 0)
        - 0.22 * density / max(float(density.max(initial=0.0)), 1e-15)
        - 0.68 * np.clip(society.transport.accessibility, 0.0, 1.0),
    ).astype(np.float32)
    transitions = _state_transition_penalties(elevation, grid.river_order, society.cultures.language_id, roads, land_mask=land)
    control = simulate_territories(
        TerritorySimulation(
            controlled, friction, transitions, roads, bridges,
            owner_constraint=society.cultures.civilization_id,
        ),
        seeds,
    )
    candidate = control.owner
    national_band = _boundary_band(original_states, 8)
    protected = _settlement_protection_mask(society.settlements, grid.shape, radius=1)
    physical = physical_transition_penalties(elevation, grid.river_order, land_mask=land)
    supported_cells = _supported_boundary_cells(original_states, physical, grid.river_order)
    supported_cells |= _supported_boundary_cells(original_provinces, physical, grid.river_order)
    protected |= supported_cells
    del physical
    # Transport changes travel cost; it does not confer sovereignty on every
    # raster cell traversed by a road.  A road between two domestic cities may
    # pass through another country after a border is corrected.
    eligible = (
        national_band & ~protected & np.isfinite(control.cost) & (candidate > 0)
        & (parent[np.maximum(candidate, 0)] != original_states)
    )
    provinces, states, national_moves = _accept_topological_changes(
        original_provinces, original_states, candidate, parent, eligible, control.cost
    )
    del control, transitions
    transitions = _province_transition_penalties(grid, roads)
    provincial_seeds = tuple(
        replace(seed, domain=int(states[seed.row, seed.column])) for seed in seeds
    )
    control = simulate_territories(
        TerritorySimulation(controlled, friction, transitions, roads, bridges, owner_constraint=states),
        provincial_seeds,
    )
    eligible = (
        _boundary_band(provinces, 8) & ~protected & np.isfinite(control.cost)
        & (control.owner > 0) & (parent[np.maximum(control.owner, 0)] == states)
        & (control.owner != provinces)
    )
    provinces, states, provincial_moves = _accept_topological_changes(
        provinces, states, control.owner, parent, eligible, control.cost
    )
    before_states = _component_counts(original_states, len(society.politics.states))
    after_states = _component_counts(states, len(society.politics.states))
    before_provinces = _component_counts(original_provinces, len(society.provinces.provinces))
    after_provinces = _component_counts(provinces, len(society.provinces.provinces))
    if before_states != after_states or before_provinces != after_provinces:
        raise AssertionError("administrative refinement changed native component counts")
    if np.any(parent[provinces[controlled]] != states[controlled]):
        raise AssertionError("administrative refinement crossed a province parent")
    for city in society.settlements:
        cell = (city.row, city.column)
        if states[cell] != original_states[cell] or provinces[cell] != original_provinces[cell]:
            raise AssertionError("administrative refinement changed settlement ownership")
    weight = np.asarray(society.population.population_weight, dtype=np.float64)
    state_weight = np.bincount(np.maximum(states, 0).ravel(), weights=weight.ravel(), minlength=len(society.politics.states) + 1)
    new_states = []
    for record in society.politics.states:
        share = float(state_weight[record.identifier])
        lower = max(1000, round(share * society.population.population_min / 10000) * 10000)
        upper = max(lower + 1000, round(share * society.population.population_max / 10000) * 10000)
        new_states.append(replace(record, population_min=lower, population_max=upper))
    counts = np.bincount(np.maximum(provinces, 0).ravel(), minlength=len(parent))
    areas = cell_areas_km2(grid)
    average_density = float(weight[controlled].sum() / areas[controlled].sum())
    new_provinces = []
    for record in society.provinces.provinces:
        selected = provinces == record.identifier
        ratio = float(weight[selected].sum() / areas[selected].sum()) / max(average_density, 1e-15)
        new_provinces.append(replace(record, area_cells=int(counts[record.identifier]),
                                    population_density_class="dense" if ratio >= 1.35 else "settled" if ratio >= .62 else "sparse"))
    updated = replace(
        society,
        politics=replace(society.politics, state_id=states.astype(np.int16), states=tuple(new_states)),
        provinces=replace(society.provinces, province_id=provinces.astype(np.int32), provinces=tuple(new_provinces)),
    )
    report = dict(
        method="native-city-travel-cultural-domains-topology-preserved-v2",
        stateCellsChanged=int(np.count_nonzero(states != original_states)),
        provinceCellsChanged=int(np.count_nonzero(provinces != original_provinces)),
        acceptedNationalMoves=national_moves,
        acceptedProvincialMoves=provincial_moves,
        supportedBoundaryCellsProtected=int(np.count_nonzero(supported_cells)),
        transportInfluence="travel-cost-and-recorded-bridge-edges",
        settlementOwnershipChanges=0,
        stateComponentsBefore=before_states[1:], stateComponentsAfter=after_states[1:],
        provinceComponentsBefore=before_provinces[1:], provinceComponentsAfter=after_provinces[1:],
        physicalFieldsUnchanged=True, identitiesUnchanged=True,
        limitation="Travel still uses eight-direction native-grid Dijkstra; no continuous propagation is claimed.",
    )
    return updated, report
