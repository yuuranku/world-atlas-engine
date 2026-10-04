"""Re-form saved institutions on the same physical graph used by new worlds."""

from dataclasses import replace
import logging
import time

import numpy as np

from .administrations import synchronize_administrative_statistics
from .politics import (
    _allocate_state_territories,
    _dominant,
    _state_control_regions,
    _state_core_strengths,
    _state_formation_fields,
)
from .population import cell_areas_km2, population_density
from .provinces import (
    _ProvinceSeed,
    _administrative_system,
    _allocate_province_territories,
    _local_state_metrics,
    _province_growth_scale,
    _province_layers_from_specs,
)


def refresh_saved_institutional_administrations(grid, thematic, society):
    """Restore physical control while retaining every named institution and seat.

    This is the ordinary hinterland/physical-compartment/formation allocation
    with already existing institutions as its input. It neither chooses new
    capitals nor re-ranks, re-names, deletes, or renumbers saved identities.
    Existing frontier peoples and the controlled opportunity domain remain
    inputs; old cross-cultural province parents are corrected through actual
    regional and maritime links by the shared state allocator.
    """
    started = time.perf_counter()
    logger = logging.getLogger(__name__)
    by_identifier = {item.identifier: item for item in society.settlements}
    states = society.politics.states
    provinces = society.provinces.provinces
    if not states or not provinces:
        raise ValueError("saved institutional formation requires countries and provinces")
    cores = {state.identifier: by_identifier[state.core_settlement_id] for state in states}
    civilizations = {state.identifier: state.civilization_identifier for state in states}
    fields = _state_formation_fields(
        grid, thematic, society.population, society.cultures, society.transport,
    )
    logger.info("Building natural control graph for %s countries and %s provincial institutions",
                len(states), len(provinces))
    governed_land = np.asarray(society.politics.state_id) > 0
    required = {record.core_settlement_id for record in (*states, *provinces)}
    occupied = set()
    controls = []
    for city in sorted(society.settlements, key=lambda item: (-item.score, item.identifier)):
        cell = (city.row, city.column)
        if cell in occupied or not governed_land[cell]:
            continue
        if city.tier == "site" and city.identifier not in required:
            continue
        if int(society.cultures.civilization_id[cell]) <= 0:
            raise ValueError("saved institutional seat requires a real cultural control domain")
        occupied.add(cell)
        controls.append(city)
    controls = tuple(sorted(controls, key=lambda city: (
        int(society.cultures.civilization_id[city.row, city.column]), city.identifier,
    )))
    graph, control_labels = _state_control_regions(
        thematic, society.cultures, society.transport, fields, governed_land=governed_land,
        control_settlements=controls,
    )
    control_seconds = time.perf_counter() - started
    logger.info("Natural control graph complete in %.3f seconds: %s regions",
                control_seconds, graph.region_count)
    institutional_seats = {
        (by_identifier[record.core_settlement_id].row,
         by_identifier[record.core_settlement_id].column): record.state_identifier
        for record in provinces
    }
    allocation_started = time.perf_counter()
    state_id, frontier = _allocate_state_territories(
        grid, society.cultures, society.transport, society.settlements, fields,
        graph, control_labels, core_by_identifier=cores,
        civilization_by_identifier=civilizations,
        strength_by_identifier=_state_core_strengths(
            cores, civilizations, society.cultures, society.transport,
        ),
        institutional_seats=institutional_seats,
    )
    state_seconds = time.perf_counter() - allocation_started
    logger.info("Fixed-institution country formation complete in %.3f seconds", state_seconds)

    province_started = time.perf_counter()
    density = population_density(grid, society.population)
    areas = cell_areas_km2(grid)
    weights = society.population.population_weight
    midpoint = (society.population.population_min + society.population.population_max) / 2
    controlled = state_id > 0
    world_density = float(weights[controlled].sum() * midpoint / areas[controlled].sum())
    governments = {record.identifier: record for record in society.politics.government_forms}
    government_by_state = {
        record.country_identifier: governments[record.government_form_identifier]
        for record in society.politics.political_entities
    }
    specs = []
    for record in provinces:
        core = by_identifier[record.core_settlement_id]
        parent = int(state_id[core.row, core.column])
        if parent <= 0:
            raise ValueError("existing province seat lost its actual institutional control")
        local_density, _, _ = _local_state_metrics(
            core, parent, state_id, density, areas, society.transport.accessibility,
        )
        specs.append(_ProvinceSeed(
            identifier=record.identifier, state_identifier=parent, core=core,
            name=record.name, region_type=record.region_type,
            administrative_system=_administrative_system(government_by_state[parent].key),
            administrative_function=record.administrative_function,
            administrative_rank=record.administrative_rank,
            growth_scale=_province_growth_scale(
                local_density / max(world_density, 1e-12),
                record.administrative_function, record.administrative_rank,
            ),
        ))
    province_id = _allocate_province_territories(
        grid, thematic, society.population, society.settlements, society.transport,
        state_id, specs,
    )
    provincial_layers = _province_layers_from_specs(
        grid, society.population, state_id, province_id, specs,
    )
    political_layers = replace(
        society.politics, state_id=state_id, frontier=frontier,
        states=tuple(replace(record, language_identifier=_dominant(
            society.cultures.language_id[state_id == record.identifier],
        )) for record in states),
    )
    updated = replace(society, politics=political_layers, provinces=provincial_layers)
    updated = synchronize_administrative_statistics(
        grid, updated, state_id=state_id, province_id=province_id,
    )
    province_seconds = time.perf_counter() - province_started
    logger.info("Fixed-institution province formation and statistics complete in %.3f seconds",
                province_seconds)
    report = {
        "method": "fixed-institutions-natural-control-region-formation",
        "stateCellsChanged": int(np.count_nonzero(state_id != society.politics.state_id)),
        "provinceCellsChanged": int(np.count_nonzero(province_id != society.provinces.province_id)),
        "countries": len(states), "provinces": len(provinces),
        "provinceParentChanges": [
            {"provinceId": before.identifier, "previousCountryId": before.state_identifier,
             "countryId": after.state_identifier}
            for before, after in zip(provinces, updated.provinces.provinces, strict=True)
            if before.state_identifier != after.state_identifier
        ],
        "identitiesUnchanged": True, "seatsUnchanged": True,
        "physicalFieldsUnchanged": True, "populationWeightsUnchanged": True,
        "stageSeconds": {"physicalControlGraph": round(control_seconds, 3),
                         "stateFormation": round(state_seconds, 3),
                         "provinceFormation": round(province_seconds, 3)},
        "elapsedSeconds": round(time.perf_counter() - started, 3),
    }
    return updated, report
