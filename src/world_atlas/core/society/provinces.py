"""Nested administrative provinces derived after country formation."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..model import WorldGrid
from ..suitability import relative_land_slope
from ..thematic import ThematicLayers
from .model import (
    CultureLayers,
    PoliticalLayers,
    PopulationLayers,
    Province,
    ProvinceLayers,
    Settlement,
    TransportLayers,
)
from .administrative_topology import reconcile_partition_components
from .governance import control_region_reach
from .onomastics import lineage_roots, lineage_style_index
from .population import cell_areas_km2, population_density
from .politics import _settlement_protection_mask
from .spatial import (
    categorical_components,
    connected_components,
    natural_compartment_ids,
    physical_transition_penalties,
    refine_partition_boundaries,
    snap_partition_to_natural_regions,
)
from .state_formation import build_control_region_graph, simulate_state_formation
from .territorial_simulation import (
    TerritorySeed,
    TerritorySimulation,
    bridge_transition_discounts,
    simulate_territories,
)
from .transport import road_network_fields


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


def _wrapped_distance(
    first: tuple[int, int],
    second: tuple[int, int],
    width: int,
) -> float:
    column_delta = abs(first[1] - second[1])
    return math.hypot(
        first[0] - second[0],
        min(column_delta, width - column_delta),
    )


def _province_core_candidates(
    state_identifier: int,
    capital_identifier: str,
    settlements: Sequence[Settlement],
    state_id: np.ndarray,
) -> list[Settlement]:
    state_region = state_id == state_identifier
    component_id, component_sizes = connected_components(state_region)
    state_area = int(np.count_nonzero(state_region))
    minimum_component_size = min(128, max(16, int(round(state_area * 0.004))))

    def is_eligible(settlement: Settlement) -> bool:
        if settlement.identifier == capital_identifier:
            return True
        identifier = int(component_id[settlement.row, settlement.column])
        if settlement.tier != "site":
            return identifier > 0
        return identifier > 0 and component_sizes[identifier - 1] >= minimum_component_size

    return sorted(
        (
            settlement
            for settlement in settlements
            if (
                settlement.tier != "site"
                or settlement.site_type in {"fortress", "pass"}
            )
            and int(state_id[settlement.row, settlement.column]) == state_identifier
            and is_eligible(settlement)
        ),
        key=lambda settlement: (
            {"metropolis": 0, "city": 1, "town": 2, "site": 3}[
                settlement.tier
            ],
            -settlement.score,
            settlement.identifier,
        ),
    )


def _major_route_groups(
    settlements: Sequence[Settlement],
    transport: TransportLayers,
    state_id: np.ndarray,
) -> dict[str, int]:
    """Group settlements linked by a same-country trunk or regional route."""

    settlement_by_identifier = {item.identifier: item for item in settlements}
    parent = {item.identifier: item.identifier for item in settlements}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    for route in transport.routes:
        if route.importance == "local" or route.target_settlement_id is None:
            continue
        first = settlement_by_identifier.get(route.source_settlement_id)
        second = settlement_by_identifier.get(route.target_settlement_id)
        if first is None or second is None:
            continue
        first_state = int(state_id[first.row, first.column])
        if first_state <= 0 or int(state_id[second.row, second.column]) != first_state:
            continue
        union(first.identifier, second.identifier)

    members_by_root: dict[str, list[str]] = {}
    for identifier in parent:
        members_by_root.setdefault(find(identifier), []).append(identifier)
    result: dict[str, int] = {}
    next_group = 1
    for root in sorted(members_by_root):
        members = members_by_root[root]
        if len(members) < 2:
            continue
        for identifier in members:
            result[identifier] = next_group
        next_group += 1
    return result


def _select_province_cores(
    candidates: Sequence[Settlement],
    *,
    capital_identifier: str,
    count: int,
    width: int,
    route_group_by_identifier: Mapping[str, int] | None = None,
    required_identifiers: frozenset[str] = frozenset(),
    service_radius: float | None = None,
    maximum_count: int | None = None,
) -> tuple[Settlement, ...]:
    if not candidates or count <= 0:
        return ()
    if maximum_count is None:
        maximum_count = len(candidates)
    by_identifier = {item.identifier: item for item in candidates}
    first = by_identifier.get(capital_identifier, candidates[0])
    chosen = [first]
    remaining = [item for item in candidates if item.identifier != first.identifier]
    for identifier in sorted(required_identifiers):
        item = by_identifier.get(identifier)
        if item is not None and item in remaining and len(chosen) < maximum_count:
            chosen.append(item)
            remaining.remove(item)

    route_groups = route_group_by_identifier or {}

    def effective_distance(item: Settlement, other: Settlement) -> float:
        distance = _wrapped_distance(
            (item.row, item.column),
            (other.row, other.column),
            width,
        )
        item_group = route_groups.get(item.identifier)
        if item_group is not None and item_group == route_groups.get(other.identifier):
            # A continuous main road, navigable river or sea lane lets one
            # centre administer farther than straight-line geography alone.
            distance *= 0.55
        return distance

    while remaining and len(chosen) < maximum_count:
        maximum_score = max((item.score for item in candidates), default=1.0)
        scale = maximum_score if maximum_score > 0.0 else 1.0
        farthest_service_distance = max(
            min(effective_distance(item, other) for other in chosen)
            for item in remaining
        )
        if (
            len(chosen) >= count
            and service_radius is not None
            and farthest_service_distance <= service_radius
        ):
            break
        if len(chosen) >= count and service_radius is None:
            break
        candidate = max(
            remaining,
            key=lambda item: (
                min(
                    effective_distance(item, other)
                    for other in chosen
                )
                * (1.0 + 0.16 * item.score / scale)
                + {
                    "metropolis": 2.0,
                    "city": 1.0,
                    "town": 0.0,
                    "site": -1.0,
                }[item.tier],
                -item.row,
                -item.column,
            ),
        )
        chosen.append(candidate)
        remaining.remove(candidate)
    return tuple(chosen)


def _province_transition_penalties(
    grid: WorldGrid,
    road_corridor: np.ndarray,
) -> np.ndarray:
    """Make administrative travel follow real landforms and drainage.

    Crossings depend on measured terrain and mapped rivers.  A D8 outlet
    label or the absence of a direct flow pointer is not a mountain barrier.
    """

    elevation = np.asarray(grid.elevation, dtype=np.float64)
    river_order = np.asarray(grid.river_order)
    land = grid.water == 0
    roads = np.asarray(road_corridor, dtype=np.float64)
    if roads.shape != grid.shape:
        raise ValueError("province road corridors must match the WorldGrid shape")
    physical = physical_transition_penalties(
        elevation,
        river_order,
        land_mask=land,
    )
    terrain_scale = 4.05
    river_scale = 3.10
    penalties = terrain_scale * physical
    height, _width = grid.shape
    for direction, (dy, dx, _distance) in enumerate(_NEIGHBORS):
        shift = (-dy, -dx)
        target_order = np.roll(river_order, shift=shift, axis=(0, 1))
        stream_bank = (river_order > 0) ^ (target_order > 0)
        stream_strength = np.maximum(river_order, target_order).astype(np.float64)
        atlas_major_bank = (river_order >= 3) ^ (target_order >= 3)
        atlas_secondary_bank = (river_order >= 2) ^ (target_order >= 2)
        physical_river = (
            atlas_major_bank
            * (5.0 + 2.8 * np.clip(stream_strength - 2.0, 0.0, 3.0))
            + atlas_secondary_bank * 1.8
        )
        penalties[direction] -= (terrain_scale - river_scale) * physical_river
        target_road = np.roll(roads, shift=shift, axis=(0, 1))
        crossing_road = np.minimum(roads, target_road)
        penalties[direction] += (
            stream_bank * (3.0 + 2.4 * np.clip(stream_strength, 0.0, 4.0))
        )
        # An authored road is evidence of an administrable corridor or a real
        # pass.  It softens an ordinary ridge/valley boundary, but does not
        # erase the defensive value of a mapped river bank.
        penalties[direction] *= np.where(
            stream_bank,
            1.0 - 0.10 * crossing_road,
            1.0 - 0.15 * crossing_road,
        )
        target_land = np.roll(land, shift=shift, axis=(0, 1))
        penalties[direction, ~(land & target_land)] = 0.0
        if dy < 0:
            penalties[direction, 0, :] = 0.0
        elif dy > 0:
            penalties[direction, -1, :] = 0.0
    return penalties


def _repair_inland_province_fragments(
    labels: np.ndarray,
    state_id: np.ndarray,
    province_specs: Sequence[_ProvinceSeed],
) -> np.ndarray:
    """Repair provincial fragments after every merge, preserving island seats."""
    states = np.asarray(state_id)
    nodes, components, _owners = categorical_components(states)
    state_components = np.zeros(states.shape, dtype=np.int32)
    state_components.ravel()[nodes] = components + 1
    domains = np.zeros(max(spec.identifier for spec in province_specs) + 1, dtype=np.int32)
    seats = {}
    for spec in province_specs:
        domains[spec.identifier] = spec.state_identifier
        seats[spec.identifier] = (spec.core.row, spec.core.column)
    return reconcile_partition_components(labels, seats, domains, state_components,
                                          prefer_local_seats=True)


def _assign_unseeded_province_islands(
    labels: np.ndarray,
    state_id: np.ndarray,
    province_specs: Sequence[_ProvinceSeed],
    *,
    settlements: Sequence[Settlement] = (),
    routes=(),
) -> np.ndarray:
    """Give an already-owned unseeded island its nearest same-country seat.

    Unlike state formation this does not create a territorial claim: the
    country's sea-route evidence has already established ownership. IDs carry
    no spatial meaning, so attachment uses spherical distance to the island.
    """
    result = np.asarray(labels).copy()
    states = np.asarray(state_id)
    nodes, components, owners = categorical_components(states)
    if not nodes.size:
        return result
    height, width = states.shape
    order = np.argsort(components, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(components[order])) + 1, len(order)]
    settlement_by_id = {item.identifier: item for item in settlements}
    route_cells = []
    for route in routes:
        if route.mode != "sea" or route.target_settlement_id is None:
            continue
        first = settlement_by_id.get(route.source_settlement_id)
        second = settlement_by_id.get(route.target_settlement_id)
        if first is not None and second is not None:
            a, b = (first.row, first.column), (second.row, second.column)
            if int(states[a]) > 0 and states[a] == states[b]:
                route_cells.append((a, b))
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        local = order[start:stop]
        cells = nodes[local]
        values = result.ravel()[cells]
        if np.all(values > 0):
            continue
        if np.any(values > 0):
            raise ValueError("a seeded provincial land component was not fully reached")
        state = int(owners[local[0]])
        candidates = [spec for spec in province_specs if spec.state_identifier == state]
        rows, columns = np.divmod(cells, width)
        latitude = np.pi * (0.5 - (rows + 0.5) / height)
        longitude = 2.0 * np.pi * (columns + 0.5) / width
        centre = np.array([np.mean(np.cos(latitude) * np.cos(longitude)),
                           np.mean(np.cos(latitude) * np.sin(longitude)),
                           np.mean(np.sin(latitude))])

        def distance_key(spec):
            lat = np.pi * (0.5 - (spec.core.row + 0.5) / height)
            lon = 2.0 * np.pi * (spec.core.column + 0.5) / width
            vector = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
            return (-float(centre @ vector), spec.identifier)

        sea_connected = set()
        component_cells = set(int(cell) for cell in cells)
        for a, b in route_cells:
            for source, destination in ((a, b), (b, a)):
                if (destination[0] * width + destination[1] in component_cells
                        and int(result[source]) > 0):
                    sea_connected.add(int(result[source]))
        maritime_candidates = [spec for spec in candidates if spec.identifier in sea_connected]
        target = min(maritime_candidates or candidates, key=distance_key)
        result.ravel()[cells] = target.identifier
    return result


def _consolidate_tiny_provinces(
    labels: np.ndarray,
    province_state: np.ndarray,
    *,
    protected_identifiers: frozenset[int],
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Absorb administrative specks without deleting real island districts."""

    source = np.asarray(labels, dtype=np.int32)
    state_by_province = np.asarray(province_state, dtype=np.int16)
    if source.ndim != 2:
        raise ValueError("province consolidation expects two-dimensional labels")
    identifiers = tuple(int(value) for value in np.unique(source) if value > 0)
    if not identifiers:
        return source.copy(), ()
    if max(identifiers) >= len(state_by_province):
        raise ValueError("province-state lookup does not cover labels")
    result = source.copy()
    areas = {
        identifier: int(np.count_nonzero(result == identifier))
        for identifier in identifiers
    }
    by_state: dict[int, list[int]] = {}
    for identifier in identifiers:
        by_state.setdefault(int(state_by_province[identifier]), []).append(identifier)
    thresholds: dict[int, int] = {}
    for state_identifier, local_identifiers in by_state.items():
        if len(local_identifiers) <= 1:
            thresholds[state_identifier] = 0
            continue
        reference = float(np.median([areas[item] for item in local_identifiers]))
        thresholds[state_identifier] = max(12, min(96, int(round(reference * 0.04))))

    for identifier in sorted(identifiers, key=lambda item: (areas[item], item)):
        if identifier in protected_identifiers:
            continue
        state_identifier = int(state_by_province[identifier])
        region = result == identifier
        area = int(np.count_nonzero(region))
        if area == 0 or area >= thresholds[state_identifier]:
            continue
        contacts: Counter[int] = Counter()
        for neighbor in (
            np.roll(result, 1, axis=1),
            np.roll(result, -1, axis=1),
            np.vstack((np.zeros((1, result.shape[1]), dtype=result.dtype), result[:-1])),
            np.vstack((result[1:], np.zeros((1, result.shape[1]), dtype=result.dtype))),
        ):
            boundary = region & (neighbor > 0) & (neighbor != identifier)
            for target in np.unique(neighbor[boundary]):
                target_identifier = int(target)
                if int(state_by_province[target_identifier]) == state_identifier:
                    contacts[target_identifier] += int(
                        np.count_nonzero(boundary & (neighbor == target_identifier))
                    )
        if not contacts:
            # A water-separated islet can be a tiny but coherent dependency.
            continue
        target = min(
            contacts,
            key=lambda item: (-contacts[item], -areas.get(item, 0), item),
        )
        result[region] = target
        areas[target] = areas.get(target, 0) + area
        areas[identifier] = 0

    kept = tuple(int(value) for value in np.unique(result) if value > 0)
    remap = np.zeros(max(kept, default=0) + 1, dtype=np.int32)
    for new_identifier, old_identifier in enumerate(kept, start=1):
        remap[old_identifier] = new_identifier
    positive = result > 0
    result[positive] = remap[result[positive]]
    return result, kept


def _province_region_type(
    settlement: Settlement,
    grid: WorldGrid,
    thematic: ThematicLayers,
) -> str:
    if settlement.site_type == "island-port":
        region_type = "archipelago"
    elif settlement.site_type == "port":
        region_type = "coastal"
    elif settlement.site_type == "lake-port":
        region_type = "lake"
    elif settlement.site_type == "river-city":
        region_type = "river"
    elif settlement.site_type == "oasis":
        region_type = "oasis"
    elif float(grid.elevation[settlement.row, settlement.column]) >= 0.48:
        region_type = "highland"
    elif float(thematic.land_potential[settlement.row, settlement.column]) >= 0.52:
        region_type = "plain"
    else:
        region_type = "interior"
    return region_type


@dataclass(frozen=True, slots=True)
class _ProvinceSeed:
    identifier: int
    state_identifier: int
    core: Settlement
    name: str
    region_type: str
    administrative_system: str
    administrative_function: str
    administrative_rank: str
    growth_scale: float


def _administrative_system(government_key: str) -> str:
    if government_key in {"court-monarchy", "bureaucratic-monarchy"}:
        return "central-bureaucracy"
    if government_key == "hereditary-feudalism":
        return "feudal-vassalage"
    if government_key in {"maritime-republic", "city-republic"}:
        return "civic-administration"
    if government_key in {
        "nomadic-confederacy",
        "clan-league",
        "tributary-chiefdom",
    }:
        return "confederal-territory"
    if government_key == "estate-monarchy":
        return "estate-administration"
    raise ValueError(f"unsupported government key for provinces: {government_key}")


def _province_density_multiplier(density_ratio: float) -> float:
    """Return how many divisions equal area supports at a given density."""

    return float(np.clip(max(density_ratio, 1.0e-6) ** 0.42, 0.58, 1.78))


def _institutional_division_multiplier(administrative_system: str) -> float:
    return {
        "central-bureaucracy": 1.20,
        "feudal-vassalage": 1.00,
        "civic-administration": 1.12,
        "confederal-territory": 0.68,
        "estate-administration": 0.96,
    }[administrative_system]


def _province_core_targets(
    state_areas: Mapping[int, float],
    density_ratios: Mapping[int, float],
    administrative_systems: Mapping[int, str],
    candidate_counts: Mapping[int, int],
) -> dict[int, int]:
    """Apportion administrative seats without forcing equal-sized countries.

    The atlas aims for about 5.6 provinces per country overall. Small states
    with only one real town consume one seat; their unused share goes to
    larger, settled states. No town or population is invented to hit a count.
    Area, population density and institutions determine governing workload,
    while urban-network size represents the number of actual local centres.
    """
    identifiers = sorted(state_areas)
    if not identifiers:
        return {}
    capacity = {identifier: max(1, candidate_counts[identifier]) for identifier in identifiers}
    target = min(round(len(identifiers) * 5.6), sum(capacity.values()))
    area_total = max(1, sum(state_areas.values()))
    town_total = max(1, sum(capacity.values()))
    workload = {
        identifier: (
            0.65 * state_areas[identifier] / area_total
            * _province_density_multiplier(density_ratios[identifier])
            + 0.35 * capacity[identifier] / town_total
        ) * _institutional_division_multiplier(administrative_systems[identifier])
        for identifier in identifiers
    }
    result = dict.fromkeys(identifiers, 1)
    for _seat in range(target - len(identifiers)):
        eligible = [identifier for identifier in identifiers if result[identifier] < capacity[identifier]]
        if not eligible:
            break
        chosen = max(eligible, key=lambda identifier: (
            workload[identifier] / (result[identifier] + 0.5), -identifier,
        ))
        result[chosen] += 1
    return result


def _local_state_metrics(
    settlement: Settlement,
    state_identifier: int,
    state_id: np.ndarray,
    density_field: np.ndarray,
    cell_areas: np.ndarray,
    accessibility: np.ndarray,
    *,
    radius: int = 18,
) -> tuple[float, float, float]:
    """Measure density, access and border exposure around one division core."""

    height, width = state_id.shape
    rows = np.arange(
        max(0, settlement.row - radius),
        min(height, settlement.row + radius + 1),
        dtype=np.int64,
    )
    columns = np.asarray(
        [(settlement.column + offset) % width for offset in range(-radius, radius + 1)],
        dtype=np.int64,
    )
    local_labels = state_id[np.ix_(rows, columns)]
    local_state = local_labels == state_identifier
    if not bool(np.any(local_state)):
        return 0.0, 0.0, 1.0
    density = float(np.average(density_field[np.ix_(rows, columns)][local_state],
                              weights=cell_areas[np.ix_(rows, columns)][local_state]))
    access = float(np.mean(accessibility[np.ix_(rows, columns)][local_state]))
    governable_land = local_labels >= 0
    border_exposure = 1.0 - float(
        np.count_nonzero(local_state) / max(1, np.count_nonzero(governable_land))
    )
    return density, access, border_exposure


def _province_role(
    *,
    administrative_system: str,
    government_key: str,
    core: Settlement,
    capital_identifier: str,
    region_type: str,
    ordinal: int,
    non_capital_count: int,
    border_exposure: float,
    access: float,
    elevation: float,
) -> tuple[str, str]:
    """Derive an office/rank from institutions, geography and strategic use."""

    if core.identifier == capital_identifier:
        rank = {
            "central-bureaucracy": "capital-district",
            "feudal-vassalage": "royal-seat",
            "civic-administration": "metropolitan-district",
            "confederal-territory": "court-domain",
            "estate-administration": "estate-seat",
        }[administrative_system]
        return "capital", rank

    # Fortresses are created only from actual political-border crossings.
    # If one is strong and remote enough to become a provincial core, its
    # institution must remain military rather than being renamed as an
    # ordinary civil, pastoral, civic or enfeoffed district.
    if core.site_type == "fortress":
        return {
            "central-bureaucracy": ("military", "military-command"),
            "feudal-vassalage": ("military", "march"),
            "civic-administration": ("frontier", "defence-district"),
            "confederal-territory": ("frontier", "frontier-wing"),
            "estate-administration": ("military", "service-march"),
        }[administrative_system]

    frontier = border_exposure >= 0.55 or (
        border_exposure >= 0.42 and access < 0.12
    )
    highland = elevation >= 0.50 or region_type == "highland"
    maritime = region_type in {"archipelago", "coastal"}
    strategic_border = border_exposure >= 0.42 and (
        highland
        or core.site_type in {"pass", "fortress"}
        or access >= 0.14
    )

    if administrative_system == "feudal-vassalage":
        if frontier:
            return "military", "march"
        # Crown lands are productive or strategic districts kept under the
        # monarch's own officers.  They are not synonyms for the capital and
        # may recur away from the royal seat among otherwise enfeoffed lands.
        if access >= 0.15 and non_capital_count >= 2 and ordinal == 1:
            return "crown", "crown-domain"
        if core.tier == "metropolis" or (
            non_capital_count >= 3 and ordinal == 2
        ):
            rank = "duchy"
        elif core.tier == "city":
            rank = "county"
        elif ordinal % 3 == 0:
            rank = "viscounty"
        else:
            rank = "barony"
        return "vassal", rank
    if administrative_system == "central-bureaucracy":
        if strategic_border:
            return "military", "military-command"
        if frontier:
            return "frontier", "frontier-command"
        if maritime:
            return "maritime", "maritime-circuit"
        return "civil", "civil-prefecture"
    if administrative_system == "civic-administration":
        if frontier:
            return "frontier", "defence-district"
        # A sea-oriented republic can govern an inland hinterland, but that
        # does not turn every one of its provinces into a port authority.
        # Maritime rank is derived from this province's own core geography.
        if maritime:
            return "maritime", "naval-trade-district"
        return "civic", "civic-district"
    if administrative_system == "confederal-territory":
        if frontier:
            return "frontier", "frontier-wing"
        if highland or government_key == "clan-league":
            return "military", "clan-territory"
        return "pastoral", "tribal-territory"
    if frontier:
        return "military", "service-march"
    if maritime:
        return "maritime", "chartered-port"
    return "civil", "estate-district"


def _province_suffixes(
    administrative_system: str,
    administrative_function: str,
    administrative_rank: str,
    style: int,
) -> tuple[str, ...]:
    """Translate one division's legal identity through its own culture.

    ``administrative_rank`` is the underlying institution; ``style`` controls
    the atlas translation used by that civilization.  A feudal county must
    therefore read as a county-equivalent, while ``道`` remains exclusive to
    the Sinitic bureaucratic tradition instead of leaking into western names.
    """

    western_styles = {1, 2, 3, 9, 10}
    if administrative_system == "feudal-vassalage":
        western_peerage = {
            "royal-seat": ("王领", "王室领"),
            "crown-domain": ("王冠领", "直辖领"),
            "march": ("边疆侯国", "边疆侯领", "侯国"),
            "duchy": ("公国", "公爵领"),
            "county": ("伯国", "伯爵领"),
            "viscounty": ("子爵领",),
            "barony": ("男爵领", "采邑"),
        }
        if style == 0:
            return {
                **western_peerage,
                "royal-seat": ("王畿",),
                "crown-domain": ("王领",),
            }[administrative_rank]
        if style in western_styles:
            return western_peerage[administrative_rank]
        cultural_peerage = {
            4: {
                "royal-seat": ("汗廷领",),
                "crown-domain": ("汗室领",),
                "march": ("边镇领",),
                "duchy": ("大贝伊领",),
                "county": ("贝伊领",),
                "viscounty": ("台吉领",),
                "barony": ("诺颜领",),
            },
            5: {
                "royal-seat": ("沙阿领",),
                "crown-domain": ("王室领",),
                "march": ("边侯领",),
                "duchy": ("大埃米尔领",),
                "county": ("埃米尔领",),
                "viscounty": ("总督领",),
                "barony": ("迪赫干领",),
            },
            6: {
                "royal-seat": ("苏丹领",),
                "crown-domain": ("王室领",),
                "march": ("边防埃米尔国",),
                "duchy": ("大埃米尔国",),
                "county": ("埃米尔国",),
                "viscounty": ("谢赫领",),
                "barony": ("部族领",),
            },
            7: {
                "royal-seat": ("王畿",),
                "crown-domain": ("王领",),
                "march": ("边邦",),
                "duchy": ("大王公领",),
                "county": ("罗阇领",),
                "viscounty": ("土邦",),
                "barony": ("封邑",),
            },
            8: {
                "royal-seat": ("苏丹领",),
                "crown-domain": ("王家领",),
                "march": ("边防邦",),
                "duchy": ("大邦",),
                "county": ("岛邦",),
                "viscounty": ("达督领",),
                "barony": ("酋领",),
            },
            11: {
                "royal-seat": ("王廷领",),
                "crown-domain": ("王家领",),
                "march": ("边境酋邦",),
                "duchy": ("大酋邦",),
                "county": ("酋邦",),
                "viscounty": ("氏族领",),
                "barony": ("村社领",),
            },
        }
        return cultural_peerage[style][administrative_rank]

    if administrative_system == "central-bureaucracy":
        by_style = {
            0: {
                "capital-district": ("府",),
                "civil-prefecture": ("州", "郡", "府"),
                "military-command": ("军府", "镇"),
                "frontier-command": ("都护府", "边镇"),
                "maritime-circuit": ("海道", "海府"),
            },
            1: {
                "capital-district": ("王廷领", "王城辖区", "宫廷直辖领"),
                "civil-prefecture": ("行政郡", "辖区", "直辖领"),
                "military-command": ("边疆伯领", "卫戍领", "要塞辖区"),
                "frontier-command": ("边境侯领", "边疆领", "关隘辖区"),
                "maritime-circuit": ("滨海郡", "港务辖区", "海防领"),
            },
            2: {
                "capital-district": ("都城省", "首都辖区", "王室直辖省"),
                "civil-prefecture": ("行省", "总督领", "行政辖区"),
                "military-command": ("边疆总督领", "戍卫总督区", "要塞辖区"),
                "frontier-command": ("边疆侯领", "边省", "边境总督领"),
                "maritime-circuit": ("滨海行省", "海岸总督领", "港务辖区"),
            },
            3: {
                "capital-district": ("王公领", "都城辖区", "大公直辖领"),
                "civil-prefecture": ("行政领", "州区", "辖区"),
                "military-command": ("边疆戍领", "守备区", "要塞领"),
                "frontier-command": ("边疆领", "边区", "边境总督领"),
                "maritime-circuit": ("滨海领", "港区", "海岸辖区"),
            },
            4: {
                "capital-district": ("汗廷",),
                "civil-prefecture": ("辖地", "州"),
                "military-command": ("军府",),
                "frontier-command": ("边镇",),
                "maritime-circuit": ("海领", "港地"),
            },
            5: {
                "capital-district": ("王都省",),
                "civil-prefecture": ("行省", "辖区", "总督领"),
                "military-command": ("军政总督领", "总督区"),
                "frontier-command": ("边疆总督领", "边镇"),
                "maritime-circuit": ("海岸省", "港区"),
            },
            6: {
                "capital-district": ("都城辖区",),
                "civil-prefecture": ("行省", "辖区"),
                "military-command": ("埃米尔领",),
                "frontier-command": ("边防区",),
                "maritime-circuit": ("港省", "海岸区"),
            },
            7: {
                "capital-district": ("都城邦",),
                "civil-prefecture": ("邦", "辖区"),
                "military-command": ("军镇",),
                "frontier-command": ("边邦",),
                "maritime-circuit": ("海邦", "港区"),
            },
            8: {
                "capital-district": ("王廷领",),
                "civil-prefecture": ("邦", "辖地"),
                "military-command": ("军镇",),
                "frontier-command": ("边地",),
                "maritime-circuit": ("岛邦", "港领"),
            },
            9: {
                "capital-district": ("王廷领", "王城领", "王冠直辖领"),
                "civil-prefecture": ("领地", "行政郡", "氏族辖区"),
                "military-command": ("边疆伯领", "卫戍领", "要塞领"),
                "frontier-command": ("边境侯领", "边疆领", "边区"),
                "maritime-circuit": ("海岸郡", "群岛领", "港务领"),
            },
            10: {
                "capital-district": ("首府辖区", "王廷领", "王室直辖领"),
                "civil-prefecture": ("辖区", "行政领", "州领"),
                "military-command": ("边疆领", "守备领", "要塞区"),
                "frontier-command": ("边地", "边境领", "边疆总督领"),
                "maritime-circuit": ("海岸领", "岛区", "港务辖区"),
            },
            11: {
                "capital-district": ("王廷领",),
                "civil-prefecture": ("部地", "辖地"),
                "military-command": ("战团领",),
                "frontier-command": ("边地",),
                "maritime-circuit": ("海领", "港地"),
            },
        }
        return by_style[style][administrative_rank]

    if administrative_system == "civic-administration":
        civic_rank = {
            "metropolitan-district": ("自由市", "都会区"),
            "naval-trade-district": ("海港市", "商港区"),
            "defence-district": ("守备区", "边防区"),
            "civic-district": ("市辖区", "共同体"),
        }
        if style == 0:
            civic_rank = {
                **civic_rank,
                "metropolitan-district": ("都会区",),
                "naval-trade-district": ("市舶区",),
                "defence-district": ("守备区",),
                "civic-district": ("市辖区",),
            }
        elif style == 4:
            civic_rank = {
                **civic_rank,
                "metropolitan-district": ("牙帐市",),
                "naval-trade-district": ("商港",),
                "defence-district": ("边旗",),
                "civic-district": ("市盟",),
            }
        elif style in {5, 6}:
            civic_rank = {
                **civic_rank,
                "metropolitan-district": ("都城市",),
                "naval-trade-district": ("商港",),
                "defence-district": ("边防区",),
                "civic-district": ("市社",),
            }
        elif style in {7, 8, 11}:
            civic_rank = {
                **civic_rank,
                "metropolitan-district": ("城邦",),
                "naval-trade-district": ("港邦",),
                "defence-district": ("边邦",),
                "civic-district": ("市邦",),
            }
        return civic_rank[administrative_rank]

    if administrative_system == "confederal-territory":
        by_style = {
            0: ("盟庭", "边旗", "宗部", "部"),
            4: ("汗庭", "翼", "旗", "牧地"),
            5: ("王帐", "边旗", "氏族领", "部族领"),
            6: ("王帐", "边防领", "氏族领", "部族领"),
            7: ("王帐", "边邦", "氏族邦", "部族邦"),
            8: ("王廷", "海防部", "氏族地", "部地"),
            11: ("王庭", "边部", "氏族地", "部地"),
        }
        terms = by_style.get(style, ("议盟", "边盟", "氏族领", "部落领"))
        rank_index = {
            "court-domain": 0,
            "frontier-wing": 1,
            "clan-territory": 2,
            "tribal-territory": 3,
        }[administrative_rank]
        return (terms[rank_index],)

    if administrative_system == "estate-administration":
        if style == 0:
            terms = ("王领", "军镇", "市舶港", "封地")
        elif style in western_styles:
            terms = ("王冠领", "侯国", "特许港", "领地")
        elif style == 4:
            terms = ("汗室领", "边镇领", "商港领", "采邑")
        elif style in {5, 6}:
            terms = ("王室领", "边侯领", "特许港", "总督领")
        elif style == 7:
            terms = ("王领", "边邦", "港邦", "土邦")
        elif style == 8:
            terms = ("王家领", "边防邦", "港邦", "封地")
        else:
            terms = ("王家领", "边境酋邦", "港地", "酋领")
        rank_index = {
            "estate-seat": 0,
            "service-march": 1,
            "chartered-port": 2,
            "estate-district": 3,
        }[administrative_rank]
        return (terms[rank_index],)

    raise ValueError(
        f"unsupported province suffix: {administrative_system}/"
        f"{administrative_function}/{administrative_rank}/style-{style}"
    )


def _province_name(
    settlement: Settlement,
    *,
    state_identifier: int,
    name_family: str,
    administrative_system: str,
    administrative_function: str,
    administrative_rank: str,
    used: set[str],
) -> str:
    """Create a cultural administrative name independent of its core city.

    Province labels previously copied the city and appended a physical term
    such as "coast" or "river valley".  At the city symbol that looked like a
    malformed city name.  Provinces now draw a separate local-language root
    and a culturally translated legal suffix; geography still shapes the
    boundary and is retained separately as ``region_type``.
    """

    # The capital relationship is stored as an institution, not printed as a
    # literal concept such as "capital district" or "direct rule".  Its actual
    # place-name follows the capital city; other provinces retain their own
    # historic/local roots.
    if administrative_function == "capital":
        roots = (settlement.name,)
    else:
        # Prefer the culture's historical place roots, then fall back to the
        # province's own unique administrative centre.  A large polity can
        # legitimately reuse the same legal rank dozens of times; exhausting
        # a finite family lexicon must not produce duplicate names or abort
        # the otherwise valid territorial simulation.
        roots = tuple(
            dict.fromkeys((*lineage_roots(name_family), settlement.name))
        )
    style = lineage_style_index(name_family)
    terms = _province_suffixes(
        administrative_system,
        administrative_function,
        administrative_rank,
        style,
    )
    digest = hashlib.sha256(
        (
            f"province:{state_identifier}:{settlement.identifier}:"
            f"{administrative_system}:{administrative_function}:"
            f"{administrative_rank}"
        ).encode("utf-8")
    ).digest()
    root_start = int.from_bytes(digest[:4], "big") % len(roots)
    suffix_start = int.from_bytes(digest[4:6], "big") % len(terms)
    for offset in range(len(roots) * len(terms)):
        root = roots[(root_start + offset) % len(roots)]
        suffix = terms[(suffix_start + offset // len(roots)) % len(terms)]
        candidate = f"{root}{suffix}"
        if candidate not in used and candidate != settlement.name:
            used.add(candidate)
            return candidate
    # Several language families deliberately share readable roots, and city
    # names may also repeat across a whole world.  When every short form is
    # occupied, form a historically plausible compound toponym instead of a
    # numeric suffix.  The centre comes first so the province remains locally
    # identifiable; the second root supplies enough deterministic global
    # uniqueness for dense imperial administrations.
    family_roots = lineage_roots(name_family)
    compound_count = len(family_roots) * len(terms)
    for offset in range(compound_count):
        qualifier = family_roots[(root_start + offset) % len(family_roots)]
        suffix = terms[(suffix_start + offset // len(family_roots)) % len(terms)]
        candidate = f"{settlement.name}{qualifier}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError("unable to resolve province-name collision")


def derive_provinces(
    grid: WorldGrid,
    thematic: ThematicLayers,
    population: PopulationLayers,
    settlements: Sequence[Settlement],
    transport: TransportLayers,
    cultures: CultureLayers,
    politics: PoliticalLayers,
) -> ProvinceLayers:
    """Partition each country around city cores without crossing its border."""

    if politics.state_id.shape != grid.shape:
        raise ValueError("political layers must match the WorldGrid shape")
    if population.population_weight.shape != grid.shape:
        raise ValueError("population layers must match the WorldGrid shape")
    if transport.accessibility.shape != grid.shape:
        raise ValueError("transport layers must match the WorldGrid shape")
    state_id = np.asarray(politics.state_id, dtype=np.int16)
    controlled = state_id > 0
    result = np.where(grid.water == 0, 0, -1).astype(np.int32)
    if not np.any(controlled) or not politics.states:
        return ProvinceLayers(province_id=result, provinces=())

    controlled_area = int(np.count_nonzero(controlled))
    population_weight = np.asarray(population.population_weight, dtype=np.float64)
    density_field = population_density(grid, population)
    cell_areas = cell_areas_km2(grid)
    midpoint_population = (population.population_min + population.population_max) / 2.0
    accessibility = np.asarray(transport.accessibility, dtype=np.float64)
    road_corridor, road_junction = road_network_fields(grid.shape, transport.routes)
    controlled_population = float(np.sum(population_weight[controlled]))
    world_controlled_density = controlled_population * midpoint_population / float(np.sum(cell_areas[controlled]))
    province_specs: list[_ProvinceSeed] = []
    settlement_by_identifier = {item.identifier: item for item in settlements}
    civilization_by_identifier = {
        item.identifier: item for item in cultures.civilizations
    }
    government_by_identifier = {
        item.identifier: item for item in politics.government_forms
    }
    entity_by_state = {
        item.country_identifier: item for item in politics.political_entities
    }
    state_areas: dict[int, float] = {}
    density_ratios: dict[int, float] = {}
    administrative_systems: dict[int, str] = {}
    candidates_by_state: dict[int, list[Settlement]] = {}
    candidate_counts: dict[int, int] = {}
    for state in politics.states:
        state_region = state_id == state.identifier
        area = int(np.count_nonzero(state_region))
        if area <= 0:
            continue
        entity = entity_by_state.get(state.identifier)
        if entity is None:
            raise ValueError("province parent country requires a political entity")
        government = government_by_identifier.get(entity.government_form_identifier)
        if government is None:
            raise ValueError("province parent country requires a government form")
        candidates = _province_core_candidates(
            state.identifier, state.core_settlement_id, settlements, state_id,
        )
        if not candidates:
            core = settlement_by_identifier.get(state.core_settlement_id)
            if core is None:
                raise ValueError("province parent country requires a real capital")
            candidates = [core]
        state_areas[state.identifier] = float(np.sum(cell_areas[state_region]))
        density_ratios[state.identifier] = (
            float(np.sum(population_weight[state_region])) * midpoint_population / state_areas[state.identifier]
            / max(world_controlled_density, 1.0e-12)
        )
        administrative_systems[state.identifier] = _administrative_system(government.key)
        candidates_by_state[state.identifier] = candidates
        candidate_counts[state.identifier] = max(1, sum(item.tier != "site" for item in candidates))
    core_targets = _province_core_targets(
        state_areas, density_ratios, administrative_systems, candidate_counts,
    )
    target_area = max(48.0, controlled_area / max(1, sum(core_targets.values())))
    route_group_by_identifier = _major_route_groups(
        settlements,
        transport,
        state_id,
    )
    used_names = {item.name for item in settlements}
    used_names.update(item.name for item in politics.states)
    used_names.update(item.formal_name for item in politics.political_entities)
    next_identifier = 1
    for state in politics.states:
        state_region = state_id == state.identifier
        state_area = int(np.count_nonzero(state_region))
        if state_area <= 0:
            continue
        entity = entity_by_state.get(state.identifier)
        if entity is None:
            raise ValueError("province parent country requires a political entity")
        government = government_by_identifier.get(entity.government_form_identifier)
        if government is None:
            raise ValueError("province parent country requires a government form")
        administrative_system = administrative_systems[state.identifier]
        candidates = candidates_by_state[state.identifier]
        density_ratio = density_ratios[state.identifier]
        desired = core_targets[state.identifier]

        component_id, component_sizes = connected_components(state_region)
        capital = settlement_by_identifier.get(state.core_settlement_id)
        capital_component = (
            int(component_id[capital.row, capital.column])
            if capital is not None
            else 0
        )
        capital_route_group = route_group_by_identifier.get(
            state.core_settlement_id
        )
        minimum_independent_component = min(
            128,
            max(16, int(round(state_area * 0.004))),
        )
        required_identifiers: set[str] = set()
        for component_identifier, component_size in enumerate(
            component_sizes,
            start=1,
        ):
            if (
                component_identifier == capital_component
                or component_size < minimum_independent_component
            ):
                continue
            component_candidates = [
                item
                for item in candidates
                if int(component_id[item.row, item.column]) == component_identifier
            ]
            if not component_candidates:
                continue
            route_connected = capital_route_group is not None and any(
                route_group_by_identifier.get(item.identifier)
                == capital_route_group
                for item in component_candidates
            )
            if not route_connected:
                required_identifiers.add(component_candidates[0].identifier)
        desired = max(desired, 1 + len(required_identifiers))
        state_access = float(np.mean(accessibility[state_region]))
        service_radius = (
            math.sqrt(target_area / math.pi)
            * 1.38
            * float(np.clip(0.88 + 0.72 * state_access, 0.88, 1.42))
            / math.sqrt(
                _province_density_multiplier(density_ratio)
                * _institutional_division_multiplier(administrative_system)
            )
        )
        cores = _select_province_cores(
            candidates,
            capital_identifier=state.core_settlement_id,
            count=desired,
            width=grid.shape[1],
            route_group_by_identifier=route_group_by_identifier,
            required_identifiers=frozenset(required_identifiers),
            service_radius=max(14.0, service_radius),
        )
        civilization = civilization_by_identifier.get(state.civilization_identifier)
        if civilization is None:
            raise ValueError("province parent country must reference a civilization")
        local_metrics = {
            core.identifier: _local_state_metrics(
                core,
                state.identifier,
                state_id,
                density_field,
                cell_areas,
                accessibility,
            )
            for core in cores
        }
        non_capital = sorted(
            (core for core in cores if core.identifier != state.core_settlement_id),
            key=lambda item: (-item.score, item.identifier),
        )
        feudal_ordinal = {
            core.identifier: ordinal
            for ordinal, core in enumerate(non_capital, start=1)
        }
        for core in cores:
            region_type = _province_region_type(core, grid, thematic)
            local_density, local_access, border_exposure = local_metrics[core.identifier]
            function, rank = _province_role(
                administrative_system=administrative_system,
                government_key=government.key,
                core=core,
                capital_identifier=state.core_settlement_id,
                region_type=region_type,
                ordinal=feudal_ordinal.get(core.identifier, 0),
                non_capital_count=len(non_capital),
                border_exposure=border_exposure,
                access=local_access,
                elevation=float(grid.elevation[core.row, core.column]),
            )
            name = _province_name(
                core,
                state_identifier=state.identifier,
                name_family=civilization.name_family,
                administrative_system=administrative_system,
                administrative_function=function,
                administrative_rank=rank,
                used=used_names,
            )
            local_density_ratio = local_density / max(world_controlled_density, 1.0e-12)
            density_growth = float(
                np.clip(max(local_density_ratio, 1.0e-6) ** -0.28, 0.62, 1.62)
            )
            function_growth = {
                "capital": 0.86,
                "civil": 0.90,
                "civic": 0.88,
                "maritime": 0.98,
                "vassal": {
                    "duchy": 1.16,
                    "county": 1.06,
                    "viscounty": 0.96,
                    "barony": 0.88,
                }.get(rank, 1.0),
                "crown": 0.94,
                "military": 1.12,
                "frontier": 1.28,
                "pastoral": 1.32,
            }[function]
            province_specs.append(
                _ProvinceSeed(
                    identifier=next_identifier,
                    state_identifier=state.identifier,
                    core=core,
                    name=name,
                    region_type=region_type,
                    administrative_system=administrative_system,
                    administrative_function=function,
                    administrative_rank=rank,
                    growth_scale=float(np.clip(density_growth * function_growth, 0.48, 2.15)),
                )
            )
            next_identifier += 1

    if not province_specs:
        return ProvinceLayers(province_id=result, provinces=())

    elevation = np.asarray(grid.elevation, dtype=np.float64)
    slope = relative_land_slope(elevation, grid.water == 0)
    neighbor_sum = np.zeros(grid.shape, dtype=np.float64)
    neighbor_count = np.zeros(grid.shape, dtype=np.uint8)
    land = grid.water == 0
    for dy, dx in (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ):
        valid_neighbor = np.roll(land, shift=(dy, dx), axis=(0, 1))
        if dy < 0:
            valid_neighbor[-1] = False
        elif dy > 0:
            valid_neighbor[0] = False
        neighbor_sum += np.where(valid_neighbor,
            np.roll(elevation, shift=(dy, dx), axis=(0, 1)), 0.0)
        neighbor_count += valid_neighbor
    neighbor_mean = np.divide(neighbor_sum, neighbor_count,
                              out=elevation.copy(), where=neighbor_count > 0)
    ridge = np.clip((elevation - neighbor_mean) / 0.035, 0.0, 1.0)
    friction = (
        1.0
        + 6.4 * np.clip(slope / 0.08, 0.0, 1.0)
        + 3.2 * np.clip((elevation - 0.45) / 0.35, 0.0, 1.0)
        + 3.0 * np.power(1.0 - accessibility, 1.35)
        + 2.8 * ridge
    )
    friction = np.clip(friction, 0.55, None)
    transitions = _province_transition_penalties(grid, road_corridor)
    road_access = np.clip(0.82 * road_corridor + 0.18 * road_junction, 0.0, 1.0)
    bridge_edges = bridge_transition_discounts(
        np.asarray(grid.river_order),
        transport.routes,
        transport.bridges,
    )
    for spec in province_specs:
        if int(state_id[spec.core.row, spec.core.column]) != spec.state_identifier:
            raise ValueError("province core must lie inside its parent country")

    # Provinces aggregate the same kind of local daily-control regions as
    # countries.  Province capitals no longer compete for raw pixels across a
    # whole state, which was the source of ruler-straight cuts and rectangular
    # wedges on open plains.  Permanent settlements and any strategic site
    # selected as a provincial core all receive an initial hinterland.
    required_core_ids = {spec.core.identifier for spec in province_specs}
    control_settlements: list[Settlement] = []
    occupied_cells: set[tuple[int, int]] = set()
    for settlement in sorted(settlements, key=lambda item: item.identifier):
        if settlement.tier == "site" and settlement.identifier not in required_core_ids:
            continue
        cell = (settlement.row, settlement.column)
        if cell in occupied_cells or int(state_id[cell]) <= 0:
            continue
        occupied_cells.add(cell)
        control_settlements.append(settlement)
    control_seeds = tuple(
        TerritorySeed(
            row=settlement.row,
            column=settlement.column,
            owner=identifier,
            strength=float(
                np.clip(
                    0.92
                    + 0.16 * settlement.score
                    + 0.10 * float(accessibility[settlement.row, settlement.column]),
                    0.82,
                    1.32,
                )
            ),
            domain=int(state_id[settlement.row, settlement.column]),
        )
        for identifier, settlement in enumerate(control_settlements, start=1)
    )
    local_control = simulate_territories(
        TerritorySimulation(
            valid=controlled,
            friction=friction,
            transition_penalty=transitions,
            road_access=road_access,
            bridge_edges=bridge_edges,
            owner_constraint=state_id,
        ),
        control_seeds,
    )
    control_labels = local_control.owner.astype(np.int32)
    control_domain = np.zeros(len(control_seeds) + 1, dtype=np.int32)
    control_anchors: dict[tuple[int, int], int] = {}
    for identifier, seed in enumerate(control_seeds, start=1):
        control_domain[identifier] = seed.domain
        control_anchors[(seed.row, seed.column)] = identifier
    control_labels = snap_partition_to_natural_regions(
        control_labels,
        controlled,
        transitions,
        control_anchors,
        owner_field=state_id,
        label_owner=control_domain,
        barrier_threshold=14.0,
        anchored_support=0.42,
        unanchored_majority=0.62,
    )
    control_labels = refine_partition_boundaries(
        control_labels,
        controlled,
        transitions,
        control_anchors,
        owner_field=state_id,
        label_owner=control_domain,
        friction=friction.astype(np.float32),
        band_radius=10,
    ).astype(np.int32)
    control_labels = natural_compartment_ids(
        controlled & (control_labels > 0),
        transitions,
        domain=control_labels,
        barrier_threshold=14.0,
    ).astype(np.int32)
    population_normalized = density_field.astype(np.float64)
    population_normalized /= max(
        float(population_normalized.max(initial=0.0)),
        1.0e-12,
    )
    local_resources = (
        0.56 * population_normalized
        + 0.26 * np.clip(thematic.land_potential, 0.0, 1.0)
        + 0.18 * np.clip(accessibility, 0.0, 1.0)
    ).astype(np.float32)
    control_graph = build_control_region_graph(
        control_labels,
        transitions,
        road_access,
        bridge_edges,
        state_id,
        local_resources,
    )
    core_region_by_province = np.zeros(len(province_specs) + 1, dtype=np.int32)
    province_strength = np.zeros(len(province_specs) + 1, dtype=np.float32)
    for spec in province_specs:
        core_region_by_province[spec.identifier] = int(
            control_labels[spec.core.row, spec.core.column]
        )
        province_strength[spec.identifier] = spec.growth_scale
    control_graph, edge_days, response_budgets = control_region_reach(
        grid, control_graph, control_labels, core_region_by_province, province_strength,
        provinces=True, routes=transport.routes,
    )
    formation = simulate_state_formation(
        control_graph,
        core_region_by_state=core_region_by_province,
        state_strength=province_strength,
        travel_days_by_edge=edge_days,
        maximum_response_days_by_state=response_budgets,
    )
    result = np.where(np.asarray(grid.water) == 0, 0, -1).astype(np.int32)
    result[controlled] = formation.region_owner[control_labels[controlled]]
    result = _assign_unseeded_province_islands(
        result, state_id, province_specs, settlements=settlements, routes=transport.routes,
    )

    province_state = np.zeros(len(province_specs) + 1, dtype=np.int16)
    for spec in province_specs:
        province_state[spec.identifier] = spec.state_identifier
    protected_rows, protected_columns = np.nonzero(
        _settlement_protection_mask(settlements, grid.shape, radius=1) & (result > 0)
    )
    result = refine_partition_boundaries(
        result,
        controlled,
        transitions * (1.0 - bridge_edges),
        {
            (int(row), int(column)): int(result[row, column])
            for row, column in zip(protected_rows, protected_columns, strict=True)
        },
        owner_field=state_id,
        label_owner=province_state,
        friction=friction.astype(np.float32),
        band_radius=8,
    ).astype(np.int32)
    result = _repair_inland_province_fragments(
        result,
        state_id,
        province_specs,
    )
    result, kept_provinces = _consolidate_tiny_provinces(
        result,
        province_state,
        protected_identifiers=frozenset(
            spec.identifier
            for spec in province_specs
            if spec.administrative_function == "capital"
        ),
    )
    province_specs = tuple(
        replace(spec, identifier=new_identifier)
        for new_identifier, old_identifier in enumerate(kept_provinces, start=1)
        for spec in (province_specs[old_identifier - 1],)
    )

    result = _repair_inland_province_fragments(result, state_id, province_specs)
    if np.any(controlled & (result <= 0)):
        raise ValueError("every country cell must belong to a province")
    province_records: list[Province] = []
    for spec in province_specs:
        region = result == spec.identifier
        area_cells = int(np.count_nonzero(region))
        density = float(np.sum(population_weight[region])) * midpoint_population / float(np.sum(cell_areas[region]))
        density_ratio = density / max(world_controlled_density, 1.0e-12)
        density_class = (
            "dense"
            if density_ratio >= 1.35
            else "settled" if density_ratio >= 0.62 else "sparse"
        )
        province_records.append(
            Province(
                identifier=spec.identifier,
                name=spec.name,
                state_identifier=spec.state_identifier,
                core_settlement_id=spec.core.identifier,
                region_type=spec.region_type,
                administrative_system=spec.administrative_system,
                administrative_function=spec.administrative_function,
                administrative_rank=spec.administrative_rank,
                population_density_class=density_class,
                area_cells=area_cells,
            )
        )
    return ProvinceLayers(
        province_id=result,
        provinces=tuple(province_records),
    )


__all__ = ["derive_provinces"]
