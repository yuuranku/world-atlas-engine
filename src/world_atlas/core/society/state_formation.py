"""Path-dependent polity formation over settlement control regions.

The pixel grid still carries the physical truth, but countries do not compete
for pixels directly.  Settlements first receive connected local hinterlands.
Those hinterlands become graph nodes that states absorb over a bounded series
of simultaneous political rounds.  This preserves physical seams and makes
administrative reach, rather than geometric proximity, determine scale.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter
from collections.abc import Sequence
import math

import numpy as np

from .model import Settlement, TransportRoute, immutable_array
from .administrative_topology import reconcile_partition_components


@dataclass(frozen=True, slots=True)
class ControlRegionEdge:
    first: int
    second: int
    contact_cells: int
    crossing_cost: float
    barrier_mean: float
    hard_barrier_share: float
    road_share: float
    bridge_strength: float

    def __post_init__(self) -> None:
        if self.first < 1 or self.second <= self.first:
            raise ValueError("control-region edge identifiers must be ordered and positive")
        if self.contact_cells < 1:
            raise ValueError("control-region edges require shared boundary cells")
        values = (
            self.crossing_cost,
            self.barrier_mean,
            self.hard_barrier_share,
            self.road_share,
            self.bridge_strength,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("control-region edge metrics must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ControlRegionGraph:
    domain_by_region: np.ndarray
    resource_by_region: np.ndarray
    area_by_region: np.ndarray
    edges: tuple[ControlRegionEdge, ...]

    def __post_init__(self) -> None:
        domains = np.asarray(self.domain_by_region)
        resources = np.asarray(self.resource_by_region)
        areas = np.asarray(self.area_by_region)
        if domains.ndim != 1 or resources.shape != domains.shape or areas.shape != domains.shape:
            raise ValueError("control-region graph arrays must be aligned vectors")
        if domains.size < 2 or int(domains[0]) != 0:
            raise ValueError("control-region graph requires a zero sentinel")
        if np.any(domains < 0) or np.any(resources < 0.0) or np.any(areas < 0):
            raise ValueError("control-region graph fields must be non-negative")
        expected = set(range(1, domains.size))
        present = {edge.first for edge in self.edges} | {edge.second for edge in self.edges}
        if not present.issubset(expected):
            raise ValueError("control-region edge references an unavailable region")
        object.__setattr__(
            self,
            "domain_by_region",
            immutable_array(domains, dtype=np.dtype(np.int32)),
        )
        object.__setattr__(
            self,
            "resource_by_region",
            immutable_array(resources, dtype=np.dtype(np.float32)),
        )
        object.__setattr__(
            self,
            "area_by_region",
            immutable_array(areas, dtype=np.dtype(np.int32)),
        )

    @property
    def region_count(self) -> int:
        return int(self.domain_by_region.size - 1)

    def with_domains(self, domains: np.ndarray) -> "ControlRegionGraph":
        """Return the same physical graph with a different test/domain partition."""

        return replace(self, domain_by_region=np.asarray(domains, dtype=np.int32))


@dataclass(frozen=True, slots=True)
class StateFormationResult:
    region_owner: np.ndarray
    acquisition_cost: np.ndarray
    acquisition_round: np.ndarray

    def __post_init__(self) -> None:
        owners = np.asarray(self.region_owner)
        costs = np.asarray(self.acquisition_cost)
        rounds = np.asarray(self.acquisition_round)
        if owners.ndim != 1 or costs.shape != owners.shape or rounds.shape != owners.shape:
            raise ValueError("state-formation result fields must be aligned vectors")
        object.__setattr__(
            self,
            "region_owner",
            immutable_array(owners, dtype=np.dtype(np.int32)),
        )
        object.__setattr__(
            self,
            "acquisition_cost",
            immutable_array(costs, dtype=np.dtype(np.float32)),
        )
        object.__setattr__(
            self,
            "acquisition_round",
            immutable_array(rounds, dtype=np.dtype(np.int32)),
        )


def _edge_samples(
    labels: np.ndarray,
    transitions: np.ndarray,
    roads: np.ndarray,
    bridges: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Collect each physical east/south border exactly once."""

    east_other = np.roll(labels, -1, axis=1)
    east_mask = (labels > 0) & (east_other > 0) & (labels != east_other)
    east_raw = np.maximum(transitions[4], np.roll(transitions[3], -1, axis=1))
    east_bridge = np.maximum(bridges[4], np.roll(bridges[3], -1, axis=1))
    east_effective = east_raw * (1.0 - east_bridge)
    east_road = np.maximum(roads, np.roll(roads, -1, axis=1))

    south_first = labels[:-1]
    south_second = labels[1:]
    south_mask = (
        (south_first > 0)
        & (south_second > 0)
        & (south_first != south_second)
    )
    south_raw = np.maximum(transitions[6, :-1], transitions[1, 1:])
    south_bridge = np.maximum(bridges[6, :-1], bridges[1, 1:])
    south_effective = south_raw * (1.0 - south_bridge)
    south_road = np.maximum(roads[:-1], roads[1:])

    first = np.concatenate((labels[east_mask], south_first[south_mask])).astype(np.int64)
    second = np.concatenate((east_other[east_mask], south_second[south_mask])).astype(np.int64)
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    effective = np.concatenate(
        (east_effective[east_mask], south_effective[south_mask])
    ).astype(np.float32)
    raw = np.concatenate((east_raw[east_mask], south_raw[south_mask])).astype(np.float32)
    road = np.concatenate((east_road[east_mask], south_road[south_mask])).astype(np.float32)
    bridge = np.concatenate(
        (east_bridge[east_mask], south_bridge[south_mask])
    ).astype(np.float32)
    return low, high, effective, raw, road, bridge


def build_control_region_graph(
    control_region_id: np.ndarray,
    transition_penalty: np.ndarray,
    road_access: np.ndarray,
    bridge_edges: np.ndarray,
    owner_domain: np.ndarray,
    resource: np.ndarray,
) -> ControlRegionGraph:
    """Aggregate native-resolution settlement hinterlands into a border graph."""

    labels = np.asarray(control_region_id, dtype=np.int32)
    transitions = np.asarray(transition_penalty, dtype=np.float32)
    roads = np.asarray(road_access, dtype=np.float32)
    bridges = np.asarray(bridge_edges, dtype=np.float32)
    domains = np.asarray(owner_domain, dtype=np.int32)
    resources = np.asarray(resource, dtype=np.float32)
    if labels.ndim != 2:
        raise ValueError("control-region labels must be two-dimensional")
    shape = labels.shape
    if transitions.shape != (8, *shape) or bridges.shape != transitions.shape:
        raise ValueError("control-region edge fields must have shape (8, H, W)")
    if any(field.shape != shape for field in (roads, domains, resources)):
        raise ValueError("control-region cell fields must align")
    if np.any(resources < 0.0) or np.any(~np.isfinite(resources)):
        raise ValueError("control-region resources must be finite and non-negative")
    region_count = int(labels.max(initial=0))
    if region_count < 1:
        raise ValueError("control-region graph requires positive regions")
    present = set(int(value) for value in np.unique(labels) if value > 0)
    if present != set(range(1, region_count + 1)):
        raise ValueError("control-region identifiers must be contiguous")

    area_by_region = np.bincount(
        labels[labels > 0],
        minlength=region_count + 1,
    ).astype(np.int32)
    resource_by_region = np.bincount(
        labels[labels > 0],
        weights=resources[labels > 0].astype(np.float64),
        minlength=region_count + 1,
    ).astype(np.float64)
    positive_resource = resource_by_region[1:]
    scale = float(np.median(positive_resource[positive_resource > 0.0])) if np.any(
        positive_resource > 0.0
    ) else 1.0
    resource_by_region = (resource_by_region / max(scale, 1.0e-12)).astype(np.float32)

    domain_by_region = np.zeros(region_count + 1, dtype=np.int32)
    positive_domain = (labels > 0) & (domains > 0)
    domain_minimum = np.full(region_count + 1, np.iinfo(np.int32).max, dtype=np.int32)
    np.maximum.at(domain_by_region, labels[positive_domain], domains[positive_domain])
    np.minimum.at(domain_minimum, labels[positive_domain], domains[positive_domain])
    if np.any((domain_by_region > 0) & (domain_minimum != domain_by_region)):
        raise ValueError("one control region cannot cross owner domains")

    low, high, effective, raw, road, bridge = _edge_samples(
        labels,
        transitions,
        roads,
        bridges,
    )
    if low.size == 0:
        return ControlRegionGraph(
            domain_by_region=domain_by_region,
            resource_by_region=resource_by_region,
            area_by_region=area_by_region,
            edges=(),
        )
    codes = low * (region_count + 1) + high
    order = np.argsort(codes, kind="stable")
    codes = codes[order]
    effective = effective[order]
    raw = raw[order]
    road = road[order]
    bridge = bridge[order]
    unique_codes, starts, counts = np.unique(
        codes,
        return_index=True,
        return_counts=True,
    )
    edges: list[ControlRegionEdge] = []
    for code, start, count in zip(unique_codes, starts, counts, strict=True):
        stop = int(start + count)
        values = effective[start:stop]
        raw_values = raw[start:stop]
        road_values = road[start:stop]
        bridge_values = bridge[start:stop]
        first = int(code // (region_count + 1))
        second = int(code % (region_count + 1))
        barrier_mean = float(np.mean(values))
        hard_share = float(np.mean(raw_values >= 5.0))
        road_share = float(np.mean(road_values))
        bridge_strength = float(np.max(bridge_values, initial=0.0))
        best_crossing = float(np.min(values))
        # A boundary remains defensive along most of its length, while one
        # real bridge or pass can establish a local connection.  The best
        # crossing therefore matters more than the mean, but cannot erase it.
        crossing_cost = max(
            0.25,
            0.85
            + 0.62 * best_crossing
            + 0.10 * barrier_mean
            + 0.85 * hard_share
            - 0.55 * road_share
            - 1.10 * bridge_strength,
        )
        edges.append(
            ControlRegionEdge(
                first=first,
                second=second,
                contact_cells=int(count),
                crossing_cost=float(crossing_cost),
                barrier_mean=barrier_mean,
                hard_barrier_share=hard_share,
                road_share=road_share,
                bridge_strength=bridge_strength,
            )
        )
    return ControlRegionGraph(
        domain_by_region=domain_by_region,
        resource_by_region=resource_by_region,
        area_by_region=area_by_region,
        edges=tuple(edges),
    )


def _state_capacity_targets(
    graph: ControlRegionGraph,
    core_region_by_state: np.ndarray,
    strengths: np.ndarray,
) -> np.ndarray:
    """Estimate soft administrative capacities without forcing exact areas."""

    state_count = len(core_region_by_state) - 1
    targets = np.ones(state_count + 1, dtype=np.float64)
    state_domains = graph.domain_by_region[core_region_by_state]
    for domain in sorted(int(value) for value in np.unique(state_domains[1:])):
        states = np.flatnonzero(state_domains == domain)
        regions = np.flatnonzero(graph.domain_by_region == domain)
        if states.size == 0 or regions.size == 0:
            continue
        weights = np.asarray(
            [
                max(0.05, float(strengths[state])) ** 1.35
                * max(
                    0.20,
                    float(graph.resource_by_region[core_region_by_state[state]]),
                )
                ** 0.28
                for state in states
            ],
            dtype=np.float64,
        )
        weights /= max(float(weights.sum()), 1.0e-12)
        targets[states] = np.maximum(
            1.0,
            float(regions.size) * weights,
        )
    return targets


def settle_state_boundaries(
    graph: ControlRegionGraph,
    region_owner: np.ndarray,
    *,
    core_region_by_state: np.ndarray,
    maximum_passes: int = 8,
    institutional_regions: dict[int, int] | None = None,
) -> np.ndarray:
    """Move fringe control regions so open seams become internal borders.

    The operation is performed on whole settlement hinterlands.  It cannot
    invent pixel tendrils, remove a political core, or disconnect the
    defender because each proposed transfer is checked against the actual
    remaining graph connectivity.
    """

    owners = np.asarray(region_owner, dtype=np.int32).copy()
    cores = np.asarray(core_region_by_state, dtype=np.int32)
    if owners.shape != (graph.region_count + 1,) or owners[0] != 0:
        raise ValueError("state-boundary ownership must match the control graph")
    if cores.ndim != 1 or cores.size < 2 or int(cores[0]) != 0:
        raise ValueError("state-boundary cores require a zero sentinel")
    if np.any(owners[1:] < 0) or np.any(owners[1:] >= cores.size):
        raise ValueError("control regions must reference an available state or frontier")
    core_state_by_region = np.zeros(graph.region_count + 1, dtype=np.int32)
    for state in range(1, cores.size):
        region = int(cores[state])
        if region < 1 or region > graph.region_count:
            raise ValueError("state-boundary core lies outside the control graph")
        core_state_by_region[region] = state
        if int(owners[region]) != state:
            raise ValueError("state-boundary core must remain owned by its state")
    for region, state in (institutional_regions or {}).items():
        if not 0 < region <= graph.region_count or not 0 < state < cores.size:
            raise ValueError("institutional regions must reference available regions and states")
        if int(owners[region]) != state:
            raise ValueError("institutional regions must remain owned by their state")
        core_state_by_region[region] = state
    neighbors: list[list[tuple[int, ControlRegionEdge]]] = [
        [] for _ in range(graph.region_count + 1)
    ]
    for edge in graph.edges:
        neighbors[edge.first].append((edge.second, edge))
        neighbors[edge.second].append((edge.first, edge))

    members: dict[int, set[int]] = {}
    for region in range(1, graph.region_count + 1):
        members.setdefault(int(owners[region]), set()).add(region)

    def removal_keeps_state_connected(region: int, state: int) -> bool:
        """Return whether a whole hinterland can change hands safely.

        Counting same-owner contacts mistakes any region with two contacts for
        an articulation point.  The actual political condition is graph
        connectivity after removal, so test exactly that on the small control
        graph.  The destination is connected automatically because it must
        already border ``region``.
        """

        remaining = members[state] - {region}
        if not remaining:
            return False
        start = min(remaining)
        visited = {start}
        queue = [start]
        while queue:
            current = queue.pop()
            for neighbor, _edge in neighbors[current]:
                if neighbor in remaining and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited == remaining

    def local_energy(region: int, candidate: int) -> float:
        energy = 0.0
        for neighbor, edge in neighbors[region]:
            length_weight = 1.0 + 0.10 * math.log1p(edge.contact_cells)
            if int(owners[neighbor]) == candidate:
                # Internal administration dislikes a barrier along most of
                # the shared seam even if one bridge offers a cheap crossing.
                energy += max(
                    0.0,
                    0.18 * edge.barrier_mean
                    + 1.80 * edge.hard_barrier_share
                    - 0.30 * edge.bridge_strength,
                ) * length_weight
            else:
                # A durable border wants the *whole seam* to be defensible.
                # Best-crossing cost is intentionally absent here; it is for
                # expansion and bridgehead warfare, not boundary settlement.
                energy += (
                    4.50 * (1.0 - edge.hard_barrier_share)
                    + 0.30 * max(0.0, 5.0 - edge.barrier_mean)
                    + 0.50 * edge.bridge_strength
                    + 0.05
                ) * length_weight
        return energy

    for _pass in range(maximum_passes):
        proposals: list[tuple[float, int, int, int]] = []
        for region in range(1, graph.region_count + 1):
            if core_state_by_region[region] > 0 or owners[region] == 0:
                continue
            defender = int(owners[region])
            candidates = sorted(
                {
                    int(owners[neighbor])
                    for neighbor, _edge in neighbors[region]
                    if int(owners[neighbor]) > 0 and int(owners[neighbor]) != defender
                    and graph.domain_by_region[neighbor]
                    == graph.domain_by_region[region]
                }
            )
            if not candidates:
                continue
            current_energy = local_energy(region, defender)
            improvements = [(current_energy - local_energy(region, candidate), candidate)
                            for candidate in candidates]
            improvements = [(gain, candidate) for gain, candidate in improvements if gain > 0.35]
            if not improvements or not removal_keeps_state_connected(region, defender):
                continue
            for improvement, candidate in improvements:
                proposals.append((improvement, region, defender, candidate))
        if not proposals:
            break
        changed = False
        for _improvement, region, defender, candidate in sorted(
            proposals,
            key=lambda item: (-item[0], item[1], item[3]),
        ):
            if int(owners[region]) != defender:
                continue
            if not removal_keeps_state_connected(region, defender):
                continue
            if not any(
                int(owners[neighbor]) == candidate
                for neighbor, _edge in neighbors[region]
            ):
                continue
            if local_energy(region, defender) - local_energy(region, candidate) <= 0.35:
                continue
            owners[region] = candidate
            members[defender].remove(region)
            members[candidate].add(region)
            changed = True
        if not changed:
            break
    return owners


def repair_same_land_state_fragments(
    state_id: np.ndarray,
    political_core_cells: np.ndarray,
    land_component_id: np.ndarray,
    *,
    state_domain: np.ndarray,
    maritime_routes: Sequence[TransportRoute] = (),
    institutional_seats: dict[tuple[int, int], int] | None = None,
    settlements: Sequence[Settlement] = (),
) -> np.ndarray:
    """Repair whole fragments without inventing corridors or cultural claims."""
    labels = np.asarray(state_id)
    cores = np.asarray(political_core_cells, dtype=bool)
    if labels.ndim != 2 or cores.shape != labels.shape:
        raise ValueError("state-fragment repair fields must align")
    rows, columns = np.nonzero(cores)
    seats = {int(labels[row, column]): (int(row), int(column))
             for row, column in zip(rows, columns, strict=True)}
    if 0 in seats or -1 in seats:
        raise ValueError("political cores require positive ownership")
    return reconcile_partition_components(labels, seats, state_domain, land_component_id,
                                          maritime_routes=maritime_routes, settlements=settlements,
                                          institutional_seats=institutional_seats)


def distribute_state_cores(
    graph: ControlRegionGraph,
    control_labels: np.ndarray,
    cores: dict[int, Settlement],
    settlements: Sequence[Settlement],
    *,
    protected_settlement_ids: frozenset[str],
    routes: Sequence[TransportRoute] = (),
) -> dict[int, Settlement]:
    """Use the requested number of cores across real inhabited components.

    Move redundant same-culture cores before making territorial claims.  A
    civilization's original centre is protected.  When the requested quota
    cannot cover every isolated community, remaining communities retain their
    local frontier institutions; the number of countries is never inflated.
    """
    parent = np.arange(graph.region_count + 1)

    def find(region):
        while parent[region] != region:
            parent[region] = parent[parent[region]]
            region = int(parent[region])
        return region

    for edge in graph.edges:
        if graph.domain_by_region[edge.first] == graph.domain_by_region[edge.second]:
            a, b = find(edge.first), find(edge.second)
            parent[max(a, b)] = min(a, b)
    settlement_by_id = {item.identifier: item for item in settlements}
    for route in routes:
        if route.mode != "sea" or route.target_settlement_id is None:
            continue
        first = settlement_by_id.get(route.source_settlement_id)
        second = settlement_by_id.get(route.target_settlement_id)
        if first is None or second is None:
            continue
        a = int(control_labels[first.row, first.column])
        b = int(control_labels[second.row, second.column])
        if a > 0 and b > 0 and graph.domain_by_region[a] == graph.domain_by_region[b]:
            a, b = find(a), find(b)
            parent[max(a, b)] = min(a, b)
    component = np.asarray([find(i) for i in range(graph.region_count + 1)])
    candidates: dict[int, list] = {}
    for settlement in settlements:
        region = int(control_labels[settlement.row, settlement.column])
        if region > 0 and settlement.tier != "site":
            candidates.setdefault(int(component[region]), []).append(settlement)
    result = dict(cores)
    core_component = {state: int(component[control_labels[core.row, core.column]])
                      for state, core in result.items()}
    coverage = Counter(core_component.values())
    unseeded = sorted((group for group in candidates if coverage[group] == 0),
                      key=lambda group: (-sum(item.population_max for item in candidates[group]), group))
    for group in unseeded:
        domain = int(graph.domain_by_region[group])
        donors = [state for state, core in result.items()
                  if coverage[core_component[state]] > 1
                  and int(graph.domain_by_region[core_component[state]]) == domain
                  and core.identifier not in protected_settlement_ids]
        if not donors:
            continue
        donor = min(donors, key=lambda state: (-coverage[core_component[state]],
                                              result[state].population_max, result[state].score, state))
        chosen = max(candidates[group], key=lambda item: (item.population_max, item.score, item.identifier))
        coverage[core_component[donor]] -= 1
        result[donor] = chosen
        core_component[donor] = group
        coverage[group] += 1
    return result


def attach_maritime_control_regions(
    graph: ControlRegionGraph,
    region_owner: np.ndarray,
    control_labels: np.ndarray,
    core_region_by_state: np.ndarray,
    settlements: Sequence[Settlement],
    routes: Sequence[TransportRoute],
    *,
    travel_days_by_edge: tuple[float, ...],
    maximum_response_days_by_state: np.ndarray,
    maritime_days_by_route: dict[str, float],
) -> np.ndarray:
    """Extend actual sea access within the capital's full response budget.

    Landing at a port never grants its whole island or continent. The sea
    crossing and every subsequent land traversal consume the same budget
    used for formation, including the capital-to-departure-port journey.
    """
    import heapq

    owners = np.asarray(region_owner, dtype=np.int32).copy()
    labels = np.asarray(control_labels)
    budgets = np.asarray(maximum_response_days_by_state, dtype=np.float64)
    edge_days = np.asarray(travel_days_by_edge, dtype=np.float64)
    if (edge_days.shape != (len(graph.edges),) or not np.isfinite(edge_days).all()
            or np.any(edge_days <= 0) or budgets.shape != core_region_by_state.shape
            or not np.isfinite(budgets[1:]).all() or np.any(budgets[1:] <= 0)):
        raise ValueError("maritime reach requires finite positive travel costs and state budgets")
    by_id = {item.identifier: item for item in settlements}
    neighbors = [[] for _ in range(graph.region_count + 1)]
    for edge, days in zip(graph.edges, edge_days, strict=True):
        neighbors[edge.first].append((edge.second, float(days), False))
        neighbors[edge.second].append((edge.first, float(days), False))
    state_domain = graph.domain_by_region[core_region_by_state]
    for route in routes:
        if route.mode != "sea" or route.target_settlement_id is None:
            continue
        first = by_id.get(route.source_settlement_id)
        second = by_id.get(route.target_settlement_id)
        if first is None or second is None:
            continue
        a = int(labels[first.row, first.column])
        b = int(labels[second.row, second.column])
        if a <= 0 or b <= 0 or a == b:
            continue
        days = float(maritime_days_by_route[route.identifier])
        if not math.isfinite(days) or days <= 0:
            raise ValueError("sea-route response costs must be finite and positive")
        neighbors[a].append((b, days, True))
        neighbors[b].append((a, days, True))

    distances = np.full(graph.region_count + 1, math.inf)
    pending = []
    # Establish capital-to-port costs through existing domestic territory.
    # Only an actual sea edge may initiate a new detached claim.
    for state, core in enumerate(core_region_by_state[1:], 1):
        core = int(core)
        if core <= 0 or owners[core] != state:
            raise ValueError("a maritime state must retain its capital region")
        queue = [(0., core)]
        distances[core] = 0.
        while queue:
            cost, current = heapq.heappop(queue)
            if cost != distances[current]:
                continue
            for neighbor, days, is_sea in neighbors[current]:
                candidate = cost + days
                if candidate > budgets[state]:
                    continue
                if owners[neighbor] == state and candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
                elif (is_sea and owners[neighbor] == 0
                      and graph.domain_by_region[neighbor] == state_domain[state]):
                    heapq.heappush(pending, (candidate, state, neighbor))
    while pending:
        cost, state, current = heapq.heappop(pending)
        if owners[current] not in (0, state) or cost >= distances[current]:
            continue
        owners[current], distances[current] = state, cost
        for neighbor, days, _is_sea in neighbors[current]:
            candidate = cost + days
            if (owners[neighbor] in (0, state) and candidate <= budgets[state]
                    and candidate < distances[neighbor]
                    and graph.domain_by_region[neighbor] == state_domain[state]):
                heapq.heappush(pending, (candidate, state, neighbor))
    return owners


def simulate_state_formation(
    graph: ControlRegionGraph,
    *,
    core_region_by_state: np.ndarray,
    state_strength: np.ndarray,
    travel_days_by_edge: tuple[float, ...],
    maximum_response_days_by_state: np.ndarray,
    maximum_rounds: int | None = None,
    institutional_regions: dict[int, int] | None = None,
) -> StateFormationResult:
    """Form contiguous states through simultaneous adjacent-region expansion."""

    cores = np.asarray(core_region_by_state, dtype=np.int32)
    strengths = np.asarray(state_strength, dtype=np.float64)
    budgets = np.asarray(maximum_response_days_by_state, dtype=np.float64)
    edge_days = np.asarray(travel_days_by_edge, dtype=np.float64)
    if cores.ndim != 1 or strengths.shape != cores.shape or cores.size < 2:
        raise ValueError("state cores and strengths must be aligned sentinel vectors")
    if int(cores[0]) != 0 or float(strengths[0]) != 0.0:
        raise ValueError("state formation vectors require a zero sentinel")
    if np.any(cores[1:] < 1) or np.any(cores[1:] > graph.region_count):
        raise ValueError("state cores must reference control regions")
    if len(set(int(value) for value in cores[1:])) != cores.size - 1:
        raise ValueError("each state requires a distinct control-region core")
    if np.any(~np.isfinite(strengths[1:])) or np.any(strengths[1:] <= 0.0):
        raise ValueError("state strengths must be finite and positive")
    if budgets.shape != cores.shape or np.any(np.isnan(budgets)) or np.any(budgets[1:] <= 0):
        raise ValueError("state response budgets must be positive and match the cores")
    if edge_days.shape != (len(graph.edges),) or np.any(~np.isfinite(edge_days)) or np.any(edge_days <= 0):
        raise ValueError("every control-region edge needs a positive travel-day estimate")
    days_for_pair = {(edge.first, edge.second): float(days) for edge, days in zip(graph.edges, edge_days, strict=True)}

    state_count = cores.size - 1
    neighbors: list[list[tuple[int, ControlRegionEdge]]] = [
        [] for _ in range(graph.region_count + 1)
    ]
    for edge in graph.edges:
        neighbors[edge.first].append((edge.second, edge))
        neighbors[edge.second].append((edge.first, edge))

    owners = np.zeros(graph.region_count + 1, dtype=np.int32)
    costs = np.full(graph.region_count + 1, np.inf, dtype=np.float64)
    rounds = np.full(graph.region_count + 1, -1, dtype=np.int32)
    depths = np.zeros(graph.region_count + 1, dtype=np.int32)
    response_days = np.full(graph.region_count + 1, np.inf, dtype=np.float64)
    state_regions: list[set[int]] = [set() for _ in range(state_count + 1)]
    state_resources = np.zeros(state_count + 1, dtype=np.float64)
    state_areas = np.zeros(state_count + 1, dtype=np.float64)
    state_domains = np.zeros(state_count + 1, dtype=np.int32)
    core_state_by_region = np.zeros(graph.region_count + 1, dtype=np.int32)
    for state in range(1, state_count + 1):
        region = int(cores[state])
        owners[region] = state
        costs[region] = 0.0
        rounds[region] = 0
        response_days[region] = 0
        state_regions[state].add(region)
        state_resources[state] = max(0.05, float(graph.resource_by_region[region]))
        state_areas[state] = max(1.0, float(graph.area_by_region[region]))
        state_domains[state] = int(graph.domain_by_region[region])
        core_state_by_region[region] = state
    for region, state in (institutional_regions or {}).items():
        if not 0 < region <= graph.region_count or not 0 < state <= state_count:
            raise ValueError("institutional regions must reference available regions and states")
        if int(graph.domain_by_region[region]) != int(state_domains[state]):
            raise ValueError("institutional regions cannot cross a physical or cultural owner domain")
        if int(owners[region]) not in (0, state):
            raise ValueError("institutional region conflicts with a rival political core")
        if int(owners[region]) == state:
            continue
        owners[region] = state
        costs[region] = 0.0
        rounds[region] = 0
        response_days[region] = 0.0
        core_state_by_region[region] = state
        state_regions[state].add(region)
        state_resources[state] += max(0.05, float(graph.resource_by_region[region]))
        state_areas[state] += max(1.0, float(graph.area_by_region[region]))
    capacity_targets = _state_capacity_targets(graph, cores, strengths)
    mean_region_area = max(1.0, float(np.mean(graph.area_by_region[1:])))
    limit = maximum_rounds or max(8, graph.region_count * 3)

    for current_round in range(1, limit + 1):
        if not np.any(owners[1:] == 0):
            break
        best_by_state: dict[int, tuple[float, int, int, ControlRegionEdge]] = {}
        for target in np.flatnonzero(owners == 0):
            target_domain = int(graph.domain_by_region[target])
            if target_domain <= 0:
                continue
            same_state_contacts: dict[int, int] = {}
            proposals: list[tuple[float, int, int, ControlRegionEdge]] = []
            for parent, edge in neighbors[int(target)]:
                state = int(owners[parent])
                if state <= 0 or state_domains[state] != target_domain:
                    continue
                next_days = response_days[parent] + days_for_pair[(edge.first, edge.second)]
                if next_days > budgets[state]:
                    continue
                same_state_contacts[state] = same_state_contacts.get(state, 0) + 1
                power = float(strengths[state]) * (
                    1.0 + 0.16 * math.log1p(state_resources[state])
                )
                load = len(state_regions[state]) / max(1.0, capacity_targets[state])
                depth = int(depths[parent]) + 1
                travel = (
                    0.50
                    + math.sqrt(max(1.0, float(graph.area_by_region[target])) / mean_region_area)
                ) / max(0.35, power)
                administrative = 0.30 * load * load + 0.055 * depth * depth
                attraction = 0.16 * math.log1p(
                    max(0.0, float(graph.resource_by_region[target]))
                )
                contact_bonus = 0.10 * math.log1p(edge.contact_cells)
                score = (
                    float(costs[parent])
                    + edge.crossing_cost
                    + travel
                    + administrative
                    - attraction
                    - contact_bonus
                    + .6 * next_days / max(1.0, budgets[state])
                )
                proposals.append((score, state, parent, edge))
            if not proposals:
                continue
            score, state, parent, edge = min(proposals, key=lambda item: (item[0], item[1]))
            score -= 0.18 * max(0, same_state_contacts[state] - 1)
            current = best_by_state.get(state)
            proposal = (score, int(target), parent, edge)
            if current is None or proposal[:2] < current[:2]:
                best_by_state[state] = proposal
        if not best_by_state:
            break

        best_by_target: dict[int, tuple[float, int, int, ControlRegionEdge]] = {}
        for state, (score, target, parent, edge) in best_by_state.items():
            proposal = (score, state, parent, edge)
            current = best_by_target.get(target)
            if current is None or proposal[:2] < current[:2]:
                best_by_target[target] = proposal
        for target, (score, state, parent, _edge) in sorted(best_by_target.items()):
            if owners[target] != 0:
                continue
            owners[target] = state
            costs[target] = score
            rounds[target] = current_round
            depths[target] = depths[parent] + 1
            response_days[target] = response_days[parent] + days_for_pair[(_edge.first, _edge.second)]
            state_regions[state].add(target)
            state_resources[state] += max(0.0, float(graph.resource_by_region[target]))
            state_areas[state] += max(1.0, float(graph.area_by_region[target]))

    # Disconnected components have no demonstrated administrative connection.
    # They remain frontier until an explicit maritime route is applied by the
    # political layer. Region identifiers are never a geographic distance.

    owners = settle_state_boundaries(
        graph,
        owners,
        core_region_by_state=cores,
        institutional_regions=institutional_regions,
    )
    state_regions = [set() for _ in range(state_count + 1)]
    state_resources[:] = 0.0
    for region in range(1, graph.region_count + 1):
        state = int(owners[region])
        state_regions[state].add(region)
        state_resources[state] += max(
            0.0,
            float(graph.resource_by_region[region]),
        )

    # Bounded border wars let a much stronger polity exploit a real bridge or
    # pass.  Only fringe regions can change hands; every rival core survives
    # and the defender remains connected to it.
    for _epoch in range(3):
        changed = False
        for edge in sorted(graph.edges, key=lambda item: (item.crossing_cost, item.first, item.second)):
            first_owner = int(owners[edge.first])
            second_owner = int(owners[edge.second])
            if first_owner <= 0 or second_owner <= 0 or first_owner == second_owner:
                continue
            if state_domains[first_owner] != state_domains[second_owner]:
                continue
            first_power = float(strengths[first_owner]) * (
                1.0 + 0.12 * math.log1p(state_resources[first_owner])
            )
            second_power = float(strengths[second_owner]) * (
                1.0 + 0.12 * math.log1p(state_resources[second_owner])
            )
            challenger, defender, target = (
                (first_owner, second_owner, edge.second)
                if first_power > second_power
                else (second_owner, first_owner, edge.first)
            )
            stronger = max(first_power, second_power)
            weaker = min(first_power, second_power)
            threshold = 1.65 + 0.18 * edge.crossing_cost
            if stronger / max(weaker, 1.0e-12) <= threshold:
                continue
            if core_state_by_region[target] > 0:
                continue
            defender_neighbors = [
                neighbor
                for neighbor, _neighbor_edge in neighbors[target]
                if int(owners[neighbor]) == defender
            ]
            if len(defender_neighbors) > 1:
                continue
            owners[target] = challenger
            state_regions[defender].discard(target)
            state_regions[challenger].add(target)
            resource_value = max(0.0, float(graph.resource_by_region[target]))
            state_resources[defender] = max(0.05, state_resources[defender] - resource_value)
            state_resources[challenger] += resource_value
            changed = True
        if not changed:
            break

    # Border transfers must not bypass the same communication constraint.
    # Recompute over each actual owned graph and leave unsupported regions
    # outside direct government. No ruler gains a remote province because the
    # allocation loop ran out of competing states.
    import heapq
    supported = np.zeros(graph.region_count + 1, dtype=bool)
    for state in range(1, state_count + 1):
        seat = int(cores[state])
        roots = {seat} | {region for region, owner in (institutional_regions or {}).items()
                          if owner == state}
        travel = dict.fromkeys(roots, 0.0)
        queue = [(0.0, region) for region in sorted(roots)]
        while queue:
            days, region = heapq.heappop(queue)
            if days != travel[region]:
                continue
            supported[region] = True
            for neighbor, edge in neighbors[region]:
                if int(owners[neighbor]) != state:
                    continue
                candidate = days + days_for_pair[(edge.first, edge.second)]
                if candidate <= budgets[state] and candidate < travel.get(neighbor, math.inf):
                    travel[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
    owners[~supported] = 0
    costs[~supported] = np.inf
    rounds[~supported] = -1
    return StateFormationResult(
        region_owner=owners,
        acquisition_cost=costs.astype(np.float32),
        acquisition_round=rounds,
    )


__all__ = [
    "ControlRegionEdge",
    "ControlRegionGraph",
    "StateFormationResult",
    "build_control_region_graph",
    "attach_maritime_control_regions",
    "distribute_state_cores",
    "repair_same_land_state_fragments",
    "settle_state_boundaries",
    "simulate_state_formation",
]
