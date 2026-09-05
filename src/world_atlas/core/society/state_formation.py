"""Path-dependent polity formation over settlement control regions.

The pixel grid still carries the physical truth, but countries do not compete
for pixels directly.  Settlements first receive connected local hinterlands.
Those hinterlands become graph nodes that states absorb over a bounded series
of simultaneous political rounds.  This preserves physical seams and makes
administrative reach, rather than geometric proximity, determine scale.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math

import numpy as np

from .model import immutable_array
from .spatial import connected_components


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
    for identifier in range(1, region_count + 1):
        values = np.unique(domains[labels == identifier])
        values = values[values > 0]
        if values.size > 1:
            raise ValueError("one control region cannot cross owner domains")
        if values.size == 1:
            domain_by_region[identifier] = int(values[0])

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
    if np.any(owners[1:] <= 0) or np.any(owners[1:] >= cores.size):
        raise ValueError("every control region must reference an available state")
    core_state_by_region = np.zeros(graph.region_count + 1, dtype=np.int32)
    for state in range(1, cores.size):
        region = int(cores[state])
        if region < 1 or region > graph.region_count:
            raise ValueError("state-boundary core lies outside the control graph")
        core_state_by_region[region] = state
        if int(owners[region]) != state:
            raise ValueError("state-boundary core must remain owned by its state")
    neighbors: list[list[tuple[int, ControlRegionEdge]]] = [
        [] for _ in range(graph.region_count + 1)
    ]
    for edge in graph.edges:
        neighbors[edge.first].append((edge.second, edge))
        neighbors[edge.second].append((edge.first, edge))

    def removal_keeps_state_connected(region: int, state: int) -> bool:
        """Return whether a whole hinterland can change hands safely.

        Counting same-owner contacts mistakes any region with two contacts for
        an articulation point.  The actual political condition is graph
        connectivity after removal, so test exactly that on the small control
        graph.  The destination is connected automatically because it must
        already border ``region``.
        """

        remaining = {
            candidate
            for candidate in range(1, graph.region_count + 1)
            if candidate != region and int(owners[candidate]) == state
        }
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
            if core_state_by_region[region] > 0:
                continue
            defender = int(owners[region])
            if not removal_keeps_state_connected(region, defender):
                continue
            current_energy = local_energy(region, defender)
            candidates = sorted(
                {
                    int(owners[neighbor])
                    for neighbor, _edge in neighbors[region]
                    if int(owners[neighbor]) != defender
                    and graph.domain_by_region[neighbor]
                    == graph.domain_by_region[region]
                }
            )
            for candidate in candidates:
                candidate_energy = local_energy(region, candidate)
                improvement = current_energy - candidate_energy
                if improvement > 0.35:
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
            changed = True
        if not changed:
            break
    return owners


def repair_same_land_state_fragments(
    state_id: np.ndarray,
    political_core_cells: np.ndarray,
    land_component_id: np.ndarray,
) -> np.ndarray:
    """Absorb detached same-land fragments without deleting overseas realms."""

    labels = np.asarray(state_id)
    cores = np.asarray(political_core_cells, dtype=bool)
    land_components = np.asarray(land_component_id, dtype=np.int32)
    if labels.ndim != 2 or cores.shape != labels.shape or land_components.shape != labels.shape:
        raise ValueError("state-fragment repair fields must align")
    result = labels.astype(np.int16, copy=True)
    height, width = labels.shape
    for state in sorted(int(value) for value in np.unique(result) if value > 0):
        component_id, sizes = connected_components(result == state)
        active = [
            identifier
            for identifier, size in enumerate(sizes, start=1)
            if int(size) > 0
        ]
        if len(active) <= 1:
            continue
        core_components = [
            identifier
            for identifier in active
            if bool(np.any(cores & (component_id == identifier)))
        ]
        if not core_components:
            raise ValueError("every fragmented state requires a political core")
        main = max(
            core_components,
            key=lambda identifier: int(sizes[identifier - 1]),
        )
        main_land_values = land_components[component_id == main]
        main_land_values = main_land_values[main_land_values > 0]
        main_land = (
            int(np.bincount(main_land_values).argmax())
            if main_land_values.size
            else 0
        )
        for identifier in active:
            if identifier == main or identifier in core_components:
                continue
            region = component_id == identifier
            values = land_components[region]
            values = values[values > 0]
            local_land = int(np.bincount(values).argmax()) if values.size else 0
            if local_land <= 0 or local_land != main_land:
                continue
            contacts: Counter[int] = Counter()
            rows, columns = np.nonzero(region)
            for row, column in zip(rows, columns, strict=True):
                for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                    next_row = int(row) + dy
                    if next_row < 0 or next_row >= height:
                        continue
                    next_column = (int(column) + dx) % width
                    neighbor = int(result[next_row, next_column])
                    if neighbor > 0 and neighbor != state:
                        contacts[neighbor] += 1
            if not contacts:
                continue
            target = min(
                contacts,
                key=lambda candidate: (-contacts[candidate], candidate),
            )
            result[region] = target
    return result


def simulate_state_formation(
    graph: ControlRegionGraph,
    *,
    core_region_by_state: np.ndarray,
    state_strength: np.ndarray,
    maximum_rounds: int | None = None,
) -> StateFormationResult:
    """Form contiguous states through simultaneous adjacent-region expansion."""

    cores = np.asarray(core_region_by_state, dtype=np.int32)
    strengths = np.asarray(state_strength, dtype=np.float64)
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
        state_regions[state].add(region)
        state_resources[state] = max(0.05, float(graph.resource_by_region[region]))
        state_areas[state] = max(1.0, float(graph.area_by_region[region]))
        state_domains[state] = int(graph.domain_by_region[region])
        core_state_by_region[region] = state
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
            state_regions[state].add(target)
            state_resources[state] += max(0.0, float(graph.resource_by_region[target]))
            state_areas[state] += max(1.0, float(graph.area_by_region[target]))

    # A domain may include an island or isolated local graph component without
    # a political core.  Attach the whole residual component to the nearest
    # same-domain state by region identifier; never invent a land corridor.
    unassigned = set(int(value) for value in np.flatnonzero(owners == 0) if value > 0)
    while unassigned:
        start = min(unassigned)
        component: set[int] = set()
        queue = [start]
        domain = int(graph.domain_by_region[start])
        while queue:
            region = queue.pop()
            if region in component or region not in unassigned:
                continue
            component.add(region)
            queue.extend(
                neighbor
                for neighbor, _edge in neighbors[region]
                if neighbor in unassigned and int(graph.domain_by_region[neighbor]) == domain
            )
        candidates = [
            state
            for state in range(1, state_count + 1)
            if state_domains[state] == domain
        ]
        if not candidates:
            raise ValueError("a control-region domain has no political core")
        state = min(
            candidates,
            key=lambda identifier: (
                min(abs(int(cores[identifier]) - region) for region in component),
                identifier,
            ),
        )
        for region in component:
            owners[region] = state
            costs[region] = float("inf")
            rounds[region] = limit + 1
        unassigned -= component

    owners = settle_state_boundaries(
        graph,
        owners,
        core_region_by_state=cores,
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
    "repair_same_land_state_fragments",
    "settle_state_boundaries",
    "simulate_state_formation",
]
