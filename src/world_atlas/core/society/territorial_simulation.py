"""Native-resolution territorial growth over physical and transport constraints."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .model import Bridge, TransportRoute, immutable_array
from .spatial import connected_components

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

# The other vertex of each native right triangle, seen from its target.
# Continuous updates become causal when its axial vertex is accepted after
# the diagonal vertex. The complementary update has an endpoint minimum and
# is already represented by the ordinary legal edge candidate.
_FRONT_PEERS = {
    1: ((5, 1, -1), (7, 1, 1)),
    3: ((2, -1, 1), (7, 1, 1)),
    4: ((0, -1, -1), (5, 1, -1)),
    6: ((0, -1, -1), (2, -1, 1)),
}


def _continuous_front_time(
    row, column, direction, owner, time, edge_slowness, strength,
    friction, transitions, costs, labels, accepted, *, symmetric_crossings=False,
):
    """Return the causal arrival time from an accepted triangular front.

    Arrival time varies linearly on the segment joining the accepted axial
    and diagonal vertices. Minimizing travel from any point on that segment
    to the target gives time + sqrt(slowness**2 - time_difference**2).
    It is an interior minimum only below a 45-degree incidence angle.

    The upper envelope of the two legal incoming edge slownesses bounds the
    angular crossing cost. A cheap bridge edge therefore cannot discount its
    unbridged neighbour; terrain barriers and owner domains retain their
    authored meaning. The original edge candidates remain available.
    """
    best = time + edge_slowness
    height, width = friction.shape
    for peer_direction, dy, dx in _FRONT_PEERS[direction]:
        peer_row = row + dy
        if peer_row < 0 or peer_row >= height:
            continue
        peer_column = (column + dx) % width
        if not accepted[peer_row, peer_column] or labels[peer_row, peer_column] != owner:
            continue
        peer_time = float(costs[peer_row, peer_column])
        difference = time - peer_time
        crossing = float(transitions[7-peer_direction, peer_row, peer_column])
        if symmetric_crossings:
            crossing = max(crossing, float(transitions[peer_direction, row, column]))
        peer_slowness = (
            .5 * (float(friction[peer_row, peer_column]) + float(friction[row, column])) / strength
            + crossing
        )
        slowness = max(edge_slowness, peer_slowness)
        if difference < 0.0 or difference >= slowness / math.sqrt(2.0):
            continue
        candidate = time + math.sqrt(slowness*slowness - difference*difference)
        if candidate < best:
            best = candidate
    return best


@dataclass(frozen=True, slots=True)
class TerritorySeed:
    row: int
    column: int
    owner: int
    strength: float = 1.0
    domain: int = 0

    def __post_init__(self) -> None:
        if self.row < 0 or self.column < 0:
            raise ValueError("territory seed coordinates must be non-negative")
        if self.owner < 1:
            raise ValueError("territory seed owners must be positive")
        if not np.isfinite(self.strength) or self.strength <= 0.0:
            raise ValueError("territory seed strength must be finite and positive")
        if self.domain < 0:
            raise ValueError("territory seed domain must be non-negative")


@dataclass(frozen=True, slots=True)
class TerritorySimulation:
    valid: np.ndarray
    friction: np.ndarray
    transition_penalty: np.ndarray
    road_access: np.ndarray
    bridge_edges: np.ndarray
    owner_constraint: np.ndarray | None = None

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid)
        friction = np.asarray(self.friction)
        transitions = np.asarray(self.transition_penalty)
        roads = np.asarray(self.road_access)
        bridges = np.asarray(self.bridge_edges)
        if valid.ndim != 2 or valid.dtype != np.bool_:
            raise ValueError("territory valid mask must be a two-dimensional bool array")
        if friction.shape != valid.shape or roads.shape != valid.shape:
            raise ValueError("territory cell fields must match the valid mask")
        if transitions.shape != (8, *valid.shape) or bridges.shape != transitions.shape:
            raise ValueError("territory edge fields must have shape (8, H, W)")
        if np.any(~np.isfinite(friction[valid])) or np.any(friction[valid] <= 0.0):
            raise ValueError("valid territory friction must be finite and positive")
        if np.any(~np.isfinite(transitions)) or np.any(transitions < 0.0):
            raise ValueError("territory transition penalties must be finite and non-negative")
        if np.any(~np.isfinite(roads)) or np.any((roads < 0.0) | (roads > 1.0)):
            raise ValueError("territory road access must lie in 0..1")
        if np.any(~np.isfinite(bridges)) or np.any((bridges < 0.0) | (bridges > 0.75)):
            raise ValueError("territory bridge discounts must lie in 0..0.75")
        constraint = self.owner_constraint
        if constraint is not None and np.asarray(constraint).shape != valid.shape:
            raise ValueError("territory owner constraint must match the valid mask")
        object.__setattr__(self, "valid", immutable_array(valid, dtype=np.dtype(np.bool_)))
        object.__setattr__(self, "friction", immutable_array(friction, dtype=np.dtype(np.float32)))
        object.__setattr__(
            self,
            "transition_penalty",
            immutable_array(transitions, dtype=np.dtype(np.float32)),
        )
        object.__setattr__(self, "road_access", immutable_array(roads, dtype=np.dtype(np.float32)))
        object.__setattr__(self, "bridge_edges", immutable_array(bridges, dtype=np.dtype(np.float32)))
        if constraint is not None:
            object.__setattr__(
                self,
                "owner_constraint",
                immutable_array(np.asarray(constraint), dtype=np.dtype(np.int32)),
            )


@dataclass(frozen=True, slots=True)
class TerritoryResult:
    owner: np.ndarray
    cost: np.ndarray
    frontier: np.ndarray
    barriers: np.ndarray

    def __post_init__(self) -> None:
        owner = np.asarray(self.owner)
        expected = owner.shape
        arrays = (
            ("cost", np.asarray(self.cost)),
            ("frontier", np.asarray(self.frontier)),
            ("barriers", np.asarray(self.barriers)),
        )
        if owner.ndim != 2 or any(value.shape != expected for _name, value in arrays):
            raise ValueError("territory result fields must share a two-dimensional shape")
        object.__setattr__(self, "owner", immutable_array(owner, dtype=np.dtype(np.int32)))
        object.__setattr__(self, "cost", immutable_array(self.cost, dtype=np.dtype(np.float32)))
        object.__setattr__(self, "frontier", immutable_array(self.frontier, dtype=np.dtype(np.bool_)))
        object.__setattr__(self, "barriers", immutable_array(self.barriers, dtype=np.dtype(np.float32)))


def simulate_bounded_reach(
    simulation: TerritorySimulation,
    seeds: tuple[TerritorySeed, ...],
    *,
    maximum_cost: float,
) -> np.ndarray:
    """Return the cells reachable from any seed inside one travel-time budget.

    Unlike a geometric dilation, the reachable footprint follows the same
    terrain, road, river and bridge costs used by territorial expansion.  The
    search stops at ``maximum_cost`` so it can model an urban hinterland
    without allocating the whole map.
    """

    if not math.isfinite(maximum_cost) or maximum_cost <= 0.0:
        raise ValueError("maximum_cost must be finite and positive")
    if not seeds:
        return np.zeros(simulation.valid.shape, dtype=bool)
    coordinates = tuple((seed.row, seed.column) for seed in seeds)
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("territory seed coordinates must be unique")
    for seed in seeds:
        if (
            seed.row >= simulation.valid.shape[0]
            or seed.column >= simulation.valid.shape[1]
            or not bool(simulation.valid[seed.row, seed.column])
        ):
            raise ValueError("territory seeds must lie on valid cells")

    friction = simulation.friction.astype(np.float32) * (
        1.0 - 0.15 * simulation.road_access
    )
    transitions = simulation.transition_penalty.astype(np.float32) * (
        1.0 - simulation.bridge_edges
    )
    labels = np.zeros(simulation.valid.shape, dtype=np.int32)
    costs = np.full(simulation.valid.shape, np.inf, dtype=np.float64)
    accepted = np.zeros(simulation.valid.shape, dtype=bool)
    heap: list[tuple[float, int, int, int]] = []
    for label, seed in enumerate(seeds, start=1):
        labels[seed.row, seed.column] = label
        costs[seed.row, seed.column] = 0.0
        heapq.heappush(heap, (0.0, label, seed.row, seed.column))

    height, width = simulation.valid.shape
    while heap:
        cost, label, row, column = heapq.heappop(heap)
        if cost != costs[row, column] or label != int(labels[row, column]):
            continue
        if accepted[row, column]:
            continue
        accepted[row, column] = True
        seed = seeds[label - 1]
        for direction, (dy, dx, distance) in enumerate(_NEIGHBORS):
            next_row = row + dy
            if next_row < 0 or next_row >= height:
                continue
            next_column = (column + dx) % width
            if accepted[next_row, next_column] or not bool(simulation.valid[next_row, next_column]):
                continue
            if simulation.owner_constraint is not None:
                required = int(simulation.owner_constraint[next_row, next_column])
                if required > 0 and required != seed.domain:
                    continue
            cell_cost = 0.5 * (
                float(friction[row, column]) + float(friction[next_row, next_column])
            ) / seed.strength
            edge_cost = (
                cell_cost + float(transitions[direction, row, column])
            ) * distance
            next_cost = cost + edge_cost
            if direction in _FRONT_PEERS:
                next_cost = _continuous_front_time(
                    next_row, next_column, direction, label, cost, edge_cost, seed.strength,
                    friction, transitions, costs, labels, accepted,
                )
            if next_cost > maximum_cost + 1.0e-12:
                continue
            current_cost = float(costs[next_row, next_column])
            current_label = int(labels[next_row, next_column])
            if next_cost < current_cost - 1.0e-12 or (
                abs(next_cost - current_cost) <= 1.0e-12
                and (current_label <= 0 or label < current_label)
            ):
                costs[next_row, next_column] = next_cost
                labels[next_row, next_column] = label
                heapq.heappush(heap, (next_cost, label, next_row, next_column))
    return labels > 0


def simulate_territories(
    simulation: TerritorySimulation,
    seeds: tuple[TerritorySeed, ...],
) -> TerritoryResult:
    """Grow owners using continuous triangular fronts on native observations.

    Single-edge arrivals obey the authored eight directional crossing costs.
    Causal same-owner triangle updates propagate between those directions,
    removing the octile travel metric without perturbing displayed borders.
    """

    if not seeds:
        owner = np.where(simulation.valid, 0, -1).astype(np.int32)
        return TerritoryResult(
            owner=owner,
            cost=np.full(owner.shape, np.inf, dtype=np.float32),
            frontier=simulation.valid,
            barriers=np.max(simulation.transition_penalty, axis=0),
        )
    coordinates = tuple((seed.row, seed.column) for seed in seeds)
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("territory seed coordinates must be unique")
    for seed in seeds:
        if (
            seed.row >= simulation.valid.shape[0]
            or seed.column >= simulation.valid.shape[1]
            or not bool(simulation.valid[seed.row, seed.column])
        ):
            raise ValueError("territory seeds must lie on valid cells")

    friction = simulation.friction.astype(np.float32) * (
        1.0 - 0.15 * simulation.road_access
    )
    transitions = simulation.transition_penalty.astype(np.float32) * (
        1.0 - simulation.bridge_edges
    )
    labels = np.full(simulation.valid.shape, -1, dtype=np.int32)
    costs = np.full(simulation.valid.shape, np.inf, dtype=np.float64)
    accepted = np.zeros(simulation.valid.shape, dtype=bool)
    heap: list[tuple[float, int, int, int]] = []
    seed_owner = {coordinate: index for index, coordinate in enumerate(coordinates, 1)}
    for index, seed in enumerate(seeds, start=1):
        labels[seed.row, seed.column] = index
        costs[seed.row, seed.column] = 0.0
        heapq.heappush(heap, (0.0, index, seed.row, seed.column))
    height, width = simulation.valid.shape
    while heap:
        cost, label, row, column = heapq.heappop(heap)
        if cost != costs[row, column] or label != int(labels[row, column]):
            continue
        if accepted[row, column]:
            continue
        accepted[row, column] = True
        seed = seeds[label - 1]
        for direction, (dy, dx, distance) in enumerate(_NEIGHBORS):
            next_row = row + dy
            if next_row < 0 or next_row >= height:
                continue
            next_column = (column + dx) % width
            if accepted[next_row, next_column] or not bool(simulation.valid[next_row, next_column]):
                continue
            fixed_seed = seed_owner.get((next_row, next_column))
            if fixed_seed is not None and fixed_seed != label:
                continue
            if simulation.owner_constraint is not None:
                required = int(simulation.owner_constraint[next_row, next_column])
                if required > 0 and required != seed.domain:
                    continue
            cell_cost = 0.5 * (
                float(friction[row, column]) + float(friction[next_row, next_column])
            ) / seed.strength
            # Administrative strength accelerates movement through ordinary
            # terrain, but cannot make a mountain or unbridged river cease to
            # be a physical barrier.  Bridge discounts have already been
            # applied only to their exact crossing edges above.
            edge_cost = cell_cost + float(transitions[direction, row, column])
            next_cost = cost + edge_cost * distance
            if direction in _FRONT_PEERS:
                next_cost = _continuous_front_time(
                    next_row, next_column, direction, label, cost, edge_cost, seed.strength,
                    friction, transitions, costs, labels, accepted,
                )
            current_cost = float(costs[next_row, next_column])
            current_label = int(labels[next_row, next_column])
            if next_cost < current_cost - 1.0e-12 or (
                abs(next_cost - current_cost) <= 1.0e-12 and label < current_label
            ):
                costs[next_row, next_column] = next_cost
                labels[next_row, next_column] = label
                heapq.heappush(heap, (next_cost, label, next_row, next_column))
    if np.any(simulation.valid & (labels <= 0)):
        constraint = simulation.owner_constraint
        domain_values = (0,) if constraint is None else tuple(
            int(value) for value in np.unique(constraint[simulation.valid])
        )
        for domain in domain_values:
            domain_mask = simulation.valid & (labels <= 0)
            if constraint is not None:
                domain_mask &= constraint == domain
            components, sizes = connected_components(domain_mask)
            seed_indices = [
                index
                for index, seed in enumerate(seeds, start=1)
                if domain <= 0 or seed.domain == domain
            ]
            if not seed_indices:
                continue
            for identifier, size in enumerate(sizes, start=1):
                if int(size) <= 0:
                    continue
                region = components == identifier
                rows, columns = np.nonzero(region)
                centre_row = float(np.mean(rows))
                angles = columns.astype(np.float64) * (2.0 * np.pi / width)
                centre_column = float(
                    (math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
                    % (2.0 * np.pi))
                    * width
                    / (2.0 * np.pi)
                )
                nearest = min(
                    seed_indices,
                    key=lambda index: (
                        math.hypot(
                            seeds[index - 1].row - centre_row,
                            min(
                                abs(seeds[index - 1].column - centre_column),
                                width - abs(seeds[index - 1].column - centre_column),
                            ),
                        ),
                        index,
                    ),
                )
                labels[region] = nearest
    owner_lookup = np.asarray([0, *(seed.owner for seed in seeds)], dtype=np.int32)
    owner = np.where(labels > 0, owner_lookup[np.maximum(labels, 0)], -1).astype(np.int32)
    frontier = simulation.valid & (owner <= 0)
    return TerritoryResult(
        owner=owner,
        cost=costs.astype(np.float32),
        frontier=frontier,
        barriers=np.max(simulation.transition_penalty, axis=0),
    )


def boundary_alignment(
    labels: np.ndarray,
    transition_penalty: np.ndarray,
    *,
    barrier_threshold: float = 5.0,
) -> float:
    """Return the share of internal border edges supported by a physical barrier."""

    owners = np.asarray(labels)
    transitions = np.asarray(transition_penalty)
    if owners.ndim != 2 or transitions.shape != (8, *owners.shape):
        raise ValueError("boundary fields must align")
    supported = 0
    total = 0
    east = (
        (owners > 0)
        & (np.roll(owners, -1, axis=1) > 0)
        & (owners != np.roll(owners, -1, axis=1))
    )
    total += int(np.count_nonzero(east))
    east_barrier = np.maximum(transitions[4], np.roll(transitions[3], -1, axis=1))
    supported += int(np.count_nonzero(east & (east_barrier >= barrier_threshold)))
    south = (
        (owners[:-1, :] > 0)
        & (owners[1:, :] > 0)
        & (owners[:-1, :] != owners[1:, :])
    )
    total += int(np.count_nonzero(south))
    south_barrier = np.maximum(transitions[6, :-1, :], transitions[1, 1:, :])
    supported += int(np.count_nonzero(south & (south_barrier >= barrier_threshold)))
    return 1.0 if total == 0 else supported / total


def _longest_true_run(values: np.ndarray, *, cyclic: bool) -> int:
    flags = np.asarray(values, dtype=bool).reshape(-1)
    if flags.size == 0 or not bool(np.any(flags)):
        return 0
    scan = np.concatenate((flags, flags)) if cyclic else flags
    best = 0
    current = 0
    limit = int(flags.size)
    for value in scan:
        current = current + 1 if bool(value) else 0
        best = max(best, min(current, limit))
    return best


def longest_unjustified_straight_run(
    labels: np.ndarray,
    transition_penalty: np.ndarray,
    *,
    barrier_threshold: float = 5.0,
) -> int:
    """Measure the longest axis-aligned internal border lacking terrain support."""

    owners = np.asarray(labels)
    transitions = np.asarray(transition_penalty)
    if owners.ndim != 2 or transitions.shape != (8, *owners.shape):
        raise ValueError("straight-run fields must align")
    east_boundary = (
        (owners > 0)
        & (np.roll(owners, -1, axis=1) > 0)
        & (owners != np.roll(owners, -1, axis=1))
    )
    east_barrier = np.maximum(transitions[4], np.roll(transitions[3], -1, axis=1))
    unsupported_east = east_boundary & (east_barrier < barrier_threshold)
    south_boundary = (
        (owners[:-1, :] > 0)
        & (owners[1:, :] > 0)
        & (owners[:-1, :] != owners[1:, :])
    )
    south_barrier = np.maximum(transitions[6, :-1, :], transitions[1, 1:, :])
    unsupported_south = south_boundary & (south_barrier < barrier_threshold)
    vertical = max(
        (_longest_true_run(unsupported_east[:, column], cyclic=False)
         for column in range(owners.shape[1])),
        default=0,
    )
    horizontal = max(
        (_longest_true_run(unsupported_south[row], cyclic=True)
         for row in range(max(0, owners.shape[0] - 1))),
        default=0,
    )
    return max(vertical, horizontal)


def ordinary_frontier_components(
    owner: np.ndarray,
    valid: np.ndarray,
    extreme_environment: np.ndarray,
    *,
    maximum_size: int = 512,
) -> int:
    """Count small ungoverned land components without extreme-terrain cause."""

    owners = np.asarray(owner)
    allowed = np.asarray(valid, dtype=bool)
    extreme = np.asarray(extreme_environment, dtype=bool)
    if owners.ndim != 2 or allowed.shape != owners.shape or extreme.shape != owners.shape:
        raise ValueError("ordinary-frontier fields must align")
    if maximum_size < 1:
        raise ValueError("maximum_size must be positive")
    components, sizes = connected_components(allowed & (owners == 0))
    count = 0
    for identifier, size in enumerate(sizes, start=1):
        if int(size) > maximum_size:
            continue
        region = components == identifier
        if float(np.mean(extreme[region])) < 0.55:
            count += 1
    return count


def extreme_frontier_environment(
    valid: np.ndarray,
    snow: np.ndarray,
    elevation: np.ndarray,
    annual_precipitation: np.ndarray,
    land_potential: np.ndarray,
    population_density: np.ndarray,
    river_order: np.ndarray,
) -> np.ndarray:
    """Identify land where permanent government may physically remain absent."""

    allowed = np.asarray(valid, dtype=bool)
    fields = tuple(
        np.asarray(field)
        for field in (
            snow,
            elevation,
            annual_precipitation,
            land_potential,
            population_density,
            river_order,
        )
    )
    if allowed.ndim != 2 or any(field.shape != allowed.shape for field in fields):
        raise ValueError("extreme-frontier fields must align")
    snow_field, heights, annual, potential, population, rivers = fields
    if (not np.issubdtype(population.dtype, np.floating)
            or np.any(~np.isfinite(population)) or np.any(population < 0.0)):
        raise ValueError("extreme-frontier density must contain finite non-negative floating persons/km²")
    return allowed & (
        snow_field.astype(bool)
        | ((heights > 0.78) & (potential < 0.20) & (population <= 0.1))
        | (
            (annual < 0.025)
            & (potential < 0.15)
            & (population == 0)
            & (rivers == 0)
        )
    )


def bridge_transition_discounts(
    river_order: np.ndarray,
    routes: tuple[TransportRoute, ...],
    bridges: tuple[Bridge, ...],
) -> np.ndarray:
    """Return directional discounts only on the river edges a bridge crosses."""

    rivers = np.asarray(river_order)
    if rivers.ndim != 2:
        raise ValueError("river_order must be two-dimensional")
    discounts = np.zeros((8, *rivers.shape), dtype=np.float32)
    route_by_identifier = {route.identifier: route for route in routes}
    direction_by_delta = {
        (-1, -1): 0,
        (-1, 0): 1,
        (-1, 1): 2,
        (0, -1): 3,
        (0, 1): 4,
        (1, -1): 5,
        (1, 0): 6,
        (1, 1): 7,
    }
    opposite = (7, 6, 5, 4, 3, 2, 1, 0)
    strength = {"local": 0.55, "regional": 0.65, "trunk": 0.75}
    # Local import keeps the transport module independent from territorial
    # simulation while reusing its exact route rasterization contract.
    from .transport import _path_cells

    for bridge in bridges:
        route = route_by_identifier.get(bridge.route_identifier)
        if route is None or route.mode != "road":
            raise ValueError("bridge must reference an available road route")
        cells = _path_cells(route.path, rivers.shape)
        indices = [
            index
            for index, cell in enumerate(cells)
            if cell == (bridge.row, bridge.column)
        ]
        if not indices:
            # A D8 flow segment may cross a road between samples: its nearest
            # river-support sample need not be one of the road's raster cells.
            # Such a geometrically verified bridge has no raster bank penalty
            # to discount when the actual road cells contain no channel.
            distances = []
            for row, column in cells:
                dx = abs(column-bridge.column)
                dx = min(dx, rivers.shape[1]-dx)
                distances.append((row-bridge.row)**2 + dx*dx)
            nearest = int(np.argmin(distances)) if distances else -1
            if nearest < 0 or distances[nearest] > 2:
                raise ValueError("bridge river support must be adjacent to its road crossing")
            if int(rivers[cells[nearest]]) == 0:
                continue
            indices = [nearest]
        centre = indices[len(indices) // 2]
        start = centre
        end = centre
        while start > 0 and int(rivers[cells[start - 1]]) > 0:
            start -= 1
        while end + 1 < len(cells) and int(rivers[cells[end + 1]]) > 0:
            end += 1
        if start == 0 or end + 1 >= len(cells):
            raise ValueError("bridge route must reach land on both river banks")
        for index in range(start - 1, end + 1):
            first = cells[index]
            second = cells[index + 1]
            first_order, second_order = int(rivers[first]), int(rivers[second])
            if all((first_order >= threshold) == (second_order >= threshold)
                   for threshold in (2, 3)):
                continue
            dy = second[0] - first[0]
            dx = second[1] - first[1]
            if dx > rivers.shape[1] * 0.5:
                dx -= rivers.shape[1]
            elif dx < -rivers.shape[1] * 0.5:
                dx += rivers.shape[1]
            dy = int(np.sign(dy))
            dx = int(np.sign(dx))
            direction = direction_by_delta.get((dy, dx))
            if direction is None:
                continue
            value = np.float32(strength[bridge.importance])
            discounts[direction, first[0], first[1]] = max(
                discounts[direction, first[0], first[1]],
                value,
            )
            reverse = opposite[direction]
            discounts[reverse, second[0], second[1]] = max(
                discounts[reverse, second[0], second[1]],
                value,
            )
    return discounts


def fill_unassigned_territory(
    simulation: TerritorySimulation,
    owner: np.ndarray,
    fill_mask: np.ndarray,
    owner_domains: np.ndarray,
) -> np.ndarray:
    """Absorb accidental holes with the same physical costs as state growth.

    Existing ownership is immutable.  Only explicitly requested zero cells are
    filled, so a separately approved wilderness frontier remains untouched.
    """

    source = np.asarray(owner, dtype=np.int32)
    requested = np.asarray(fill_mask, dtype=bool)
    domains = np.asarray(owner_domains, dtype=np.int32)
    if source.shape != simulation.valid.shape or requested.shape != source.shape:
        raise ValueError("territory repair fields must align")
    positive = source > 0
    if np.any(source[positive] >= len(domains)):
        raise ValueError("territory owner-domain lookup does not cover labels")
    target = requested & simulation.valid & ~positive
    if not np.any(target):
        return source.copy()

    friction = simulation.friction.astype(np.float32) * (
        1.0 - 0.15 * simulation.road_access
    )
    transitions = simulation.transition_penalty.astype(np.float32) * (
        1.0 - simulation.bridge_edges
    )
    constraint = simulation.owner_constraint
    opposite = (7, 6, 5, 4, 3, 2, 1, 0)
    labels = source.copy()
    costs = np.full(source.shape, np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int, int]] = []
    height, width = source.shape

    def permitted(label: int, row: int, column: int) -> bool:
        if constraint is None:
            return True
        required = int(constraint[row, column])
        return required <= 0 or int(domains[label]) == required

    def step_cost(
        row: int,
        column: int,
        next_row: int,
        next_column: int,
        direction: int,
        distance: float,
    ) -> float:
        cell_cost = 0.5 * (
            float(friction[row, column]) + float(friction[next_row, next_column])
        )
        return (
            cell_cost + float(transitions[direction, row, column])
        ) * distance

    for row, column in zip(*np.nonzero(target), strict=True):
        active_row = int(row)
        active_column = int(column)
        for direction, (dy, dx, distance) in enumerate(_NEIGHBORS):
            neighbor_row = active_row + dy
            if neighbor_row < 0 or neighbor_row >= height:
                continue
            neighbor_column = (active_column + dx) % width
            label = int(source[neighbor_row, neighbor_column])
            if label <= 0 or not permitted(label, active_row, active_column):
                continue
            cost = step_cost(
                neighbor_row,
                neighbor_column,
                active_row,
                active_column,
                opposite[direction],
                distance,
            )
            if cost < costs[active_row, active_column] - 1.0e-12 or (
                abs(cost - costs[active_row, active_column]) <= 1.0e-12
                and label < int(labels[active_row, active_column])
            ):
                costs[active_row, active_column] = cost
                labels[active_row, active_column] = label
                heapq.heappush(
                    heap,
                    (cost, label, active_row, active_column),
                )

    while heap:
        cost, label, row, column = heapq.heappop(heap)
        if cost != costs[row, column] or label != int(labels[row, column]):
            continue
        for direction, (dy, dx, distance) in enumerate(_NEIGHBORS):
            next_row = row + dy
            if next_row < 0 or next_row >= height:
                continue
            next_column = (column + dx) % width
            if not bool(target[next_row, next_column]):
                continue
            if not permitted(label, next_row, next_column):
                continue
            next_cost = cost + step_cost(
                row,
                column,
                next_row,
                next_column,
                direction,
                distance,
            )
            current_cost = float(costs[next_row, next_column])
            current_label = int(labels[next_row, next_column])
            if next_cost < current_cost - 1.0e-12 or (
                abs(next_cost - current_cost) <= 1.0e-12
                and label < current_label
            ):
                costs[next_row, next_column] = next_cost
                labels[next_row, next_column] = label
                heapq.heappush(
                    heap,
                    (next_cost, label, next_row, next_column),
                )

    # A water-separated unseeded island has no land edge from which the heap
    # can enter.  Attach the island as a whole to the nearest compatible
    # polity; never draw a one-cell corridor across the water.
    unresolved = target & (labels <= 0)
    if np.any(unresolved):
        required_values = (
            (0,)
            if constraint is None
            else tuple(int(value) for value in np.unique(constraint[unresolved]))
        )
        for required in required_values:
            domain_unresolved = unresolved.copy()
            if constraint is not None:
                domain_unresolved &= constraint == required
            components, sizes = connected_components(domain_unresolved)
            candidates = tuple(
                identifier
                for identifier in sorted(int(value) for value in np.unique(source[positive]))
                if required <= 0 or int(domains[identifier]) == required
            )
            if not candidates:
                continue
            centres: dict[int, tuple[float, float]] = {}
            for identifier in candidates:
                rows, columns = np.nonzero(source == identifier)
                angles = columns.astype(np.float64) * (2.0 * np.pi / width)
                centres[identifier] = (
                    float(np.mean(rows)),
                    float(
                        (math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
                        % (2.0 * np.pi))
                        * width
                        / (2.0 * np.pi)
                    ),
                )
            for component_identifier, size in enumerate(sizes, start=1):
                if int(size) <= 0:
                    continue
                region = components == component_identifier
                rows, columns = np.nonzero(region)
                centre_row = float(np.mean(rows))
                angles = columns.astype(np.float64) * (2.0 * np.pi / width)
                centre_column = float(
                    (math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
                    % (2.0 * np.pi))
                    * width
                    / (2.0 * np.pi)
                )
                nearest = min(
                    candidates,
                    key=lambda identifier: (
                        math.hypot(
                            centres[identifier][0] - centre_row,
                            min(
                                abs(centres[identifier][1] - centre_column),
                                width - abs(centres[identifier][1] - centre_column),
                            ),
                        ),
                        identifier,
                    ),
                )
                labels[region] = nearest
    return labels.astype(np.int32)


def classify_justified_frontier(
    owner: np.ndarray,
    valid: np.ndarray,
    frontier_candidate: np.ndarray,
    extreme_environment: np.ndarray,
    land_potential: np.ndarray,
    accessibility: np.ndarray,
    transition_penalty: np.ndarray,
    anchors: np.ndarray,
    *,
    stateless_society_candidate: np.ndarray | None = None,
    minimum_large_component: int = 513,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve only physically defensible frontier from an already complete partition."""

    owners = np.asarray(owner, dtype=np.int32)
    allowed = np.asarray(valid, dtype=bool)
    candidates = np.asarray(frontier_candidate, dtype=bool) & allowed
    extreme = np.asarray(extreme_environment, dtype=bool)
    potential = np.asarray(land_potential, dtype=np.float32)
    access = np.asarray(accessibility, dtype=np.float32)
    transitions = np.asarray(transition_penalty, dtype=np.float32)
    anchored = np.asarray(anchors, dtype=bool)
    stateless = (
        np.zeros(owners.shape, dtype=bool)
        if stateless_society_candidate is None
        else np.asarray(stateless_society_candidate, dtype=bool)
    )
    shape = owners.shape
    if any(
        field.shape != shape
        for field in (allowed, candidates, extreme, potential, access, anchored, stateless)
    ):
        raise ValueError("frontier cell fields must align")
    if transitions.shape != (8, *shape):
        raise ValueError("frontier transition field must have shape (8, H, W)")
    if minimum_large_component < 1:
        raise ValueError("minimum_large_component must be positive")

    components, sizes = connected_components(candidates)
    frontier = np.zeros(shape, dtype=bool)
    opposite = (7, 6, 5, 4, 3, 2, 1, 0)

    def perimeter_barriers(region: np.ndarray) -> np.ndarray:
        """Return physical costs on cardinal edges leaving one component."""

        values: list[np.ndarray] = []
        height, width = region.shape
        for direction in (1, 3, 4, 6):
            dy, dx, _distance = _NEIGHBORS[direction]
            neighbor_region = np.roll(region, (-dy, -dx), axis=(0, 1))
            neighbor_allowed = np.roll(allowed, (-dy, -dx), axis=(0, 1))
            reverse_cost = np.roll(
                transitions[opposite[direction]],
                (-dy, -dx),
                axis=(0, 1),
            )
            edge = region & neighbor_allowed & ~neighbor_region
            if dy < 0:
                edge[0, :] = False
            elif dy > 0:
                edge[height - 1, :] = False
            costs = np.maximum(transitions[direction], reverse_cost)
            if np.any(edge):
                values.append(costs[edge])
        if not values:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(values).astype(np.float32, copy=False)

    for identifier, size in enumerate(sizes, start=1):
        region = components == identifier
        if not np.any(region) or np.any(anchored[region]):
            continue
        extreme_share = float(np.mean(extreme[region]))
        boundary_costs = perimeter_barriers(region)
        enclosure_supported = bool(
            boundary_costs.size
            and float(np.mean(boundary_costs)) >= 5.0
            and float(np.mean(boundary_costs >= 5.0)) >= 0.55
        )
        large_remote_enclosure = (
            int(size) >= minimum_large_component
            and float(np.mean(potential[region])) < 0.24
            and float(np.mean(access[region])) < 0.05
            and enclosure_supported
        )
        large_stateless_society = (
            int(size) >= minimum_large_component
            and float(np.mean(stateless[region])) >= 0.55
            and float(np.mean(potential[region])) < 0.38
            and float(np.mean(access[region])) < 0.065
        )
        if extreme_share >= 0.55 or large_remote_enclosure or large_stateless_society:
            frontier[region] = True
    result = owners.copy()
    result[frontier] = 0
    result[~allowed] = -1
    return result.astype(np.int32), frontier


__all__ = [
    "TerritoryResult",
    "TerritorySeed",
    "TerritorySimulation",
    "boundary_alignment",
    "bridge_transition_discounts",
    "classify_justified_frontier",
    "extreme_frontier_environment",
    "fill_unassigned_territory",
    "longest_unjustified_straight_run",
    "ordinary_frontier_components",
    "simulate_territories",
]
