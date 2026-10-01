"""Reconcile administrative components through real four-neighbour contact."""

from collections import Counter
from collections.abc import Mapping

import numpy as np

from .spatial import categorical_components


def maritime_component_support(component_grid, owners, core_components, settlements, routes):
    """Return the components reachable from their capital over recorded sea routes."""
    by_id = {item.identifier: item for item in settlements}
    evidence = []
    maritime_neighbors = [[] for _ in range(len(owners))]
    for route in routes:
        if route.mode != "sea" or route.target_settlement_id is None:
            continue
        first = by_id.get(route.source_settlement_id)
        second = by_id.get(route.target_settlement_id)
        if first is None or second is None:
            continue
        a = int(component_grid[first.row, first.column])
        b = int(component_grid[second.row, second.column])
        if a >= 0 and b >= 0 and owners[a] == owners[b]:
            evidence.append((a, b, route))
            maritime_neighbors[a].append(b)
            maritime_neighbors[b].append(a)
    queue = list(int(value) for value in core_components if value >= 0)
    reached = set(queue)
    while queue:
        current = queue.pop()
        for neighbor in maritime_neighbors[current]:
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    supported = np.zeros(len(owners), dtype=bool)
    supported[list(reached)] = True
    route_evidence = [dict(routeId=route.identifier, ownerIdentifier=int(owners[a]),
                           sourceSettlementId=route.source_settlement_id,
                           targetSettlementId=route.target_settlement_id)
                      for a, b, route in evidence if a != b and a in reached and b in reached]
    return supported, route_evidence


def reconcile_partition_components(
    labels: np.ndarray,
    seats: Mapping[int, tuple[int, int]],
    owner_domain: np.ndarray,
    land_component_id: np.ndarray,
    *,
    maritime_routes=(),
    settlements=(),
    prefer_local_seats: bool = False,
) -> np.ndarray:
    """Join inland fragments to a rooted neighbour, or retain them as frontier.

    Transfers happen on whole components and only towards an already rooted
    component of the same parent domain.  Consequently transfer order cannot
    create a new inland exclave or cross a cultural/country boundary.  Islands
    are retained; their political access must be established by the caller.
    """
    result = np.asarray(labels).copy()
    land = np.asarray(land_component_id)
    domains = np.asarray(owner_domain)
    if result.ndim != 2 or land.shape != result.shape or domains.ndim != 1:
        raise ValueError("administrative topology fields must align")
    nodes, component, owners = categorical_components(result)
    if not nodes.size:
        return result
    count = int(component.max()) + 1
    component_grid = np.full(result.shape, -1, dtype=np.int32)
    component_grid.ravel()[nodes] = component
    component_owner = np.zeros(count, dtype=np.int32)
    component_land = np.zeros(count, dtype=np.int32)
    np.maximum.at(component_owner, component, owners)
    np.maximum.at(component_land, component, land.ravel()[nodes])
    if component_owner.max() >= domains.size:
        raise ValueError("administrative domains must cover every owner")
    core_component = np.full(domains.size, -1, dtype=np.int32)
    for owner, cell in seats.items():
        if int(result[cell]) != owner:
            raise ValueError("administrative core must remain inside its territory")
        core_component[owner] = component_grid[cell]
    if np.any(core_component[component_owner] < 0):
        raise ValueError("every administrative owner requires a core")
    owner_core = core_component[component_owner]
    remote = component_land != component_land[owner_core]
    if prefer_local_seats:
        local_seat_land = component_land[core_component[core_component >= 0]]
        remote &= ~np.isin(component_land, local_seat_land)
    rooted = (np.arange(count) == owner_core) | remote
    supported, _evidence = maritime_component_support(
        component_grid, component_owner, core_component, settlements, maritime_routes,
    )
    rooted |= supported
    component_domain = domains[component_owner]

    # Collect each shared boundary once. Longitude wraps; latitude does not.
    pairs = []
    for first, second in (
        (component_grid, np.roll(component_grid, -1, axis=1)),
        (component_grid[:-1], component_grid[1:]),
    ):
        valid = (first >= 0) & (second >= 0) & (first != second)
        pairs.append(np.column_stack((first[valid], second[valid])))
    pairs = np.concatenate(pairs)
    contacts: list[Counter[int]] = [Counter() for _ in range(count)]
    if pairs.size:
        pairs.sort(axis=1)
        unique, lengths = np.unique(pairs, axis=0, return_counts=True)
        for (first, second), length in zip(unique, lengths, strict=True):
            if component_domain[first] == component_domain[second]:
                contacts[first][int(second)] = int(length)
                contacts[second][int(first)] = int(length)
    pending = set(int(value) for value in np.flatnonzero(~rooted))
    while pending:
        proposals = []
        for fragment in sorted(pending):
            support: Counter[int] = Counter()
            for neighbor, length in contacts[fragment].items():
                if rooted[neighbor]:
                    support[int(component_owner[neighbor])] += length
            if support:
                target = min(support, key=lambda owner: (-support[owner], owner))
                proposals.append((fragment, target))
        if not proposals:
            break
        for fragment, target in proposals:
            component_owner[fragment] = target
            rooted[fragment] = True
            pending.remove(fragment)
    if pending:
        component_owner[list(pending)] = 0
    result.ravel()[nodes] = component_owner[component]
    return result
