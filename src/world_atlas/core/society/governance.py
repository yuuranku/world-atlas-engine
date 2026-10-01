"""Era-dependent administration over the world's recorded transport network.

Travel rates and response cycles are explicit simulation parameters, not
historical measurements or universal maximum empire sizes. Ownership remains
the actual local partition; suzerainty is a separate, evidenced relation.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ..model import WorldGrid
from .model import SocietyLayers


@dataclass(frozen=True, slots=True)
class AdministrationEra:
    label: str
    response_days: float
    walking_km_day: float
    road_km_day: float
    river_km_day: float
    sea_km_day: float
    rail_km_day: float
    delegation: float


ADMINISTRATION_ERAS = {
    "tribal": AdministrationEra("部落时代", 12, 18, 24, 35, 50, 0, .25),
    "ancient": AdministrationEra("古典时代", 60, 22, 70, 60, 110, 0, .75),
    "medieval": AdministrationEra("中古时代", 45, 20, 55, 50, 85, 0, .55),
    "early-modern": AdministrationEra("近代早期", 75, 24, 85, 75, 150, 0, .85),
    "preindustrial": AdministrationEra("前工业时代", 80, 24, 90, 80, 160, 0, .90),
    "industrial": AdministrationEra("工业时代", 18, 28, 140, 150, 300, 400, 1.0),
    "contemporary": AdministrationEra("近现代", 7, 30, 240, 240, 400, 900, 1.0),
}


def administration_era(grid: WorldGrid) -> AdministrationEra:
    era = grid.metadata["worldProfile"]["technologyEra"]
    if era not in ADMINISTRATION_ERAS:
        raise ValueError("administration requires a supported canonical technology era")
    return ADMINISTRATION_ERAS[era]


def path_length_km(grid: WorldGrid, path) -> float:
    """Sum great-circle segments, including short crossings of the map seam."""
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("transport length requires a finite column/row polyline")
    extents = grid.metadata["extents"]
    longitude = np.radians(float(extents["west"]) + values[:, 0] / grid.shape[1]
                           * (float(extents["east"]) - float(extents["west"])))
    latitude = np.radians(float(extents["north"]) - values[:, 1] / grid.shape[0]
                          * (float(extents["north"]) - float(extents["south"])))
    angle = np.sin(np.diff(latitude)/2)**2 + np.cos(latitude[:-1])*np.cos(latitude[1:])*np.sin(np.diff(longitude)/2)**2
    return float(np.sum(2*float(grid.metadata["planet"]["radiusKm"])*np.arcsin(np.sqrt(np.clip(angle, 0, 1)))))


def _route_days(grid, route, era):
    speed = {"road": era.road_km_day, "river": era.river_km_day,
             "sea": era.sea_km_day, "rail": era.rail_km_day}[route.mode]
    if speed <= 0:
        raise ValueError("recorded transport mode is unavailable in this era")
    path = np.asarray(route.path)
    rows = np.clip(np.floor(path[:, 1]).astype(int), 0, grid.shape[0]-1)
    columns = np.floor(path[:, 0]).astype(int) % grid.shape[1]
    # A relay message is quicker than a supply convoy. Both must be feasible:
    # use the slower response cost, with terrain and seasonal exposure.
    terrain = float(np.mean(np.clip(grid.elevation[rows, columns], 0, 1)))
    snow = float(np.mean(grid.snow[rows, columns] > 0))
    exposure = (1 + .8*terrain + .7*snow) if route.mode == "road" else (1.25 if route.mode == "sea" else 1.15)
    importance = {"trunk": 1, "regional": 1.15, "local": 1.35}[route.importance]
    return max(.05, path_length_km(grid, route.path) / speed * exposure * importance)


def maritime_response_days(grid: WorldGrid, routes) -> dict[str, float]:
    """Use the same spherical, era-specific sea costs during state formation."""
    era = administration_era(grid)
    return {route.identifier: _route_days(grid, route, era)
            for route in routes if route.mode == "sea"}


def _control_route_links(grid, labels, routes, era):
    """Measure links on recorded routes, splitting at exact native grid lines.

    A navigable channel is a transport link, not an extra defensive bank.
    Region runs use their actual route arclength; no square-root area estimate
    can make a short landing into a journey across the whole adjoining basin.
    A zero region breaks the link, so recorded transit never claims frontier.
    """
    height, width = labels.shape
    links = {}
    for route in routes:
        if route.mode == "sea":
            continue
        runs = []
        for first, second in zip(route.path, route.path[1:]):
            first = np.asarray(first, dtype=np.float64)
            delta = np.asarray(second, dtype=np.float64) - first
            delta[0] = (delta[0] + width / 2) % width - width / 2
            cuts = [0., 1.]
            for axis in (0, 1):
                if delta[axis] == 0:
                    continue
                lower, upper = sorted((first[axis], first[axis] + delta[axis]))
                cuts.extend((line-first[axis])/delta[axis]
                            for line in range(math.floor(lower)+1, math.ceil(upper))
                            if 0 < (line-first[axis])/delta[axis] < 1)
            cuts = sorted(set(cuts))
            for start, end in zip(cuts, cuts[1:]):
                midpoint = first + .5*(start+end)*delta
                row = min(height-1, max(0, math.floor(midpoint[1])))
                column = math.floor(midpoint[0]) % width
                region = int(labels[row, column])
                length = path_length_km(grid, (first+start*delta, first+end*delta))
                if runs and runs[-1][0] == region:
                    runs[-1][1] += length
                else:
                    runs.append([region, length])
        total = sum(run[1] for run in runs)
        if not total:
            continue
        days_per_km = _route_days(grid, route, era) / total
        for (first, first_length), (second, second_length) in zip(runs, runs[1:]):
            if first <= 0 or second <= 0 or first == second:
                continue
            pair = tuple(sorted((first, second)))
            cost = max(.01, .5*(first_length+second_length)*days_per_km)
            links[pair] = min(links.get(pair, math.inf), cost)
    return links


def control_region_reach(grid, graph, labels, cores, strengths, *, provinces: bool, routes):
    """Physical travel estimates and era budgets for political formation.

    A control-region traversal is estimated from its spherical area, actual
    shared barrier and mapped road share. It is deliberately an estimate, not
    a straight-line claim across water. Province allocation divides land
    already governed by its parent, so it requires complete coverage rather
    than creating another band of stateless land within a country.
    """
    from .population import cell_areas_km2
    era = administration_era(grid)
    labels = np.asarray(labels)
    active = labels > 0
    areas = np.bincount(labels[active], weights=cell_areas_km2(grid)[active],
                        minlength=graph.region_count+1)
    spans = np.sqrt(areas)
    from .state_formation import ControlRegionEdge
    links = _control_route_links(grid, labels, routes, era)
    existing = {(edge.first, edge.second) for edge in graph.edges}
    # A real diagonal route can connect two regions at a native corner even
    # when they have no east/south boundary. Its recorded traversal is the
    # evidence for that graph edge; geometric proximity never adds a link.
    additions = tuple(ControlRegionEdge(a, b, 1, .85, 0., 0., 0., 0.)
                      for (a, b) in sorted(links) if (a, b) not in existing
                      and graph.domain_by_region[a] == graph.domain_by_region[b])
    if additions:
        from dataclasses import replace
        graph = replace(graph, edges=tuple(sorted((*graph.edges, *additions), key=lambda e: (e.first, e.second))))
    days = []
    for edge in graph.edges:
        distance = .5*(spans[edge.first]+spans[edge.second])
        road = float(np.clip(edge.road_share, 0, 1))
        # Pay exposed terrain at walking speed; a documented road seam
        # permits a relay fraction. A bridge reduces its actual barrier.
        inverse_speed = (1-road)/era.walking_km_day + road/era.road_km_day
        estimate = max(.01, distance*inverse_speed*(1+.12*edge.crossing_cost))
        days.append(min(estimate, links.get((edge.first, edge.second), math.inf)))
    budgets = np.zeros(len(cores), dtype=np.float64)
    for state in range(1, len(cores)):
        if provinces:
            budgets[state] = math.inf
        else:
            resource = float(graph.resource_by_region[cores[state]])
            budgets[state] = era.response_days * float(strengths[state]) * (1+.18*math.log1p(resource))
    return graph, tuple(days), budgets


def derive_governance(grid: WorldGrid, society: SocietyLayers) -> dict:
    """Measure central reach and form acyclic, route-supported nominal realms.

    No sea, frontier or foreign country is filled by capital proximity. A
    disconnected governor retains local authority and is marked autonomous;
    an independent country's actual ownership is never overwritten by its
    overlord. Nominal relations require an existing inter-country route.
    """
    era = administration_era(grid)
    cities = tuple(society.settlements)
    indices = {city.identifier: i for i, city in enumerate(cities)}
    owners = np.asarray([society.politics.state_id[city.row, city.column] for city in cities], dtype=int)
    states = {state.identifier: state for state in society.politics.states}
    provinces = society.provinces.provinces
    edges, evidence = {}, {}
    node_owners = list(owners)
    river_nodes = {}
    cross_country = set()

    def add_edge(first, second, cost, identifier):
        if first == second:
            return
        pair = tuple(sorted((first, second)))
        if cost < edges.get(pair, math.inf):
            edges[pair], evidence[pair] = cost, identifier
        first_owner, second_owner = node_owners[first], node_owners[second]
        if first_owner > 0 and second_owner > 0 and first_owner != second_owner:
            cross_country.add(tuple(sorted((int(first_owner), int(second_owner)))))

    for route in society.transport.routes:
        if route.mode == "river":
            # One-ended navigable reaches meet at their actual confluence
            # vertices. Omitting them would wrongly mark river cities as
            # isolated; proximity alone must never invent a tributary link.
            total_length = max(.001, path_length_km(grid, route.path))
            route_days = _route_days(grid, route, era)
            previous = indices[route.source_settlement_id]
            previous_point = route.path[0]
            for position, point in enumerate(route.path):
                key = (round(float(point[0]) % grid.shape[1], 8), round(float(point[1]), 8))
                if key not in river_nodes:
                    river_nodes[key] = len(node_owners)
                    column, row = int(math.floor(key[0])) % grid.shape[1], int(np.clip(math.floor(key[1]), 0, grid.shape[0]-1))
                    node_owners.append(int(society.politics.state_id[row, column]))
                current = river_nodes[key]
                length = path_length_km(grid, (previous_point, point)) if position else 0
                add_edge(previous, current, max(.01, length/total_length*route_days), route.identifier)
                previous, previous_point = current, point
            if route.target_settlement_id is not None:
                add_edge(previous, indices[route.target_settlement_id], .01, route.identifier)
            continue
        if route.target_settlement_id is None:
            raise ValueError("a road, sea or rail route needs both recorded endpoints")
        first, second = indices[route.source_settlement_id], indices[route.target_settlement_id]
        cost = _route_days(grid, route, era)
        add_edge(first, second, cost, route.identifier)
    node_owners = np.asarray(node_owners, dtype=int)

    def graph_for(allowed_countries):
        selected = [(pair, cost) for pair, cost in edges.items()
                    if node_owners[pair[0]] in allowed_countries
                    and node_owners[pair[1]] in allowed_countries]
        starts, ends, costs = [], [], []
        for (first, second), cost in selected:
            starts.extend((first, second)); ends.extend((second, first)); costs.extend((cost, cost))
        return csr_matrix((costs, (starts, ends)), shape=(len(node_owners), len(node_owners)))

    central_days = np.full(len(cities), np.inf, dtype=np.float64)
    budgets = {}
    summaries = []
    forms = {item.identifier: item.key for item in society.politics.government_forms}
    governments = {item.country_identifier: forms[item.government_form_identifier]
                   for item in society.politics.political_entities}
    institution = {"bureaucratic-monarchy": 1.35, "court-monarchy": 1.05,
                   "hereditary-feudalism": .85, "maritime-republic": 1.15,
                   "city-republic": .75, "nomadic-confederacy": .80,
                   "clan-league": .65, "tributary-chiefdom": .60, "estate-monarchy": 1.05}
    for identifier, state in states.items():
        core = indices[state.core_settlement_id]
        distance = dijkstra(graph_for({identifier}), indices=core, directed=True)[:len(cities)]
        central_days[owners == identifier] = distance[owners == identifier]
        local = owners == identifier
        urban = sum(city.tier != "site" for city, selected in zip(cities, local) if selected)
        fiscal = math.sqrt(max(0, (state.population_min+state.population_max)/2) / 1e6)
        # Larger, supported tax bases can fund relays, governors and garrisons.
        # The logarithm prevents population from making travel cost disappear.
        budget = era.response_days * institution[governments[identifier]] * (1+.14*math.log1p(fiscal)+.08*math.log1p(urban))
        budgets[identifier] = budget
        summaries.append({"id": identifier, "responseBudgetDays": round(budget, 3),
                          "connectedSettlements": int(np.count_nonzero(local & np.isfinite(distance))),
                          "autonomousSettlements": int(np.count_nonzero(local & ~np.isfinite(distance)))})

    # A strictly increasing fiscal rank prevents circular vassal hierarchies.
    power = {identifier: (state.population_min+state.population_max)/2
             * institution[governments[identifier]] for identifier, state in states.items()}
    parents = {identifier: identifier for identifier in states}
    relations = []
    for child, state in states.items():
        candidates = []
        for parent, stronger in states.items():
            if parent == child or power[parent] <= 1.8*max(1, power[child]):
                continue
            pair = tuple(sorted((parent, child)))
            if pair not in cross_country:
                continue
            # An unrelated country never supplies unrecorded transit rights.
            # A real bilateral link and a route through these two countries
            # are both required before nominal power can be projected.
            distance = dijkstra(graph_for({parent, child}),
                                indices=indices[stronger.core_settlement_id], directed=True)
            days = float(distance[indices[state.core_settlement_id]])
            cultural_distance = 1 if stronger.civilization_identifier == state.civilization_identifier else 1.35
            projected = days * cultural_distance
            budget = budgets[parent] * era.delegation
            if not math.isfinite(projected) or projected > budget:
                continue
            ratio = power[parent] / max(1, power[child])
            # Nominal subordination remains costly across cultural frontiers.
            strength = math.log(ratio) * math.exp(-projected / max(1, budget)) / cultural_distance
            if strength >= .75:
                candidates.append((strength, parent, days))
        if candidates:
            strength, parent, days = max(candidates, key=lambda item: (item[0], -item[1]))
            parents[child] = parent
            route_ids = sorted({evidence[pair] for pair in edges
                                if tuple(sorted((int(node_owners[pair[0]]), int(node_owners[pair[1]])))) == tuple(sorted((parent, child)))})
            relations.append({"countryId": child, "suzerainId": parent, "kind": "tributary",
                              "capitalTravelDays": round(days, 3), "supportingRouteIds": route_ids})
    nominal = {identifier: identifier for identifier in states}
    for identifier in states:
        root = identifier
        while parents[root] != root:
            root = parents[root]
        nominal[identifier] = root
    for summary in summaries:
        summary["suzerainId"] = parents[summary["id"]]
        summary["nominalRealmId"] = nominal[summary["id"]]
    settlement_records = []
    for index, city in enumerate(cities):
        owner, days = int(owners[index]), float(central_days[index])
        status = ("frontier" if owner <= 0 else "autonomous" if not math.isfinite(days)
                  else "central" if days <= budgets[owner] else "delegated")
        settlement_records.append({"id": city.identifier, "countryId": max(0, owner),
                                   "centralTravelDays": round(days, 3) if math.isfinite(days) else None,
                                   "administration": status, "nominalRealmId": nominal.get(owner, 0)})
    return {"schema": "world-atlas-governance-v1", "era": grid.metadata["worldProfile"]["technologyEra"],
            "eraLabel": era.label, "method": "recorded-transport-and-local-administration",
            "calibration": "Simulation travel rates and response budgets; not reconstructed historical boundaries.",
            "countries": summaries, "relations": relations, "settlements": settlement_records,
            "provinceSeats": [{"id": item.identifier, "settlementId": item.core_settlement_id} for item in provinces]}


def nominal_owner_field(society: SocietyLayers, governance: dict) -> np.ndarray:
    """Map each real country to its separate nominal overlord; leave frontier."""
    if governance["schema"] != "world-atlas-governance-v1":
        raise ValueError("unsupported governance model")
    lookup = np.zeros(len(society.politics.states)+1, dtype=np.int16)
    for country in governance["countries"]:
        lookup[country["id"]] = country["nominalRealmId"]
    if np.any(lookup[1:] <= 0):
        raise ValueError("governance must describe every actual country")
    owners = society.politics.state_id
    return np.where(owners < 0, -1, lookup[np.maximum(owners, 0)]).astype(np.int16)


def write_governance(output: Path, grid: WorldGrid, society: SocietyLayers, governance: dict) -> None:
    from ..presentation import society_content_digest
    payload = {**governance, "gridDigest": grid.content_digest(), "societyDigest": society_content_digest(society)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "governance.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n", encoding="utf-8")
