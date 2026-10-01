"""Countries require local control, sea access, and stable administrative seats."""
from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.society.administrative_topology import reconcile_partition_components
from world_atlas.core.society.administrative_audit import audit_administrative_topology
from world_atlas.core.society.model import TransportRoute
from world_atlas.core.society.provinces import _assign_unseeded_province_islands
from world_atlas.core.society.state_formation import (
    ControlRegionEdge, ControlRegionGraph, attach_maritime_control_regions,
    build_control_region_graph, distribute_state_cores, simulate_state_formation,
)


def graph(domains, pairs):
    return ControlRegionGraph(np.array(domains), np.ones(len(domains)),
        np.ones(len(domains), dtype=np.int32),
        tuple(ControlRegionEdge(a, b, 2, 1., 0., 0., 0., 0.) for a, b in pairs))


def settlement(identifier, column, population=10_000):
    return SimpleNamespace(identifier=identifier, row=0, column=column,
        population_max=population, score=1., tier="city")


class StateFormationTopologyTests(unittest.TestCase):
    def test_domain_aggregation_rejects_a_mixed_civilization_control_region(self):
        labels = np.ones((2, 3), dtype=np.int32)
        fields = np.zeros_like(labels, dtype=np.float32)
        edges = np.zeros((8, *labels.shape), dtype=np.float32)
        domains = np.array([[1, 1, 1], [1, 1, 2]])
        with self.assertRaisesRegex(ValueError, "cross owner domains"):
            build_control_region_graph(labels, edges, fields, edges, domains, fields)
        domains[1, 2] = 0
        result = build_control_region_graph(labels, edges, fields, edges, domains, fields)
        np.testing.assert_array_equal(result.domain_by_region, [0, 1])

    def test_graph_identifiers_do_not_claim_disconnected_territory(self):
        source = graph([0, 1, 1, 1, 1], [(1, 2), (3, 4)])
        result = simulate_state_formation(source, core_region_by_state=np.array([0, 1]),
                                         state_strength=np.array([0., 1.]),
                                         travel_days_by_edge=(1., 1.),
                                         maximum_response_days_by_state=np.array([0., 12.]))
        np.testing.assert_array_equal(result.region_owner, [0, 1, 1, 0, 0])
        self.assertTrue(np.all(np.isinf(result.acquisition_cost[3:])))

    def test_redundant_cores_move_to_inhabited_components_without_changing_count(self):
        source = graph([0, 1, 1, 1, 1, 2], [(1, 2), (3, 4)])
        cities = tuple(settlement(f"city-{i}", i - 1, i * 10_000) for i in range(1, 6))
        labels = np.array([[1, 2, 3, 4, 5]])
        cores = {1: cities[0], 2: cities[1], 3: cities[4]}
        result = distribute_state_cores(source, labels, cores, cities,
            protected_settlement_ids=frozenset({cities[0].identifier, cities[4].identifier}))
        self.assertEqual(set(result), set(cores))
        self.assertEqual(result[1], cities[0])
        self.assertEqual(result[2], cities[3])
        self.assertEqual(result[3], cities[4])

    def test_actual_sea_route_can_support_a_detached_same_continent_port(self):
        source = graph([0, 1, 1, 1, 1], [(1, 2), (3, 4)])
        labels = np.array([[1, 2, 0, 3, 4]])
        cities = (settlement("home", 1), settlement("port", 3))
        route = TransportRoute("coastal-link", "sea", "regional", "home", "port",
                               ((1., 0.), (2., 1.), (3., 0.)))
        owners = np.array([0, 1, 1, 0, 0])
        costs = dict(travel_days_by_edge=(1., 1.),
                     maximum_response_days_by_state=np.array([0., 12.]),
                     maritime_days_by_route={route.identifier: 2.})
        linked = attach_maritime_control_regions(source, owners, labels, np.array([0, 1]), cities, (route,), **costs)
        np.testing.assert_array_equal(linked, [0, 1, 1, 1, 1])
        no_evidence = attach_maritime_control_regions(source, owners, labels, np.array([0, 1]), cities, (), **costs)
        np.testing.assert_array_equal(no_evidence, owners)

    def test_sea_connected_island_does_not_consume_a_relocated_capital(self):
        source = graph([0, 1, 1, 1, 1], [(1, 2)])
        cities = tuple(settlement(f"city-{i}", i - 1, i * 10_000) for i in range(1, 5))
        labels = np.array([[1, 2, 3, 4]])
        route = TransportRoute("sea", "sea", "regional", "city-1", "city-4", ((0., 0.), (3., 0.)))
        result = distribute_state_cores(source, labels, {1: cities[0], 2: cities[1]}, cities,
            protected_settlement_ids=frozenset({"city-1"}), routes=(route,))
        self.assertEqual(result[2], cities[2])

    def test_sea_routes_do_not_cross_civilization_domains(self):
        source = graph([0, 1, 2], [])
        cities = (settlement("home", 0), settlement("port", 2))
        route = TransportRoute("sea", "sea", "regional", "home", "port", ((0., 0.), (2., 0.)))
        owners = np.array([0, 1, 0])
        result = attach_maritime_control_regions(source, owners, np.array([[1, 0, 2]]),
            np.array([0, 1]), cities, (route,), travel_days_by_edge=(),
            maximum_response_days_by_state=np.array([0., 12.]),
            maritime_days_by_route={route.identifier: 2.})
        np.testing.assert_array_equal(result, owners)

    def test_landing_does_not_grant_the_whole_island_beyond_response_budget(self):
        source = graph([0, 1, 1, 1, 1], [(1, 2), (3, 4)])
        labels = np.array([[1, 2, 0, 3, 4]])
        cities = (settlement("home", 1), settlement("port", 3))
        route = TransportRoute("sea", "sea", "regional", "home", "port", ((1., 0.), (3., 0.)))
        result = attach_maritime_control_regions(source, np.array([0, 1, 1, 0, 0]),
            labels, np.array([0, 1]), cities, (route,), travel_days_by_edge=(4., 4.),
            maximum_response_days_by_state=np.array([0., 10.]),
            maritime_days_by_route={"sea": 4.})
        # Four days to the departure port, four at sea: the island interior
        # would require twelve days, beyond the capital's ten-day budget.
        np.testing.assert_array_equal(result, [0, 1, 1, 1, 0])

    def test_repeated_sea_landings_consume_one_cumulative_capital_budget(self):
        source = graph([0, 1, 1, 1], [])
        labels = np.array([[1, 0, 2, 0, 3]])
        cities = (settlement("home", 0), settlement("middle", 2), settlement("remote", 4))
        routes = tuple(TransportRoute(identifier, "sea", "regional", first, second,
            ((float(a), 0.), (float(b), 0.))) for identifier, first, second, a, b in
            (("first", "home", "middle", 0, 2), ("second", "middle", "remote", 2, 4)))
        result = attach_maritime_control_regions(source, np.array([0, 1, 0, 0]),
            labels, np.array([0, 1]), cities, routes, travel_days_by_edge=(),
            maximum_response_days_by_state=np.array([0., 10.]),
            maritime_days_by_route={"first": 6., "second": 6.})
        np.testing.assert_array_equal(result, [0, 1, 1, 0])

    def test_fragment_transfers_only_to_rooted_same_domain_neighbor(self):
        labels = np.array([[1, 1, 2, 2, 1, 3, 3, 0]], dtype=np.int16)
        seats = {1: (0, 0), 2: (0, 2), 3: (0, 6)}
        domains = np.array([0, 1, 1, 2])
        result = reconcile_partition_components(labels, seats, domains, np.ones_like(labels))
        self.assertEqual(result[0, 4], 2)
        for owner, cell in seats.items():
            self.assertEqual(result[cell], owner)

    def test_unsupported_fragment_becomes_frontier_but_island_survives(self):
        labels = np.array([[1, 0, 1, 0, 1, 0]], dtype=np.int16)
        land = np.array([[1, 1, 1, 0, 2, 0]])
        result = reconcile_partition_components(labels, {1: (0, 0)}, np.array([0, 1]), land)
        np.testing.assert_array_equal(result, [[1, 0, 0, 0, 1, 0]])

    def test_fragment_chain_grows_from_existing_seats_without_swapping_exclaves(self):
        labels = np.array([[1, 0, 2, 1, 3, 2, 0, 3, 0]], dtype=np.int16)
        result = reconcile_partition_components(labels, {1: (0, 0), 2: (0, 2), 3: (0, 7)},
            np.array([0, 1, 1, 1]), np.ones_like(labels))
        np.testing.assert_array_equal(result, [[1, 0, 2, 2, 2, 2, 0, 3, 0]])

    def test_detached_component_with_capital_sea_access_is_kept_and_audited(self):
        labels = np.array([[1, 0, 0, 1, 0, 0]], dtype=np.int16)
        cities = (settlement("home", 0), settlement("port", 3))
        route = TransportRoute("sea", "sea", "regional", "home", "port", ((0., 0.), (3., 0.)))
        result = reconcile_partition_components(labels, {1: (0, 0)}, np.array([0, 1]),
            np.ones_like(labels), maritime_routes=(route,), settlements=cities)
        np.testing.assert_array_equal(result, labels)
        record = SimpleNamespace(identifier=1, core_settlement_id="home", state_identifier=1)
        society = SimpleNamespace(settlements=cities, transport=SimpleNamespace(routes=(route,)),
            politics=SimpleNamespace(state_id=result, states=(record,)),
            provinces=SimpleNamespace(province_id=result.astype(np.int32), provinces=(record,)))
        audit = audit_administrative_topology(SimpleNamespace(water=np.zeros_like(labels)), society)
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(audit["states"]["summary"]["seaSupportedSameLandComponentCount"], 1)
        self.assertEqual(audit["states"]["owners"][0]["seaRouteEvidence"][0]["routeId"], "sea")

    def test_unseeded_province_island_uses_distance_not_identifier(self):
        states = np.zeros((9, 20), dtype=np.int16)
        states[4, 2] = states[4, 13] = states[4, 15] = 1
        labels = np.zeros_like(states, dtype=np.int32)
        labels[4, 2], labels[4, 15] = 1, 2
        specs = tuple(SimpleNamespace(identifier=identifier, state_identifier=1,
            core=SimpleNamespace(row=4, column=column)) for identifier, column in ((1, 2), (2, 15)))
        result = _assign_unseeded_province_islands(labels, states, specs)
        self.assertEqual(result[4, 13], 2)
        self.assertEqual(result[4, 2], 1)

    def test_remote_provincial_speck_joins_the_existing_local_seat(self):
        labels = np.array([[1, 0, 1, 2, 2, 0]], dtype=np.int32)
        # The island has its own provincial seat. A three-cell-style leftover
        # of the mainland province must not remain a second island district.
        result = reconcile_partition_components(labels, {1: (0, 0), 2: (0, 4)},
            np.array([0, 1, 1]), np.array([[1, 0, 2, 2, 2, 0]]), prefer_local_seats=True)
        np.testing.assert_array_equal(result, [[1, 0, 2, 2, 2, 0]])


if __name__ == "__main__":
    unittest.main()
