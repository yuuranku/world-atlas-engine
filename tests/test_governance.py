"""Administration needs actual transport and never replaces local ownership."""

from types import SimpleNamespace as NS
import unittest

import numpy as np

from world_atlas.core.society.governance import (
    ADMINISTRATION_ERAS, control_region_reach, derive_governance, nominal_owner_field, path_length_km,
)
from world_atlas.core.society.model import TransportRoute
from world_atlas.core.society.state_formation import ControlRegionEdge, ControlRegionGraph, simulate_state_formation


def world(era="ancient", route=True):
    shape = (10, 20)
    grid = NS(shape=shape, water=np.zeros(shape), elevation=np.zeros(shape), snow=np.zeros(shape),
              metadata={"worldProfile": {"technologyEra": era}, "extents": {"west": -10, "east": 10, "north": 5, "south": -5},
                        "planet": {"radiusKm": 6400}})
    city = lambda ident, col: NS(identifier=ident, row=5, column=col, tier="city")
    cities = (city("capital", 7), city("client", 9), city("isolated", 18))
    owners = np.ones(shape, dtype=np.int16); owners[:, 9:] = 2; owners[:, 18:] = 3
    owners[0, 0] = -1; owners[0, 1] = 0
    state = lambda ident, capital, pop: NS(identifier=ident, core_settlement_id=capital,
                                         population_min=pop, population_max=pop, civilization_identifier=ident)
    states = (state(1, "capital", 8_000_000), state(2, "client", 80_000), state(3, "isolated", 50_000))
    routes = (TransportRoute("real-road", "road", "trunk", "capital", "client", ((7.5, 5.5), (9.5, 5.5))),) if route else ()
    society = NS(settlements=cities, transport=NS(routes=routes), provinces=NS(provinces=()),
                 politics=NS(states=states, state_id=owners,
                             government_forms=(NS(identifier=1, key="bureaucratic-monarchy"),),
                             political_entities=tuple(NS(country_identifier=i, government_form_identifier=1) for i in (1, 2, 3))))
    return grid, society


class GovernanceTests(unittest.TestCase):
    def test_relation_needs_a_recorded_route_and_preserves_actual_owner(self):
        grid, society = world()
        before = society.politics.state_id.copy()
        result = derive_governance(grid, society)
        self.assertEqual([(i["countryId"], i["suzerainId"]) for i in result["relations"]], [(2, 1)])
        self.assertEqual(result["relations"][0]["supportingRouteIds"], ["real-road"])
        field = nominal_owner_field(society, result)
        self.assertEqual(field[5, 9], 1)
        self.assertEqual(society.politics.state_id[5, 9], 2)
        self.assertEqual(field[0, 0], -1); self.assertEqual(field[0, 1], 0)
        np.testing.assert_array_equal(society.politics.state_id, before)

    def test_proximity_does_not_invent_a_nominal_empire(self):
        grid, society = world(route=False)
        result = derive_governance(grid, society)
        self.assertEqual(result["relations"], [])
        np.testing.assert_array_equal(nominal_owner_field(society, result), society.politics.state_id)

    def test_nominal_reach_does_not_assume_transit_through_a_third_country(self):
        grid, society = world()
        society.settlements += (NS(identifier='detached-port', row=5, column=8, tier='city'),)
        society.transport.routes = (
            TransportRoute('to-third-country', 'road', 'trunk', 'capital', 'isolated',
                           ((7.5, 5.5), (18.5, 5.5))),
            TransportRoute('from-third-country', 'road', 'trunk', 'isolated', 'detached-port',
                           ((18.5, 5.5), (8.5, 5.5))),
            TransportRoute('real-bilateral-link', 'road', 'trunk', 'detached-port', 'client',
                           ((8.5, 5.5), (9.5, 5.5))),
        )
        result = derive_governance(grid, society)
        self.assertFalse(any(record['countryId'] == 2 for record in result['relations']))

    def test_unavailable_rail_is_rejected_in_ancient_world(self):
        grid, society = world()
        society.transport.routes = (TransportRoute("future", "rail", "trunk", "capital", "client", ((7., 5.), (9., 5.))),)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            derive_governance(grid, society)

    def test_one_ended_river_routes_join_only_at_a_shared_channel_vertex(self):
        grid, society = world()
        society.transport.routes = (
            TransportRoute("upper", "river", "trunk", "capital", None, ((7.5, 5.5), (9.5, 5.5), (10.5, 5.5))),
            TransportRoute("lower", "river", "trunk", "client", None, ((9.5, 5.5), (10.5, 5.5))),
        )
        connected = derive_governance(grid, society)
        self.assertEqual(len(connected['relations']), 1)
        self.assertTrue(set(connected['relations'][0]['supportingRouteIds']) <= {'upper', 'lower'})
        society.transport.routes = (
            society.transport.routes[0],
            TransportRoute("nearby-but-separate", "river", "trunk", "client", None, ((9.6, 5.5), (10.6, 5.5))),
        )
        separate = derive_governance(grid, society)
        self.assertEqual(separate['relations'], [])

    def test_seam_distance_is_short_and_uses_actual_planet_radius(self):
        grid, _ = world()
        grid.metadata["extents"] = {"west": -180, "east": 180, "north": 90, "south": -90}
        length = path_length_km(grid, ((19.9, 5), (.1, 5)))
        self.assertAlmostEqual(length, 6400*np.radians(3.6), places=8)
        grid.metadata["planet"]["radiusKm"] = 3200
        self.assertAlmostEqual(path_length_km(grid, ((19.9, 5), (.1, 5))), length/2, places=8)

    def test_era_changes_communication_days_and_nominal_reach(self):
        ancient_grid, society = world("ancient")
        tribal_grid, _ = world("tribal")
        ancient = derive_governance(ancient_grid, society)
        tribal = derive_governance(tribal_grid, society)
        self.assertEqual(len(ancient["relations"]), 1)
        self.assertEqual(tribal["relations"], [])
        self.assertEqual(set(ADMINISTRATION_ERAS), {"tribal", "ancient", "medieval", "early-modern", "preindustrial", "industrial", "contemporary"})

    def test_too_distant_regions_remain_outside_actual_government(self):
        edges = tuple(ControlRegionEdge(a, a+1, 1, 1., 0., 0., 0., 0.) for a in range(1, 6))
        graph = ControlRegionGraph(np.array([0, 1, 1, 1, 1, 1, 1]), np.ones(7), np.ones(7, dtype=int), edges)
        result = simulate_state_formation(graph, core_region_by_state=np.array([0, 1]), state_strength=np.array([0., 1.]),
                                         travel_days_by_edge=(10.,)*5, maximum_response_days_by_state=np.array([0., 25.]))
        np.testing.assert_array_equal(result.region_owner, [0, 1, 1, 1, 0, 0, 0])
        self.assertTrue(np.all(np.isinf(result.acquisition_cost[4:])))

    def test_supported_tax_base_can_expand_an_empire_without_a_fixed_area_limit(self):
        grid, _ = world()
        labels = np.repeat(np.arange(1, 21)[None], 10, axis=0)
        graph = ControlRegionGraph(np.array([0]+[1]*20), np.ones(21), np.ones(21, dtype=int),
                                   tuple(ControlRegionEdge(a, a+1, 1, 1., 0., 0., 1., 0.) for a in range(1, 20)))
        graph, days, budgets = control_region_reach(grid, graph, labels, np.array([0, 1]), np.array([0., 2.]), provinces=False, routes=())
        self.assertTrue(np.isfinite(budgets[1]))
        stronger = simulate_state_formation(graph, core_region_by_state=np.array([0, 1]), state_strength=np.array([0., 2.]),
                                             travel_days_by_edge=days, maximum_response_days_by_state=budgets)
        weaker = simulate_state_formation(graph, core_region_by_state=np.array([0, 1]), state_strength=np.array([0., .3]),
                                           travel_days_by_edge=days, maximum_response_days_by_state=budgets*.15)
        self.assertGreater(np.count_nonzero(stronger.region_owner), np.count_nonzero(weaker.region_owner))

    def test_recorded_river_access_claims_channel_without_discounting_the_other_bank(self):
        grid, _ = world()
        labels = np.ones(grid.shape, dtype=np.int32)
        labels[:, 9] = 2
        labels[:, 10:] = 3
        edges = (ControlRegionEdge(1, 2, 10, 40., 40., 1., 0., 0.),
                 ControlRegionEdge(2, 3, 10, 40., 40., 1., 0., 0.))
        graph = ControlRegionGraph(np.array([0, 1, 1, 1]), np.ones(4),
                                   np.array([0, 90, 10, 100]), edges)
        cores, strengths = np.array([0, 1]), np.array([0., 1.])
        _, ordinary, budgets = control_region_reach(grid, graph, labels, cores, strengths,
                                                    provinces=False, routes=())
        river = TransportRoute("river-access", "river", "trunk", "capital", None,
                               ((8.5, 5.5), (9.5, 5.5), (9.5, 6.5)))
        enriched, days, _ = control_region_reach(grid, graph, labels, cores, strengths,
                                                 provinces=False, routes=(river,))
        self.assertLess(days[0], ordinary[0])
        self.assertEqual(days[1], ordinary[1])
        self.assertEqual(enriched.edges, graph.edges)
        result = simulate_state_formation(enriched, core_region_by_state=cores,
                                          state_strength=strengths, travel_days_by_edge=days,
                                          maximum_response_days_by_state=np.array([0., days[0]+.1]))
        np.testing.assert_array_equal(result.region_owner, [0, 1, 1, 0])

    def test_recorded_route_does_not_jump_over_reserved_frontier(self):
        grid, _ = world()
        labels = np.ones(grid.shape, dtype=np.int32)
        labels[:, 9] = 0
        labels[:, 10:] = 2
        graph = ControlRegionGraph(np.array([0, 1, 1]), np.ones(3), np.array([0, 90, 100]), ())
        route = TransportRoute("frontier-transit", "river", "trunk", "capital", None,
                               ((8.5, 5.5), (10.5, 5.5)))
        enriched, days, _ = control_region_reach(grid, graph, labels, np.array([0, 1]),
                                                 np.array([0., 1.]), provinces=False, routes=(route,))
        self.assertEqual(enriched.edges, ())
        self.assertEqual(days, ())

    def test_exact_diagonal_route_adds_only_its_observed_region_contact(self):
        grid, _ = world()
        labels = np.ones(grid.shape, dtype=np.int32)
        labels[6:, 9:] = 2
        graph = ControlRegionGraph(np.array([0, 1, 1]), np.ones(3), np.array([0, 100, 100]), ())
        route = TransportRoute("diagonal", "road", "local", "capital", "client",
                               ((8.5, 5.5), (9.5, 6.5)))
        enriched, days, _ = control_region_reach(grid, graph, labels, np.array([0, 1]),
                                                 np.array([0., 1.]), provinces=False, routes=(route,))
        self.assertEqual([(edge.first, edge.second) for edge in enriched.edges], [(1, 2)])
        self.assertEqual(enriched.edges[0].road_share, 0.)
        self.assertGreater(days[0], 0)


if __name__ == "__main__":
    unittest.main()
