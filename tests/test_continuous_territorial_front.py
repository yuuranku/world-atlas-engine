"""Generated arrival fields must avoid octile bias and retain crossing semantics."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from world_atlas.core.society.territorial_simulation import (
    TerritorySeed, TerritorySimulation, simulate_territories, simulate_bounded_reach,
    _continuous_front_time, _FRONT_PEERS, _NEIGHBORS,
    bridge_transition_discounts,
)
from world_atlas.core.society.spatial import refine_partition_boundaries


def simulation(shape, *, valid=None, domains=None, transitions=None, bridges=None):
    return TerritorySimulation(
        valid=np.ones(shape, bool) if valid is None else valid,
        friction=np.ones(shape, np.float32),
        transition_penalty=np.zeros((8, *shape), np.float32) if transitions is None else transitions,
        road_access=np.zeros(shape, np.float32),
        bridge_edges=np.zeros((8, *shape), np.float32) if bridges is None else bridges,
        owner_constraint=domains,
    )


class ContinuousTerritorialFrontTests(unittest.TestCase):
    def test_flat_arrivals_reduce_direction_bias_without_coordinate_noise(self):
        shape = (101, 101)
        result = simulate_territories(simulation(shape), (TerritorySeed(50, 50, 7),))
        rows, columns = np.indices(shape)
        dx, dy = abs(columns-50), abs(rows-50)
        true = np.hypot(dx, dy)
        octile = np.maximum(dx, dy)+(math.sqrt(2)-1)*np.minimum(dx, dy)
        ring = (true >= 20) & (true <= 40)
        error = (result.cost[ring]-true[ring])/true[ring]
        original_error = (octile[ring]-true[ring])/true[ring]
        self.assertGreaterEqual(float(error.min()), -1e-6)
        self.assertLess(float(error.max()), .015)
        self.assertLess(float(error.mean()), float(original_error.mean())/5)
        np.testing.assert_allclose(result.cost[:, 50], abs(np.arange(101)-50))
        self.assertEqual(set(np.unique(result.owner)), {7})

    def test_continuous_same_owner_front_updates_all_four_cardinal_directions(self):
        shape = (7, 7)
        friction = np.ones(shape)
        transitions = np.zeros((8, *shape))
        for direction, peers in _FRONT_PEERS.items():
            labels = np.ones(shape, np.int32)
            costs = np.full(shape, np.inf)
            accepted = np.zeros(shape, bool)
            for _, dy, dx in peers:
                accepted[3+dy, 3+dx] = True
                costs[3+dy, 3+dx] = 9.75
            result = _continuous_front_time(3, 3, direction, 1, 10., 1., 1.,
                                            friction, transitions, costs, labels, accepted)
            self.assertAlmostEqual(result, 10+math.sqrt(1-.25**2))
            labels.fill(2)
            unchanged = _continuous_front_time(3, 3, direction, 1, 10., 1., 1.,
                                               friction, transitions, costs, labels, accepted)
            self.assertEqual(unchanged, 11., "different owners cannot form a combined front")

    def test_angular_crossing_cannot_spread_one_cheap_bridge_edge(self):
        shape = (7, 7)
        friction = np.ones(shape)
        for direction, peers in _FRONT_PEERS.items():
            labels = np.ones(shape, np.int32)
            costs = np.zeros(shape)
            accepted = np.ones(shape, bool)
            transitions = np.zeros((8, *shape))
            for peer_direction, dy, dx in peers:
                transitions[7-peer_direction, 3+dy, 3+dx] = 20.
            bridged_edge_slowness = 1.+20.*.25
            result = _continuous_front_time(3, 3, direction, 1, 0., bridged_edge_slowness, 1.,
                                            friction, transitions, costs, labels, accepted)
            self.assertEqual(result, bridged_edge_slowness,
                             "the unbridged incoming edges disable the cheaper angular update")
            transitions.fill(0)
            result = _continuous_front_time(3, 3, direction, 1, 0., 21., 1.,
                                            friction, transitions, costs, labels, accepted)
            self.assertEqual(result, 21., "a costly authored axial crossing remains costly")

    def test_bounded_reach_uses_continuous_arrivals_and_respects_its_budget(self):
        reached = simulate_bounded_reach(simulation((61, 61)), (TerritorySeed(30, 30, 1),),
                                        maximum_cost=20.)
        self.assertTrue(reached[38, 48], "a 19.70-cell diagonal reach is not an octile 21.31-cell reach")
        self.assertFalse(reached[31, 50])
        self.assertFalse(reached[30, 51])

    def test_invalid_land_and_hard_cultural_domains_block_front_propagation(self):
        shape = (31, 51)
        domains = np.ones(shape, np.int32)
        domains[:, 26:] = 2
        valid = np.ones(shape, bool)
        valid[15, 2:24] = False
        seeds = (TerritorySeed(8, 10, 5, domain=1), TerritorySeed(23, 39, 9, domain=2))
        result = simulate_territories(simulation(shape, valid=valid, domains=domains), seeds)
        np.testing.assert_array_equal(result.owner[valid & (domains == 1)], 5)
        np.testing.assert_array_equal(result.owner[valid & (domains == 2)], 9)
        np.testing.assert_array_equal(result.owner[~valid], -1)
        reached = simulate_bounded_reach(simulation(shape, valid=valid, domains=domains), seeds[:1],
                                        maximum_cost=1000.)
        self.assertFalse(bool(np.any(reached[:, 26:])))

    def test_directional_barriers_and_exact_bridge_discounts_keep_real_crossing_costs(self):
        shape = (31, 51)
        transitions = np.zeros((8, *shape), np.float32)
        for direction, (dy, dx, _) in enumerate(_NEIGHBORS):
            for row in range(shape[0]):
                target = row+dy
                if 0 <= target < shape[0] and (row < 15) != (target < 15):
                    transitions[direction, row] = 30.
        seed = (TerritorySeed(10, 25, 1),)
        blocked = simulate_bounded_reach(simulation(shape, transitions=transitions), seed,
                                        maximum_cost=20.)
        self.assertFalse(bool(np.any(blocked[15:])))
        bridges = np.zeros_like(transitions)
        bridges[6, 14, 25] = .75
        bridges[1, 15, 25] = .75
        bridged = simulate_bounded_reach(simulation(shape, transitions=transitions, bridges=bridges),
                                        seed, maximum_cost=20.)
        self.assertTrue(bridged[16, 25])
        self.assertFalse(bridged[15, 0], "the single bridge discount cannot become a whole river discount")

    def test_strength_scales_ordinary_terrain_but_not_crossing_penalties(self):
        shape = (41, 61)
        ordinary = simulation(shape)
        first = simulate_territories(ordinary, (TerritorySeed(20, 30, 1),))
        second = simulate_territories(ordinary, (TerritorySeed(20, 30, 1, strength=2),))
        np.testing.assert_allclose(second.cost, first.cost*.5, atol=1e-6)
        transitions = np.full((8, *shape), 10., np.float32)
        result = simulate_territories(simulation(shape, transitions=transitions),
                                     (TerritorySeed(20, 30, 1, strength=2),))
        self.assertEqual(result.cost[20, 31], 10.5)

    def test_periodic_longitude_and_protected_seed_cells_remain_authoritative(self):
        shape = (41, 61)
        seeds = (TerritorySeed(20, 0, 4), TerritorySeed(20, 30, 8))
        result = simulate_territories(simulation(shape), seeds)
        self.assertEqual(result.cost[20, -1], 1.)
        self.assertEqual(result.owner[20, 0], 4)
        self.assertEqual(result.owner[20, 30], 8)
        np.testing.assert_array_equal(result.cost[:, 1], result.cost[:, -1])

    def test_flow_crossing_support_outside_road_cells_does_not_invent_a_raster_bank(self):
        rivers = np.zeros((7, 11), np.uint8)
        rivers[1, 4] = 3
        road = SimpleNamespace(identifier="road", mode="road", path=((1.5, 2.5), (6.5, 2.5)))
        bridge = SimpleNamespace(route_identifier="road", row=1, column=4, importance="regional")
        discounts = bridge_transition_discounts(rivers, (road,), (bridge,))
        self.assertFalse(bool(np.any(discounts)))
        far = SimpleNamespace(route_identifier="road", row=6, column=4, importance="regional")
        with self.assertRaisesRegex(ValueError, "adjacent"):
            bridge_transition_discounts(rivers, (road,), (far,))

    def test_bridge_discounts_only_the_two_actual_native_bank_edges(self):
        rivers = np.zeros((7, 11), np.uint8)
        rivers[2, 4] = 3
        road = SimpleNamespace(identifier="road", mode="road", path=((1.5, 2.5), (6.5, 2.5)))
        bridge = SimpleNamespace(route_identifier="road", row=2, column=4, importance="regional")
        discounts = bridge_transition_discounts(rivers, (road,), (bridge,))
        self.assertAlmostEqual(float(discounts[4, 2, 3]), .65)
        self.assertAlmostEqual(float(discounts[4, 2, 4]), .65)
        self.assertEqual(np.count_nonzero(discounts), 4, "two bank edges and their reverse directions")

    def test_boundary_regrowth_uses_geometric_arrivals_between_fixed_cores(self):
        shape = (51, 81)
        rows, columns = np.indices(shape)
        first_dx = np.minimum(abs(columns-20), shape[1]-abs(columns-20))
        second_dx = np.minimum(abs(columns-56), shape[1]-abs(columns-56))
        first_distance = np.hypot(rows-12, first_dx)
        second_distance = np.hypot(rows-35, second_dx)
        geometric = np.where(first_distance <= second_distance, 1, 2).astype(np.int32)
        result = refine_partition_boundaries(
            geometric, np.ones(shape, bool), np.zeros((8, *shape), np.float32),
            {(12, 20): 1, (35, 56): 2}, band_radius=100,
        )
        first_octile = np.maximum(abs(rows-12), first_dx)+(math.sqrt(2)-1)*np.minimum(abs(rows-12), first_dx)
        second_octile = np.maximum(abs(rows-35), second_dx)+(math.sqrt(2)-1)*np.minimum(abs(rows-35), second_dx)
        octile = np.where(first_octile <= second_octile, 1, 2)
        self.assertLess(np.count_nonzero(result != geometric), np.count_nonzero(octile != geometric)/5)
        self.assertEqual(result[12, 46], 2, "the final refinement must not restore the old octile bisector")
        self.assertEqual(result[12, 20], 1)
        self.assertEqual(result[35, 56], 2)


if __name__ == "__main__":
    unittest.main()
