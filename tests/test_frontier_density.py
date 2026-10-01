"""Frontier preferences cannot erase populated agricultural corridors."""
import unittest

import numpy as np

from world_atlas.core.society.frontiers import derive_stateless_frontier
from world_atlas.core.thematic import BIOME_TEMPERATE_GRASSLAND


class FrontierDensityTests(unittest.TestCase):
    def frontier(self, population, *, target=0.75, protected=None, access=None):
        shape = population.shape
        zeros = np.zeros(shape)
        return derive_stateless_frontier(
            np.ones(shape, dtype=bool), np.ones(shape, dtype=np.int16),
            np.full(shape, BIOME_TEMPERATE_GRASSLAND), zeros.astype(bool),
            np.full(shape, 0.2), np.full(shape, 0.15), np.full(shape, 0.3),
            population, np.ones(shape), zeros if access is None else access, zeros,
            np.zeros((8, *shape)), zeros.astype(bool) if protected is None else protected,
            target_share=target,
        )

    def test_dense_countryside_survives_an_unfulfillable_frontier_preference(self):
        population = np.ones((64, 128))
        population[:, :40] = 0.02
        frontier = self.frontier(population)
        self.assertFalse(np.any(frontier[:, 40:]))
        self.assertGreater(np.count_nonzero(frontier), 0)
        self.assertLess(np.mean(frontier), 0.4)

    def test_uniform_settled_plain_is_not_forced_to_supply_frontier(self):
        self.assertFalse(np.any(self.frontier(np.ones((64, 128)), target=0.95)))

    def test_real_transport_corridor_stays_connected_through_sparse_land(self):
        population = np.ones((64, 128))
        population[:, :48] = 0.02
        protected = np.zeros(population.shape, dtype=bool)
        protected[31:34, :] = True
        frontier = self.frontier(population, protected=protected)
        self.assertFalse(np.any(frontier[protected]))
        self.assertGreater(np.count_nonzero(frontier[:, :48]), 0)

    def test_density_decision_is_independent_of_density_units(self):
        population = np.ones((64, 128))
        population[:, :40] = 0.05
        np.testing.assert_array_equal(self.frontier(population), self.frontier(population / population.sum()))


if __name__ == "__main__":
    unittest.main()
