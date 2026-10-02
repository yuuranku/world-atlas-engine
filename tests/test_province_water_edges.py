"""Large river mouths must not invalidate province territory simulation."""
from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.society.provinces import _province_transition_penalties
from world_atlas.core.society.territorial_simulation import TerritorySimulation


class ProvinceWaterEdgeTests(unittest.TestCase):
    def grid(self, water_kind):
        shape = (5, 8)
        water = np.zeros(shape, dtype=np.uint8)
        water[2, 3] = water_kind
        rivers = np.zeros(shape, dtype=np.int8)
        rivers[2, 2:4] = 5
        return SimpleNamespace(shape=shape, elevation=np.full(shape, .2),
                               river_order=rivers, water=water)

    def test_large_rivers_entering_sea_or_lake_allow_province_simulation(self):
        for water_kind in (1, 2):
            with self.subTest(water_kind=water_kind):
                grid = self.grid(water_kind)
                roads = np.zeros(grid.shape, dtype=np.float32)
                penalties = _province_transition_penalties(grid, roads)
                TerritorySimulation(valid=grid.water == 0,
                    friction=np.ones(grid.shape, dtype=np.float32),
                    transition_penalty=penalties, road_access=roads,
                    bridge_edges=np.zeros_like(penalties))

    def test_inland_river_banks_remain_costly(self):
        grid = self.grid(0)
        river = _province_transition_penalties(grid, np.zeros(grid.shape))
        grid.river_order[:] = 0
        plain = _province_transition_penalties(grid, np.zeros(grid.shape))
        self.assertGreater(float(river[:, 2, 1:5].max()), float(plain.max()))


if __name__ == '__main__':
    unittest.main()
