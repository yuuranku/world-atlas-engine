"""Settlement water-source and river-channel safety regression tests."""

from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.society.model import PopulationLayers
from world_atlas.core.society.population import (
    _settlement_freshwater_access,
    derive_settlements,
)


def _water_fixture() -> tuple[SimpleNamespace, SimpleNamespace, PopulationLayers]:
    height, width = 32, 48
    water = np.zeros((height, width), dtype=np.uint8)
    water[0, :] = 1
    water[-1, :] = 1
    water[:, 0] = 1
    water[:, -1] = 1
    river_order = np.zeros((height, width), dtype=np.uint8)
    river_order[16, 5:-5] = 1
    discharge = np.ones((height, width), dtype=np.float32)
    discharge[16, 5:-5] = 90.0
    discharge[8, 32] = 120.0
    weights = np.full((height, width), 0.15, dtype=np.float32)
    weights[15, 7:-7] = 1.0
    weights[17, 7:-7] = 0.95
    weights[water != 0] = 0.0
    grid = SimpleNamespace(
        shape=(height, width),
        water=water,
        river_order=river_order,
        discharge=discharge,
        flow_to=np.full((height, width), -1, dtype=np.int32),
        seasonal_river_strength=np.zeros((4, height, width), dtype=np.uint8),
        elevation=np.full((height, width), 0.20, dtype=np.float32),
        snow=np.zeros((height, width), dtype=bool),
        metadata={
            "extents": {"north": 60.0, "south": -60.0},
            "societyGeneration": {"humanSeed": 41},
        },
    )
    climate = SimpleNamespace(
        annual_precipitation=np.full((height, width), 0.12, dtype=np.float64),
        precipitation_range=np.full((height, width), 0.08, dtype=np.float64),
        temperature=np.full((height, width), 0.60, dtype=np.float64),
    )
    thematic = SimpleNamespace(
        climate=climate,
        land_potential=np.full((height, width), 0.58, dtype=np.float32),
        habitability=np.full((height, width), 0.72, dtype=np.float32),
    )
    population = PopulationLayers(
        population_weight=weights,
        population_band=np.where(weights > 0.0, 3, 0).astype(np.uint8),
        population_min=1_000_000,
        population_max=2_000_000,
    )
    return grid, thematic, population


class SettlementWaterTests(unittest.TestCase):
    def test_low_order_stream_and_groundwater_are_distinct_water_sources(self):
        grid, _thematic, _population = _water_fixture()
        land = grid.water == 0
        reliability = np.ones(grid.shape, dtype=np.float64)
        (
            stream_access,
            major_stream_access,
            stream_bank,
            groundwater_access,
            freshwater_access,
        ) = _settlement_freshwater_access(grid, reliability, land)

        self.assertGreater(float(stream_access[15, 16]), 0.0)
        self.assertEqual(float(major_stream_access.max()), 0.0)
        self.assertTrue(bool(stream_bank[15, 16]))
        self.assertGreater(float(groundwater_access[8, 32]), 0.0)
        self.assertGreaterEqual(
            float(freshwater_access[8, 32]),
            float(groundwater_access[8, 32]),
        )

    def test_settlements_use_low_order_banks_but_never_channels_or_water(self):
        grid, thematic, population = _water_fixture()
        first = derive_settlements(grid, thematic, population, target_count=20)
        second = derive_settlements(grid, thematic, population, target_count=20)

        self.assertEqual(
            tuple((item.row, item.column, item.site_type) for item in first),
            tuple((item.row, item.column, item.site_type) for item in second),
        )
        self.assertTrue(any(item.site_type == "river-city" for item in first))
        for settlement in first:
            self.assertEqual(int(grid.water[settlement.row, settlement.column]), 0)
            self.assertEqual(int(grid.river_order[settlement.row, settlement.column]), 0)


if __name__ == "__main__":
    unittest.main()
