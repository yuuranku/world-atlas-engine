"""Physical causes of fractional cover, including dry riparian vegetation."""

from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.vegetation import derive_vegetation_cover
from ecology_fixtures import declared_freshwater_sources


def fixture(rain=900.0, temperature=18.0):
    shape = (8, 16)
    grid = SimpleNamespace(shape=shape, water=np.zeros(shape, np.uint8),
        snow=np.zeros(shape, bool), elevation=np.full(shape, .2, np.float32),
        river_order=np.zeros(shape, np.uint8),
        flow_to=np.array([[(row+1)*shape[1]+column if row+1<shape[0] else -1
                           for column in range(shape[1])] for row in range(shape[0])]))
    climate = SimpleNamespace(mean_annual_temperature_c=np.full(shape, temperature),
        warmest_month_temperature_c=np.full(shape, temperature + 10.0),
        annual_precipitation_mm=np.full(shape, rain),
        seasonal_precipitation=np.full((4, *shape), rain / 8400.0))
    return grid, climate


def cover(grid,climate):
    return derive_vegetation_cover(grid,climate,ecological_sources=declared_freshwater_sources(grid))


class VegetationTests(unittest.TestCase):
    def test_water_supply_increases_cover_without_rainfall_cliffs(self):
        fractions = [float(cover(*fixture(rain=rain)).fraction[4, 8])
                     for rain in (0.0, 100.0, 500.0, 1500.0, 3000.0)]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[0], 0.0)
        self.assertGreater(fractions[-1], .95)
        before = cover(*fixture(rain=500.0 - .001)).fraction
        after = cover(*fixture(rain=500.0 + .001)).fraction
        self.assertLess(float((after - before).max()), 1e-5)

    def test_river_support_stays_local_and_snow_water_remain_bare(self):
        grid, climate = fixture(rain=0.0)
        grid.river_order[:, 8] = 3
        grid.water[3, 8] = 2
        grid.snow[4, 8] = True
        fraction = cover(grid, climate).fraction
        self.assertGreater(float(fraction[2, 8]), .7)
        self.assertGreater(float(fraction[2, 9]), 0.0)
        self.assertEqual(float(fraction[2, 11]), 0.0)
        self.assertEqual(float(fraction[3, 8]), 0.0)
        self.assertEqual(float(fraction[4, 8]), 0.0)

    def test_cold_and_dry_seasons_limit_cover_continuously(self):
        cold = cover(*fixture(temperature=-10.0)).fraction
        warm = cover(*fixture()).fraction
        grid, climate = fixture()
        climate.seasonal_precipitation[0] = 0.0
        seasonal = cover(grid, climate).fraction
        self.assertTrue(np.all(cold == 0.0))
        self.assertTrue(np.all(seasonal < warm))
        self.assertTrue(np.all(seasonal > 0.0))

    def test_coastal_display_zero_does_not_create_false_bare_land(self):
        grid, climate = fixture()
        grid.water[:, :4] = 1
        grid.elevation[:, :4] = 0.0
        elevation = grid.elevation.copy()
        fraction = cover(grid, climate).fraction
        self.assertEqual(float(fraction[4, 4]), float(fraction[4, 8]))
        np.testing.assert_array_equal(elevation, grid.elevation)
        self.assertFalse(fraction.flags.writeable)
