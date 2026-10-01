"""Drainage position, retention and hydroperiod cause regional marsh support."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.thematic import ClimateDrivers, derive_physiographic_features, KOPPEN_CFB
from world_atlas.core.wetlands import derive_wetland_support, _lake_bank_level


def valley_fixture(rain=500.0):
    height, width = shape = (11, 15)
    row, column = np.indices(shape)
    elevation = (.14 + .0007 * (column - 7) ** 2 + .0003 * (height - row)).astype(np.float32)
    flow = (row * width + column + np.sign(7 - column)).astype(np.int32)
    flow[:, 7] = np.minimum(row[:, 7] + 1, height - 1) * width + 7
    flow[-1, 7] = -1
    river = np.zeros(shape, dtype=np.uint8)
    river[:, 7] = 2
    grid = SimpleNamespace(shape=shape, elevation=elevation,
        metadata={"extents": {"west": 0., "east": .15, "south": -.055, "north": .055},
                  "planet": {"radiusKm": 6400.}},
        water=np.zeros(shape, dtype=np.uint8), snow=np.zeros(shape, dtype=bool),
        flow_to=flow, river_order=river,
        discharge=np.full(shape, 500., dtype=np.float32),
        seasonal_river_strength=np.where(river[None, :, :] > 0, 160, 0).repeat(4, axis=0).astype(np.uint8))
    climate = ClimateDrivers(temperature=np.full(shape,.6),
        seasonal_precipitation=np.full((4,*shape),rain/8400),
        annual_precipitation=np.full(shape,rain/8400), precipitation_range=np.zeros(shape),
        mean_annual_temperature_c=np.full(shape,20.,dtype=np.float32),
        annual_precipitation_mm=np.full(shape,rain,dtype=np.float32),
        coldest_month_temperature_c=np.full(shape,10.,dtype=np.float32),
        warmest_month_temperature_c=np.full(shape,28.,dtype=np.float32),
        koppen_code=np.full(shape,KOPPEN_CFB,dtype=np.int8))
    return grid, climate


class WetlandSupportTests(unittest.TestCase):
    def test_supplied_valley_floor_is_wet_but_sloping_banks_are_not(self):
        grid, climate = valley_fixture()
        score = derive_wetland_support(grid, climate)
        self.assertGreater(float(score[5, 6]), .5)
        self.assertLess(float(score[5, 2]), .5)
        self.assertEqual(score.dtype, np.dtype(np.float32))
        self.assertGreater(np.unique(score).size, 10)

    def test_spatially_near_water_cannot_cross_a_drainage_divide(self):
        grid, climate = valley_fixture()
        grid.flow_to[:, :7] = -1
        score = derive_wetland_support(grid, climate)
        self.assertEqual(float(score[5, 6]), 0.)
        self.assertGreater(float(score[5, 8]), .5)

    def test_high_discharge_and_rain_without_a_channel_do_not_invent_marshes(self):
        grid, climate = valley_fixture(rain=2200.)
        grid.river_order.fill(0)
        grid.discharge.fill(100000.)
        self.assertFalse(np.any(derive_wetland_support(grid, climate)))

    def test_a_perennial_channel_can_supply_dryland_marshes(self):
        grid, climate = valley_fixture(rain=20.)
        wet = derive_wetland_support(grid, climate)
        grid.seasonal_river_strength[0].fill(0)
        seasonal = derive_wetland_support(grid, climate)
        self.assertGreater(float(wet[5, 6]), .5)
        self.assertEqual(float(seasonal[5, 6]), 0.)

    def test_lake_zero_display_height_does_not_drain_a_flat_shore(self):
        grid, climate = valley_fixture()
        grid.river_order.fill(0)
        grid.water[:, 7] = 2
        grid.elevation.fill(.25)
        grid.elevation[:, 7] = 0.
        score = derive_wetland_support(grid, climate)
        self.assertGreater(float(score[5, 6]), .5)
        self.assertFalse(score[:, 7].any())

    def test_one_lake_keeps_one_stage_across_the_longitude_seam(self):
        elevation = np.tile(np.asarray([0., .2, .6, .4, 0.]), (3, 1))
        lake = elevation == 0.
        level = _lake_bank_level(elevation, ~lake, lake)
        np.testing.assert_allclose(level[lake], .2)

    def test_remote_high_ground_does_not_change_local_classification(self):
        grid, climate = valley_fixture()
        before = derive_wetland_support(grid, climate)
        grid.elevation[:, :3] = .9
        after = derive_wetland_support(grid, climate)
        np.testing.assert_array_equal(before[:, 6:10], after[:, 6:10])

    def test_snow_and_no_growing_season_have_no_vegetated_marsh(self):
        grid, climate = valley_fixture()
        grid.snow[5, 6] = True
        self.assertEqual(float(derive_wetland_support(grid, climate)[5, 6]), 0.)
        climate = replace(climate,
            warmest_month_temperature_c=np.zeros(grid.shape,dtype=np.float32),
            coldest_month_temperature_c=np.full(grid.shape,-10.,dtype=np.float32))
        self.assertFalse(np.any(derive_wetland_support(grid, climate)))

    def test_longitude_roll_preserves_drainage_and_support(self):
        grid, climate = valley_fixture()
        before = derive_wetland_support(grid, climate)
        width = grid.shape[1]
        shift = 9
        flow = grid.flow_to.copy()
        valid = flow >= 0
        flow[valid] = (flow[valid] // width) * width + (flow[valid] % width + shift) % width
        for name in ("elevation", "water", "snow", "river_order", "discharge"):
            setattr(grid, name, np.roll(getattr(grid,name), shift, axis=1))
        grid.flow_to = np.roll(flow,shift,axis=1)
        grid.seasonal_river_strength = np.roll(grid.seasonal_river_strength,shift,axis=2)
        after = derive_wetland_support(grid, climate)
        np.testing.assert_allclose(np.roll(before,shift,axis=1),after,atol=1e-7)

    def test_cycles_are_rejected_instead_of_assigning_arbitrary_water(self):
        grid, climate = valley_fixture()
        grid.flow_to[5, 4] = 5 * 15 + 5
        grid.flow_to[5, 5] = 5 * 15 + 4
        with self.assertRaisesRegex(ValueError, "cycle"):
            derive_wetland_support(grid, climate)

    def test_physiography_mask_and_continuous_support_share_a_threshold(self):
        grid, climate = valley_fixture()
        features = derive_physiographic_features(grid, climate)
        np.testing.assert_array_equal(features.wetland, features.wetland_support >= .5)
        self.assertFalse(features.wetland_support.flags.writeable)


if __name__ == "__main__":
    unittest.main()
