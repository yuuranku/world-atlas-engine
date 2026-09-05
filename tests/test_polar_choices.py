"""North/south choices must survive the physical and metadata boundaries."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.api import load_recipe
from world_atlas.core.polar import derive_polar_regions, polar_continent_mask
from world_atlas.core.procedural_planet import _polar_crust_bias


class PolarChoiceTests(unittest.TestCase):
    def test_each_pole_changes_only_its_own_crust_domain(self):
        recipe = load_recipe(Path(__file__).resolve().parents[1] / "examples/terrain.json")
        latitudes = 90. - (np.arange(72) + .5) * 180. / 72
        field = np.random.default_rng(71).normal(0, .2, (72, 144)).astype(np.float32)
        for north in (False, True):
            for south in (False, True):
                choice = replace(recipe, north_polar_continent=north, south_polar_continent=south)
                output = _polar_crust_bias(field, latitudes, choice)
                self.assertTrue(np.all((output[0] > 0) == north))
                self.assertTrue(np.all((output[-1] > 0) == south))
                np.testing.assert_array_equal(output[np.abs(latitudes) < 58], field[np.abs(latitudes) < 58])
                opposite = _polar_crust_bias(field, latitudes, replace(choice, south_polar_continent=not south))
                np.testing.assert_array_equal(output[latitudes > 0], opposite[latitudes > 0])

    def test_single_polar_continent_and_ocean_ice_are_independent(self):
        shape = (36, 72)
        extents = dict(west=-180., east=180., south=-90., north=90.)
        for north in (False, True):
            for south in (False, True):
                north_land = np.zeros(shape, bool)
                south_land = np.zeros(shape, bool)
                north_land[:3] = north
                south_land[-3:] = south
                water = (~(north_land | south_land)).astype(np.uint8)
                result = derive_polar_regions(shape, extents, water, north_land, south_land)
                self.assertFalse((result.sea_ice & (water == 0)).any())
                self.assertTrue(np.isfinite(result.north_coast_latitude).all())
                self.assertTrue(np.isfinite(result.south_coast_latitude).all())
                grid = SimpleNamespace(shape=shape, water=water, metadata={
                    "extents": extents, "planet": {"axialTiltDegrees":18.},
                    "polarRegions": {"arctic":{"surface":"continental-land" if north else "ocean"},
                                     "antarctic":{"surface":"continental-land" if south else "ocean"}}})
                np.testing.assert_array_equal(polar_continent_mask(grid), north_land | south_land)


if __name__ == "__main__":
    unittest.main()
