"""Physical outlet edges share generator semantics and accepted metre slopes."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.river_network import hydrologic_outlet_targets


def outlet_fixture(*, latitude=0.):
    shape = (3, 3)
    grid = SimpleNamespace(shape=shape, water=np.zeros(shape, dtype=np.uint8),
        flow_to=np.full(shape, -1, dtype=np.int32), river_order=np.zeros(shape, dtype=np.uint8),
        metadata={"extents": {"west": -1.5, "east": 1.5, "north": latitude + 1.5, "south": latitude - 1.5},
                  "planet": {"radiusKm": 6371.}})
    grid.river_order[1, 1] = 2
    return grid, np.ones(shape)


class RiverNetworkTests(unittest.TestCase):
    def test_ocean_descent_uses_physical_distance_not_nearest_wet_cell(self):
        grid, raw = outlet_fixture()
        grid.water[0, 1] = grid.water[0, 2] = grid.water[1, 2] = 1
        raw[0, 1], raw[0, 2], raw[1, 2] = -1., -5., -3.
        self.assertEqual(hydrologic_outlet_targets(grid, raw), {4: 2})
        # Same D8 candidates at high latitude make east physically shorter.
        grid, raw = outlet_fixture(latitude=70.)
        grid.water[0, 1] = grid.water[1, 2] = 1
        raw[0, 1], raw[1, 2] = -1., -.1
        self.assertEqual(hydrologic_outlet_targets(grid, raw), {4: 5})

    def test_equal_flat_outlet_choices_keep_generator_d8_tie_order(self):
        grid, raw = outlet_fixture()
        grid.water[0, 1] = grid.water[1, 2] = 1
        raw[0, 1] = raw[1, 2] = -1.
        self.assertEqual(hydrologic_outlet_targets(grid, raw), {4: 1})
        self.assertEqual(hydrologic_outlet_targets(grid, raw), {4: 1})

    def test_fresh_lake_inland_sea_and_inland_sink_do_not_get_guessed_edges(self):
        for code in (0, 2, 3):
            grid, raw = outlet_fixture()
            grid.water[0, 1] = code
            raw[0, 1] = 1. if code == 0 else -1.
            self.assertEqual(hydrologic_outlet_targets(grid, raw), {})

    def test_explicit_flow_target_and_unrepresented_coast_remain_unchanged(self):
        grid, raw = outlet_fixture()
        grid.water[0, 1], raw[0, 1] = 1, -1.
        grid.flow_to[1, 1] = 1
        self.assertEqual(hydrologic_outlet_targets(grid, raw), {})
        grid.flow_to[1, 1], grid.river_order[1, 1] = -1, 0
        self.assertEqual(hydrologic_outlet_targets(grid, raw), {})

    def test_outlet_requires_matching_accepted_sign_and_negative_water(self):
        grid, raw = outlet_fixture()
        grid.water[0, 1], raw[0, 1] = 1, 0.
        with self.assertRaisesRegex(ValueError, "no negative accepted-ground outlet"):
            hydrologic_outlet_targets(grid, raw)
        raw[0, 1] = .1
        with self.assertRaisesRegex(ValueError, "native physical land sign"):
            hydrologic_outlet_targets(grid, raw)
        raw[0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite accepted metres"):
            hydrologic_outlet_targets(grid, raw)

    def test_canonical_arrays_and_accepted_ground_are_not_mutated(self):
        grid, raw = outlet_fixture()
        grid.water[0, 1], raw[0, 1] = 1, -1.
        before = [array.copy() for array in (grid.water, grid.flow_to, grid.river_order, raw)]
        for array in (grid.water, grid.flow_to, grid.river_order, raw):
            array.flags.writeable = False
        self.assertEqual(hydrologic_outlet_targets(grid, raw), {4: 1})
        for expected, array in zip(before, (grid.water, grid.flow_to, grid.river_order, raw), strict=True):
            np.testing.assert_array_equal(expected, array)

    def test_all_381_saved_v91_ocean_terminals_keep_the_same_physical_outlet(self):
        document = json.loads((Path(__file__).parent / "fixtures/river-outlets-v91.json").read_text(encoding="utf-8"))
        self.assertEqual(len(document["outlets"]), 381)
        height, width = document["shape"]
        source_extent = document["extents"]
        dy = (source_extent["north"] - source_extent["south"]) / height
        dx = (source_extent["east"] - source_extent["west"]) / width
        for saved in document["outlets"]:
            grid, _ = outlet_fixture()
            grid.water = np.asarray(saved["water"], dtype=np.uint8)
            grid.river_order[1, 1] = saved["order"]
            north = source_extent["north"] - (saved["row"] - 1) * dy
            west = source_extent["west"] + (saved["column"] - 1) * dx
            grid.metadata = {"extents": {"north": north, "south": north - 3 * dy, "west": west, "east": west + 3 * dx},
                             "planet": {"radiusKm": document["radiusKm"]}}
            raw = np.asarray(saved["raw"])
            row_delta, column_delta = saved["direction"]
            expected = (1 + row_delta) * 3 + 1 + column_delta
            with self.subTest(source=(saved["row"], saved["column"])):
                self.assertEqual(hydrologic_outlet_targets(grid, raw), {4: expected})


if __name__ == "__main__":
    unittest.main()
