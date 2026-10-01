import math
from types import SimpleNamespace
import unittest

import numpy as np
import shapely

from world_atlas.core.cartographic_rivers import (
    river_channel_surface,
    river_width_field,
    river_width_profile,
)


def grid_fixture(shape, *, extents=None):
    return SimpleNamespace(
        shape=shape,
        water=np.zeros(shape, dtype=np.uint8),
        flow_to=np.full(shape, -1, dtype=np.int32),
        discharge=np.ones(shape, dtype=np.float32),
        river_order=np.ones(shape, dtype=np.uint8),
        metadata={"planet": {"radiusKm": 6400}, "extents": extents or {
            "west": -180, "east": 180, "north": 90, "south": -90}},
    )


class CartographicRiverTests(unittest.TestCase):
    def test_same_drainage_area_retains_width_when_the_grid_is_refined(self):
        outlet_widths = []
        for shape in ((4, 8), (8, 16)):
            grid = grid_fixture(shape)
            # One acyclic, locally adjacent drainage chain receives the same
            # entire planet area on either resolution.
            chain = [row * shape[1] + column
                     for row in range(shape[0])
                     for column in (range(shape[1]) if row % 2 == 0
                                    else range(shape[1] - 1, -1, -1))]
            grid.flow_to.ravel()[chain[:-1]] = chain[1:]
            before = grid.flow_to.copy()
            widths = river_width_field(grid)
            outlet_widths.append(widths.flat[chain[-1]])
            np.testing.assert_array_equal(grid.flow_to, before)
            self.assertTrue(np.all(np.diff(widths.ravel()[chain]) > 0))
        self.assertAlmostEqual(outlet_widths[0], outlet_widths[1], places=9)
        # Half a metre times sqrt(km²), based on planetary area rather than
        # the unrelated stored cell-count discharge.
        self.assertAlmostEqual(outlet_widths[0], 0.5 * math.sqrt(4 * math.pi * 6400**2))

    def test_equal_cell_counts_use_latitude_correct_catchment_area(self):
        grid = grid_fixture((4, 8))
        widths = river_width_field(grid)
        self.assertGreater(widths[1, 2], widths[0, 2])
        self.assertAlmostEqual(widths[0, 2], widths[3, 2])
        grid.water[1, 1] = 2
        grid.river_order[1, 2] = 0
        widths = river_width_field(grid)
        self.assertEqual(widths[1, 1], 0)
        self.assertEqual(widths[1, 2], 0)

    def test_confluence_widths_and_mouth_are_preserved_on_off_cell_curves(self):
        grid = grid_fixture((5, 5))
        grid.water[:] = 1
        grid.river_order[:] = 0
        for row, column in ((1, 1), (1, 3), (2, 2), (3, 2)):
            grid.water[row, column] = 0
            grid.river_order[row, column] = 1
        grid.flow_to[1, 1] = grid.flow_to[1, 3] = 12
        grid.flow_to[2, 2] = 17
        field = river_width_field(grid)
        incoming = np.array(((1.5, 1.5), (2.5, 2.5)))
        incoming_curve = np.array(((1.5, 1.5), (1.4, 2.2), (2.5, 2.5)))
        downstream = np.array(((2.5, 2.5), (2.5, 3.5), (2.25, 4.1)))
        downstream_curve = np.array(((2.5, 2.5), (2.2, 3.1), (2.5, 3.5), (2.25, 4.1)))
        upper_profile = river_width_profile(grid, incoming, incoming_curve, field)
        lower_profile = river_width_profile(grid, downstream, downstream_curve, field)
        self.assertEqual(upper_profile[-1], lower_profile[0])
        self.assertGreater(upper_profile[-1], upper_profile[0])
        self.assertTrue(np.all(np.diff(upper_profile) >= 0))
        self.assertEqual(lower_profile[-1], field[3, 2])
        self.assertTrue(np.isfinite(lower_profile).all())
        upper = river_channel_surface(grid, incoming_curve, upper_profile)
        lower = river_channel_surface(grid, downstream_curve, lower_profile)
        self.assertGreater(upper.intersection(lower).area, 0)
        self.assertTrue(shapely.is_valid(shapely.union_all((upper, lower))))

    def test_channel_cross_sections_follow_metres_and_projection_not_screen_ink(self):
        grid = grid_fixture((8, 16), extents={"west": -8, "east": 8, "north": 4, "south": -4})
        points = np.array(((8., 1.), (8., 4.), (8., 7.)))
        profile = np.array((20., 100., 180.))
        channel = river_channel_surface(grid, points, profile)
        self.assertTrue(shapely.is_valid(channel))
        for point, expected_width in zip(points, profile, strict=True):
            cross_section = channel.intersection(shapely.LineString(((0, point[1]), (16, point[1]))))
            latitude = math.radians(4 - point[1])
            actual_metres = cross_section.length * 6_400_000 * math.radians(1) * math.cos(latitude)
            self.assertAlmostEqual(actual_metres, expected_width, places=6)
        self.assertGreater(channel.area, 0)
        with self.assertRaises(ValueError):
            river_channel_surface(grid, points, np.array((1., 0., 1.)))
        with self.assertRaises(ValueError):
            river_width_profile(grid, points, points + (0.1, 0), np.ones(grid.shape))


if __name__ == "__main__":
    unittest.main()
