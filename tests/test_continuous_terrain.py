import unittest

import contourpy
import numpy as np

from world_atlas.core.continuous_terrain import PhysicalTerrainField


def field(values):
    values = np.asarray(values, dtype=np.float64)
    return PhysicalTerrainField(values, land_mask=values > 0, sea_level_m=700,
                                elevation_scale_m=1200, elevation_exponent=1.06)


class ContinuousTerrainTests(unittest.TestCase):
    def test_native_samples_and_input_are_preserved_without_shore_noise(self):
        values = np.array([[-50, -20, 2, 100, 20, -30],
                           [-40, -5, 10, 600, 30, -70],
                           [-60, -3, 40, 400, 50, -80]], dtype=float)
        before = values.copy()
        terrain = field(values)
        sampled = terrain.sample_rect(np.arange(6) + .5, np.arange(3) + .5)
        np.testing.assert_array_equal(sampled, values)
        np.testing.assert_array_equal(values, before)
        self.assertFalse(terrain.native_m.flags.writeable)

    def test_tensor_values_stay_within_containing_native_corners(self):
        values = np.array([[-100, 1, 400, -20, 2], [-5, 30, 7, -80, 9],
                           [10, 4000, 1, -1, 90], [-300, 2, -500, 3, 20]], dtype=float)
        x = np.arange(.5, 4.5 + .01, .05)
        y = np.arange(.5, 3.5 + .01, .05)
        sampled = field(values).sample_rect(x, y)
        columns = np.minimum(np.floor(x - .5).astype(int), 3)
        rows = np.minimum(np.floor(y - .5).astype(int), 2)
        corners = np.stack([values[rows[:, None] + dy, columns[None, :] + dx]
                            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))])
        self.assertTrue(np.all(sampled >= corners.min(axis=0) - 1e-9))
        self.assertTrue(np.all(sampled <= corners.max(axis=0) + 1e-9))

    def test_longitude_and_shared_tile_edges_use_identical_samples(self):
        values = np.array([[10, 20, -30, -60, -5, 2], [20, 30, -20, -40, -2, 4],
                           [5, 6, -20, -80, -9, 8]], dtype=float)
        terrain = field(values)
        x, y = np.linspace(0, 6, 97), np.linspace(0, 3, 49)
        full = terrain.sample_rect(x, y)
        np.testing.assert_allclose(full[:, 0], full[:, -1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(terrain.sample_rect(x + 6, y), full, rtol=0, atol=1e-12)
        np.testing.assert_allclose(terrain.sample_rect(x[40:], y[18:]), full[18:, 40:],
                                   rtol=0, atol=1e-12)

    def test_lake_and_ocean_zero_lines_and_land_contours_use_one_ground(self):
        values = np.full((8, 12), -40.)
        values[1:7, 2:10] = 120
        values[3:5, 5:7] = -3  # The source defines this enclosed water at its shared datum.
        terrain = field(values)
        x, y = np.arange(193) / 16, np.arange(129) / 16
        heights = terrain.sample_rect(x, y)
        generator = contourpy.contour_generator(x=x, y=y, z=heights)
        lines = generator.lines(0)
        self.assertEqual(len(lines), 2)
        level = .15
        physical = terrain.contour_height_m(level)
        self.assertAlmostEqual(float(terrain.palette_elevation(physical)), level)
        source_colours = terrain.palette_elevation(values)
        np.testing.assert_array_equal(source_colours[values < 0], 0)
        # No artificial finite shore band changes the constant native plateau.
        self.assertEqual(float(heights[24, 40]), 120)

    def test_invalid_or_unmatched_physical_sources_are_rejected(self):
        with self.assertRaises(ValueError):
            PhysicalTerrainField([[1, -2]], land_mask=[[True, True]], sea_level_m=0,
                                 elevation_scale_m=1, elevation_exponent=1.06)
        for values in ([[float('nan')]], [], [1, 2]):
            with self.assertRaises(ValueError):
                field(values)
        with self.assertRaises(ValueError):
            field([[1, -1]]).sample_rect([1, 0], [.5])

    def test_paired_samples_match_rectangles_including_poles_and_longitude(self):
        values = np.array([[10, -40, 500, 30], [-200, 4, -70, 3], [4, 6, -80, 60]])
        terrain = field(values)
        x, y = np.linspace(-1, 5, 71), np.linspace(0, 3, 61)
        xx, yy = np.meshgrid(x, y)
        np.testing.assert_allclose(terrain.sample_points(xx, yy), terrain.sample_rect(x, y),
                                   rtol=0, atol=1e-12)
        np.testing.assert_array_equal(terrain.sample_points(np.arange(4) + .5, .5), values[0])


if __name__ == '__main__':
    unittest.main()
