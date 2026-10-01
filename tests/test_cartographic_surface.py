import unittest
from types import SimpleNamespace

import numpy as np
import shapely

from world_atlas.core.cartographic_surface import continuous_land_surface
from world_atlas.core.continuous_terrain import PhysicalTerrainField


def surface_grid(mask, elevation=10):
    mask = np.asarray(mask)
    values = np.where(mask, elevation, -20.).astype(float)
    terrain = PhysicalTerrainField(values, land_mask=mask, sea_level_m=700,
                                   elevation_scale_m=1000, elevation_exponent=1.06)
    grid = SimpleNamespace(water=np.where(mask, 0, 1).astype(np.uint8))
    return grid, terrain


def surface(mask, elevation=10):
    grid, terrain = surface_grid(mask, elevation)
    return continuous_land_surface(grid, terrain_field=terrain)


class CartographicSurfaceTests(unittest.TestCase):
    def assert_native_centers(self, mask, geometry):
        rows, columns = np.indices(mask.shape)
        points = shapely.points(columns.ravel() + .5, rows.ravel() + .5)
        np.testing.assert_array_equal(shapely.contains(geometry, points).reshape(mask.shape), mask)

    def test_empty_and_full_masks_have_exact_map_frames(self):
        for shape in ((1, 1), (1, 8), (6, 1), (8, 16)):
            self.assertTrue(surface(np.zeros(shape, dtype=bool)).is_empty)
            full = surface(np.ones(shape, dtype=bool))
            self.assertTrue(full.equals(shapely.box(0, 0, shape[1], shape[0])))

    def test_single_cell_islands_and_lake_holes_keep_their_native_centers(self):
        mask = np.zeros((16, 32), dtype=bool)
        mask[4:12, 4:20] = True
        mask[8, 10] = False
        mask[3, 26] = True
        original = mask.copy()
        geometry = surface(mask)
        self.assertTrue(geometry.is_valid)
        self.assertEqual(len(shapely.get_parts(geometry)), 2)
        self.assertEqual(sum(len(part.interiors) for part in shapely.get_parts(geometry)), 1)
        self.assert_native_centers(mask, geometry)
        np.testing.assert_array_equal(mask, original)
        self.assertTrue(geometry.equals_exact(surface(mask), 0.0))

    def test_periodic_coast_meets_at_identical_longitude_frame_intervals(self):
        mask = np.zeros((16, 32), dtype=bool)
        mask[4:12, :5] = True
        mask[4:12, -5:] = True
        mask[6:10, :8] = True
        mask[6:10, -8:] = True
        geometry = surface(mask)
        left = geometry.intersection(shapely.LineString(((0, 0), (0, 16))))
        right = geometry.intersection(shapely.LineString(((32, 0), (32, 16))))
        shifted = shapely.transform(right, lambda xy: xy - np.array((32.0, 0.0)))
        self.assertTrue(left.equals_exact(shifted, 1e-8, normalize=True))
        self.assert_native_centers(mask, geometry)
        self.assertEqual(geometry.bounds[0], 0)
        self.assertEqual(geometry.bounds[2], 32)

    def test_staircase_uses_ground_zero_without_changing_native_ownership(self):
        mask = np.zeros((24, 48), dtype=bool)
        for row in range(4, 20):
            mask[row, 6:row + 12] = True
        geometry = surface(mask)
        self.assert_native_centers(mask, geometry)
        self.assertTrue(geometry.is_valid)
        native = shapely.union_all([shapely.box(column, row, column + 1, row + 1)
                                   for row, column in np.argwhere(mask)])
        self.assertLess(geometry.boundary.hausdorff_distance(native.boundary), .75)

    def test_one_cell_land_bridge_stays_connected_and_water_channel_stays_open(self):
        mask = np.zeros((16, 32), dtype=bool)
        mask[3:13, 3:12] = True
        mask[3:13, 19:28] = True
        mask[7, 12:19] = True
        geometry = surface(mask)
        self.assert_native_centers(mask, geometry)
        self.assertEqual(len(shapely.get_parts(geometry)), 1)
        self.assertTrue(geometry.covers(shapely.LineString(((7.5, 7.5), (24.5, 7.5)))))
        mask[7, 12:19] = False
        mask[:, 12:19] = True
        mask[:, 15] = False
        geometry = surface(mask)
        self.assert_native_centers(mask, geometry)
        self.assertTrue(geometry.disjoint(shapely.LineString(((15.5, 0), (15.5, 16)))))

    def test_raw_relief_changes_the_sea_level_crossing_with_the_same_land_mask(self):
        mask = np.zeros((16, 32), dtype=bool)
        mask[3:9, 5:27] = True
        low = surface(mask, 2)
        high = surface(mask, 120)
        self.assert_native_centers(mask, low)
        self.assert_native_centers(mask, high)
        self.assertGreater(low.symmetric_difference(high).area, 4)
        self.assertGreater(low.boundary.hausdorff_distance(high.boundary), .25)

    def test_ground_is_required_and_must_match_the_same_native_grid(self):
        grid, terrain = surface_grid(np.ones((4, 8), dtype=bool))
        grid.water[2, 3] = 1
        with self.assertRaises(ValueError):
            continuous_land_surface(grid, terrain_field=terrain)


if __name__ == "__main__":
    unittest.main()
