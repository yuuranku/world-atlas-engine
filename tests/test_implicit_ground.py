"""True physical zero branches retain tiny features and solved map cuts."""

from types import SimpleNamespace
import unittest

import numpy as np
import shapely
from scipy.optimize import brentq

from world_atlas.core.cartographic_surface import continuous_land_surface
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.implicit_ground import ground_zero_paths
from tests.test_terrain_refinement import fixture


def base_field(values):
    return PhysicalTerrainField(values, land_mask=values > 0,
        sea_level_m=0, elevation_scale_m=10, elevation_exponent=1)


def surface(field):
    return continuous_land_surface(SimpleNamespace(water=(field.native_m <= 0).astype(np.uint8)),
                                   terrain_field=field)


class ImplicitGroundTests(unittest.TestCase):
    def test_real_tensor_slope_switch_is_a_shared_zero_port(self):
        values = np.random.default_rng(89323).uniform(100, 2000, (9, 11))-350.001
        field = base_field(values)
        coordinates = np.concatenate(ground_zero_paths(field))
        corner = np.array((4.27165524165717, 1.9827907997308))
        self.assertGreaterEqual(int((np.linalg.norm(coordinates-corner, axis=1) < 1e-12).sum()), 2)
        self.assertLess(float(abs(field.sample_points(coordinates[:, 0], coordinates[:, 1])).max()), 1e-10)
        geometry = surface(field)
        self.assertTrue(geometry.is_valid)
        rows, columns = np.indices(values.shape)
        np.testing.assert_array_equal(shapely.contains_xy(geometry, columns+.5, rows+.5), values > 0)

    def test_barely_positive_island_and_negative_lake_use_true_curved_zero(self):
        for sign in (1., -1.):
            values = np.full((7, 7), -sign)
            values[3, 3] = sign*.0001
            field = base_field(values)
            geometry = surface(field)
            ring = (geometry.exterior if sign > 0 else geometry.interiors[0])
            coordinates = np.asarray(ring.coords)
            self.assertGreater(len(coordinates), 32)
            root = brentq(lambda distance: float(field.sample_points(3.5+distance, 3.5)), 0., .25)
            self.assertAlmostEqual(coordinates[:, 0].max()-3.5, root, places=12)
            self.assertAlmostEqual(3.5-coordinates[:, 0].min(), root, places=12)
            self.assertLess(float(abs(field.sample_points(coordinates[:, 0], coordinates[:, 1])).max()), 1e-12)
            self.assertEqual(geometry.contains(shapely.Point(3.5, 3.5)), sign > 0)

    def test_constant_zero_shore_retains_native_edge_root_roundoff_conditioning(self):
        values = np.full((12, 24), -20.)
        values[3:9, 3:20] = 10.
        field = base_field(values)
        coordinates = np.concatenate(ground_zero_paths(field))
        self.assertLess(float(abs(field.sample_points(coordinates[:, 0], coordinates[:, 1])).max()), 1e-12)
        geometry = surface(field)
        rows, columns = np.indices(values.shape)
        np.testing.assert_array_equal(shapely.contains_xy(geometry, columns+.5, rows+.5), values > 0)

    def test_exact_zero_plateau_is_water_and_is_not_moved_below_zero(self):
        values = np.zeros((7, 12))
        values[2:5, 3:6] = 10.
        geometry = surface(base_field(values))
        self.assertTrue(geometry.is_valid)
        self.assertTrue(geometry.contains(shapely.Point(4.5, 3.5)))
        self.assertFalse(geometry.contains(shapely.Point(8.5, 3.5)))
        self.assertFalse(geometry.contains(shapely.Point(2.5, 3.5)))
        self.assertTrue(geometry.disjoint(shapely.box(7., 0., 12., 7.)))
        self.assertTrue(surface(base_field(np.zeros((7, 12)))).is_empty)

    def test_transported_meridian_ports_are_true_roots_shared_across_period(self):
        rows, columns = np.indices((24, 48))
        values = (rows-11.2)*30+150*np.cos((columns+.5)/48*2*np.pi*3+.7)
        base, field = fixture(values, seed=2)
        old_root = brentq(lambda y: float(base.sample_points(0., y)), 0., 24.)
        self.assertGreater(abs(float(field.sample_points(0., old_root))), .5)
        coordinates = np.concatenate(ground_zero_paths(field))
        left = coordinates[coordinates[:, 0] == 0]
        right = coordinates[coordinates[:, 0] == 48]
        self.assertGreater(len(left), 0)
        np.testing.assert_array_equal(np.sort(left[:, 1]), np.sort(right[:, 1]))
        self.assertGreater(abs(float(left[0, 1])-old_root), .04)
        self.assertLess(float(abs(field.sample_points(coordinates[:, 0], coordinates[:, 1])).max()), 1e-10)
        geometry = surface(field)
        self.assertTrue(geometry.is_valid)
        np.testing.assert_array_equal(shapely.contains_xy(geometry, columns+.5, rows+.5), values > 0)


if __name__ == "__main__":
    unittest.main()
