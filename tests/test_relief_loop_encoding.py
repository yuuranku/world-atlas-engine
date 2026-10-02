import unittest

import numpy as np
import shapely

from world_atlas.core.cartographic_relief import _closed_relief_curves
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.implicit_curves import level_bands
from world_atlas.core.implicit_terrain import terrain_level_curves


class ReliefLoopEncodingTests(unittest.TestCase):
    def setUp(self):
        rows, columns = np.indices((9, 11))
        native = 100 + 800*np.exp(-((rows-4)**2+(columns-5)**2)/4)
        self.field = PhysicalTerrainField(native, land_mask=native > 0,
            sea_level_m=0, elevation_scale_m=4000, elevation_exponent=1)
        self.level = 350.
        self.curves = terrain_level_curves(self.field, [self.level])

    def test_root_roundoff_loop_closes_without_moving_saved_curve_vertices(self):
        chain = self.curves[0][0].copy()
        chain[-1, 0] = np.nextafter(chain[-1, 0], np.inf)
        original = chain.copy()
        with self.assertRaisesRegex(ValueError, 'closed shared coverage'):
            level_bands(self.field, [self.level], [[chain]])
        encoded = _closed_relief_curves(self.field, [[chain]])
        np.testing.assert_array_equal(chain, original)
        np.testing.assert_array_equal(encoded[0][0][:-1], original)
        np.testing.assert_array_equal(encoded[0][0][0], encoded[0][0][-1])
        bands = level_bands(self.field, [self.level], encoded)
        self.assertTrue(shapely.coverage_is_valid(np.asarray(bands)))
        self.assertTrue(bands[1].covers(shapely.Point(5.5, 4.5)))
        self.assertEqual(shapely.union_all(bands).area, 99.)

    def test_real_gaps_and_frame_curves_are_not_closed(self):
        chain = self.curves[0][0].copy()
        chain[-1, 0] += 1e-4
        encoded = _closed_relief_curves(self.field, [[chain]])
        np.testing.assert_array_equal(encoded[0][0], chain)
        with self.assertRaisesRegex(ValueError, 'closed shared coverage'):
            level_bands(self.field, [self.level], encoded)
        frame = np.array(((0., 2.), (0., 3.), (0., np.nextafter(2., np.inf))))
        encoded = _closed_relief_curves(self.field, [[frame]])
        np.testing.assert_array_equal(encoded[0][0], frame)


if __name__ == '__main__':
    unittest.main()
