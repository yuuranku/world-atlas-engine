import unittest
import numpy as np
from scipy import ndimage
from world_atlas.core.coastal_drainage import drown_coastal_valleys, shoreline_distance


class CoastalDrainageTests(unittest.TestCase):
    def test_coastal_support_decays_into_both_interior_and_ocean(self):
        land = np.zeros((20, 80), dtype=bool)
        land[:, 40:] = True
        inside = ndimage.distance_transform_edt(land) * 20
        outside = ndimage.distance_transform_edt(~land) * 20
        distance = shoreline_distance(inside, outside)
        self.assertEqual(distance[10, 39], 20)
        self.assertEqual(distance[10, 40], 20)
        self.assertEqual(distance[10, 0], 800)
        self.assertEqual(distance[10, 79], 800)
        support = np.exp(-0.5 * (distance / 128)**2)
        self.assertLess(support[10, 0], 1e-8)
        self.assertLess(support[10, 79], 1e-8)
        self.assertGreater(support[10, 40], 0.98)

    def test_floods_existing_valley_but_preserves_ridges_and_lake_isolation(self):
        z = np.full((40, 60), 650.0, dtype=np.float32)
        z[:, :10] = -100.0
        z[18:22, 10:28] = 35.0
        z[4:7, 20:24] = 20.0
        flow = np.ones_like(z)
        flow[18:22, 10:28] = 200
        flow[4:7, 20:24] = 200
        result, metrics = drown_coastal_valleys(z, flow, np.zeros_like(z), 10)
        self.assertGreater(metrics['drownedCells'], 0)
        self.assertTrue((result[18:22, 10:20] < 0).all())
        self.assertTrue((result[4:7, 20:24] > 0).all())
        self.assertTrue((result[10:14, 10:28] > 0).all())
        np.testing.assert_array_equal(result[z < 0], z[z < 0])
        labels, _ = ndimage.label(result <= 0)
        self.assertTrue((labels[(z > 0) & (result <= 0)] == labels[0, 0]).all())
        repeated, _ = drown_coastal_valleys(z, flow, np.zeros_like(z), 10)
        np.testing.assert_array_equal(result, repeated)
