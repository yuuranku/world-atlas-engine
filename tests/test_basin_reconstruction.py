import unittest
import numpy as np
from world_atlas.core.basin_reconstruction import reconstruct_basin


class BasinReconstructionTests(unittest.TestCase):
    def test_constant_rim_and_target_stay_constant(self):
        z = np.full((20, 25), 120.0)
        mask = np.zeros_like(z, dtype=bool)
        mask[3:17, 3:22] = True
        z[mask] = -50
        np.testing.assert_allclose(reconstruct_basin(z, mask, 120), 120)

    def test_different_rims_blend_without_nearest_neighbor_seam(self):
        z = np.full((30, 30), 100.0)
        z[:, 16:] = 800
        mask = np.zeros_like(z, dtype=bool)
        mask[3:27, 3:27] = True
        z[mask] = -100
        original = z.copy()
        result = reconstruct_basin(z, mask, 150)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(result.min(), 100)
        self.assertLessEqual(result.max(), 800)
        z[mask] = result
        self.assertLess(np.max(np.abs(np.diff(z[15, 6:24]))), 80)
        np.testing.assert_array_equal(original[~mask], z[~mask])
        np.testing.assert_array_equal(result, reconstruct_basin(original, mask, 150))
