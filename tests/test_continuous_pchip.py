"""Cache reuse preserves SciPy's binary64 model and bounded worker payloads."""
import pickle
import unittest

import numpy as np
from scipy.interpolate import PchipInterpolator

from world_atlas.core.continuous_scalar import ContinuousScalarField
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.implicit_terrain_bounds import _Dual, _polynomial


class ContinuousPchipTests(unittest.TestCase):
    def test_cached_intervals_match_independent_four_neighbour_scipy_polynomials(self):
        rng = np.random.default_rng(9271)
        for shape in ((1, 1), (1, 2), (2, 1), (65, 19)):
            values = rng.normal(size=shape)*800
            values[::3, ::2] = 0
            values[1::4, ::3] = -0.
            fields = (PhysicalTerrainField(values, land_mask=values > 0,
                sea_level_m=0, elevation_scale_m=1, elevation_exponent=1),
                ContinuousScalarField(values, np.ones(shape, bool)))
            for field in fields:
                with self.subTest(shape=shape, field=type(field).__name__):
                    coefficients = field.horizontal_coefficients
                    self.assertFalse(coefficients.flags.writeable)
                    self.assertIs(coefficients, field.horizontal_coefficients)
                    columns = np.arange(shape[1])
                    local = values[:, (columns[:, None]+np.arange(-1, 3)) % shape[1]]
                    expected = PchipInterpolator(np.arange(-1, 3), local, axis=2).c[:, 1]
                    np.testing.assert_array_equal(coefficients.view(np.uint64), expected.view(np.uint64))
                    self.assertEqual(field.sample_points([], []).shape, (0,))
                    x = rng.uniform(-3*shape[1], 4*shape[1], 17000)
                    y = rng.uniform(0, shape[0], len(x))
                    column = np.floor(x-.5).astype(int)
                    row = np.floor(y-.5).astype(int)
                    # Independent local construction used before coefficient
                    # reuse, including poles, wrapping and chunk boundaries.
                    local = values[np.clip(row[None, None, :]+np.arange(-1, 3)[:, None, None], 0, shape[0]-1),
                                   (column[None, None, :]+np.arange(-1, 3)[None, :, None]) % shape[1]]
                    horizontal = PchipInterpolator(np.arange(-1, 3), local, axis=1).c[:, 1]
                    tx = x-.5-column
                    horizontal = ((horizontal[0]*tx+horizontal[1])*tx+horizontal[2])*tx+horizontal[3]
                    vertical = PchipInterpolator(np.arange(-1, 3), horizontal, axis=0).c[:, 1]
                    ty = y-.5-row
                    expected = ((vertical[0]*ty+vertical[1])*ty+vertical[2])*ty+vertical[3]
                    np.testing.assert_array_equal(field.sample_points(x, y).view(np.uint64), expected.view(np.uint64))
                    self.assertEqual(field.sample_points(.75, .25).shape, ())

    def test_spawn_payload_excludes_derived_cache_and_preserves_immutable_samples(self):
        values = np.arange(65*11, dtype=float).reshape(65, 11)-300
        fields = (PhysicalTerrainField(values, land_mask=values > 0,
            sea_level_m=0, elevation_scale_m=1, elevation_exponent=1),
            ContinuousScalarField(values, np.ones(values.shape, bool)))
        for field in fields:
            before = pickle.dumps(field)
            expected = field.sample_points([-.3, 2.7, 24.4], [0, 34.5, 65])
            self.assertEqual(pickle.dumps(field), before)
            restored = pickle.loads(before)
            self.assertNotIn('horizontal_coefficients', restored.__dict__)
            native = restored.native_m if isinstance(restored, PhysicalTerrainField) else restored.native
            self.assertFalse(native.flags.writeable)
            np.testing.assert_array_equal(restored.sample_points([-.3, 2.7, 24.4], [0, 34.5, 65]).view(np.uint64),
                                          expected.view(np.uint64))

    def test_batched_outward_polynomial_bounds_match_separate_rows_bit_for_bit(self):
        rng = np.random.default_rng(17)
        coefficients = rng.normal(size=(4, 4, 101))
        coefficients[:, 1, ::3] = 0
        position = _Dual(rng.uniform(0, .4, 101), rng.uniform(.6, 1, 101),
                         rng.normal(size=(101, 2)), rng.normal(size=(101, 2)))
        batched = _polynomial(coefficients, position)
        for row in range(4):
            separate = _polynomial(coefficients[:, row], position)
            for attribute in ('low', 'high', 'gradient_low', 'gradient_high'):
                np.testing.assert_array_equal(getattr(batched[row], attribute).view(np.uint64),
                                              getattr(separate, attribute).view(np.uint64))


from world_atlas.core.continuous_pchip import interior_uniform_coefficients

class UniformInteriorPchipTests(unittest.TestCase):
    def test_middle_coefficients_match_scipy_for_switches_plateaus_and_scales(self):
        rng = np.random.default_rng(823173)
        values = rng.normal(size=(4, 4096))*10**rng.uniform(-100,100,(1,4096))
        values[:, :5] = np.asarray(((0.,0.,0.,0.),(1.,1.,1.,1.),
                                   (0.,1.,1.,2.),(1.,3.,2.,5.),
                                   (3.,2.,1.,0.))).T
        expected = PchipInterpolator(np.arange(-1,3), values, axis=0).c[:,1]
        actual = interior_uniform_coefficients(values)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


if __name__ == '__main__':
    unittest.main()
