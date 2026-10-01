"""Transport proximity must not inject Manhattan direction bias into control."""
import unittest

import numpy as np

from world_atlas.core.society.model import Settlement
from world_atlas.core.society.transport import _accessibility_field


def city(identifier, row, column, tier="city"):
    return Settlement(identifier, identifier, row, column, tier, "market",
                      1., 10_000, 20_000)


class TransportAccessibilityTests(unittest.TestCase):
    def test_equal_distance_angles_and_diagonal_have_euclidean_access(self):
        field = _accessibility_field((252, 252), (city("centre", 126, 126),), ())
        self.assertEqual(field[126, 131], field[129, 130])
        self.assertEqual(field[126, 131], field[131, 126])
        self.assertAlmostEqual(float(field[127, 127]), .84 * .88**np.sqrt(2), places=7)

    def test_longitude_wraps_and_poles_do_not(self):
        field = _accessibility_field((252, 252), (city("edge", 0, 0),), ())
        self.assertEqual(field[0, 1], field[0, -1])
        self.assertEqual(field[-1, 0], 0)
        shifted = _accessibility_field((252, 252), (city("edge", 0, 17),), ())
        np.testing.assert_array_equal(shifted, np.roll(field, 17, axis=1))

    def test_source_strength_competes_with_distance(self):
        field = _accessibility_field((252, 252),
            (city("town", 126, 126, "town"), city("capital", 126, 129, "metropolis")), ())
        self.assertAlmostEqual(float(field[126, 126]), .88**3, places=7)
        self.assertEqual(field[126, 129], 1)
        self.assertEqual(field[126, 136], 0)


if __name__ == "__main__":
    unittest.main()
