import unittest
import numpy as np
from world_atlas.core.geomorphology import coastal_scarp_transport


class CoastalTransportTests(unittest.TestCase):
    def test_steep_coast_transfers_to_sea_but_preserves_interior(self):
        z = np.full((180, 80), -1000., dtype=np.float32)
        z[:, :40] = 7000
        lat = np.linspace(70, -70, 180)
        out = coastal_scarp_transport(z, lat)
        self.assertLess(out[90, 39], 7000)
        self.assertGreater(out[90, 40], -1000)
        np.testing.assert_array_equal(out[:, 10:20], z[:, 10:20])
        area = np.cos(np.radians(lat))[:, None]
        self.assertLess(abs(np.sum((out-z)*area)), 1)
        np.testing.assert_array_equal(out, coastal_scarp_transport(z, lat))

    def test_low_coast_is_not_forced_down(self):
        z = np.full((180, 80), -100., dtype=np.float32)
        z[:, :40] = 100
        np.testing.assert_array_equal(coastal_scarp_transport(z, np.zeros(180)), z)
