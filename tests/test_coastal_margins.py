"""Physical-coordinate checks for genetically distinct coastal footprints."""
import importlib
import importlib.util
import unittest

import numpy as np


class CoastalMarginTests(unittest.TestCase):
    def test_short_high_slip_sectors_do_not_fold_surface(self):
        from world_atlas.core.coastal_margins import margin_motion
        coordinate = np.linspace(-650, 650, 301)
        along, inland = np.meshgrid(coordinate, coordinate)
        for family in ('rift', 'active', 'passive', 'transform'):
            for seed in (12, 817, 819):
                normal, tangent, _ = margin_motion(
                    along, inland, 180, 210, 520, seed=seed, family=family)
                dn_y, dn_x = np.gradient(normal, coordinate, coordinate)
                dt_y, dt_x = np.gradient(tangent, coordinate, coordinate)
                determinant = (1 + dt_x) * (1 + dn_y) - dt_y * dn_x
                self.assertGreater(float(determinant.min()), 0.0)

    def test_motion_budget_does_not_collapse_plate_speeds_to_one_slip(self):
        from world_atlas.core import coastal_margins
        self.assertTrue(hasattr(coastal_margins, 'motion_budget_km'))
        speeds = np.array([0.,1.,3.,6.,9.,12.,16.8])
        budget = coastal_margins.motion_budget_km(speeds, 3.)
        self.assertEqual(budget[0], 0)
        self.assertTrue(np.all(np.diff(budget)>0))
        self.assertGreater(budget[-1]-budget[3],100)
        self.assertLess(budget[-1],280)
        self.assertLess(coastal_margins.motion_budget_km(9.,1.), budget[4])

    def test_fault_displacement_changes_macro_direction_without_sine_arc(self):
        from world_atlas.core import coastal_margins
        self.assertTrue(hasattr(coastal_margins, 'margin_motion'),
                        'Macro coast needs structural displacement, not inlet stamps')
        along = np.linspace(-2400,2400,481)
        normal, tangent, support = coastal_margins.margin_motion(along, np.zeros_like(along), 2100, 1500, 520, seed=817, family='rift')
        # Absolute displacement used to be rewarded even when adjacent
        # sectors folded. Preserve alternating motion and bound its strain.
        self.assertGreater(float(normal.max()), 0)
        self.assertLess(float(normal.min()), 0)
        self.assertLess(float(np.max(np.abs(np.gradient(normal, along)))), 0.45)
        self.assertEqual(normal[0], 0)
        self.assertEqual(normal[-1], 0)
        interior, _, _ = coastal_margins.margin_motion(along, np.full_like(along,2000), 2100, 1500, 520, seed=817, family='rift')
        np.testing.assert_array_equal(interior,0)

    def test_motion_is_seeded_and_independent_of_sampling_resolution(self):
        from world_atlas.core.coastal_margins import margin_motion
        across, along = np.meshgrid(np.linspace(-2000,2000,129),np.linspace(-2800,2800,129))
        for family in ('rift','active','passive','transform'):
            result = margin_motion(along,across,2100,1600,480,seed=819,family=family)
            coarse = margin_motion(along[::2,::2],across[::2,::2],2100,1600,480,seed=819,family=family)
            repeated = margin_motion(along,across,2100,1600,480,seed=819,family=family)
            other = margin_motion(along,across,2100,1600,480,seed=820,family=family)
            for i in range(3):
                np.testing.assert_array_equal(result[i],repeated[i])
                np.testing.assert_array_equal(result[i][::2,::2],coarse[i])
                self.assertTrue(np.isfinite(result[i]).all())
            self.assertGreater(np.count_nonzero(result[0] != other[0]),100)
            self.assertEqual(np.count_nonzero(result[0][np.abs(across)>1600]),0)

    def test_evolution_preserves_interior_and_deep_ocean(self):
        from world_atlas.core import procedural_planet as planet
        self.assertTrue(hasattr(planet, '_evolve_coastal_margins'),
                        'Coast repair must be bounded to the final coastal surface')
        rows, columns = np.indices((360, 720))
        relative = np.broadcast_to((columns - 360) * 45., (360, 720)).astype(np.float32)
        quiet = np.zeros_like(relative)
        boundary = np.zeros(relative.shape, dtype=np.int8)
        velocity = np.ones_like(relative)
        delta, metrics = planet._evolve_coastal_margins(relative, quiet, boundary, velocity, velocity, 819, .28)
        # Structural motion carries a coastal cliff too. Protect by geographic
        # distance, not altitude. This periodic fixture has shores at 0 and 360.
        latitude = np.pi/2-(rows+.5)*np.pi/360
        distance_from_shore_km = 6400*np.arcsin(np.abs(np.sin(columns*np.pi/360))*np.cos(latitude))
        protected = distance_from_shore_km > 3000
        np.testing.assert_array_equal(delta[protected], 0)
        self.assertGreater(np.count_nonzero((relative + delta > 0) != (relative > 0)), 10)
        self.assertGreater(metrics['modifiedShoreSegments'], 0)
        self.assertLessEqual(metrics['maximumDisplacementGradient'], 0.650001)
        repeated, _ = planet._evolve_coastal_margins(relative, quiet, boundary, velocity, velocity, 819, .28)
        np.testing.assert_array_equal(delta, repeated)

    def test_directional_orogen_is_seeded_and_has_parallel_variation(self):
        from world_atlas.core import procedural_planet as planet
        from world_atlas.physical.planetary_grid import build_lat_lon_grid

        grid = build_lat_lon_grid(36, 72)
        sources = np.zeros(grid.shape, dtype=bool)
        sources[17:19, 12:60] = True
        affinity = np.ones(grid.shape, dtype=np.float64)
        first = planet._directional_orogenic_belt(
            grid, sources, affinity, seed=819, width_km=260.0,
            amplitude_m=2400.0,
        )
        repeated = planet._directional_orogenic_belt(
            grid, sources, affinity, seed=819, width_km=260.0,
            amplitude_m=2400.0,
        )
        other = planet._directional_orogenic_belt(
            grid, sources, affinity, seed=820, width_km=260.0,
            amplitude_m=2400.0,
        )
        np.testing.assert_array_equal(first, repeated)
        self.assertGreater(float(np.max(first)), 100.0)
        self.assertGreater(float(np.ptp(first[14:23, 12:60])), 200.0)
        self.assertGreater(int(np.count_nonzero(first != other)), 100)


if __name__ == '__main__':
    unittest.main()
