"""Physical contours must follow the queried ground, including tiny peaks."""
import unittest

import numpy as np
from scipy.optimize import brentq

from world_atlas.core.cartographic_relief import physical_relief_paths
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.terrain_refinement import RefinedTerrainField


def base_field(values):
    return PhysicalTerrainField(values, land_mask=values > 0, sea_level_m=0,
                                elevation_scale_m=4000, elevation_exponent=1)


class PhysicalReliefTests(unittest.TestCase):
    def test_barely_superlevel_peak_keeps_its_true_curved_footprint(self):
        values = np.full((7, 7), .1)
        values[3, 3] = 2.0001
        field = base_field(values)
        palette = float(field.palette_elevation(2.0))
        level = float(field.contour_height_m(palette))
        bands, lines, levels = physical_relief_paths(field, [palette])
        self.assertTrue(bands[0])
        self.assertEqual(len(lines), 1)
        self.assertEqual(levels, [palette])
        points = lines[0]
        self.assertGreater(len(points), 16)
        residual = field.sample_points(points[:, 0], points[:, 1]) - level
        self.assertLess(float(np.max(abs(residual))), 2e-11)
        radius = brentq(lambda x: float(field.sample_points(x, 3.5)) - level,
                        3.5, 4.5, xtol=1e-14) - 3.5
        self.assertAlmostEqual(float(points[:, 0].max()) - 3.5, radius, delta=2e-10)
        self.assertAlmostEqual(3.5 - float(points[:, 0].min()), radius, delta=2e-10)

    def test_refined_contours_query_actual_ground_instead_of_native_reinterpolation(self):
        values = np.full((9, 9), 100.)
        values[3:6, 3:6] = np.array(((700., 1100., 800.),
                                            (1200., 1900., 900.),
                                            (600., 800., 500.)))
        base = base_field(values)
        flow = np.full(values.shape, -1, dtype=np.int32)
        flow[4, 4], flow[4, 5] = 4*9+5, 5*9+5
        discharge = np.zeros(values.shape)
        discharge[4, 4], discharge[4, 5] = 64., 128.
        segments = np.array((((4.5,4.5),(5.5,4.5)),((5.5,4.5),(5.5,5.5))))
        field = RefinedTerrainField(base, seed=1, radius_km=12,
            flow_to=flow, discharge=discharge, river_segments=segments,
            river_anchors=np.empty((0,2)))
        palette = float(field.palette_elevation(1000.))
        level = float(field.contour_height_m(palette))
        _, lines, _ = physical_relief_paths(field, [palette])
        self.assertTrue(lines)
        points = np.concatenate(lines)
        actual = field.sample_points(points[:,0], points[:,1])
        unrefined = base.sample_points(points[:,0], points[:,1])
        self.assertGreater(float(np.max(abs(actual-unrefined))), 1.,
                           'this fixture must distinguish the actual refined model')
        self.assertLess(float(np.max(abs(actual-level))), 2e-8)


if __name__ == '__main__':
    unittest.main()
