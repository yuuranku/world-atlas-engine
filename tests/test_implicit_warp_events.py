"""True source switch curves retain the accepted inverse W and its periods."""

import unittest

import numpy as np

from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.implicit_warp_events import warped_pchip_events
from world_atlas.core.terrain_refinement import RefinedTerrainField


def terrain(native):
    base = PhysicalTerrainField(native, land_mask=native > 0, sea_level_m=0,
                                elevation_scale_m=4000, elevation_exponent=1)
    return RefinedTerrainField(base, seed=2, radius_km=6371,
        flow_to=np.full(native.shape, -1, dtype=int), discharge=np.zeros(native.shape),
        river_segments=np.empty((0, 2, 2)), river_anchors=np.empty((0, 2)))


class ImplicitWarpEventTests(unittest.TestCase):
    def test_actual_source_switches_are_transported_by_the_original_inverse(self):
        native = np.random.default_rng(89323).uniform(-300, 2000, (9, 11))
        field = terrain(native)
        events = warped_pchip_events(field, [[0., 0.]], [[11., 9.]])
        self.assertGreater(len(events), 0)
        source = np.array([(event["sourceX"], (event["sourceYLower"]+event["sourceYUpper"])/2)
                           for event in events])
        x, y = field.inverse_ground_coordinates(source[:, 0], source[:, 1])
        physical = np.column_stack((x, y))
        self.assertGreater(float(np.max(np.linalg.norm(physical-source, axis=1))), .001)
        wx, wy = field.forward_ground_coordinates(x, y)
        np.testing.assert_allclose(np.column_stack((wx, wy)), source, rtol=0, atol=1e-12)
        for event, point in zip(events, physical, strict=True):
            self.assertLessEqual(event["sourceYUpper"]-event["sourceYLower"], 1)
            patch = event["patchIndex"]
            self.assertLess(np.linalg.norm(point-field._landforms.centres[patch]), field._landforms.radius[patch])
            row = int(np.floor((event["sourceYLower"]+event["sourceYUpper"])/2-.5))
            rows = np.clip(np.arange(row-1, row+3), 0, field.height-1)+.5
            values = field.base.sample_points(np.full(4, event["sourceX"]), rows)
            self.assertLess(float(np.min(abs(np.diff(values)))), 1e-10)
        np.testing.assert_array_equal(field.native_m, native)

    def test_periodic_candidates_share_primary_roots_without_subtracting_a_width(self):
        field = terrain(np.random.default_rng(89323).uniform(-300, 2000, (9, 11)))
        events = warped_pchip_events(field, [[-11., 0.], [0., 0.], [11., 0.]],
                                     [[0., 9.], [11., 9.], [22., 9.]])
        owners = [{(event["patchIndex"] % field._landforms.count, event["canonicalSourceX"])
                   for event in events if event["owner"] == owner} for owner in range(3)]
        self.assertGreater(len(owners[0]), 0)
        self.assertEqual(owners[0], owners[1])
        self.assertEqual(owners[1], owners[2])
        for event in events:
            period = (event["owner"]-1)*field.width
            self.assertEqual(event["sourceX"], event["canonicalSourceX"]+period)

    def test_empty_support_and_disjoint_query_do_not_invent_warped_events(self):
        field = terrain(np.ones((9, 11)))
        self.assertEqual(warped_pchip_events(field, [[0., 0.]], [[11., 9.]]), [])
        field = terrain(np.random.default_rng(89323).uniform(-300, 2000, (9, 11)))
        self.assertEqual(warped_pchip_events(field, [[0., 0.]], [[.01, .01]]), [])
        self.assertEqual(warped_pchip_events(field, np.empty((0, 2)), np.empty((0, 2))), [])


if __name__ == "__main__":
    unittest.main()
