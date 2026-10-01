"""Native nearest-river corners remain true terrain model sections."""

from types import SimpleNamespace
import unittest

import numpy as np
import shapely

from world_atlas.core.implicit_river_events import (
    interior_medial_is_nearest, interior_river_medials)
from world_atlas.core.terrain_refinement import RefinedTerrainField


def source_field(segments, *, width=2176):
    segments = np.asarray(segments, dtype=float)
    segments[:, 0, 0] %= width
    segments[:, 1, 0] = segments[:, 0, 0] + ((segments[:, 1, 0]-segments[:, 0, 0]
                                           + width/2) % width-width/2)
    periodic = np.concatenate((segments-(width,0), segments, segments+(width,0)))
    return SimpleNamespace(_rivers=shapely.STRtree(shapely.linestrings(periodic)), width=width,
                           _river_support_distance=np.zeros((1088, width)))


class ImplicitRiverEventTests(unittest.TestCase):
    def test_actual_parallel_d8_medial_is_a_source_protection_corner(self):
        field = source_field(np.array((((1835.5, 582.5), (1836.5, 583.5)),
                                       ((1835.5, 583.5), (1836.5, 584.5)))))
        events = interior_river_medials(field, [[1835.5, 582.5]], [[1836.5, 584.5]])
        self.assertEqual(len(events), 1)
        event = events[0]
        np.testing.assert_array_equal(event["normal"], (1., -1.))
        self.assertEqual(event["offset"], 1252.5)
        self.assertEqual(event["sourceRiverIndices"], (2, 3))
        # Exact represented station on the same actual v97 switching line.
        point = np.array((1836., 583.5))
        self.assertTrue(interior_medial_is_nearest(field, event, point))
        along_normal = np.array((1., -1.))*1e-4
        points = np.stack((point-along_normal, point, point+along_normal))
        weights = RefinedTerrainField._river_weight(field, points[:, 0], points[:, 1])
        self.assertGreater(weights[1], weights[0])
        self.assertGreater(weights[1], weights[2])
        self.assertAlmostEqual(weights[0], weights[2], places=10)

    def test_projection_endpoints_and_shadowed_pairs_do_not_license_events(self):
        segments = np.array((((.5, .5), (1.5, 1.5)),
                             ((.5, 1.5), (1.5, 2.5)),
                             ((.5, 1.), (1.5, 2.))))
        field = source_field(segments)
        events = interior_river_medials(field, [[.5, .5]], [[1.5, 2.5]])
        outer = next(event for event in events if event["sourceRiverIndices"] == (3, 4))
        self.assertFalse(interior_medial_is_nearest(field, outer, (1., 1.5)))
        self.assertFalse(interior_medial_is_nearest(field, outer, (.75, 1.25)))
        self.assertEqual(interior_river_medials(field, [[1.6, 2.1]], [[1.9, 2.4]]), [])

    def test_constant_profile_and_coincident_sources_have_no_switch(self):
        cases = (
            (((.5, .5), (1.5, .5)), ((.5, 1.5), (1.5, 1.5))),
            (((.5, .5), (1.5, 1.5)), ((1.5, 1.5), (.5, .5))),
        )
        for segments in cases:
            with self.subTest(segments=segments):
                self.assertEqual(interior_river_medials(source_field(segments), [[0., 0.]], [[2., 2.]]), [])

    def test_actual_nonparallel_medial_keeps_the_native_protection_corner(self):
        segments = np.array((((1037.5, 652.5), (1036.5, 652.5)),
                             ((1036.5, 652.5), (1037.5, 653.5))))
        field = source_field(segments)
        events = interior_river_medials(field, [[1037., 652.5]], [[1037.5, 653.]])
        self.assertEqual(len(events), 1)
        event = events[0]
        expected = np.array((1037.249808600642, 652.810580891569))
        normal = event['normal']
        solved = int(np.argmax(abs(normal)))
        free = 1-solved
        point = expected.copy()
        point[solved] = (event['offset']-normal[free]*point[free])/normal[solved]
        self.assertLess(float(np.linalg.norm(point-expected)), 1e-10)
        self.assertTrue(interior_medial_is_nearest(field, event, point))
        nearest = field._rivers.query_nearest(shapely.Point(point), all_matches=True)
        self.assertTrue(set(nearest).issubset({2, 3}))
        displacement = normal / np.linalg.norm(normal) * 1e-4
        stations = np.stack((point-displacement, point, point+displacement))
        weights = RefinedTerrainField._river_weight(field, stations[:, 0], stations[:, 1])
        self.assertGreater(weights[1], weights[0])
        self.assertGreater(weights[1], weights[2])
        # Moving one represented coordinate loses the exact source event.
        moved = point.copy()
        moved[solved] = np.nextafter(moved[solved], np.inf)
        self.assertFalse(interior_medial_is_nearest(field, event, moved))

    def test_two_angle_bisectors_retain_distinct_interior_source_sectors(self):
        field = source_field(np.array((((-1., 0.), (1., 0.)),
                                       ((0., -1.), (0., 1.)))))
        events = interior_river_medials(field, [[-.6, -.6]], [[.6, .6]])
        self.assertEqual(len(events), 4)
        stations = ((.3, .3), (-.3, -.3), (.3, -.3), (-.3, .3))
        for station in stations:
            self.assertEqual(sum(interior_medial_is_nearest(field, event, station) for event in events), 1)
        # The same native junction inside the flat protection core is smooth.
        self.assertEqual(interior_river_medials(field, [[-.17, -.17]], [[.17, .17]]), [])

    def test_periodic_source_copies_keep_their_actual_unwrapped_identity(self):
        segments = np.array((((.5, .5), (1.5, 1.5)), ((.5, 1.5), (1.5, 2.5))))
        field = source_field(segments, width=10)
        events = interior_river_medials(field, [[.5, .5], [10.5, .5]], [[1.5, 2.5], [11.5, 2.5]])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["sourceRiverIndices"], (2, 3))
        self.assertEqual(events[1]["sourceRiverIndices"], (4, 5))
        self.assertEqual(events[0]["owner"], 0)
        self.assertEqual(events[1]["owner"], 1)
        self.assertEqual(events[0]["offset"]+10, events[1]["offset"])
        self.assertTrue(interior_medial_is_nearest(field, events[1], (11., 1.5)))
        self.assertEqual(events[0]['canonicalLineKey'], events[1]['canonicalLineKey'])
        self.assertEqual(events[0]['canonicalOffset'], events[1]['canonicalOffset'])
        self.assertEqual((events[0]['longitudePeriod'], events[1]['longitudePeriod']), (0, 1))

    def test_irrational_bisectors_keep_primary_identity_without_offset_modulo(self):
        width = 2176
        field = source_field(np.array((((512.5, 81.5), (513.5, 81.5)),
                                       ((512.5, 81.5), (513.5, 82.5)))), width=width)
        events = interior_river_medials(field, [[512.5, 81.5], [width+512.5, 81.5]],
                                       [[513.5, 82.5], [width+513.5, 82.5]])
        self.assertEqual(len(events), 2)
        first, last = events
        np.testing.assert_array_equal(first['normal'], last['normal'])
        self.assertEqual(first['canonicalLineKey'], last['canonicalLineKey'])
        self.assertEqual(first['canonicalOffset'], last['canonicalOffset'])
        self.assertEqual((first['longitudePeriod'], last['longitudePeriod']), (0, 1))
        self.assertEqual(last['offset'], first['canonicalOffset']+first['normal'][0]*width)
        # This actual pair demonstrates why modulo cannot recover identity.
        self.assertNotEqual(first['offset'] % (first['normal'][0]*width),
                            last['offset'] % (last['normal'][0]*width))

    def test_cross_meridian_pairs_use_primary_source_order(self):
        width = 2176
        field = source_field(np.array((((.5, .5), (1.5, 1.5)),
                                       ((width-.5, .5), (width+.5, .5)))), width=width)
        events = interior_river_medials(field, [[-.25, .6], [width-.25, .6]],
                                       [[.5, 1.], [width+.5, 1.]])
        self.assertEqual(len(events), 2)
        first, last = events
        self.assertEqual(first['sourceRiverIndices'], (1, 2))
        self.assertEqual(last['sourceRiverIndices'], (3, 4))
        self.assertEqual(first['canonicalLineKey'], last['canonicalLineKey'])
        self.assertEqual(first['canonicalLineKey'][:3], (0, 1, -1))
        self.assertEqual(first['canonicalOffset'], last['canonicalOffset'])
        self.assertEqual((first['longitudePeriod'], last['longitudePeriod']), (0, 1))


if __name__ == "__main__":
    unittest.main()
