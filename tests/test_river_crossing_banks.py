"""Source crossing stations connect the route's actual native bank faces."""

import unittest

import numpy as np
import shapely

from world_atlas.core.river_crossings import (
    endpoint_bank_direction, river_bank_owner, source_crossing_points,
    source_crossing_transitions,
)


def station_coordinates(road, river):
    return [tuple(point.coords[0]) for point in source_crossing_points(road, river)]


class RiverCrossingBankTests(unittest.TestCase):
    def setUp(self):
        # Translated native v94 road0041 confluence, not a screen-space curve.
        self.river = shapely.MultiLineString((
            ((-2.5, 1.5), (-2, 1), (-1, 1), (0, 0)),
            ((0, 3.5), (0, 3), (1, 2), (1, 1), (0, 0)),
            ((0, 0), (1, -1)),
        ))

    def test_confluence_transition_banks_reverse_without_changing_station(self):
        road = shapely.LineString(((3, 3), (1, 1), (0, 0), (0, -1)))
        forward = source_crossing_transitions(road, self.river)
        reverse = source_crossing_transitions(shapely.reverse(road), self.river)
        self.assertEqual([point.coords[0] for point, _, _ in forward], [(0., 0.)])
        self.assertEqual([point.coords[0] for point, _, _ in reverse], [(0., 0.)])
        point, incoming, outgoing = forward[0]
        self.assertTrue(np.allclose(incoming, reverse[0][2]))
        self.assertTrue(np.allclose(outgoing, reverse[0][1]))
        samples = [shapely.Point(np.asarray(point.coords[0]) + np.asarray(direction)*.001)
                   for direction in (incoming, outgoing)]
        owners = river_bank_owner(self.river, point, samples)
        self.assertNotIn(None, owners)
        self.assertNotEqual(owners[0], owners[1])
        # The incoming road follows a native flow edge at the station.
        # Its valid incoming bank therefore cannot come from that tangent.
        self.assertGreater(abs(incoming[1]-incoming[0]), .1)

    def test_transverse_transition_records_both_original_approach_banks(self):
        river = shapely.LineString(((0, -2), (0, 2)))
        road = shapely.LineString(((-1, 0), (1, 0)))
        point, incoming, outgoing = source_crossing_transitions(road, river)[0]
        self.assertTrue(np.allclose(incoming, (-1., 0.)))
        self.assertTrue(np.allclose(outgoing, (1., 0.)))

    def test_endpoint_overlap_uses_the_first_proven_off_channel_bank(self):
        river = shapely.LineString(((-1, 0), (3, 0)))
        road = shapely.LineString(((0, 0), (2, 0), (2, 1)))
        point = shapely.Point(0, 0)
        direction = endpoint_bank_direction(road, river, point)
        self.assertTrue(np.allclose(direction, (0., 1.)))
        self.assertTrue(np.allclose(direction,
            endpoint_bank_direction(shapely.reverse(road), river, point)))
        self.assertIsNone(endpoint_bank_direction(
            shapely.LineString(((0, 0), (2, 0))), river, point))

    def test_endpoint_overlap_cannot_skip_a_branch_between_approach_and_city(self):
        river = shapely.MultiLineString((((-1, 0), (3, 0)), ((1, 0), (1, 1))))
        road = shapely.LineString(((0, 0), (2, 0), (2, 1)))
        with self.assertRaisesRegex(ValueError, 'incident fan sector'):
            endpoint_bank_direction(road, river, shapely.Point(0, 0))

    def test_actual_confluence_exit_connects_third_bank(self):
        road = shapely.LineString(((3, 3), (1, 1), (0, 0), (0, -1)))
        self.assertEqual(station_coordinates(road, self.river), [(0., 0.)])
        self.assertEqual(station_coordinates(shapely.reverse(road), self.river), [(0., 0.)])

    def test_another_bank_pair_requires_confluence_entry(self):
        road = shapely.LineString(((-1, 0), (0, 0), (1, 1), (2, 1)))
        self.assertEqual(station_coordinates(road, self.river), [(0., 0.)])
        self.assertEqual(station_coordinates(shapely.reverse(road), self.river), [(0., 0.)])

    def test_two_branch_barriers_require_two_facilities_in_one_overlap(self):
        river = shapely.MultiLineString((
            ((-1, 0), (1, 0), (2, 0), (4, 0)),
            ((1, 2), (1, 0)), ((2, -2), (2, 0)),
        ))
        road = shapely.LineString(((0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (3, -1)))
        self.assertEqual(station_coordinates(road, river), [(0., 0.), (2., 0.)])
        self.assertEqual(station_coordinates(shapely.reverse(road), river), [(2., 0.), (0., 0.)])

    def test_staying_on_the_same_bank_needs_no_facility(self):
        river = shapely.LineString(((-1, 0), (3, 0)))
        road = shapely.LineString(((0, 1), (0, 0), (2, 0), (2, 1)))
        self.assertEqual(station_coordinates(road, river), [])

    def test_transverse_intersection_and_tangent_visit_have_different_banks(self):
        river = shapely.LineString(((0, -2), (0, 2)))
        transverse = shapely.LineString(((-1, 0), (1, 0)))
        tangent = shapely.LineString(((-1, 1), (0, 0), (-1, -1)))
        self.assertEqual(station_coordinates(transverse, river), [(0., 0.)])
        self.assertEqual(station_coordinates(tangent, river), [])

    def test_native_city_endpoint_remains_endpoint_star_responsibility(self):
        river = shapely.LineString(((0, -2), (0, 2)))
        road = shapely.LineString(((-1, 0), (0, 0)))
        self.assertEqual(station_coordinates(road, river), [])

    def test_classification_uses_one_noding_operation_at_world_coordinates(self):
        road = shapely.LineString(((1141.5, 381.5), (1139.5, 379.5),
                                   (1138.5, 378.5), (1138.5, 377.5)))
        river = shapely.MultiLineString((
            ((1136., 380.), (1136.5, 379.5), (1137.5, 379.5), (1138.5, 378.5)),
            ((1138.5, 382.), (1138.5, 381.5), (1139.5, 380.5),
             (1139.5, 379.5), (1138.5, 378.5)),
            ((1138.5, 378.5), (1139.5, 377.5)),
        ))
        self.assertEqual(station_coordinates(road, river), [(1138.5, 378.5)])


if __name__ == '__main__':
    unittest.main()
