from types import SimpleNamespace
import unittest

from world_atlas.core.city_transport import CityCorridor, CityTransportIndex, refined_land_routes
from world_atlas.core.society.model import TransportRoute


def route(identifier, mode, points):
    return TransportRoute(identifier, mode, "regional", "city", "neighbor", tuple(points))


def transport_index(routes, width):
    return CityTransportIndex(SimpleNamespace(routes=routes, bridges=()), width,
        corridors=(CityCorridor(item.identifier, item.mode, item.path) for item in routes))


class CityTransportTests(unittest.TestCase):
    def test_subcell_access_preserves_the_entire_canonical_route(self):
        roads = (route("land", "road", [(1.5, 3.5), (2.5, 4.5), (3.5, 4.5)]),
                 route("boat", "sea", [(1.5, 3.5), (2.5, 4.5)]))
        result = refined_land_routes(roads, {"city": (3.5, 1.1), "neighbor": (4.1, 3.5)})
        self.assertEqual(result[0].path[1:-1], roads[0].path)
        self.assertEqual(result[0].path[0], (1.1, 3.5))
        self.assertEqual(result[0].path[-1], (3.5, 4.1))
        self.assertEqual(result[1], roads[1])

    def test_nearby_through_route_is_included_without_false_endpoint_connection(self):
        roads = (route("through", "road", [(0, 5), (12, 5)]),
                 route("far", "road", [(0, 20), (12, 20)]),
                 route("shipping", "sea", [(0, 5), (12, 5)]))
        index = transport_index(roads, 100)
        corridors = index.corridors((4, 4, 6, 6))
        self.assertEqual([item.identifier for item in corridors], ["through"])
        self.assertEqual(index.connections["unconnected"]["road"], 0)
        self.assertAlmostEqual(corridors[0].points[0][0], 3.6)
        self.assertAlmostEqual(corridors[0].points[-1][0], 6.4)

    def test_longitude_seam_uses_short_local_corridor(self):
        index = transport_index((route("wrapped", "rail", [(98, 5), (1, 5)]),), 100)
        corridors = index.corridors((4, -2, 6, 2))
        self.assertEqual(len(corridors), 1)
        self.assertEqual(corridors[0].points, ((-2.0, 5.0), (1.0, 5.0)))
        self.assertEqual(index.corridors((4, 40, 6, 60)), ())

    def test_reentry_is_split_without_a_shortcut(self):
        index = transport_index((route("loop", "road", [(1, 4), (8, 4), (8, 6), (1, 6)]),), 100)
        corridors = index.corridors((3, 2, 7, 4))
        self.assertEqual(len(corridors), 2)
        self.assertTrue(all(len(item.points) == 2 for item in corridors))

    def test_final_corridors_retain_logical_connections_without_inventing_endpoints(self):
        index = CityTransportIndex(SimpleNamespace(
            routes=(route("logical", "road", [(0, 20), (12, 20)]),), bridges=()), 100,
            corridors=(CityCorridor("built", "road", ((0, 5), (12, 5))),))
        self.assertEqual(index.connections["city"]["road"], 1)
        self.assertEqual([item.identifier for item in index.corridors((4, 4, 6, 6))], ["built"])
        self.assertEqual(index.corridors((19, 4, 21, 6)), ())
