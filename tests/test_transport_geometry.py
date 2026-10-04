"""Display network topology and tangent corners share the visible coastline."""

from types import SimpleNamespace
import json
from pathlib import Path
import unittest

import numpy as np
import shapely

from world_atlas.core.transport_geometry import (
    _anchor_parts, _bank_boundary_roundoff, _bank_route, _curved_transport_points, _road_native_corridor, river_navigation_attributes, river_navigation_segments, shared_transport_paths,
)


def route(identifier, mode, points, importance="regional"):
    return SimpleNamespace(identifier=identifier, mode=mode, importance=importance, path=tuple(points))


class TransportGeometryTests(unittest.TestCase):
    def setUp(self):
        self.grid = SimpleNamespace(shape=(32, 64))

    def test_bridge_connects_to_overlay_bank_without_adding_a_water_crossing(self):
        record=json.loads((Path(__file__).parent/'fixtures/bridge-overlay-bank-dev12.json').read_text())
        corridor=shapely.from_wkb(bytes.fromhex(record['corridorWkb']))
        channel=shapely.from_wkb(bytes.fromhex(record['channelWkb']))
        points=np.asarray(record['points'])
        portal=shapely.Point(record['coordinates'])
        dry=corridor.difference(channel)
        # This actual channel bank is outside the dry overlay by 1000 ulps.
        self.assertFalse(channel.covers(portal))
        self.assertGreater(dry.distance(portal),16*np.spacing(portal.x))
        path=_bank_route(points,channel,record['passages'],corridor)
        line=shapely.LineString(path)
        self.assertTrue(corridor.covers(line))
        np.testing.assert_array_equal(path[[0,-1]],points[[0,-1]])
        self.assertIn(tuple(record['centre']),tuple(map(tuple,path)))
        decks=shapely.union_all([shapely.LineString(deck)for _station,deck in record['passages']])
        water=line.intersection(channel).difference(channel.boundary)
        unlicensed=water.difference(decks)
        certificates=_bank_boundary_roundoff(unlicensed,channel)
        from world_atlas.core.transport_geometry import _uncertified_water
        self.assertTrue(_uncertified_water(unlicensed,certificates).is_empty)

    def test_bridge_portal_extension_cannot_cross_unlicensed_water(self):
        channel=shapely.box(4.,0.,6.,10.)
        points=np.asarray(((2.,5.),(8.,5.)))
        # These claimed bank endpoints are physically inside the river.
        with self.assertRaisesRegex(ValueError,'crosses physical water'):
            _bank_route(points,channel,((3.,((4.1,5.),(5.,5.),(5.9,5.))),),
                        shapely.box(0.,0.,10.,10.))

    def test_coastal_access_admits_actual_dry_ground_in_a_coarse_water_cell(self):
        water=np.zeros((3,3),dtype=np.uint8)
        water[:,2]=1
        grid=SimpleNamespace(shape=water.shape,water=water)
        points=np.array(((.5,1.5),(1.5,1.5),(2.6,1.5)))
        access=(((1.5,1.5),(2.6,1.5)),)
        corridor=_road_native_corridor(grid,points,shapely.box(0,0,2.8,3),endpoint_access=access)
        self.assertTrue(corridor.covers(shapely.LineString(points)))
        self.assertFalse(corridor.covers(shapely.Point(2.9,1.5)))
        limited=_road_native_corridor(grid,points,shapely.box(0,0,2.4,3),endpoint_access=access)
        self.assertFalse(limited.covers(shapely.Point(2.6,1.5)))

    def test_diagonal_coastal_access_has_a_connected_corridor_at_grid_corners(self):
        water=np.zeros((3,4),dtype=np.uint8)
        water[:,2:]=1
        grid=SimpleNamespace(shape=water.shape,water=water)
        points=np.array(((.5,.5),(1.5,.5),(3.3,2.3)))
        corridor=_road_native_corridor(grid,points,shapely.box(0,0,3.8,3),
            endpoint_access=(((1.5,.5),(3.3,2.3)),))
        self.assertEqual(corridor.geom_type,'Polygon')
        self.assertTrue(corridor.covers(shapely.LineString(points)))

    def test_bank_roundoff_certifies_only_same_edge_binary64_limits(self):
        channel=shapely.box(0.,0.,1.,1.)
        bank_x=float(np.nextafter(1.,0.))
        limit=shapely.LineString(((bank_x,.2),(bank_x,.8)))
        records=_bank_boundary_roundoff(limit,channel)
        self.assertEqual(len(records),1)
        self.assertLessEqual(records[0]["maxDistance"],records[0]["ulpBound"])
        actual_cut=shapely.LineString(((.98,.2),(.98,.8)))
        self.assertEqual(_bank_boundary_roundoff(actual_cut,channel),())
        # Both endpoints on banks still do not license an interior shortcut.
        chord=shapely.LineString(((0.,.2),(1.,.8)))
        self.assertEqual(_bank_boundary_roundoff(chord,channel),())

    def test_anchor_split_retains_a_distinct_bank_corner_and_its_short_edge(self):
        coordinates=((0.,0.),(1.,0.),(1.,2e-11),(2.,2e-11))
        line=shapely.LineString(coordinates)
        anchors=shapely.STRtree([shapely.Point(point)for point in coordinates[1:3]])
        parts=_anchor_parts(line,anchors)
        network=shapely.union_all([shapely.LineString(part)for part in parts])
        self.assertEqual(network.difference(line).length,0.)
        self.assertEqual(line.difference(network).length,0.)
        self.assertTrue(any(np.array_equal(part,np.asarray(coordinates[1:3]))for part in parts))

    def test_each_native_bank_transition_uses_both_physical_deck_portals(self):
        # Centre-only navigation would visit both facilities from the west
        # bank and leave the east-bank portion of the native route unused.
        channel=shapely.box(4.9,0.,5.1,10.)
        points=np.asarray(((2.,4.),(5.,4.),(8.,4.),(8.,6.),(5.,6.),(2.,6.)))
        passages=((3.,((4.9,4.),(5.,4.),(5.1,4.))),
                  (11.,((5.1,6.),(5.,6.),(4.9,6.))))
        path=_bank_route(points,channel,passages,shapely.box(0.,0.,10.,10.))
        water=shapely.LineString(path).intersection(channel).difference(channel.boundary)
        self.assertAlmostEqual(water.length,.4)
        for _station,deck in passages:
            self.assertTrue(shapely.LineString(path).covers(shapely.LineString(deck)))
        np.testing.assert_array_equal(path[[0,-1]],points[[0,-1]])

    def test_bank_repair_retains_terrain_selected_dry_turns_before_and_after_bridge(self):
        channel=shapely.box(4.9,0.,5.1,10.)
        points=np.asarray(((1.,1.),(1.,4.),(5.,4.),(8.,4.),(8.,7.)))
        source=shapely.LineString(points)
        crossing=source.project(shapely.Point(5.,4.))
        deck=((4.9,4.),(5.,4.),(5.1,4.))
        path=_bank_route(points,channel,((crossing,deck),),shapely.box(0.,0.,10.,10.))
        self.assertIn((1.,4.),tuple(map(tuple,path)))
        self.assertIn((8.,4.),tuple(map(tuple,path)))
        line=shapely.LineString(path)
        self.assertTrue(line.covers(shapely.LineString(deck)))
        water=line.intersection(channel).difference(channel.boundary)
        self.assertAlmostEqual(water.length,.2)
        np.testing.assert_array_equal(path[[0,-1]],points[[0,-1]])

    def test_navigability_is_a_single_physical_reach_attribute_with_all_routes_retained(self):
        water = np.zeros((8, 16), dtype=np.uint8)
        order = np.zeros_like(water)
        order[3,2:9] = 3
        grid = SimpleNamespace(shape=water.shape, water=water, river_order=order)
        physical = np.array([(x+.5,3.5) for x in range(2,9)])
        routes = (route("one","river",((2.5,3.5),(8.5,3.5))),
                  route("two","river",((2.5,3.5),(8.5,3.5))))
        before = tuple(item.path for item in routes)
        attrs = river_navigation_attributes(grid,[physical],routes)[0]
        self.assertEqual(attrs["navigation-route-ids"],"one,two")
        self.assertEqual(attrs["navigation-cell-count"],"7")
        self.assertEqual(attrs["navigable"],"true")
        self.assertEqual(tuple(item.path for item in routes),before)

    def test_partial_navigation_cannot_designate_the_rest_of_a_reach(self):
        water = np.zeros((8, 16), dtype=np.uint8)
        order = np.zeros_like(water)
        order[3,2:9] = 3
        grid = SimpleNamespace(shape=water.shape, water=water, river_order=order)
        physical = np.array([(x+.5,3.5) for x in range(2,9)])
        attrs = river_navigation_attributes(grid,[physical],
                    (route("short","river",((2.5,3.5),(4.5,3.5))),))[0]
        self.assertEqual(attrs["navigable"],"false")
        self.assertEqual(attrs["navigation-cell-count"],"3")
        self.assertEqual(attrs["navigation-support-cell-count"],"7")
        self.assertEqual(river_navigation_attributes(grid,[physical],
                    (route("road","road",((2.5,3.5),(8.5,3.5))),)),[{}])

    def test_periodic_navigation_never_adds_support_across_the_world_interior(self):
        water = np.zeros((8, 16), dtype=np.uint8)
        order = np.zeros_like(water)
        order[3,:] = 3
        grid = SimpleNamespace(shape=water.shape, water=water, river_order=order)
        reaches = [np.array([(15.5,3.5),(.5,3.5)]),np.array([(8.5,3.5),(9.5,3.5)])]
        attrs = river_navigation_attributes(grid,reaches,
                    (route("seam","river",((15.5,3.5),(.5,3.5))),))
        self.assertEqual(attrs[0]["navigable"],"true")
        self.assertEqual(attrs[1],{})

    def test_partial_navigation_sections_cover_each_physical_edge_exactly_once(self):
        water = np.zeros((8,16),dtype=np.uint8)
        order = np.zeros_like(water); order[3,2:9] = 3
        grid = SimpleNamespace(shape=water.shape,water=water,river_order=order)
        source = np.array([(x+.5,3.5) for x in range(2,9)])
        curve = np.array([(x,3.5) for x in np.arange(2.5,8.51,.25)])
        sections = river_navigation_segments(grid,[source],[curve],
                     (route("short","river",((2.5,3.5),(4.5,3.5))),))[0]
        self.assertEqual([attrs["navigable"] for _,attrs in sections],["true","false"])
        lines = [shapely.LineString(points) for points,_ in sections]
        self.assertTrue(shapely.union_all(lines).equals(shapely.LineString(curve)))
        self.assertAlmostEqual(sum(line.length for line in lines),shapely.LineString(curve).length)
        np.testing.assert_array_equal(np.vstack([sections[0][0],sections[1][0][1:]]),curve)
        self.assertEqual(sections[0][1]["navigation-route-ids"],"short")
        self.assertEqual(sections[1][1]["navigation-route-ids"],"")

    def test_shared_prefix_is_drawn_once_and_all_ports_and_junctions_are_anchored(self):
        routes = (
            route("one", "sea", ((2, 2), (10, 2), (15, 6), (21, 6))),
            route("two", "sea", ((2, 2), (10, 2), (15, 6), (21, 12))),
        )
        paths = shared_transport_paths(self.grid, routes, land_geometry=shapely.GeometryCollection(), road_surface=shapely.GeometryCollection(), river_geometry=shapely.GeometryCollection(), anchors=())
        self.assertEqual(len(paths), 3)
        endpoints = {tuple(point) for _, _, points in paths for point in points[[0, -1]]}
        self.assertEqual(endpoints, {(2., 2.), (15., 6.), (21., 6.), (21., 12.)})
        display = [shapely.LineString(points) for _, _, points in paths]
        self.assertAlmostEqual(sum(line.length for line in display), shapely.union_all(display).length)

    def test_city_bridge_and_degree_two_nodes_survive_simplification(self):
        routes = (
            route("one", "road", ((2, 4), (12, 4))),
            route("two", "road", ((12, 4), (22, 4))),
        )
        paths = shared_transport_paths(self.grid, routes, land_geometry=shapely.box(0, 0, 64, 32), road_surface=shapely.box(0, 0, 64, 32), river_geometry=shapely.GeometryCollection(), anchors=((7, 4),))
        self.assertEqual(len(paths), 3)
        endpoints = {tuple(point) for _, _, points in paths for point in points[[0, -1]]}
        self.assertEqual(endpoints, {(2., 4.), (7., 4.), (12., 4.), (22., 4.)})

    def test_navigation_simplification_cannot_cut_an_island_or_change_endpoints(self):
        island = shapely.box(9, 5, 13, 15)
        source = ((4., 10.), (4., 3.), (8., 3.), (14., 3.), (20., 3.), (20., 10.))
        paths = shared_transport_paths(self.grid, (route("cape", "sea", source),), land_geometry=island, road_surface=island, river_geometry=shapely.GeometryCollection(), anchors=())
        points = paths[0][2]
        self.assertLess(len(points), len(source))
        np.testing.assert_array_equal(points[[0, -1]], np.asarray(source)[[0, -1]])
        self.assertFalse(shapely.LineString(points).intersects(island))
        curve = _curved_transport_points(points,"sea",island,shapely.GeometryCollection())
        self.assertGreater(len(curve),len(points))
        self.assertTrue(shapely.LineString(curve).disjoint(island))

    def test_parallel_redundant_sailing_bends_do_not_collapse_into_duplicate_ink(self):
        routes = (
            route("first", "sea", ((2., 4.), (7., 4.), (12., 4.))),
            route("second", "sea", ((2., 4.), (7., 5.), (12., 4.))),
        )
        paths = shared_transport_paths(self.grid, routes, land_geometry=shapely.GeometryCollection(), road_surface=shapely.GeometryCollection(), river_geometry=shapely.GeometryCollection(), anchors=())
        lines = [shapely.LineString(points) for _, _, points in paths]
        self.assertEqual(len(paths), 1)
        self.assertAlmostEqual(sum(line.length for line in lines), shapely.union_all(lines).length)

    def test_open_water_and_straight_roads_do_not_get_decorative_waves(self):
        for mode, land in (("sea", shapely.GeometryCollection()), ("road", shapely.box(0, 0, 64, 32))):
            points = np.array(((2., 4.), (22., 4.)))
            paths = shared_transport_paths(self.grid, (route("direct", mode, points),), land_geometry=land, road_surface=land, river_geometry=shapely.GeometryCollection(), anchors=())
            np.testing.assert_array_equal(paths[0][2], points)
            np.testing.assert_array_equal(_curved_transport_points(points,mode,land,shapely.GeometryCollection()),points)

    def test_quadratic_control_triangles_remain_in_visible_water(self):
        # The drawn shore extends beyond a binary land cell: checks must use
        # this vector surface, rather than just sampling the original grid.
        land = shapely.box(9.7, 0, 30, 9.7)
        points = np.array(((4., 12.), (9.5, 12.), (9.5, 4.)))
        curve = _curved_transport_points(points,"sea",land,shapely.GeometryCollection())
        self.assertGreater(len(curve),len(points))
        self.assertTrue(shapely.LineString(curve).disjoint(land))
        np.testing.assert_array_equal(curve[[0,-1]],points[[0,-1]])

    def test_long_road_bend_rounds_within_its_accepted_corridor(self):
        points=np.asarray(((2.,10.),(12.,10.),(22.,15.)))
        land=shapely.box(0.,0.,32.,32.)
        curve=_curved_transport_points(points,"road",land,shapely.GeometryCollection())
        # Long shallow bends can use a broader tangent instead of a fixed
        # sub-cell corner radius, while retaining the terrain-selected lane.
        self.assertGreater(np.linalg.norm(curve[1]-points[1]),.45)
        self.assertTrue(shapely.LineString(points).buffer(.35).covers(shapely.LineString(curve)))
        np.testing.assert_array_equal(curve[[0,-1]],points[[0,-1]])

    def test_road_rounding_still_requires_the_grade_profile_to_be_accepted(self):
        points=np.asarray(((2.,10.),(12.,10.),(22.,15.)))
        curve=_curved_transport_points(points,"road",shapely.box(0.,0.,32.,32.),
            shapely.GeometryCollection(),grade_check=lambda original,proposed:False)
        np.testing.assert_array_equal(curve,points)

    def test_road_cannot_straighten_across_a_valley_or_a_lake(self):
        land = shapely.box(0, 0, 64, 32).difference(shapely.box(9, 5, 13, 15))
        points = ((4., 10.), (4., 3.), (20., 3.), (20., 10.))
        paths = shared_transport_paths(self.grid, (route("valley", "road", points),), land_geometry=land, road_surface=land, river_geometry=shapely.GeometryCollection(), anchors=())
        np.testing.assert_array_equal(paths[0][2], points)

    def test_tiny_noded_bank_edge_does_not_erase_the_actual_turn(self):
        lake=shapely.box(8.,8.,11.,11.)
        land=shapely.box(0.,0.,32.,32.).difference(lake)
        points=np.array(((8.,12.),(12.,12.),(12.+2e-11,12.),(12.,8.)))
        curve=_curved_transport_points(points,"road",land,shapely.GeometryCollection())
        self.assertTrue(land.covers(shapely.LineString(curve)))
        self.assertTrue(shapely.LineString(curve).disjoint(lake))
        self.assertTrue(np.any(np.all(curve==points[2],axis=1)))
        np.testing.assert_array_equal(curve[[0,-1]],points[[0,-1]])

    def test_shared_routes_adopt_stronger_importance_and_leave_sources_unchanged(self):
        source = ((2., 4.), (12., 4.))
        routes = (route("local", "road", source, "local"), route("trunk", "road", source, "trunk"))
        paths = shared_transport_paths(self.grid, routes, land_geometry=shapely.box(0, 0, 64, 32), road_surface=shapely.box(0, 0, 64, 32), river_geometry=shapely.GeometryCollection(), anchors=())
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0][1], "trunk")
        self.assertEqual(routes[0].path, source)

    def test_wrapped_routes_are_split_before_network_union(self):
        source = ((60., 10.), (63., 10.), (1., 10.), (4., 10.))
        paths = shared_transport_paths(self.grid, (route("wrapped", "sea", source),), land_geometry=shapely.GeometryCollection(), road_surface=shapely.GeometryCollection(), river_geometry=shapely.GeometryCollection(), anchors=())
        self.assertEqual(len(paths), 2)
        for _, _, points in paths:
            self.assertLessEqual(shapely.LineString(points).length, 4.)
        endpoints=[tuple(p) for _,_,points in paths for p in (points[0],points[-1])]
        self.assertIn((64.,10.),endpoints)
        self.assertIn((0.,10.),endpoints)

    def test_one_wrapped_sailing_edge_retains_both_half_edges(self):
        paths=shared_transport_paths(self.grid,(route('seam','sea',((63.,8.),(1.,12.))),),
            land_geometry=shapely.GeometryCollection(),road_surface=shapely.GeometryCollection(),
            river_geometry=shapely.GeometryCollection(),anchors=())
        self.assertEqual(len(paths),2)
        endpoints=[tuple(p) for _,_,points in paths for p in (points[0],points[-1])]
        self.assertIn((64.,10.),endpoints)
        self.assertIn((0.,10.),endpoints)
        self.assertAlmostEqual(sum(shapely.LineString(p).length for _,_,p in paths),np.hypot(2,4))



if __name__ == "__main__":
    unittest.main()
