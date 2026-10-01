from types import SimpleNamespace
import unittest

import numpy as np
import shapely

from world_atlas.core.society.model import TransportRoute
from world_atlas.core.society.transport import (
    _least_cost_path, _road_river_constraints, _terrain_safe_simplification,
    derive_bridges,
)
from world_atlas.core.transport_geometry import (
    native_river_geometry, prepare_transport_geometry, source_crossing_points,
)


def fixture(points, order=2):
    shape=(16,64)
    river_order=np.zeros(shape,dtype=np.uint8)
    flow_to=np.full(shape,-1,dtype=np.int32)
    cells=[(int(y),int(x))for x,y in points]
    for cell in cells:
        river_order[cell]=order
    for first,last in zip(cells[:-1],cells[1:],strict=True):
        flow_to[first]=last[0]*shape[1]+last[1]
    return SimpleNamespace(shape=shape,water=np.zeros(shape,dtype=np.uint8),
                           river_order=river_order,flow_to=flow_to,
                           metadata={'planet':{'radiusKm':6400},'worldProfile':{'technologyEra':'preindustrial'}})


def raw(grid):
    return np.where(grid.water==0,1.,-1.)


def road(identifier,points,importance="regional",mode="road"):
    return TransportRoute(identifier,mode,importance,"west","east",tuple(points))


def prepare(grid,routes,display=None,bridges=None):
    from world_atlas.core.continuous_terrain import PhysicalTerrainField
    field=PhysicalTerrainField(raw(grid),land_mask=raw(grid)>0,sea_level_m=0,elevation_scale_m=1,elevation_exponent=1)
    source=native_river_geometry(grid,raw_elevation_m=raw(grid))
    return prepare_transport_geometry(grid,routes,derive_bridges(grid,routes,raw_elevation_m=raw(grid)) if bridges is None else bridges,
                                      locations={},land_surface=shapely.box(0,0,64,16),
                                      river_source_geometry=source,river_geometry=source if display is None else display,
                                      river_channel_geometry=(source if display is None else display).buffer(.01),raw_elevation_m=raw(grid),terrain_field=field)


class TransportCrossingTests(unittest.TestCase):
    def test_facility_is_the_actual_shared_road_and_shifted_channel_point(self):
        grid=fixture([(12.5,y+.5)for y in range(2,14)])
        routes=(road("cross",((5.5,7.5),(19.5,7.5))),)
        display=shapely.MultiLineString([[(12.7,2.5),(12.7,13.5)]])
        prepared=prepare(grid,routes,display)
        self.assertEqual(prepared.bridges,derive_bridges(grid,routes,raw_elevation_m=raw(grid)))
        self.assertEqual(len(prepared.bridges),1)
        point=shapely.Point(prepared.positions[prepared.bridges[0].identifier])
        network=shapely.union_all([shapely.LineString(path)for mode,_importance,path in prepared.paths if mode=="road"])
        self.assertLess(point.distance(network),1e-8)
        self.assertLess(point.distance(display),1e-8)
        self.assertAlmostEqual(point.x,12.7)
        self.assertEqual(prepared.crossings[0]["nativePoint"],[12.5,7.5])
        transition=prepared.crossings[0]["sourceBankTransitions"][0]
        self.assertEqual(transition["routeIdentifier"],"cross")
        self.assertEqual(len(transition["nativeDirections"]),2)
        self.assertEqual(len(set(map(tuple,transition["bankPoints"]))),2)
        self.assertEqual(prepared.tangents[prepared.bridges[0].identifier],(1.,0.))
        span=prepared.spans[prepared.bridges[0].identifier]
        geometry=shapely.geometry.shape(span["geometry"])
        self.assertLess(geometry.distance(point),1e-8)
        self.assertEqual(len(span["bankPoints"]),2)
        self.assertAlmostEqual(geometry.length,.02)
        self.assertEqual({owner for portal in span["bankPortals"]for owner in portal["sourceRoadIds"]},{"cross"})
        self.assertEqual(routes[0].path,((5.5,7.5),(19.5,7.5)))

    def test_nearby_distinct_crossings_retain_two_source_facilities(self):
        grid=fixture([(12.5,y+.5)for y in range(2,14)])
        routes=(road("one",((5.5,7.5),(19.5,7.5))),road("two",((5.5,9.5),(19.5,9.5))))
        prepared=prepare(grid,routes)
        self.assertEqual(len(prepared.bridges),2)
        self.assertEqual(len(set(prepared.positions)),2)

    def test_shared_physical_crossing_has_one_facility_and_all_source_routes(self):
        grid=fixture([(12.5,y+.5)for y in range(2,14)])
        routes=(road("weak",((5.5,7.5),(19.5,7.5)),"local"),road("strong",((5.5,7.5),(19.5,7.5)),"trunk"))
        prepared=prepare(grid,routes)
        self.assertEqual(len(prepared.bridges),1)
        self.assertEqual(prepared.bridges[0].route_identifier,"strong")
        self.assertEqual(prepared.crossings[0]["sourceRoadIds"],["strong","weak"])

    def test_oblique_crossing_before_an_overlap_is_classified_locally(self):
        stream=[(1.5,9.5),(2.5,9.5),(3.5,8.5),(4.5,7.5),(5.5,6.5),(6.5,6.5),(7.5,6.5),(8.5,6.5)]
        grid=fixture(stream,order=1)
        routes=(road("mixed",((7.5,5.5),(4.5,7.5),(3.5,8.5),(3.5,9.5)),"local"),)
        prepared=prepare(grid,routes)
        self.assertEqual(len(prepared.bridges),1)
        self.assertEqual(prepared.crossings[0]["nativePoint"],[6.,6.5])
        stream_geometry=native_river_geometry(grid,raw_elevation_m=raw(grid))
        network=shapely.union_all([shapely.LineString(path)for _mode,_importance,path in prepared.paths])
        self.assertEqual(shapely.intersection(network,stream_geometry).geom_type,"Point")
        np.testing.assert_array_equal(prepared.routes[0].path[0],routes[0].path[0])
        np.testing.assert_array_equal(prepared.routes[0].path[-1],routes[0].path[-1])

    def test_missing_facility_is_rejected_before_any_drawing(self):
        grid=fixture([(12.5,y+.5)for y in range(2,14)])
        with self.assertRaisesRegex(ValueError,"recorded bridges"):
            prepare(grid,(road("cross",((5.5,7.5),(19.5,7.5))),),bridges=())

    def test_true_outflow_edge_is_source_evidence_for_a_coastal_crossing(self):
        grid=fixture([(12.5,10.5),(12.5,11.5)])
        grid.river_order[11,12]=0
        grid.water[11,12]=1
        routes=(road("mouth",((10.5,11.),(14.5,11.))),)
        self.assertEqual(len(derive_bridges(grid,routes,raw_elevation_m=raw(grid))),1)

    def test_local_route_goes_round_large_stream_and_simplification_preserves_it(self):
        grid=fixture([(12.5,y+.5)for y in range(3,13)],order=3)
        valid,blocked,barrier=_road_river_constraints(grid,"local",raw_elevation_m=raw(grid))
        friction=np.ones(grid.shape)
        path=_least_cost_path(friction,valid,(7,9),(7,15),blocked_edges=blocked)
        self.assertTrue(path)
        shortened=_terrain_safe_simplification(path,friction,valid,river_barrier=barrier)
        line=shapely.LineString([(column+.5,row+.5)for row,column in shortened])
        self.assertEqual(source_crossing_points(line,barrier),())
        self.assertGreater(line.length,6.)

    def test_opposed_d8_flow_edges_cannot_be_crossed_between_cell_centres(self):
        grid=fixture([(4.5,4.5),(5.5,5.5)],order=3)
        valid,blocked,_barrier=_road_river_constraints(grid,"local",raw_elevation_m=raw(grid))
        self.assertTrue(valid[4,5] and valid[5,4])
        self.assertTrue(blocked[4,5] & (1<<5))
        self.assertTrue(blocked[5,4] & (1<<2))

    def test_odd_degree_confluence_is_a_crossing_even_on_multiline_boundary(self):
        grid=fixture([(5.5,y+.5)for y in range(2,9)])
        grid.river_order[4,4]=2
        grid.flow_to[4,4]=5*grid.shape[1]+5
        routes=(road("junction",((4.5,5.5),(6.5,5.5))),)
        river=native_river_geometry(grid,raw_elevation_m=raw(grid))
        self.assertFalse(shapely.crosses(shapely.LineString(routes[0].path),river))
        self.assertEqual(len(derive_bridges(grid,routes,raw_elevation_m=raw(grid))),1)

    def test_road_touching_the_channel_and_returning_to_the_same_bank_has_no_bridge(self):
        grid=fixture([(5.5,y+.5)for y in range(2,9)])
        routes=(road("bank",((4.5,5.5),(5.5,5.5),(4.5,6.5))),)
        self.assertEqual(derive_bridges(grid,routes,raw_elevation_m=raw(grid)),())
        prepared=prepare(grid,routes)
        network=shapely.union_all([shapely.LineString(path)for _mode,_importance,path in prepared.paths])
        self.assertTrue(network.disjoint(native_river_geometry(grid,raw_elevation_m=raw(grid))))

    def test_opposite_bank_roads_meeting_at_a_river_city_need_a_source_facility(self):
        grid=fixture([(12.5,y+.5)for y in range(2,14)])
        routes=(road("west",((5.5,7.5),(12.5,7.5))),road("east",((12.5,7.5),(19.5,7.5))))
        bridges=derive_bridges(grid,routes,raw_elevation_m=raw(grid))
        self.assertEqual(len(bridges),1)
        prepared=prepare(grid,routes)
        self.assertEqual(prepared.crossings[0]["nativePoint"],[12.5,7.5])
        self.assertEqual(prepared.crossings[0]["sourceRoadIds"],["east","west"])

    def test_single_bank_city_access_is_proven_separately_from_bridges(self):
        grid=fixture([(12.5,y+.5)for y in range(2,14)])
        routes=(road("access",((5.5,7.5),(12.5,7.5))),)
        prepared=prepare(grid,routes)
        self.assertEqual(prepared.bridges,())
        self.assertEqual(prepared.crossings,())
        self.assertEqual(len(prepared.endpoint_touches),1)
        proof=prepared.endpoint_touches[0]
        self.assertEqual(proof["nativePoint"],[12.5,7.5])
        self.assertEqual(proof["displayBankCount"],1)
        self.assertEqual(proof["nativeBankCount"],1)
        self.assertEqual(proof["sourceSettlementIds"],["east"])
        self.assertEqual(len(proof["bankPoints"]),1)
        self.assertAlmostEqual(shapely.geometry.shape(proof["channelAccessGeometry"]).length,.01)

    def test_recorded_confluence_facility_reaches_the_actual_narrow_bank_sector(self):
        grid=fixture([(12.5,y+.5)for y in range(2,14)])
        grid.river_order[5,10]=grid.river_order[6,11]=1
        grid.flow_to[5,10]=6*64+11
        grid.flow_to[6,11]=7*64+12
        routes=(road("junction",((11.5,5.5),(11.5,6.5),(12.5,7.5),(19.5,7.5))),)
        prepared=prepare(grid,routes)
        self.assertEqual(len(prepared.bridges),1)
        self.assertEqual(prepared.crossings[0]["nativePoint"],[12.5,7.5])
        span=prepared.spans[prepared.bridges[0].identifier]
        self.assertEqual(len(span["bankPoints"]),2)
        # The northwestern approach occupies the bank between a tributary
        # and the main channel; a normal to either one alone misses it.
        self.assertTrue(any(x<12.5 and y<7.5 for x,y in span["bankPoints"]))

    def test_large_coordinate_overlap_keeps_shared_bank_noding_closed(self):
        river=shapely.LineString(((916.,242.5),(915.5,242.5),(914.5,241.5),
                                  (913.5,242.5),(912.5,241.5),(912.,242.)))
        road_line=shapely.LineString(((912.5,243.5),(914.5,241.5),(914.5,239.5)))
        self.assertEqual([tuple(point.coords)[0]for point in source_crossing_points(road_line,river)],
                         [(913.5,242.5)])


if __name__=="__main__":
    unittest.main()
