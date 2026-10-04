import math
import unittest

import numpy as np
import shapely

from world_atlas.core.mountain_roads import _route, engineer_mountain_path
from world_atlas.core.road_engineering import RoadEngineering, segment_lengths_km


class MountainRoadTests(unittest.TestCase):
    def test_confined_hillside_requires_real_switchbacks_in_both_directions(self):
        shape=(24,48)
        engineering=RoadEngineering(np.ones(shape),4,'ancient')
        class Hillside:
            def sample_points(self,x,y):
                return 100+(np.asarray(y)-8)*math.pi*4/24*1000*.18
        field=Hillside()
        endpoints=np.array([[10.5,9.5],[10.5,15.5]])
        domain=shapely.box(10,9,11,16)
        route=engineer_mountain_path(endpoints,engineering,terrain_field=field,road_surface=domain)
        np.testing.assert_array_equal(route[[0,-1]],endpoints)
        self.assertTrue(domain.covers(shapely.LineString(route)))
        dx=np.diff(route[:,0]);directions=np.sign(dx[abs(dx)>1e-10])
        self.assertGreaterEqual(np.count_nonzero(np.diff(directions)),4)
        direct=segment_lengths_km(endpoints,shape,4).sum()
        self.assertGreater(segment_lengths_km(route,shape,4).sum(),direct*1.8)
        for path in (route,route[::-1]):
            _,_,_,grade=engineering.profile(path,terrain_field=field)
            self.assertLessEqual(np.max(abs(grade)),.10)
        self.assertFalse(engineering.preserves_profile(route,endpoints,terrain_field=field))

    def test_flat_land_retains_direct_route(self):
        engineering=RoadEngineering(np.ones((24,48)),4,'ancient')
        class Plain:
            def sample_points(self,x,y):return np.ones(np.asarray(x).shape)
        path=np.array([[10.,10.],[15.,15.]])
        np.testing.assert_array_equal(engineer_mountain_path(path,engineering,
            terrain_field=Plain(),road_surface=shapely.box(0,0,48,24)),path)

    def test_continuous_ridge_rejects_first_graph_route_and_uses_valid_detour(self):
        engineering=RoadEngineering(np.ones((24,48)),4,'ancient')
        class NarrowRidge:
            def sample_points(self,x,y):
                return 100+6*np.exp(-((np.asarray(x)-5.55)/.016)**2-((np.asarray(y)-5)/.06)**2)
        field=NarrowRidge()
        endpoints=np.array([[5.,5.],[6.,5.]])
        _,_,_,direct_grade=engineering.profile(endpoints,terrain_field=field)
        self.assertGreater(np.max(abs(direct_grade)),engineering.maximum_grade)
        # The ridge falls between these search nodes; their interpolated
        # heights alone admit a direct edge that the real ground disallows.
        domain=shapely.box(4.9,4.5,6.1,5.5)
        route=_route(*endpoints,domain,engineering,field,margin=.5,resolution=8)
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(route[[0,-1]],endpoints)
        self.assertTrue(domain.covers(shapely.LineString(route)))
        self.assertGreater(float(np.max(abs(route[:,1]-5))),.05)
        _,_,_,grade=engineering.profile(route,terrain_field=field)
        self.assertLessEqual(np.max(abs(grade)),engineering.maximum_grade)

    def test_continuous_cliff_does_not_return_an_infeasible_graph_route(self):
        engineering=RoadEngineering(np.ones((24,48)),4,'ancient')
        class NarrowCliff:
            def sample_points(self,x,y):
                return 100+6*np.exp(-((np.asarray(x)-5.55)/.016)**2)
        route=_route(np.array([5.,5.]),np.array([6.,5.]),
                     shapely.box(4.9,4.99,6.1,5.01),engineering,NarrowCliff(),margin=.5,resolution=8)
        self.assertIsNone(route)

    def test_sharp_bank_uses_real_corner_when_lattice_cannot_reach_anchor(self):
        engineering=RoadEngineering(np.ones((24,48)),4,'ancient')
        corner=np.array([10.02,10.03])
        water=shapely.Polygon(((9.5,9.5),(10.54,9.5),corner))
        first=corner*.98+np.array([9.5,9.5])*.02
        last=corner*.98+np.array([10.54,9.5])*.02
        class SteepBanks:
            def sample_points(self,x,y):
                return 100+1000*shapely.distance(shapely.points(x,y),water)
        field=SteepBanks()
        domain=shapely.box(9.4,9.4,10.6,10.2).difference(water)
        route=_route(first,last,domain,engineering,field,margin=.5,resolution=32)
        self.assertIsNotNone(route)
        np.testing.assert_array_equal(route[[0,-1]],np.array([first,last]))
        self.assertTrue(domain.covers(shapely.LineString(route)))
        self.assertTrue(np.any(np.all(route==corner,axis=1)))
        _,_,_,grade=engineering.profile(route,terrain_field=field)
        self.assertLessEqual(np.max(abs(grade)),engineering.maximum_grade)
