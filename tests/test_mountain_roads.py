import math
import unittest

import numpy as np
import shapely

from world_atlas.core.mountain_roads import engineer_mountain_path
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
