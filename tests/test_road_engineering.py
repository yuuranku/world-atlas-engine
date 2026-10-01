import unittest
import numpy as np

from world_atlas.core.road_engineering import RoadEngineering
from world_atlas.core.society.transport import _least_cost_path
from world_atlas.core.continuous_terrain import PhysicalTerrainField


class RoadEngineeringTests(unittest.TestCase):
    def test_cart_road_uses_low_pass_instead_of_climbing_a_cliff(self):
        z=np.ones((24,48));z[:,24:]=401
        z[4:7]=1+np.clip((np.arange(48)-10)/25,0,1)*400
        valid=np.ones(z.shape,dtype=bool)
        engine=RoadEngineering(z,4,'ancient');friction=np.ones(z.shape)
        path=_least_cost_path(friction,valid,(12,5),(12,40),edge_costs=engine.edge_costs(friction))
        self.assertTrue(path)
        self.assertTrue(any(row in range(4,7) for row,col in path))
        for a,b in zip(path[:-1],path[1:],strict=True):
            if a[1]<24<=b[1]:self.assertIn(a[0],range(4,7))

    def test_same_hillside_has_different_climbing_and_contour_costs(self):
        z=1+np.broadcast_to(np.arange(48)*10.,(24,48))
        engine=RoadEngineering(z,1,'ancient');costs=engine.edge_costs(np.ones(z.shape)).values
        self.assertGreater(costs[12,20,4],costs[12,20,6])

    def test_display_cannot_replace_level_detour_with_a_mountain_chord(self):
        z=np.ones((24,48));z[8:16,20:28]=700
        field=PhysicalTerrainField(z,land_mask=z>0,sea_level_m=0,elevation_scale_m=1,elevation_exponent=1)
        engine=RoadEngineering(z,4,'preindustrial')
        original=np.array([[14.5,12.5],[14.5,5.5],[33.5,5.5],[33.5,12.5]])
        self.assertFalse(engine.preserves_profile(original,original[[0,-1]],terrain_field=field))

    def test_impossible_grade_is_disconnected_instead_of_a_fake_road(self):
        z=np.ones((16,32));z[:,16:]=5000
        engine=RoadEngineering(z,1,'preindustrial');friction=np.ones(z.shape)
        self.assertEqual(_least_cost_path(friction,np.ones(z.shape,dtype=bool),(8,10),(8,20),edge_costs=engine.edge_costs(friction)),())
