from types import SimpleNamespace
import unittest
import numpy as np
import shapely
from world_atlas.core.city_ground_locations import refine_city_ground_locations


class CityGroundLocationTests(unittest.TestCase):
    def locate(self,source,display,channel,land=None):
        grid=SimpleNamespace(shape=(2,4))
        city=SimpleNamespace(identifier='city',row=0,column=1)
        society=SimpleNamespace(settlements=(city,))
        terrain=SimpleNamespace(sample_points=lambda x,y:np.ones(np.shape(x))*100)
        locations={'city':(.5,1.5)}
        refine_city_ground_locations(grid,society,locations,terrain,
            land_surface=shapely.box(0,0,4,2) if land is None else land,
            river_source_geometry=shapely.MultiLineString([source]),
            river_geometry=shapely.MultiLineString([display]),river_channel_geometry=channel)
        return locations['city'],city

    def test_river_passing_a_coarse_centre_retains_its_original_bank_in_the_cell(self):
        location,city=self.locate(((0,1.2),(3,1.2)),((0,.4),(3,.4)),shapely.box(0,.37,3,.43))
        row,column=location
        self.assertTrue(0<row<.37)
        self.assertTrue(1<column<2)
        self.assertEqual((city.row,city.column),(0,1))

    def test_centre_on_a_native_flow_line_chooses_actual_dry_ground(self):
        location,_=self.locate(((0,.5),(3,.5)),((0,.5),(3,.5)),shapely.box(0,.45,3,.55))
        self.assertFalse(.45<=location[0]<=.55)
        self.assertTrue(0<location[0]<1 and 1<location[1]<2)

    def test_unmoved_dry_centre_keeps_its_exact_position(self):
        location,_=self.locate(((0,1.2),(3,1.2)),((0,1.2),(3,1.2)),shapely.box(0,1.17,3,1.23))
        self.assertEqual(location,(.5,1.5))

    def test_headwater_ray_does_not_create_an_infinite_bank_barrier(self):
        location,_=self.locate(((2,1.2),(3,1.2)),((2,.4),(3,.4)),shapely.box(2,.37,3,.43))
        self.assertEqual(location,(.5,1.5))

    def test_source_cell_without_actual_dry_ground_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'source cell has no dry ground'):
            self.locate(((0,1.2),(3,1.2)),((0,1.2),(3,1.2)),shapely.GeometryCollection(),land=shapely.box(0,0,1,2))
