from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import shapely

from world_atlas.core.city_harbors import derive_harbors, connect_harbor_routes
from world_atlas.core.society.model import TransportRoute
from test_presentation import _fixture


class CityHarborTests(unittest.TestCase):
    def test_shared_zero_shore_and_dry_city_and_wet_route(self):
        grid,society=_fixture()
        society=replace(society,settlements=(replace(society.settlements[0],row=3,column=1,site_type='port',population_min=1000,population_max=2000),))
        terrain=SimpleNamespace(sample_points=lambda x,y:(1.8-np.asarray(x))*100)
        locations={'port-city':(3.5,1.5)}
        harbors=derive_harbors(grid,society,locations,terrain,
            road_surface=shapely.box(0,0,1.8,grid.shape[0]))
        harbor=harbors['port-city']
        self.assertAlmostEqual(harbor['shore']['column'],1.8,places=6)
        self.assertGreater(terrain.sample_points(harbor['landPoint']['column'],harbor['landPoint']['row']),0)
        self.assertLess(terrain.sample_points(harbor['seaPoint']['column'],harbor['seaPoint']['row']),0)
        self.assertTrue(1<=locations['port-city'][1]<2)
        route=TransportRoute('sail','sea','regional','port-city',None,((1.5,3.5),(2.,3.5),(3.,3.5)))
        refined=connect_harbor_routes((route,),harbors,terrain)[0]
        self.assertEqual(refined.path[0],(harbor['seaPoint']['column'],harbor['seaPoint']['row']))
        self.assertEqual(route.path[0],(1.5,3.5))
        for a,b in zip(refined.path,refined.path[1:]):
            t=np.linspace(0,1,33)
            self.assertTrue(np.all(terrain.sample_points(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)<0))

    def test_coastal_refinement_keeps_the_city_and_dock_on_its_native_bank(self):
        grid,society=_fixture()
        society=replace(society,settlements=(replace(society.settlements[0],row=3,column=1,
            site_type='port',population_min=1000,population_max=2000),))
        terrain=SimpleNamespace(sample_points=lambda x,y:(1.8-np.asarray(x))*100)
        dry=shapely.box(0,0,1.8,grid.shape[0]).difference(shapely.box(1.6,3.49,1.79,3.51))
        locations={'port-city':(3.5,1.5)}
        harbor=derive_harbors(grid,society,locations,terrain,road_surface=dry)['port-city']
        row,column=locations['port-city']
        self.assertTrue(dry.covers(shapely.LineString(((1.5,3.5),(column,row)))))
        self.assertTrue(dry.covers(shapely.LineString([(p['column'],p['row']) for p in harbor['access']])))
