from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.city_harbors import derive_harbors, connect_harbor_routes
from world_atlas.core.society.model import TransportRoute
from test_presentation import _fixture


class CityHarborTests(unittest.TestCase):
    def test_shared_zero_shore_and_dry_city_and_wet_route(self):
        grid,society=_fixture()
        society=replace(society,settlements=(replace(society.settlements[0],row=3,column=1,site_type='port',population_min=1000,population_max=2000),))
        terrain=SimpleNamespace(sample_points=lambda x,y:(1.8-np.asarray(x))*100)
        locations={'port-city':(3.5,1.5)}
        harbors=derive_harbors(grid,society,locations,terrain)
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
