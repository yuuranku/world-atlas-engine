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
            road_surface=shapely.box(0,0,1.8,grid.shape[0]),land_surface=shapely.box(0,0,1.8,grid.shape[0]))
        harbor=harbors['port-city']
        self.assertAlmostEqual(harbor['shore']['column'],1.8,places=6)
        self.assertGreater(terrain.sample_points(harbor['landPoint']['column'],harbor['landPoint']['row']),0)
        self.assertLess(terrain.sample_points(harbor['seaPoint']['column'],harbor['seaPoint']['row']),0)
        self.assertTrue(1<=locations['port-city'][1]<2)
        route=TransportRoute('sail','sea','regional','port-city',None,((1.5,3.5),(2.,3.5),(3.,3.5)))
        grid.water=np.ones(grid.shape,dtype=np.uint8);grid.water[:,:2]=0
        harbor['seaPoint']={'column':2.01,'row':3.5}
        route=replace(route,target_settlement_id='other-port')
        harbors['other-port']={**harbor,'seaPoint':{'column':3.,'row':3.5}}
        refined=connect_harbor_routes((route,),harbors,grid,
            land_surface=shapely.box(0,0,1.8,grid.shape[0]))[0]
        self.assertEqual(refined.path[0],(harbor['seaPoint']['column'],harbor['seaPoint']['row']))
        self.assertEqual(route.path[0],(1.5,3.5))
        for a,b in zip(refined.path,refined.path[1:]):
            t=np.linspace(0,1,33)
            self.assertTrue(np.all(terrain.sample_points(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t)<0))

    def test_sailing_route_avoids_unsampled_islands_and_native_land_corners(self):
        grid,_=_fixture()
        water=np.ones(grid.shape,dtype=np.uint8);water[2,3]=0
        grid.water=water
        island=shapely.box(4.01,2.99,4.03,3.01)
        land=shapely.union_all((shapely.box(3,2,4,3),island))
        harbors={'a':{'kind':'sea','seaPoint':{'column':1.5,'row':2.5}},
                 'b':{'kind':'sea','seaPoint':{'column':6.5,'row':3.5}}}
        route=TransportRoute('sail','sea','regional','a','b',((1.5,2.5),(6.5,3.5)))
        result=connect_harbor_routes((route,),harbors,grid,land_surface=land)[0]
        self.assertFalse(shapely.LineString(result.path).intersects(land))
        from world_atlas.acceptance import _transport_surface_violations
        self.assertFalse(_transport_surface_violations(grid,(result,),sea_land_geometry=land)['sea'])
        self.assertEqual(result.path[0],route.path[0]);self.assertEqual(result.path[-1],route.path[-1])

    def test_sea_route_requires_both_endpoints_to_have_real_berths(self):
        grid,_=_fixture()
        route=TransportRoute('sail','sea','regional','a','b',((1.5,2.5),(6.5,3.5)))
        with self.assertRaisesRegex(ValueError,'no real berth'):
            connect_harbor_routes((route,),{},grid,land_surface=shapely.Polygon())

    def test_berth_is_wet_on_the_published_coast_and_the_ground_field(self):
        grid,society=_fixture()
        society=replace(society,settlements=(replace(society.settlements[0],row=3,column=1,
            site_type='port',population_min=1000,population_max=2000),))
        terrain=SimpleNamespace(sample_points=lambda x,y:(1.8-np.asarray(x))*100)
        land=shapely.union_all((shapely.box(0,0,1.8,grid.shape[0]),shapely.box(1.8,3.43,2.2,3.57)))
        harbor=derive_harbors(grid,society,{'port-city':(3.5,1.5)},terrain,
            road_surface=land,land_surface=land)['port-city']
        point=harbor['seaPoint'];coordinates=(point['column'],point['row'])
        self.assertLess(float(terrain.sample_points(*coordinates)),0)
        self.assertFalse(land.covers(shapely.Point(coordinates)))

    def test_coastal_refinement_keeps_the_city_and_dock_on_its_native_bank(self):
        grid,society=_fixture()
        society=replace(society,settlements=(replace(society.settlements[0],row=3,column=1,
            site_type='port',population_min=1000,population_max=2000),))
        terrain=SimpleNamespace(sample_points=lambda x,y:(1.8-np.asarray(x))*100)
        dry=shapely.box(0,0,1.8,grid.shape[0]).difference(shapely.box(1.6,3.49,1.79,3.51))
        locations={'port-city':(3.5,1.5)}
        harbor=derive_harbors(grid,society,locations,terrain,road_surface=dry,
            land_surface=shapely.box(0,0,1.8,grid.shape[0]))['port-city']
        row,column=locations['port-city']
        from world_atlas.core.polygon_navigation import PolygonNavigator
        relocation=PolygonNavigator(dry).path((1.5,3.5),(column,row))
        self.assertTrue(dry.covers(shapely.LineString(relocation)))
        self.assertTrue(dry.covers(shapely.LineString([(p['column'],p['row']) for p in harbor['access']])))

    def test_harbor_access_turns_around_river_on_the_same_dry_bank(self):
        grid,society=_fixture()
        society=replace(society,settlements=(replace(society.settlements[0],row=3,column=1,
            site_type='port',population_min=1000,population_max=2000),))
        terrain=SimpleNamespace(sample_points=lambda x,y:(1.8-np.asarray(x))*100)
        land=shapely.box(0,0,1.8,grid.shape[0])
        river=shapely.box(1.6,2.5,1.7,4.5)
        dry=land.difference(river)
        harbor=derive_harbors(grid,society,{'port-city':(3.5,1.5)},terrain,
            road_surface=dry,land_surface=land)['port-city']
        landing=(harbor['landPoint']['column'],harbor['landPoint']['row'])
        self.assertFalse(dry.covers(shapely.LineString(((1.5,3.5),landing))))
        from world_atlas.core.polygon_navigation import PolygonNavigator
        original_access=PolygonNavigator(dry).path((1.5,3.5),landing)
        self.assertGreater(len(original_access),2)
        access=shapely.LineString([(p['column'],p['row']) for p in harbor['access']])
        self.assertTrue(dry.covers(access))

    def test_coarse_city_cell_refines_to_a_real_coastal_bank(self):
        grid,society=_fixture()
        city=replace(society.settlements[0],row=3,column=1,
            site_type='port',population_min=1000,population_max=2000)
        society=replace(society,settlements=(city,))
        terrain=SimpleNamespace(sample_points=lambda x,y:(1.8-np.asarray(x))*100)
        land=shapely.box(0,0,1.8,grid.shape[0])
        dry=land.difference(shapely.box(1.6,0,1.7,grid.shape[0]))
        locations={'port-city':(3.5,1.5)}
        harbor=derive_harbors(grid,society,locations,terrain,
            road_surface=dry,land_surface=land)['port-city']
        row,column=locations['port-city']
        self.assertTrue(1.7<column<1.8)
        self.assertEqual((city.row,city.column),(3,1))
        self.assertTrue(dry.covers(shapely.LineString(
            [(p['column'],p['row']) for p in harbor['access']])))
        self.assertEqual((harbor['access'][0]['row'],harbor['access'][0]['column']),(row,column))

    def test_port_keeps_a_coastal_bank_already_served_by_source_roads(self):
        grid,society=_fixture()
        city=replace(society.settlements[0],row=3,column=1,
            site_type='port',population_min=1000,population_max=2000)
        society=replace(society,settlements=(city,))
        terrain=SimpleNamespace(sample_points=lambda x,y:np.minimum(
            1.8-np.asarray(x),4.0-np.asarray(y))*100)
        land=shapely.box(0,0,1.8,4.)
        dry=land.difference(shapely.box(1.6,0,1.7,4.))
        source_bank=next(p for p in shapely.get_parts(dry) if p.covers(shapely.Point(1.5,3.5)))
        locations={'port-city':(3.5,1.5)}
        harbor=derive_harbors(grid,society,locations,terrain,
            road_surface=dry,land_surface=land)['port-city']
        row,column=locations['port-city']
        self.assertTrue(source_bank.covers(shapely.Point(column,row)))
        self.assertTrue(source_bank.covers(shapely.Point(
            harbor['landPoint']['column'],harbor['landPoint']['row'])))
