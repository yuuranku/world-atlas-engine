import unittest
from types import SimpleNamespace
import numpy as np
import shapely
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.navigation import build_navigation,travel_profile


class NavigationTests(unittest.TestCase):
    def build(self,lines,locations,*,sea_paths=(),land_surface=None,harbors=None,field=None):
        z=np.ones((24,48))
        if field is None:
            field=PhysicalTerrainField(z,land_mask=z>0,sea_level_m=0,elevation_scale_m=1,elevation_exponent=1)
        grid=SimpleNamespace(shape=z.shape,metadata={'planet':{'radiusKm':4,'rotationPeriodHours':26},
            'worldProfile':{'technologyEra':'preindustrial'}},content_digest=lambda:'test-ground')
        society=SimpleNamespace(settlements=[SimpleNamespace(identifier=k,name=k) for k in locations])
        return build_navigation(grid,society,shapely.MultiLineString(lines),terrain_field=field,
            settlement_locations=locations,profile=travel_profile('preindustrial',capabilities=('magic-flight','dragon')),
            sea_paths=sea_paths,land_surface=shapely.GeometryCollection() if land_surface is None else land_surface,
            harbors={} if harbors is None else harbors)

    def test_true_junction_and_city_anchor_are_queryable_nodes(self):
        result=self.build([[(2,12),(40,12)],[(20,3),(20,21)]],{'town':(12,10)})
        self.assertEqual(len(result['edges']),5)
        self.assertIn([20.,12.],result['nodes'])
        self.assertIsNotNone(result['cities'][0]['node'])
        self.assertTrue(all(edge['distanceKm']>0 and edge['maximumGrade']==0 for edge in result['edges']))

    def test_close_opposite_banks_never_become_a_junction(self):
        result=self.build([[(2,12),(40,12)],[(2,12.01),(40,12.01)]],{})
        self.assertEqual(len(result['edges']),2)
        self.assertEqual(len(result['nodes']),4)
        self.assertEqual(len({n for e in result['edges'] for n in (e['a'],e['b'])}),4)

    def test_magic_is_explicit_and_not_inferred_from_a_fantasy_name(self):
        ordinary=travel_profile('preindustrial')
        self.assertEqual([m['id'] for m in ordinary['modes']],['walk','horse','carriage','boat'])
        rare=travel_profile('preindustrial',capabilities=('magic-flight','dragon'))
        self.assertEqual([m['speedKmh']*m['dailyHours'] for m in rare['modes'] if m['kind']=='flight'],[120,390])

    def test_sea_and_road_crossing_is_not_a_transfer(self):
        result=self.build([[(2,12),(40,12)]],{},sea_paths=[[(20,3),(20,21)]])
        road={n for e in result['edges'] if e['kind']=='road' for n in (e['a'],e['b'])}
        sea={n for e in result['edges'] if e['kind']=='sea' for n in (e['a'],e['b'])}
        self.assertTrue(road.isdisjoint(sea))
        self.assertFalse(any(e['kind']=='port' for e in result['edges']))

    def test_saved_harbor_is_the_only_connection_and_its_dry_access_has_real_grade(self):
        harbor=dict(kind='sea',name='甲港',seaPoint={'column':6.1,'row':12},shore={'column':6,'row':12},
                    access=[{'column':4,'row':12},{'column':5.9,'row':12}])
        terrain=SimpleNamespace(sample_points=lambda x,y:(6-np.asarray(x))*10)
        result=self.build([[(2,12),(4,12)]],{'A':(12,4),'island':(6,35)},
            sea_paths=[[(6.1,12),(35,12)]],land_surface=shapely.box(0,0,6,24),harbors={'A':harbor},field=terrain)
        city=result['cities'][0]
        port=next(e for e in result['edges'] if e['kind']=='port')
        self.assertEqual(city['node'],port['a'])
        self.assertEqual(city['portNode'],port['a'])
        self.assertGreater(port['descentM'],19)
        self.assertGreater(port['maximumGrade'],0)
        self.assertEqual(port['segments'][-1][4],0,'wet berth span is boarding, not walking on water')
        self.assertIsNone(result['cities'][1]['node'])
        self.assertIsNone(result['cities'][1]['portNode'],'proximity is not a harbor')

    def test_land_blocks_sea_without_inventing_a_land_crossing(self):
        obstacle=shapely.box(20,10,22,14)
        result=self.build([],{},sea_paths=[[(10,12),(30,12)]],land_surface=obstacle)
        edges=result['edges']
        self.assertEqual(len(edges),2)
        self.assertTrue({edges[0]['a'],edges[0]['b']}.isdisjoint({edges[1]['a'],edges[1]['b']}))
        self.assertTrue(all(shapely.LineString(e['points']).intersection(obstacle).length==0 for e in edges))

    def test_dateline_connections_share_a_node_without_a_world_spanning_line(self):
        result=self.build([],{},sea_paths=[[(46,12),(48,12)],[(0,12),(2,12)]])
        self.assertEqual(len(result['nodes']),3)
        self.assertEqual(len(result['edges']),2)
        self.assertTrue(all(np.ptp(np.array(e['points'])[:,0])==2 for e in result['edges']))

    def test_ship_type_follows_saved_era(self):
        self.assertIn('独木舟',travel_profile('tribal')['modes'][-1]['label'])
        self.assertIn('动力船',travel_profile('contemporary')['modes'][-1]['label'])
