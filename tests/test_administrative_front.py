import unittest
import numpy as np
from world_atlas.core.society.administrative_front import administrative_front
from world_atlas.core.society.territorial_simulation import TerritorySimulation,TerritorySeed
from world_atlas.core.society.spatial import connected_components


def simulation(shape=(17,31), *, valid=None, domain=None):
    return TerritorySimulation(valid=np.ones(shape,dtype=bool) if valid is None else valid,
         friction=np.ones(shape,dtype=np.float32),transition_penalty=np.zeros((8,*shape),dtype=np.float32),
         road_access=np.zeros(shape,dtype=np.float32),bridge_edges=np.zeros((8,*shape),dtype=np.float32),owner_constraint=domain)


class AdministrativeFrontTests(unittest.TestCase):
    def test_uniform_metric_is_isotropic_and_keeps_distinct_arrivals(self):
        model=simulation()
        seeds=(TerritorySeed(5,7,1),TerritorySeed(11,23,2))
        front=administrative_front(model,seeds)
        yy,xx=np.indices(model.valid.shape)
        exact=np.stack([np.hypot(yy-seed.row,np.minimum(abs(xx-seed.column),model.valid.shape[1]-abs(xx-seed.column))) for seed in seeds])
        self.assertTrue(np.all(front.arrivals[0]>=exact.min(axis=0)-1e-12))
        self.assertLess(float(np.max(front.arrivals[0]-exact.min(axis=0))),1.5)
        self.assertTrue(np.all(front.arrivals[1]>=front.arrivals[0]))
        self.assertTrue(np.all(front.owners[0]!=front.owners[1]))

    def test_three_competitors_have_actual_core_connected_winning_domains(self):
        model=simulation()
        seeds=(TerritorySeed(5,4,1),TerritorySeed(11,14,2),TerritorySeed(8,24,3))
        front=administrative_front(model,seeds)
        self.assertTrue(np.all(front.owners[0]!=front.owners[1]))
        for seed in seeds:
            components,sizes=connected_components(front.owner==seed.owner)
            self.assertEqual(len(sizes),1)
            self.assertEqual(components[seed.row,seed.column],1)

    def test_diagonal_cheap_route_cannot_create_a_detached_winning_cell(self):
        model=simulation((23,39))
        friction=np.ones(model.valid.shape,np.float32)
        friction[np.arange(3,20),np.arange(4,21)]=.1
        model=TerritorySimulation(model.valid,friction,model.transition_penalty,model.road_access,model.bridge_edges)
        seeds=(TerritorySeed(3,4,1),TerritorySeed(12,24,2),TerritorySeed(19,7,3))
        front=administrative_front(model,seeds)
        for seed in seeds:
            components,sizes=connected_components(front.owner==seed.owner)
            self.assertEqual(len(sizes),1)
            self.assertEqual(components[seed.row,seed.column],1)

    def test_same_state_towns_cooperate_without_becoming_runner_up(self):
        front=administrative_front(simulation(),(TerritorySeed(5,4,1),TerritorySeed(11,14,1),TerritorySeed(8,24,2)))
        self.assertTrue(np.all(front.owners[0]!=front.owners[1]))
        self.assertEqual(front.owner[5,4],1)
        self.assertEqual(front.owner[11,14],1)
        self.assertEqual(front.owner[8,24],2)

    def test_closed_unseeded_component_remains_uncontrolled(self):
        valid=np.ones((9,15),dtype=bool);valid[4]=False
        front=administrative_front(simulation(valid.shape,valid=valid),(TerritorySeed(2,2,1),))
        self.assertTrue(np.all(front.owner[5:]==0))
        self.assertTrue(np.all(np.isinf(front.arrivals[:,5:])))
        self.assertTrue(np.all(front.owner[4]==-1))

    def test_remote_authority_arrives_by_actual_sea_path_cost(self):
        valid=np.zeros((9,21),bool);valid[2:7,1:6]=True;valid[2:7,14:19]=True
        model=simulation(valid.shape,valid=valid)
        seed=TerritorySeed(4,2,1)
        unconnected=administrative_front(model,(seed,))
        self.assertEqual(unconnected.owner[4,16],0)
        connected=administrative_front(model,(seed,),maritime_links=(((4,4),(4,15),17.),))
        self.assertEqual(connected.owner[4,16],1)
        self.assertAlmostEqual(float(connected.arrivals[0,4,15]),19.)
        with self.assertRaisesRegex(ValueError,'positive actual route cost'):
            administrative_front(model,(seed,),maritime_links=(((4,4),(4,15),0.),))

    def test_cultural_parent_is_a_real_domain_constraint(self):
        domain=np.ones((9,15),dtype=np.int32);domain[:,8:]=2
        front=administrative_front(simulation(domain.shape,domain=domain),
                                  (TerritorySeed(2,2,1,domain=1),TerritorySeed(6,12,2,domain=2)))
        np.testing.assert_array_equal(front.owner,domain)
        self.assertTrue(np.all(front.owners[1]==0))

    def test_owner_strength_cannot_change_the_shared_metric(self):
        with self.assertRaisesRegex(ValueError,'shared travel metric'):
            administrative_front(simulation(),(TerritorySeed(2,2,1,strength=2),))


if __name__=='__main__':unittest.main()
