import unittest
import numpy as np
import shapely
from world_atlas.core.society.administrative_front import AdministrativeFront,administrative_front
from world_atlas.core.society.administrative_coverage import administrative_coverage
from world_atlas.core.society.territorial_simulation import TerritorySeed
from tests.test_administrative_front import simulation


class AdministrativeCoverageTests(unittest.TestCase):
    def test_real_arrival_difference_moves_shared_front_off_midpoint(self):
        owners=np.array([[[1,2],[1,2]],[[2,1],[2,1]]])
        times=np.array([[[2.,1.],[2.,1.]],[[4.,5.],[4.,5.]]])
        front=AdministrativeFront(owners,times,np.ones((8,2,2)),np.ones((2,2),bool),20.,None,{1:0,2:0})
        faces,labels=administrative_coverage(front)
        first=shapely.union_all([face for face,label in zip(faces,labels) if label==1])
        second=shapely.union_all([face for face,label in zip(faces,labels) if label==2])
        shared=first.boundary.intersection(second.boundary)
        self.assertAlmostEqual(shared.bounds[0],.5+2./6.,places=7)
        self.assertAlmostEqual(shared.bounds[2],.5+2./6.,places=7)
        self.assertTrue(shapely.coverage_is_valid(faces))
        self.assertEqual(shapely.union_all(faces).area,4.)

    def test_three_actual_competitors_keep_native_winners_and_shared_complete_frame(self):
        front=administrative_front(simulation((19,37)),(TerritorySeed(3,4,1),TerritorySeed(12,16,2),TerritorySeed(7,29,3)))
        faces,labels=administrative_coverage(front)
        self.assertTrue(shapely.coverage_is_valid(faces))
        self.assertAlmostEqual(shapely.union_all(faces).area,19*37,places=8)
        for row in range(19):
            for column in range(37):
                point=shapely.Point(column+.5,row+.5)
                owned=[int(label) for face,label in zip(faces,labels) if face.covers(point)]
                self.assertIn(front.owner[row,column],owned)

    def test_shared_edges_do_not_depend_on_64_cell_work_block_boundary(self):
        front=administrative_front(simulation((17,130)),(TerritorySeed(2,40,1),TerritorySeed(12,84,2),TerritorySeed(7,121,3)))
        faces,labels=administrative_coverage(front)
        self.assertTrue(shapely.coverage_is_valid(faces))
        self.assertAlmostEqual(shapely.union_all(faces).area,17*130,places=7)
        self.assertEqual(set(labels),{1,2,3})

    def test_frontier_domain_is_explicit_and_is_not_filled_by_nearest_town(self):
        valid=np.ones((9,15),bool);valid[4]=False
        front=administrative_front(simulation(valid.shape,valid=valid),(TerritorySeed(2,2,1),))
        faces,labels=administrative_coverage(front)
        south=shapely.Point(7.5,7.5)
        self.assertEqual({int(label) for face,label in zip(faces,labels) if face.covers(south)},{0})

    def test_remote_runner_up_cannot_materialise_as_a_detached_subcell_country(self):
        owners=np.array([[[1,2],[1,2]],[[3,3],[3,3]]])
        times=np.array([[[0.,0.],[0.,0.]],[[.01,.01],[.01,.01]]])
        front=AdministrativeFront(owners,times,np.ones((8,2,2)),np.ones((2,2),bool),20.,None,{1:0,2:0,3:0})
        faces,labels=administrative_coverage(front)
        self.assertEqual(set(labels),{1,2})
        self.assertTrue(shapely.coverage_is_valid(faces))
        self.assertEqual(shapely.union_all(faces).area,4.)


if __name__=='__main__':unittest.main()
