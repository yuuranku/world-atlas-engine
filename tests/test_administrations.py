"""One final arrival partition drives country and province ownership together."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import shapely

from tests.test_world_identity import _society
from world_atlas.core.society.administrations import (
    administrative_source, administrative_paint_coverage, derive_administrations)


def fixture():
    society = _society()
    shape = society.politics.state_id.shape
    civilization = np.ones(shape, dtype=np.int16)
    society = replace(society,
        cultures=replace(society.cultures, civilization_id=civilization,
            language_id=civilization, language_family_id=civilization),
        politics=replace(society.politics, states=tuple(replace(record, civilization_identifier=1)
            for record in society.politics.states)))
    grid = SimpleNamespace(shape=shape, water=np.zeros(shape,dtype=np.uint8),
        elevation=np.full(shape,.2), river_order=np.zeros(shape,dtype=np.uint8),
        metadata=dict(extents=dict(west=-180,east=180,north=80,south=-80),planet=dict(radiusKm=6400)))
    thematic = SimpleNamespace(land_potential=np.full(shape,.8),habitability=np.full(shape,.7))
    return grid, thematic, society


class AdministrativeSourceTests(unittest.TestCase):
    def test_final_categorical_seam_is_not_an_input_to_the_arrival_field(self):
        grid, thematic, society = fixture()
        labels = society.provinces.province_id.copy()
        labels[0:2,2:6] = 1
        labels[3,1:7] = 2
        labels[5,2:5] = 1
        changed = replace(society,
            provinces=replace(society.provinces, province_id=labels),
            politics=replace(society.politics,state_id=labels.astype(np.int16)))
        first = administrative_source(grid,thematic,society)
        second = administrative_source(grid,thematic,changed)
        for layer in ('countries','provinces'):
            np.testing.assert_array_equal(getattr(first,layer).owners,getattr(second,layer).owners)
            np.testing.assert_array_equal(getattr(first,layer).arrivals,getattr(second,layer).arrivals)
        faces,owners=administrative_paint_coverage(first,grid.water==0)
        self.assertTrue(shapely.coverage_is_valid(faces))
        for row,column in np.ndindex(grid.shape):
            point=shapely.Point(column+.5,row+.5)
            observed={int(owner)for face,owner in zip(faces,owners)if face.covers(point)}
            expected=int(first.provinces.owner[row,column])
            if any(layer.arrivals[0,row,column]==layer.arrivals[1,row,column]
                   for layer in (first.countries,first.provinces)):
                # An exact metric tie lies on the shared, zero-area boundary.
                self.assertIn(expected,observed)
            else:
                self.assertEqual(observed,{expected})

    def test_coast_paint_extends_arrivals_only_in_water(self):
        grid, thematic, society = fixture()
        front = administrative_source(grid,thematic,society).provinces
        land = np.ones(grid.shape,bool);land[:,0]=False
        # This water-only observation is excluded from the source front.
        owners=front.owners.copy();owners[:,:,0]=-1
        arrivals=front.arrivals.copy();arrivals[:,:,0]=np.inf
        valid=front.valid.copy();valid[:,0]=False
        front=replace(front,owners=owners,arrivals=arrivals,valid=valid)
        before=(front.owners.copy(),front.arrivals.copy())
        faces, labels = administrative_paint_coverage(front,land)
        np.testing.assert_array_equal(front.owners,before[0])
        np.testing.assert_array_equal(front.arrivals,before[1])
        self.assertTrue(shapely.coverage_is_valid(faces))
        self.assertAlmostEqual(shapely.union_all(faces).area,grid.shape[0]*grid.shape[1])
        for row,column in zip(*np.where(land)):
            owned=[int(label) for face,label in zip(faces,labels) if face.covers(shapely.Point(column+.5,row+.5))]
            self.assertIn(int(front.owner[row,column]),owned)


if __name__=='__main__':unittest.main()
