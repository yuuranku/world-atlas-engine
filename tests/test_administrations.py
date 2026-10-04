"""Formed native ownership is the only administrative drawing source."""
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import shapely

from tests.test_world_identity import _society
from world_atlas.core.coastal_partition import enforce_homogeneous_components, clip_partition_to_surface
from world_atlas.core.society.administrations import (
    administrative_source, administrative_paint_coverage,
    write_administrative_stage, load_administrative_coverage)


def fixture():
    society = _society()
    shape = society.politics.state_id.shape
    grid = SimpleNamespace(shape=shape, water=np.zeros(shape,dtype=np.uint8),
                           content_digest=lambda:'same-physical-world')
    return grid,society


class AdministrativeOwnershipTests(unittest.TestCase):
    def assert_native(self,faces,labels,values,land):
        self.assertTrue(shapely.coverage_is_valid(faces))
        for row,column in zip(*np.where(land)):
            point=shapely.Point(column+.5,row+.5)
            observed={int(label) for face,label in zip(faces,labels) if face.covers(point)}
            self.assertEqual(observed,{int(values[row,column])})

    def test_drawing_keeps_the_authoritative_formed_seam(self):
        grid,society=fixture()
        labels=society.provinces.province_id.copy()
        labels[0:2,2:6]=1;labels[3,1:7]=2;labels[5,2:5]=1
        changed=replace(society,provinces=replace(society.provinces,province_id=labels),
                        politics=replace(society.politics,state_id=labels.astype(np.int16)))
        before=(changed.politics.state_id.copy(),changed.provinces.province_id.copy())
        source=administrative_source(grid,changed)
        faces,owners=administrative_paint_coverage(source,grid.water==0)
        self.assert_native(faces,owners,labels,grid.water==0)
        np.testing.assert_array_equal(changed.politics.state_id,before[0])
        np.testing.assert_array_equal(changed.provinces.province_id,before[1])
        for face,label in zip(faces,owners):
            self.assertEqual(source.province_to_state[label],label)

    def test_frontier_and_water_working_band_do_not_change_native_ownership(self):
        grid,society=fixture()
        grid.water[:,0]=1
        labels=society.provinces.province_id.copy();labels[:,0]=-1;labels[1:3,2]=0
        society=replace(society,provinces=replace(society.provinces,province_id=labels),
                        politics=replace(society.politics,state_id=labels.astype(np.int16)))
        source=administrative_source(grid,society)
        before=source.province_id.copy()
        faces,owners=administrative_paint_coverage(source,grid.water==0)
        self.assert_native(faces,owners,labels,grid.water==0)
        np.testing.assert_array_equal(source.province_id,before)
        self.assertAlmostEqual(shapely.union_all(faces).area,grid.shape[0]*grid.shape[1])

    def test_country_parent_must_match_its_actual_province_union(self):
        grid,society=fixture()
        labels=society.politics.state_id.copy();labels[3,3]=2
        society=SimpleNamespace(politics=SimpleNamespace(state_id=labels),provinces=society.provinces)
        with self.assertRaisesRegex(ValueError,'union'):
            administrative_source(grid,society)

    def test_homogeneous_island_keeps_its_own_province_under_the_shared_shore(self):
        water=np.ones((9,13),np.uint8);water[1:8,1:5]=0;water[4,6]=0
        provinces=np.where(water==0,1,-1).astype(np.int32);provinces[4,6]=2
        states=np.where(water==0,1,-1).astype(np.int16);states[4,6]=2
        grid=SimpleNamespace(shape=water.shape,water=water)
        society=SimpleNamespace(politics=SimpleNamespace(state_id=states),
            provinces=SimpleNamespace(province_id=provinces,provinces=(
                SimpleNamespace(identifier=1,state_identifier=1),
                SimpleNamespace(identifier=2,state_identifier=2))))
        source=administrative_source(grid,society)
        faces,labels=administrative_paint_coverage(source,water==0)
        mainland=shapely.box(.9,.9,5.1,8.1);island=shapely.box(6.02,4.02,6.98,4.98)
        land_surface=shapely.MultiPolygon((mainland,island))
        faces,labels=enforce_homogeneous_components(faces,labels,source.province_id,water==0,land_surface)
        clipped,owners=clip_partition_to_surface(faces,labels,land_surface)
        actual=shapely.union_all([face for face,label in zip(clipped,owners) if label==2])
        self.assertTrue(actual.equals(island))
        self.assertTrue(shapely.union_all(clipped).equals(land_surface))

    def test_island_correction_preserves_one_shared_province_country_graph(self):
        land=np.zeros((7,11),bool);land[1:6,1:5]=True;land[3,6]=True
        values=np.where(land,1,-1).astype(np.int32);values[3,6]=2
        mainland=shapely.box(.9,.9,5.1,6.1);island=shapely.box(6.02,3.02,6.98,3.98)
        surface=shapely.MultiPolygon((mainland,island))
        # A prior shared display arc intrudes into a homogeneous physical
        # island. The correction must replace both owners on that surface.
        faces=(shapely.box(0,0,6.6,7),shapely.box(6.6,0,11,7))
        labels=np.asarray((1,2))
        corrected,owners=enforce_homogeneous_components(faces,labels,values,land,surface)
        self.assertTrue(shapely.coverage_is_valid(corrected))
        visible,visible_ids=clip_partition_to_surface(corrected,owners,surface)
        provinces={owner:shapely.union_all([face for face,label in zip(visible,visible_ids) if label==owner])
                   for owner in (1,2)}
        self.assertTrue(provinces[1].equals(mainland))
        self.assertTrue(provinces[2].equals(island))
        parents=np.asarray((0,1,1))
        country=shapely.union_all([face for face,label in zip(visible,visible_ids) if parents[label]==1])
        self.assertTrue(country.equals(surface))
        self.assertEqual(provinces[1].intersection(provinces[2]).area,0.)

    def test_stage_persists_formed_bindings_without_obsolete_arrivals(self):
        grid,society=fixture();source=administrative_source(grid,society)
        with tempfile.TemporaryDirectory() as directory:
            faces,labels,report=write_administrative_stage(directory,grid,society,source)
            loaded,owners,parent=load_administrative_coverage(directory)
            self.assertEqual(report['nativeWrong'],0)
            self.assertEqual(report['schema'],'administrative-ownership-stage-v1')
            self.assertEqual(set(path.name for path in Path(directory).iterdir()),
                             {'manifest.json','shared-coverage.npz'})
            np.testing.assert_array_equal(labels,owners)
            np.testing.assert_array_equal(parent,source.province_to_state)
            self.assertTrue(all(a.equals(b) for a,b in zip(faces,loaded)))


if __name__=='__main__':unittest.main()
