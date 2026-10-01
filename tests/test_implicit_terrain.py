import unittest

import numpy as np
import shapely
from scipy.optimize import brentq

from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.implicit_curves import level_bands
from world_atlas.core.implicit_terrain import terrain_height_range, terrain_level_curves, _SharedPorts, _query_cells, _cell_boxes
from world_atlas.core.terrain_refinement import RefinedTerrainField
from world_atlas.core.implicit_pchip import pchip_switch_abscissae
from tests.test_terrain_refinement import fixture


def physical(values):
    return PhysicalTerrainField(values,land_mask=values>0,sea_level_m=0,
                                elevation_scale_m=4000,elevation_exponent=1)


class ImplicitTerrainTests(unittest.TestCase):
    def test_actual_point_gradient_roundoff_keeps_zero_proposal_for_every_owner(self):
        from world_atlas.core.implicit_terrain import _gradient_directions
        low=np.array(((0.,-5.53551657e-307),(0.,0.),(0.,0.),(0.,-1.10176639e-321)))
        high=np.array(((0.,5.53551657e-307),(0.,0.),(0.,0.),(0.,7.46727657e-307)))
        np.testing.assert_array_equal(_gradient_directions(low,high),((0.,0.),))

    def test_direction_proposals_condition_subnormals_and_keep_degenerate_owners(self):
        from world_atlas.core.implicit_terrain import _gradient_directions
        fan=np.array(((-3.,1.),(1.,-.1),(-3.,1.),(1.,-.1)))
        low=np.stack((fan,np.zeros((4,2)),np.tile((2.,1.),(4,1)),
                      np.array(((1.,0.),(-1.,0.),(0.,1.),(0.,-1.)))))
        saved=low.copy()
        for factor in (1e-320,1e-160,1.,1e160,1e307):
            actual=_gradient_directions(low*factor,low*factor)
            self.assertEqual(actual.shape,(4,2))
            self.assertTrue(np.isfinite(actual).all())
            self.assertGreater(float((fan@actual[0]).min()),.17)
            np.testing.assert_array_equal(actual[1],(0.,0.))
            np.testing.assert_array_equal(actual[2],(1.,.5))
            np.testing.assert_array_equal(actual[3],(0.,0.))
        np.testing.assert_array_equal(low,saved)
        invalid=low.copy();invalid[0,0,0]=np.inf
        with self.assertRaisesRegex(ValueError,'finite'):
            _gradient_directions(invalid,low)

    def test_model_line_ports_share_native_identity_across_irrational_period_offsets(self):
        class Ground:
            width=2176
            @staticmethod
            def sample_points(x,y):return y
        for normal in ((1/np.sqrt(2),1-1/np.sqrt(2)),(1-1/np.sqrt(2),1/np.sqrt(2))):
            normal=np.asarray(normal);offset=.5;solved=int(np.argmax(abs(normal)));free=1-solved
            first,last=np.empty((2,2)),np.empty((2,2))
            for i,period in enumerate((0,1)):
                translation=period*Ground.width
                actual_offset=offset+normal[0]*translation
                a,b=((0.,1.)if free==1 else ((offset-normal[1])/normal[0],offset/normal[0]))
                first[i,free]=a+(translation if free==0 else 0.)
                last[i,free]=b+(translation if free==0 else 0.)
                first[i,solved]=(actual_offset-normal[free]*first[i,free])/normal[solved]
                last[i,solved]=(actual_offset-normal[free]*last[i,free])/normal[solved]
            saved_first,saved_last=first.copy(),last.copy()
            ports=_SharedPorts(Ground(),.25)
            points=ports.solve_model_lines(first,last,np.repeat(normal[None],2,axis=0),
                np.array((offset,offset+normal[0]*Ground.width)),((7,9,0,0),(7,9,0,0)),
                np.array((0,1)),np.array((offset,offset)))
            self.assertEqual(points[1,free],points[0,free]+(Ground.width if free==0 else 0.))
            self.assertEqual(len(ports.model_roots),1)
            self.assertLess(float(abs(points[:,1]-.25).max()),2e-13)
            np.testing.assert_array_equal(first,saved_first);np.testing.assert_array_equal(last,saved_last)

    def test_directional_chart_finds_a_real_gradient_cone_missed_by_axes_and_diagonals(self):
        from world_atlas.core.implicit_terrain import _separating_directions,_direction_sections
        class Ground:
            @staticmethod
            def sample_points(x,y):return np.maximum(-3*x+y,x-.1*y)
        field=Ground();lower=np.array(((1.,5.),));upper=np.array(((2.,6.5),))
        def bounds(ground,a,b):
            first=-3*a[:,0]+a[:,1];second=a[:,0]-.1*a[:,1]
            gradients=np.where((first>second)[:,None],(-3.,1.),(1.,-.1))
            return first,second,gradients,gradients
        gradients=np.array(((-3.,1.),(1.,-.1)))
        fixed=np.array(((1.,0.),(0.,1.),(1.,1.),(1.,-1.)))
        derivatives=gradients@fixed.T
        self.assertTrue(np.all(derivatives.min(axis=0)<0))
        self.assertTrue(np.all(derivatives.max(axis=0)>0))
        direction=_separating_directions(field,lower,upper,bounds)[0]
        self.assertGreater(float((gradients@direction).min()),.17)
        first=np.array((1.5,5.));last=np.array((11/6,6.5))
        probes=first+(last-first)*np.array((.25,.5,.75))[:,None]
        saved=probes.copy()
        points=_direction_sections(field,1.,probes,np.repeat(lower,3,axis=0),
                                   np.repeat(upper,3,axis=0),np.repeat(direction[None],3,axis=0))
        np.testing.assert_allclose(field.sample_points(points[:,0],points[:,1]),1.,rtol=0,atol=2e-14)
        self.assertGreater(float(np.linalg.norm(points-probes,axis=1).max()),.07)
        np.testing.assert_array_equal(probes,saved)

    def test_warped_source_switch_remains_an_explicit_port_on_the_true_ground(self):
        native=np.random.default_rng(89323).uniform(-500.,1500.,(9,11))
        base=physical(native)
        field=RefinedTerrainField(base,seed=1,radius_km=12,
            flow_to=np.full(native.shape,-1,dtype=np.int32),discharge=np.zeros(native.shape),
            river_segments=np.empty((0,2,2)),river_anchors=np.empty((0,2)))
        source=np.array((2.902347299137181,6.065684587852189))
        corner=np.array(field.inverse_ground_coordinates(*source))
        self.assertGreater(float(np.linalg.norm(corner-source)),.05)
        level=float(field.sample_points(*corner))
        self.assertGreater(abs(float(base.sample_points(*corner))-level),1.)
        curves=terrain_level_curves(field,[level],query_bounds=(2.84,6.04,2.87,6.07))[0]
        points=np.concatenate(curves)
        self.assertLess(float(abs(field.sample_points(points[:,0],points[:,1])-level).max()),2e-8)
        station=points[np.linalg.norm(points-corner,axis=1).argmin()]
        self.assertLess(float(np.linalg.norm(station-corner)),2e-10)
        mapped=np.array(field.forward_ground_coordinates(*station))
        np.testing.assert_allclose(mapped,source,rtol=0,atol=2e-12)
        np.testing.assert_array_equal(field.native_m,native)

    def test_coarse_gain_exclusion_requires_both_zero_pchip_and_no_drainage_support(self):
        from world_atlas.core.implicit_terrain import _zero_detail_boxes, _selected_ranges, _coarse_ranges
        native=np.full((9,9),1800.)
        flow=np.full(native.shape,-1,dtype=np.int32);discharge=np.zeros(native.shape)
        dry=RefinedTerrainField(physical(native),seed=1,radius_km=1000,
            flow_to=flow,discharge=discharge,river_segments=np.empty((0,2,2)),river_anchors=np.empty((0,2)))
        low,high=_coarse_ranges(dry)
        np.testing.assert_array_equal(low,np.full(low.shape,1800.))
        np.testing.assert_array_equal(high,np.full(high.shape,1800.))
        native[5,5]=200.;flow[4,4]=5*9+5;discharge[4,4]=128.
        field=RefinedTerrainField(physical(native),seed=1,radius_km=1000,
            flow_to=flow,discharge=discharge,river_segments=np.empty((0,2,2)),river_anchors=np.empty((0,2)))
        cells=np.array(((2,2),(5,5)))
        lower,upper=_cell_boxes(field,cells)
        zero=_zero_detail_boxes(field,cells[:,0]-1,cells[:,1]-1,lower,upper)
        np.testing.assert_array_equal(zero,(True,False))
        low,high=_selected_ranges(field,cells)
        self.assertEqual(low[0],1800.);self.assertEqual(high[0],1800.)
        self.assertLess(low[1],200.);self.assertGreater(high[1],1800.)
        points=lower[0]+np.array(((.1,.2),(.7,.8),(.43,.29)))
        np.testing.assert_array_equal(field.sample_points(points[:,0],points[:,1]),np.full(3,1800.))

    def test_real_nearest_river_corner_uses_a_certified_diagonal_chart(self):
        from world_atlas.core.implicit_terrain_bounds import field_range_gradient_bounds, directional_bounds
        native=np.full((9,9),100.);native[4:6,4:6]=1900.
        flow=np.full(native.shape,-1,dtype=np.int32);flow[5,4]=6*9+5
        discharge=np.zeros(native.shape);discharge[5,4]=128.
        rivers=np.array((((4.5,4.5),(5.5,5.5)),((4.5,5.5),(5.5,6.5))))
        field=RefinedTerrainField(physical(native),seed=1,radius_km=12,
            flow_to=flow,discharge=discharge,river_segments=rivers,river_anchors=np.empty((0,2)))
        corner=np.array((4.8,5.3));lower=corner[None,:]-1e-5;upper=corner[None,:]+1e-5
        _,_,glo,ghi=field_range_gradient_bounds(field,lower,upper)
        self.assertTrue(np.all(glo<0));self.assertTrue(np.all(ghi>0))
        dlo,dhi=directional_bounds(field,lower,upper,((1.,1.),(1.,-1.)))
        self.assertLess(dhi[0,0],-22.)
        self.assertLess(dlo[0,1],0.);self.assertGreater(dhi[0,1],0.)
        level=float(field.sample_points(*corner))
        paths=terrain_level_curves(field,[level],query_bounds=(4.79,5.29,4.81,5.31))[0]
        points=np.concatenate(paths)
        self.assertLess(float(abs(field.sample_points(points[:,0],points[:,1])-level).max()),2e-8)
        station=points[np.linalg.norm(points-corner,axis=1).argmin()]
        self.assertLess(float(np.linalg.norm(station-corner)),2e-10)
        self.assertEqual(float(station[0]-station[1]),-.5)
        np.testing.assert_array_equal(field.native_m,native)

    def test_native_queries_share_the_original_graph_and_deduplicate_overlaps(self):
        rows,columns=np.indices((7,11))
        field=physical(100.+columns*180+rows*115)
        queries=np.array(((0.,2.,1.,5.),(10.,2.,11.,5.),(3.2,1.8,7.2,5.4),(3.2,1.8,7.2,5.4)))
        level=1250.
        full=shapely.MultiLineString(terrain_level_curves(field,[level])[0])
        paths=terrain_level_curves(field,[level],query_bounds=queries)[0]
        selected=shapely.MultiLineString(paths)
        lower,upper=_cell_boxes(field,_query_cells(field,queries))
        native_queries=shapely.union_all(shapely.box(lower[:,0],lower[:,1],upper[:,0],upper[:,1]))
        expected=full.intersection(native_queries)
        self.assertLess(float(selected.hausdorff_distance(expected)),2e-10)
        self.assertAlmostEqual(selected.length,expected.length,places=9)
        single=terrain_level_curves(field,[level],query_bounds=queries[:-1])[0]
        self.assertEqual(shapely.normalize(selected).wkb,shapely.normalize(shapely.MultiLineString(single)).wkb)
        points=np.concatenate(paths)
        self.assertLess(float(abs(field.sample_points(points[:,0],points[:,1])-level).max()),2e-8)
        lo,hi=terrain_height_range(field,queries)
        self.assertLessEqual(lo,level);self.assertGreaterEqual(hi,level)
        self.assertEqual(terrain_level_curves(field,[level],query_bounds=np.empty((0,4))),[[]])
        with self.assertRaisesRegex(ValueError,'ordered rectangles'):
            terrain_level_curves(field,[level],query_bounds=(-1.,0.,2.,2.))

    def test_real_tensor_slope_switch_is_an_explicit_true_curve_port(self):
        native=np.random.default_rng(89323).uniform(100.,2000.,(9,11))
        field=physical(native)
        levels=(350.,350.001)
        curves=terrain_level_curves(field,levels)
        for level,paths in zip(levels,curves,strict=True):
            points=np.concatenate(paths)
            self.assertLess(float(abs(field.sample_points(points[:,0],points[:,1])-level).max()),2e-8)
        corner=np.array((4.27165524165717,1.9827907997308))
        self.assertLess(float(np.linalg.norm(np.concatenate(curves[1])-corner,axis=1).min()),2e-10)
        np.testing.assert_array_equal(field.native_m,native)

    def test_refined_corner_keeps_the_certified_bracket_and_source_event(self):
        native=np.random.default_rng(89323).uniform(100.,2000.,(9,11))
        base=physical(native)
        flow=np.full(native.shape,-1,dtype=np.int32);discharge=np.zeros(native.shape)
        source,target=((1,4),(2,4)) if native[1,4]>native[2,4] else ((2,4),(1,4))
        flow[source]=target[0]*11+target[1];discharge[source]=64.
        field=RefinedTerrainField(base,seed=1,radius_km=12,flow_to=flow,discharge=discharge,
                                  river_segments=np.empty((0,2,2)),river_anchors=np.empty((0,2)))
        lower,upper=np.array(((3.5,1.5),)),np.array(((4.5,2.5),))
        events=pchip_switch_abscissae(base,native,lower,upper)[0]
        event=float(events[np.argmin(abs(events-4.27165524165717))])
        ordinate=1.9827907997308
        level=float(field.sample_points(event,ordinate))
        self.assertGreater(abs(level-float(base.sample_points(event,ordinate))),.01)
        curves=terrain_level_curves(field,[level],query_bounds=(3.5,1.5,4.5,2.5))[0]
        points=np.concatenate(curves)
        self.assertTrue(bool(np.any(points[:,0]==event)))
        self.assertLess(float(abs(field.sample_points(points[:,0],points[:,1])-level).max()),2e-8)
        np.testing.assert_array_equal(field.native_m,native)

    def test_periodic_ports_are_identical_across_distinct_root_brackets_and_batches(self):
        rows,columns=np.indices((24,48))
        native=(rows-11.2)*30+150*np.cos((columns+.5)/48*2*np.pi*3+.7)
        _,field=fixture(native,seed=2)
        for batch in (False,True):
            ports=_SharedPorts(field,5.)
            first=np.array(((0.,7.),(48.,7.5)))
            last=np.array(((0.,9.),(48.,8.5)))
            if batch:
                points=ports.solve(first,last)
            else:
                points=np.vstack((ports.solve(first[:1],last[:1]),ports.solve(first[1:],last[1:])))
            self.assertEqual(float(points[0,1]),float(points[1,1]))
            np.testing.assert_array_equal(points[:,0],(0.,48.))
            self.assertEqual(len(ports.roots),1)
            self.assertLess(float(abs(field.sample_points(points[:,0],points[:,1])-5.).max()),2e-10)

    def test_refined_subgrid_peak_above_every_native_height_is_discovered(self):
        native=np.full((9,9),100.)
        native[3:6,3:6]=((700.,1100.,800.),(1200.,1900.,900.),(600.,800.,500.))
        base=physical(native)
        flow=np.full(native.shape,-1,dtype=np.int32)
        flow[4,4],flow[4,5]=4*9+5,5*9+5
        discharge=np.zeros(native.shape)
        discharge[4,4],discharge[4,5]=64.,128.
        field=RefinedTerrainField(base,seed=1,radius_km=12,flow_to=flow,discharge=discharge,
                                  river_segments=np.empty((0,2,2)),
                                  river_anchors=np.array(((3.71,3.89),)))
        level=1900.25
        peak=np.array((4.516311,4.50029469))
        self.assertLess(float(native.max()),level)
        self.assertGreater(float(field.sample_points(*peak)),level)
        curves=terrain_level_curves(field,[level])
        self.assertEqual(len(curves[0]),1)
        query=np.array(((4.4,4.4,4.7,4.7),(4.4,4.4,4.7,4.7)))
        local=terrain_level_curves(field,[level],query_bounds=query)
        self.assertEqual(shapely.normalize(shapely.MultiLineString(curves[0])).wkb,
                         shapely.normalize(shapely.MultiLineString(local[0])).wkb)
        lo,hi=terrain_height_range(field,query)
        self.assertLessEqual(lo,float(field.sample_points(*peak)))
        self.assertGreaterEqual(hi,float(field.sample_points(*peak)))
        levels=(1500.,1800.,level)
        combined=terrain_level_curves(field,levels)
        independent=[terrain_level_curves(field,[cut])[0]for cut in levels]
        for cached,separate in zip(combined,independent,strict=True):
            self.assertEqual(shapely.normalize(shapely.MultiLineString(cached)).wkb,
                             shapely.normalize(shapely.MultiLineString(separate)).wkb)
        points=curves[0][0]
        self.assertGreater(len(points),16)
        np.testing.assert_array_equal(points[0],points[-1])
        self.assertLess(float(abs(field.sample_points(points[:,0],points[:,1])-level).max()),2e-8)
        bands=level_bands(field,[level],curves)
        self.assertTrue(bands[1].covers(shapely.Point(peak)))
        self.assertFalse(bool(shapely.covers(bands[1],shapely.points(4.5,4.5))))
        self.assertTrue(bool(shapely.coverage_is_valid(np.asarray(bands))))
        np.testing.assert_array_equal(field.native_m,native)

    def test_true_base_peak_is_a_single_closed_shared_curve(self):
        native=np.full((7,7),.1);native[3,3]=2.0001
        field=physical(native)
        curves=terrain_level_curves(field,[2.])
        self.assertEqual(len(curves[0]),1)
        points=curves[0][0]
        self.assertGreater(len(points),16)
        np.testing.assert_array_equal(points[0],points[-1])
        self.assertLess(float(abs(field.sample_points(points[:,0],points[:,1])-2.).max()),2e-11)
        root=brentq(lambda x:float(field.sample_points(x,3.5))-2.,3.5,4.5,xtol=1e-14)
        self.assertAlmostEqual(float(points[:,0].max()),root,places=10)
        bands=level_bands(field,[2.],curves)
        self.assertTrue(bool(shapely.coverage_is_valid(np.asarray(bands))))
        self.assertEqual(shapely.box(0,0,7,7).symmetric_difference(shapely.union_all(bands)).area,0.)
        self.assertTrue(bands[1].covers(shapely.Point(3.5,3.5)))
        np.testing.assert_array_equal(field.native_m,native)

    def test_base_nested_levels_use_one_noded_coverage_and_true_holes(self):
        row,column=np.indices((11,17))
        field=physical(1.+np.hypot(row-5,column-8))
        levels=[3.3,5.1,7.2]
        curves=terrain_level_curves(field,levels)
        bands=level_bands(field,levels,curves)
        self.assertTrue(bool(shapely.coverage_is_valid(np.asarray(bands))))
        self.assertEqual(shapely.box(0,0,17,11).symmetric_difference(shapely.union_all(bands)).area,0.)
        self.assertEqual(len(bands[1].interiors),1)
        for level,paths in zip(levels,curves,strict=True):
            p=np.concatenate(paths)
            self.assertLess(float(abs(field.sample_points(p[:,0],p[:,1])-level).max()),1e-11)

    def test_metres_zero_constant_vertical_shore_uses_native_roundoff_scale(self):
        native=np.full((8,24),-20.)
        native[:,4:20]=10.
        field=physical(native)
        curves=terrain_level_curves(field,[0.])
        self.assertEqual(len(curves[0]),2)
        for points in curves[0]:
            self.assertLess(float(abs(field.sample_points(points[:,0],points[:,1])).max()),1e-12)
            self.assertEqual(float(np.ptp(points[:,0])),0.)
            self.assertEqual(float(points[:,1].min()),0.)
            self.assertEqual(float(points[:,1].max()),8.)

    def test_exact_native_height_plateau_keeps_its_perimeter_and_owner(self):
        native=np.full((5,5),100.);native[1:4,1:4]=200.
        base=physical(native)
        refined=RefinedTerrainField(base,seed=1,radius_km=1000,
            flow_to=np.full(native.shape,-1,dtype=np.int32),discharge=np.zeros(native.shape),
            river_segments=np.empty((0,2,2)),river_anchors=np.empty((0,2)))
        for field in (base,refined):
            with self.subTest(model=type(field).__name__):
                curves=terrain_level_curves(field,[200.])
                self.assertEqual(len(curves[0]),1)
                bands=level_bands(field,[200.],curves)
                self.assertGreater(bands[1].area,3.999999)
                self.assertLess(bands[1].area,4.00001)
                self.assertFalse(bands[1].covers(shapely.Point(.5,.5)))
                self.assertTrue(bands[1].covers(shapely.Point(2.5,2.5)))
                self.assertEqual(shapely.box(0,0,5,5).symmetric_difference(shapely.union_all(bands)).area,0.)

    def test_unsupported_field_and_invalid_levels_fail(self):
        field=physical(np.ones((3,4)))
        with self.assertRaisesRegex(ValueError,'increasing finite'):
            terrain_level_curves(field,[2.,1.])
        with self.assertRaisesRegex(ValueError,'accepted physical'):
            terrain_level_curves(object(),[1.])


if __name__=='__main__':
    unittest.main()
