import unittest

import numpy as np
import shapely

from world_atlas.core.continuous_ecology import ContinuousEcologyField, FreshwaterCorridors


class ContinuousEcologyTests(unittest.TestCase):
    def test_world_without_freshwater_sources_keeps_only_its_background(self):
        shape=(8,16)
        sources=FreshwaterCorridors(16,8,(),(),shapely.GeometryCollection())
        field=ContinuousEcologyField(np.full(shape,.2),np.full(shape,.9),np.ones(shape,bool),sources,
            supply_capacity=.72,river_radii=(),lake_radius=2.)
        self.assertTrue(field.supply_superlevel(.3).is_empty)
        np.testing.assert_allclose(field.native,.2)
    def field(self,background=.1,ceiling=.95,capacity=.72):
        shape=(16,32)
        sources=FreshwaterCorridors(32,16,(3,),
            (shapely.LineString(((4.5,4.5),(12.5,12.5))),),shapely.GeometryCollection())
        return ContinuousEcologyField(np.full(shape,background),np.full(shape,ceiling),np.ones(shape,bool),sources,
            supply_capacity=capacity,river_radii=(2.,),lake_radius=2.)

    def test_diagonal_river_has_no_native_spacing_pulses(self):
        field=self.field()
        t=np.linspace(4.5,12.5,129)
        values=field.sample_points(t,t)
        np.testing.assert_array_equal(values,np.full(t.shape,float(np.float32(.72))))
        offset=np.sqrt(.5)
        side=field.sample_points(t-offset,t+offset)
        expected=float(np.float32(.72))*(1.-(1./2.)**2)**2
        np.testing.assert_allclose(side,expected,atol=2e-15,rtol=0)

    def test_quantitative_superlevel_identity_and_shared_native_ownership(self):
        field=self.field()
        cuts=(.2,.35,.5,.65,.8)
        frame=shapely.box(0,0,32,16)
        regions=field.class_regions(cuts,working_surface=frame)
        self.assertTrue(shapely.coverage_is_valid(np.asarray(regions,dtype=object)))
        self.assertTrue(shapely.union_all(regions).equals(frame))
        y,x=np.indices((16,32),dtype=float)
        probes=shapely.points(x.ravel()+.5,y.ravel()+.5)
        expected=np.digitize(field.native.ravel(),cuts)
        counts=np.zeros(len(probes),dtype=int)
        for label,region in enumerate(regions):
            inside=shapely.covers(region,probes)
            self.assertTrue(np.all(expected[inside]==label))
            counts+=inside
        np.testing.assert_array_equal(counts,1)

    def test_source_ceiling_and_background_remain_separate_physical_limits(self):
        field=self.field(background=.2,ceiling=.4)
        self.assertAlmostEqual(float(field.sample_points(7.3,7.3)),float(np.float32(.4)))
        self.assertAlmostEqual(float(field.sample_points(25.,7.3)),float(np.float32(.2)))
        self.assertTrue(field.superlevel(.5).is_empty)

    def test_barely_superthreshold_source_keeps_a_true_curved_capsule(self):
        field=self.field(background=0.,ceiling=1.,capacity=.50028479)
        footprint=field.superlevel(.5)
        self.assertTrue(footprint.is_valid)
        self.assertGreater(len(footprint.exterior.coords),30)
        self.assertTrue(footprint.covers(shapely.Point(7.25,7.25)))
        points=shapely.get_coordinates(footprint)
        np.testing.assert_allclose(field.sample_points(points[:,0],points[:,1]),.5,atol=2e-14,rtol=0)

    def test_inscribed_arc_precision_retains_true_native_distance_class(self):
        shape=(8,8)
        # Put a native witness just inside a circular end-cap, between the
        # coarse GEOS chord vertices. Its source value stays authoritative.
        centre=(3.5-1.9999*np.cos(np.pi/64),3.5-1.9999*np.sin(np.pi/64))
        sources=FreshwaterCorridors(8,8,(2,),(shapely.LineString((centre,(centre[0]-.3,centre[1]))),),shapely.GeometryCollection())
        field=ContinuousEcologyField(np.zeros(shape),np.ones(shape),np.ones(shape,bool),sources,
            supply_capacity=.72,river_radii=(3.,),lake_radius=2.)
        level=field.supply_capacity*(1-(2./3.)**2)**2
        self.assertGreater(float(field.supply_points(3.5,3.5)),level)
        self.assertTrue(field.supply_superlevel(level).covers(shapely.Point(3.5,3.5)))


if __name__=='__main__':
    unittest.main()
