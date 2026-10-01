import unittest

import numpy as np
import shapely
from scipy.optimize import brentq

from world_atlas.core.cartographic_features import filled_geometries
from world_atlas.core.continuous_scalar import ContinuousScalarField, scalar_band_paths


class ContinuousScalarTests(unittest.TestCase):
    def test_land_observations_are_unchanged_and_coast_has_no_false_zero(self):
        values = np.zeros((6, 12))
        land = np.zeros_like(values, dtype=bool)
        land[:, :4] = True
        values[land] = .7
        before = values.copy()
        field = ContinuousScalarField(values, land)
        np.testing.assert_array_equal(values, before)
        np.testing.assert_allclose(field.sample_rect(np.arange(12)+.5, np.arange(6)+.5), .7)
        np.testing.assert_allclose(field.sample_rect([3.5, 4., 4.5], [1., 2.5, 5.]), .7)

    def test_tile_edges_and_longitude_seam_use_identical_neighbours(self):
        values = np.random.default_rng(12).uniform(size=(8, 16))
        field = ContinuousScalarField(values, np.ones_like(values, dtype=bool))
        whole = field.sample_rect(np.arange(0, 16.01, .25), np.arange(0, 8.01, .25))
        left = field.sample_rect(np.arange(0, 8.01, .25), np.arange(0, 8.01, .25))
        right = field.sample_rect(np.arange(8, 16.01, .25), np.arange(0, 8.01, .25))
        np.testing.assert_allclose(whole[:, :33], left, atol=1e-12)
        np.testing.assert_allclose(whole[:, 32:], right, atol=1e-12)
        np.testing.assert_allclose(whole[:, 0], whole[:, -1], atol=1e-12)
        np.testing.assert_allclose(whole[2::4, 2::4], values, atol=1e-12)

    def test_numeric_gradient_changes_class_between_cell_centres(self):
        values = np.tile(np.arange(8, dtype=float)+.5, (6, 1))
        paths = scalar_band_paths(values, np.ones_like(values, dtype=bool), [3.2, 6.1])
        regions = [shapely.union_all(list(filled_geometries(band))) for band in paths]
        interior = shapely.box(.5, 0, 7.5, 6)
        self.assertAlmostEqual(regions[0].intersection(interior).bounds[2], 3.2, places=8)
        self.assertAlmostEqual(regions[1].intersection(interior).bounds[2], 6.1, places=8)
        rectangle = shapely.box(0, 0, 8, 6)
        self.assertLess(shapely.symmetric_difference(shapely.union_all(regions), rectangle).area, 1e-12)
        for first, second in zip(regions[:-1], regions[1:], strict=True):
            self.assertLess(first.intersection(second).area, 1e-12)

    def test_constant_and_all_water_fields_are_complete(self):
        values = np.full((4, 8), .5)
        paths = scalar_band_paths(values, np.ones_like(values, dtype=bool), [.25, .75])
        self.assertEqual([bool(band) for band in paths], [False, True, False])
        self.assertTrue(shapely.equals(shapely.union_all(list(filled_geometries(paths[1]))), shapely.box(0,0,8,4)))
        self.assertEqual(scalar_band_paths(values, np.zeros_like(values, dtype=bool), [.25, .75]), [[],[],[]])

    def test_exact_threshold_plateaus_belong_to_the_upper_class(self):
        values = np.full((4, 8), .5)
        paths = scalar_band_paths(values, np.ones_like(values, dtype=bool), [.5])
        self.assertEqual([bool(band) for band in paths], [False, True])
        self.assertTrue(shapely.equals(shapely.union_all(list(filled_geometries(paths[1]))),
                                       shapely.box(0, 0, 8, 4)))

    def test_paired_points_match_the_same_rectangular_surface_in_bounded_batches(self):
        values = np.random.default_rng(73).uniform(size=(9, 17))
        field = ContinuousScalarField(values, np.ones(values.shape, bool))
        x, y = np.linspace(0, 17, 129), np.linspace(0, 9, 129)
        sampled = field.sample_rect(x, y)
        np.testing.assert_allclose(field.sample_points(x[None, :], y[:, None]), sampled, atol=1e-13)
        np.testing.assert_array_equal(field.sample_points(np.arange(17)+.5, np.arange(9)[:,None]+.5), values)
        np.testing.assert_allclose(field.sample_points(x+17, 4.3), field.sample_points(x, 4.3), atol=1e-13)

    def test_barely_super_threshold_peaks_retain_their_true_pchip_footprint(self):
        for excess in (np.spacing(np.float32(.5)), .0002847909927368, .05):
            with self.subTest(excess=excess):
                values = np.full((7, 7), .04)
                values[3, 3] = .5+float(excess)
                land = np.ones(values.shape, bool)
                field = ContinuousScalarField(values, land)
                paths = scalar_band_paths(values, land, [.5])
                wet = shapely.union_all(list(filled_geometries(paths[1])))
                vertices = np.asarray(wet.exterior.coords)
                actual_root = brentq(lambda x: float(field.sample_points(x, 3.5))-.5,
                                     3.5, 4.5, xtol=1e-14)
                self.assertAlmostEqual(wet.bounds[2], actual_root, places=11)
                self.assertGreater(len(vertices), 16, "a true curved peak cannot be four linear mesh crossings")
                self.assertLess(float(abs(field.sample_points(vertices[:,0], vertices[:,1])-.5).max()), 1e-12)
                midpoints = (vertices[:-1]+vertices[1:])*.5
                self.assertLess(float(abs(field.sample_points(midpoints[:,0],midpoints[:,1])-.5).max()), float(excess)*.03)
                self.assertTrue(wet.covers(shapely.Point(3.5,3.5)))
                self.assertTrue(wet.is_valid)

    def test_nested_thresholds_share_complete_coverage_and_keep_the_basin_hole(self):
        rows, columns = np.indices((11, 17))
        values = np.hypot(rows-5, columns-8)
        paths = scalar_band_paths(values, np.ones(values.shape, bool), [2.3, 4.1, 6.2])
        regions = np.asarray([shapely.union_all(list(filled_geometries(band))) for band in paths])
        self.assertTrue(bool(shapely.coverage_is_valid(regions)))
        self.assertEqual(shapely.box(0,0,17,11).symmetric_difference(shapely.union_all(regions)).area, 0.)
        self.assertEqual(len(regions[1].interiors), 1)
        centres = shapely.points(columns.ravel()+.5, rows.ravel()+.5)
        expected = np.searchsorted([2.3,4.1,6.2], values.ravel(), side="right")
        for identifier, region in enumerate(regions):
            np.testing.assert_array_equal(shapely.covers(region, centres), expected == identifier)

    def test_periodic_threshold_components_meet_both_frame_edges_identically(self):
        values = np.full((7, 9), .1)
        values[2:5, [0,8]] = .8
        paths = scalar_band_paths(values, np.ones(values.shape, bool), [.5])
        wet = shapely.union_all(list(filled_geometries(paths[1])))
        left = shapely.intersection(wet.boundary, shapely.LineString(((0,0),(0,7))))
        right = shapely.intersection(wet.boundary, shapely.LineString(((9,0),(9,7))))
        first, last = shapely.get_coordinates(left), shapely.get_coordinates(right)
        np.testing.assert_allclose(np.sort(first[:,1]), np.sort(last[:,1]), atol=1e-12)
        self.assertTrue(wet.covers(shapely.Point(.5,3.5)))
        self.assertTrue(wet.covers(shapely.Point(8.5,3.5)))
        self.assertTrue(wet.is_valid)

    def test_native_checkerboards_and_threshold_junctions_have_shared_edges(self):
        for lower_right in (.8, .9):
            with self.subTest(lower_right=lower_right):
                values = np.full((5,7), .2)
                values[1,2], values[2,3] = .8, lower_right
                regions = np.asarray([shapely.union_all(list(filled_geometries(band)))
                                      for band in scalar_band_paths(values, np.ones(values.shape,bool), [.5])])
                self.assertTrue(bool(np.all(shapely.is_valid(regions))))
                self.assertTrue(bool(shapely.coverage_is_valid(regions)))
                self.assertEqual(regions[0].intersection(regions[1]).area, 0.)
                self.assertEqual(shapely.box(0,0,7,5).difference(shapely.union_all(regions)).area, 0.)
                rows,columns = np.indices(values.shape)
                np.testing.assert_array_equal(shapely.covers(regions[1],shapely.points(columns.ravel()+.5,rows.ravel()+.5)),
                                              values.ravel() >= .5)


if __name__ == '__main__':
    unittest.main()
