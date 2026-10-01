import unittest

import numpy as np
import shapely

from world_atlas.core.cartographic_generalization import _native_safe_chain, generalize_display_surface


class CartographicGeneralizationTests(unittest.TestCase):
    def assert_native_ownership(self, source, result, shape):
        rows, columns = np.meshgrid(np.arange(shape[0]) + .5, np.arange(shape[1]) + .5, indexing="ij")
        np.testing.assert_array_equal(
            shapely.contains_xy(source, columns, rows), shapely.contains_xy(result, columns, rows),
        )

    def test_dp_chord_cannot_remove_a_native_centre_in_its_sweep(self):
        source = shapely.Polygon([(3, 3), (5.48, 3), (5.48, 5), (5.54, 5.5), (5.48, 6), (3, 6)])
        before = shapely.to_wkb(source)
        result = generalize_display_surface(source, frame_shape=(10, 10), tolerance=.08)
        self.assertTrue(shapely.contains_xy(result, 5.5, 5.5))
        self.assertFalse(shapely.contains_xy(shapely.simplify(source, .08, preserve_topology=True), 5.5, 5.5))
        self.assertIn((5.54, 5.5), result.exterior.coords)
        self.assert_native_ownership(source, result, (10, 10))
        self.assertEqual(shapely.to_wkb(source), before)

    def test_one_native_sensitive_peak_does_not_restore_the_entire_dense_arc(self):
        yy = np.linspace(1, 9, 257)
        xx = 5.48 + .001*np.sin(yy*3)
        xx[np.argmin(abs(yy-5.5))] = 5.54
        chain = np.column_stack((xx, yy))
        source = shapely.Polygon([(3, 1), *chain, (3, 9)])
        result = generalize_display_surface(source, frame_shape=(12, 12), tolerance=.08)
        self.assertTrue(result.is_valid)
        self.assertTrue(result.contains(shapely.Point(5.5, 5.5)))
        self.assertLess(shapely.get_num_coordinates(result), 20)
        self.assert_native_ownership(source, result, (12, 12))
        self.assertTrue(set(result.exterior.coords).issubset(set(source.exterior.coords)))
        forward = _native_safe_chain(chain, frame_shape=(12, 12))
        backward = _native_safe_chain(chain[::-1], frame_shape=(12, 12))
        np.testing.assert_array_equal(forward, backward[::-1])

    def test_equal_deviation_shared_chord_is_canonical_in_both_directions(self):
        chain = np.asarray([(2.48, 1), (2.54, 1.5), (2.48, 2),
                            (2.54, 2.5), (2.48, 3)])
        forward = _native_safe_chain(chain, frame_shape=(4, 4))
        backward = _native_safe_chain(chain[::-1], frame_shape=(4, 4))
        np.testing.assert_array_equal(forward, backward[::-1])
        left = shapely.Polygon([(0, 1), *chain, (0, 3)])
        summarized = shapely.Polygon([(0, 1), *forward, (0, 3)])
        self.assert_native_ownership(left, summarized, (4, 4))

    def test_safe_subchain_is_reduced_using_only_original_vertices(self):
        source = shapely.Polygon([(1, 1), (3, 1.03), (5, 1), (5, 5), (1, 5)])
        result = generalize_display_surface(source, frame_shape=(8, 8), tolerance=.08)
        self.assertLess(shapely.get_num_coordinates(result), shapely.get_num_coordinates(source))
        self.assertTrue(set(result.exterior.coords).issubset(set(source.exterior.coords)))
        self.assert_native_ownership(source, result, (8, 8))

    def test_frame_segments_and_periodic_intersections_are_exact(self):
        source = shapely.MultiPolygon([shapely.box(0, 1.25, 2.3, 7.75), shapely.box(29.7, 1.25, 32, 7.75)])
        result = generalize_display_surface(source, frame_shape=(10, 32), tolerance=.12)
        for x in (0, 32):
            cut = shapely.LineString([(x, 0), (x, 10)])
            self.assertTrue(shapely.equals(source.intersection(cut), result.intersection(cut)))
        self.assert_native_ownership(source, result, (10, 32))
        frame = shapely.box(0, 0, 32, 10)
        self.assertTrue(shapely.equals(frame, generalize_display_surface(frame, frame_shape=(10, 32), tolerance=.12)))

    def test_single_cell_island_bridge_and_water_channel_stay_distinct(self):
        bridge = shapely.union_all([shapely.box(2, 3, 8, 13), shapely.box(12, 3, 18, 13), shapely.box(8, 7, 12, 8)])
        result = generalize_display_surface(bridge, frame_shape=(16, 24), tolerance=.12)
        self.assertEqual(result.geom_type, "Polygon")
        self.assertTrue(result.covers(shapely.LineString([(6.5, 7.5), (14.5, 7.5)])))
        self.assert_native_ownership(bridge, result, (16, 24))
        source = shapely.MultiPolygon([shapely.box(0, 0, 8, 16), shapely.box(9, 0, 16, 16), shapely.box(20, 6, 21, 7)])
        result = generalize_display_surface(source, frame_shape=(16, 24), tolerance=.12)
        self.assertEqual(len(result.geoms), 3)
        self.assertTrue(result.disjoint(shapely.LineString([(8.5, 0), (8.5, 16)])))
        self.assertTrue(shapely.contains_xy(result, 20.5, 6.5))
        self.assert_native_ownership(source, result, (16, 24))

    def test_lake_hole_is_retained(self):
        source = shapely.Polygon(shapely.box(1, 1, 15, 15).exterior, [shapely.box(6, 6, 10, 10).exterior])
        result = generalize_display_surface(source, frame_shape=(16, 16), tolerance=.12)
        self.assertEqual(len(result.interiors), 1)
        self.assertFalse(shapely.contains_xy(result, 8.5, 8.5))
        self.assert_native_ownership(source, result, (16, 16))


if __name__ == "__main__":
    unittest.main()
