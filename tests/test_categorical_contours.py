"""Shared contour reconstruction must preserve observations and material topology."""

import unittest

import numpy as np
import shapely
from scipy import ndimage

from world_atlas.core.raster_topology import categorical_coverage, _indicator_scores


class CategoricalContourTests(unittest.TestCase):
    def assert_native_ownership(self, values, active, faces, labels):
        row, column = np.nonzero(active)
        samples = shapely.points(column+.5, row+.5)
        owners = np.full(len(samples), -1, dtype=np.int32)
        first, last = shapely.STRtree(faces).query(samples, predicate="within")
        self.assertEqual(len(first), len(samples), "each native observation has exactly one interior owner")
        owners[first] = labels[last]
        np.testing.assert_array_equal(owners, values[row, column])
        row, column = np.nonzero(~active)
        self.assertFalse(bool(np.any(shapely.intersects_xy(shapely.union_all(faces), column+.5, row+.5))))

    def test_three_material_junctions_share_exact_edges_and_fill_the_frame(self):
        values = np.zeros((17, 21), dtype=np.int16)
        values[3:14, 6:16] = 5
        values[8:, 11:] = 11
        values[2:5, 17:20] = 5
        faces, labels = categorical_coverage(values, np.ones(values.shape, bool), category_count=12)
        self.assertTrue(shapely.coverage_is_valid(faces))
        self.assertTrue(shapely.union_all(faces).equals(shapely.box(0, 0, 21, 17)))
        self.assertEqual(set(labels), {0, 5, 11})
        self.assert_native_ownership(values, np.ones(values.shape, bool), faces, labels)

    def test_isolated_material_is_reconstructed_from_samples_without_a_diamond(self):
        values = np.zeros((9, 9), dtype=np.int16)
        values[4, 4] = 5
        faces, labels = categorical_coverage(values, np.ones(values.shape, bool), category_count=6)
        island = faces[int(np.flatnonzero(labels == 5)[0])]
        self.assertTrue(island.is_valid)
        self.assertTrue(island.contains(shapely.Point(4.5, 4.5)))
        edges = np.diff(np.asarray(island.exterior.coords), axis=0)
        self.assertTrue(bool(np.any((np.abs(edges[:, 0]) > 1e-6)
                                   & (np.abs(edges[:, 1]) > 1e-6)
                                   & (np.abs(np.abs(edges[:, 0])-np.abs(edges[:, 1])) > 1e-6))))
        self.assertGreater(len(edges), 16)
        self.assert_native_ownership(values, np.ones(values.shape, bool), faces, labels)

    def test_holes_and_thin_material_connections_keep_every_native_owner(self):
        values = np.zeros((13, 18), dtype=np.int16)
        values[6, 2:15] = 3
        values[2:11, 2] = 3
        active = np.ones(values.shape, dtype=bool)
        active[3:5, 8:10] = False
        active[9, 13] = False
        faces, labels = categorical_coverage(values, active, category_count=4)
        self.assertTrue(shapely.coverage_is_valid(faces))
        self.assert_native_ownership(values, active, faces, labels)

    def test_periodic_support_and_tile_boundaries_do_not_open_cracks(self):
        row, column = np.mgrid[:39, :71]
        values = np.where(((column+7*np.sin(row/5)) % 71) < 29, 4, 9).astype(np.int16)
        faces, labels = categorical_coverage(values, np.ones(values.shape, bool), category_count=10)
        self.assertTrue(shapely.coverage_is_valid(faces))
        self.assertTrue(shapely.union_all(faces).equals(shapely.box(0, 0, 71, 39)))
        self.assert_native_ownership(values, np.ones(values.shape, bool), faces, labels)
        identifiers, scores = _indicator_scores(values, np.array((0., 71.)), np.array((.25, 12., 38.75)))
        np.testing.assert_array_equal(scores[:, :, 0], scores[:, :, 1])

    def test_indicator_reconstruction_interpolates_and_partitions_without_numeric_id_blending(self):
        values = np.array(((1, 1001, 1), (1001, 1, 1001)), dtype=np.int16)
        identifiers, scores = _indicator_scores(values, np.arange(3)+.5, np.arange(2)+.5)
        np.testing.assert_array_equal(identifiers[np.argmax(scores, axis=0)], values)
        np.testing.assert_allclose(scores.sum(axis=0), 1.)
        identifiers, scores = _indicator_scores(values, np.arange(13)/4, np.arange(9)/4)
        np.testing.assert_allclose(scores.sum(axis=0), 1., atol=1e-14)
        self.assertEqual(set(identifiers), {1, 1001})

    def test_non_square_and_single_cell_frames_remain_complete(self):
        for shape in ((1, 1), (1, 9), (8, 1)):
            values = np.full(shape, 7, dtype=np.int16)
            faces, labels = categorical_coverage(values, np.ones(shape, bool), category_count=8)
            self.assertEqual(labels.tolist(), [7])
            self.assertTrue(shapely.union_all(faces).equals(shapely.box(0, 0, shape[1], shape[0])))

    def test_crowded_materials_cannot_disconnect_the_shared_contour_graph(self):
        # Losers at all triangle vertices can still own an interval of a
        # shared edge. Checking only corner winners breaks this actual case.
        values = np.random.default_rng(7).integers(0, 4, (10, 10), dtype=np.int16)
        active = np.ones(values.shape, dtype=bool)
        faces, labels = categorical_coverage(values, active, category_count=4)
        self.assertTrue(shapely.coverage_is_valid(faces))
        self.assertTrue(shapely.union_all(faces).equals(shapely.box(0, 0, 10, 10)))
        self.assert_native_ownership(values, active, faces, labels)

    def test_opposite_triangle_edge_directions_share_half_grid_intersections(self):
        generator = np.random.default_rng(1)
        for _ in range(4):
            values = np.argmax(np.stack([
                ndimage.gaussian_filter(generator.random((40, 49)), 1)
                for _ in range(8)
            ]), axis=0).astype(np.int16)
        active = np.ones(values.shape, dtype=bool)
        faces, labels = categorical_coverage(values, active, category_count=8)
        self.assertTrue(shapely.coverage_is_valid(faces))
        self.assert_native_ownership(values, active, faces, labels)

    def test_shared_delivery_coordinates_do_not_snap_the_physical_shoreline(self):
        values = np.zeros((8, 12), dtype=np.int16)
        values[:, 6:] = 3
        faces, labels = categorical_coverage(values, np.ones(values.shape, bool), category_count=4)
        physical = shapely.Polygon(((1.123456789123, 1.987654321123),
                                    (10.123456789123, 2.987654321123),
                                    (9.123456789123, 6.987654321123),
                                    (1.123456789123, 5.987654321123)))
        self.assertTrue(all(shapely.get_precision(face) == 0 for face in faces))
        visible = shapely.union_all([shapely.intersection(face, physical) for face in faces])
        self.assertLess(shapely.difference(physical, visible).area, 1e-12)
        self.assertLess(shapely.difference(visible, physical).area, 1e-12)


if __name__ == "__main__":
    unittest.main()
