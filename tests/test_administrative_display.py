import os
import unittest

import numpy as np
import shapely

from world_atlas.core.society.administrative_display import administrative_display_coverage


def partition(frame, *lines):
    graph = shapely.union_all((frame.boundary, *lines))
    return shapely.get_parts(shapely.polygonize(shapely.get_parts(graph)))


@unittest.skipUnless(os.environ.get('WORLD_ATLAS_MAPSHAPER'), 'locked Mapshaper runtime required')
class AdministrativeDisplayTests(unittest.TestCase):
    def assert_preserved(self, source, result, frame_shape):
        self.assertEqual(len(source), len(result))
        self.assertTrue(np.all(shapely.is_valid(result)))
        self.assertTrue(shapely.coverage_is_valid(result))
        self.assertTrue(shapely.coverage_union_all(source[~shapely.is_empty(source)]).equals(
            shapely.coverage_union_all(result[~shapely.is_empty(result)])))
        height, width = frame_shape
        west, north, east, south = shapely.coverage_union_all(source[~shapely.is_empty(source)]).bounds
        yy, xx = np.meshgrid(np.arange(max(0, int(np.floor(north))), min(height, int(np.ceil(south)))),
                             np.arange(max(0, int(np.floor(west))), min(width, int(np.ceil(east)))), indexing='ij')
        centres = shapely.points(np.column_stack((xx.ravel()+.5, yy.ravel()+.5)))
        for before, after in zip(source, result, strict=True):
            np.testing.assert_array_equal(shapely.covers(before, centres), shapely.covers(after, centres))
            first, last = shapely.get_parts(before), shapely.get_parts(after)
            self.assertEqual(len(first), len(last))
            self.assertEqual(sorted(len(part.interiors) for part in first),
                             sorted(len(part.interiors) for part in last))

    def test_rounds_actual_grid_turn_on_the_shared_fill_boundary(self):
        border = shapely.LineString(((0, 1.9), (2, 1.9), (2, 2.9), (3, 2.9),
                                     (3, 3.9), (6, 3.9)))
        source = partition(shapely.box(0, 0, 6, 6), border)
        result = administrative_display_coverage(source, frame_shape=(6, 6))
        self.assert_preserved(source, result, (6, 6))
        shared = result[0].boundary.intersection(result[1].boundary)
        self.assertFalse(border.equals(shared))
        self.assertLessEqual(border.hausdorff_distance(shared), .35000001)
        # The original one-unit shelf acquires a varying tangent. Extra
        # vertices on the same horizontal segment would fail this assertion.
        shelf = shared.intersection(shapely.box(2, 2.5, 3, 3.3))
        points = shapely.get_coordinates(shelf)
        self.assertGreater(np.ptp(points[:, 1]), .2)

    def test_three_way_junction_and_exterior_are_fixed(self):
        junction = (4.2, 4.1)
        lines = (shapely.LineString(((0, 2.1), (2.2, 2.1), (2.2, 3.1), junction)),
                 shapely.LineString((junction, (5.2, 5.1), (5.2, 8))),
                 shapely.LineString((junction, (6.2, 3.1), (8, 3.1))))
        source = partition(shapely.box(0, 0, 8, 8), *lines)
        result = administrative_display_coverage(source, frame_shape=(8, 8))
        self.assert_preserved(source, result, (8, 8))
        for face in result:
            self.assertTrue(face.boundary.covers(shapely.Point(junction)))

    def test_narrow_adjacent_arcs_keep_their_corridor(self):
        first = np.asarray(((2.9, 0), (2.9, 1.9), (3.9, 1.9),
                            (3.9, 3.9), (2.9, 3.9), (2.9, 8)))
        second = first+(0.08, 0)
        source = partition(shapely.box(0, 0, 8, 8), shapely.LineString(first), shapely.LineString(second))
        result = administrative_display_coverage(source, frame_shape=(8, 8))
        self.assert_preserved(source, result, (8, 8))
        self.assertGreater(min(face.area for face in result), 0)

    def test_small_enclaves_and_multiple_components_survive(self):
        tiny = shapely.Polygon(((2.08, 2.06), (2.3, 2.06), (2.3, 2.18),
                                (2.22, 2.18), (2.22, 2.3), (2.08, 2.3)))
        island = shapely.box(5.08, 5.08, 5.32, 5.3)
        owned = shapely.MultiPolygon((tiny, island))
        source = np.asarray((shapely.box(0, 0, 8, 8).difference(owned), owned), dtype=object)
        result = administrative_display_coverage(source, frame_shape=(8, 8))
        self.assert_preserved(source, result, (8, 8))

    def test_constraint_is_local_on_a_long_arc(self):
        border = shapely.LineString(((0, 1.9), (1.9, 1.9), (1.9, 2.5), (2.5, 2.5),
                                     (3.9, 2.5), (3.9, 3.9), (6.9, 3.9),
                                     (6.9, 5.9), (10, 5.9)))
        source = partition(shapely.box(0, 0, 10, 8), border)
        result = administrative_display_coverage(source, frame_shape=(8, 10))
        self.assert_preserved(source, result, (8, 10))
        shared = result[0].boundary.intersection(result[1].boundary)
        self.assertTrue(shared.covers(shapely.Point(2.5, 2.5)))
        unconstrained = shapely.box(5, 3, 9, 7)
        self.assertFalse(border.intersection(unconstrained).equals(shared.intersection(unconstrained)))

    def test_continuous_competitive_front_retains_all_measured_owners(self):
        from world_atlas.core.society.administrative_front import administrative_front
        from world_atlas.core.society.administrative_coverage import administrative_coverage
        from world_atlas.core.society.territorial_simulation import TerritorySeed
        from tests.test_administrative_front import simulation
        front = administrative_front(simulation((19, 37)),
            (TerritorySeed(3, 4, 1), TerritorySeed(12, 16, 2), TerritorySeed(7, 29, 3)))
        source, _ = administrative_coverage(front)
        from world_atlas.core.cartographic_features import shared_display_coverage
        source = shared_display_coverage(source)
        result = administrative_display_coverage(source, frame_shape=front.valid.shape)
        self.assert_preserved(source, result, front.valid.shape)

    def test_empty_face_records_keep_their_original_positions_and_types(self):
        source = np.asarray((shapely.GeometryCollection(), shapely.box(0, 0, 3, 6),
                             shapely.Polygon(), shapely.box(3, 0, 6, 6),
                             shapely.GeometryCollection()), dtype=object)
        result = administrative_display_coverage(source, frame_shape=(6, 6))
        self.assert_preserved(source, result, (6, 6))
        for index in (0, 2, 4):
            self.assertEqual(result[index].geom_type, source[index].geom_type)
            self.assertTrue(result[index].is_empty)

    def test_microscopic_nonempty_enclaves_cannot_collapse(self):
        for size in (1e-6, 1e-8):
            with self.subTest(size=size):
                enclave = shapely.box(2.08, 2.08, 2.08+size, 2.08+size)
                source = np.asarray((shapely.box(0, 0, 6, 6).difference(enclave), enclave), dtype=object)
                result = administrative_display_coverage(source, frame_shape=(6, 6))
                self.assert_preserved(source, result, (6, 6))
                self.assertGreater(result[1].area, 0)

    def test_thin_long_and_world_coordinate_enclaves_survive(self):
        examples = ((shapely.box(2.08, 2.08, 4.08, 2.08000001),
                     shapely.box(0, 0, 6, 6), (6, 6)),
                    (shapely.box(1120.08, 805.08, 1120.08000001, 805.08000001),
                     shapely.box(1117, 803, 1123, 809), (1088, 2176)))
        for enclave, frame, frame_shape in examples:
            with self.subTest(enclave=enclave.bounds):
                source = np.asarray((frame.difference(enclave), enclave), dtype=object)
                result = administrative_display_coverage(source, frame_shape=frame_shape)
                self.assert_preserved(source, result, frame_shape)
                self.assertGreater(result[1].area, 0)


if __name__ == '__main__':
    unittest.main()
