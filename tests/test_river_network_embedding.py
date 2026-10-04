"""Terrain-guided river motion preserves the source network embedding."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from collections import Counter

import numpy as np
import shapely

from world_atlas.core.river_courses import solve_river_courses
from world_atlas.core.continuous_terrain import PhysicalTerrainField


def field_fixture(values):
    raw = np.asarray(values, dtype=np.float64)
    grid = SimpleNamespace(shape=raw.shape, water=(raw <= 0).astype(np.uint8))
    field = PhysicalTerrainField(raw, land_mask=raw > 0, sea_level_m=0,
                                 elevation_scale_m=10000., elevation_exponent=1.)
    return grid, field


class RiverNetworkEmbeddingTests(unittest.TestCase):
    def test_actual_0069_confluence_retains_source_fan_and_has_no_extra_cycle(self):
        data = json.loads((Path(__file__).parent / 'fixtures/river-confluence-0069-v96.json').read_text())
        grid, field = field_fixture(data['ground'])
        paths = [np.asarray(points) for points in data['paths']]
        original = [shapely.LineString(path) for path in paths]
        old = [shapely.LineString(path) for path in data['oldIndependentDisplay']]
        junction = shapely.Point(paths[0][0])
        self.assertFalse(old[0].intersection(old[1]).equals(junction))

        curves = solve_river_courses(grid, field, source_paths=paths).paths
        delivered = [shapely.LineString(path) for path in curves]
        for first in range(len(paths)):
            self.assertTrue(delivered[first].is_simple)
            np.testing.assert_array_equal(curves[first][[0, -1]], paths[first][[0, -1]])
            for second in range(first):
                self.assertTrue(delivered[first].intersection(delivered[second]).equals(
                    original[first].intersection(original[second])))

    def test_intrinsic_nonadjacent_intersection_is_retained(self):
        rows, columns = np.indices((20, 20))
        grid, field = field_fixture(100 + 8 * (rows - 8) ** 2 + .4 * columns ** 2)
        path = np.array(((3.5, 3.5), (13.5, 13.5), (3.5, 13.5), (13.5, 3.5)))
        curve = solve_river_courses(grid, field, source_paths=(path,)).paths[0]
        self.assertFalse(shapely.LineString(path).is_simple)
        self.assertFalse(shapely.LineString(curve).is_simple)
        def intrinsic_nodes(points):
            degree = Counter()
            for line in shapely.get_parts(shapely.node(shapely.LineString(points))):
                degree.update((tuple(line.coords[0]), tuple(line.coords[-1])))
            return {point for point, count in degree.items() if count > 2}
        self.assertEqual(intrinsic_nodes(path), {(8.5, 8.5)})
        self.assertEqual(intrinsic_nodes(curve), intrinsic_nodes(path))

    def test_network_order_and_reach_reversal_do_not_change_geometry(self):
        rows, columns = np.indices((20, 20))
        grid, field = field_fixture(300 + 5 * (rows - 10) ** 2 + columns ** 2)
        paths = [np.array(((3.5, 10.5), (10.5, 10.5))),
                 np.array(((10.5, 3.5), (10.5, 10.5))),
                 np.array(((10.5, 10.5), (15.5, 14.5)))]
        curves = solve_river_courses(grid, field, source_paths=paths).paths
        reversed_curves = solve_river_courses(grid, field, source_paths=[p[::-1] for p in paths[::-1]]).paths
        for curve, reverse in zip(curves, reversed_curves[::-1], strict=True):
            np.testing.assert_allclose(curve, reverse[::-1], rtol=0, atol=1e-11)

    def test_distant_valley_motion_survives_the_shared_network_constraints(self):
        rows = np.arange(64) + .5
        grid, field = field_fixture(np.broadcast_to(200 + 40 * (rows[:, None] - 15.5) ** 2, (64, 64)))
        paths = [np.array(((3.5, 15.8), (27.5, 15.8))),
                 np.array(((3.5, 25.5), (27.5, 25.5)))]
        curves = solve_river_courses(grid, field, source_paths=paths).paths
        self.assertGreater(float(np.max(15.8 - curves[0][1:-1, 1])), .01)
        self.assertTrue(shapely.LineString(curves[0]).disjoint(shapely.LineString(curves[1])))


if __name__ == '__main__':
    unittest.main()
