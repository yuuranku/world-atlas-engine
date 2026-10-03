"""Parallel batches retain the exact serial curves and query domain."""
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_terrain_refinement import fixture
from world_atlas.core.implicit_terrain import (
    _cell_boxes, _query_cells, _refined_curves, _selected_ranges, terrain_level_curves,
)
from world_atlas.core.implicit_terrain_parallel import parallel_height_curves


class ParallelTerrainTests(unittest.TestCase):
    def test_spawned_height_batches_retain_order_empty_levels_and_exact_vertices(self):
        y, x = np.indices((9,11))
        _, field = fixture(600 + x*80 + y*30)
        query = np.array(((3.,2.,7.,6.),))
        cells = _query_cells(field, query)
        lower, upper = _cell_boxes(field, cells)
        low, high = _selected_ranges(field, cells)
        levels = np.array((100.,900.,1100.,1800.))
        expected = _refined_curves(field, levels, lower, upper, low, high)
        actual = parallel_height_curves(field, levels, lower, upper, low, high)
        self.assertEqual(len(expected), len(actual))
        self.assertEqual(actual[0], [])
        self.assertEqual(actual[-1], [])
        self.assertTrue(actual[1])
        for wanted, generated in zip(expected, actual, strict=True):
            self.assertEqual(len(wanted), len(generated))
            for first, last in zip(wanted, generated, strict=True):
                np.testing.assert_array_equal(first.view(np.uint64), last.view(np.uint64))

    def test_large_local_query_uses_the_same_parallel_route_as_global_queries(self):
        y, x = np.indices((48,48))
        _, field = fixture(600 + x*8 + y*3)
        query = np.array(((0.,0.,48.,48.),))
        with patch('world_atlas.core.implicit_terrain_parallel.parallel_height_curves',
                   return_value=['complete source graphs']) as run:
            actual = terrain_level_curves(field, [700.,900.], query_bounds=query)
        self.assertEqual(actual, ['complete source graphs'])
        source, levels, lower, upper, low, high = run.call_args.args
        self.assertIs(source, field)
        self.assertGreater(len(lower), 2048)
        np.testing.assert_array_equal(levels, (700.,900.))
        cells = _query_cells(field, query)
        np.testing.assert_array_equal(lower, _cell_boxes(field, cells)[0])
        np.testing.assert_array_equal(upper, _cell_boxes(field, cells)[1])
        np.testing.assert_array_equal(low, _selected_ranges(field, cells)[0])
        np.testing.assert_array_equal(high, _selected_ranges(field, cells)[1])


if __name__ == '__main__':
    unittest.main()
