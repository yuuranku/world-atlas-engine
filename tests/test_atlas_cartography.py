import unittest
import numpy as np
from shapely import LineString, Point

from world_atlas.core.cartographic_colors import POLITICAL_PALETTE, area_colors
from world_atlas.core.render import _river_path_data
from world_atlas.core.society.transport import _terrain_safe_simplification


class AtlasCartographyTests(unittest.TestCase):
    def test_all_territory_colors_remain_distinguishable_when_palette_is_reused(self):
        colors = np.array([
            [int(color[index:index + 2], 16) for index in (1, 3, 5)]
            for color in POLITICAL_PALETTE
        ])
        for index, first in enumerate(colors):
            for second in colors[index + 1:]:
                self.assertGreater(np.linalg.norm(first - second), 65)

    def test_neighbouring_countries_have_contrast_and_seam_is_included(self):
        values = np.repeat(np.arange(1, 81, dtype=np.int32).reshape(8, 10), 3, axis=0)
        colors = area_colors(values, range(1, 81))
        self.assertEqual(colors, area_colors(values, range(1, 81)))
        self.assertGreaterEqual(len(set(colors.values())), 8)
        for first, second in ((values[:, :-1], values[:, 1:]), (values[:-1], values[1:]),
                              (values[:, -1:], values[:, :1])):
            for left, right in zip(first.ravel(), second.ravel()):
                if left != right:
                    self.assertNotEqual(colors[int(left)], colors[int(right)])
                    a = np.array([int(colors[int(left)][i:i+2], 16) for i in (1, 3, 5)])
                    b = np.array([int(colors[int(right)][i:i+2], 16) for i in (1, 3, 5)])
                    self.assertGreater(np.linalg.norm(a-b), 65)

    def test_road_simplification_preserves_valley_bends_even_on_equal_friction(self):
        path = ((2, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 5),
                (6, 6), (6, 7), (6, 8), (5, 9), (4, 9), (3, 9), (2, 10))
        simplified = _terrain_safe_simplification(path, np.ones((16, 16)), np.ones((16, 16), dtype=bool))
        line = LineString(simplified)
        self.assertLessEqual(max(Point(p).distance(line) for p in path), .35 + 1e-9)
        self.assertGreater(len(simplified), 4)
        self.assertEqual((simplified[0], simplified[-1]), (path[0], path[-1]))

    def test_river_label_curve_preserves_reach_endpoints(self):
        path = np.array([[2.5, 2.5], [5.5, 2.5], [5.5, 6.5]])
        river = _river_path_data(path)
        self.assertTrue(river.startswith('M2.5,2.5'))
        self.assertTrue(river.endswith('5.5,6.5'))


if __name__ == '__main__':
    unittest.main()
