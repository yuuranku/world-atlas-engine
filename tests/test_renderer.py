import unittest
import numpy as np
import shapely

from world_atlas.core.render import (
    _categorical_overlay_pixels,
)
from world_atlas.core.raster_topology import categorical_coverage


class RendererTests(unittest.TestCase):
    def test_population_overlay_keeps_native_cells_in_compact_rgba(self):
        values = np.array([[0, 1], [2, 1]], dtype=np.uint8)
        active = np.array([[False, True], [True, True]])
        pixels = _categorical_overlay_pixels(
            values,
            active,
            (("empty", "#000000"), ("one", "#112233"), ("two", "#abcdef")),
            opacity=0.5,
        )
        self.assertEqual(pixels.shape, (2, 2, 4))
        self.assertEqual(tuple(pixels[0, 0]), (0, 0, 0, 0))
        self.assertEqual(tuple(pixels[0, 1]), (0x11, 0x22, 0x33, 128))
        self.assertEqual(tuple(pixels[1, 0]), (0xAB, 0xCD, 0xEF, 128))

    def test_category_junction_has_one_shared_coverage_and_native_owners(self):
        categories = np.full((12, 20), 1, dtype=np.int16)
        categories[:, 10:] = 2
        categories[6:, :10] = 3
        original = categories.copy()
        faces, labels = categorical_coverage(
            categories, np.ones(categories.shape, dtype=bool), category_count=4,
        )
        self.assertEqual(set(labels), {1, 2, 3})
        self.assertTrue(shapely.coverage_is_valid(faces))
        self.assertLess(shapely.symmetric_difference(shapely.union_all(faces),
                                                    shapely.box(0, 0, 20, 12)).area, 1e-10)
        for row, column in np.ndindex(categories.shape):
            point = shapely.Point(column+.5, row+.5)
            owners = [int(label) for face, label in zip(faces, labels, strict=True)
                      if face.contains(point)]
            self.assertEqual(owners, [int(categories[row, column])])
        by_label = {label: shapely.union_all([face for face, owner in zip(faces, labels, strict=True)
                                             if owner == label]) for label in set(labels)}
        for first, second in ((1, 2), (1, 3), (2, 3)):
            shared = by_label[first].boundary.intersection(by_label[second].boundary)
            self.assertGreater(shared.length, 0)
            self.assertLess(by_label[first].intersection(by_label[second]).area, 1e-12)
        np.testing.assert_array_equal(categories, original)
