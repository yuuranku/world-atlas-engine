import os
from pathlib import Path
import unittest
import numpy as np
import shapely

from world_atlas.core.render import (
    _categorical_overlay_pixels,
    _mapshaper_smooth_coverage,
)


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

    def test_shared_geometry_runs_without_original_project_directory(self):
        entry = Path(__file__).resolve().parents[1] / 'runtime/node_modules/mapshaper/bin/mapshaper'
        if not entry.is_file():
            self.skipTest('install runtime npm dependencies to exercise Mapshaper')
        old = os.environ.get('WORLD_ATLAS_MAPSHAPER')
        os.environ['WORLD_ATLAS_MAPSHAPER'] = str(entry)
        try:
            faces, labels = _mapshaper_smooth_coverage([shapely.box(0,0,10,10), shapely.box(10,0,20,10)], [1,2], distance=.2)
            self.assertEqual(set(labels), {1,2})
            self.assertTrue(shapely.coverage_is_valid(faces))
        finally:
            if old is None:
                os.environ.pop('WORLD_ATLAS_MAPSHAPER', None)
            else:
                os.environ['WORLD_ATLAS_MAPSHAPER'] = old
