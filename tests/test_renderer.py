import os
from pathlib import Path
import unittest
import numpy as np
import shapely

from world_atlas.core.render import _mapshaper_smooth_coverage


class RendererTests(unittest.TestCase):
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
