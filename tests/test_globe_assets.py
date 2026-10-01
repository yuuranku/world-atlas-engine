from copy import deepcopy
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import shapely

from world_atlas.core.globe_assets import THEME_EXPORT_FILENAMES, globe_theme_documents
from world_atlas.core.render import (
    _filled_path_data, _frontier_overlay, _geometry_filled_paths,
    _line_overlay_svg_document, _partition_boundary_overlay,
    _partition_overlay_svg_document,
)


class GlobeAssetTests(unittest.TestCase):
    def setUp(self):
        self.grid = SimpleNamespace(shape=(8, 16))
        polygon = shapely.Polygon(((1.12345678, 1), (9, 1), (9, 7), (1.12345678, 7)),
                                  holes=[((3, 3), (4, 3), (4, 4), (3, 4))])
        self.land = shapely.MultiPolygon((polygon, shapely.box(12, 2, 14, 4)))
        self.paths = _geometry_filled_paths(self.land)
        self.base = _line_overlay_svg_document(self.grid,
            '<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
            + ''.join(f'<path d="{_filled_path_data(points, codes)}" '
                      'fill-rule="evenodd" clip-rule="evenodd" />' for points, codes in self.paths)
            + '</clipPath></defs>', title='base')
        self.full = _partition_overlay_svg_document(self.grid, [self.paths],
            land_surface=self.land, title='current full-quality export',
            partition_id='test-partition', data_attribute='test-band',
            zones=(('zone', '#123456'),), extra_overlay=_frontier_overlay(self.paths),
            fill_opacity=.91)
        self.themes = {key: self.full for key in THEME_EXPORT_FILENAMES}
        ink = _partition_boundary_overlay('state-boundaries', [shapely.get_coordinates(
            shapely.LineString(((5, 1), (5, 2), (5, 7))))], color='#46413b', width=.4)
        self.boundaries = {'political': ink, 'provinces': ink}

    def test_production_export_keeps_all_paint_and_only_base_owns_the_clip(self):
        documents = globe_theme_documents(self.base, self.themes, boundary_bodies=self.boundaries)
        expected_paint = ET.fromstring(self.full)[1]
        expected_ink = ET.fromstring('<svg xmlns="http://www.w3.org/2000/svg">'
                                   + self.boundaries['political'] + '</svg>')[0]
        self.assertEqual(len(documents), 13)
        for theme, document in documents.items():
            root = ET.fromstring(document)
            self.assertEqual(root.attrib['viewBox'], '0 0 16 8')
            self.assertFalse(root.findall('.//{*}svg'))
            self.assertFalse(root.findall('.//{*}clipPath'))
            composed = ET.fromstring(self.base)
            composed.append(deepcopy(root))
            self.assertEqual(len(composed.findall('.//{*}clipPath[@id="land-silhouette-clip"]')), 1)
            if theme == 'terrain':
                self.assertEqual(len(root), 1)
            else:
                self.assertEqual(ET.tostring(root[1]), ET.tostring(expected_paint))
                self.assertEqual([node.attrib for node in root[1].iter()],
                                 [node.attrib for node in expected_paint.iter()])
                # Frontier hatching retains its paint-server definition.
                self.assertEqual(len(root.findall('.//{*}pattern[@id="frontier-hatch"]')), 1)
            if theme in self.boundaries:
                self.assertEqual(ET.tostring(root[2]), ET.tostring(expected_ink))

    def test_rejects_duplicate_or_different_authoritative_base_clip(self):
        base = ET.fromstring(self.base)
        definitions = base.find('{*}defs')
        definitions.append(deepcopy(definitions[0]))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            globe_theme_documents(ET.tostring(base, encoding='unicode'), self.themes,
                                  boundary_bodies=self.boundaries)
        with self.assertRaisesRegex(ValueError, 'exact base physical clip'):
            globe_theme_documents(self.base.replace('1.12345678', '1.12345679'), self.themes,
                                  boundary_bodies=self.boundaries)

    def test_rejects_unowned_paint_or_partial_current_theme_sets(self):
        with self.assertRaisesRegex(ValueError, 'all twelve'):
            globe_theme_documents(self.base, {'climate': self.full}, boundary_bodies=self.boundaries)
        broken = dict(self.themes)
        broken['climate'] = self.full.replace('data-surface-contract="shared-land-clip"', '')
        with self.assertRaisesRegex(ValueError, 'exact base physical clip'):
            globe_theme_documents(self.base, broken, boundary_bodies=self.boundaries)

    def test_rejects_boundary_definitions_and_different_world_dimensions(self):
        with self.assertRaisesRegex(ValueError, 'only source vector ink'):
            globe_theme_documents(self.base, self.themes,
                                  boundary_bodies={'political': '<defs/>', 'provinces': ''})
        broken = dict(self.themes)
        broken['population'] = self.full.replace('viewBox="0 0 16 8"', 'viewBox="0 0 8 4"')
        with self.assertRaisesRegex(ValueError, 'exact base physical clip'):
            globe_theme_documents(self.base, broken, boundary_bodies=self.boundaries)


if __name__ == '__main__':
    unittest.main()
