import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import shapely

from world_atlas.core.cartographic_tiles import geometry_path_data
from world_atlas.core.render import _bridge_overlay

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location('bridge_int_decoder', _ROOT/'scripts/reencode_globe_paths.py')
_DECODER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DECODER)


class BridgePresentationTests(unittest.TestCase):
    def test_actual_decks_use_one_global_integer_encoding_with_exact_translation_cancellation(self):
        records = json.loads((_ROOT/'tests/fixtures/bridge-native-decks-v97.json').read_text(encoding='utf-8'))['bridges']
        for record in records:
            with self.subTest(bridge=record['identifier']):
                bridge = SimpleNamespace(identifier=record['identifier'], route_identifier=record['routeIdentifier'], importance='regional', river_order=2)
                prepared = SimpleNamespace(bridges=(bridge,), positions={bridge.identifier:record['position']}, tangents={bridge.identifier:record['tangent']}, spans={bridge.identifier:record['bankSpan']})
                node = ET.fromstring(_bridge_overlay(prepared)).find('g')
                x = node.get('data-map-x'); y = node.get('data-map-y')
                self.assertEqual(node.get('transform'), f'translate({x} {y})')
                close = node.find("g[@class='bridge-close']")
                self.assertEqual(close.get('transform'), f'translate(-{x} -{y})')
                span = shapely.from_geojson(json.dumps(record['bankSpan']['geometry']))
                paths = close.findall('path')
                self.assertEqual(len(paths),3)
                self.assertEqual([path.get('stroke-width') for path in paths],['3.2','2.2','1.0'])
                expected = []
                for part in shapely.get_parts(span):
                    points = []
                    for point in part.coords:
                        integer = tuple(int(round(float(value)*100_000_000)) for value in point)
                        if not points or integer != points[-1]:
                            points.append(integer)
                    expected.append(points)
                for path in paths:
                    self.assertEqual(path.get('vector-effect'),'non-scaling-stroke')
                    self.assertIsNone(path.get('transform'))
                    self.assertEqual(path.get('d'),geometry_path_data(span))
                    decoded = list(_DECODER.linear_subpaths(path.get('d')))
                    self.assertTrue(all(not closed for _,closed in decoded))
                    self.assertEqual([points for points,_ in decoded],expected)
                self.assertEqual(node.find("g[@class='bridge-far']").get('transform'),f"rotate({node.get('data-base-angle')})")


if __name__ == '__main__':
    unittest.main()
