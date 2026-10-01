import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from world_atlas.core.globe import write_globe


class GlobeTests(unittest.TestCase):
    def test_globe_is_offline_and_keeps_exact_atlas_vector_geometry(self):
        surface = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 8">'
                   '<path id="shared-coast" d="M1.25,2.5 L3.125,4.75 Z" fill="#abcdef" />'
                   '</svg>')
        overlay = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 8" />'
        ink = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 8">'
               '<path id="shared-ink" d="M1.25,2.5 L3.125,4.75" stroke="#123456" />'
               '</svg>')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_globe(output, surface=surface, ink=ink, textures={"terrain": overlay, "provinces": overlay},
                        grid_digest="fixture", world_name="<世界>")
            payload = (output / 'globe-data.js').read_text(encoding='utf-8')
            data = json.loads(payload.removeprefix('window.WorldAtlasGlobe=').strip().removesuffix(';'))
            self.assertEqual(data['gridDigest'], 'fixture')
            self.assertEqual(data['projection'], 'equirectangular')
            self.assertEqual(data['surface'], surface)
            self.assertEqual(data['ink'], ink)
            self.assertEqual(data['textures']['terrain'], 'globe-theme-terrain.svg')
            for url in data['textures'].values():
                self.assertEqual((output / url).read_text(encoding='utf-8'), overlay)
                self.assertEqual(ET.parse(output / url).getroot().attrib['viewBox'], '0 0 16 8')
            self.assertEqual(payload.count('shared-coast'), 1)
            self.assertEqual(payload.count('shared-ink'), 1)
            document = (output / 'globe.html').read_text(encoding='utf-8')
            self.assertIn('&lt;世界&gt;', document)
            self.assertNotIn('https://', document)
            self.assertTrue((output / 'globe.js').is_file())
            self.assertTrue((output / 'three-license.txt').is_file())
            self.assertNotIn('value="climate"', document)
            self.assertIn('id="globe-graticule"', document)
            self.assertIn('width:100%;height:100%', document)

    def test_thirteen_local_theme_sources_have_chinese_labels_and_preserve_geometry(self):
        labels = {"terrain":"地形", "climate":"柯本气候", "biome":"生态群系",
            "watershed":"水文流域", "potential":"农业潜力", "habitability":"宜居度",
            "vegetation":"植被覆盖", "population":"人口分布", "civilizations":"文明区",
            "languages":"语言分布", "religions":"宗教分布", "political":"国家政区",
            "provinces":"省份政区"}
        surface = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 8" />'
        textures = {key: f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 8"><path id="{key}-original" d="M1.23456789,2.34567891 L3.45678912,4.56789123" /></svg>'
                    for key in labels}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_globe(output, surface=surface, ink=surface, textures=textures,
                        grid_digest="fixture", world_name="world")
            payload = (output / 'globe-data.js').read_text(encoding='utf-8')
            data = json.loads(payload.removeprefix('window.WorldAtlasGlobe=').strip().removesuffix(';'))
            self.assertEqual(len(data['textures']), 13)
            self.assertNotIn('-original', payload)
            document = (output / 'globe.html').read_text(encoding='utf-8')
            for key, label in labels.items():
                self.assertIn(f'<option value="{key}">{label}</option>', document)
                self.assertEqual((output / data['textures'][key]).read_text(encoding='utf-8'), textures[key])
            self.assertLess(len(payload), 2048)

    def test_globe_rejects_partial_or_mismatched_map_projections(self):
        def surface(viewbox):
            return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}" />'
        with tempfile.TemporaryDirectory() as directory:
            for textures in ({"terrain": surface("0 0 8 8")},
                             {"climate": surface("0 0 16 8")},
                             {"terrain": surface("0 0 16 8"), "climate": surface("0 0 8 4")},
                             {"terrain": surface("1 0 16 8")},
                             {"terrain": surface("0 0 0 0")},
                             {"terrain": surface("0 0 inf inf")},
                             {"terrain": '<svg />'},
                             {"terrain": '<svg'}):
                with self.subTest(textures=textures):
                    with self.assertRaises(ValueError):
                        write_globe(Path(directory), surface=surface("0 0 16 8"), ink=surface("0 0 16 8"), textures=textures,
                                    grid_digest="fixture", world_name="world")

    def test_globe_rejects_invalid_or_mismatched_shared_surface(self):
        overlay = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 8" />'
        with tempfile.TemporaryDirectory() as directory:
            for surface in ('<svg />', '<svg',
                            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 4" />'):
                with self.subTest(surface=surface):
                    with self.assertRaises(ValueError):
                        write_globe(Path(directory), surface=surface, ink=overlay, textures={"terrain": overlay},
                                    grid_digest="fixture", world_name="world")

    def test_globe_rejects_invalid_or_mismatched_shared_ink(self):
        overlay = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 8" />'
        with tempfile.TemporaryDirectory() as directory:
            for ink in ('<svg />', '<svg',
                        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 4" />'):
                with self.subTest(ink=ink):
                    with self.assertRaises(ValueError):
                        write_globe(Path(directory), surface=overlay, ink=ink, textures={"terrain": overlay},
                                    grid_digest="fixture", world_name="world")
