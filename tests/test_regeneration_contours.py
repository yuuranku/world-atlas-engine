"""Human-only runs reuse current physical graphs without changing their binding."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.implicit_terrain import terrain_level_curves
from world_atlas.core import physical_contour_stage as contour
from world_atlas.rebuild import prepare_regeneration
from world_atlas.settings import load_world_settings


def fixture(root, *, beside_bundle=False):
    settings = load_world_settings(Path(__file__).resolve().parents[1] / 'examples/world-settings.json')
    source_dir = root / 'source'
    source_dir.mkdir()
    image, bundle = source_dir / 'image.png', source_dir / 'fields.npz'
    image.write_bytes(b'immutable-terrain-image')
    bundle.write_bytes(b'immutable-physical-bundle')
    config = {'source': {'path': 'source/image.png',
                         'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                         'fieldBundle': {'path': 'source/fields.npz',
                                         'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest()}},
              'planet': {**settings.planet, 'orbitalEccentricity': 0.0},
              'output': {'directory': '.'}}
    config_path = root / 'worldgen.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    provenance = source_dir / 'provenance.json'
    provenance.write_text('{}', encoding='utf-8')
    y, x = np.indices((5, 6), dtype=float)
    values = 80 + 35*x + 19*y
    physical = SimpleNamespace(relative_elevation_m=values, diagnostics={'seed': 37})
    field = PhysicalTerrainField(values, land_mask=values > 0, sea_level_m=0.,
                                 elevation_scale_m=4000., elevation_exponent=1.)
    identity = {'rawElevationSha256': hashlib.sha256(values.tobytes()).hexdigest(),
                'physicalDiagnostics': physical.diagnostics}
    binding = contour._binding(field, [133.25], identity)
    fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    directory = (source_dir if beside_bundle else root) / 'physical-contours'
    directory.mkdir()
    contour._write_graph(directory, 0, 133.25, fingerprint, terrain_level_curves(field, [133.25])[0])
    manifest = {'schema': contour._SCHEMA, 'binding': binding, 'fingerprint': fingerprint,
                'status': 'complete', 'completedHeights': 1, 'totalHeights': 1}
    (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return SimpleNamespace(settings=settings, config=config, config_path=config_path,
                           provenance=provenance, physical=physical, field=field,
                           directory=directory, identity=identity, output=root / 'new-world')


def prepare(data, settings=None):
    with patch('world_atlas.rebuild.load_surface_bundle', return_value=data.physical):
        return prepare_regeneration(data.config_path, data.provenance, data.output, [],
                                    settings=settings or data.settings)


class RegenerationContourTests(unittest.TestCase):
    def test_human_seed_changes_copy_exact_current_graphs_from_supported_locations(self):
        for beside_bundle in (False, True):
            with self.subTest(beside_bundle=beside_bundle), tempfile.TemporaryDirectory() as directory:
                data = fixture(Path(directory), beside_bundle=beside_bundle)
                before = {path.name: path.read_bytes() for path in data.directory.iterdir()}
                settings = replace(data.settings, human_seed=371, naming_seed=812)
                prepared = prepare(data, settings)
                self.assertEqual(prepared['physicalContoursCheckpoint']['state'], 'hit')
                copied = data.output / 'physical-contours'
                self.assertEqual({path.name: path.read_bytes() for path in copied.iterdir()}, before)
                with patch.object(contour, 'cartographic_curve_batches', side_effect=AssertionError('reuse must not extract')):
                    contour.staged_height_curves(data.field, [133.25], copied, source_identity=data.identity)
                self.assertEqual({path.name: path.read_bytes() for path in data.directory.iterdir()}, before)

    def test_changed_planet_or_raw_ground_does_not_copy_graphs(self):
        for changed in ('planet', 'ground'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                data = fixture(Path(directory))
                settings = data.settings
                if changed == 'planet':
                    settings = replace(settings, planet={**settings.planet,
                                                         'radiusKm': settings.planet['radiusKm']+1})
                else:
                    data.physical.relative_elevation_m = data.physical.relative_elevation_m+1
                self.assertEqual(prepare(data, settings)['physicalContoursCheckpoint']['state'], 'miss')
                self.assertFalse((data.output / 'physical-contours').exists())

    def test_changed_code_or_corrupted_graph_never_gets_rebound(self):
        for changed in ('code', 'graph'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                data = fixture(Path(directory))
                if changed == 'code':
                    path = data.directory / 'manifest.json'
                    record = json.loads(path.read_text(encoding='utf-8'))
                    record['binding']['sourceModulesSha256']['implicit_terrain'] = '0'*64
                    record['fingerprint'] = hashlib.sha256(json.dumps(record['binding'], sort_keys=True,
                                                                   separators=(',', ':')).encode()).hexdigest()
                    path.write_text(json.dumps(record), encoding='utf-8')
                else:
                    path = data.directory / 'level-000.npz'
                    path.write_bytes(path.read_bytes()+b'changed')
                before = {path.name: path.read_bytes() for path in data.directory.iterdir()}
                self.assertEqual(prepare(data)['physicalContoursCheckpoint']['state'], 'miss')
                self.assertFalse((data.output / 'physical-contours').exists())
                self.assertEqual({path.name: path.read_bytes() for path in data.directory.iterdir()}, before)


if __name__ == '__main__':
    unittest.main()
