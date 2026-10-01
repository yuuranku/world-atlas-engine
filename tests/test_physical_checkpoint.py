import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from world_atlas.rebuild import _load_verified_physical_checkpoint


class PhysicalCheckpointTests(unittest.TestCase):
    def test_exact_fingerprint_required_and_manifest_cannot_bypass_grid_digest(self):
        grid = SimpleNamespace(metadata={"inputFingerprint": "physical-a"}, content_digest=lambda: "grid-a")
        config = SimpleNamespace(source=SimpleNamespace(path=Path('source.png')))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            manifest = checkpoint / 'checkpoint.json'
            manifest.write_text(json.dumps({"schema":"world-atlas-physical-checkpoint-v1", "inputFingerprint":"physical-a", "gridDigest":"grid-a"}))
            with patch('world_atlas.rebuild.WorldGrid.load', return_value=grid), patch('world_atlas.rebuild.load_world_config', return_value=config), patch('world_atlas.rebuild.compute_input_fingerprint', return_value='physical-a') as fingerprint:
                self.assertIs(_load_verified_physical_checkpoint(Path('worldgen.json'), checkpoint), grid)
                fingerprint.assert_called_once_with(config.source.path, config)
                fingerprint.return_value = 'changed-climate-source-or-code'
                self.assertIsNone(_load_verified_physical_checkpoint(Path('worldgen.json'), checkpoint))
                fingerprint.return_value = 'physical-a'
                grid.metadata['inputFingerprint'] = 'other-grid'
                self.assertIsNone(_load_verified_physical_checkpoint(Path('worldgen.json'), checkpoint))

    def test_missing_checkpoint_is_an_explicit_cache_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(_load_verified_physical_checkpoint(Path('worldgen.json'), Path(directory)))
