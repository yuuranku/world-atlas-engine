import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from world_atlas import __version__
from world_atlas.api import reproduce_world


class ReplayContractTests(unittest.TestCase):
    def test_replay_reads_the_fingerprint_where_generation_writes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            output = Path(directory) / 'output'
            (source / 'review').mkdir(parents=True)
            (output / 'review').mkdir(parents=True)
            (source / 'regeneration.json').write_text(json.dumps({'engineVersion': __version__}))
            expected = {'gridArrayDigest': 'grid', 'societyArrayDigest': 'society', 'societyDocumentDigest': 'entities'}
            (source / 'review/release-checks.json').write_text(json.dumps(expected))
            with patch('world_atlas.api.generate_world', return_value={}) as generate:
                with patch('world_atlas.checks.semantic_checks', return_value=expected):
                    self.assertTrue(reproduce_world(source, output)['replay']['ok'])
            self.assertEqual(generate.call_args.args[1], source.resolve() / 'world-settings.json')

    def test_replay_refuses_another_engine_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / 'regeneration.json').write_text(json.dumps({'engineVersion': '0.0.0'}))
            with self.assertRaisesRegex(ValueError, 'original engine version'):
                reproduce_world(source, source / 'output')
            self.assertFalse((source / 'output').exists())
