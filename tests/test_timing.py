import json
from unittest.mock import patch

from pathlib import Path
import tempfile
import unittest

from world_atlas.timing import elapsed_seconds, measure_stage


class TimingTests(unittest.TestCase):
    def test_resume_counts_work_not_time_between_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch('world_atlas.timing.time.perf_counter', side_effect=[10., 13., 1000., 1002.]):
                with measure_stage(output, 'physical'):
                    pass
                with measure_stage(output, 'society'):
                    pass
            self.assertEqual(elapsed_seconds(output), 5.)
            self.assertEqual([s['status'] for s in json.loads((output / 'timing.json').read_text())['stages']], ['complete', 'complete'])


    def test_failure_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch('world_atlas.timing.time.perf_counter', side_effect=[10., 12.]):
                with self.assertRaisesRegex(ValueError, 'fixture'):
                    with measure_stage(output, 'society'):
                        raise ValueError('fixture')
            self.assertEqual(elapsed_seconds(output), 2.)
            self.assertEqual(json.loads((output / 'timing.json').read_text())['stages'][0]['status'], 'failed')
