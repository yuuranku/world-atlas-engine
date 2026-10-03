import json
import multiprocessing
from unittest.mock import patch

from pathlib import Path
import tempfile
import time
import unittest

from world_atlas.timing import elapsed_seconds, measure_stage


def parallel_stage(output, barrier, index):
    with measure_stage(Path(output), f"parallel-{index}"):
        barrier.wait(timeout=10)
        time.sleep(.05)


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

    def test_nested_stages_keep_both_records_and_do_not_double_count(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch('world_atlas.timing.time.perf_counter', side_effect=[10., 11., 13., 15.]):
                with measure_stage(output, 'render'):
                    with measure_stage(output, 'tiles'):
                        pass
            document = json.loads((output / 'timing.json').read_text())
            parent, child = document['stages']
            self.assertEqual([stage['name'] for stage in document['stages']], ['render', 'tiles'])
            self.assertEqual([stage['elapsedSeconds'] for stage in document['stages']], [5., 2.])
            self.assertEqual(child['parentId'], parent['id'])
            self.assertIsNone(parent['parentId'])
            self.assertEqual(document['elapsedSeconds'], 5.)

    def test_processes_atomically_merge_records_and_count_overlapping_work_once(self):
        context = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as directory:
            barrier = context.Barrier(3)
            processes = [context.Process(target=parallel_stage, args=(directory, barrier, index))
                         for index in range(3)]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=15)
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join()
            document = json.loads((Path(directory) / 'timing.json').read_text())
            stages = document['stages']
            self.assertEqual({stage['name'] for stage in stages}, {f'parallel-{index}' for index in range(3)})
            self.assertTrue(all(stage['status'] == 'complete' for stage in stages))
            self.assertGreater(document['elapsedSeconds'], 0.)
            self.assertLess(document['elapsedSeconds'], sum(stage['elapsedSeconds'] for stage in stages) - .04)
            self.assertAlmostEqual(document['elapsedSeconds'],
                                   max(stage['finishedAtMonotonic'] for stage in stages)
                                   - min(stage['startedAtMonotonic'] for stage in stages), places=3)

    def test_old_timing_document_is_not_silently_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / 'timing.json'
            path.write_text('{"stages":[],"elapsedSeconds":7}', encoding='utf-8')
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'obsolete timing document'):
                with measure_stage(output, 'render'):
                    self.fail('obsolete timing may not be resumed')
            self.assertEqual(path.read_bytes(), before)
