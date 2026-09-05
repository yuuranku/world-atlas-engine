import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import re
import json

SKILL = os.environ.get('WORLD_ATLAS_SKILL', str(Path(__file__).resolve().parents[1] / 'skills/generate-world-atlas'))
SCRIPT = Path(SKILL or '.') / 'scripts/prepare_brief.py'


class BriefTests(unittest.TestCase):
    def test_skill_starts_with_exactly_fifteen_questions(self):
        root = Path(SKILL)
        questions = (root/'references/questionnaire.md').read_text(encoding='utf-8')
        self.assertEqual([int(x) for x in re.findall(r'^(\d+)\. \*\*', questions, flags=re.M)], list(range(1,16)))
        instructions = (root/'SKILL.md').read_text(encoding='utf-8')
        self.assertIn('第一项行动', instructions)
        self.assertIn('只有玩家明确说', instructions)

    def test_brief_remains_unconfirmed_until_parameters_reviewed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'brief.json'
            answers = ' '.join(f'{i}B' for i in range(1,16))
            result = subprocess.run([sys.executable,str(SCRIPT),'--answers',answers,'--seed','42','--output',str(path)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['status'],'awaiting-parameter-confirmation')
    def test_requires_all_fifteen_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(SCRIPT), '--answers', '1A 2B', '--seed', '42', '--output', str(Path(folder)/'brief.json')], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('15', result.stderr)

    def test_random_is_explicit_and_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            outputs = []
            for name in ('a.json', 'b.json'):
                path = Path(folder)/name
                result = subprocess.run([sys.executable, str(SCRIPT), '--random', '--seed', '42', '--output', str(path)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                outputs.append(path.read_bytes())
            self.assertEqual(*outputs)


if __name__ == '__main__':
    unittest.main()
