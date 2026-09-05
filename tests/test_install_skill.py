from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

SKILL = os.environ.get('WORLD_ATLAS_SKILL', str(Path(__file__).resolve().parents[1] / 'skills/generate-world-atlas'))
SCRIPT = Path(SKILL or '.') / 'scripts/install_engine.py'


class InstallationTests(unittest.TestCase):
    def test_install_refuses_existing_user_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(SCRIPT), '--target', folder], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('exists', result.stderr)

    def test_check_only_never_creates_target_or_installs(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'new-runtime'
            result = subprocess.run([sys.executable, str(SCRIPT), '--target', str(target), '--check-only'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn('assets', result.stdout)
            self.assertFalse(target.exists())
