"""Public distribution contract: run without the original project on sys.path."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PackageContractTests(unittest.TestCase):
    def test_source_input_hash_check_rejects_tampered_bundle(self):
        from world_atlas.rebuild import prepare_regeneration
        from world_atlas.settings import load_world_settings
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'image.png').write_bytes(b'not used as artwork')
            (root/'fields.npz').write_bytes(b'tampered')
            (root/'config.json').write_text(json.dumps({'source':{'path':'image.png','sha256':hashlib.sha256(b'not used as artwork').hexdigest(),'fieldBundle':{'path':'fields.npz','sha256':'0'*64}}}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'physical field hash mismatch'):
                prepare_regeneration(
                    root/'config.json',
                    root/'provenance.json',
                    root/'output',
                    [],
                    settings=load_world_settings(
                        Path(__file__).resolve().parents[1]/'examples/world-settings.json'
                    ),
                )
            self.assertFalse((root/'output').exists())

    def test_invalid_recipe_numbers_rejected(self):
        from world_atlas.api import load_recipe
        record = json.loads((Path(__file__).resolve().parents[1]/'examples/terrain.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'recipe.json'
            for field,value in [('seed',True),('seed',-1),('coastline_detail',float('nan')),('plate_count',7.5)]:
                candidate={**record,field:value}
                path.write_text(json.dumps(candidate),encoding='utf-8')
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    load_recipe(path)
    def test_engine_is_installable_namespace_not_old_scripts(self):
        self.assertIsNotNone(importlib.util.find_spec("world_atlas"), "standalone world_atlas package is missing")

    def test_cli_exposes_both_generation_modes(self):
        result = subprocess.run([sys.executable, "-m", "world_atlas", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("doctor", "seeds", "terrain", "world", "reproduce", "verify"):
            self.assertIn(command, result.stdout)

    def test_import_does_not_mutate_module_search_path(self):
        result = subprocess.run([sys.executable, "-c", "import sys; before=list(sys.path); import world_atlas; from world_atlas.core import procedural_planet, baseline; assert before==sys.path; assert 'scripts' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_recipe_field_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "recipe.json"
            recipe.write_text(json.dumps({"magic": 123}), encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "world_atlas", "terrain", "--recipe", str(recipe), "--output", str(root / "output")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("recipe", result.stderr.lower())
            self.assertFalse((root / "output").exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "keep.txt"
            marker.write_text("user data", encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "world_atlas", "terrain", "--recipe", str(Path(__file__).resolve().parents[1] / "examples/terrain.json"), "--output", str(root)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exists", result.stderr)
            self.assertEqual(marker.read_text(), "user data")

    def test_doctor_reports_missing_renderer_without_original_project_lookup(self):
        result = subprocess.run([sys.executable, "-m", "world_atlas", "doctor", "--mapshaper", "absent/mapshaper"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mapshaper", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
