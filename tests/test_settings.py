import json
from pathlib import Path
import tempfile
import unittest

from world_atlas.settings import load_world_settings


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/world-settings.json"


class WorldSettingsTests(unittest.TestCase):
    def test_example_connects_independent_human_and_naming_seeds(self):
        settings = load_world_settings(EXAMPLE)
        self.assertEqual(settings.human_seed, 445034353)
        self.assertEqual(settings.naming_seed, 3188500066)
        self.assertEqual(settings.society["stateCount"], 45)
        self.assertEqual(settings.society["frontierTargetShare"], 0.25)

    def test_unknown_fields_are_rejected(self):
        document = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        document["society"]["ignored"] = 1
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "society must contain exactly"):
                load_world_settings(path)

    def test_frontier_share_is_bounded(self):
        document = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        document["society"]["frontierTargetShare"] = 1.0
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frontierTargetShare"):
                load_world_settings(path)


if __name__ == "__main__":
    unittest.main()
