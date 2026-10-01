"""An interface rebuild must reuse only the matching, unchanged map scene."""

from pathlib import Path
import tempfile
import unittest

from world_atlas.core.review_interface import load_interface_snapshot, write_interface_snapshot


class ReviewInterfaceTests(unittest.TestCase):
    def test_interface_scene_round_trip_keeps_geometry_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            overlay = '<g id="shore"><path d="M0.12345678 1L2 3"/></g>'
            write_interface_snapshot(
                target, overlay, grid_digest="physical", society_digest="human",
                width=800, height=400, viewbox_width=80, viewbox_height=40,
                tectonic_diagnostics={"plateCount": 2}, territorial_qa={"alignment": .8},
                settlement_locations={"city": (12.5, 32.5)},
            )
            restored, presentation = load_interface_snapshot(
                target, grid_digest="physical", society_digest="human"
            )
            self.assertEqual(restored, overlay)
            self.assertEqual(presentation["viewboxWidth"], 80)
            with self.assertRaisesRegex(ValueError, "different world"):
                load_interface_snapshot(target, grid_digest="different", society_digest="human")
            with self.assertRaisesRegex(ValueError, "different world"):
                load_interface_snapshot(target, grid_digest="physical", society_digest="different")
            with (target / "atlas-scene.svg").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(ValueError, "content changed"):
                load_interface_snapshot(target, grid_digest="physical", society_digest="human")


if __name__ == "__main__":
    unittest.main()
