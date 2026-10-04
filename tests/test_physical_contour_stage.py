"""A resumed physical extraction must use exact, current, complete graphs."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.cartographic_contours import cartographic_level_curves
from world_atlas.core.terrain_refinement import RefinedTerrainField
from world_atlas.core.physical_contour_stage import (
    _binding, _read_graph, _write_graph,
    current_height_graphs, staged_height_curves,
)


def field(shift=0.):
    y, x = np.indices((5, 6), dtype=float)
    values = 80 + 35*x + 19*y + shift
    return PhysicalTerrainField(values, land_mask=values > 0, sea_level_m=0,
                                elevation_scale_m=4000, elevation_exponent=1)


class PhysicalContourStageTests(unittest.TestCase):
    def test_actual_source_graphs_resume_without_repeating_extraction(self):
        source = field()
        levels = np.array((133.25, 189.5))
        expected = cartographic_level_curves(source, levels)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = staged_height_curves(source, levels, root, source_identity={"source": "immutable-fixture"})
            with patch("world_atlas.core.physical_contour_stage.cartographic_curve_batches",
                       side_effect=AssertionError("complete source graphs must not be recalculated")):
                resumed = staged_height_curves(source, levels, root, source_identity={"source": "immutable-fixture"})
            for wanted, generated, saved in zip(expected, actual, resumed, strict=True):
                self.assertEqual(len(wanted), len(generated))
                self.assertEqual(len(wanted), len(saved))
                for a, b, c in zip(wanted, generated, saved, strict=True):
                    np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
                    np.testing.assert_array_equal(a.view(np.uint64), c.view(np.uint64))
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completedHeights"], 2)
            self.assertTrue(current_height_graphs(root))
            with patch("world_atlas.core.physical_contour_stage._extraction_runtime_binding",
                       return_value={"runtime": {"changed": True}}):
                self.assertFalse(current_height_graphs(root))
            original = source.native_m.copy()
            for cut, paths in zip(levels, resumed, strict=True):
                for path in paths:
                    self.assertLess(np.max(np.abs(source.sample_points(path[:, 0], path[:, 1]) - cut)), 2e-8)
            np.testing.assert_array_equal(original, source.native_m)

    def test_changed_native_ground_rejects_even_reused_external_identity(self):
        identity = {"source": "same-caller-label"}
        with tempfile.TemporaryDirectory() as temporary:
            staged_height_curves(field(), [133.25], temporary, source_identity=identity)
            with self.assertRaisesRegex(ValueError, "obsolete physical contour stage"):
                staged_height_curves(field(1.), [133.25], temporary, source_identity=identity)

    def test_changed_input_or_height_recipe_rejects_complete_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = field()
            staged_height_curves(source, [133.25], temporary, source_identity={"seed": 1})
            for identity, levels in (({"seed": 2}, [133.25]), ({"seed": 1}, [134.25])):
                with self.assertRaisesRegex(ValueError, "obsolete physical contour stage"):
                    staged_height_curves(source, levels, temporary, source_identity=identity)

    def test_same_native_samples_cannot_reuse_changed_continuous_refinement(self):
        base = field()
        flow = np.arange(base.height*base.width, dtype=np.int32).reshape(base.height, base.width)-1
        flow[:, 0] = -1
        arguments = dict(seed=37, radius_km=60., flow_to=flow,
                         discharge=np.full(base.native_m.shape, 10.),
                         river_segments=np.empty((0, 2, 2)), river_anchors=np.empty((0, 2)))
        source = RefinedTerrainField(base, **arguments)
        changes = ({"seed": 38}, {"radius_km": 61.},
                   {"discharge": np.full(base.native_m.shape, 12.)},
                   {"river_segments": np.array([[[2.5, .5], [2.5, 4.5]]])},
                   {"river_anchors": np.array([[2.71, 2.19]])})
        with tempfile.TemporaryDirectory() as temporary:
            staged_height_curves(source, [], temporary, source_identity={"source": "same-native"})
            for change in changes:
                changed = RefinedTerrainField(base, **(arguments | change))
                np.testing.assert_array_equal(changed.native_m, source.native_m)
                with self.subTest(change=list(change)), self.assertRaisesRegex(ValueError, "obsolete physical contour stage"):
                    staged_height_curves(changed, [], temporary, source_identity={"source": "same-native"})

    def test_corrupted_saved_bytes_raise_and_are_never_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = field()
            staged_height_curves(source, [133.25], root, source_identity={"seed": 1})
            path = root / "level-000.npz"
            path.write_bytes(path.read_bytes() + b"changed")
            self.assertFalse(current_height_graphs(root))
            with patch("world_atlas.core.physical_contour_stage.cartographic_curve_batches",
                       side_effect=AssertionError("invalid source bytes must fail closed")):
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    staged_height_curves(source, [133.25], root, source_identity={"seed": 1})

    def test_river_bed_change_rejects_a_checkpoint_with_identical_native_heights(self):
        base = field()
        arguments = dict(seed=37, radius_km=60.,
                         flow_to=np.full(base.native_m.shape, -1, dtype=np.int32),
                         discharge=np.ones(base.native_m.shape),
                         river_segments=np.empty((0, 2, 2)), river_anchors=np.empty((0, 2)))
        source = RefinedTerrainField(base, **arguments)
        source._river_bed = SimpleNamespace(
            segments=np.array([[[2.5, .5], [2.5, 4.5]]]), owners=np.array([0]),
            beds=np.array([[80., 40.]]), radii=np.array([[.2, .3]]))
        with tempfile.TemporaryDirectory() as temporary:
            staged_height_curves(source, [], temporary, source_identity={"same-native": True})
            source._river_bed.beds[0, 1] = 30.
            with self.assertRaisesRegex(ValueError, "obsolete physical contour stage"):
                staged_height_curves(source, [], temporary, source_identity={"same-native": True})

    def test_failed_source_extraction_does_not_commit_header_or_erase_other_graph(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = field()
            staged_height_curves(source,[133.25,189.5],root,source_identity={"source":"immutable"})
            paths = cartographic_level_curves(source,[133.25])[0]
            original = (root / "level-000.npz").read_bytes()
            fingerprint = json.loads((root / "manifest.json").read_text())["fingerprint"]
            (root / "level-001.npz").unlink()
            (root / "level-001.json").unlink()
            with patch("world_atlas.core.physical_contour_stage.cartographic_curve_batches",
                       side_effect=ValueError("uncertified physical critical point")):
                with self.assertRaisesRegex(ValueError, "uncertified"):
                    staged_height_curves(source,[133.25,189.5],root,source_identity={"source":"immutable"})
            self.assertFalse((root / "level-001.json").exists())
            self.assertFalse((root / "level-001.npz").exists())
            self.assertEqual(original, (root / "level-000.npz").read_bytes())
            saved = _read_graph(root, 0, 133.25, fingerprint)
            for wanted, actual in zip(paths, saved, strict=True):
                np.testing.assert_array_equal(wanted.view(np.uint64), actual.view(np.uint64))

    def test_datum_and_current_source_code_are_part_of_the_binding(self):
        source = field()
        first = _binding(source, [133.25], {"seed": 1})
        source.elevation_scale_m = 5000
        second = _binding(source, [133.25], {"seed": 1})
        self.assertNotEqual(first, second)
        with patch("world_atlas.core.physical_contour_stage._sha", return_value="changed-source"):
            third = _binding(source, [133.25], {"seed": 1})
        self.assertNotEqual(second, third)

    def test_cartographic_sampling_contract_is_bound_and_old_exact_schema_rejected(self):
        source = field()
        binding = _binding(source,[133.25],{"seed":1})
        self.assertEqual(binding["cartographicContract"]["subdivisionPerNativeCell"],4)
        self.assertFalse(binding["cartographicContract"]["physicalResolutionChanged"])
        with tempfile.TemporaryDirectory() as temporary:
            staged_height_curves(source,[133.25],temporary,source_identity={"seed":1})
            path=Path(temporary)/"manifest.json"
            record=json.loads(path.read_text())
            record["schema"]="physical-height-graphs-v1"
            path.write_text(json.dumps(record))
            self.assertFalse(current_height_graphs(temporary))
            with self.assertRaisesRegex(ValueError,"obsolete physical contour stage"):
                staged_height_curves(source,[133.25],temporary,source_identity={"seed":1})


if __name__ == "__main__":
    unittest.main()
