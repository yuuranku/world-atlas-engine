"""Delivered SVG vertices remain exact while browser accumulation is bounded."""

import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from world_atlas.core.render import _svg_group
from world_atlas.core.svg_paths import COORDINATE_SCALE, integer_subpath_data


spec = importlib.util.spec_from_file_location(
    "globe_path_reencoder", Path(__file__).resolve().parents[1] / "scripts" / "reencode_globe_paths.py"
)
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


class SvgPathsTests(unittest.TestCase):
    def test_major_contour_keeps_collinear_source_event_vertices(self):
        points = np.array(((512.5, 81.5), (512.625, 81.625), (512.75, 81.75),
                           (512.8312345678, 81.8623456789)))
        cut = 4085.313054156516
        group = ET.fromstring(_svg_group("elevation-contours", [points], "#abcdef",
                                        path_levels=[cut]))
        path = group.find("path")
        expected = [(round(x*COORDINATE_SCALE), round(y*COORDINATE_SCALE)) for x, y in points]
        self.assertEqual(decoder.linear_subpaths(path.get("d")), [(expected, False)])
        self.assertEqual(float(path.get("data-level")), cut)

    def test_frame_edges_are_absolute_without_moving_delivered_vertices(self):
        points = [(0, 0), (COORDINATE_SCALE, 10), (2*COORDINATE_SCALE, 30),
                  (2176*COORDINATE_SCALE, 10), (2176*COORDINATE_SCALE, 20), (0, 20)]
        encoded = integer_subpath_data(points, closed=True)
        self.assertEqual(decoder.linear_subpaths(encoded), [(points, True)])
        self.assertIn("L2 .0000003 2176 .0000001", encoded)
        self.assertIn("2176 .0000002", encoded)

    def test_long_interior_chain_has_absolute_anchors_and_exact_round_trip(self):
        points = [(0, 0)] + [(i*132659, 100000+i*771) for i in range(1, 240)] + [(100000000, 100000000)]
        encoded = integer_subpath_data(points, closed=False)
        self.assertEqual(decoder.linear_subpaths(encoded), [(points, False)])
        tokens = decoder.TOKEN.findall(encoded)
        command = None
        run = maximum = 0
        for token in tokens:
            if token in ("M", "L", "l"):
                command = token
                if command != "l": run = 0
            elif command == "l":
                run += 1
                maximum = max(maximum, run // 2)
        self.assertLessEqual(maximum, 64)
        self.assertGreater(encoded.count("L"), 2)

    def test_number_packing_keeps_integer_and_leading_decimal_distinct(self):
        points = [(3*COORDINATE_SCALE, COORDINATE_SCALE//2),
                  (0, 1), (COORDINATE_SCALE, 0)]
        encoded = integer_subpath_data(points, closed=True)
        self.assertTrue(encoded.startswith("M3 .5"))
        self.assertNotIn("0.", encoded)
        self.assertEqual(decoder.linear_subpaths(encoded), [(points, True)])

    def test_signs_and_second_decimal_points_separate_exact_numbers(self):
        points = [(60_000_000, 50_000_000), (-60_000_000, -50_000_000)]
        encoded = integer_subpath_data(points, closed=False)
        self.assertEqual(encoded, "M.6.5L-.6-.5")
        self.assertEqual(decoder.linear_subpaths(encoded), [(points, False)])

    def test_compact_relative_pairs_preserve_zero_steps_and_anchor_runs(self):
        points = [(0,0), (COORDINATE_SCALE,COORDINATE_SCALE)]
        points[1:1] = [(i*200_001, (i//2)*30_003) for i in range(1,180)]
        encoded = integer_subpath_data(points, closed=False)
        self.assertEqual(decoder.linear_subpaths(encoded), [(points, False)])
        self.assertIn("l", encoded)
        self.assertGreater(encoded.count("L"), 2)

    def test_standard_relative_subpaths_holes_and_close_restore_the_start(self):
        original = "M1,2h8v8h-8z m2,2 0,4h4v-4z"
        paths = decoder.linear_subpaths(original)
        self.assertEqual(paths, [([(100000000,200000000),(900000000,200000000),
                                  (900000000,1000000000),(100000000,1000000000)],True),
                                 ([(300000000,400000000),(300000000,800000000),
                                   (700000000,800000000),(700000000,400000000)],True)])
        rewritten, count, curves = decoder.reencode(f'<svg><path fill-rule="evenodd" d="{original}"/></svg>')
        root = ET.fromstring(rewritten)
        self.assertEqual(decoder.linear_subpaths(root.find("path").get("d")), paths)
        self.assertEqual((count,curves),(1,0))
        self.assertEqual(root.find("path").get("fill-rule"), "evenodd")

    def test_explicit_curve_commands_and_precision_are_preserved(self):
        curve = "M1,2 Q3,4 5,6"
        rewritten, count, curves = decoder.reencode(f'<svg><path d="{curve}"/></svg>')
        self.assertEqual(ET.fromstring(rewritten).find("path").get("d"), curve)
        self.assertEqual((count,curves),(0,1))
        with self.assertRaises(ValueError):
            decoder.linear_subpaths("M0,0l0.000000001,1L1,0Z")


if __name__ == "__main__":
    unittest.main()
