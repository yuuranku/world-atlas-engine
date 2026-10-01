"""Independent published-shore acceptance catches gaps that clipping misses."""
import importlib.util
import json
from pathlib import Path
import tempfile

import unittest
import numpy as np
import shapely

from world_atlas.core.svg_artifacts import encode_svgz, read_svgz

spec = importlib.util.spec_from_file_location(
    "coastal_coverage_check", Path(__file__).resolve().parents[1] / "scripts" / "check_coastal_coverage.py"
)
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


class CoastalCoverageTests(unittest.TestCase):
    def test_missing_canonical_compressed_physical_export_fails_with_old_file_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            review=Path(temporary)
            self.published_fixture(review)
            current=review/'physical-surface.svgz'
            (review/'physical-surface.svg').write_text(read_svgz(current),encoding='utf-8')
            current.unlink()
            with self.assertRaisesRegex(ValueError,'physical-surface.svgz'):
                coverage.check_review(review)

    def test_corrupt_compressed_physical_export_is_not_accepted_as_empty_paint(self):
        with tempfile.TemporaryDirectory() as temporary:
            review=Path(temporary)
            self.published_fixture(review)
            current=review/'physical-surface.svgz'
            payload=bytearray(current.read_bytes());payload[-8]^=1
            current.write_bytes(payload)
            with self.assertRaises(OSError):
                coverage.check_review(review)

    def published_fixture(self, review):
        land = "M0,0 10,0 10,10 0,10Z M2,2 4,2 4,4 2,4Z"
        markup = (
            '<svg><defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
            f'<path d="{land}" clip-rule="evenodd"/>'
            '</clipPath></defs><g id="lakes"><path d="M2,2 4,2 4,4 2,4Z" '
            'fill="#eefcff"/></g></svg>'
        )
        images = ''.join(f'<image data-theme-map="{theme}" data-source="{theme}.svg"/>'
                         for theme in coverage.THEMES)
        (review / "index.html").write_text(
            '<html><div id="physical-surface"></div><svg>'
            '<g id="physical-theme-overlays" clip-path="url(#land-silhouette-clip)">'
            + images + '</g></svg></html>', encoding="utf-8",
        )
        (review / "physical-surface.svgz").write_bytes(encode_svgz(markup))
        document = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
            f'<g data-partition="mutually-exclusive"><path d="{land}" fill="#abcdef"/></g></svg>'
        )
        for theme in coverage.THEMES:
            (review / f"{theme}.svg").write_text(document, encoding="utf-8")
        globe = {
            "surface": '<svg xmlns="http://www.w3.org/2000/svg">' + markup[5:-6] + '</svg>',
            "ink": '<svg xmlns="http://www.w3.org/2000/svg"/>',
            "textures": {theme: f"globe-theme-{theme}.svg" for theme in coverage.GLOBE_THEMES},
        }
        for theme, filename in globe["textures"].items():
            markup = '<svg xmlns="http://www.w3.org/2000/svg"/>' if theme == "terrain" else (
                '<svg xmlns="http://www.w3.org/2000/svg"><g clip-path="url(#land-silhouette-clip)">'
                + document + '</g></svg>')
            (review / filename).write_text(markup, encoding="utf-8")
        (review / "globe-data.js").write_text(
            "window.WorldAtlasGlobe=" + json.dumps(globe) + ";", encoding="utf-8",
        )
        # A raster export cannot prove vector coverage.
        (review / "population.png").write_bytes(b"not a PNG")

    def test_missing_coastal_strip_fails_even_when_paint_never_crosses_water(self):
        land = shapely.box(0, 0, 10, 10)
        paint = shapely.box(.3, 0, 10, 10)
        result = coverage.coverage_metrics(land, paint)
        self.assertEqual(result["paintOutsideLandAreaBeforeClip"], 0)
        self.assertAlmostEqual(result["uncoveredLandArea"], 3)
        self.assertAlmostEqual(result["uncoveredLandPercent"], 3)

    def test_even_odd_lakes_and_islands_match_svg_fill(self):
        geometry = coverage.path_geometry(
            "M0,0 H10 V10 H0 Z M2,2 8,2 8,8 2,8Z M4,4 6,4 6,6 4,6Z"
        )
        self.assertAlmostEqual(geometry.area, 68)
        self.assertTrue(geometry.covers(shapely.Point(5, 5)))
        self.assertFalse(geometry.covers(shapely.Point(3, 3)))

    def test_absolute_and_relative_polygons_have_identical_even_odd_shapes(self):
        absolute = coverage.path_geometry(
            "M10,20 H30 V40 H10 Z M14,24 26,24 26,36 14,36Z M18,28 22,28 22,32 18,32Z"
        )
        relative = coverage.path_geometry(
            "m10,20 h20 v20 h-20 z m4,4 l12,0 0,12 -12,0z m4,4 l4,0 0,4 -4,0z"
        )
        self.assertTrue(relative.equals(absolute))
        self.assertAlmostEqual(relative.area, 272)
        self.assertTrue(relative.covers(shapely.Point(20, 30)))
        self.assertFalse(relative.covers(shapely.Point(16, 26)))

    def test_relative_move_after_close_uses_the_previous_ring_start(self):
        absolute = coverage.path_geometry("M2,3 12,3 12,13 2,13Z M22,3 27,3 27,8 22,8Z")
        relative = coverage.path_geometry("M2,3 l10,0 0,10 -10,0z m20,0 5,0 0,5 -5,0z")
        self.assertTrue(relative.equals(absolute))
        self.assertAlmostEqual(relative.area, 125)

    def test_relative_decimal_steps_recover_the_exact_shared_vertex(self):
        absolute = coverage.path_geometry("M0,0 .3,0 .3,1 0,1z M.3,0 .6,0 .6,1 .3,1z")
        relative = coverage.path_geometry("M0,0 l.1,0 .1,0 .1,0 0,1 -.3,0z M.3,0 l.3,0 0,1 -.3,0z")
        self.assertTrue(relative.equals(absolute))
        self.assertAlmostEqual(relative.area, .6)

    def test_touching_ring_vertex_is_noded_on_the_other_ring_edge(self):
        geometry = coverage.path_geometry("M1,1 2,1 2,2 1,2z M1,3 3,1 3,3z")
        self.assertTrue(geometry.is_valid)
        self.assertAlmostEqual(geometry.area, 3)
        self.assertTrue(geometry.covers(shapely.Point(1.5, 1.5)))
        self.assertTrue(geometry.covers(shapely.Point(2.8, 2.8)))

    def test_even_odd_paint_excludes_twice_covered_faces(self):
        geometry = coverage.path_geometry("M0,0 10,0 10,10 0,10z M5,0 15,0 15,10 5,10z")
        self.assertAlmostEqual(geometry.area, 100)
        self.assertTrue(geometry.covers(shapely.Point(2, 5)))
        self.assertTrue(geometry.covers(shapely.Point(12, 5)))
        self.assertFalse(geometry.covers(shapely.Point(7, 5)))

    def test_border_paint_and_physical_clip_cover_tiny_shore_without_claiming_overflow(self):
        land = shapely.box(0, 0, 10, 10)
        raw = {"d": "M.1,0 10,0 10,10 .1,10Z", "stroke": "#123456", "stroke-width": ".2"}
        filled = coverage.union_paths([raw])
        painted = coverage.union_paths([raw], paint=True)
        self.assertAlmostEqual(coverage.coverage_metrics(land, filled)["uncoveredLandArea"], 1)
        self.assertAlmostEqual(land.difference(painted).area, 0)
        self.assertGreater(painted.difference(land).area, 0)
        self.assertAlmostEqual(painted.intersection(land).difference(land).area, 0)

    def test_checker_rejects_curves_instead_of_silently_undercounting_land(self):
        with self.assertRaisesRegex(ValueError, "polygonal"):
            coverage.path_geometry("M0,0 Q5,5 10,0Z")

    def test_native_shoreline_check_counts_every_source_centre_and_reports_changed_classifications(self):
        water = np.ones((5, 6), dtype=np.uint8)
        water[2, 2] = 0
        good = coverage.native_shoreline_metrics(shapely.box(2, 2, 3, 3), water)
        self.assertEqual(good["checkedNativeCenters"], 30)
        self.assertEqual(good["mismatchCount"], 0)
        moved = coverage.native_shoreline_metrics(shapely.box(3, 2, 4, 3), water)
        self.assertEqual(moved["landCentersDisplayedAsWater"], 1)
        self.assertEqual(moved["waterCentersDisplayedAsLand"], 1)
        self.assertEqual(moved["mismatchCount"], 2)

    def test_physical_and_population_svg_exports_are_the_full_quality_surfaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            self.published_fixture(review)
            report = coverage.check_review(review)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["visibleLandArea"], 96)
        self.assertEqual(report["visibleLandUnderLakeFillArea"], 0)
        self.assertEqual(report["themes"]["population"]["uncoveredLandArea"], 0)
        self.assertTrue(report["clipContract"]["sharedLandClipIdentical"])

    def test_working_export_paint_is_measured_inside_the_shared_physical_shore(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            self.published_fixture(review)
            path = review / "population.svg"
            path.write_text(path.read_text(encoding="utf-8").replace(
                'M0,0 10,0 10,10 0,10Z M2,2 4,2 4,4 2,4Z',
                'M-2,-2 12,-2 12,12 -2,12Z',
            ), encoding="utf-8")
            report = coverage.check_review(review)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["themes"]["population"]["visiblePaintOutsideLandArea"], 0)
        self.assertGreater(report["themes"]["population"]["paintOutsideLandAreaBeforeClip"], 0)

    def test_physical_export_requires_one_authoritative_native_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            self.published_fixture(review)
            path = review / "physical-surface.svgz"
            path.write_bytes(encode_svgz(read_svgz(path).replace(
                'id="land-silhouette-clip"', 'id="other-clip"')))
            with self.assertRaisesRegex(ValueError, "exactly one land-silhouette-clip"):
                coverage.check_review(review)

    def test_globe_must_share_the_exact_flat_clip_and_clip_its_thematic_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            self.published_fixture(review)
            path = review / "globe-data.js"
            original = coverage.globe_payload(path)
            for modified, expected in (
                ({**original, "surface": original["surface"].replace('10,10', '9,10')}, "same physical land clip"),
            ):
                path.write_text("window.WorldAtlasGlobe=" + json.dumps(modified) + ";", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    coverage.check_review(review)
            path.write_text("window.WorldAtlasGlobe=" + json.dumps(original) + ";", encoding="utf-8")
            theme_path = review / original["textures"]["political"]
            theme_path.write_text(theme_path.read_text(encoding="utf-8").replace('clip-path="url(#land-silhouette-clip)"', ''), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lacks the physical land clip"):
                coverage.check_review(review)

    def test_globe_shared_ink_cannot_repeat_a_thematic_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            self.published_fixture(review)
            path = review / "globe-data.js"
            payload = coverage.globe_payload(path)
            payload["ink"] = (review / payload["textures"]["political"]).read_text(encoding="utf-8")
            path.write_text("window.WorldAtlasGlobe=" + json.dumps(payload) + ";", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "without thematic partitions"):
                coverage.check_review(review)

    def test_population_vector_is_required_even_when_a_raster_export_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            self.published_fixture(review)
            (review / "population.svg").unlink()
            with self.assertRaisesRegex(ValueError, "population.svg"):
                coverage.check_review(review)

    def test_globe_carrier_accepts_only_one_json_object_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "globe-data.js"
            for payload in ('window.WorldAtlasGlobe="<svg/>";',
                            'window.WorldAtlasGlobe={}; extra();'):
                path.write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "JSON object assignment"):
                    coverage.globe_payload(path)


if __name__ == "__main__":
    unittest.main()
