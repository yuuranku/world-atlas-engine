"""Numeric-paint acceptance detects wrong classes independently of contours."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import shapely

from scripts import check_scalar_capacity as audit


class ScalarCapacityCheckTests(unittest.TestCase):
    def test_absolute_legends_and_zero_population_are_not_relative_display_ranks(self):
        np.testing.assert_array_equal(
            audit.expected_classes("population", np.array([[0, .05, .1, .5, 1, 2, 5, 500]])),
            [[0, 1, 2, 3, 4, 5, 6, 6]],
        )
        near_thresholds = np.array([[.5, np.nextafter(np.float32(1), np.float32(-np.inf)),
                                     np.nextafter(np.float32(2), np.float32(-np.inf))]], dtype=np.float32)
        np.testing.assert_array_equal(audit.expected_classes("population", near_thresholds), [[3, 3, 4]])
        np.testing.assert_array_equal(
            audit.expected_classes("vegetation", np.array([[0, .09, .1, .5, .9, 1]])),
            [[0, 0, 1, 5, 9, 9]],
        )
        np.testing.assert_array_equal(
            audit.expected_classes("potential", np.array([[0, .2, .35, .5, .65, .8, 1]])),
            [[0, 1, 2, 3, 4, 5, 5]],
        )

    def test_sparse_classes_preserve_zero_and_do_not_invent_intermediate_categories(self):
        values = np.array([[0, 1]])
        land = np.ones_like(values, dtype=bool)
        regions = [(0, shapely.box(0, 0, 1, 1)), (5, shapely.box(1, 0, 2, 1))]
        result, paint = audit.scalar_metrics("potential", values, land, regions)
        self.assertEqual(result["mismatchCount"], 0)
        self.assertEqual(result["paintedClassLandCells"], {"0": 1, "5": 1})
        self.assertAlmostEqual(paint.area, 2)
        wrong, _ = audit.scalar_metrics("potential", values, land, [(5, shapely.box(0, 0, 2, 1))])
        self.assertEqual(wrong["wrongClassCenters"], 1)
        missing, _ = audit.scalar_metrics("potential", values, land, regions[1:])
        self.assertEqual(missing["unpaintedLandCenters"], 1)

    def test_shared_threshold_boundary_uses_actual_svg_paint_order(self):
        values = np.array([[.2]])
        land = np.ones_like(values, dtype=bool)
        lower = (0, shapely.box(0, 0, .5, 1))
        upper = (1, shapely.box(.5, 0, 1, 1))
        result, _ = audit.scalar_metrics("potential", values, land, [lower, upper])
        self.assertEqual(result["mismatchCount"], 0)
        self.assertEqual(result["classOverlapArea"], 0)
        reversed_order, _ = audit.scalar_metrics("potential", values, land, [upper, lower])
        self.assertEqual(reversed_order["wrongClassCenters"], 1)

    def test_export_numeric_ancestors_and_physical_lake_hole_are_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vegetation-cover.svg"
            path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2" '
                'data-surface-contract="shared-land-clip">'
                '<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
                '<path d="M0,0H2V2H0ZM1,0H2V1H1Z" clip-rule="evenodd"/>'
                '</clipPath></defs><g clip-path="url(#land-silhouette-clip)">'
                '<g data-partition="mutually-exclusive" data-vegetation-band="partition">'
                '<defs><path data-vegetation-band="999" d="M0,0H2V2H0Z" '
                'fill="red" fill-rule="evenodd"/></defs>'
                '<g data-vegetation-band="7"><path d="M0,0H2V2H0Z" fill="#246b3b" '
                'fill-rule="evenodd"/></g></g></g></svg>', encoding="utf-8",
            )
            land = shapely.box(0, 0, 2, 2).difference(shapely.box(1, 0, 2, 1))
            regions = audit.export_regions(path, "data-vegetation-band", physical_land=land, dimensions=(2, 2))
            self.assertEqual(len(regions), 1)
            self.assertEqual(regions[0][0], 7)
            self.assertAlmostEqual(regions[0][1].area, 3)
            self.assertFalse(regions[0][1].covers(shapely.Point(1.5, .5)))
            path.write_text(path.read_text(encoding="utf-8").replace('data-vegetation-band="7"', ''), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "numeric class"):
                audit.export_regions(path, "data-vegetation-band", physical_land=land, dimensions=(2, 2))

    def test_export_owned_clip_definitions_and_references_are_required(self):
        document = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2" data-surface-contract="shared-land-clip">'
            '<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
            '<path d="M0,0H2V2H0Z" clip-rule="evenodd"/></clipPath></defs>'
            '<g clip-path="url(#land-silhouette-clip)"><g data-partition="mutually-exclusive">'
            '<g data-potential-band="3"><path d="M0,0H2V2H0Z" fill="green" fill-rule="evenodd"/>'
            '</g></g></g></svg>'
        )
        failures = {
            'missing definition': (document.replace('<defs>', '<g>').replace('</defs>', '</g>'), 'own one'),
            'foreign reference': (document.replace('url(#land-silhouette-clip)', 'url(#foreign)'), 'foreign'),
            'missing clip': (document.replace(' clip-path="url(#land-silhouette-clip)"', ''), 'lacks its owned'),
            'secondary clip': (document.replace('<g data-potential-band="3">', '<g data-potential-band="3" clip-path="url(#land-silhouette-clip)">'), 'secondary'),
            'object coordinates': (document.replace('userSpaceOnUse', 'objectBoundingBox'), 'own one'),
            'implicit winding': (document.replace('clip-rule="evenodd"', ''), 'explicit evenodd'),
            'moved clip': (document.replace('<clipPath ', '<clipPath transform="translate(1,0)" '), 'own one'),
            'moved definitions': (document.replace('<defs>', '<defs transform="translate(1,0)">'), 'own one'),
            'extra definition': (document.replace('</defs>', '<clipPath id="foreign"/></defs>'), 'own one'),
            'duplicate ID': (document.replace('<defs>', '<defs><path id="land-silhouette-clip"/>'), 'must be unique'),
            'wrong namespace': (document.replace('http://www.w3.org/2000/svg', 'http://example.invalid'), 'shared-land-clip'),
            'moved paint': (document.replace('<g data-potential-band="3">', '<g data-potential-band="3" transform="translate(.1,0)">'), 'unexpected transform'),
            'unclassified actual paint': (document.replace('<g data-potential-band="3">', '<g data-potential-band="3"><rect width="2" height="2"/>'), 'unsupported visible'),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'potential.svg'
            for name, (markup, message) in failures.items():
                with self.subTest(name=name):
                    path.write_text(markup, encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, message):
                        audit.export_regions(path, 'data-potential-band', physical_land=shapely.box(0,0,2,2), dimensions=(2,2))
            path.write_text(document, encoding='utf-8')
            regions = audit.export_regions(path, 'data-potential-band', physical_land=shapely.box(0,0,2,2), dimensions=(2,2))
            self.assertAlmostEqual(regions[0][1].area, 4)
            path.write_text(document.replace('<g clip-path=', '<g opacity="0" clip-path='), encoding='utf-8')
            self.assertEqual(audit.export_regions(path, 'data-potential-band', physical_land=shapely.box(0,0,2,2), dimensions=(2,2)), [])

    def test_export_clip_is_strictly_the_published_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'potential.svg'
            path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2" data-surface-contract="shared-land-clip">'
                '<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
                '<path d="M0,0H2V2H0Z" clip-rule="evenodd"/></clipPath></defs>'
                '<g data-partition="mutually-exclusive" clip-path="url(#land-silhouette-clip)">'
                '<path data-potential-band="3" d="M0,0H2V2H0Z" fill="green" fill-rule="evenodd"/></g></svg>',
                encoding='utf-8',
            )
            authority = shapely.box(0,0,2,2).difference(shapely.Polygon([(0,0),(1e-8,0),(0,1e-8)]))
            with self.assertRaisesRegex(ValueError, 'differs from the published'):
                audit.export_regions(path, 'data-potential-band', physical_land=authority, dimensions=(2,2))

    def test_actual_detail_payload_is_checked_against_numeric_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory)
            (review / "tiles/detail").mkdir(parents=True)
            land = "M0,0H1V1H2V2H0Z"
            water = "M1,0H2V1H1Z"
            themes = {"none": ""}
            classes = {"potential": 3, "habitability": 3, "vegetation": 6, "population": 2}
            fields = {}
            wet = np.array([[0, 1], [0, 0]], dtype=np.uint8)
            for theme, spec in audit.SPECS.items():
                fields[spec["field"]] = np.where(wet == 0, .6 if theme != "population" else .2, 0)
                themes[theme] = (f'<g clip-path="url(#tile-land-detail-0-0)"><path d="{land}" '
                                 f'{spec["attribute"]}="{classes[theme]}" fill="#629b7e" fill-rule="evenodd"/></g>')
            payload = {"bounds": [0, 0, 2, 2], "surface": '<defs>'
                       '<clipPath id="tile-land-detail-0-0" clipPathUnits="userSpaceOnUse">'
                       f'<path d="{land}" clip-rule="evenodd"/></clipPath>'
                       '<clipPath id="tile-water-detail-0-0" clipPathUnits="userSpaceOnUse">'
                       f'<path d="{water}" clip-rule="evenodd"/></clipPath></defs>',
                       "themes": themes, "ink": ""}
            tile_path = review / "tiles/detail/0-0.json"
            tile_path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = {"width": 2, "height": 2, "tileSize": 32, "rows": 1, "columns": 1,
                        "themes": list(themes), "levels": [{"id": "detail", "minScale": 64}]}
            result = audit._check_detail(review, manifest, wet, fields)
            self.assertEqual(result["checkedTiles"], 1)
            self.assertTrue(all(metric["mismatchCount"] == 0 for metric in result["themes"].values()))
            payload["themes"]["vegetation"] = payload["themes"]["vegetation"].replace('data-vegetation-band="6"', 'data-vegetation-band="0"')
            tile_path.write_text(json.dumps(payload), encoding="utf-8")
            failed = audit._check_detail(review, manifest, wet, fields)
            self.assertEqual(failed["themes"]["vegetation"]["wrongClassCenters"], 3)
            self.assertEqual(failed["themes"]["vegetation"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
