"""Actual mounted administrative tiles must retain canonical city ownership."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import shapely


spec = importlib.util.spec_from_file_location(
    "city_coverage_check", Path(__file__).resolve().parents[1] / "scripts" / "check_city_coverage.py",
)
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def published_city_fixture(review, *, tile_owner=1, canonical_owner=1, lake=False):
    (review / "tiles").mkdir()
    (review / "atlas-manifest.json").write_text(json.dumps({"width":10,"height":2,
        "tileSize":32,"columns":1,"rows":1,"levels":[{"id":"regional","minScale":8},
        {"id":"local","minScale":32},{"id":"detail","minScale":64},
        {"id":"city-detail","minScale":128,"kind":"relief-overlay","publishedTiles":[]}],
        "themes":["none","political","provinces"]}),encoding="utf-8")
    hole = " M4,0 6,0 6,2 4,2Z" if lake else ""
    land = "M0,0 10,0 10,2 0,2Z" + hole
    payload = {"bounds":[0,0,10,2],"surface":(
        '<defs><clipPath id="tile-land-0-0" clipPathUnits="userSpaceOnUse">'
        f'<path d="{land}" clip-rule="evenodd"/></clipPath>'
        '<clipPath id="tile-water-0-0" clipPathUnits="userSpaceOnUse">'
        f'<path d="M0,0 10,0 10,2 0,2Z {land}" clip-rule="evenodd"/></clipPath></defs>'),
        "themes":{"none":"",**{
            theme:'<g clip-path="url(#tile-land-0-0)">'
            f'<g {attribute}="{tile_owner}"><path d="M0,0 10,0 10,2 0,2Z" fill="#abc"/></g></g>'
            for theme,attribute in (("political","data-state"),("provinces","data-province"))
        }},"ink":""}
    for level in ("regional","local","detail"):
        (review / "tiles" / level).mkdir()
        (review / "tiles" / level / "0-0.json").write_text(
            json.dumps(payload).replace("tile-land-0-0",f"tile-land-{level}-0-0")
            .replace("tile-water-0-0",f"tile-water-{level}-0-0"),encoding="utf-8")
    (review / "tiles" / "city-detail").mkdir()
    sizes = [path.stat().st_size for path in (review/"tiles").rglob("*.json")]
    manifest_path = review / "atlas-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stats"] = {"totalTiles":len(sizes),"maxTileBytes":max(sizes),"totalTileBytes":sum(sizes)}
    manifest_path.write_text(json.dumps(manifest),encoding="utf-8")
    cities = [{"identifier":"city-one","name":"Port","row":0,"column":4}]
    (review / "society.json").write_text(json.dumps({"gridDigest":"source-grid","settlements":cities}),encoding="utf-8")
    owners = np.full((2,10),canonical_owner,dtype=np.uint8)
    np.savez(review / "society.npz",state_id=owners,province_id=owners)
    overview_land = ('<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
        f'<path d="{land}" clip-rule="evenodd"/></clipPath>'
        '<clipPath id="water-silhouette-clip" clipPathUnits="userSpaceOnUse">'
        f'<path d="M0,0 10,0 10,2 0,2Z {land}" clip-rule="evenodd"/></clipPath></defs>')
    for theme,attribute in (("political","data-state"),("provinces","data-province")):
        (review / f"overview-{theme}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg">' + overview_land
            + '<g clip-path="url(#land-silhouette-clip)">'
            + f'<path {attribute}="{canonical_owner}" d="M0,0 10,0 10,2 0,2Z" fill="#abc"/></g></svg>',encoding="utf-8")


class CityCoverageTests(unittest.TestCase):
    def test_displayed_country_cannot_take_a_city_from_its_canonical_owner(self):
        cities = [{"identifier":"city-one","name":"East","row":0,"column":4}]
        owners = np.full((2,10),2)
        result = coverage.owner_metrics(cities,owners,{1:shapely.box(0,0,5,2),2:shapely.box(5,0,10,2)})
        self.assertEqual(result["mismatchCount"],1)
        self.assertEqual(result["mismatches"][0]["expectedOwnerId"],2)
        self.assertEqual(result["mismatches"][0]["displayedOwnerIds"],[1])

    def test_unpainted_or_ambiguous_city_is_a_real_mismatch(self):
        cities = [{"identifier":"city-one","name":"Centre","row":0,"column":4}]
        owners = np.ones((2,10))
        for regions in ({1:shapely.box(5,0,10,2)},
                        {1:shapely.box(0,0,5,2),2:shapely.box(4,0,10,2)}):
            self.assertEqual(coverage.owner_metrics(cities,owners,regions)["mismatchCount"],1)

    def test_frontier_cities_are_owned_by_explicit_tile_owner_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            published_city_fixture(review,tile_owner=0,canonical_owner=0)
            report = coverage.check_review(review)
        self.assertEqual(report["status"],"ok")
        self.assertEqual(report["countries"]["checkedCities"],1)
        self.assertEqual(report["checkedCityTiles"],3)
        self.assertEqual(report["countries"]["checkedCityViews"],4)

    def test_actual_tile_owner_is_checked_even_when_exported_svg_is_correct(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            published_city_fixture(review,tile_owner=2,canonical_owner=1)
            for name in ("political","provinces"):
                (review / f"{name}.svg").write_text('<svg><g data-state="1"><path d="M0,0 10,0 10,2 0,2Z"/></g></svg>')
            report = coverage.check_review(review)
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["countries"]["mismatches"][0]["displayedOwnerIds"],[2])
        self.assertEqual(report["provinces"]["mismatchCount"],3)

    def test_actual_physical_lake_clip_removes_working_region_paint(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            published_city_fixture(review,lake=True)
            report = coverage.check_review(review)
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["countries"]["mismatches"][0]["displayedOwnerIds"],[])

    def test_missing_actual_tile_is_an_error_with_no_svg_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            published_city_fixture(review)
            (review / "tiles" / "regional" / "0-0.json").unlink()
            with self.assertRaises(FileNotFoundError):
                coverage.check_review(review)

    def test_overview_owner_error_fails_even_when_all_detail_levels_are_correct(self):
        with tempfile.TemporaryDirectory() as temporary:
            review = Path(temporary)
            published_city_fixture(review)
            path = review / "overview-political.svg"
            path.write_text(path.read_text().replace('data-state="1"','data-state="2"'),encoding="utf-8")
            report = coverage.check_review(review)
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["countries"]["mismatchCount"],1)
        self.assertEqual(report["countries"]["mismatches"][0]["view"],"overview")
        self.assertTrue(all(report["views"][level]["countries"]["mismatchCount"] == 0
                            for level in ("regional","local","detail")))


if __name__ == "__main__":
    unittest.main()
