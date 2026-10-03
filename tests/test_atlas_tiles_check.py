"""Independent consumer acceptance rejects wrong native centres and paint gaps."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "actual_atlas_tiles_check", Path(__file__).resolve().parents[1] / "scripts" / "check_atlas_tiles.py",
)
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def tile_payload(row, column, width, height, land, level="detail"):
    x, y = column * 32, row * 32
    rectangle = f"M{x},{y}h{width}v{height}h-{width}Z"
    land_id, water_id = f"tile-land-{level}-{row}-{column}", f"tile-water-{level}-{row}-{column}"
    return {
        "bounds":[x,y,width,height],
        "surface":('<defs><clipPath id="%s" clipPathUnits="userSpaceOnUse"><path d="%s" clip-rule="evenodd"/></clipPath>'
                    '<clipPath id="%s" clipPathUnits="userSpaceOnUse"><path d="%s %s" clip-rule="evenodd"/></clipPath></defs>'
                    % (land_id,land,water_id,rectangle,land)),
        "themes":{"none":"", **{
            theme:f'<g clip-path="url(#{land_id})"><g data-tile-layer="{theme}"><path {attribute}="{column+1}" d="{rectangle}" fill="#abc"/></g></g>'
            for theme,attribute in (("political","data-state"),("provinces","data-province"))
        }},
        "ink":"",
    }


def fixture(root, *, rows=1):
    review = root / "review"
    grid = root / "grid"
    (review / "tiles").mkdir(parents=True)
    grid.mkdir()
    (review / "atlas-manifest.json").write_text(json.dumps({"width":64,"height":32*rows,"tileSize":32,
        "columns":2,"rows":rows,"levels":[{"id":"regional","minScale":8},
        {"id":"local","minScale":32},{"id":"detail","minScale":64},
        {"id":"city-detail","minScale":128,"kind":"relief-overlay","publishedTiles":["0-0"]}],
        "themes":["none","political","provinces"]}),encoding="utf-8")
    lands = ("M0,0 32,0 32,8 24,8 24,24 32,24 32,32 0,32Z",
             "M32,0 64,0 64,32 32,32 32,24 40,24 40,8 32,8Z")
    for level in ("regional","local","detail"):
        (review / "tiles" / level).mkdir()
        for row in range(rows):
            for column in range(2):
                land = lands[column] if row == 0 else f"M{column*32},{row*32}h32v32h-32Z"
                (review / "tiles" / level / f"{row}-{column}.json").write_text(json.dumps(tile_payload(row,column,32,32,land,level)),encoding="utf-8")
    (review / "tiles" / "city-detail").mkdir()
    (review / "tiles" / "city-detail" / "0-0.json").write_text(json.dumps(city_payload(0,0)),encoding="utf-8")
    update_stats(review)
    water = np.zeros((32*rows,64),dtype=np.uint8)
    water[8:24,24:40] = 1
    np.savez(grid / "world-grid.npz",water=water)
    for theme in ("political","provinces"):
        land = " ".join(lands) + (f" M0,32h64v{32*(rows-1)}h-64Z" if rows > 1 else "")
        (review / f"overview-{theme}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><defs>'
            '<clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
            f'<path d="{land}" clip-rule="evenodd"/></clipPath>'
            '<clipPath id="water-silhouette-clip" clipPathUnits="userSpaceOnUse">'
            f'<path d="M0,0 64,0 64,{32*rows} 0,{32*rows}Z {land}" clip-rule="evenodd"/></clipPath></defs>'
            f'<g clip-path="url(#land-silhouette-clip)"><path d="M0,0 64,0 64,{32*rows} 0,{32*rows}Z" fill="#abc"/></g></svg>',encoding="utf-8")
    return review,grid


def city_payload(row,column):
    x,y = column*32,row*32
    key = f"{row}-{column}"
    gradient = f"city-gradient-settlement-test-city-detail-{key}"
    mask = f"city-relief-mask-city-detail-{key}"
    definitions = (f'<defs><radialGradient id="{gradient}" gradientUnits="userSpaceOnUse" '
                   f'gradientTransform="matrix(12 0 0 10 {x+16} {y+16})" cx="0" cy="0" r="1">'
                   '<stop offset="0" stop-color="white"/><stop offset=".65" stop-color="white"/>'
                   '<stop offset="1" stop-color="white" stop-opacity="0"/></radialGradient>'
                   f'<mask id="{mask}" maskUnits="userSpaceOnUse" maskContentUnits="userSpaceOnUse" '
                   f'x="{x}" y="{y}" width="32" height="32" style="mask-type:alpha">'
                   f'<ellipse cx="{x+16}" cy="{y+16}" rx="12" ry="10" fill="url(#{gradient})"/>'
                   '</mask></defs>')
    wrapper = (f'<g data-city-refinement="true" mask="url(#{mask})" '
               f'clip-path="url(#tile-land-detail-{key})">')
    return {"bounds":[x,y,32,32],
            "surface":definitions,
            "ink":wrapper+'<g data-tile-layer="elevation-contours">'
                  f'<path data-city-relief="true" data-contour-kind="minor" data-height-m="100" data-source-field="accepted-continuous-ground" '
                  f'd="M{x+5},{y+5}l2,2" fill="none" vector-effect="non-scaling-stroke"/></g></g>'}


def update_stats(review):
    path = review / "atlas-manifest.json"
    manifest = json.loads(path.read_text())
    sizes = [item.stat().st_size for item in (review/"tiles").rglob("*.json")]
    manifest["stats"] = {"totalTiles":len(sizes),"maxTileBytes":max(sizes),"totalTileBytes":sum(sizes)}
    path.write_text(json.dumps(manifest),encoding="utf-8")


class AtlasTilesCoverageTests(unittest.TestCase):
    def test_pattern_definitions_are_not_classified_ground_and_refs_are_local(self):
        with tempfile.TemporaryDirectory() as temporary:
            review,grid=fixture(Path(temporary))
            target=review/"tiles/detail/0-0.json"
            payload=json.loads(target.read_text(encoding="utf8"))
            identifier="landform-pattern-wetland-tile-detail-0-0"
            definition=(f'<defs><pattern id="{identifier}" patternUnits="userSpaceOnUse" '
                        'data-landform-pattern="wetland"><path data-state="999" '
                        'd="M0,0h64v32h-64Z" fill="#bad"/></pattern></defs>')
            payload["surface"]+=definition
            payload["ink"]=(f'<g clip-path="url(#tile-land-detail-0-0)">'
                            f'<path d="M0,0h3v3h-3Z" fill="url(#{identifier})"/></g>')
            target.write_text(json.dumps(payload),encoding="utf8")
            update_stats(review)
            tile=coverage.read_tile(review,coverage.read_manifest(review),"detail",0,0)
            self.assertEqual(len(coverage.painted_regions(definition,tile,owner_attribute="data-state")),0)
            regions=coverage.tile_owner_regions(tile,"political","data-state")
            self.assertEqual(set(regions),{1})
            payload["ink"]=payload["ink"].replace(identifier,"foreign-pattern-tile-detail-0-1")
            target.write_text(json.dumps(payload),encoding="utf8")
            with self.assertRaisesRegex(ValueError,"paint-server"):
                coverage.read_tile(review,coverage.read_manifest(review),"detail",0,0)

    def test_actual_tile_clips_match_every_native_centre_and_shared_lake(self):
        with tempfile.TemporaryDirectory() as temporary:
            review,grid = fixture(Path(temporary))
            report = coverage.check_review(review,grid=grid)
        self.assertEqual(report["status"],"ok")
        self.assertEqual(report["checkedTiles"],7)
        self.assertEqual(report["nativeShoreline"]["checkedNativeCenters"],7168)
        self.assertEqual(report["nativeShoreline"]["mismatchCount"],0)
        self.assertEqual(report["levels"]["detail"]["physicalCoverage"]["landWaterOverlapArea"],0)
        self.assertEqual(report["levels"]["detail"]["themes"]["political"]["missingArea"],0)
        self.assertEqual(report["overview"]["nativeShoreline"]["mismatchCount"],0)
        self.assertEqual(report["levels"]["city-detail"]["composedBase"],"detail")
        self.assertEqual(report["levels"]["city-detail"]["checkedTiles"],1)
        self.assertEqual(report["levels"]["city-detail"]["themes"]["political"]["missingArea"],0)

    def test_actual_tile_gap_fails_even_when_a_separate_export_is_correct(self):
        with tempfile.TemporaryDirectory() as temporary:
            review,grid = fixture(Path(temporary))
            path = review / "tiles" / "detail" / "0-0.json"
            payload = json.loads(path.read_text())
            payload["themes"]["political"] = ('<g clip-path="url(#tile-land-detail-0-0)">'
                '<path data-state="1" d="M0,0 31,0 31,32 0,32Z" fill="#abc"/></g>')
            path.write_text(json.dumps(payload),encoding="utf-8")
            update_stats(review)
            (review / "political.svg").write_text('<svg><path d="M0,0 64,0 64,32 0,32Z"/></svg>',encoding="utf-8")
            report = coverage.check_review(review,grid=grid)
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["levels"]["detail"]["themes"]["political"]["missingArea"],16)
        self.assertEqual(report["levels"]["detail"]["themes"]["political"]["examples"][0]["tile"],"detail/0-0")
        self.assertEqual(report["nativeShoreline"]["mismatchCount"],0)

    def test_wrong_actual_shore_cannot_hide_behind_a_complete_thematic_fill(self):
        with tempfile.TemporaryDirectory() as temporary:
            review,grid = fixture(Path(temporary))
            path = review / "tiles" / "detail" / "0-0.json"
            path.write_text(json.dumps(tile_payload(0,0,32,32,"M0,0 31,0 31,32 0,32Z")),encoding="utf-8")
            update_stats(review)
            report = coverage.check_review(review,grid=grid)
        self.assertEqual(report["status"],"failed")
        self.assertGreater(report["nativeShoreline"]["mismatchCount"],0)
        self.assertEqual(report["levels"]["detail"]["themes"]["political"]["missingArea"],0)

    def test_foreign_clip_or_missing_land_clip_is_rejected(self):
        for markup in ('<g clip-path="url(#tile-land-detail-0-1)"><path d="M0,0 32,0 32,32 0,32Z"/></g>',
                       '<g><path d="M0,0 32,0 32,32 0,32Z"/></g>'):
            with self.subTest(markup=markup),tempfile.TemporaryDirectory() as temporary:
                review,grid = fixture(Path(temporary))
                path = review / "tiles" / "detail" / "0-0.json"
                payload = json.loads(path.read_text())
                payload["themes"]["political"] = markup
                path.write_text(json.dumps(payload),encoding="utf-8")
                update_stats(review)
                with self.assertRaisesRegex(ValueError,"clip"):
                    coverage.check_review(review,grid=grid)

    def test_bounds_and_duplicate_mounted_ids_are_rejected(self):
        for mutate in (lambda payload:payload["bounds"].__setitem__(2,33),
                       lambda payload:payload.__setitem__("ink",'<path id="tile-land-detail-0-0" d="M1,1l1,1" fill="none"/>'),
                       lambda payload:payload.__setitem__("ink",'<path id="global-coast" d="M1,1l1,1" fill="none"/>')):
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as temporary:
                review,grid = fixture(Path(temporary))
                path = review / "tiles" / "detail" / "0-0.json"
                payload = json.loads(path.read_text())
                mutate(payload)
                path.write_text(json.dumps(payload),encoding="utf-8")
                update_stats(review)
                with self.assertRaises(ValueError):
                    coverage.check_review(review,grid=grid)

    def test_owner_attributes_inherit_through_groups_and_clip_lake_water(self):
        with tempfile.TemporaryDirectory() as temporary:
            review,_ = fixture(Path(temporary))
            manifest = coverage.read_manifest(review)
            tile = coverage.read_tile(review,manifest,"detail",0,0)
            tile["payload"]["themes"]["political"] = ('<g clip-path="url(#tile-land-detail-0-0)" data-state="0">'
                '<path d="M0,0 32,0 32,32 0,32Z" fill="#abc"/></g>')
            regions = coverage.tile_owner_regions(tile,"political","data-state")
        self.assertEqual(set(regions),{0})
        self.assertTrue(regions[0].equals(tile["land"]))

    def test_actual_overview_gap_fails_even_when_every_detail_level_is_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            review,grid = fixture(Path(temporary))
            path = review / "overview-political.svg"
            source = path.read_text()
            source = source.replace('<path d="M0,0 64,0 64,32 0,32Z" fill="#abc"/>',
                                    '<path d="M0,0 63,0 63,32 0,32Z" fill="#abc"/>')
            path.write_text(source,encoding="utf-8")
            report = coverage.check_review(review,grid=grid)
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["overview"]["themes"]["political"]["missingArea"],32)
        self.assertTrue(all(level["status"] == "ok" for level in report["levels"].values()))

    def test_unreported_payload_change_fails_measured_manifest_stats(self):
        with tempfile.TemporaryDirectory() as temporary:
            review,grid = fixture(Path(temporary))
            path = review / "tiles" / "detail" / "0-0.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError,"tile bytes differ"):
                coverage.check_review(review,grid=grid)

    def test_sparse_city_level_in_two_by_two_world_does_not_publish_wilderness(self):
        with tempfile.TemporaryDirectory() as temporary:
            review,grid = fixture(Path(temporary),rows=2)
            report = coverage.check_review(review,grid=grid)
            files = sorted(path.name for path in (review/"tiles"/"city-detail").iterdir())
        self.assertEqual(report["status"],"ok")
        self.assertEqual(files,["0-0.json"])
        self.assertEqual(report["checkedTiles"],13)
        self.assertEqual(report["levels"]["city-detail"]["checkedTiles"],1)
        self.assertEqual(report["levels"]["city-detail"]["nativeShoreline"]["checkedNativeCenters"],1024)
        self.assertEqual(report["payloadBudget"]["totalTiles"],13)

    def test_undeclared_or_missing_sparse_file_is_rejected(self):
        for missing in (False,True):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as temporary:
                review,grid = fixture(Path(temporary),rows=2)
                directory = review/"tiles"/"city-detail"
                if missing:
                    (directory/"0-0.json").unlink()
                else:
                    (directory/"0-1.json").write_text(json.dumps(city_payload(0,1)))
                with self.assertRaisesRegex(ValueError,"files differ"):
                    coverage.check_review(review,grid=grid)

    def test_obsolete_three_levels_and_duplicate_sparse_coordinates_are_rejected(self):
        for mutate in (lambda manifest:manifest["levels"].pop(),
                       lambda manifest:manifest["levels"][3]["publishedTiles"].append("0-0"),
                       lambda manifest:manifest["levels"][3].__setitem__("publishedTiles",["1-0"])):
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as temporary:
                review,_ = fixture(Path(temporary))
                path = review/"atlas-manifest.json"
                manifest = json.loads(path.read_text())
                mutate(manifest)
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    coverage.read_manifest(review)

    def test_city_overlay_cannot_duplicate_themes_shore_or_escape_shared_support(self):
        mutations = (
            lambda payload:payload.__setitem__("themes",{"none":""}),
            lambda payload:payload.__setitem__("ink",payload["ink"].replace("elevation-contours","coastline")),
            lambda payload:payload.__setitem__("ink",payload["ink"].replace("clip-path=\"url(#tile-land-detail-0-0)\"","")),
            lambda payload:payload.__setitem__("surface",payload["surface"]+payload["ink"]),
            lambda payload:payload.__setitem__("ink",payload["ink"].replace('data-contour-kind="minor"','data-contour-kind="major"')),
            lambda payload:payload.__setitem__("ink",payload["ink"].replace('data-source-field="accepted-continuous-ground"','')),
            lambda payload:payload.__setitem__("ink",payload["ink"].replace("mask=\"url(#city-relief-mask-city-detail-0-0)\"","")),
            lambda payload:payload.__setitem__("surface",payload["surface"].replace("stop-opacity=\"0\"","stop-opacity=\"1\"")),
            lambda payload:payload.__setitem__("surface",payload["surface"].replace("matrix(12 0 0 10 16 16)","matrix(12 0 0 10 16 17)")),
            lambda payload:payload.__setitem__("ink",payload["ink"].replace("tile-land-detail-0-0","tile-land-local-0-0")),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as temporary:
                review,_ = fixture(Path(temporary))
                manifest = coverage.read_manifest(review)
                path = review/"tiles"/"city-detail"/"0-0.json"
                payload = json.loads(path.read_text())
                mutate(payload)
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    coverage.read_city_overlay(review,manifest,0,0)

    def test_city_payload_over_three_mib_is_rejected_by_measured_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            review,_ = fixture(Path(temporary))
            manifest = coverage.read_manifest(review)
            path = review/"tiles"/"city-detail"/"0-0.json"
            path.write_bytes(path.read_bytes()+b" "*coverage.MAX_TILE_BYTES)
            with self.assertRaisesRegex(ValueError,"3 MiB"):
                coverage.read_city_overlay(review,manifest,0,0)


if __name__ == "__main__":
    unittest.main()
