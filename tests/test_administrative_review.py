"""Selective publication keeps existing physical assets and shared admin arcs."""

from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import shapely

from world_atlas.core import administrative_review as refresh
from world_atlas.core import render
from world_atlas.core.cartographic_features import line_features, overview_markup
from world_atlas.core.cartographic_tiles import TileFeature, TileLevel, write_atlas_tiles
from world_atlas.core.globe_assets import THEME_EXPORT_FILENAMES
from world_atlas.core.review_interface import load_interface_snapshot, write_interface_snapshot
from world_atlas.core.svg_groups import extract_group


def document(text):
    return ET.fromstring(text.replace('xmlns="http://www.w3.org/2000/svg"', ""))


class AdministrativeReviewTests(unittest.TestCase):
    def setUp(self):
        self.grid = SimpleNamespace(shape=(32, 64), water=np.zeros((32, 64), dtype=np.int16),
                                    content_digest=lambda: "physical")
        states = np.where(np.indices((32, 64))[1] < 32, 1, 2).astype(np.int16)
        provinces = np.where(states == 2, 3,
            np.where(np.indices((32, 64))[0] < 16, 1, 2)).astype(np.int16)
        self.society = SimpleNamespace(
            politics=SimpleNamespace(state_id=states, states=[SimpleNamespace(identifier=i) for i in (1, 2)]),
            provinces=SimpleNamespace(province_id=provinces, provinces=[
                SimpleNamespace(identifier=i, state_identifier=1 if i < 3 else 2) for i in (1, 2, 3)]))
        self.front = SimpleNamespace(province_id=provinces)
        self.land = shapely.box(0, 0, 64, 32)
        self.labels = np.asarray((1, 2, 3), dtype=np.int32)
        self.parents = np.asarray((0, 1, 1, 2), dtype=np.int32)
        self.original_faces = np.asarray((shapely.box(0, 0, 32, 16),
            shapely.box(0, 16, 32, 32), shapely.box(32, 0, 64, 32)))
        self.changed_faces = np.asarray((
            shapely.Polygon(((0, 0), (32, 0), (32.2, 8), (32.15, 16), (0, 16))),
            shapely.Polygon(((0, 16), (32.15, 16), (32.1, 24), (32, 32), (0, 32))),
            shapely.Polygon(((32, 0), (64, 0), (64, 32), (32, 32),
                (32.1, 24), (32.15, 16), (32.2, 8))),
        ))
        self.state_zones = (("frontier", "#eee"), ("Alpha", "#abc"), ("Beta", "#def"))
        self.province_zones = (("frontier", "#eee"), ("One", "#aab"), ("Two", "#bbc"), ("Three", "#ccd"))

    def features(self, faces):
        paths = render._shared_topology_zone_paths(faces, self.labels,
            category_count=4, include_zero=True)
        political = render._shared_topology_zone_paths(faces, self.parents[self.labels],
            category_count=3, include_zero=True)
        _, states, provinces = render._administrative_boundary_paths(faces, self.labels, self.parents)
        return (render._administrative_theme_features(political, paths, self.state_zones, self.province_zones),
                render._administrative_ink_features(states, provinces), political)

    def governance(self, directory, grid, society, faces, labels):
        geometry = shapely.union_all([face for face, owner in zip(faces, labels) if owner in (1, 2)])
        features = line_features([np.asarray(geometry.exterior.coords)], "nominal-realms", "#73583f", 2.2,
                                 clip="land")
        body = overview_markup(features, self.land, 64, 32, section="ink")
        (directory / "governance-overlay.svg").write_text(
            render._line_overlay_svg_document(grid, body, title="nominal realm"), encoding="utf-8")
        report = {"status": "ok", "compoundRealms": 1}
        (directory / "governance-render-check.json").write_text(json.dumps(report), encoding="utf-8")
        (directory / "governance.json").write_text('{"countries":[1,2]}', encoding="utf-8")
        return {}, report, features

    def create_source(self, source):
        themes, ink, political_paths = self.features(self.original_faces)
        source.mkdir()
        _, _, nominal = self.governance(source, self.grid, self.society,
            self.original_faces, self.labels)
        preserved_ink = line_features([np.asarray(((2, 2), (60, 26)))], "transport-network", "#777", 1)
        preserved_ink += line_features([np.asarray(((5, 0), (12, 32)))], "rivers", "#789", 1)
        biome = TileFeature(self.land, {"fill": "#aca", "clip": "land"},
                            "theme-fill", "biome", "theme")
        features = [*themes, *ink, *preserved_ink, *nominal, biome]
        write_atlas_tiles(source, 64, 32, [TileLevel(name, scale, self.land, features)
            for name, scale in (("regional", 8), ("local", 32), ("detail", 64))])
        (source / "tiles/city-detail").mkdir()
        (source / "tiles/city-detail/0-0.json").write_bytes(b'{"bounds":[0,0,32,32],"ink":"city-relief"}')
        manifest = json.loads((source / "atlas-manifest.json").read_text(encoding="utf-8"))
        manifest["levels"].append({"id": "city-detail", "kind": "relief-overlay",
                                  "minScale": 128, "publishedTiles": ["0-0"]})
        (source / "atlas-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        standalone = render._partition_overlay_svg_document(self.grid, political_paths,
            land_surface=self.land, title="old", partition_id="old-partition",
            data_attribute="state", zones=self.state_zones)
        for theme, filename in THEME_EXPORT_FILENAMES.items():
            (source / filename).write_text(standalone, encoding="utf-8")
            (source / f"globe-theme-{theme}.svg").write_text("old globe " + theme, encoding="utf-8")
            (source / f"overview-{theme}.svg").write_text("old overview " + theme, encoding="utf-8")
        (source / "physical-surface.svgz").write_bytes(gzip.compress(standalone.encode("utf-8")))
        overlay = ('<svg id="overview-source-svg"><g id="overview-source"><g id="physical-surface">'
            '<path d="M0 0L64 0L64 32L0 32Z"/></g>'
            + overview_markup([*ink, *preserved_ink, *nominal], self.land, 64, 32, section="ink")
            + '</g></svg><g id="city-layer"><path d="M8 9L10 11"/></g>')
        write_interface_snapshot(source, overlay, grid_digest="physical", society_digest="human",
            width=640, height=320, viewbox_width=64, viewbox_height=32,
            tectonic_diagnostics={}, territorial_qa={}, settlement_locations={"city": (8, 9)})
        untouched = {"index.html": b'<html><g id="overview-source"></g>overview-source.svg</html>',
            "navigation-network.json": b'{"roads":"original"}', "transport-crossings.json": b'bridges',
            "river-channels.svg": b'river geometry', "travel-profile.json": b'travel',
            "society.npz": b'native arrays', "society.json": b'original entities',
            "city-maps/city.json": b'original city recipe'}
        for filename, value in untouched.items():
            path = source / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        (source / "timing.json").write_text('{"schema":"old-timing"}', encoding="utf-8")
        (source / "qa.json").write_text(json.dumps({"gridDigest": "physical",
            "artifacts": {name: (source / name).stat().st_size for name in ("political.svg", "provinces.svg", "index.html")},
            "viewportTiles": manifest, "territorialQA": {"native": "unchanged"}}), encoding="utf-8")

    def run_refresh(self, source, target):
        with ExitStack() as stack:
            for name, value in (("society_content_digest", "human"), ("terrain_from_source", object()),
                ("continuous_land_surface", self.land), ("administrative_source", self.front),
                ("administrative_paint_coverage", (self.original_faces, self.labels)),
                ("administrative_display_coverage", self.changed_faces)):
                stack.enter_context(patch.object(refresh, name, return_value=value))
            stack.enter_context(patch.object(render, "_political_zones", return_value=self.state_zones))
            stack.enter_context(patch.object(render, "_province_zones", return_value=self.province_zones))
            stack.enter_context(patch.object(refresh, "write_governance_overlay", side_effect=self.governance))
            full_render = stack.enter_context(patch.object(render, "render_review", side_effect=AssertionError("full render")))
            result = refresh.refresh_review_administrations(self.grid, source, target,
                administrative_society=self.society, rendered_society=self.society,
                physical_source=SimpleNamespace(relative_elevation_m=np.ones(self.grid.shape)))
            full_render.assert_not_called()
            return result

    def test_refresh_synchronizes_admin_assets_and_preserves_physical_city_and_navigation_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            self.create_source(source)
            original = {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()}
            result = self.run_refresh(source, target)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["geometry"]["paintedGeometryChangedCount"], 3)
            self.assertEqual(original, {path.relative_to(source).as_posix(): path.read_bytes()
                                       for path in source.rglob("*") if path.is_file()})
            changed = {"political.svg", "provinces.svg", "overview-political.svg", "overview-provinces.svg",
                "globe-theme-political.svg", "globe-theme-provinces.svg", "governance-overlay.svg",
                "atlas-scene.svg", "overview-source.svg", "interface-presentation.json", "atlas-manifest.json",
                "qa.json", "timing.json"}
            for filename, value in original.items():
                if not filename.startswith("tiles/") and filename not in changed:
                    self.assertEqual((target / filename).read_bytes(), value, filename)
            for path in (source / "tiles").rglob("*.json"):
                relative = path.relative_to(source)
                if "city-detail" in relative.parts:
                    self.assertEqual((target / relative).read_bytes(), original[relative.as_posix()])
                    continue
                before = json.loads(original[relative.as_posix()])
                after = json.loads((target / relative).read_text(encoding="utf-8"))
                self.assertEqual(after["surface"], before["surface"])
                self.assertEqual(after["bounds"], before["bounds"])
                self.assertEqual(after["themes"]["biome"], before["themes"]["biome"])
                for layer in ("transport-network", "rivers"):
                    if f'data-tile-layer="{layer}"' in before["ink"]:
                        self.assertEqual(extract_group(after["ink"], "data-tile-layer", layer),
                                         extract_group(before["ink"], "data-tile-layer", layer))
                    else:
                        self.assertNotIn(f'data-tile-layer="{layer}"', after["ink"])
            overlay, presentation = load_interface_snapshot(target, grid_digest="physical", society_digest="human")
            self.assertEqual(presentation["settlementLocations"], {"city": [8, 9]})
            self.assertEqual(extract_group(overlay, "id", "city-layer"),
                extract_group((source / "atlas-scene.svg").read_text(encoding="utf-8"), "id", "city-layer"))
            _, expected_states, expected_provinces = render._administrative_boundary_paths(
                self.changed_faces, self.labels, self.parents)
            for layer, paths in (("state-boundaries", expected_states), ("province-boundaries", expected_provinces)):
                expected = {render._path_data(path) for path in paths}
                for theme in ("political", "provinces"):
                    if theme == "political" and layer == "province-boundaries":
                        continue
                    globe = document((target / f"globe-theme-{theme}.svg").read_text(encoding="utf-8"))
                    self.assertEqual({node.get("d") for node in globe.findall(f'.//g[@id="{layer}"]/path')}, expected)
            summary = document((target / "overview-source.svg").read_text(encoding="utf-8"))
            expected_ink = render._administrative_ink_features(expected_states, expected_provinces)
            for layer in ("state-boundaries", "province-boundaries"):
                expected = document('<svg>' + overview_markup([f for f in expected_ink if f.layer == layer],
                    self.land, 64, 32, section="ink") + '</svg>')
                self.assertEqual([node.get("d") for node in summary.findall(f'.//g[@id="{layer}"]/path')],
                    [node.get("d") for node in expected.findall(".//path")])
            with np.load(target / "administrative-display.npz", allow_pickle=False) as saved:
                data, offsets = saved["wkb"].tobytes(), saved["offsets"]
                faces = shapely.from_wkb([data[int(a):int(b)] for a, b in zip(offsets, offsets[1:])])
                self.assertTrue(np.all(shapely.equals(faces, self.changed_faces)))
                np.testing.assert_array_equal(saved["province_ids"], self.labels)
            qa = json.loads((target / "qa.json").read_text(encoding="utf-8"))
            self.assertEqual(qa["territorialQA"], {"native": "unchanged"})
            self.assertEqual(qa["artifacts"]["political.svg"], (target / "political.svg").stat().st_size)
            self.assertEqual(result["phases"]["schema"], "world-atlas-timing-v2")

    def test_refresh_rejects_a_source_that_differs_from_saved_native_ownership(self):
        owners = self.society.provinces.province_id.copy()
        owners[0, 0] = 2
        self.front = SimpleNamespace(province_id=owners)
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            self.create_source(source)
            with self.assertRaisesRegex(ValueError, "must use the saved native ownership"):
                self.run_refresh(source, target)
            record = json.loads((target / "administrative-refresh.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failed")
            self.assertEqual(self.society.provinces.province_id[0, 0], 1)
            self.assertFalse((target / "tiles").exists())

    def test_refresh_rejects_changed_native_ownership_before_creating_target(self):
        other = SimpleNamespace(politics=self.society.politics,
            provinces=SimpleNamespace(province_id=self.society.provinces.province_id.copy(),
                                      provinces=self.society.provinces.provinces))
        other.provinces.province_id[0, 0] = 2
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            with self.assertRaisesRegex(ValueError, "native ownership"):
                refresh.refresh_review_administrations(self.grid, Path(directory) / "source", target,
                    administrative_society=self.society, rendered_society=other, physical_source=object())
            self.assertFalse(target.exists())

    def test_refresh_rejects_replaced_scene_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            self.create_source(source)
            with (source / "atlas-scene.svg").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "content changed"), patch.object(refresh, "society_content_digest", return_value="human"):
                refresh.refresh_review_administrations(self.grid, source, target,
                    administrative_society=self.society, rendered_society=self.society, physical_source=object())
            self.assertFalse(target.exists())
            target.mkdir()
            with self.assertRaises(FileExistsError):
                refresh.refresh_review_administrations(self.grid, source, target,
                    administrative_society=self.society, rendered_society=self.society, physical_source=object())


if __name__ == "__main__":
    unittest.main()
