from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from world_atlas.core.city_map_assets import CITY_MAP_SCHEMA, write_city_map_assets
from world_atlas.core.presentation import derive_presentation_assets
from world_atlas.core.society.model import TransportRoute

from test_presentation import _fixture
from test_city_site import _terrain


class CityMapAssetsTests(unittest.TestCase):
    def test_window_recipe_uses_atlas_anchor_for_bounds_cores_and_access_road(self):
        grid, society = _fixture()
        anchors = {"port-city": (3.6, 1.7), "river-town": (4.5, 6.5)}
        route = TransportRoute("road-to-port", "road", "regional", "port-city", "river-town",
                               ((1.5, 3.5), (3.5, 3.5), (6.5, 4.5)))
        society = replace(society, transport=replace(society.transport, routes=(route,)))
        recipe = next(item for item in derive_presentation_assets(
            grid, society, settlement_locations=anchors,
        ).city_recipes if item.identifier == "port-city")
        self.assertEqual((recipe.center_row, recipe.center_column), anchors["port-city"])
        self.assertAlmostEqual((recipe.footprint_bounds[0] + recipe.footprint_bounds[2]) / 2, 3.6)
        self.assertAlmostEqual((recipe.footprint_bounds[1] + recipe.footprint_bounds[3]) / 2, 1.7)
        self.assertEqual((recipe.core_zones[0].row, recipe.core_zones[0].column), anchors["port-city"])
        self.assertIn((1.7, 3.6), recipe.transport_corridors[0].points)
        self.assertEqual(recipe.road_connections, 1)
        self.assertIn((3, 1), recipe.buildable_cells)

    def test_export_is_deterministic_individual_json_without_an_eager_payload(self):
        grid, society = _fixture()
        anchors = {city.identifier: (city.row + .5, city.column + .5) for city in society.settlements}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            terrain = _terrain(grid.elevation * 100)
            write_city_map_assets(output, grid, society, anchors, terrain_field=terrain,harbors={},physical_paths=())
            filenames = sorted(path.name for path in (output / "city-maps").iterdir())
            self.assertEqual(filenames, ["port-city.json", "river-town.json"])
            first = (output / "city-maps/port-city.json").read_bytes()
            payload = json.loads(first)
            self.assertEqual(payload["schema"], CITY_MAP_SCHEMA)
            self.assertEqual(payload["grid"], {"height": 8, "width": 12})
            self.assertEqual(payload["gridDigest"], grid.content_digest())
            self.assertEqual(payload["recipe"]["location"], {"row": 3.5, "column": 1.5})
            self.assertEqual(payload["recipe"]["population"]["minimum"], 200_000)
            self.assertEqual(payload["recipe"]["era"], "industrial")
            self.assertEqual(payload["recipe"]["water"]["kind"], "seaport")
            self.assertEqual(payload["siteContext"]["rows"], 17)
            self.assertEqual(len(payload["siteContext"]["elevationMetres"]), 289)
            self.assertNotIn("localSite", payload["recipe"])
            self.assertFalse((output / "city-recipes.js").exists())
            self.assertFalse((output / "presentation.json").exists())
            write_city_map_assets(output, grid, society, anchors, terrain_field=terrain,harbors={},physical_paths=())
            self.assertEqual((output / "city-maps/port-city.json").read_bytes(), first)

    def test_export_rejects_missing_or_displaced_anchors_before_writing(self):
        grid, society = _fixture()
        terrain = _terrain(grid.elevation * 100)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "every settlement"):
                write_city_map_assets(directory, grid, society, {"port-city": (3.5, 1.5)}, terrain_field=terrain,harbors={},physical_paths=())
            with self.assertRaisesRegex(ValueError, "native land cell"):
                write_city_map_assets(directory, grid, society,
                                      {"port-city": (3.5, .5), "river-town": (4.5, 6.5)}, terrain_field=terrain,harbors={},physical_paths=())
            self.assertFalse((Path(directory) / "city-maps").exists())
