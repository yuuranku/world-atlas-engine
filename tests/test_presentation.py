from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from world_atlas.core.presentation import (
    CITY_RECIPES_GLOBAL,
    PRESENTATION_SCHEMA,
    derive_presentation_assets,
    write_presentation_assets,
)
from world_atlas.core.society.model import (
    Civilization,
    CultureLayers,
    GovernmentForm,
    Language,
    PoliticalEntity,
    PoliticalLayers,
    PopulationLayers,
    Province,
    ProvinceLayers,
    Religion,
    ReligionLayers,
    Settlement,
    SocietyLayers,
    State,
    TransportLayers,
)


def _fixture() -> tuple[SimpleNamespace, SocietyLayers]:
    height, width = 8, 12
    water = np.zeros((height, width), dtype=np.uint8)
    water[0, :] = 1
    water[:, 0] = 1
    river_order = np.zeros((height, width), dtype=np.uint8)
    river_order[4, 5] = 4
    river_order[5, 5] = 3
    elevation = (
        np.arange(height, dtype=np.float32)[:, np.newaxis] * 0.7
        + np.arange(width, dtype=np.float32)[np.newaxis, :] * 0.12
    )
    grid = SimpleNamespace(
        shape=(height, width),
        water=water,
        river_order=river_order,
        flow_to=np.full((height,width),-1,dtype=np.int32),
        elevation=elevation,
        metadata={
            "societyGeneration": {"humanSeed": 123456789},
            "worldProfile": {
                "name": "Test World",
                "seed": 987654321,
                "technologyEra": "industrial",
            },
            "planet": {"radiusKm": 40.0},
            "extents":{"west":-180.,"east":180.,"north":90.,"south":-90.},
        },
        content_digest=lambda: "grid-test-digest",
    )
    population = PopulationLayers(
        population_weight=np.ones((height, width), dtype=np.float32),
        population_band=np.ones((height, width), dtype=np.uint8),
        population_min=100_000,
        population_max=400_000,
    )
    civilization = Civilization(
        identifier=1,
        name="Test Civilization",
        name_family="test",
        core_settlement_id="port-city",
        internal_diversity=("north", "south", "coast"),
    )
    language = Language(
        identifier=1,
        name="Test Tongue",
        name_family="test",
        family_identifier=1,
        core_settlement_id="port-city",
    )
    cultures = CultureLayers(
        civilization_id=np.ones((height, width), dtype=np.int16),
        civilization_influence=np.ones((height, width), dtype=np.uint8),
        language_family_id=np.ones((height, width), dtype=np.int16),
        language_id=np.ones((height, width), dtype=np.int16),
        language_contact=np.zeros((height, width), dtype=bool),
        civilizations=(civilization,),
        languages=(language,),
    )
    religion = Religion(
        identifier=1,
        name="Test Faith",
        tradition="solar",
        holy_settlement_id="port-city",
        origin_civilization_identifier=1,
        worldview="ordered cosmos",
        moral_ideal="stewardship",
        institution="temple network",
    )
    religions = ReligionLayers(
        religion_id=np.ones((height, width), dtype=np.int16),
        religions=(religion,),
    )
    state = State(
        identifier=1,
        name="Test State",
        core_settlement_id="port-city",
        size_class="major",
        population_min=100_000,
        population_max=400_000,
        civilization_identifier=1,
        language_identifier=1,
    )
    government = GovernmentForm(1, "civic", "Civic", "A civic government")
    politics = PoliticalLayers(
        state_id=np.ones((height, width), dtype=np.int16),
        frontier=np.zeros((height, width), dtype=bool),
        states=(state,),
        government_forms=(government,),
        political_entities=(PoliticalEntity(1, 1, 1, "Test State"),),
        frontier_groups=(),
    )
    province = Province(
        identifier=1,
        name="Test District",
        state_identifier=1,
        core_settlement_id="port-city",
        region_type="coastal",
        administrative_system="civic-administration",
        administrative_function="capital",
        administrative_rank="district",
        population_density_class="dense",
        area_cells=height * width,
    )
    settlements = (
        Settlement(
            identifier="port-city",
            name="Port City",
            row=3,
            column=1,
            tier="metropolis",
            site_type="port",
            score=1.0,
            population_min=200_000,
            population_max=500_000,
            holy_religion_identifier=1,
        ),
        Settlement(
            identifier="river-town",
            name="River Town",
            row=4,
            column=6,
            tier="city",
            site_type="river-city",
            score=0.7,
            population_min=60_000,
            population_max=120_000,
        ),
    )
    society = SocietyLayers(
        population=population,
        settlements=settlements,
        transport=TransportLayers(
            accessibility=np.zeros((height, width), dtype=np.float32),
            routes=(),
        ),
        cultures=cultures,
        religions=religions,
        geographic_features=(),
        politics=politics,
        provinces=ProvinceLayers(
            province_id=np.ones((height, width), dtype=np.int32),
            provinces=(province,),
        ),
    )
    return grid, society


class PresentationAssetsTests(unittest.TestCase):
    def test_two_sided_waterfront_keeps_the_selected_bank_and_all_cores_on_land(self):
        grid, society = _fixture()
        grid.water[3, 2] = 1
        recipes = derive_presentation_assets(grid, society).city_recipes
        port = next(item for item in recipes if item.identifier == "port-city")
        self.assertEqual(port.water_bearing_degrees, 180)
        self.assertLess(port.center_column, port.column + 0.5)
        allowed = set(port.buildable_cells)
        for core in port.core_zones:
            self.assertIn((int(np.floor(core.row)), int(np.floor(core.column)) % grid.shape[1]), allowed)

    def test_city_recipes_are_deterministic_and_context_aware(self):
        grid, society = _fixture()
        first = derive_presentation_assets(grid, society, engine_version="test-engine")
        second = derive_presentation_assets(grid, society, engine_version="test-engine")

        self.assertEqual(first.manifest_document(), second.manifest_document())
        manifest = first.manifest_document()
        self.assertEqual(manifest["schema"], PRESENTATION_SCHEMA)
        self.assertEqual(manifest["humanSeed"], 123456789)
        self.assertEqual(manifest["worldSeed"], 987654321)
        self.assertEqual(manifest["era"], "industrial")
        self.assertEqual(len(manifest["cityRecipes"]), len(society.settlements))
        self.assertFalse(manifest["lazyAssets"]["preRenderedCityDom"])
        self.assertEqual(len(manifest["urbanClusters"]), 1)
        recipes = {recipe["id"]: recipe for recipe in manifest["cityRecipes"]}
        port = recipes["port-city"]
        river = recipes["river-town"]
        self.assertEqual(port["water"]["kind"], "seaport")
        self.assertIn("harbor", port["landmarks"])
        self.assertIn("government-seat", port["landmarks"])
        self.assertEqual(port["anchorCell"], {"row": 3, "column": 1})
        self.assertEqual(port["location"]["row"], 3.5)
        self.assertGreater(port["location"]["column"], 1.0)
        self.assertLess(port["location"]["column"], 1.5)
        self.assertLess(port["urban"]["bounds"]["west"], 1.0,
                        "waterfront footprint must meet the shared coast and be clipped to land")
        self.assertEqual(port["anchorCell"], {"row": 3, "column": 1})
        self.assertEqual(port["urban"]["coordinateSpace"], "world-grid-cells")
        self.assertTrue(port["urban"]["clipToLand"])
        self.assertGreater(port["urban"]["gridCellKilometres"]["row"], 0.0)
        self.assertGreater(port["urban"]["streetWidthsMetres"]["local"], 0.0)
        self.assertEqual(len(port["urban"]["coreZones"]), port["urban"]["coreCount"])
        self.assertEqual(port["urban"]["coreZones"][0]["row"], 3.5)
        self.assertEqual(port["urban"]["coreZones"][0]["column"], port["location"]["column"])
        self.assertEqual(port["urban"]["cluster"]["memberCount"], 2)
        self.assertNotIn("rail-reserve", port["transport"]["interfaces"])
        self.assertEqual(port["transport"]["railConnections"], 0)
        self.assertEqual(river["water"]["kind"], "riverfront")
        self.assertGreater(river["water"]["riverOrder"], 0)
        self.assertGreater(port["urban"]["targetFootprintKm2"], 0.0)
        self.assertEqual(
            port["urban"]["areaSemantics"],
            "target-before-native-terrain-mask",
        )
        self.assertGreaterEqual(port["urban"]["coreCount"], 1)
        self.assertIn(port["terrain"]["slopeClass"], {"flat", "gentle", "rolling", "steep", "escarpment"})
        buildable = port["terrain"]["buildable"]
        self.assertEqual(buildable["coordinateSpace"], "world-grid-cells")
        self.assertEqual(buildable["precision"], "native-grid-cell")
        self.assertEqual(buildable["allowedCellCount"], len(buildable["allowedCells"]))
        self.assertEqual(buildable["blockedCellCount"], len(buildable["blockedCells"]))
        self.assertTrue(buildable["clipCityGeometry"])
        self.assertIn([3, 1], buildable["allowedCells"])
        for candidate_row, candidate_column in buildable["allowedCells"]:
            self.assertEqual(int(grid.water[candidate_row, candidate_column]), 0)
            self.assertEqual(int(grid.river_order[candidate_row, candidate_column]), 0)

    def test_writer_emits_data_only_external_script(self):
        grid, society = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            assets = write_presentation_assets(directory, grid, society, engine_version="test-engine")
            output = Path(directory)
            manifest = json.loads((output / "presentation.json").read_text(encoding="utf-8"))
            script = (output / "city-recipes.js").read_text(encoding="utf-8")
            self.assertEqual(manifest, assets.manifest_document())
            self.assertIn(CITY_RECIPES_GLOBAL, script)
            self.assertIn("load-on-zoom", script)
            self.assertNotIn("createElement", script)
            self.assertNotIn("<svg", script)

    def test_writer_rejects_a_city_on_a_represented_river_channel(self):
        grid, society = _fixture()
        channel_city = replace(society.settlements[1], row=4, column=5)
        invalid = replace(society, settlements=(society.settlements[0], channel_city))
        with self.assertRaisesRegex(ValueError, "river channel"):
            derive_presentation_assets(grid, invalid)

    def test_each_formal_technology_era_has_a_distinct_strict_city_profile(self):
        grid, society = _fixture()
        for era in (
            "tribal",
            "ancient",
            "medieval",
            "early-modern",
            "preindustrial",
            "industrial",
            "contemporary",
        ):
            profile = dict(grid.metadata["worldProfile"])
            profile["technologyEra"] = era
            era_grid = SimpleNamespace(**{**grid.__dict__, "metadata": {**grid.metadata, "worldProfile": profile}})
            manifest = derive_presentation_assets(era_grid, society).manifest_document()
            self.assertEqual(manifest["era"], era)
            self.assertTrue(all(recipe["era"] == era for recipe in manifest["cityRecipes"]))


if __name__ == "__main__":
    unittest.main()
