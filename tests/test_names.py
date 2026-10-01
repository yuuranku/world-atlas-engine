"""Naming-layer topology and physical-feature contracts."""

from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.society.model import GeographicFeature, Language, NameLexicon
from world_atlas.core.society.names import _interior_depth, extract_geographic_features


def _erosion_oracle(mask: np.ndarray) -> np.ndarray:
    """The former periodic-x / finite-y erosion definition of interior depth."""

    current = np.asarray(mask, dtype=bool).copy()
    depth = np.zeros(current.shape, dtype=np.float64)
    level = 1.0
    while np.any(current):
        depth[current] = level
        padded = np.pad(current, ((1, 1), (0, 0)), mode="constant")
        current = (
            current
            & np.roll(current, 1, axis=1)
            & np.roll(current, -1, axis=1)
            & padded[:-2]
            & padded[2:]
        )
        level += 1.0
    return depth


def _lexicon() -> NameLexicon:
    return NameLexicon(
        major_country_names=tuple(f"大名{index}" for index in range(19)),
        major_country_roles=tuple("regional" for _index in range(19)),
        minor_country_names=tuple(f"小名{index}" for index in range(120)),
        place_roots=("岚芜", "澄宛", "祁棠"),
    )


class InteriorDepthTests(unittest.TestCase):
    def test_distance_transform_matches_previous_wrapped_erosion(self):
        rng = np.random.default_rng(163)
        for shape in ((1, 1), (1, 9), (9, 1), (5, 7), (17, 31), (31, 17)):
            for density in (0.0, 0.08, 0.45, 0.92, 1.0):
                mask = rng.random(shape) < density
                np.testing.assert_array_equal(
                    _interior_depth(mask),
                    _erosion_oracle(mask),
                )


class GeographicFeatureNamingTests(unittest.TestCase):
    def test_physical_inventory_and_anchors_do_not_depend_on_agricultural_capacity(self):
        shape = (48, 96)
        water = np.ones(shape, dtype=np.uint8)
        water[12:34, 20:70] = 0
        water[18:21, 78:81] = 0
        water[27:30, 83:86] = 0
        rows, columns = np.indices(shape)
        elevation = np.where(water == 0, .2 + .0002*rows + .0001*columns, 0.).astype(np.float32)
        grid = SimpleNamespace(shape=shape, metadata={
            "extents": {"north": 90., "south": -90., "east": 180., "west": -180.},
            "planet": {"axialTiltDegrees": 23.44, "radiusKm": 6371.},
        }, water=water, elevation=elevation,
            river_order=np.zeros(shape,dtype=np.uint8), flow_to=np.full(shape,-1,dtype=np.int32),
            discharge=np.zeros(shape,dtype=np.float32), snow=np.zeros(shape,dtype=bool))
        empty = np.zeros(shape,dtype=bool)
        thematic = SimpleNamespace(land_potential=np.zeros(shape,dtype=np.float32),
            drainage_basin=np.full(shape,-1,dtype=np.int32),
            climate=SimpleNamespace(annual_precipitation_mm=np.full(shape,720.,dtype=np.float32)),
            physiography=SimpleNamespace(desert=empty,wetland=empty,wetland_support=empty.astype(np.float32)))
        cultures = SimpleNamespace(language_id=np.ones(shape,dtype=np.int16), languages=(
            Language(1,"测试语","lineage-s00-01",1,"core"),))
        raw = np.where(water == 0, elevation.astype(float)*1000., -100.)
        expected = extract_geographic_features(grid,thematic,cultures,_lexicon(),raw_elevation_m=raw)
        self.assertTrue(any(item.feature_type == "plain" for item in expected))
        self.assertTrue(any(item.feature_type == "island" for item in expected))
        for potential in (np.ones(shape,dtype=np.float32),
                          np.random.default_rng(173).random(shape).astype(np.float32)):
            with self.subTest(potential=potential[0,0]):
                thematic.land_potential = potential
                self.assertEqual(extract_geographic_features(
                    grid,thematic,cultures,_lexicon(),raw_elevation_m=raw),expected)

    def test_desert_wetland_and_peak_features_stay_on_their_physical_masks(self):
        shape = (30, 40)
        water = np.zeros(shape, dtype=np.uint8)
        elevation = np.full(shape, 0.20, dtype=np.float32)
        elevation[20, 25] = 0.94
        elevation[19:22, 24:27] = np.maximum(elevation[19:22, 24:27], 0.48)
        elevation[20, 25] = 0.94
        desert = np.zeros(shape, dtype=bool)
        desert[3:11, 4:14] = True
        wetland = np.zeros(shape, dtype=bool)
        wetland[12:20, 20:33] = True
        annual_precipitation = np.full(shape, 720.0, dtype=np.float32)
        annual_precipitation[desert] = 120.0
        grid = SimpleNamespace(
            shape=shape,
            metadata={
                "extents": {"north": 90.0, "south": -90.0, "east": 180.0, "west": -180.0},
                "planet": {"axialTiltDegrees": 23.44, "radiusKm": 6371.0},
            },
            water=water,
            elevation=elevation,
            river_order=np.zeros(shape, dtype=np.uint8),
            flow_to=np.full(shape, -1, dtype=np.int32),
            discharge=np.zeros(shape, dtype=np.float32),
            snow=np.zeros(shape, dtype=bool),
        )
        thematic = SimpleNamespace(
            land_potential=np.full(shape, 0.55, dtype=np.float32),
            habitability=np.full(shape, 0.65, dtype=np.float32),
            drainage_basin=np.full(shape, -1, dtype=np.int32),
            climate=SimpleNamespace(annual_precipitation_mm=annual_precipitation),
            physiography=SimpleNamespace(
                desert=desert,
                wetland=wetland,
                wetland_support=wetland.astype(np.float32),
            ),
        )
        language = Language(
            identifier=1,
            name="测试语",
            name_family="lineage-s00-01",
            family_identifier=1,
            core_settlement_id="core",
        )
        cultures = SimpleNamespace(
            language_id=np.ones(shape, dtype=np.int16),
            languages=(language,),
        )
        raw_elevation = np.full(shape, 100.0, dtype=np.float64)
        raw_elevation[19:22, 24:27] = 500.0
        raw_elevation[20, 25] = 1800.0
        features = extract_geographic_features(
            grid, thematic, cultures, _lexicon(), raw_elevation_m=raw_elevation,
        )
        by_type = {
            feature_type: [
                feature for feature in features if feature.feature_type == feature_type
            ]
            for feature_type in ("desert", "wetland", "peak")
        }
        self.assertTrue(by_type["desert"])
        self.assertTrue(by_type["wetland"])
        self.assertTrue(by_type["peak"])
        self.assertTrue(
            all(desert[item.row, item.column] for item in by_type["desert"])
        )
        self.assertTrue(
            all(wetland[item.row, item.column] for item in by_type["wetland"])
        )
        self.assertTrue(all((item.row, item.column) == (20, 25) for item in by_type["peak"]))
        self.assertTrue(all(item.name.endswith("沙漠") for item in by_type["desert"]))
        self.assertTrue(all(item.name.endswith("湿地") for item in by_type["wetland"]))

    def test_model_accepts_new_physical_feature_types(self):
        for feature_type in ("desert", "wetland"):
            item = GeographicFeature(
                identifier=f"{feature_type}-01",
                feature_type=feature_type,
                name=f"测试{feature_type}",
                row=2,
                column=3,
                language_identifier=1,
                tier="secondary",
            )
            self.assertEqual(item.feature_type, feature_type)


if __name__ == "__main__":
    unittest.main()
