"""Deterministic climate and physiography contracts."""

from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.thematic import (
    ClimateDrivers,
    KOPPEN_AF,
    KOPPEN_AM,
    KOPPEN_AS,
    KOPPEN_BWH,
    KOPPEN_BWK,
    KOPPEN_CFB,
    KOPPEN_CSA,
    KOPPEN_CWA,
    KOPPEN_DFC,
    KOPPEN_EF,
    KOPPEN_ET,
    classify_koppen_geiger,
    derive_monthly_climate_estimate,
    derive_physiographic_features,
)


def _climate_drivers(shape: tuple[int, int], codes: np.ndarray) -> ClimateDrivers:
    return ClimateDrivers(
        temperature=np.full(shape, 0.60, dtype=np.float64),
        seasonal_precipitation=np.full((4, *shape), 0.10, dtype=np.float64),
        annual_precipitation=np.full(shape, 0.10, dtype=np.float64),
        precipitation_range=np.zeros(shape, dtype=np.float64),
        mean_annual_temperature_c=np.full(shape, 12.0, dtype=np.float32),
        annual_precipitation_mm=np.full(shape, 720.0, dtype=np.float32),
        coldest_month_temperature_c=np.full(shape, 3.0, dtype=np.float32),
        warmest_month_temperature_c=np.full(shape, 20.0, dtype=np.float32),
        koppen_code=codes.astype(np.int8),
    )


class ThematicClimateTests(unittest.TestCase):
    def test_rainforest_is_not_overwritten_by_wetter_winter(self):
        temperature = np.full((12, 1, 1), 26.0, dtype=np.float32)
        rain = np.full_like(temperature, 200.0)
        rain[5] = 80.0
        codes = classify_koppen_geiger(temperature, rain, latitude_degrees=np.array([[10.0]]), land_mask=np.ones((1, 1), bool))
        self.assertEqual(codes[0, 0], KOPPEN_AF)

    def test_double_dry_season_uses_half_year_totals_in_both_hemispheres(self):
        temperature = np.array([5, 7, 12, 16, 20, 24, 25, 24, 20, 14, 10, 6], dtype=np.float32)[:, None, None]
        rain = np.array([1, 90, 90, 300, 300, 5, 300, 300, 300, 90, 90, 90], dtype=np.float32)[:, None, None]
        for latitude, offset in ((35.0, 0), (-35.0, 6)):
            codes = classify_koppen_geiger(np.roll(temperature, offset, axis=0), np.roll(rain, offset, axis=0), latitude_degrees=np.array([[latitude]]), land_mask=np.ones((1, 1), bool))
            self.assertEqual(codes[0, 0], KOPPEN_CWA)

    def test_aridity_precedes_polar_temperature_class(self):
        temperature = np.full((12, 1, 1), 5.0, dtype=np.float32)
        rain = np.full_like(temperature, 1.0)
        codes = classify_koppen_geiger(temperature, rain, latitude_degrees=np.array([[60.0]]), land_mask=np.ones((1, 1), bool))
        self.assertEqual(codes[0, 0], KOPPEN_BWK)

    def test_koppen_rules_cover_tropical_arid_temperate_continental_and_polar(self):
        count = 10
        temperature = np.full((12, 1, count), 25.0, dtype=np.float32)
        precipitation = np.full((12, 1, count), 80.0, dtype=np.float32)

        # Af, Am, Aw, BWh, BWk, Csa, Cfb, Dfc, ET, EF.
        precipitation[:, 0, 0] = 200.0
        precipitation[:, 0, 1] = 200.0
        precipitation[0, 0, 1] = 50.0
        precipitation[:, 0, 2] = 120.0
        precipitation[5, 0, 2] = 5.0
        precipitation[:, 0, 3] = 10.0
        temperature[:, 0, 4] = 10.0
        precipitation[:, 0, 4] = 10.0
        temperature[:, 0, 5] = np.asarray(
            (5, 6, 10, 14, 18, 22, 25, 24, 20, 15, 10, 6),
            dtype=np.float32,
        )
        precipitation[:, 0, 5] = np.asarray(
            (100, 100, 100, 10, 10, 10, 10, 10, 10, 100, 100, 100),
            dtype=np.float32,
        )
        temperature[:, 0, 6] = np.asarray(
            (5, 6, 8, 11, 14, 17, 19, 18, 15, 11, 8, 6),
            dtype=np.float32,
        )
        temperature[:, 0, 7] = np.asarray(
            (-16, -15, -8, 0, 7, 12, 15, 14, 8, 2, -6, -13),
            dtype=np.float32,
        )
        temperature[:, 0, 8] = np.asarray(
            (-5, -4, -2, 0, 2, 4, 6, 5, 3, 0, -3, -4),
            dtype=np.float32,
        )
        temperature[:, 0, 9] = np.asarray(
            (-20, -19, -17, -14, -10, -6, -5, -6, -9, -13, -17, -19),
            dtype=np.float32,
        )
        precipitation[:, 0, 9] = 20.0

        codes = classify_koppen_geiger(
            temperature,
            precipitation,
            latitude_degrees=np.full((1, count), 45.0, dtype=np.float32),
            land_mask=np.ones((1, count), dtype=bool),
        )
        np.testing.assert_array_equal(
            codes[0],
            np.asarray(
                (
                    KOPPEN_AF,
                    KOPPEN_AM,
                    KOPPEN_AS,
                    KOPPEN_BWH,
                    KOPPEN_BWK,
                    KOPPEN_CSA,
                    KOPPEN_CFB,
                    KOPPEN_DFC,
                    KOPPEN_ET,
                    KOPPEN_EF,
                ),
                dtype=np.int8,
            ),
        )
        self.assertFalse(codes.flags.writeable)

    def test_monthly_estimate_preserves_four_season_annual_precipitation(self):
        shape = (3, 4)
        seasonal = np.empty((4, *shape), dtype=np.uint8)
        for index, value in enumerate((10, 20, 30, 40)):
            seasonal[index].fill(value)
        grid = SimpleNamespace(
            shape=shape,
            metadata={"extents": {"north": 45.0, "south": -45.0}},
            elevation=np.full(shape, 0.20, dtype=np.float32),
            water=np.zeros(shape, dtype=np.uint8),
            snow=np.zeros(shape, dtype=bool),
            seasonal_precipitation=seasonal,
        )
        estimate = derive_monthly_climate_estimate(grid)
        expected_annual = 8400.0 * np.mean(np.asarray((10, 20, 30, 40)) / 255.0)
        self.assertEqual(estimate.temperature_c.shape, (12, *shape))
        self.assertEqual(estimate.precipitation_mm.shape, (12, *shape))
        np.testing.assert_allclose(
            estimate.precipitation_mm.sum(axis=0),
            expected_annual,
            rtol=2.0e-6,
        )
        self.assertFalse(estimate.temperature_c.flags.writeable)
        self.assertFalse(estimate.precipitation_mm.flags.writeable)

    def test_zero_axial_tilt_has_no_synthetic_annual_temperature_cycle(self):
        shape = (3, 4)
        grid = SimpleNamespace(
            shape=shape,
            metadata={
                "extents": {"north": 70.0, "south": -70.0},
                "planet": {"axialTiltDegrees": 0.0},
            },
            elevation=np.full(shape, 0.20, dtype=np.float32),
            water=np.zeros(shape, dtype=np.uint8),
            snow=np.zeros(shape, dtype=bool),
            seasonal_precipitation=np.full((4, *shape), 32, dtype=np.uint8),
        )
        estimate = derive_monthly_climate_estimate(grid)
        np.testing.assert_allclose(
            np.ptp(estimate.temperature_c, axis=0),
            0.0,
            atol=0.0,
        )

    def test_physiographic_masks_are_physical_and_land_limited(self):
        shape = (9, 9)
        water = np.zeros(shape, dtype=np.uint8)
        water[4, 4] = 2
        elevation = np.full(shape, 0.20, dtype=np.float32)
        codes = np.full(shape, KOPPEN_CFB, dtype=np.int8)
        codes[1, 1] = KOPPEN_BWH
        climate = _climate_drivers(shape, codes)
        grid = SimpleNamespace(
            shape=shape,
            metadata={"extents": {"west": 0.0, "east": .9, "south": -.45, "north": .45},
                      "planet": {"radiusKm": 6400.0}},
            water=water,
            elevation=elevation,
            snow=np.zeros(shape, dtype=bool),
            river_order=np.zeros(shape, dtype=np.uint8),
            discharge=np.zeros(shape, dtype=np.float32),
            flow_to=np.asarray([((row + np.sign(4 - row)) * 9
                                  + column + np.sign(4 - column))
                                 if (row, column) != (4, 4) else -1
                                 for row in range(9) for column in range(9)],
                                dtype=np.int32).reshape(shape),
            seasonal_river_strength=np.zeros((4, *shape), dtype=np.uint8),
        )
        features = derive_physiographic_features(grid, climate)
        self.assertTrue(features.desert[1, 1])
        self.assertFalse(features.desert[4, 4])
        self.assertGreater(int(features.wetland.sum()), 0)
        self.assertFalse(features.wetland[4, 4])
        self.assertFalse(features.desert.flags.writeable)
        self.assertFalse(features.wetland_support.flags.writeable)


if __name__ == "__main__":
    unittest.main()
