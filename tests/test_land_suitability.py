"""Physical capacity, river supply and area-normalized population regressions."""

from types import SimpleNamespace
import unittest

import numpy as np
from ecology_fixtures import declared_freshwater_sources

from world_atlas.core.polar import polar_continent_mask
from world_atlas.core.society.model import PopulationLayers
from world_atlas.core.society.population import (
    POPULATION_DENSITY_THRESHOLDS,
    _population_bands,
    derive_population,
    population_density,
)
from world_atlas.core.suitability import continuous_proximity, relative_land_slope
from world_atlas.core.thematic import (
    HABITABILITY_THRESHOLDS,
    LAND_POTENTIAL_THRESHOLDS,
    derive_habitability,
    derive_land_potential,
)


def fixture(*, rain=0.12, temperature=0.60):
    shape = (32, 64)
    grid = SimpleNamespace(
        shape=shape,
        water=np.zeros(shape, dtype=np.uint8),
        snow=np.zeros(shape, dtype=bool),
        elevation=np.full(shape, 0.20, dtype=np.float32),
        river_order=np.zeros(shape, dtype=np.uint8),
        metadata={
            "extents": {"north": 40.0, "south": -40.0, "west": -180.0, "east": 180.0},
            "planet": {"radiusKm": 6400.0, "axialTiltDegrees": 23.44},
        },
    )
    climate = SimpleNamespace(
        annual_precipitation=np.full(shape, rain, dtype=np.float64),
        precipitation_range=np.full(shape, 0.08, dtype=np.float64),
        temperature=np.full(shape, temperature, dtype=np.float64),
    )
    return grid, climate


def themes(grid, climate):
    sources=declared_freshwater_sources(grid)
    potential, potential_band = derive_land_potential(grid,climate,ecological_sources=sources)
    habitability, habitability_band = derive_habitability(grid,climate,ecological_sources=sources)
    return SimpleNamespace(
        climate=climate,
        land_potential=potential,
        land_potential_band=potential_band,
        habitability=habitability,
        habitability_band=habitability_band,
    )


class LandSuitabilityTests(unittest.TestCase):
    def test_agricultural_rainfall_boundary_has_no_capacity_cliff(self):
        low_grid, low_climate = fixture(rain=0.035 - 1.0e-6)
        high_grid, high_climate = fixture(rain=0.035 + 1.0e-6)
        low, _ = derive_land_potential(low_grid,low_climate,ecological_sources=declared_freshwater_sources(low_grid))
        high, _ = derive_land_potential(high_grid,high_climate,ecological_sources=declared_freshwater_sources(high_grid))
        self.assertGreater(float(low[16, 16]), 0.25)
        self.assertLess(float(np.max(high - low)), 0.0001)
        self.assertTrue(np.all(high >= low))

    def test_dry_river_bank_support_survives_former_population_cutoff(self):
        populations = []
        for rain in (0.018 - 1.0e-6, 0.018 + 1.0e-6):
            grid, climate = fixture(rain=rain)
            grid.river_order[:, 32] = 3
            thematic = themes(grid, climate)
            population = derive_population(grid, thematic,
                population_min=100_000_000, population_max=160_000_000)
            self.assertGreater(float(thematic.land_potential[16, 31]), 0.45)
            self.assertGreater(float(thematic.habitability[16, 31]), 0.55)
            self.assertGreater(float(population.population_weight[16, 31]), 0.0)
            self.assertGreater(float(population.population_weight[16, 31]),
                               float(population.population_weight[16, 42]))
            populations.append(population.population_weight)
        relative_change = np.abs(populations[1] / populations[0] - 1.0)
        self.assertLess(float(relative_change.max()), 0.001)

    def test_residential_and_agricultural_capacity_have_distinct_water_needs(self):
        grid, climate = fixture(rain=0.02, temperature=0.95)
        thematic = themes(grid, climate)
        self.assertGreater(float(thematic.habitability[16, 16]),
                           float(thematic.land_potential[16, 16]) + 0.1)
        self.assertFalse(thematic.habitability.flags.writeable)
        self.assertFalse(thematic.land_potential.flags.writeable)
        self.assertEqual(LAND_POTENTIAL_THRESHOLDS, (0.2, 0.35, 0.5, 0.65, 0.8))
        self.assertEqual(HABITABILITY_THRESHOLDS, LAND_POTENTIAL_THRESHOLDS)

    def test_warm_polar_land_snow_and_water_remain_outside_human_capacity(self):
        grid, climate = fixture(rain=0.20)
        grid.metadata["extents"].update(north=85.0, south=40.0)
        grid.metadata["polarRegions"] = {"arctic": {"surface": "continental-land"}}
        grid.water[25, 25] = 2
        grid.water[25, 26] = 1
        grid.snow[26, 25] = True
        forbidden = polar_continent_mask(grid) | grid.snow | (grid.water != 0)
        thematic = themes(grid, climate)
        population = derive_population(grid, thematic,
            population_min=100_000_000, population_max=160_000_000)
        for field in (thematic.land_potential, thematic.habitability,
                      population.population_weight, population.population_band):
            self.assertTrue(np.all(field[forbidden] == 0))
        self.assertGreater(float(population.population_weight[29, 40]), 0.0)
        self.assertAlmostEqual(float(population.population_weight.sum(dtype=np.float64)),
                               1.0, places=7)
        self.assertEqual((population.population_min, population.population_max),
                         (100_000_000, 160_000_000))

    def test_water_zero_elevation_does_not_turn_flat_coasts_into_steep_land(self):
        grid, climate = fixture()
        grid.water[:, :16] = 1
        grid.elevation[:, :16] = 0.0
        thematic = themes(grid, climate)
        np.testing.assert_array_equal(relative_land_slope(grid.elevation, grid.water == 0), 0.0)
        self.assertEqual(float(thematic.land_potential[16, 16]),
                         float(thematic.land_potential[16, 30]))
        self.assertGreaterEqual(float(thematic.habitability[16, 16]),
                                float(thematic.habitability[16, 30]))

    def test_real_land_slope_is_retained_with_one_sided_coastal_difference(self):
        elevation = np.array([[0.0, 0.0, 0.2, 0.3, 0.4, 0.5]], dtype=np.float64)
        land = np.array([[False, False, True, True, True, True]])
        slope = relative_land_slope(elevation, land)
        np.testing.assert_allclose(slope[land], 0.1, atol=1.0e-15)
        self.assertTrue(np.all(slope[~land] == 0))
        # An isolated island has no measured land neighbours, not an invented
        # zero-elevation sea cliff.
        land[:] = False
        land[0, 2] = True
        np.testing.assert_array_equal(relative_land_slope(elevation, land), 0.0)

    def test_proximity_is_isotropic_wrapped_and_has_a_smooth_compact_tail(self):
        mask = np.zeros((25, 40), dtype=bool)
        mask[12, 20] = True
        proximity = continuous_proximity(mask, 8)
        self.assertAlmostEqual(float(proximity[15, 24]), float(proximity[12, 25]))
        self.assertEqual(float(proximity[12, 28]), 0.0)
        self.assertLess(float(proximity[12, 27]), 0.06)
        mask[:] = False
        mask[12, 0] = True
        proximity = continuous_proximity(mask, 8)
        self.assertEqual(float(proximity[12, 1]), float(proximity[12, -1]))
        self.assertEqual(float(proximity[12, 0]), 1.0)
        np.testing.assert_array_equal(continuous_proximity(np.zeros_like(mask), 8), 0.0)

    def test_source_capacity_derivation_never_mutates_physical_or_climate_arrays(self):
        grid, climate = fixture()
        grid.river_order[:, 32] = 3
        arrays = (grid.water, grid.snow, grid.elevation, grid.river_order,
                  climate.temperature, climate.annual_precipitation,
                  climate.precipitation_range)
        copies = tuple(field.copy() for field in arrays)
        thematic = themes(grid, climate)
        first = derive_population(grid, thematic,
            population_min=100_000_000, population_max=160_000_000)
        second = derive_population(grid, thematic,
            population_min=100_000_000, population_max=160_000_000)
        for original, copy in zip(arrays, copies, strict=True):
            np.testing.assert_array_equal(original, copy)
        np.testing.assert_array_equal(first.population_weight, second.population_weight)
        np.testing.assert_array_equal(first.population_band, second.population_band)
        self.assertFalse(first.population_weight.flags.writeable)
        self.assertFalse(first.population_band.flags.writeable)


class PopulationDensityTests(unittest.TestCase):
    def test_equal_density_has_same_colour_despite_different_latitude_cell_area(self):
        grid, _climate = fixture()
        grid.metadata["extents"].update(north=80.0, south=-80.0)
        latitude_edges = np.linspace(80.0, -80.0, grid.shape[0] + 1)
        area_rows = 6400.0 ** 2 * np.radians(360.0 / grid.shape[1]) * np.abs(
            np.diff(np.sin(np.radians(latitude_edges))))
        weights = np.broadcast_to(area_rows[:, None], grid.shape).copy()
        weights = (weights / weights.sum()).astype(np.float32)
        population = PopulationLayers(weights, np.zeros(grid.shape, dtype=np.uint8),
                                      100_000_000, 160_000_000)
        density = population_density(grid, population)
        expected = 130_000_000 / (area_rows.sum() * grid.shape[1])
        np.testing.assert_allclose(density, expected, rtol=1.0e-6)
        self.assertGreater(float(weights[16, 0]), float(weights[0, 0]) * 4.0)
        bands = _population_bands(density, np.ones(grid.shape, dtype=bool))
        self.assertEqual(np.unique(bands).size, 1)
        self.assertFalse(density.flags.writeable)

    def test_density_classes_are_absolute_and_reserve_zero_for_zero_density(self):
        density = np.array([[0.0, 0.05, 0.15, 0.75, 1.5, 3.0, 6.0]])
        bands = _population_bands(density, np.ones(density.shape, dtype=bool))
        np.testing.assert_array_equal(bands, [[0, 1, 2, 3, 4, 5, 6]])
        extended = np.concatenate((density, [[1000.0]]), axis=1)
        np.testing.assert_array_equal(
            _population_bands(extended, np.ones(extended.shape, dtype=bool))[:, :7], bands)
        self.assertEqual(POPULATION_DENSITY_THRESHOLDS, (0.0, 0.1, 0.5, 1.0, 2.0, 5.0))


if __name__ == "__main__":
    unittest.main()
