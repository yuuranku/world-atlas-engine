"""Demographic changes must reach culture, territory and administration."""

from types import SimpleNamespace
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from world_atlas.core.society import culture, politics, provinces, religion
from world_atlas.core.society.administrative_centres import derive_administrative_centres
from world_atlas.core.society.frontiers import (
    derive_frontier_groups,
    derive_stateless_frontier,
)
from world_atlas.core.society.model import (
    NameLexicon,
    PopulationLayers,
    Settlement,
    TransportLayers,
    TransportRoute,
)
from world_atlas.core.society.population import cell_areas_km2, population_density
from world_atlas.core.society.territorial_simulation import extreme_frontier_environment
from world_atlas.core.thematic import BIOME_TEMPERATE_GRASSLAND, BIOME_TEMPERATE_MIXED_FOREST


def fixture(*, rain=0.05, north=40.0, south=-40.0):
    shape = (32, 64)
    grid = SimpleNamespace(
        shape=shape,
        water=np.zeros(shape, dtype=np.uint8),
        snow=np.zeros(shape, dtype=bool),
        elevation=np.full(shape, 0.2, dtype=np.float32),
        river_order=np.zeros(shape, dtype=np.uint8),
        flow_to=np.full(shape, -1, dtype=np.int32),
        metadata={
            "planet": {"radiusKm": 6400.0},
            "extents": {"north": north, "south": south, "west": -180.0, "east": 180.0},
            "worldProfile": {"technologyEra": "preindustrial"},
        },
    )
    thematic = SimpleNamespace(
        land_potential=np.full(shape, 0.6, dtype=np.float32),
        habitability=np.full(shape, 0.8, dtype=np.float32),
        biome_zone=np.full(shape, BIOME_TEMPERATE_GRASSLAND, dtype=np.int8),
        drainage_basin=np.zeros(shape, dtype=np.int32),
        climate=SimpleNamespace(
            annual_precipitation=np.full(shape, rain),
            temperature=np.full(shape, 0.6),
        ),
    )
    areas = cell_areas_km2(grid)
    # Constant density deliberately isolates the latitude-area effect from
    # climate and food; weights still represent actual local population mass.
    weights = (areas / areas.sum()).astype(np.float32)
    population = PopulationLayers(weights, np.full(shape, 3, dtype=np.uint8),
                                  100_000_000, 160_000_000)
    towns = tuple(
        Settlement(str(index), f"Town{index}", row, column, "city", "market",
                   1.0, 10_000, 20_000)
        for index, (row, column) in enumerate(((4, 8), (4, 24), (20, 40), (20, 56)))
    )
    transport = TransportLayers(np.full(shape, 0.25, dtype=np.float32), ())
    lexicon = NameLexicon(tuple(f"Country{index}" for index in range(19)),
        tuple("central" for _ in range(19)),
        tuple(f"Realm{index}" for index in range(120)), ("Place",))
    return grid, thematic, population, towns, transport, lexicon


def cultures_for(grid, thematic, population, towns, transport):
    return culture.derive_cultures(grid, thematic, population, towns, transport=transport,
        civilization_count=2, minimum_civilization_count=2, language_count=4)


class PopulationDownstreamTests(unittest.TestCase):
    def test_remote_road_does_not_confer_a_fixed_width_strip_of_sovereignty(self):
        grid, thematic, population, towns, transport, lexicon = fixture()
        cultures = cultures_for(grid, thematic, population, towns, transport)
        weights = population.population_weight.astype(float)
        weights[10:20] *= .01
        weights /= weights.sum()
        population = replace(population, population_weight=weights.astype(np.float32))
        thematic.land_potential[10:20] = .3
        road = TransportRoute("remote-road", "road", "regional", towns[0].identifier,
            towns[-1].identifier,
            ((8.5, 4.5), (8.5, 16.5), (56.5, 16.5), (56.5, 20.5)))
        transport = replace(transport, routes=(road,))
        countries = politics.derive_politics(grid, thematic, population, towns,
            cultures, transport, lexicon, state_count=2, frontier_target_share=.5)
        # This inhabited route connects cities across sparse land;
        # its centre and shoulder may remain beyond either city's control.
        self.assertEqual(countries.state_id[16, 28], 0)
        self.assertEqual(countries.state_id[15, 28], 0)
        self.assertGreater(countries.state_id[towns[0].row, towns[0].column], 0)
        self.assertEqual(len(countries.states), 2)

    def test_cultural_transmission_recognizes_density_below_one_person_per_km2(self):
        shape = (32, 64)
        land = np.ones(shape, dtype=bool)
        hearths = np.zeros(shape, dtype=bool)
        hearths[16, 4] = True
        zeros = np.zeros(shape)
        support = culture._civilization_transmission_mask(
            land, hearths, np.full(shape, 0.12), zeros, np.full(shape, 0.20),
            np.full(shape, 0.2), zeros, zeros, zeros, zeros.astype(bool),
            np.full(shape, 0.7),
        )
        self.assertTrue(support[16, 50])
        self.assertTrue(np.all(support))

    def test_supplied_valleys_do_not_get_a_rainfall_threshold_wall(self):
        cultural_friction = []
        territorial_friction = []
        original_cultural = culture.simulate_territories
        original_language = culture.allocate_regions_by_proximity
        original_bounded = politics.simulate_bounded_reach
        for rainfall in (0.05 - 1.0e-6, 0.05 + 1.0e-6):
            grid, thematic, population, towns, transport, lexicon = fixture(rain=rainfall)
            local_culture = []
            local_territory = []

            def cultural_run(simulation, seeds):
                local_culture.append(simulation.friction.copy())
                return original_cultural(simulation, seeds)

            def bounded_run(simulation, seeds, **kwargs):
                local_territory.append(simulation.friction.copy())
                return original_bounded(simulation, seeds, **kwargs)

            def language_run(friction, valid, **kwargs):
                local_culture.append(friction.copy())
                return original_language(friction, valid, **kwargs)

            with patch.object(culture, "simulate_territories", side_effect=cultural_run), \
                 patch.object(culture, "allocate_regions_by_proximity", side_effect=language_run):
                cultures = cultures_for(grid, thematic, population, towns, transport)
            with patch.object(politics, "simulate_bounded_reach", side_effect=bounded_run):
                countries = politics.derive_politics(grid, thematic, population, towns,
                    cultures, transport, lexicon, state_count=2, frontier_target_share=0)
            self.assertEqual(len(countries.states), 2)
            self.assertGreaterEqual(len(local_culture), 3)  # civilization and both languages
            self.assertTrue(local_territory)
            cultural_friction.append(local_culture)
            territorial_friction.append(local_territory)
        for runs in (cultural_friction, territorial_friction):
            self.assertEqual(len(runs[0]), len(runs[1]))
            for below, above in zip(runs[0], runs[1], strict=True):
                np.testing.assert_allclose(below, above, atol=1.0e-6)

    def test_equal_density_polar_and_equatorial_land_have_same_frontier_protection(self):
        grid, thematic, population, _towns, transport, _lexicon = fixture(north=80, south=-80)
        density = population_density(grid, population)
        frontier = derive_stateless_frontier(
            grid.water == 0, np.ones(grid.shape, dtype=np.int16), thematic.biome_zone,
            grid.snow, grid.elevation, thematic.climate.annual_precipitation,
            np.full(grid.shape, 0.3), density, cell_areas_km2(grid),
            transport.accessibility, grid.river_order, np.zeros((8, *grid.shape)),
            np.zeros(grid.shape, dtype=bool), target_share=0.95,
        )
        self.assertFalse(frontier.any())
        self.assertGreater(float(population.population_weight[16, 0]),
                           4.0 * float(population.population_weight[0, 0]))

    def test_frontier_anchor_locations_do_not_depend_on_display_colour_classes(self):
        grid, thematic, population, towns, transport, _lexicon = fixture()
        cultures = cultures_for(grid, thematic, population, towns, transport)
        frontier = np.ones(grid.shape, dtype=bool)
        first = derive_frontier_groups(grid, thematic, population, cultures, frontier)
        alternate = PopulationLayers(population.population_weight,
            np.random.default_rng(53).integers(0, 7, grid.shape, dtype=np.uint8),
            population.population_min, population.population_max)
        second = derive_frontier_groups(grid, thematic, alternate, cultures, frontier)
        self.assertTrue(first)
        self.assertEqual(first, second)

    def test_province_density_classes_and_local_metrics_use_physical_area(self):
        grid, thematic, population, towns, transport, lexicon = fixture(north=80, south=-80)
        cultures = cultures_for(grid, thematic, population, towns, transport)
        countries = politics.derive_politics(grid, thematic, population, towns,
            cultures, transport, lexicon, state_count=2, frontier_target_share=0)
        captured = {}
        original_targets = provinces._province_core_targets

        def targets(areas, density_ratios, systems, capacities):
            captured["areas"] = areas.copy()
            captured["density_ratios"] = density_ratios.copy()
            return original_targets(areas, density_ratios, systems, capacities)

        with patch.object(provinces, "_province_core_targets", side_effect=targets):
            result = provinces.derive_provinces(grid, thematic, population, towns,
                                               transport, cultures, countries)
        self.assertTrue(result.provinces)
        self.assertTrue(all(item.population_density_class == "settled" for item in result.provinces))
        areas = cell_areas_km2(grid)
        for identifier, area in captured["areas"].items():
            self.assertAlmostEqual(area, float(areas[countries.state_id == identifier].sum()), places=5)
            self.assertAlmostEqual(captured["density_ratios"][identifier], 1.0, places=6)
        density = population_density(grid, population)
        labels = np.ones(grid.shape, dtype=np.int16)
        northern = provinces._local_state_metrics(towns[0], 1, labels, density, areas,
                                                 transport.accessibility, radius=2)
        equatorial = provinces._local_state_metrics(towns[2], 1, labels, density, areas,
                                                   transport.accessibility, radius=2)
        self.assertAlmostEqual(northern[0], equatorial[0], places=6)

    def test_administrative_centres_and_tiers_are_independent_of_display_band(self):
        grid, thematic, population, towns, transport, _lexicon = fixture()
        thematic.biome_zone[:] = BIOME_TEMPERATE_MIXED_FOREST
        countries = SimpleNamespace(state_id=np.ones(grid.shape, dtype=np.int16),
                                    states=(SimpleNamespace(identifier=1),))
        first = derive_administrative_centres(grid, thematic, population,
                                              towns[:1], transport, countries)
        alternate = replace(population, population_band=
            np.random.default_rng(59).integers(0, 7, grid.shape, dtype=np.uint8))
        second = derive_administrative_centres(grid, thematic, alternate,
                                               towns[:1], transport, countries)
        self.assertTrue(first)
        self.assertEqual(first, second)

    def test_admin_centres_use_flat_coastal_land_when_inland_slopes_are_unbuildable(self):
        grid, thematic, _population, towns, transport, _lexicon = fixture()
        grid.water[:, :4] = 1
        grid.elevation[:] = 0.80
        grid.elevation[:, :4] = 0.0
        grid.elevation[:, 4:6] = 0.30
        grid.elevation[:, -2:] = 0.30
        thematic.biome_zone[:] = BIOME_TEMPERATE_MIXED_FOREST
        areas = np.where(grid.water == 0, cell_areas_km2(grid), 0.0)
        population = PopulationLayers((areas / areas.sum()).astype(np.float32),
            np.where(grid.water == 0, 3, 0).astype(np.uint8), 100_000_000, 160_000_000)
        countries = SimpleNamespace(state_id=np.where(grid.water == 0, 1, -1).astype(np.int16),
                                    states=(SimpleNamespace(identifier=1),))
        home = replace(towns[0], row=16, column=4)
        result = derive_administrative_centres(grid, thematic, population,
                                               (home,), transport, countries)
        self.assertTrue(result)
        self.assertTrue(all(item.column in (4, 63) for item in result))
        self.assertTrue(all(item.site_type == "port" for item in result))

    def test_religious_diffusion_has_no_false_coastal_slope_penalty(self):
        grid, thematic, population, towns, transport, _lexicon = fixture()
        grid.water[:, :4] = 1
        grid.elevation[:] = 0.30
        grid.elevation[:, :4] = 0.0
        cultures = SimpleNamespace(civilization_id=np.where(grid.water == 0, 1, -1))
        captured = []
        original = religion.simulate_territories

        def run(simulation, seeds):
            captured.append(simulation.friction.copy())
            return original(simulation, seeds)

        with patch.object(religion, "simulate_territories", side_effect=run):
            result, _ = religion.derive_religions(grid, thematic, population, towns[:1],
                                                cultures, transport, religion_count=1)
        self.assertEqual(len(result.religions), 1)
        self.assertAlmostEqual(float(captured[0][16, 4]), float(captured[0][16, 16]), places=6)

    def test_extreme_frontier_uses_absolute_density_and_rejects_old_colour_classes(self):
        shape = (1, 4)
        valid = np.ones(shape, dtype=bool)
        snow = np.zeros(shape, dtype=bool)
        snow[0, 3] = True
        heights = np.full(shape, 0.85)
        rain = np.full(shape, 0.05)
        potential = np.full(shape, 0.10)
        density = np.array([[0.099, 0.101, 1.0, 8.0]], dtype=np.float32)
        rivers = np.zeros(shape, dtype=np.uint8)
        result = extreme_frontier_environment(valid, snow, heights, rain, potential,
                                             density, rivers)
        np.testing.assert_array_equal(result, [[True, False, False, True]])
        with self.assertRaisesRegex(ValueError, "floating"):
            extreme_frontier_environment(valid, snow, heights, rain, potential,
                                          density.astype(np.uint8), rivers)


if __name__ == "__main__":
    unittest.main()
