"""Continents are physical landmasses, not an arbitrary owner partition."""

import unittest
from dataclasses import replace

import numpy as np

from world_atlas.core.planet_morphology import sample_world_morphology
from world_atlas.core.procedural_planet import (
    PlanetRecipe,
    _boundary_class_field,
    _continental_crust,
    _continental_topology,
    _continent_owners,
    _curved_boundary_corridors,
    _derive_seed,
    _grow_continental_crust,
    _plate_definitions,
    _select_crust_roots,
    generate_planet_surface,
)
from world_atlas.core.raster_topology import periodic_component_labels
from world_atlas.physical.planetary_grid import build_lat_lon_grid
from world_atlas.physical.tectonics import build_plate_fields


class ContinentConstructionTests(unittest.TestCase):
    def test_frozen_recipe_solves_requested_physical_continents(self):
        # The failed five-continent recipe also exercises the supported count
        # range without random seed retries or full world rendering.
        recipe = PlanetRecipe(273763461, 120, 60, 14, 5, 0.35, 0.34, 0.82, 0.18, False, False)
        grid = build_lat_lon_grid(recipe.height, recipe.width)
        morphology = sample_world_morphology(_derive_seed(recipe.seed, "morphology"))
        plate_seed = _derive_seed(recipe.seed, "plates")
        anchors = _plate_definitions(recipe.plate_count, morphology, plate_seed)
        initial = build_plate_fields(grid, plate_count=recipe.plate_count, anchors=anchors,
                                     stage_seed=plate_seed, radius_km=6400.0, boundary_corridors=())
        plates = build_plate_fields(grid, plate_count=recipe.plate_count, anchors=anchors,
                                    stage_seed=plate_seed, radius_km=6400.0,
                                    boundary_corridors=_curved_boundary_corridors(initial, morphology, plate_seed))
        boundaries = _boundary_class_field(plates)
        weights = np.cos(np.radians(grid.latitude_degrees[:, 0]))[:, None]
        for count in (3, 5, 8, 12):
            with self.subTest(count=count):
                selected = replace(recipe, continent_count=count)
                land, field, level, metrics = _continental_crust(
                    grid, plates, boundaries, morphology, selected, _derive_seed(recipe.seed, "crust"),
                )
                np.testing.assert_array_equal(field > level, land)
                valid, topology = _continental_topology(land, weights, selected)
                self.assertTrue(valid, topology)
                self.assertLess(abs(topology["landFraction"] - recipe.land_fraction), 2 / grid.cell_count)
                self.assertEqual(np.unique(_continent_owners(land, count)[land]).size, count)
                self.assertGreaterEqual(metrics["continentConstruction"]["retainedAmplitude"], 0.0)
                self.assertLessEqual(metrics["continentConstruction"]["retainedAmplitude"], 1.0)

    def test_frozen_recipe_survives_native_relief_and_shoreline_evolution(self):
        recipe = PlanetRecipe(273763461, 240, 120, 14, 5, 0.35, 0.34, 0.82, 0.18, False, False)
        surface = generate_planet_surface(recipe)
        np.testing.assert_array_equal(surface.relative_elevation_m > 0.0, surface.land_mask)
        self.assertEqual(np.unique(surface.continent_id[surface.land_mask]).size, 5)
        construction = surface.diagnostics["nativeContinentConstruction"]
        self.assertEqual(construction["result"]["connectivity"]["4"]["majorCount"], 5)
        self.assertEqual(construction["result"]["connectivity"]["8"]["majorCount"], 5)
        self.assertTrue(np.isfinite(surface.relative_elevation_m).all())

    def test_polar_roots_are_included_in_total_continent_count(self):
        grid = build_lat_lon_grid(60, 120)
        morphology = sample_world_morphology(2026)
        roots = _select_crust_roots(grid, 1, morphology, 2026, 0.35) + (0, 59 * 120)
        land = _grow_continental_crust(
            grid, roots, np.zeros(grid.shape, dtype=int),
            np.zeros(grid.shape, dtype=np.int8), morphology, 0.35, 2026,
        )
        recipe = PlanetRecipe(2026, 120, 60, 14, 3, 0.35, 0.34, 0.82, 0.18, True, True)
        valid, metrics = _continental_topology(
            land, np.cos(np.radians(grid.latitude_degrees[:, 0]))[:, None], recipe,
        )
        self.assertTrue(valid, metrics)
        self.assertTrue(land[0].all() and land[-1].all())

    def test_seam_certification_accounts_for_noninteger_native_sampling(self):
        land = np.zeros((360, 720), dtype=bool)
        for start in (50, 150, 250):
            land[start:start + 50, 18:] = True
        recipe = PlanetRecipe(2026, 2176, 1088, 14, 3, 0.35, 0.34, 0.82, 0.18, False, False)
        weights = np.cos(np.radians(90.0 - (np.arange(360) + 0.5) / 2.0))[:, None]
        valid, metrics = _continental_topology(land, weights, recipe)
        # Eighteen construction columns satisfy 2.5% at 720, but become only
        # 54 native columns; the delivery grid requires 55.
        self.assertFalse(valid)
        self.assertFalse(metrics["mapSeamValid"])
        land[:, 18] = False
        valid, metrics = _continental_topology(land, weights, recipe)
        self.assertTrue(valid, metrics)
        self.assertGreaterEqual(metrics["projectedOceanCorridorWidthColumns"], 55)

    def test_accretion_fronts_keep_three_distinct_roots_and_area(self):
        grid = build_lat_lon_grid(60, 120)
        morphology = sample_world_morphology(2026)
        roots = _select_crust_roots(grid, 3, morphology, 2026, 0.35)
        crust = _grow_continental_crust(
            grid, roots, np.zeros(grid.shape, dtype=int),
            np.zeros(grid.shape, dtype=np.int8), morphology, 0.425, 2026,
        )
        labels, count = periodic_component_labels(crust, 8)
        self.assertEqual(count, 3)
        self.assertEqual(len(set(labels.ravel()[list(roots)])), 3)
        weight = np.cos(np.radians(grid.latitude_degrees))
        self.assertAlmostEqual(float(np.sum(crust * weight) / np.sum(weight)), 0.425, places=3)

    def test_each_mainland_and_island_has_one_owner(self):
        land = np.zeros((24, 48), dtype=bool)
        land[5:19, 3:12] = True
        land[5:19, 19:28] = True
        land[5:19, 35:44] = True
        land[3, 4:6] = True
        owners = _continent_owners(land, 3)
        for start in (3, 19, 35):
            self.assertEqual(np.unique(owners[5:19, start:start+9]).size, 1)
        self.assertEqual(owners[3, 4], owners[6, 4])
        self.assertTrue(np.all(owners[~land] == 0))

    def test_labels_cannot_disguise_a_supercontinent(self):
        land = np.zeros((24, 48), dtype=bool)
        land[5:19, 3:44] = True
        with self.assertRaisesRegex(ValueError, "1 major continents"):
            _continent_owners(land, 3)

    def test_diagonal_land_bridge_cannot_pass_as_two_continents(self):
        land = np.zeros((12, 24), dtype=bool)
        land[2:6, 2:6] = True
        land[6:10, 6:10] = True
        with self.assertRaisesRegex(ValueError, "diagonal"):
            _continent_owners(land, 2)
