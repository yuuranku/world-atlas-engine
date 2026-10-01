"""Continents are physical landmasses, not an arbitrary owner partition."""

import unittest

import numpy as np

from world_atlas.core.planet_morphology import sample_world_morphology
from world_atlas.core.procedural_planet import (
    _continent_owners,
    _grow_continental_crust,
    _select_crust_roots,
)
from world_atlas.core.raster_topology import periodic_component_labels
from world_atlas.physical.planetary_grid import build_lat_lon_grid


class ContinentConstructionTests(unittest.TestCase):
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
