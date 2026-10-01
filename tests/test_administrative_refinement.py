import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from world_atlas.core.society.administrative_refinement import (
    _accept_topological_changes, _component_counts, _supported_boundary_cells,
    refine_saved_administrations,
)
from world_atlas.core.society.spatial import physical_transition_penalties, refine_partition_boundaries


class AdministrativeTerrainEvidenceTests(unittest.TestCase):
    def test_flat_outlet_partition_is_not_a_ridge_even_on_a_high_plateau(self):
        # D8 may route either half to different outlets, but neither this
        # valley floor nor a flat high plateau has a cross-edge mountain.
        for height in (.12, .82):
            terrain = np.full((16, 32), height)
            penalties = physical_transition_penalties(terrain, np.zeros(terrain.shape, dtype=np.int8), land_mask=np.ones(terrain.shape, dtype=bool))
            np.testing.assert_array_equal(penalties, 0)

    def test_flat_narrow_coast_does_not_acquire_a_ridge_from_water_zero_height(self):
        terrain = np.zeros((16, 32))
        terrain[:, 14:16] = .24
        dry = terrain > 0
        penalties = physical_transition_penalties(terrain, np.zeros(terrain.shape, dtype=np.int8), land_mask=dry)
        np.testing.assert_array_equal(penalties, 0)

    def test_measured_crest_is_directional_and_rotates_with_terrain(self):
        terrain = np.full((16, 16), .2)
        terrain[7:9] = .75
        rivers = np.zeros(terrain.shape, dtype=np.int8)
        penalties = physical_transition_penalties(terrain, rivers, land_mask=np.ones(terrain.shape, dtype=bool))
        self.assertGreater(penalties[6, 7, 8], 20)
        self.assertEqual(penalties[4, 7, 8], 0)
        rotated = physical_transition_penalties(terrain.T, rivers.T, land_mask=np.ones(terrain.T.shape, dtype=bool))
        np.testing.assert_array_equal(penalties[6, 3:13, 3:13].T, rotated[4, 3:13, 3:13])

    def test_uniform_slope_has_no_invented_crest(self):
        terrain = np.broadcast_to(np.linspace(.1, .4, 16)[:, None], (16, 32))
        penalties = physical_transition_penalties(terrain, np.zeros(terrain.shape, dtype=np.int8), land_mask=np.ones(terrain.shape, dtype=bool))
        expected = 34 * abs(terrain[6, 8] - terrain[7, 8]) ** 1.2
        self.assertAlmostEqual(float(penalties[6, 6, 8]), expected, places=6)

    def test_real_major_river_still_separates_banks(self):
        terrain = np.full((16, 32), .12)
        rivers = np.zeros(terrain.shape, dtype=np.int8)
        rivers[:, 14] = 4
        penalties = physical_transition_penalties(terrain, rivers, land_mask=np.ones(terrain.shape, dtype=bool))
        self.assertGreater(penalties[4, 8, 13], 10)
        self.assertEqual(penalties[6, 8, 14], 0)

    def test_existing_mapped_river_border_is_fixed_but_flat_outlet_seam_is_not(self):
        terrain = np.full((16, 32), .12)
        labels = np.ones(terrain.shape, dtype=np.int16); labels[:, 14:] = 2
        rivers = np.zeros(terrain.shape, dtype=np.int8)
        self.assertFalse(_supported_boundary_cells(labels, physical_transition_penalties(terrain, rivers, land_mask=np.ones(terrain.shape, dtype=bool)), rivers).any())
        rivers[:, 14] = 2
        protected = _supported_boundary_cells(labels, physical_transition_penalties(terrain, rivers, land_mask=np.ones(terrain.shape, dtype=bool)), rivers)
        self.assertTrue(protected[:, 13:15].all())
        self.assertFalse(protected[:, :13].any())

    def test_border_refinement_is_longitude_translation_equivariant_without_wave_noise(self):
        rows, columns = np.indices((24, 48))
        labels = np.where((columns > 10) & (columns < 33), 1, 2).astype(np.int32)
        elevation = .1 + .15 * np.exp(-((rows - 10) / 4)**2)
        elevation += .03 * np.exp(-((columns - 18) / 5)**2)
        penalties = physical_transition_penalties(elevation, np.zeros(labels.shape, dtype=np.int8), land_mask=np.ones(labels.shape, dtype=bool))
        anchors = {(5, 20): 1, (16, 40): 2}
        result = refine_partition_boundaries(labels, labels > 0, penalties, anchors, band_radius=4)
        offset = 7
        shifted = refine_partition_boundaries(
            np.roll(labels, offset, axis=1), labels > 0, np.roll(penalties, offset, axis=2),
            {(r, (c + offset) % labels.shape[1]): owner for (r, c), owner in anchors.items()},
            band_radius=4,
        )
        np.testing.assert_array_equal(result, np.roll(shifted, -offset, axis=1))


class AdministrativeTopologyTests(unittest.TestCase):
    def saved_travel_fixture(self, *, separate_civilizations):
        from test_world_identity import _society
        from world_atlas.core.society.model import TransportRoute

        society = _society()
        shape = (16, 40)
        labels = np.full(shape, 2, dtype=np.int16)
        labels[:, :12] = 1
        labels[8, 12:24] = 1
        civilization = labels if separate_civilizations else np.ones(shape, dtype=np.int16)
        positions = ((8, 4), (6, 10), (8, 30), (11, 30))
        cities = tuple(replace(city, row=row, column=column)
                       for city, (row, column) in zip(society.settlements, positions, strict=True))
        road = TransportRoute(
            'domestic-detour', 'road', 'regional', cities[0].identifier, cities[1].identifier,
            ((4.5, 8.5), (23.5, 8.5), (23.5, 6.5), (10.5, 6.5)),
        )
        states = tuple(replace(state, civilization_identifier=state.identifier if separate_civilizations else 1,
                               language_identifier=state.identifier if separate_civilizations else 1)
                       for state in society.politics.states)
        society = replace(
            society,
            settlements=cities,
            politics=replace(society.politics, state_id=labels, frontier=np.zeros(shape, dtype=bool), states=states),
            provinces=replace(society.provinces, province_id=labels.astype(np.int32)),
            population=replace(society.population, population_weight=np.ones(shape, dtype=np.float32),
                               population_band=np.ones(shape, dtype=np.uint8)),
            cultures=replace(society.cultures, civilization_id=civilization,
                             civilization_influence=np.ones(shape, dtype=np.uint8),
                             language_family_id=civilization, language_id=civilization,
                             language_contact=np.zeros(shape, dtype=bool)),
            religions=replace(society.religions, religion_id=np.ones(shape, dtype=np.int16)),
            transport=replace(society.transport, accessibility=np.zeros(shape, dtype=np.float32),
                              routes=(road,)),
        )
        grid = SimpleNamespace(
            shape=shape, water=np.zeros(shape, dtype=np.uint8), elevation=np.full(shape, .2),
            river_order=np.zeros(shape, dtype=np.int8),
            metadata=dict(extents=dict(west=-180, east=180, north=80, south=-80),
                          planet=dict(radiusKm=6400)),
        )
        thematic = SimpleNamespace(land_potential=np.full(shape, .8), habitability=np.full(shape, .7))
        return society, grid, thematic

    def apply(self, provinces, states, candidate, eligible, parent):
        return _accept_topological_changes(
            provinces, states, candidate, np.asarray(parent, dtype=np.int16), eligible,
            np.zeros(provinces.shape),
        )

    def test_country_and_province_move_together_without_a_diagonal_possession(self):
        provinces = np.ones((12, 20), dtype=np.int32)
        provinces[:, 10:] = 2
        states = provinces.astype(np.int16)
        wanted = provinces.copy()
        wanted[5:7, 9] = 2
        eligible = wanted != provinces
        after, countries, count = self.apply(provinces, states, wanted, eligible, [0, 1, 2])
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(countries, after)
        self.assertEqual(_component_counts(provinces, 2), _component_counts(after, 2))
        np.testing.assert_array_equal(provinces[:, :9], 1)

    def test_narrow_land_governance_bridge_and_last_island_cell_cannot_be_cut(self):
        provinces = np.full((9, 15), 2, dtype=np.int32)
        provinces[2:7, 2:6] = 1
        provinces[2:7, 9:13] = 1
        provinces[4, 6:9] = 1
        wanted = provinces.copy()
        wanted[4, 7] = 2
        after, _, count = self.apply(provinces, provinces.astype(np.int16), wanted, wanted != provinces, [0, 1, 2])
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(after, provinces)
        provinces[1, 0] = 3
        wanted = provinces.copy(); wanted[1, 0] = 2
        after, _, count = self.apply(provinces, provinces.astype(np.int16), wanted, wanted != provinces, [0, 1, 2, 3])
        self.assertEqual(count, 0)
        self.assertEqual(after[1, 0], 3)

    def test_saved_society_keeps_real_city_ownership_identifiers_and_input_arrays(self):
        from test_world_identity import _society
        society = _society()
        shape = society.politics.state_id.shape
        grid = SimpleNamespace(
            shape=shape, water=np.zeros(shape, dtype=np.uint8), elevation=np.full(shape, .2),
            river_order=np.zeros(shape, dtype=np.int8),
            metadata=dict(extents=dict(west=-180,east=180,north=80,south=-80), planet=dict(radiusKm=6400)),
        )
        thematic = SimpleNamespace(land_potential=np.full(shape, .8), habitability=np.full(shape, .7))
        original_states = society.politics.state_id.copy()
        original_provinces = society.provinces.province_id.copy()
        updated, report = refine_saved_administrations(grid, society, thematic=thematic)
        self.assertEqual(updated.settlements, society.settlements)
        self.assertEqual(updated.transport, society.transport)
        self.assertEqual([(r.identifier,r.name,r.core_settlement_id) for r in updated.politics.states],
                         [(r.identifier,r.name,r.core_settlement_id) for r in society.politics.states])
        self.assertEqual(report['settlementOwnershipChanges'], 0)
        self.assertEqual(report['stateComponentsBefore'], report['stateComponentsAfter'])
        self.assertEqual(report['provinceComponentsBefore'], report['provinceComponentsAfter'])
        np.testing.assert_array_equal(society.politics.state_id, original_states)
        np.testing.assert_array_equal(society.provinces.province_id, original_provinces)

    def test_saved_seam_keeps_civilization_ownership_despite_closer_foreign_cities(self):
        society, grid, thematic = self.saved_travel_fixture(separate_civilizations=True)
        updated, report = refine_saved_administrations(grid, society, thematic=thematic)
        np.testing.assert_array_equal(updated.politics.state_id, society.politics.state_id)
        self.assertEqual(report['acceptedNationalMoves'], 0)

    def test_domestic_road_does_not_reserve_an_unsupported_territorial_appendage(self):
        society, grid, thematic = self.saved_travel_fixture(separate_civilizations=False)
        updated, report = refine_saved_administrations(grid, society, thematic=thematic)
        self.assertGreater(report['acceptedNationalMoves'], 0)
        self.assertEqual(updated.politics.state_id[8, 23], 2)
        self.assertEqual(updated.politics.state_id[8, 4], 1)
        self.assertEqual(updated.politics.state_id[6, 10], 1)
        self.assertEqual(updated.transport, society.transport)
        self.assertEqual(report['stateComponentsBefore'], report['stateComponentsAfter'])


if __name__ == '__main__':
    unittest.main()
