"""Saved identity is retained while authority follows the physical institutions."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from tests.test_world_identity import _society
from world_atlas.core.society.administrative_topology import reconcile_partition_components
from world_atlas.core.society.politics import _resolve_institutional_seats
from world_atlas.core.society.saved_formation import refresh_saved_institutional_administrations
from world_atlas.core.society.state_formation import (
    ControlRegionEdge, ControlRegionGraph, simulate_state_formation,
)


def fixture():
    society = _society()
    society = replace(society, politics=replace(society.politics,
        government_forms=(replace(society.politics.government_forms[0],
                                  key="bureaucratic-monarchy"),)))
    shape = society.politics.state_id.shape
    grid = SimpleNamespace(shape=shape, water=np.zeros(shape, dtype=np.uint8),
        elevation=np.full(shape, .2, dtype=np.float32),
        river_order=np.zeros(shape, dtype=np.uint8), snow=np.zeros(shape, dtype=np.uint8),
        metadata={"worldProfile": {"technologyEra": "ancient"},
                  "planet": {"radiusKm": 20.},
                  "extents": {"west": -180., "east": 180., "north": 60., "south": -60.}})
    thematic = SimpleNamespace(land_potential=np.full(shape, .8),
                               habitability=np.full(shape, .7))
    return grid, thematic, society


class SavedInstitutionalFormationTests(unittest.TestCase):
    def test_real_graph_refresh_retains_identity_and_coherent_demographics(self):
        grid, thematic, society = fixture()
        updated, report = refresh_saved_institutional_administrations(grid, thematic, society)
        self.assertEqual(updated.settlements, society.settlements)
        self.assertIs(updated.population, society.population)
        self.assertIs(updated.transport, society.transport)
        self.assertIs(updated.cultures, society.cultures)
        self.assertIs(updated.religions, society.religions)
        self.assertEqual(updated.politics.political_entities, society.politics.political_entities)
        self.assertEqual([(r.identifier, r.name, r.core_settlement_id, r.size_class)
                          for r in updated.politics.states],
                         [(r.identifier, r.name, r.core_settlement_id, r.size_class)
                          for r in society.politics.states])
        self.assertEqual([(r.identifier, r.name, r.core_settlement_id)
                          for r in updated.provinces.provinces],
                         [(r.identifier, r.name, r.core_settlement_id)
                          for r in society.provinces.provinces])
        np.testing.assert_array_equal(updated.politics.state_id, society.cultures.civilization_id)
        self.assertEqual(sum(r.area_cells for r in updated.provinces.provinces), grid.water.size)
        for state in updated.politics.states:
            share = float(updated.population.population_weight[updated.politics.state_id == state.identifier].sum())
            self.assertEqual(state.population_min, round(share * updated.population.population_min / 10_000) * 10_000)
        self.assertEqual(report["provinceParentChanges"], [])
        self.assertEqual(set(report["stageSeconds"]),
                         {"physicalControlGraph", "stateFormation", "provinceFormation"})

    def test_old_cross_cultural_parent_is_corrected_without_erasing_or_renaming_the_seat(self):
        grid, thematic, society = fixture()
        states = society.politics.state_id.copy()
        provinces = society.provinces.province_id.copy()
        states[4, 2] = 2
        provinces[4, 2] = 3
        third = replace(society.provinces.provinces[0], identifier=3,
                        name="Retained historical district", state_identifier=2,
                        core_settlement_id="east-market", administrative_function="civil",
                        administrative_rank="civil-district", area_cells=1)
        society = replace(society, politics=replace(society.politics, state_id=states),
                          provinces=replace(society.provinces, province_id=provinces,
                                            provinces=(*society.provinces.provinces, third)))
        updated, report = refresh_saved_institutional_administrations(grid, thematic, society)
        restored = updated.provinces.provinces[2]
        self.assertEqual((restored.identifier, restored.name, restored.core_settlement_id),
                         (third.identifier, third.name, third.core_settlement_id))
        self.assertEqual(restored.state_identifier, 1)
        self.assertEqual(int(updated.provinces.province_id[4, 2]), 3)
        self.assertEqual(report["provinceParentChanges"],
                         [{"provinceId": 3, "previousCountryId": 2, "countryId": 1}])

    def test_existing_local_institution_is_a_protected_temporal_control_root(self):
        graph = ControlRegionGraph(np.array([0, 1, 1, 1, 1, 1, 1]),
            np.ones(7), np.ones(7, dtype=np.int32),
            tuple(ControlRegionEdge(a, a + 1, 2, 1., 0., 0., 0., 0.) for a in range(1, 6)))
        parameters = dict(core_region_by_state=np.array([0, 1, 6]),
                          state_strength=np.array([0., 1., 8.]),
                          travel_days_by_edge=(1.,) * 5,
                          maximum_response_days_by_state=np.array([0., 1., 2.]))
        ordinary = simulate_state_formation(graph, **parameters)
        restored = simulate_state_formation(graph, **parameters, institutional_regions={3: 1})
        self.assertEqual(restored.region_owner[3], 1)
        self.assertNotEqual(ordinary.region_owner[3], 1)
        self.assertEqual(restored.region_owner[1], 1)
        self.assertEqual(restored.region_owner[6], 2)

    def test_local_institutions_cannot_assert_authority_across_a_cultural_domain(self):
        graph = ControlRegionGraph(np.array([0, 1, 2]), np.ones(3),
                                  np.ones(3, dtype=np.int32), ())
        with self.assertRaisesRegex(ValueError, "cultural owner domain"):
            simulate_state_formation(graph, core_region_by_state=np.array([0, 1]),
                state_strength=np.array([0., 1.]), travel_days_by_edge=(),
                maximum_response_days_by_state=np.array([0., 1.]), institutional_regions={2: 1})

    def test_actual_local_seat_survives_component_reconciliation(self):
        labels = np.array([[1, 0, 1, 2]], dtype=np.int16)
        parameters = dict(seats={1: (0, 0), 2: (0, 3)},
                          owner_domain=np.array([0, 1, 1]),
                          land_component_id=np.ones_like(labels))
        ordinary = reconcile_partition_components(labels, **parameters)
        existing_institution = reconcile_partition_components(
            labels, **parameters, institutional_seats={(0, 2): 1},
        )
        self.assertEqual(ordinary[0, 2], 2)
        self.assertEqual(existing_institution[0, 2], 1)

    def test_reparenting_requires_an_actual_connection_to_the_new_state(self):
        grid, _, _ = fixture()
        graph = ControlRegionGraph(np.array([0, 1, 2, 2]), np.ones(4),
                                  np.ones(4, dtype=np.int32), ())
        with self.assertRaisesRegex(ValueError, "no evidenced state"):
            _resolve_institutional_seats(grid, graph, np.array([[1, 2, 3]]),
                np.array([0, 1, 2]), {(0, 2): 1}, (), (), ())


if __name__ == "__main__":
    unittest.main()
