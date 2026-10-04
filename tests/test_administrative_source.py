from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from tests.test_world_identity import _society
from world_atlas.core.society.administrations import (
    AdministrativeHierarchy, _apply_administrative_ownership, _travel_simulation,
)
from world_atlas.core.society.administrative_front import AdministrativeFront


class AdministrativeSourceTests(unittest.TestCase):
    def fixture(self):
        society = _society()
        grid = SimpleNamespace(shape=society.politics.state_id.shape,
            water=np.zeros(society.politics.state_id.shape,dtype=np.uint8),
            elevation=np.full(society.politics.state_id.shape,.2),
            river_order=np.zeros(society.politics.state_id.shape,dtype=np.uint8),
            metadata={'extents':dict(west=-180,east=180,north=80,south=-80),
                      'planet':dict(radiusKm=6400)})
        thematic = SimpleNamespace(land_potential=np.full(grid.shape,.8),
                                   habitability=np.full(grid.shape,.7))
        return grid,thematic,society

    def test_provincial_reach_respects_mapped_minor_river_without_national_language_seam(self):
        grid,thematic,society = self.fixture()
        grid.river_order[:,2] = 1
        # Within a single language, a minor river constrains local reach even
        # when it is not a national frontier. Along-bank travel stays cheap.
        countries = _travel_simulation(grid,thematic,society)
        provinces = _travel_simulation(grid,thematic,society,domains=society.politics.state_id)
        self.assertEqual(float(countries.transition_penalty[4,2,1]),0.)
        self.assertGreater(float(provinces.transition_penalty[4,2,1]),5.)
        self.assertEqual(float(provinces.transition_penalty[6,2,2]),0.)

    def test_native_refresh_updates_provincial_counts_with_same_seats_and_population_source(self):
        grid,_,society = self.fixture()
        old = society.provinces.province_id.copy()
        old[3:,:4] = 3
        records = (replace(society.provinces.provinces[0],area_cells=12),
                   society.provinces.provinces[1],
                   replace(society.provinces.provinces[0],identifier=3,
                           core_settlement_id='east-market',area_cells=12))
        society = replace(society,provinces=replace(society.provinces,
                          province_id=old,provinces=records))
        new = old.copy();new[3,:4] = 1
        front = AdministrativeFront(np.stack((new,np.zeros_like(new))),
            np.zeros((2,*grid.shape)),np.ones((8,*grid.shape)),
            np.ones(grid.shape,bool),100.,None,{1:1,2:2,3:1})
        hierarchy = AdministrativeHierarchy(front,front,np.array((0,1,2,1)))
        updated = _apply_administrative_ownership(grid,society,hierarchy)
        np.testing.assert_array_equal(updated.politics.state_id,society.politics.state_id)
        np.testing.assert_array_equal(updated.provinces.province_id,new)
        self.assertEqual([item.area_cells for item in updated.provinces.provinces],[16,24,8])
        self.assertEqual([item.core_settlement_id for item in updated.provinces.provinces],
                         [item.core_settlement_id for item in records])
        self.assertIs(updated.population,society.population)
        self.assertEqual(updated.politics.states[0].population_min,2_400_000)


if __name__ == '__main__':
    unittest.main()
