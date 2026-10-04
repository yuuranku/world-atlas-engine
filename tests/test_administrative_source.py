from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from tests.test_world_identity import _society
from world_atlas.core.society.administrations import (
    administrative_source, synchronize_administrative_statistics,
)


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

    def test_source_uses_formed_observations_without_a_second_territorial_race(self):
        grid,_,society = self.fixture()
        first = administrative_source(grid,society)
        grid.river_order[:,2] = 1
        grid.elevation[:,2] = .9
        second = administrative_source(grid,society)
        np.testing.assert_array_equal(first.province_id,society.provinces.province_id)
        np.testing.assert_array_equal(first.province_id,second.province_id)
        self.assertFalse(first.province_id.flags.writeable)
        self.assertFalse(first.province_to_state.flags.writeable)

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
        updated = synchronize_administrative_statistics(grid,society,
            state_id=society.politics.state_id,province_id=new)
        np.testing.assert_array_equal(updated.politics.state_id,society.politics.state_id)
        np.testing.assert_array_equal(updated.provinces.province_id,new)
        self.assertEqual([item.area_cells for item in updated.provinces.provinces],[16,24,8])
        self.assertEqual([item.core_settlement_id for item in updated.provinces.provinces],
                         [item.core_settlement_id for item in records])
        self.assertIs(updated.population,society.population)
        self.assertEqual(updated.politics.states[0].population_min,2_400_000)


if __name__ == '__main__':
    unittest.main()
