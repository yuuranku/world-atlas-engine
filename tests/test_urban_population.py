from dataclasses import replace
import unittest

from world_atlas.core.society.urban_population import allocate_urban_population
from test_presentation import _fixture


class UrbanPopulationTests(unittest.TestCase):
    def test_hierarchy_and_budget_and_replay(self):
        grid,society=_fixture()
        society=replace(society,settlements=tuple(replace(s,tier='city',score=1.,population_min=10000,population_max=10000)
                                                for s in society.settlements))
        result=allocate_urban_population(grid,society)
        cities={s.identifier:s for s in result.settlements}
        self.assertGreater(cities['port-city'].population_min,cities['river-town'].population_min*2)
        self.assertLessEqual(sum(s.population_min for s in result.settlements),35000)
        self.assertLessEqual(sum(s.population_max for s in result.settlements),140000)
        self.assertEqual(result.settlements,allocate_urban_population(grid,result).settlements)
        self.assertIs(result.population,society.population)
