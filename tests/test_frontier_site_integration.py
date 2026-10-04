"""New ports must respect local capacity and enter the real transport graph."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from world_atlas.core.society import pipeline
from world_atlas.core.society.model import PopulationLayers, Settlement, TransportLayers
from world_atlas.core.society.state_formation import ControlRegionGraph, attach_maritime_control_regions
from world_atlas.core.society.strategic_sites import _population_interval, derive_strategic_sites


def fixture():
    shape = (31, 61)
    water = np.ones(shape, dtype=np.uint8)
    water[:, :15] = 0
    water[12:19, 35:43] = 0
    grid = SimpleNamespace(shape=shape, water=water,
        elevation=np.zeros(shape, dtype=np.float32), snow=np.zeros(shape, dtype=bool),
        river_order=np.zeros(shape, dtype=np.uint8),
        flow_to=np.full(shape, -1, dtype=np.int32),
        metadata={"worldProfile": {"technologyEra": "preindustrial"},
                  "planet": {"radiusKm": 6400.0},
                  "extents": {"north": 60.0, "south": -60.0, "west": -180.0, "east": 180.0}})
    thematic = SimpleNamespace(land_potential=np.full(shape, 0.5, dtype=np.float32),
        habitability=np.full(shape, 0.65, dtype=np.float32),
        biome_zone=np.zeros(shape, dtype=np.uint8),
        climate=SimpleNamespace(annual_precipitation=np.full(shape, 0.2)))
    weights = np.where(water == 0, 1.0e-5, 0).astype(np.float32)
    population = PopulationLayers(weights, np.where(water == 0, 3, 0).astype(np.uint8),
                                  100_000_000, 160_000_000)
    home = Settlement("home", "Home", 15, 14, "city", "port", 1.0, 10_000, 20_000)
    town = Settlement("new-port", "New Port", 15, 35, "town", "port", 0.8, 2_500, 4_000)
    return grid, thematic, population, home, town


class FrontierSiteIntegrationTests(unittest.TestCase):
    def test_small_island_cannot_borrow_population_across_the_sea(self):
        grid, _thematic, population, _home, town = fixture()
        grid.water[:, 35:] = 1
        grid.water[15:17, 35] = 0
        population = PopulationLayers(
            np.where(grid.water == 0, 1.0e-7, 0).astype(np.float32),
            np.where(grid.water == 0, 6, 0).astype(np.uint8),
            100_000_000, 160_000_000,
        )
        low, high = _population_interval(grid, population, town.row, town.column,
                                         frontier=True, density=0.001)
        self.assertLessEqual(low, 20)
        self.assertLessEqual(high, 32)
        self.assertGreater(high, 0)

    def test_coast_and_display_band_alone_do_not_create_an_eight_thousand_person_town(self):
        grid, thematic, _population, home, _town = fixture()
        grid.water[:, 35:] = 1
        grid.water[15:17, 40] = 0
        population = PopulationLayers(np.where(grid.water == 0, 1.0e-7, 0).astype(np.float32),
            np.where(grid.water == 0, 6, 0).astype(np.uint8), 100_000_000, 160_000_000)
        states = np.where(grid.water == 0, 1, -1).astype(np.int16)
        states[15:17, 40] = 0
        politics = SimpleNamespace(state_id=states, frontier=states == 0)
        cultures = SimpleNamespace(civilization_id=np.where(grid.water == 0, 1, -1))
        transport = TransportLayers(np.zeros(grid.shape, dtype=np.float32), ())
        additions = derive_strategic_sites(grid, thematic, population, (home,), cultures,
            transport, politics, border_count=0, frontier_count=1)
        self.assertEqual(len(additions), 1)
        self.assertEqual(additions[0].tier, "site")
        self.assertLessEqual(additions[0].population_max, 32)
        self.assertEqual(derive_strategic_sites(grid, thematic, population, (home,), cultures,
            transport, politics, border_count=0, frontier_count=0), ())

    def test_late_town_gets_a_real_sea_route_and_administered_island_before_provinces(self):
        grid, thematic, population, home, town = fixture()
        labels = np.zeros(grid.shape, dtype=np.int32)
        labels[:, :15] = 1
        labels[12:19, 35:43] = 2
        graph = ControlRegionGraph(np.array([0, 1, 1]), np.ones(3),
            np.array([0, 465, 56], dtype=np.int32), ())
        cultures = SimpleNamespace(civilization_id=np.where(grid.water == 0, 1, -1))
        calls = []

        def countries(_grid, _thematic, _population, towns, _cultures, transport, _lexicon, **kwargs):
            owners = attach_maritime_control_regions(graph, np.array([0, 1, 0]), labels,
                np.array([0, 1]), towns, transport.routes, travel_days_by_edge=(),
                maximum_response_days_by_state=np.array([0., 12.]),
                maritime_days_by_route={route.identifier: 2. for route in transport.routes
                                        if route.mode == "sea"})
            state_id = owners[labels].astype(np.int16)
            state_id[grid.water != 0] = -1
            calls.append(("countries", tuple(item.identifier for item in towns)))
            return SimpleNamespace(state_id=state_id, frontier=(grid.water == 0) & (state_id == 0),
                states=(SimpleNamespace(identifier=1),))

        def strategic(_grid, _thematic, _population, towns, _cultures, transport, politics, **kwargs):
            if kwargs.get("border_count") == 0:
                self.assertEqual(int(politics.state_id[town.row, town.column]), 0)
                return (town,)
            self.assertEqual(kwargs["frontier_count"], 0)
            self.assertEqual(int(politics.state_id[town.row, town.column]), 1)
            calls.append(("final-borders", ()))
            return ()

        def provinces(_grid, _thematic, _population, towns, transport, _cultures, politics):
            self.assertEqual(int(politics.state_id[town.row, town.column]), 1)
            calls.append(("provinces", ()))
            return SimpleNamespace(province_id=politics.state_id)

        stubs = {
            "derive_population": lambda *a, **k: population,
            "load_name_lexicon": lambda *a: SimpleNamespace(),
            "derive_settlements": lambda *a, **k: (home,),
            "derive_cultures": lambda *a, **k: cultures,
            "assign_culture_lineages": lambda value, *a: value,
            "name_settlements": lambda value, *a: value,
            "align_culture_names": lambda value, *a: value,
            "derive_religions": lambda *a, **k: (SimpleNamespace(), a[3]),
            "derive_state_formation_profiles": lambda *a: {},
            "derive_politics": countries,
            "derive_administrative_centres": lambda *a, **k: (),
            "promote_border_settlements": lambda *a: a[-1],
            "derive_strategic_sites": strategic,
            "derive_provinces": provinces,
            "extract_geographic_features": lambda *a, **k: (),
            "SocietyLayers": lambda **kwargs: SimpleNamespace(**kwargs),
        }
        with ExitStack() as stack:
            stack.enter_context(patch('world_atlas.core.society.urban_population.allocate_urban_population', side_effect=lambda grid,society:society))
            for name, substitute in stubs.items():
                stack.enter_context(patch.object(pipeline, name, side_effect=substitute))
            result = pipeline.derive_society_layers(
                grid, thematic, "fixture", raw_elevation_m=np.where(grid.water==0, 100., -100.), state_count=1,
            )
        self.assertEqual([call[0] for call in calls], ["countries", "countries", "final-borders", "provinces"])
        self.assertEqual(result.settlements, (home, town))
        self.assertTrue(any(route.mode == "sea" and town.identifier in
            (route.source_settlement_id, route.target_settlement_id) for route in result.transport.routes))
        self.assertEqual(int(result.politics.state_id[town.row, town.column]), 1)
        self.assertIs(result.provinces.province_id, result.politics.state_id)


if __name__ == "__main__":
    unittest.main()
