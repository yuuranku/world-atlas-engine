"""Defended passes need important through traffic and a real terrain choke."""

from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.society.model import PopulationLayers, Settlement, TransportLayers, TransportRoute
from world_atlas.core.society.strategic_sites import (
    _road_bottleneck_scores,
    _route_cells,
    derive_strategic_sites,
    promote_border_settlements,
)
from world_atlas.core.society.population import derive_settlements


def fixture():
    shape = (61, 121)
    elevation = np.full(shape, 0.20, dtype=np.float32)
    elevation[:, 57:64] = 0.62
    elevation[29:32, 57:64] = 0.34
    grid = SimpleNamespace(shape=shape, elevation=elevation,
        water=np.zeros(shape, dtype=np.uint8), snow=np.zeros(shape, dtype=bool),
        river_order=np.zeros(shape, dtype=np.uint8),
        metadata={"planet": {"radiusKm": 6400.0},
                  "extents": {"north": 60.0, "south": -60.0, "west": -180.0, "east": 180.0}})
    thematic = SimpleNamespace(land_potential=np.full(shape, 0.50, dtype=np.float32),
                               habitability=np.full(shape,0.65,dtype=np.float32))
    population = PopulationLayers(np.full(shape, 1.0 / np.prod(shape), dtype=np.float32),
        np.full(shape, 3, dtype=np.uint8), 1_000_000, 2_000_000)
    state_id = np.ones(shape, dtype=np.int16)
    state_id[:, 60:] = 2
    politics = SimpleNamespace(state_id=state_id, frontier=np.zeros(shape, dtype=bool))
    cultures = SimpleNamespace(civilization_id=np.ones(shape, dtype=np.int16))
    settlements = (
        Settlement("west", "West", 30, 10, "city", "market", 1.0, 8_000, 12_000),
        Settlement("east", "East", 30, 110, "city", "market", 1.0, 8_000, 12_000),
    )
    # Intermediate points identify this long road's intended direction around
    # the wrapped world, rather than the shorter route across its date line.
    route = TransportRoute("road", "road", "regional", "west", "east",
        tuple((column + 0.5, 30.5) for column in range(10, 111)))
    transport = TransportLayers(np.ones(shape, dtype=np.float32), (route,))
    return grid, thematic, population, cultures, politics, settlements, transport


class StrategicPassTests(unittest.TestCase):
    def test_display_density_classes_do_not_change_strategic_sites(self):
        first = list(fixture())
        second = list(fixture())
        second[2] = PopulationLayers(second[2].population_weight,
            np.full(second[0].shape, 6, dtype=np.uint8),
            second[2].population_min, second[2].population_max)
        self.assertEqual(self.additions(first), self.additions(second))

    def additions(self, values):
        grid, thematic, population, cultures, politics, settlements, transport = values
        return derive_strategic_sites(grid, thematic, population, settlements, cultures,
            transport, politics, border_count=8, frontier_count=0)

    def test_major_through_road_over_saddle_earns_a_gate(self):
        values = fixture()
        sites = self.additions(values)
        self.assertTrue(any(site.site_type == "pass" for site in sites))
        for site in sites:
            if site.site_type == "pass":
                self.assertEqual(site.row, 30)
                self.assertLessEqual(abs(site.column - 60), 3)

    def test_saddle_inside_one_country_is_still_strategic(self):
        values = fixture()
        values[4].state_id[:] = 1
        self.assertTrue(any(site.site_type == "pass" for site in self.additions(values)))

    def test_hillside_plateau_and_uniform_valley_do_not_earn_gates(self):
        for terrain in ("hillside", "plateau", "valley"):
            with self.subTest(terrain=terrain):
                values = fixture()
                grid, _thematic, _population, _cultures, _politics, _settlements, transport = values
                if terrain == "hillside":
                    grid.elevation[:] = np.linspace(0.0, 1.0, grid.shape[0])[:, None]
                elif terrain == "plateau":
                    grid.elevation[:] = 0.55
                else:
                    grid.elevation[:] = 0.62
                    grid.elevation[29:32, :] = 0.34
                scores = _road_bottleneck_scores(grid, _route_cells(transport.routes[0].path, grid.shape))
                self.assertFalse(np.any(scores > 0.0))
                self.assertFalse(any(site.site_type == "pass" for site in self.additions(values)))

    def test_local_road_or_road_to_minor_site_cannot_create_a_strategic_gate(self):
        for minor_endpoint in (False, True):
            with self.subTest(minor_endpoint=minor_endpoint):
                values = list(fixture())
                route = values[6].routes[0]
                if minor_endpoint:
                    east = values[5][1]
                    values[5] = (values[5][0], Settlement(east.identifier, east.name, east.row,
                        east.column, "site", "pass", east.score, 300, 500))
                    importance = "regional"
                else:
                    importance = "local"
                values[6] = TransportLayers(values[6].accessibility,
                    (TransportRoute(route.identifier, route.mode, importance,
                        route.source_settlement_id, route.target_settlement_id, route.path),))
                self.assertEqual(self.additions(values), ())

    def test_saved_pass_roles_are_rechecked_without_moving_admin_cores_or_ports(self):
        grid, thematic, _population, cultures, politics, settlements, transport = fixture()
        old_pass = Settlement("old-pass", "Old Pass", 12, 40, "site", "pass", 1.0, 300, 500)
        gate_town = Settlement("gate-town", "Gate Town", 30, 60, "town", "market", 1.0, 3_000, 5_000)
        port = Settlement("port", "Port", 30, 59, "town", "port", 1.0, 3_000, 5_000)
        before = settlements + (old_pass, gate_town, port)
        after = promote_border_settlements(grid, thematic, cultures, politics, transport, before)
        self.assertEqual([(s.identifier, s.name, s.row, s.column) for s in before],
                         [(s.identifier, s.name, s.row, s.column) for s in after])
        by_id = {s.identifier: s for s in after}
        self.assertEqual(by_id["old-pass"].site_type, "market")
        self.assertEqual(by_id["gate-town"].site_type, "pass")
        self.assertEqual(by_id["port"], port)

    def test_pre_road_population_selection_never_invents_a_mountain_pass_quota(self):
        from test_population_water import _water_fixture
        grid, thematic, population = _water_fixture()
        grid.elevation[:] = 0.52
        grid.elevation[::2, :] += 0.022
        towns = derive_settlements(grid, thematic, population, target_count=60)
        self.assertTrue(towns)
        self.assertFalse(any(town.site_type == "pass" for town in towns))


if __name__ == "__main__":
    unittest.main()
