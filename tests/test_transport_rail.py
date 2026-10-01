from types import SimpleNamespace
import unittest

import numpy as np

from world_atlas.core.society.model import Settlement, TransportLayers
from world_atlas.core.society.transport import (
    RoutingCache,
    _path_cells,
    _rail_routes,
    _rail_engineering_fields,
    conform_transport_to_politics,
    derive_bridges,
)


def _fixture(era: str):
    height, width = 31, 41
    water = np.zeros((height, width), dtype=np.uint8)
    water[0, :] = 1
    water[-1, :] = 1
    snow = np.zeros((height, width), dtype=bool)
    elevation = np.zeros((height, width), dtype=np.float32)
    river_order = np.zeros((height, width), dtype=np.uint8)
    river_order[1:-1, 20] = 3
    flow_to = np.full((height, width), -1, dtype=np.int32)
    for row in range(1, height - 2):
        flow_to[row, 20] = (row + 1) * width + 20
    grid = SimpleNamespace(
        shape=(height, width),
        water=water,
        snow=snow,
        elevation=elevation,
        river_order=river_order,
        flow_to=flow_to,
        metadata={"worldProfile": {"technologyEra": era},"extents":{"west":-180.,"east":180.,"north":90.,"south":-90.},"planet":{"radiusKm":6371.}},
    )
    thematic = SimpleNamespace(land_potential=np.full((height, width), 0.72, dtype=np.float32),
                               habitability=np.full((height, width), 0.75, dtype=np.float32))
    settlements = (
        Settlement(
            identifier="west-metropolis",
            name="West Metropolis",
            row=15,
            column=15,
            tier="metropolis",
            site_type="market",
            score=1.0,
            population_min=900_000,
            population_max=1_300_000,
        ),
        Settlement(
            identifier="east-metropolis",
            name="East Metropolis",
            row=15,
            column=25,
            tier="metropolis",
            site_type="market",
            score=1.0,
            population_min=800_000,
            population_max=1_200_000,
        ),
    )
    return grid, thematic, settlements


class RailwayGenerationTests(unittest.TestCase):
    def test_flat_coastal_ground_is_traversable_while_real_land_grade_remains_costly(self):
        grid, thematic, _settlements = _fixture("industrial")
        grid.elevation[grid.water == 0] = .4
        friction, valid = _rail_engineering_fields(grid, thematic)
        self.assertTrue(np.all(valid[grid.water == 0]))
        self.assertFalse(np.any(valid[grid.water != 0]))
        # The water display value is absent from the land derivative; actual
        # adjacent dry samples still create an engineering grade.
        flat_cost = float(friction[1, 10])
        grid.elevation[1:5, 10:15] = .8
        uphill_cost, uphill_valid = _rail_engineering_fields(grid, thematic)
        self.assertGreater(float(uphill_cost[1, 10]), flat_cost)
        self.assertFalse(uphill_valid[1, 10])

    def test_industrial_rail_is_deterministic_independent_and_bridge_validated(self):
        grid, thematic, settlements = _fixture("industrial")
        first = _rail_routes(grid, thematic, settlements, routing_cache=RoutingCache())
        second = _rail_routes(grid, thematic, settlements, routing_cache=RoutingCache())

        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        route = first[0]
        self.assertEqual(route.mode, "rail")
        self.assertEqual(route.importance, "trunk")
        cells = _path_cells(route.path, grid.shape)
        self.assertTrue(all(int(grid.water[row, column]) == 0 for row, column in cells))
        river_cells = [(row, column) for row, column in cells if grid.river_order[row, column] > 0]
        self.assertGreaterEqual(len(river_cells), 1)
        self.assertLessEqual(len(river_cells), 3)
        bridges = derive_bridges(grid, first,raw_elevation_m=np.where(grid.water==0,grid.elevation*1000+1.,-1.))
        self.assertEqual(len(bridges), 1)
        self.assertEqual(bridges[0].route_identifier, route.identifier)

    def test_nonindustrial_eras_emit_no_rail(self):
        for era in ("tribal", "ancient", "medieval", "early-modern", "preindustrial"):
            grid, thematic, settlements = _fixture(era)
            self.assertEqual(
                _rail_routes(grid, thematic, settlements, routing_cache=RoutingCache()),
                (),
                era,
            )

    def test_rail_requires_a_strict_formal_era(self):
        grid, thematic, settlements = _fixture("industrial")
        grid.metadata["worldProfile"]["technologyEra"] = "steam-punk"
        with self.assertRaisesRegex(ValueError, "technologyEra"):
            _rail_routes(grid, thematic, settlements, routing_cache=RoutingCache())

    def test_domestic_rail_is_rerouted_or_removed_when_a_state_blocks_its_corridor(self):
        grid, thematic, settlements = _fixture("industrial")
        routes = _rail_routes(grid, thematic, settlements, routing_cache=RoutingCache())
        state_id = np.ones(grid.shape, dtype=np.int16)
        state_id[1:-1, 20] = 2
        politics = SimpleNamespace(state_id=state_id)
        transport = TransportLayers(
            accessibility=np.zeros(grid.shape, dtype=np.float32),
            routes=routes,
        )

        conformed = conform_transport_to_politics(
            grid,
            thematic,
            settlements,
            transport,
            politics,
            raw_elevation_m=np.where(grid.water==0,grid.elevation*1000+1.,-1.),
        )

        self.assertEqual(len(conformed.routes), 1)
        route = conformed.routes[0]
        self.assertEqual(route.mode, "rail")
        self.assertTrue(
            all(
                int(state_id[row, column]) == 1
                for row, column in _path_cells(route.path, grid.shape)
            )
        )
        self.assertEqual(conformed.bridges, ())

    def test_international_rail_never_transits_an_unrelated_state(self):
        grid, thematic, settlements = _fixture("contemporary")
        routes = _rail_routes(grid, thematic, settlements, routing_cache=RoutingCache())
        state_id = np.full(grid.shape, 3, dtype=np.int16)
        state_id[:, :20] = 1
        state_id[:, 25:] = 2
        politics = SimpleNamespace(state_id=state_id)
        transport = TransportLayers(
            accessibility=np.zeros(grid.shape, dtype=np.float32),
            routes=routes,
        )

        conformed = conform_transport_to_politics(
            grid,
            thematic,
            settlements,
            transport,
            politics,
            raw_elevation_m=np.where(grid.water==0,grid.elevation*1000+1.,-1.),
        )

        self.assertEqual(len(conformed.routes), 1)
        route = conformed.routes[0]
        self.assertEqual(route.mode, "rail")
        self.assertTrue(
            all(
                int(state_id[row, column]) in {1, 2}
                for row, column in _path_cells(route.path, grid.shape)
            )
        )


if __name__ == "__main__":
    unittest.main()
