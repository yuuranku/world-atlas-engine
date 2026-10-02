from types import SimpleNamespace
import unittest

import numpy as np
import shapely

from world_atlas.acceptance import (
    _settlement_surface_violations,
    _transport_surface_violations,
)
from world_atlas.core.society.model import Settlement, TransportRoute


class ReleaseAcceptanceSurfaceTests(unittest.TestCase):
    def setUp(self):
        water = np.zeros((4, 5), dtype=np.uint8)
        water[1, 2] = 1
        river_order = np.zeros((4, 5), dtype=np.uint8)
        river_order[2, 3] = 2
        self.grid = SimpleNamespace(
            shape=water.shape,
            water=water,
            river_order=river_order,
        )

    def test_rails_are_checked_separately_from_roads_and_sea_lanes(self):
        routes = (
            TransportRoute(
                identifier="road-water",
                mode="road",
                importance="local",
                source_settlement_id="a",
                target_settlement_id="b",
                path=((0.5, 1.5), (2.5, 1.5)),
            ),
            TransportRoute(
                identifier="rail-water",
                mode="rail",
                importance="regional",
                source_settlement_id="a",
                target_settlement_id="b",
                path=((0.5, 1.5), (2.5, 1.5)),
            ),
            TransportRoute(
                identifier="sea-land",
                mode="sea",
                importance="regional",
                source_settlement_id="a",
                target_settlement_id="b",
                path=((0.5, 0.5), (1.5, 0.5), (2.5, 0.5)),
            ),
        )

        violations = _transport_surface_violations(self.grid, routes,
            sea_land_geometry=shapely.box(0,0,5,4).difference(shapely.box(2,1,3,2)))

        self.assertEqual([item["route"] for item in violations["road"]], ["road-water"])
        self.assertEqual([item["route"] for item in violations["rail"]], ["rail-water"])
        self.assertEqual([item["route"] for item in violations["sea"]], ["sea-land"])

    def test_fine_sea_channels_use_actual_shore_and_reject_tiny_islands(self):
        route=TransportRoute('sea','sea','regional','a','b',((1.5,.5),(3.5,.5)))
        empty=shapely.Polygon()
        self.assertFalse(_transport_surface_violations(self.grid,(route,),sea_land_geometry=empty)['sea'])
        island=shapely.box(2.001,.499,2.002,.501)
        self.assertTrue(_transport_surface_violations(self.grid,(route,),sea_land_geometry=island)['sea'])

    def test_sailing_audit_cannot_substitute_the_native_cell_mask(self):
        route=TransportRoute('sea','sea','regional','a','b',((1.5,.5),(3.5,.5)))
        with self.assertRaisesRegex(ValueError,'actual continuous land geometry'):
            _transport_surface_violations(self.grid,(route,))

    def test_settlement_surface_checks_reject_water_and_represented_river_cells(self):
        settlements = (
            Settlement(
                identifier="on-water",
                name="On Water",
                row=1,
                column=2,
                tier="town",
                site_type="market",
                score=1.0,
                population_min=10,
                population_max=20,
            ),
            Settlement(
                identifier="on-river",
                name="On River",
                row=2,
                column=3,
                tier="town",
                site_type="market",
                score=1.0,
                population_min=10,
                population_max=20,
            ),
        )

        violations = _settlement_surface_violations(self.grid, settlements)

        self.assertEqual([item["settlement"] for item in violations["water"]], ["on-water"])
        self.assertEqual([item["settlement"] for item in violations["river"]], ["on-river"])
        self.assertEqual(violations["river"][0]["riverOrder"], 2)


if __name__ == "__main__":
    unittest.main()
