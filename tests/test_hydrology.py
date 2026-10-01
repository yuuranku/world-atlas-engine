"""Regression contracts for canonical tributaries and seasonal water flow."""

import unittest

import numpy as np

from world_atlas.core.baseline import _seasonal_river_strength
from world_atlas.physical.hydrology import HydrologyConfig, compute_hydrology, _strahler_order


def _sloped_catchment() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one deterministic terrain with several supported tributaries."""

    height, width = 48, 64
    rows, columns = np.indices((height, width))
    land = np.zeros((height, width), dtype=bool)
    land[1:-1, 1:-1] = True
    ocean = ~land
    elevation = (
        (height - 1 - rows) * 0.020
        + 0.040 * np.abs(columns - (width // 2)) / (width // 2)
    ).astype(np.float64)
    elevation[rows < 8] += 0.25
    elevation[rows > 38] -= 0.10
    return elevation, land, ocean


class CanonicalRiverNetworkTests(unittest.TestCase):
    def test_remote_dry_land_does_not_erase_a_qualified_short_watercourse(self):
        # Keep the same local terrain, rain and drainage graph. Adding a dry
        # continent far away formerly raised the global source-length cutoff
        # from the configured two cells to three and erased this whole stream.
        shape = (190, 210)
        land = np.zeros(shape, dtype=bool)
        land[4:6, 4] = True
        elevation = np.zeros(shape)
        elevation[4:6, 4] = (0.95, 0.85)
        runoff = np.zeros(shape)
        runoff[4:6, 4] = 1.0
        config = HydrologyConfig(
            minimum_headwater_length=2,
            stream_threshold_fraction=0.0,
            tributary_threshold_fraction=0.0,
            mainstem_threshold_fraction=0.0,
            minimum_stream_span_factor=0.0,
            valley_support_threshold=0.0,
            valley_window_fraction=0.0,
            major_rivers_per_continent=0,
            closure_budget_fraction=0.0,
        )
        local = compute_hydrology(
            elevation, land, ~land, config=config, runoff_source_yield=runoff,
        )
        land[10:185, 20:205] = True
        elevation[10:185, 20:205] = np.linspace(0.65, 0.15, 175)[:, None]
        expanded = compute_hydrology(
            elevation, land, ~land, config=config, runoff_source_yield=runoff,
        )
        np.testing.assert_array_equal(local.downstream_index[4:6, 4], expanded.downstream_index[4:6, 4])
        self.assertTrue(np.all(local.stream_mask[4:6, 4]))
        np.testing.assert_array_equal(local.stream_mask[4:6, 4], expanded.stream_mask[4:6, 4])
        self.assertFalse(np.any(expanded.stream_mask[10:185, 20:205]))

    def test_strahler_order_counts_equal_tributaries_only(self):
        # Four headwaters form two order-2 reaches. Their confluence becomes
        # order 3; a later order-1 tributary must leave that order unchanged.
        downstream = np.array([[4, 4, 5, 5, 6, 6, 8, 8, -1]], dtype=np.int32)
        order = _strahler_order(np.ones(downstream.shape, bool), downstream, list(range(9)))
        np.testing.assert_array_equal(order, [[1, 1, 1, 1, 2, 2, 3, 1, 3]])

    def test_seasonality_does_not_erase_supported_canonical_tributaries(self):
        elevation, land, ocean = _sloped_catchment()
        config = HydrologyConfig(
            stream_threshold_fraction=0.002,
            tributary_threshold_fraction=0.01,
            mainstem_threshold_fraction=0.08,
            minimum_headwater_length=3,
            minimum_stream_span_factor=0.15,
            valley_support_threshold=0.05,
            lowland_headwater_flow_multiplier=1.1,
            closure_budget_fraction=0.10,
        )
        dry = compute_hydrology(
            elevation,
            land,
            ocean,
            config=config,
            seasonality_index=np.zeros(land.shape, dtype=np.float64),
        )
        monsoon = compute_hydrology(
            elevation,
            land,
            ocean,
            config=config,
            seasonality_index=np.full(land.shape, 0.95, dtype=np.float64),
        )

        np.testing.assert_array_equal(dry.stream_mask, monsoon.stream_mask)
        np.testing.assert_array_equal(dry.stream_order, monsoon.stream_order)
        self.assertGreater(int(dry.stream_mask.sum()), 0)
        self.assertLess(int(dry.stream_mask.sum()), int(land.sum()))
        self.assertTrue(np.all(dry.stream_order[dry.stream_mask] > 0))
        self.assertTrue(np.all(~dry.stream_mask | land))
        self.assertEqual(
            dry.diagnostics["localStreamThreshold"]["seasonalityTopologyGate"],
            "none; climate runoff yield remains applied",
        )

    def test_low_order_flow_is_retained_when_present_and_zero_when_dry(self):
        shape = (3, 7)
        terrain = np.zeros(shape, dtype=bool)
        terrain[1, 1:6] = True
        downstream = np.full(shape, -1, dtype=np.int32)
        for column in range(1, 5):
            downstream[1, column] = 1 * shape[1] + column + 1
        order = np.zeros(shape, dtype=np.uint8)
        order[1, 1:6] = 1
        runoff = np.zeros((4, *shape), dtype=np.float64)
        runoff[0, 1, 1] = 1.0
        runoff[2, 1, 1] = 0.5

        strength = _seasonal_river_strength(runoff, terrain, downstream, order)

        self.assertTrue(np.all(strength[0, 1, 1:6] > 0))
        self.assertTrue(np.all(strength[2, 1, 1:6] > 0))
        self.assertTrue(np.all(strength[1, 1, 1:6] == 0))
        self.assertTrue(np.all(strength[3, 1, 1:6] == 0))
        self.assertTrue(np.all(strength[:, ~terrain] == 0))


if __name__ == "__main__":
    unittest.main()
