"""Regression contracts for causal, terrain-owned coastal transgression."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage

from world_atlas.core.tectonic_coasts import (
    selective_tectonic_transgression,
    validate_coastal_surface,
)


def _shore_distance(land: np.ndarray, cell_km: float) -> np.ndarray:
    padded = np.concatenate((land[:, -1:], land, land[:, :1]), axis=1)
    inside = ndimage.distance_transform_edt(padded)[:, 1:-1]
    outside = ndimage.distance_transform_edt(~padded)[:, 1:-1]
    return np.maximum(inside, outside) * cell_km


def _coastal_fixture(*, active: bool) -> tuple[np.ndarray, ...]:
    """A lowland with three river-valley-scale coastal troughs."""

    height, width = 240, 480
    surface = np.full((height, width), 620.0, dtype=np.float32)
    surface[:, :125] = -220.0
    rows = np.arange(height, dtype=np.float32)[:, None]
    columns = np.arange(width, dtype=np.float32)[None, :]
    valleys = sum(
        np.exp(-0.5 * np.square((rows - centre) / 5.5))
        for centre in (50.0, 120.0, 190.0)
    )
    surface -= 590.0 * valleys * np.exp(-0.5 * np.square((columns - 126.0) / 22.0))
    land = surface > 0.0
    distance = _shore_distance(land, 20.0)
    boundaries = np.zeros((height, width), dtype=np.int8)
    velocity_east = np.zeros_like(surface)
    if active:
        # A plate-velocity discontinuity and a convergent trace sit on the
        # coastal side of the same terrain.  Only the tectonic context differs
        # from the quiet reference fixture.
        boundaries[:, 127] = 1
        velocity_east[:, 127:] = 5.0
    velocity_north = np.zeros_like(surface)
    quiet = np.zeros_like(surface)
    latitude = np.linspace(80.0, -80.0, height)
    return (
        surface,
        distance,
        boundaries,
        velocity_east,
        velocity_north,
        latitude,
        quiet,
    )


class TectonicCoastTests(unittest.TestCase):
    def _run(self, active: bool):
        return selective_tectonic_transgression(*_coastal_fixture(active=active), 0.65)

    def test_drowns_preexisting_lowlands_not_an_independent_coast_noise_field(self):
        source, *_ = _coastal_fixture(active=True)
        result, metrics = self._run(True)
        new_water = (source > 0.0) & (result <= 0.0)
        self.assertGreater(int(np.count_nonzero(new_water)), 0)
        # The valleys at rows 50/120/190 are expanded; the high interfluves
        # remain land, which distinguishes this from a uniform shoreline wiggle.
        self.assertLess(float(result[50, 125]), 0.0)
        self.assertGreater(float(result[85, 125]), 0.0)
        self.assertGreater(metrics["lowlandWeightedIncisionShare"], 0.90)
        self.assertEqual(metrics["outsideCoastalBandChangedCellCount"], 0)

    def test_active_kinematics_increase_response_for_identical_relief(self):
        active, active_metrics = self._run(True)
        quiet, quiet_metrics = self._run(False)
        active_lowering = float(np.sum(_coastal_fixture(active=True)[0] - active))
        quiet_lowering = float(np.sum(_coastal_fixture(active=False)[0] - quiet))
        self.assertGreater(active_lowering, quiet_lowering * 1.4)
        self.assertGreater(
            active_metrics["meanIncisionActiveMarginMeters"],
            quiet_metrics["meanIncisionActiveMarginMeters"],
        )

    def test_new_water_is_always_connected_and_replay_is_exact(self):
        source, *_ = _coastal_fixture(active=True)
        first, first_metrics = self._run(True)
        second, second_metrics = self._run(True)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_metrics, second_metrics)
        seed_water = source <= 0.0
        water = first <= 0.0
        padded_seed = np.concatenate((seed_water[:, -1:], seed_water, seed_water[:, :1]), axis=1)
        padded_water = np.concatenate((water[:, -1:], water, water[:, :1]), axis=1)
        connected = ndimage.binary_propagation(padded_seed, mask=padded_water)[:, 1:-1]
        self.assertTrue(np.all(connected[(source > 0.0) & water]))
        self.assertEqual(first_metrics["rejectedIsolatedFloodCellCount"], 0)

    def test_post_synthesis_validation_uses_actual_surface_not_render_geometry(self):
        land = np.zeros((48, 96), dtype=bool)
        land[10:38, 16:80] = True
        elevation = np.zeros_like(land, dtype=np.float32)
        # A quiet, cliff-like highland touching the ocean is reported for
        # review; adding a plate-boundary trace makes its tectonic setting
        # explicit without altering a pixel of the source terrain.
        elevation[23, 16] = 0.95
        boundaries = np.zeros_like(land, dtype=np.int8)
        latitude = np.linspace(88.0, -88.0, land.shape[0])
        quiet = validate_coastal_surface(land, elevation, boundaries, latitude)
        self.assertEqual(quiet["leftMapEdgeOceanFraction"], 1.0)
        self.assertEqual(quiet["rightMapEdgeOceanFraction"], 1.0)
        self.assertGreater(quiet["unsupportedCoastalHighlandTerminalFraction"], 0.9)
        boundaries[23, 17] = 1
        active = validate_coastal_surface(land, elevation, boundaries, latitude)
        self.assertLess(
            active["unsupportedCoastalHighlandTerminalFraction"],
            quiet["unsupportedCoastalHighlandTerminalFraction"],
        )


if __name__ == "__main__":
    unittest.main()
