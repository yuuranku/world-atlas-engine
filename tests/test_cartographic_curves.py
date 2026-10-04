"""Accepted terrain guides channel geometry; unobserved waves are not added."""
from types import SimpleNamespace
from pathlib import Path
import json
import unittest

import numpy as np
from shapely import LineString, Point, box

from world_atlas.core.cartographic_curves import _rounded_channel, terrain_channel_paths
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.cartographic_rivers import river_channel_surface


def _curve(grid, path, *, terrain_field):
    return terrain_channel_paths(grid, (path,), terrain_field=terrain_field)[0]


def fixture(values=None):
    if values is None:
        values = np.full((64, 64), 200.)
    raw = np.asarray(values, dtype=np.float64)
    grid = SimpleNamespace(shape=raw.shape, water=np.where(raw > 0, 0, 1).astype(np.uint8),
                           elevation=np.clip(raw / 10000., 0., 1.).astype(np.float32))
    field = PhysicalTerrainField(raw, land_mask=raw > 0, sea_level_m=0,
                                 elevation_scale_m=10000., elevation_exponent=1.06)
    return grid, field


def valley_fixture():
    rows = np.arange(64) + .5
    raw = np.broadcast_to(200. + 40. * (rows[:, None] - 30.5) ** 2, (64, 64)).copy()
    return fixture(raw)


class CartographicCurveTests(unittest.TestCase):
    def test_flat_ground_preserves_straight_geometry_without_waves(self):
        grid, field = fixture()
        path = np.array(((4.5, 30.5), (28.5, 30.5)))
        curve = _curve(grid, path, terrain_field=field)
        self.assertGreater(len(curve), len(path))
        self.assertTrue(LineString(curve).equals(LineString(path)))
        np.testing.assert_array_equal(curve[[0, -1]], path[[0, -1]])

    def test_supported_native_elbows_round_without_moving_endpoints_or_leaving_corridor(self):
        grid, field = fixture()
        for endpoint in ((20.5, 48.5), (12.5, 38.5)):
            with self.subTest(endpoint=endpoint):
                path = np.array(((4.5, 30.5), (20.5, 30.5), endpoint))
                curve = _curve(grid, path, terrain_field=field)
                self.assertFalse(LineString(curve).equals(LineString(path)))
                self.assertLess(max(Point(point).distance(LineString(path)) for point in curve), .25)
                self.assertTrue(LineString(curve).is_simple)
                np.testing.assert_array_equal(curve[[0, -1]], path[[0, -1]])
                vectors = np.diff(curve, axis=0)
                turn = np.abs(np.arctan2(vectors[:-1, 0]*vectors[1:, 1]-vectors[:-1, 1]*vectors[1:, 0],
                                        np.einsum('ij,ij->i', vectors[:-1], vectors[1:])))
                self.assertLess(float(np.rad2deg(turn.max())), 40.)
                reverse = _curve(grid, path[::-1], terrain_field=field)
                np.testing.assert_allclose(curve, reverse[::-1], rtol=0., atol=1e-11)

    def test_rounding_does_not_climb_an_actual_hill_or_add_an_uphill_step(self):
        rows, columns = np.indices((64, 64), dtype=float)+.5
        elbow = np.array(((29.5, 30.5), (30.5, 30.5), (30.5, 31.5)))
        for direction in (1., -1.):
            with self.subTest(direction=direction):
                ground = 100 + direction*10*np.maximum(0., 30.5-columns)*np.maximum(0., rows-30.5)
                grid, field = fixture(ground)
                # A hill leaves the original river floor; a dip then rises
                # again and would add an uphill leg to this level reach.
                self.assertNotEqual(float(field.sample_points(30.4, 30.6)), 100.)
                curve = _rounded_channel(grid, elbow, np.array((0., .25, 0.)), terrain_field=field)
                np.testing.assert_array_equal(curve, elbow)

    def test_rounding_preserves_confluence_embedding_and_fractional_mouth(self):
        grid, field = fixture()
        junction = np.array((18.5, 26.5))
        paths = (np.array(((4.5, 30.5), (14.5, 30.5), (14.5, 26.5), junction)),
                 np.array(((18.5, 14.5), junction)),
                 np.array((junction, (28.5, 26.5), (28.5, 38.73))))
        curves = terrain_channel_paths(grid, paths, terrain_field=field)
        for source, curve in zip(paths, curves, strict=True):
            np.testing.assert_array_equal(curve[[0, -1]], source[[0, -1]])
        for first in range(len(curves)):
            for second in range(first+1, len(curves)):
                self.assertTrue(LineString(curves[first]).intersection(LineString(curves[second])).equals(Point(junction)))

    def test_adjacent_elbows_remain_disjoint_in_the_shared_network(self):
        grid, field = fixture()
        paths = (np.array(((4.5, 30.5), (14.5, 30.5), (14.5, 40.5))),
                 np.array(((4.5, 30.65), (14.35, 30.65), (14.35, 40.5))))
        curves = terrain_channel_paths(grid, paths, terrain_field=field)
        self.assertTrue(LineString(curves[0]).disjoint(LineString(curves[1])))
        self.assertTrue(all(LineString(curve).is_simple for curve in curves))
        for source, curve in zip(paths, curves, strict=True):
            np.testing.assert_array_equal(curve[[0, -1]], source[[0, -1]])

    def test_convex_valley_follows_lower_accepted_ground_within_quarter_cell(self):
        grid, field = valley_fixture()
        path = np.array(((4.5, 30.8), (28.5, 30.8)))
        curve = _curve(grid, path, terrain_field=field)
        offset = 30.8 - curve[1:-1, 1]
        self.assertGreater(float(offset.max()), .01)
        self.assertTrue(np.all(offset >= 0))
        self.assertLessEqual(float(offset.max()), .25 + 1e-12)
        before = field.sample_points(curve[:, 0], np.full(len(curve), 30.8))
        after = field.sample_points(curve[:, 0], curve[:, 1])
        self.assertTrue(np.all(after <= before + 1e-9))
        self.assertLess(float(after[1:-1].mean()), float(before[1:-1].mean()))
        np.testing.assert_array_equal(curve[[0, -1]], path)

    def test_planar_slope_does_not_invent_a_valley(self):
        rows = np.arange(64) + .5
        raw = np.broadcast_to(200. + 20. * rows[:, None], (64, 64))
        grid, field = fixture(raw)
        path = np.array(((4.5, 30.8), (28.5, 30.8)))
        curve = _curve(grid, path, terrain_field=field)
        np.testing.assert_allclose(curve[:, 1], 30.8, atol=1e-12)

    def test_reconstruction_and_reversal_are_deterministic(self):
        grid, field = valley_fixture()
        path = np.array(((4.5, 30.8), (20.5, 30.8), (28.5, 34.5)))
        curve = _curve(grid, path, terrain_field=field)
        np.testing.assert_array_equal(curve, _curve(grid, path, terrain_field=field))
        reverse = _curve(grid, path[::-1], terrain_field=field)
        np.testing.assert_allclose(curve, reverse[::-1], rtol=0., atol=1e-11)

    def test_reach_endpoints_keep_exact_shared_confluences(self):
        grid, field = valley_fixture()
        junction = (18.5, 30.8)
        incoming = (np.array(((4.5, 30.8), junction)),
                    np.array(((18.5, 7.5), junction)))
        outgoing = np.array((junction, (30.5, 30.8)))
        for path in incoming:
            curve = _curve(grid, path, terrain_field=field)
            np.testing.assert_array_equal(curve[-1], junction)
            np.testing.assert_array_equal(curve[0], path[0])
        curve = _curve(grid, outgoing, terrain_field=field)
        np.testing.assert_array_equal(curve[0], junction)
        np.testing.assert_array_equal(curve[-1], outgoing[-1])

    def test_adjacent_native_water_prevents_cross_bank_displacement(self):
        raw = np.full((64, 64), 1000.)
        raw[30] = 500.
        raw[31:] = -10.
        grid, field = fixture(raw)
        path = np.array(((4.5, 30.95), (28.5, 30.95)))
        curve = _curve(grid, path, terrain_field=field)
        np.testing.assert_array_equal(curve[:, 1], np.full(len(curve), 30.95))
        self.assertTrue(LineString(curve).disjoint(box(0, 31, 64, 64)))
        self.assertTrue(np.all(field.sample_points(curve[:, 0], curve[:, 1]) > 0))

    def test_native_land_cannot_license_moving_below_continuous_shoreline(self):
        raw = np.full((64, 64), 1000.)
        raw[30] = 1.
        raw[31:] = -5.
        grid, field = fixture(raw)
        path = np.array(((4.5, 30.5), (28.5, 30.5)))
        curve = _curve(grid, path, terrain_field=field)
        # The PCHIP zero contour lies inside native row 30; a convex
        # cross-section would otherwise move to negative ground there.
        self.assertTrue(np.all(field.sample_points(curve[:, 0], curve[:, 1]) > 0))
        self.assertTrue(LineString(curve).equals(LineString(path)))

    def test_diagonal_corridor_remains_within_its_land_cells(self):
        raw = np.full((64, 64), -100.)
        np.fill_diagonal(raw, 1000.)
        grid, field = fixture(raw)
        path = np.array(((4.5, 4.5), (28.5, 28.5)))
        curve = _curve(grid, path, terrain_field=field)
        self.assertTrue(np.all(grid.water[np.floor(curve[:, 1]).astype(int),
                                         np.floor(curve[:, 0]).astype(int)] == 0))
        self.assertTrue(np.all(field.sample_points(curve[:, 0], curve[:, 1]) > 0))
        self.assertLessEqual(max(Point(point).distance(LineString(path)) for point in curve), .25 + 1e-12)

    def test_meridian_jumps_are_rejected_and_already_split_reaches_work(self):
        grid, field = fixture()
        seam = np.array(((1.5, 30.5), (62.5, 30.5)))
        with self.assertRaisesRegex(ValueError, "split.*seam"):
            _curve(grid, seam, terrain_field=field)
        for path in (np.array(((1.5, 30.5), (0., 30.5))),
                     np.array(((64., 30.5), (62.5, 30.5)))):
            curve = _curve(grid, path, terrain_field=field)
            np.testing.assert_array_equal(curve[[0, -1]], path)
            self.assertTrue(np.all((curve[:, 0] >= 0) & (curve[:, 0] <= 64)))

    def test_all_inputs_remain_unchanged_even_with_read_only_anchors(self):
        grid, field = valley_fixture()
        path = np.array(((4.5, 30.8), (28.5, 30.8)))
        path.flags.writeable = False
        before = path.copy(), grid.water.copy(), grid.elevation.copy(), field.native_m.copy()
        _curve(grid, path, terrain_field=field)
        for actual, expected in zip((path, grid.water, grid.elevation, field.native_m), before, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_invalid_anchors_and_misaligned_terrain_are_rejected(self):
        grid, field = fixture()
        for path in (np.array([1., 2.]), np.array([[1., np.nan], [2., 3.]]), np.ones((3, 3))):
            with self.assertRaisesRegex(ValueError, "anchors"):
                _curve(grid, path, terrain_field=field)
        _, smaller = fixture(np.ones((8, 8)))
        with self.assertRaisesRegex(ValueError, "aligned"):
            _curve(grid, np.array(((4.5, 4.5), (8.5, 8.5))), terrain_field=smaller)

    def test_saved_v91_valley_elbow_preserves_station_order_and_valid_banks(self):
        for name in ("river-elbow-v91.json", "river-stations-v91.json"):
            with self.subTest(name=name):
                data = json.loads((Path(__file__).parent / "fixtures" / name).read_text())
                grid, field = fixture(data["ground"])
                grid.metadata = {"planet": {"radiusKm": 6400.}, "extents": data["extents"]}
                path = np.asarray(data["points"])
                curve = _curve(grid, path, terrain_field=field)
                np.testing.assert_allclose(curve, _curve(grid,path[::-1],terrain_field=field)[::-1],
                                           rtol=0.,atol=1e-10)
                # The first saved elbow used to fold two stations onto
                # (1751.25,598.75); the second caught sequential order bias.
                self.assertTrue(np.all(np.linalg.norm(np.gradient(curve,axis=0),axis=1)>0))
                self.assertTrue(LineString(curve).is_simple)
                channel = river_channel_surface(grid,curve,np.full(len(curve),100.))
                self.assertTrue(channel.is_valid)
                self.assertGreater(channel.area,0.)
                np.testing.assert_array_equal(curve[[0,-1]],path[[0,-1]])


if __name__ == "__main__":
    unittest.main()
