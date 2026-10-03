"""Real tensor-PCHIP corners are native source sections, not smoothing."""

import unittest

import numpy as np

from world_atlas.core.continuous_scalar import _threshold_segments, pchip_curve_sections
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.implicit_curves import adaptive_curve_paths
from world_atlas.core.implicit_pchip import (
    pchip_native_switch_abscissae, pchip_switch_abscissae, split_pchip_branches)


def field_for(native):
    return PhysicalTerrainField(native, land_mask=native > 0, sea_level_m=0,
                                elevation_scale_m=4000, elevation_exponent=1)


def branches(field, level):
    x = np.r_[0., np.arange(field.width)+.5, float(field.width)]
    y = np.r_[0., np.arange(field.height)+.5, float(field.height)]
    return _threshold_segments(field, x, y, field.sample_rect(x, y), level)


class ImplicitPchipTests(unittest.TestCase):
    def test_inclusive_native_events_keep_isolated_shared_ports_not_zero_relations(self):
        native = np.vstack((np.arange(-2., 4.), np.zeros(6), np.full(6, 3.),
                            np.full(6, 4.), np.full(6, 5.)))
        field = field_for(native)
        lower, upper = np.array(((1.5, 1.5), (2.5, 1.5))), np.array(((2.5, 2.5), (3.5, 2.5)))
        events = pchip_native_switch_abscissae(field, lower, upper)
        self.assertTrue(all(2.5 in event for event in events))
        self.assertTrue(all(len(event) == 0 for event in pchip_switch_abscissae(
            field, lower, upper)))
        field = field_for(np.tile(np.arange(6.), (5, 1)))
        self.assertTrue(all(len(event) == 0 for event in pchip_native_switch_abscissae(
            field, lower, upper)))

    def test_public_event_longitudes_are_true_adjacent_row_equalities(self):
        native = np.random.default_rng(89323).uniform(100, 2000, (9, 11))
        field = field_for(native)
        lower, upper = np.array(((3.5, 1.5),)), np.array(((4.5, 2.5),))
        events = pchip_switch_abscissae(field, lower, upper)[0]
        self.assertGreater(len(events), 0)
        self.assertLess(float(np.min(abs(events-4.27165524165717))), 1e-12)
        self.assertTrue(np.all((events > 3.5) & (events < 4.5)))
        rows = np.arange(4)+.5
        values = field.sample_points(events[:, None], rows[None, :])
        self.assertLess(float(np.min(abs(np.diff(values, axis=1)), axis=1).max()), 1e-10)

    def test_true_slope_switch_corner_is_an_exact_shared_port(self):
        native = np.random.default_rng(89323).uniform(100, 2000, (9, 11))
        field, level = field_for(native), 350.001
        segments, lower, upper = branches(field, level)
        sections, low, high = split_pchip_branches(field, field.native_m, segments, lower, upper, level)
        self.assertGreater(len(sections), len(segments))
        expected = np.array((4.27165524165717, 1.9827907997308))
        ports = sections.reshape(-1, 2)
        near = np.linalg.norm(ports-expected, axis=1) < 1e-12
        self.assertEqual(int(near.sum()), 2, "the true corner is one shared end/start port")
        np.testing.assert_array_equal(ports[near][0], ports[near][1])
        np.testing.assert_array_equal(field.native_m, native)

        def evaluate(starts, ends, owners):
            return pchip_curve_sections(field, field.native_m, starts, ends, low[owners], high[owners], level)

        paths = adaptive_curve_paths(sections[:, 0], sections[:, 1], evaluate)
        points = np.vstack(paths)
        self.assertLess(float(np.max(abs(field.sample_points(points[:, 0], points[:, 1])-level))), 1e-10)
        self.assertTrue(all(any(np.array_equal(port, point) for point in ports)
                            for port in segments.reshape(-1, 2)))

    def test_reversing_a_branch_preserves_all_shared_source_points(self):
        native = np.random.default_rng(89323).uniform(100, 2000, (9, 11))
        field, level = field_for(native), 350.001
        segments, lower, upper = branches(field, level)
        forward, _, _ = split_pchip_branches(field, field.native_m, segments[4:5], lower[4:5], upper[4:5], level)
        reverse, _, _ = split_pchip_branches(field, field.native_m, segments[4:5, ::-1], lower[4:5], upper[4:5], level)
        np.testing.assert_array_equal(reverse[::-1, ::-1], forward)
        # The event longitude is solved from the same cubic, independently of
        # the route through the branch; the native end ports stay exact.
        np.testing.assert_array_equal(reverse[::-1, ::-1, 0], forward[:, :, 0])

    def test_identically_equal_rows_and_vertical_branches_need_no_false_events(self):
        native = np.tile(np.array((100., 100., 250., 350., 350., 100.)), (5, 1))
        field, level = field_for(native), 200.
        segments, lower, upper = branches(field, level)
        sections, low, high = split_pchip_branches(field, field.native_m, segments, lower, upper, level)
        np.testing.assert_array_equal(sections, segments)
        np.testing.assert_array_equal(low, lower)
        np.testing.assert_array_equal(high, upper)


if __name__ == "__main__":
    unittest.main()
