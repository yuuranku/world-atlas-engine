"""Real summit evidence, identity preservation and contained range lettering."""
from dataclasses import replace
from types import SimpleNamespace
import re
import unittest
from xml.etree import ElementTree

import numpy as np
import shapely

from world_atlas.core.render import _mountain_label_path, _river_path_data, _toponymy_overlay
from world_atlas.core.society.geographic_features import (
    mountain_components, prominent_raw_summits, refine_saved_geographic_features,
)
from world_atlas.core.society.model import GeographicFeature
from test_presentation import _fixture


def grid_with(elevation):
    return SimpleNamespace(shape=elevation.shape, elevation=elevation,
                           water=np.zeros(elevation.shape, dtype=np.uint8))


def quadratic_samples(data):
    """Decode the existing M/Q label output independently of its producer."""
    result = []
    current = None
    for command, coordinates in re.findall(r"([MQ])([^MQ]*)", data):
        values = np.asarray([float(value) for value in re.findall(
            r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", coordinates)])
        if command == "M":
            current = values
            result.append(current)
        else:
            control, endpoint = values[:2], values[2:]
            t = np.linspace(0, 1, 41)[1:, None]
            result.extend((1-t)**2*current + 2*(1-t)*t*control + t*t*endpoint)
            current = endpoint
    return np.asarray(result)


class RawSummitTests(unittest.TestCase):
    def test_clipped_crown_uses_raw_summit_and_real_local_ridge_drop(self):
        elevation = np.full((21, 31), .2, dtype=np.float32)
        elevation[5:16, 8:23] = 1.0
        grid = grid_with(elevation)
        raw = np.full(grid.shape, 300.0)
        rows, columns = np.indices(grid.shape)
        raw[elevation == 1] = (7000 - 14*(rows-10)**2 - 9*(columns-16)**2)[elevation == 1]
        summits = prominent_raw_summits(grid, raw, components=mountain_components(grid))
        self.assertEqual([(item.row, item.column) for item in summits], [(10, 16)])
        self.assertEqual(summits[0].elevation_m, 7000)
        self.assertGreater(summits[0].local_prominence_m, 80)

    def test_flat_high_plateau_has_no_point_summit(self):
        elevation = np.full((21, 31), .2, dtype=np.float32)
        elevation[5:16, 8:23] = .9
        grid = grid_with(elevation)
        raw = np.where(elevation == .9, 7000.0, 300.0)
        self.assertEqual(prominent_raw_summits(grid, raw, components=mountain_components(grid)), ())

    def test_longitude_neighbor_and_land_only_relief_prevent_false_peak(self):
        elevation = np.full((15, 25), .8, dtype=np.float32)
        grid = grid_with(elevation)
        raw = np.full(grid.shape, 1000.0)
        raw[7, 0] = 1600
        raw[7, -1] = 2000
        peaks = prominent_raw_summits(grid, raw, components=mountain_components(grid))
        self.assertEqual([(item.row, item.column) for item in peaks], [(7, 24)])
        grid.water[:] = 1
        grid.water[7, 0] = 0
        raw[:] = -5000
        raw[7, 0] = 1600
        self.assertEqual(prominent_raw_summits(grid, raw, components=mountain_components(grid)), ())

    def test_raw_metres_are_required_and_alignment_is_strict(self):
        grid = grid_with(np.full((12, 18), .8, dtype=np.float32))
        for raw in (np.zeros(grid.shape), np.full(grid.shape, np.nan), np.ones((11, 18))):
            with self.assertRaises(ValueError):
                prominent_raw_summits(grid, raw, components=mountain_components(grid))

    def test_saved_identity_and_all_other_society_layers_stay_exact(self):
        grid, society = _fixture()
        grid.elevation[:] = .2
        grid.elevation[2:7, 3:10] = 1.0
        raw = np.where(grid.water == 0, 100.0, -5.0)
        raw[4, 6] = 1300
        raw[4, 7] = 2100
        feature = GeographicFeature("peak-01", "peak", "保留山名", 4, 6, 1, "detail")
        plain = GeographicFeature("plain-01", "plain", "保留平原", 3, 2, 1, "secondary")
        society = replace(society, geographic_features=(feature, plain))
        physical_before = grid.elevation.copy()
        refined, report = refine_saved_geographic_features(grid, society, raw_elevation_m=raw)
        self.assertEqual(refined.geographic_features[0], replace(feature, column=7))
        self.assertIs(refined.geographic_features[1], plain)
        for field in ("population", "settlements", "transport", "cultures", "religions", "politics", "provinces"):
            self.assertIs(getattr(refined, field), getattr(society, field))
        np.testing.assert_array_equal(grid.elevation, physical_before)
        self.assertEqual(report["removedFeatures"], [])
        replay, replay_report = refine_saved_geographic_features(grid, refined, raw_elevation_m=raw)
        self.assertEqual(replay.geographic_features, refined.geographic_features)
        self.assertEqual(replay_report["movedFeatures"], [])

    def test_unsupported_saved_peak_is_reported_instead_of_invented(self):
        grid, society = _fixture()
        grid.elevation[:] = .8
        society = replace(society, geographic_features=(
            GeographicFeature("peak-01", "peak", "无证据峰", 4, 6, 1, "detail"),))
        raw = np.where(grid.water == 0, 1000.0, -5.0)
        refined, report = refine_saved_geographic_features(grid, society, raw_elevation_m=raw)
        self.assertEqual(refined.geographic_features, ())
        self.assertEqual(report["removedFeatures"][0]["identifier"], "peak-01")


class MountainLetteringTests(unittest.TestCase):
    def test_neighboring_larger_range_cannot_steal_the_label(self):
        elevation = np.full((40, 80), .2, dtype=np.float32)
        elevation[10:30, 5:9] = .7
        elevation[10:14, 5:30] = .7
        elevation[26:30, 5:30] = .7
        elevation[3:37, 33:74] = 1.0
        grid = grid_with(elevation)
        feature = GeographicFeature("mountain-01", "mountain", "自己的山脉", 20, 6, 1, "major")
        components = mountain_components(grid)
        path = _mountain_label_path(grid, feature, components)
        self.assertIsNotNone(path)
        self.assertTrue(np.all(np.abs(np.diff(path, axis=0)).sum(axis=1) == 1))
        mask = components == components[feature.row, feature.column]
        cells = np.column_stack(np.nonzero(mask))
        support = shapely.union_all([shapely.box(column, row, column+1, row+1) for row, column in cells])
        samples = quadratic_samples(_river_path_data(path))
        self.assertTrue(np.all(shapely.covers(support, shapely.points(samples))))
        self.assertLess(path[:, 0].max(), 30)
        self.assertGreaterEqual(path[:, 0].min(), 5)

    def test_no_support_means_no_fabricated_path(self):
        grid = grid_with(np.full((20, 30), .2, dtype=np.float32))
        feature = GeographicFeature("mountain-01", "mountain", "不存在山脉", 10, 10, 1, "major")
        self.assertIsNone(_mountain_label_path(grid, feature, mountain_components(grid)))

    def test_compact_massif_uses_normal_text_at_its_true_anchor(self):
        grid = grid_with(np.full((20, 30), .2, dtype=np.float32))
        grid.elevation[6:12, 7:13] = .8
        feature = GeographicFeature("mountain-01", "mountain", "紧凑山系名称", 8, 9, 1, "major")
        markup = _toponymy_overlay(grid, SimpleNamespace(geographic_features=(feature,)))
        label = ElementTree.fromstring(markup).find("text")
        self.assertEqual(label.text, feature.name)
        self.assertEqual((label.attrib["x"], label.attrib["y"]), ("9.50", "8.50"))
        self.assertIsNone(label.find("textPath"))
        self.assertNotIn("lengthAdjust", markup)

    def test_extended_range_keeps_natural_glyphs_and_waits_until_its_name_fits(self):
        grid = grid_with(np.full((20, 110), .2, dtype=np.float32))
        grid.elevation[6:14, 5:100] = .8
        feature = GeographicFeature("mountain-01", "mountain", "长风山脉", 9, 40, 1, "major")
        markup = _toponymy_overlay(grid, SimpleNamespace(geographic_features=(feature,)))
        label = ElementTree.fromstring(markup).find("text")
        text_path = label.find("textPath")
        self.assertEqual(text_path.text, feature.name)
        self.assertNotIn("textLength", text_path.attrib)
        self.assertNotIn("lengthAdjust", text_path.attrib)
        self.assertGreater(float(label.attrib["data-min-visible-scale"]), 0)
        self.assertLessEqual(float(label.attrib["data-min-visible-scale"]), 1)


if __name__ == "__main__":
    unittest.main()
