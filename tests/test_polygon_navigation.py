"""Exact accepted-domain navigation, including real river-bank polygons."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import shapely

from world_atlas.core.polygon_navigation import PolygonNavigator, polygon_path


FIXTURES = Path(__file__).parent / "fixtures"


class PolygonNavigationTests(unittest.TestCase):
    def test_bank_passages_share_a_mesh_without_changing_exact_paths(self):
        polygon = shapely.Polygon(((0,0),(5,0),(5,5),(0,5)),
                                  holes=[((2,1),(3,1),(3,4),(2,4))])
        passages = (((.5,2.5),(4.5,2.5)), ((4.5,3),(.5,3)))
        expected = [polygon_path(polygon, first, last) for first, last in passages]
        navigator = PolygonNavigator(polygon)
        before = polygon.wkb
        with patch('world_atlas.core.polygon_navigation.shapely.constrained_delaunay_triangles',
                   wraps=shapely.constrained_delaunay_triangles) as triangulate:
            for (first, last), path in zip(passages, expected, strict=True):
                np.testing.assert_array_equal(navigator.path(first, last), path)
            self.assertEqual(triangulate.call_count, 1)
        self.assertEqual(polygon.wkb, before)

    def check_path(self, polygon, first, last):
        before = polygon.wkb
        first, last = np.asarray(first, dtype=float), np.asarray(last, dtype=float)
        original_first, original_last = first.copy(), last.copy()
        points = polygon_path(polygon, first, last)
        line = shapely.LineString(points)
        self.assertTrue(polygon.covers(line))
        self.assertTrue(line.is_simple)
        np.testing.assert_array_equal(points[0], first)
        np.testing.assert_array_equal(points[-1], last)
        np.testing.assert_array_equal(first, original_first)
        np.testing.assert_array_equal(last, original_last)
        self.assertEqual(polygon.wkb, before)
        self.assertTrue(np.all(np.linalg.norm(np.diff(points, axis=0), axis=1) > 0))
        np.testing.assert_array_equal(polygon_path(polygon, last, first), points[::-1])
        return line

    def test_visible_path_keeps_only_original_anchors(self):
        polygon = shapely.box(0, 0, 4, 4)
        points = polygon_path(polygon, (0, 0), (4, 4))
        np.testing.assert_array_equal(points, ((0, 0), (4, 4)))

    def test_concave_turn_uses_actual_inner_corner(self):
        polygon = shapely.Polygon(((0,0),(4,0),(4,1),(1,1),(1,4),(0,4)))
        line = self.check_path(polygon, (.5,3.5), (3.5,.5))
        self.assertEqual(tuple(line.coords[1]), (1.,1.))
        self.assertAlmostEqual(line.length, 2*np.hypot(.5,2.5))

    def test_hole_boundary_and_two_symmetric_corridors(self):
        polygon = shapely.Polygon(((0,0),(5,0),(5,5),(0,5)),
                                  holes=[((2,1),(3,1),(3,4),(2,4))])
        line = self.check_path(polygon, (.5,2.5), (4.5,2.5))
        self.assertFalse(line.relate_pattern(shapely.Polygon(polygon.interiors[0]), "T********"))
        self.assertAlmostEqual(line.length, 1+2*np.hypot(1.5,1.5))
        self.check_path(polygon, (2,2), (3,2))

    def test_multiple_holes_preserve_winding_domain(self):
        polygon = shapely.Polygon(((0,0),(8,0),(8,6),(0,6)),
            holes=[((2,.5),(3,.5),(3,4),(2,4)),((5,2),(6,2),(6,5.5),(5,5.5))])
        self.check_path(polygon, (.5,1), (7.5,5))
        self.check_path(shapely.reverse(polygon), (.5,1), (7.5,5))

    def test_very_narrow_turn_is_not_erased_by_coordinate_tolerance(self):
        width = 1e-10
        polygon = shapely.Polygon(((0,0),(2,0),(2,width),(width,width),(width,2),(0,2)))
        line = self.check_path(polygon, (width/2,1.5), (1.5,width/2))
        self.assertEqual(tuple(line.coords[1]), (width,width))

    def test_outside_anchors_and_invalid_domains_fail(self):
        for polygon, first, last in (
            (shapely.box(0,0,1,1), (-1e-12,.5), (.5,.5)),
            (shapely.Polygon(((0,0),(1,1),(1,0),(0,1))), (.2,.2), (.8,.8)),
            (shapely.MultiPolygon((shapely.box(0,0,1,1),shapely.box(2,0,3,1))),(.5,.5),(2.5,.5)),
            (shapely.Polygon(), (0,0), (1,1)),
            (shapely.box(0,0,1,1), (float("nan"),0), (.5,.5)),
        ):
            with self.subTest(polygon=polygon, first=first):
                with self.assertRaises(ValueError):
                    polygon_path(polygon, first, last)

    def test_same_anchor_is_preserved(self):
        np.testing.assert_array_equal(polygon_path(shapely.box(0,0,1,1),(.5,.5),(.5,.5)),
                                      ((.5,.5),(.5,.5)))

    def test_real_2354_coordinate_bank_uses_exact_domain(self):
        fixture = json.loads((FIXTURES/"polygon-navigation-bank-v97.json").read_text())
        polygon = shapely.geometry.shape(fixture["polygon"])
        self.assertEqual(len(shapely.get_coordinates(polygon)), fixture["coordinates"])
        self.check_path(polygon, fixture["first"], fixture["last"])

    def test_actual_bridge_lane_connectivity_is_not_relaxed(self):
        fixture = json.loads((FIXTURES/"polygon-navigation-road-0069.json").read_text())
        polygons = [part for part in shapely.get_parts(shapely.geometry.shape(fixture["free"]))
                    if part.geom_type == "Polygon"]
        stations = fixture["stations"]
        for first, last in zip(stations[:-2], stations[1:-1]):
            polygon = next(p for p in polygons if p.covers(shapely.MultiPoint((first,last))))
            self.check_path(polygon, first, last)
        first, last = stations[-2:]
        self.assertFalse(any(p.covers(shapely.MultiPoint((first,last))) for p in polygons))
        polygon = next(p for p in polygons if p.covers(shapely.Point(first)))
        with self.assertRaisesRegex(ValueError, "anchors must belong"):
            polygon_path(polygon, first, last)


if __name__ == "__main__":
    unittest.main()
