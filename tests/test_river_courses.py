"""The resolved course, terrain bed and logical graph remain one river."""
from types import SimpleNamespace
from pathlib import Path
import json
import unittest

import numpy as np
import shapely

from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.river_courses import solve_river_courses, RiverBedRelief, native_river_paths, _dry_course_spans
from world_atlas.core.terrain_refinement import terrain_from_source


def fixture(raw):
    raw = np.asarray(raw, dtype=float)
    base = PhysicalTerrainField(raw, land_mask=raw > 0, sea_level_m=0,
                                elevation_scale_m=10000., elevation_exponent=1.)
    grid = SimpleNamespace(shape=raw.shape, water=np.where(raw > 0, 0, 1),
        river_order=np.zeros(raw.shape, dtype=np.uint8),
        flow_to=np.full(raw.shape, -1, dtype=np.int32),
        discharge=np.full(raw.shape, 100.),
        metadata={"planet":{"radiusKm":6400.},
                  "extents":{"west":-180.,"east":180.,"north":90.,"south":-90.}})
    return grid, base


class RiverCourseTests(unittest.TestCase):
    def test_straight_valley_and_flat_ground_do_not_acquire_waves(self):
        y, x = np.indices((32, 32), dtype=float)+.5
        source = np.array(((16.5, 3.5), (16.5, 27.5)))
        for raw in (np.full((32, 32), 100.), 1000.-10*y+50*(x-16.5)**2):
            grid, base = fixture(raw)
            result = solve_river_courses(grid, base, source_paths=(source,))
            np.testing.assert_allclose(result.paths[0][:, 0], 16.5, atol=1e-12)
            self.assertTrue(np.all(np.diff(result.beds[0]) <= 0))
            self.assertEqual(result.diagnostics["uphillBedSteps"], 0)

    def test_curved_continuous_valley_guides_a_straight_logical_reach(self):
        y, x = np.indices((48, 48), dtype=float)+.5
        valley = 22.5+.8*np.sin(np.pi*(y-5.5)/32.)
        grid, base = fixture(1800.-10*y+120*(x-valley)**2)
        source = np.array(((22.5, 5.5), (22.5, 37.5)))
        result = solve_river_courses(grid, base, source_paths=(source,))
        path = result.paths[0]
        self.assertGreater(float(path[:, 0].max()-22.5), .3)
        self.assertTrue(shapely.LineString(path).is_simple)
        np.testing.assert_array_equal(path[[0, -1]], source)
        old = base.sample_points(np.full(len(path), 22.5), path[:, 1])
        new = base.sample_points(path[:, 0], path[:, 1])
        self.assertLess(float(new.mean()), float(old.mean()))
        repeated = solve_river_courses(grid, base, source_paths=(source,))
        np.testing.assert_array_equal(path, repeated.paths[0])

    def test_uphill_filled_depression_gets_auditable_descending_bed(self):
        y, x = np.indices((32, 32), dtype=float)+.5
        grid, base = fixture(100.+12*y+20*(x-16.5)**2)
        source = np.array(((16.5, 3.5), (16.5, 27.5)))
        result = solve_river_courses(grid, base, source_paths=(source,))
        self.assertGreater(result.diagnostics["uphillSourceSteps"], 0)
        self.assertGreater(result.diagnostics["maximumIncisionMeters"], 200.)
        self.assertTrue(np.all(result.beds[0] > 0))
        self.assertTrue(np.all(np.diff(result.beds[0]) <= 0))
        bed = RiverBedRelief(result, grid, radius_km=6400.)
        path = result.paths[0]
        ground = base.sample_points(path[:, 0], path[:, 1])
        actual = bed.sample(path[:, 0], path[:, 1], ground)
        np.testing.assert_allclose(actual, result.beds[0], atol=1e-10)
        np.testing.assert_array_equal(base.native_m, 100.+12*y+20*(x-16.5)**2)

    def test_confluence_has_one_position_and_one_bed_height(self):
        y, x = np.indices((32, 32), dtype=float)+.5
        grid, base = fixture(400.+8*y+2*x)
        junction = (16.5, 16.5)
        sources = (np.array(((5.5, 5.5), junction)),
                   np.array(((27.5, 5.5), junction)),
                   np.array((junction, (16.5, 27.5))))
        result = solve_river_courses(grid, base, source_paths=sources)
        self.assertEqual(result.beds[0][-1], result.beds[1][-1])
        self.assertEqual(result.beds[0][-1], result.beds[2][0])
        for first in range(3):
            for second in range(first+1, 3):
                self.assertTrue(shapely.LineString(result.paths[first]).intersection(
                    shapely.LineString(result.paths[second])).equals(shapely.Point(junction)))
        relief = RiverBedRelief(result, grid, radius_km=6400.)
        for path, bed in zip(result.paths, result.beds, strict=True):
            heights = relief.sample(path[:, 0], path[:, 1],
                                   base.sample_points(path[:, 0], path[:, 1]))
            np.testing.assert_allclose(heights, bed, atol=1e-10)

    def test_existing_fractional_bridge_portal_is_inserted_and_fixed(self):
        y, x = np.indices((48, 48), dtype=float)+.5
        valley = 22.5+.8*np.sin(np.pi*(y-5.5)/32.)
        grid, base = fixture(1800.-10*y+120*(x-valley)**2)
        source = np.array(((22.5, 5.5), (22.5, 37.5)))
        portal = np.array((22.5, 19.271))
        result = solve_river_courses(grid, base, source_paths=(source,), locked_points=(portal,))
        self.assertTrue(np.any(np.all(result.paths[0] == portal, axis=1)))
        self.assertGreater(float(result.paths[0][:, 0].max()-22.5), .1)

    def test_old_geometric_crossing_ties_beds_without_changing_native_flow(self):
        y, x = np.indices((32, 32), dtype=float)+.5
        grid, base = fixture(500.+10*y+3*x)
        before = grid.flow_to.copy()
        sources = (np.array(((8.5, 8.5), (22.5, 22.5))),
                   np.array(((22.5, 8.5), (8.5, 22.5))))
        result = solve_river_courses(grid, base, source_paths=sources)
        self.assertEqual(result.diagnostics["sourceGeometricCrossings"], 1)
        relief = RiverBedRelief(result, grid, radius_km=60.)
        for path, bed in zip(result.paths, result.beds, strict=True):
            actual = relief.sample(path[:, 0], path[:, 1],
                                   base.sample_points(path[:, 0], path[:, 1]))
            np.testing.assert_allclose(actual, bed, atol=1e-10)
            self.assertTrue(np.all(np.diff(actual) <= 1e-10))
        np.testing.assert_array_equal(grid.flow_to, before)

    def test_actual_shore_cove_between_positive_stations_gets_a_dry_course(self):
        data = json.loads((Path(__file__).parent/'fixtures/river-subcell-cove-dev20.json').read_text())
        _, base = fixture(data['ground'])
        source = np.asarray(data['points'])
        t = np.linspace(0., 1., 65)
        original = source[0]+(source[1]-source[0])*t[:, None]
        self.assertLess(float(base.sample_points(original[:, 0], original[:, 1]).min()), 0.)
        result = _dry_course_spans(base, source.copy(), source.copy(), np.full(2, .4))
        dense = (result[:-1, None]+np.diff(result, axis=0)[:, None]*t[None, :, None]).reshape(-1, 2)
        self.assertGreater(float(base.sample_points(dense[:, 0], dense[:, 1]).min()), 0.)
        np.testing.assert_array_equal(result[[0, -1]], source)

    def test_between_station_ground_trough_cannot_create_an_uphill_bed(self):
        y, x = np.indices((32, 32), dtype=float)+.5
        grid, base = fixture(200.+20*(y-16.5)**2+20*(x-16.5)**2)
        source = np.array(((16.5, 3.7), (16.5, 27.31)))
        courses = solve_river_courses(grid, base, source_paths=(source,))
        relief = RiverBedRelief(courses, grid, radius_km=6400.)
        path = courses.paths[0]
        t = np.linspace(0., 1., 17)
        dense = (path[:-1, None]+np.diff(path, axis=0)[:, None]*t[None, :, None]).reshape(-1, 2)
        heights = relief.sample(dense[:, 0], dense[:, 1], base.sample_points(dense[:, 0], dense[:, 1]))
        self.assertTrue(np.all(np.diff(heights) <= 1e-7))

    def test_factory_only_changes_native_centres_in_the_bed_corridor(self):
        y, x = np.indices((32, 32), dtype=float)+.5
        raw = 300.+10*y+20*(x-16.5)**2
        grid, _ = fixture(raw)
        grid.river_order[3:28, 16] = 2
        for row in range(3, 27):
            grid.flow_to[row, 16] = (row+1)*32+16
        source = SimpleNamespace(relative_elevation_m=raw,
            diagnostics={"seed":17,"seaLevelMeters":0.,
                         "elevationScaleMeters":10000.,"elevationExponent":1.})
        old_flow = grid.flow_to.copy()
        field = terrain_from_source(grid, source)
        actual = field.sample_points(x, y)
        self.assertLess(actual[20, 16], raw[20, 16])
        np.testing.assert_array_equal(actual[:, :15], raw[:, :15])
        np.testing.assert_array_equal(actual[:, 18:], raw[:, 18:])
        np.testing.assert_array_equal(grid.flow_to, old_flow)
        np.testing.assert_array_equal(field.native_m, raw)
        path = field.river_paths[0]
        heights = field.sample_points(path[:, 0], path[:, 1])
        np.testing.assert_allclose(heights, field.river_bed_profiles[0], atol=1e-10)

    def test_mouth_remains_on_the_original_shore_and_land_stays_positive(self):
        y, x = np.indices((32, 32), dtype=float)+.5
        grid, base = fixture(20.*(27.2-y)+10*(x-16.5)**2)
        grid.river_order[3:27, 16] = 2
        for row in range(3, 26):
            grid.flow_to[row, 16] = (row+1)*32+16
        grid.flow_to[26, 16] = 27*32+16
        sources = native_river_paths(grid, base)
        mouth = sources[0][-1]
        self.assertLess(abs(float(base.sample_points(*mouth))), 1e-7)
        result = solve_river_courses(grid, base, source_paths=sources)
        np.testing.assert_array_equal(result.paths[0][-1], mouth)
        relief = RiverBedRelief(result, grid, radius_km=6400.)
        xx, yy = np.meshgrid(np.linspace(16.1, 16.9, 35), np.linspace(4., 28., 101))
        original = base.sample_points(xx, yy)
        revised = relief.sample(xx, yy, original)
        np.testing.assert_array_equal(revised > 0, original > 0)


if __name__ == "__main__":
    unittest.main()
