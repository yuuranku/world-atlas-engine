"""A finer shared ground must preserve saved samples, anchors and topology."""
from types import SimpleNamespace
import unittest

import numpy as np
import shapely

from world_atlas.core.cartographic_surface import continuous_land_surface
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.terrain_refinement import (
    RefinedTerrainField, terrain_from_source, _PATCH_FRACTION, _PATCH_DERIVATIVE,
)


def fixture(raw, *, segments=(), anchors=(), seed=37):
    base = PhysicalTerrainField(raw, land_mask=np.asarray(raw) > 0,
        sea_level_m=671, elevation_scale_m=6000, elevation_exponent=1.06)
    height, width = base.native_m.shape
    flow = np.arange(height*width, dtype=np.int32).reshape(height, width)-1
    flow[:, 0] = -1
    refined = RefinedTerrainField(base, seed=seed, radius_km=60,
        flow_to=flow, discharge=np.full((height, width), 10.),
        river_segments=np.asarray(segments, dtype=float).reshape(-1, 2, 2),
        river_anchors=np.asarray(anchors, dtype=float).reshape(-1, 2))
    return base, refined


class TerrainRefinementTests(unittest.TestCase):
    def test_every_saved_native_height_remains_exact_and_immutable(self):
        y, x = np.indices((24, 48))
        raw = (x - 22.3)*35 + (y - 10)*7
        original = raw.copy()
        base, refined = fixture(raw)
        np.testing.assert_array_equal(refined.sample_points(x+.5, y+.5), raw)
        np.testing.assert_array_equal(refined.sample_rect(np.arange(48)+.5, np.arange(24)+.5), raw)
        np.testing.assert_array_equal(raw, original)
        self.assertIs(refined.native_m, base.native_m)
        self.assertFalse(base.native_m.flags.writeable)

    def test_overlapping_tiles_and_different_query_density_share_the_same_ground(self):
        y, x = np.indices((24, 48))
        _, refined = fixture(600 + x*20 + y*10)
        xx, yy = np.arange(10, 23, .125), np.arange(8, 17, .125)
        whole = refined.sample_rect(xx, yy)
        first = refined.sample_rect(xx[:65], yy)
        second = refined.sample_rect(xx[64:], yy)
        np.testing.assert_array_equal(first[:, -1], second[:, 0])
        np.testing.assert_array_equal(np.column_stack((first[:, :-1], second)), whole)
        np.testing.assert_allclose(refined.sample_points(xx[None, :], yy[:, None]), whole, atol=1e-12)
        np.testing.assert_array_equal(refined.sample_rect(xx[::4], yy[::4]), whole[::4, ::4])

    def test_longitude_is_periodic_and_polar_frame_remains_fixed(self):
        y, x = np.indices((24, 48))
        base, refined = fixture(500*np.sin(x/48*2*np.pi) + y*20)
        yy = np.linspace(0, 24, 151)
        np.testing.assert_allclose(refined.sample_points(np.zeros_like(yy), yy),
                                   refined.sample_points(np.full_like(yy, 48), yy), atol=1e-12)
        xx = np.linspace(0, 48, 101)
        for latitude in (0, 24):
            np.testing.assert_allclose(refined.sample_points(xx, latitude),
                                       base.sample_points(xx, latitude), atol=1e-12)

    def test_river_valley_and_non_native_mouth_anchor_do_not_move(self):
        raw = np.full((24, 48), 850.)
        segments = (((20.5,.5),(20.5,23.5)),)
        mouth = (9.271, 13.119)
        base, refined = fixture(raw, segments=segments, anchors=(mouth,))
        for x, y in ((20.5, 12.13), mouth):
            self.assertEqual(float(refined.sample_points(x, y)), float(base.sample_points(x, y)))

    def test_thin_bridge_water_channel_island_and_lake_keep_native_topology(self):
        land = np.zeros((24, 48), dtype=bool)
        land[4:20, 4:16] = True
        land[4:20, 24:37] = True
        land[11, 16:24] = True
        land[8, 10] = False
        land[4, 43] = True
        base, refined = fixture(np.where(land, 60., -25.))
        grid = SimpleNamespace(water=np.where(land, 0, 1))
        before = continuous_land_surface(grid, terrain_field=base)
        after = continuous_land_surface(grid, terrain_field=refined)
        self.assertTrue(after.is_valid)
        self.assertEqual(len(shapely.get_parts(before)), len(shapely.get_parts(after)))
        self.assertEqual(sum(len(p.interiors) for p in shapely.get_parts(before)),
                         sum(len(p.interiors) for p in shapely.get_parts(after)))
        y, x = np.indices(land.shape)
        np.testing.assert_array_equal(shapely.contains_xy(after, x+.5, y+.5), land)
        self.assertTrue(after.covers(shapely.LineString(((8.5,11.5),(29.5,11.5)))))
        self.assertTrue(after.contains(shapely.Point(43.5,4.5)))

    def test_new_subgrid_heights_exist_without_view_scale_or_density_parameters(self):
        y, x = np.indices((24, 48))
        base, refined = fixture(600 + 35*x + 25*y)
        xx, yy = np.linspace(5, 40, 141), np.linspace(4, 20, 65)
        difference = refined.sample_rect(xx, yy) - base.sample_rect(xx, yy)
        self.assertGreater(float(np.max(np.abs(difference))), 1.)
        _, repeat = fixture(base.native_m)
        np.testing.assert_array_equal(repeat.sample_rect(xx, yy), refined.sample_rect(xx, yy))

    def test_landform_kernel_has_the_declared_bound_and_native_exclusion(self):
        t = np.linspace(0, 1, 1001)
        derivative = 6*t*(1-t*t)**2
        self.assertLessEqual(float(derivative.max()), _PATCH_DERIVATIVE)
        self.assertLess(_PATCH_DERIVATIVE*_PATCH_FRACTION, 1)
        y, x = np.indices((24, 48))
        _, field = fixture(60*np.sin(x*.31)+y*15-130)
        centres = field._landforms.centres
        nearest = np.linalg.norm((centres-.5)-np.round(centres-.5), axis=1)
        self.assertTrue(np.all(field._landforms.radius < nearest))

    def test_public_ground_inverse_solves_actual_patches_without_resampling(self):
        rows, columns = np.indices((24, 48))
        _, field = fixture(60*np.sin(columns*.31)+rows*15-130)
        count = field._landforms.count
        self.assertGreater(count, 0)
        q = field._landforms.centres[count:2*count]
        p = np.column_stack(field.inverse_ground_coordinates(q[:, 0], q[:, 1]))
        reconstructed = np.column_stack(field.forward_ground_coordinates(p[:, 0], p[:, 1]))
        np.testing.assert_allclose(reconstructed, q, rtol=0, atol=2e-14)
        for index, point in enumerate(p):
            scalar = field.forward_ground_coordinates(float(point[0]), float(point[1]))
            np.testing.assert_array_equal(np.asarray(scalar), reconstructed[index],
                                          err_msg="scalar coordinates must apply the same ground deformation")
        split = np.concatenate([np.column_stack(field.inverse_ground_coordinates(
            part[:, 0], part[:, 1])) for part in np.array_split(q, 3)])
        np.testing.assert_array_equal(split, p)
        for period in (-48, 48):
            moved = np.column_stack(field.inverse_ground_coordinates(q[:, 0]+period, q[:, 1]))
            np.testing.assert_allclose(moved, p+[period, 0], rtol=0, atol=2e-14)
        for latitude in (0., 24.):
            x, y = field.inverse_ground_coordinates(np.arange(49.), latitude)
            np.testing.assert_array_equal(x, np.arange(49.))
            np.testing.assert_array_equal(y, np.full(49, latitude))

    def test_invalid_radius_rivers_and_anchors_fail_before_sampling(self):
        base, _ = fixture(np.full((24, 48), 100.))
        for radius, segments, anchors in ((0, np.empty((0,2,2)), np.empty((0,2))),
                                         (60, np.zeros((2,2)), np.empty((0,2))),
                                         (60, np.empty((0,2,2)), [[0,np.inf]])):
            with self.assertRaises(ValueError):
                RefinedTerrainField(base, seed=1, radius_km=radius,
                                    flow_to=np.full(base.native_m.shape, -1, dtype=np.int32),
                                    discharge=np.ones(base.native_m.shape),
                                    river_segments=segments, river_anchors=anchors)

    def test_factory_shares_the_verified_source_datum_and_real_drainage(self):
        y, x = np.indices((24, 48))
        raw = (x-22.3)*35+(y-10)*7
        flow = np.arange(raw.size, dtype=np.int32).reshape(raw.shape)-1
        flow[:, 0] = -1
        grid = SimpleNamespace(shape=raw.shape, water=np.where(raw > 0, 0, 1), flow_to=flow,
            discharge=np.full(raw.shape, 10.), river_order=np.zeros(raw.shape, dtype=np.uint8),
            metadata={"planet":{"radiusKm":60}})
        source = SimpleNamespace(relative_elevation_m=raw, diagnostics={"seed":37,
            "seaLevelMeters":671, "elevationScaleMeters":6000, "elevationExponent":1.06})
        field = terrain_from_source(grid, source)
        self.assertEqual(field.sea_level_m, 671)
        self.assertEqual(field.diagnostics["reliefModel"], "inherited-drainage-valley-profiles")
        np.testing.assert_array_equal(field.sample_points(x+.5, y+.5), raw)
        source.diagnostics.pop("seaLevelMeters")
        with self.assertRaises(KeyError):
            terrain_from_source(grid, source)

    def test_declared_ocean_mouth_has_the_same_ground_anchor_and_flow_edge(self):
        raw = np.tile((11.5-np.arange(24))*30., (12, 1))
        flow = np.full(raw.shape, -1, dtype=np.int32)
        order = np.zeros(raw.shape, dtype=np.uint8)
        order[6, 11] = 2
        grid = SimpleNamespace(shape=raw.shape, water=np.where(raw>0, 0, 1),
            flow_to=flow, river_order=order, discharge=np.ones(raw.shape),
            metadata={"planet":{"radiusKm":6400},
                      "extents":{"west":-180,"east":180,"north":60,"south":-60}})
        source = SimpleNamespace(relative_elevation_m=raw, diagnostics={"seed":37,
            "seaLevelMeters":671, "elevationScaleMeters":6000, "elevationExponent":1.06})
        field = terrain_from_source(grid, source)
        self.assertIsNotNone(field._anchors, "a declared mouth must protect its physical zero datum")
        self.assertIsNotNone(field._rivers, "the mouth belongs to the shared source river network")
        anchor = field._anchors.data[1]
        self.assertLess(abs(float(field.sample_points(anchor[0],anchor[1]))), 1e-7)
        np.testing.assert_array_equal(grid.flow_to, np.full(raw.shape, -1))


if __name__ == "__main__":
    unittest.main()
