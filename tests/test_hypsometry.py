import unittest
from types import SimpleNamespace

import numpy as np
import shapely

from world_atlas.core.hypsometry import (
    ELEVATION_DISPLAY_LEVELS, elevation_display_indices, elevation_display_thresholds,
)
from world_atlas.core.render import (
    _elevation_band_count, _elevation_band_paths, _elevation_palette_for,
    _elevation_paths, _terrain_pixels,
)
from world_atlas.core.cartographic_surface import continuous_land_surface
from world_atlas.core.continuous_terrain import PhysicalTerrainField


def physical_field(raw):
    raw = np.asarray(raw, dtype=float)
    return PhysicalTerrainField(raw, land_mask=raw > 0, sea_level_m=671,
                                elevation_scale_m=1000, elevation_exponent=1.06)


def filled_geometry(paths):
    polygons = []
    for points, codes in paths:
        starts = np.flatnonzero(codes == 1)
        rings = [points[first:last] for first, last in
                 zip(starts, (*starts[1:], len(points)), strict=True)]
        polygons.append(shapely.Polygon(rings[0], rings[1:]))
    return shapely.union_all(polygons)


class HypsometryTests(unittest.TestCase):
    def test_display_covers_lowlands_through_summits_without_modifying_dem(self):
        elevation = np.linspace(0.0, 1.0, 1001, dtype=np.float32).reshape(7, 143)
        original = elevation.copy()
        indices = elevation_display_indices(elevation)
        np.testing.assert_array_equal(elevation, original)
        self.assertEqual(indices.shape, elevation.shape)
        np.testing.assert_array_equal(np.unique(indices), np.arange(ELEVATION_DISPLAY_LEVELS))
        self.assertTrue(np.all(np.diff(indices.ravel().astype(int)) >= 0))
        self.assertEqual(int(indices[0, 0]), 0)
        self.assertEqual(int(indices[-1, -1]), ELEVATION_DISPLAY_LEVELS - 1)

    def test_raster_transition_is_exactly_the_vector_contour_threshold(self):
        thresholds = np.asarray(elevation_display_thresholds())
        np.testing.assert_array_equal(elevation_display_indices(np.nextafter(thresholds, -np.inf)), np.arange(23))
        np.testing.assert_array_equal(elevation_display_indices(thresholds), np.arange(1, 24))
        np.testing.assert_array_equal(elevation_display_indices(np.nextafter(thresholds, np.inf)), np.arange(1, 24))

    def test_lowland_relief_gets_more_intervals_without_random_detail(self):
        lowland = np.linspace(0.0, 0.25, 1000, endpoint=False)
        old_bands = np.floor(lowland * 16)
        new_bands = elevation_display_indices(lowland)
        self.assertGreaterEqual(len(np.unique(new_bands)), 2 * len(np.unique(old_bands)))
        np.testing.assert_array_equal(new_bands, elevation_display_indices(lowland))

    def test_constant_plain_stays_one_band(self):
        self.assertEqual(len(np.unique(elevation_display_indices(np.full((30, 40), 0.2334)))), 1)

    def test_raster_fills_and_contours_use_the_same_physical_field_and_mapping(self):
        relative = np.repeat(np.linspace(.025, 1, 25)[:, None], 3, axis=1)
        raw = 1000 * ((relative - .02) / .98)**(1 / 1.06)
        terrain = physical_field(raw)
        # Simulation palette values deliberately differ from the saved ground.
        grid = SimpleNamespace(shape=raw.shape, elevation=np.full(raw.shape, .2),
            elevation_band=np.zeros(raw.shape, dtype=np.uint8), water=np.zeros(raw.shape, dtype=np.uint8),
            bathymetry_band=np.full(raw.shape, -1, dtype=np.int8), snow=np.zeros(raw.shape, dtype=bool),
            metadata={"elevation": {"levels": 16}, "bathymetry": {"levels": 8},
                      "classification": {"landPalette": ["#000000", "#ffffff"]}})
        before = grid.elevation.copy()
        palette, count = _elevation_palette_for(grid)
        bands = _elevation_band_paths(grid, terrain_field=terrain)
        contours, levels = _elevation_paths(grid, terrain_field=terrain)
        self.assertEqual(count, 24)
        self.assertEqual(_elevation_band_count(grid), 24)
        self.assertEqual(len(bands), 24)
        expected_levels = [value for value in elevation_display_thresholds() if value > .02]
        self.assertEqual(levels, expected_levels)
        self.assertEqual(len(contours), len(expected_levels))
        np.testing.assert_array_equal(_terrain_pixels(grid, terrain_field=terrain, bathymetry=np.zeros(raw.shape)),
                                      palette[elevation_display_indices(terrain.palette_elevation(raw))])
        for level, contour in zip(levels, contours, strict=True):
            band = bands[1 + elevation_display_thresholds().index(level)]
            self.assertAlmostEqual(float(band[0][0][:, 1].min()), float(contour[0, 1]))
        np.testing.assert_array_equal(grid.elevation, before)
        self.assertEqual(grid.metadata["elevation"]["levels"], 16)

    def test_native_coastal_ground_is_not_tapered_and_zero_is_the_same_shore(self):
        raw = np.full((12, 24), -40.)
        raw[2:10, 3:15] = 650
        terrain = physical_field(raw)
        grid = SimpleNamespace(shape=raw.shape, water=np.where(raw > 0, 0, 1).astype(np.uint8),
            elevation=np.where(raw > 0, .65, 0), metadata={"bathymetry": {"levels": 8}})
        before = grid.elevation.copy()
        surface = continuous_land_surface(grid, terrain_field=terrain)
        bands = _elevation_band_paths(grid, terrain_field=terrain)
        contours, levels = _elevation_paths(grid, terrain_field=terrain)
        self.assertEqual(float(terrain.sample_rect([14.5], [5.5])[0, 0]), 650)
        self.assertEqual(float(terrain.sample_rect([8.5], [5.5])[0, 0]), 650)
        self.assertLess(surface.symmetric_difference(filled_geometry(bands[0])).area, 1e-10)
        # A relative boundary below the source .02 floor uses the shared zero
        # silhouette, rather than inventing a second near-coast green rim.
        self.assertLess(surface.symmetric_difference(filled_geometry(bands[1])).area, 1e-10)
        self.assertTrue(contours)
        self.assertTrue(all(level > .02 for level in levels))
        np.testing.assert_array_equal(grid.elevation, before)
        lake_grid = SimpleNamespace(**{**vars(grid), "water": np.where(raw > 0, 0, 2).astype(np.uint8)})
        lake_surface = continuous_land_surface(lake_grid, terrain_field=terrain)
        self.assertTrue(surface.equals_exact(lake_surface, 0))


if __name__ == "__main__":
    unittest.main()
