"""Spawned render stages retain canonical arrays and exact population paths."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest

import numpy as np
import shapely

from world_atlas.core.model import WorldGrid
from world_atlas.core.render import (
    _population_zone_paths,
    _render_grid_payload,
    _render_population_stage,
)


def small_grid():
    shape = (8, 16)
    seasons = (4, *shape)
    return WorldGrid(
        elevation=np.full(shape, .2, dtype=np.float32),
        elevation_band=np.ones(shape, dtype=np.uint8),
        water=np.zeros(shape, dtype=np.uint8),
        bathymetry_band=np.full(shape, -1, dtype=np.int8),
        flow_to=np.full(shape, -1, dtype=np.int32),
        discharge=np.zeros(shape, dtype=np.float32),
        river_order=np.zeros(shape, dtype=np.uint8),
        snow=np.zeros(shape, dtype=bool),
        sea_ice=np.zeros(shape, dtype=bool),
        seasonal_precipitation=np.full(seasons, 100, dtype=np.uint8),
        seasonal_wind_east=np.zeros(seasons, dtype=np.int8),
        seasonal_wind_north=np.zeros(seasons, dtype=np.int8),
        seasonal_river_strength=np.zeros(seasons, dtype=np.uint8),
        metadata={
            "format": "eirenor-world-grid", "schema": "world-grid-v3",
            "inputFingerprint": "render-spawn-fixture",
            "planet": {"radiusKm": 6400.},
            "extents": {"west": -180., "east": 180., "north": 90., "south": -90.},
            "elevation": {"levels": 16}, "bathymetry": {"levels": 8},
            "climate": {
                "seasons": ["vernal", "june", "autumnal", "december"],
                "solarLongitudeDegrees": [0., 90., 180., 270.],
                "precipitationQuantization": "uint8-relative",
                "windQuantization": "int8-relative",
                "riverQuantization": "uint8-relative",
                "model": "synthetic-test",
            },
        },
    )


def spawned_population(output, payload, density):
    grid = WorldGrid(**payload)
    paths = _render_population_stage(Path(output), density, grid.water == 0,
                                    shapely.box(0, 0, grid.shape[1], grid.shape[0]))
    return (grid.content_digest(), isinstance(grid.metadata, MappingProxyType),
            all(not getattr(grid, name).flags.writeable for name in grid._ARRAY_NAMES), paths)


class RenderConcurrencyTests(unittest.TestCase):
    def test_spawn_restores_frozen_grid_and_retains_population_geometry(self):
        grid = small_grid()
        self.assertIsInstance(grid.metadata, MappingProxyType)
        self.assertIsInstance(grid.metadata["climate"], MappingProxyType)
        rows, columns = np.indices(grid.shape)
        densities = ((1. + rows * columns).astype(float),
                     (3. + rows * (16 - columns)).astype(float))
        surface = shapely.box(0, 0, grid.shape[1], grid.shape[0])
        expected = [_population_zone_paths(density, grid.water == 0, working_surface=surface)
                    for density in densities]
        payload = _render_grid_payload(grid)
        with tempfile.TemporaryDirectory() as directory:
            with ProcessPoolExecutor(max_workers=2,
                    mp_context=multiprocessing.get_context("spawn")) as executor:
                futures = [executor.submit(spawned_population, directory, payload, density)
                           for density in densities]
                results = [future.result(timeout=30) for future in futures]
            for (digest, frozen_metadata, frozen_arrays, actual), reference in zip(results, expected, strict=True):
                self.assertEqual(digest, grid.content_digest())
                self.assertTrue(frozen_metadata)
                self.assertTrue(frozen_arrays)
                self.assertEqual(len(actual), len(reference))
                for observed_band, expected_band in zip(actual, reference, strict=True):
                    self.assertEqual(len(observed_band), len(expected_band))
                    for (points, codes), (expected_points, expected_codes) in zip(observed_band, expected_band, strict=True):
                        self.assertEqual(points.tobytes(), expected_points.tobytes())
                        self.assertEqual(codes.tobytes(), expected_codes.tobytes())
            timing = json.loads((Path(directory) / "timing.json").read_text(encoding="utf-8"))
            self.assertEqual(len(timing["stages"]), 2)
            self.assertTrue(all(stage["status"] == "complete" for stage in timing["stages"]))
            self.assertEqual({stage["name"] for stage in timing["stages"]}, {"numeric-population-theme"})


if __name__ == "__main__":
    unittest.main()
