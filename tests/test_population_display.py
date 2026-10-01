"""Delivered numeric paint preserves zero support and absolute density cuts."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import shapely

from scripts import check_scalar_capacity as consumer
from world_atlas.core.render import _population_zone_paths, _partition_overlay_svg_document, _POPULATION_ZONES


class PopulationDisplayTests(unittest.TestCase):
    def check_density(self, density):
        height, width = density.shape
        land = np.ones_like(density, dtype=bool)
        surface = shapely.box(0, 0, width, height)
        paths = _population_zone_paths(density, land, working_surface=surface)
        grid = SimpleNamespace(shape=density.shape, metadata={})
        document = _partition_overlay_svg_document(grid, paths, land_surface=surface, title='density',
            partition_id='population', data_attribute='population-band', zones=_POPULATION_ZONES)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'population.svg'
            filename.write_text(document, encoding='utf-8')
            regions = consumer.export_regions(filename, 'data-population-band', physical_land=surface, dimensions=density.shape)
        metrics, paint = consumer.scalar_metrics('population', density, land, regions)
        self.assertEqual(metrics['mismatchCount'], 0, metrics['examples'])
        self.assertLess(metrics['classOverlapArea'], 1e-12)
        self.assertLess(shapely.symmetric_difference(paint, surface).area, 1e-12)

    def test_isolated_and_bounded_zero_support_are_actually_painted_as_uninhabited(self):
        for size in (1, 3):
            with self.subTest(size=size):
                density = np.full((7, 7), .05, dtype=np.float32)
                density[3-size//2:4+size//2, 3-size//2:4+size//2] = 0
                self.check_density(density)
        self.check_density(np.zeros((4, 8), dtype=np.float32))

    def test_threshold_plateaus_and_float32_neighbours_keep_their_absolute_class(self):
        for value in (.5, np.nextafter(np.float32(1), np.float32(0)),
                      np.nextafter(np.float32(2), np.float32(0))):
            with self.subTest(value=value):
                self.check_density(np.full((4, 8), value, dtype=np.float32))
