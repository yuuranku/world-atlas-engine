"""The society renderer reuses the exact layers bound to its physical inputs."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import shapely

from test_render_concurrency import small_grid
from world_atlas.core.ecological_sources import derive_ecological_sources, prepare_world_layers
from world_atlas.core.procedural_planet import ProceduralSurface
from world_atlas.core.render import _render_review
from world_atlas.core.thematic import derive_thematic_layers


def source_for(grid):
    values = np.zeros(grid.shape, dtype=np.float32)
    return ProceduralSurface(plate_id=values, plate_velocity_east_cm_per_year=values,
        plate_velocity_north_cm_per_year=values, boundary_class=values, crust_kind=values,
        ocean_age_myr=values, continent_id=values, land_mask=grid.water == 0,
        elevation=np.full(grid.shape, .2), bathymetry=values,
        relative_elevation_m=np.full(grid.shape, 200.),
        diagnostics={'seed':17,'seaLevelMeters':0.,'elevationScaleMeters':10000.,
                     'elevationExponent':1.})


class WorldPreparationTests(unittest.TestCase):
    def test_preparation_retains_the_same_ecological_and_thematic_values(self):
        grid = small_grid()
        source = source_for(grid)
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_world_layers(grid, source, timing_output=Path(directory))
            expected_sources = derive_ecological_sources(grid, source)
            expected = derive_thematic_layers(grid, ecological_sources=expected_sources)
            prepared.validate(grid, source)
            self.assertEqual(prepared.ecological_sources.river_orders, expected_sources.river_orders)
            self.assertTrue(prepared.ecological_sources.lake_geometry.equals(expected_sources.lake_geometry))
            np.testing.assert_array_equal(prepared.thematic.habitability, expected.habitability)
            np.testing.assert_array_equal(prepared.thematic.land_potential, expected.land_potential)
            timing = json.loads((Path(directory)/'timing.json').read_text())
            self.assertEqual([stage['name'] for stage in timing['stages']],
                ['physical-ground','physical-shoreline','ecological-sources','thematic-layers'])

    def test_all_physical_arrays_and_diagnostics_are_bound(self):
        grid = small_grid()
        source = source_for(grid)
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_world_layers(grid, source, timing_output=Path(directory))
            changed = replace(source, bathymetry=np.ones(grid.shape))
            with self.assertRaisesRegex(ValueError, 'physical inputs'):
                prepared.validate(grid, changed)
            changed = replace(source, diagnostics={**source.diagnostics,'seed':18})
            with self.assertRaisesRegex(ValueError, 'physical inputs'):
                prepared.validate(grid, changed)
            changed_grid = replace(grid, discharge=np.ones(grid.shape,dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'physical inputs'):
                prepared.validate(changed_grid, source)

    def test_render_uses_the_same_prepared_shore_without_starting_another_factory(self):
        class MarkupReached(Exception):
            pass
        grid = small_grid()
        source = source_for(grid)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = prepare_world_layers(grid, source, timing_output=root)
            def shoreline_markup(surface):
                self.assertIs(surface, prepared.land_surface)
                raise MarkupReached()
            with patch('world_atlas.core.render.prepare_world_layers', side_effect=AssertionError('duplicate factory')), \
                    patch('world_atlas.core.render._geometry_filled_paths', side_effect=shoreline_markup):
                with self.assertRaises(MarkupReached):
                    _render_review(grid, root/'review', physical_source=source, society=None,
                                   travel_capabilities=(), executor=None, prepared_layers=prepared)
            record = json.loads((root/'review/preparation-reuse.json').read_text())
            self.assertEqual(record['gridDigest'], grid.content_digest())
            self.assertEqual(Path(record['sourceTiming']), (root/'timing.json').resolve())
            self.assertEqual(record['reusedStages'],
                ['physical-ground','physical-shoreline','ecological-sources','thematic-layers'])

