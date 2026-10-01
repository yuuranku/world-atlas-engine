from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from world_atlas.physical.build_source import BuildContext, prepare_physical_fields


class SourceBathymetryTests(unittest.TestCase):
    def test_complete_world_import_preserves_depth_across_map_frame(self):
        height, width = 32, 64
        land = np.zeros((height, width), dtype=bool)
        land[12:20, 24:40] = True
        bathymetry = np.broadcast_to(
            np.repeat([0.2, 0.45, 0.7, 0.95], 8)[:, None], land.shape
        ).copy()
        original_bathymetry = bathymetry.copy()
        elevation = np.zeros(land.shape, dtype=np.float64)
        elevation[land] = 0.2
        surface = SimpleNamespace(
            land_mask=land,
            elevation=elevation,
            bathymetry=bathymetry,
            diagnostics={"polarContinents": {"north": False, "south": False}},
        )
        context = replace(
            BuildContext.reviewed(),
            source_dimensions=(width, height),
            board_dimensions=(width, height),
            grid_dimensions=(width, height),
            padding=(0, 0, 0, 0),
            sampling_step_px=1,
            complete_globe=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "world.png"
            Image.new("RGB", (width, height), context.bathymetry_palette[0]).save(source_path)
            fields = prepare_physical_fields(
                source_path, context=context, procedural_surface=surface
            )

        expected = np.floor(original_bathymetry * context.bathymetry_levels).astype(np.int16)
        expected[land] = -1
        np.testing.assert_array_equal(fields.bathymetry, expected)
        np.testing.assert_array_equal(fields.land_mask, land)
        np.testing.assert_array_equal(fields.ocean_mask, ~land)
        np.testing.assert_array_equal(surface.bathymetry, original_bathymetry)
        self.assertEqual(set(fields.bathymetry[:, 0]), {1, 3, 5, 7})
        self.assertNotIn("board-edge", fields.generation_parameters["bathymetryRule"])


if __name__ == "__main__":
    unittest.main()
