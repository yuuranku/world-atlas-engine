from pathlib import Path
import tempfile
import unittest

import numpy as np

from world_atlas.core.procedural_planet import (
    ProceduralSurface, load_surface_bundle, save_surface_bundle,
)


def physical_surface():
    values = np.array([[-120, -20, 4, 850], [-300, 3, 20, -40]], dtype=np.float32)
    zeros = np.zeros(values.shape)
    return ProceduralSurface(
        plate_id=zeros, plate_velocity_east_cm_per_year=zeros,
        plate_velocity_north_cm_per_year=zeros, boundary_class=zeros,
        crust_kind=zeros, ocean_age_myr=zeros, continent_id=zeros,
        land_mask=values > 0, elevation=np.maximum(values, 0) / 1000,
        bathymetry=np.maximum(-values, 0) / 1000,
        relative_elevation_m=values,
        diagnostics={"seaLevelMeters": 671, "elevationScaleMeters": 1000,
                     "elevationExponent": 1.06,
                     "relativeElevationUnits": "model-metres-above-sea-level"},
    )


class PhysicalSurfaceBundleTests(unittest.TestCase):
    def test_raw_ground_and_palette_mapping_survive_bundle_roundtrip(self):
        original = physical_surface()
        with tempfile.TemporaryDirectory() as directory:
            path = save_surface_bundle(original, Path(directory) / 'surface.npz')
            with np.load(path) as archive:
                self.assertIn('relative_elevation_m', archive.files)
                self.assertNotIn('signed_height', archive.files)
                np.testing.assert_array_equal(archive['relative_elevation_m'], original.relative_elevation_m)
            loaded = load_surface_bundle(path)
        np.testing.assert_array_equal(loaded.relative_elevation_m, original.relative_elevation_m)
        self.assertEqual(loaded.diagnostics['seaLevelMeters'], 671)
        self.assertEqual(loaded.diagnostics['elevationScaleMeters'], 1000)
        self.assertFalse(loaded.relative_elevation_m.flags.writeable)

    def test_palette_glued_old_bundle_is_rejected_without_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.npz'
            np.savez(path, signed_height=np.array([[.03, -.08]]))
            with self.assertRaisesRegex(ValueError, 'current schema'):
                load_surface_bundle(path)


if __name__ == '__main__':
    unittest.main()
