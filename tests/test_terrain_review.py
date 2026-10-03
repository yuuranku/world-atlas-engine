"""The initial publisher consumes the same physical field and SVG codec."""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import shapely

from scripts.check_coastal_coverage import path_geometry
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.procedural_planet import ProceduralSurface, save_surface_bundle
from world_atlas.core.terrain_review import publish_terrain_review
from world_atlas.core.cartographic_contours import CARTOGRAPHIC_CONTOUR_CONTRACT


class TerrainReviewTests(unittest.TestCase):
    def test_published_tiny_peak_uses_true_metres_and_native_map_coordinates(self):
        values = np.full((7, 7), 80.)
        template = PhysicalTerrainField(values, land_mask=values > 0,
            sea_level_m=671, elevation_scale_m=4000, elevation_exponent=1)
        threshold = float(template.contour_height_m(.11))
        values[3, 3] = threshold+.0001
        terrain = PhysicalTerrainField(values, land_mask=values > 0,
            sea_level_m=671, elevation_scale_m=4000, elevation_exponent=1)
        zeros = np.zeros_like(values)
        surface = ProceduralSurface(
            plate_id=zeros, plate_velocity_east_cm_per_year=zeros,
            plate_velocity_north_cm_per_year=zeros, boundary_class=zeros,
            crust_kind=zeros, ocean_age_myr=zeros, continent_id=zeros,
            land_mask=values > 0, elevation=terrain.palette_elevation(values),
            bathymetry=zeros, relative_elevation_m=values,
            diagnostics={"seed":38,"seaLevelMeters":671,"elevationScaleMeters":4000,
                         "elevationExponent":1,"relativeElevationUnits":"model-metres-above-sea-level"})
        terrain = PhysicalTerrainField(surface.relative_elevation_m,
            land_mask=surface.land_mask, sea_level_m=671,
            elevation_scale_m=4000, elevation_exponent=1)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'source', Path(directory)/'review'
            source.mkdir()
            save_surface_bundle(surface, source/'physical-fields.npz')
            Image.new('RGB', (7, 7), '#dde6ce').save(source/'physical-reference.procedural.png')
            provenance = '{"fixture":"actual-physical-bundle"}\n'
            (source/'provenance.json').write_text(provenance, encoding='utf-8')
            record = publish_terrain_review(source, output, version='tiny-field-test')
            document = ET.parse(output/'contours.svg')
            paths = document.findall('.//{http://www.w3.org/2000/svg}path')
            self.assertEqual(record['contourCount'], len(paths))
            self.assertEqual(record['contourModel'],CARTOGRAPHIC_CONTOUR_CONTRACT['schema'])
            self.assertEqual(record['cartographicContract'],CARTOGRAPHIC_CONTOUR_CONTRACT)
            self.assertEqual((output/'terrain.png').read_bytes(),
                             (source/'physical-reference.procedural.png').read_bytes())
            self.assertEqual((output/'provenance.json').read_text(encoding='utf-8'), provenance)
            tiny = next(path for path in paths if abs(float(path.attrib['data-elevation-level'])-.11) < 1e-12)
            geometry = path_geometry(tiny.attrib['d'])
            self.assertTrue(geometry.is_valid)
            self.assertTrue(geometry.contains(shapely.Point(3.5,3.5)))
            self.assertFalse(geometry.contains(shapely.Point(4.,4.)),
                             'the source model already owns native n+.5 coordinates')
            self.assertGreaterEqual(len(geometry.exterior.coords),5)
            points = np.asarray(geometry.exterior.coords)
            self.assertLess(float(abs(terrain.sample_points(points[:,0],points[:,1])-threshold).max()),3e-8)
            self.assertEqual(record['contourBytes'],(output/'contours.svg').stat().st_size)


if __name__ == '__main__':
    unittest.main()
