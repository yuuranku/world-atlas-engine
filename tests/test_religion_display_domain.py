from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import shapely

from world_atlas.core.render import _coastal_partition_topology, _religion_display_values, _write_map_previews


class ReligionDisplayDomainTests(unittest.TestCase):
    def test_preview_uses_same_neutral_religion_domain_as_the_full_map(self):
        from PIL import Image
        from world_atlas.core.atlas_previews import write_theme_previews
        native=np.array(((-1,1),(1,-1)),dtype=np.int16)
        field=np.zeros(native.shape,dtype=np.uint8)
        grid=SimpleNamespace(water=field)
        thematic=SimpleNamespace(climate=SimpleNamespace(koppen_code=field),biome_zone=field,
            major_basin_rank=field,land_potential_band=field,habitability_band=field)
        society=SimpleNamespace(religions=SimpleNamespace(religion_id=native),
            cultures=SimpleNamespace(language_id=field),politics=SimpleNamespace(state_id=field),
            provinces=SimpleNamespace(province_id=field))
        def save(output,*,water,values,palettes):
            return write_theme_previews(output,water=water,values={'religions':values['religions']},
                                       palettes={'religions':(('none','#ffffff'),('faith','#000000'))})
        with TemporaryDirectory()as temporary, \
             patch('world_atlas.core.render._civilization_display_values',return_value=field), \
             patch('world_atlas.core.render._map_theme_palettes',return_value={}), \
             patch('world_atlas.core.atlas_previews.write_theme_previews',side_effect=save):
            output=Path(temporary)
            Image.new('RGB',(2,2),'#808080').save(output/'terrain.png')
            _write_map_previews(output,grid,thematic,society,field,field)
            self.assertTrue((output/'map-previews/religions.png').exists())
        np.testing.assert_array_equal(native,((-1,1),(1,-1)))

    def test_uninhabitable_land_uses_neutral_religion_coverage_without_changing_model(self):
        native = np.array(((-1, -1, 1, 1), (-1, -1, 1, 1),
                           (-1, -1, 16, 16), (-1, -1, 16, 16)), dtype=np.int16)
        society = SimpleNamespace(religions=SimpleNamespace(religion_id=native))
        land = np.ones(native.shape, dtype=bool)
        frame = shapely.box(0, 0, 4, 4)
        with self.assertRaisesRegex(ValueError, 'category_count'):
            _coastal_partition_topology(native, land, category_count=17, land_surface=frame)
        values = _religion_display_values(society)
        partition = _coastal_partition_topology(values, land, category_count=17, land_surface=frame)
        self.assertEqual(set(partition.visible_labels), {0, 1, 16})
        self.assertLess(shapely.union_all(partition.visible_faces).symmetric_difference(frame).area, 1e-12)
        self.assertTrue(shapely.coverage_is_valid(np.asarray(partition.visible_faces)))
        np.testing.assert_array_equal(native, ((-1, -1, 1, 1), (-1, -1, 1, 1),
                                              (-1, -1, 16, 16), (-1, -1, 16, 16)))

    def test_unrecognized_religion_ids_still_fail_the_renderer(self):
        native = np.array(((0, 17), (-2, 1)), dtype=np.int16)
        society = SimpleNamespace(religions=SimpleNamespace(religion_id=native))
        values = _religion_display_values(society)
        with self.assertRaisesRegex(ValueError, 'category_count'):
            _coastal_partition_topology(values, np.ones(native.shape, dtype=bool),
                category_count=17, land_surface=shapely.box(0, 0, 2, 2))


if __name__ == '__main__':
    unittest.main()
