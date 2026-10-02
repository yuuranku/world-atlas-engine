from types import SimpleNamespace
import unittest

import numpy as np
import shapely

from world_atlas.core.render import _coastal_partition_topology, _religion_display_values


class ReligionDisplayDomainTests(unittest.TestCase):
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
