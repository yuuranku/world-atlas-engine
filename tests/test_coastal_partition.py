import unittest
import numpy as np
import shapely

from world_atlas.core.coastal_partition import (
    extend_coastal_partition, clip_partition_to_surface,
    enforce_homogeneous_components,
)


class CoastalPartitionTests(unittest.TestCase):
    def test_coastal_extension_keeps_original_land_labels_including_frontiers(self):
        land = np.zeros((6, 8), dtype=bool)
        land[1:5, 2:6] = True
        labels = np.full(land.shape, -1, dtype=np.int32)
        labels[land] = 3
        labels[2:4, 4:6] = 0
        before = labels.copy()
        display, active = extend_coastal_partition(labels, land, margin=2, wrap_longitude=False)
        np.testing.assert_array_equal(display[land], labels[land])
        np.testing.assert_array_equal(labels, before)
        self.assertTrue(np.all(active[land]))
        self.assertEqual(display[2, 6], 0)
        self.assertEqual(display[2, 1], 3)

    def test_periodic_coastal_extension_uses_the_same_land_across_the_meridian(self):
        land = np.zeros((5, 10), dtype=bool)
        land[2, 0] = True
        labels = np.zeros(land.shape, dtype=np.int32)
        labels[2, 0] = 8
        display, active = extend_coastal_partition(labels, land, margin=2, wrap_longitude=True)
        self.assertTrue(active[2, -1])
        self.assertEqual(display[2, -1], 8)

    def test_partition_clipping_covers_curved_land_and_keeps_lakes_clear(self):
        outer = shapely.Point(5, 5).buffer(4)
        lake = shapely.Point(6, 5).buffer(.8)
        land = outer.difference(lake)
        faces = (shapely.box(0, 0, 5, 10), shapely.box(5, 0, 10, 10))
        clipped, labels = clip_partition_to_surface(faces, (1, 2), land)
        union = shapely.union_all(clipped)
        self.assertLess(land.symmetric_difference(union).area, 1e-10)
        self.assertLess(union.intersection(lake).area, 1e-10)
        self.assertEqual(set(labels), {1, 2})
        self.assertLess(clipped[0].intersection(clipped[1]).area, 1e-10)

    def test_a_shrunk_thematic_coast_is_rejected_even_when_nothing_spills_into_water(self):
        land = shapely.box(0, 0, 10, 5)
        shortened = (shapely.box(0, 0, 5, 4.5), shapely.box(5, 0, 10, 4.5))
        with self.assertRaisesRegex(ValueError, "uncovered"):
            clip_partition_to_surface(shortened, (1, 2), land)

    def test_foreign_mainland_category_cannot_split_a_native_single_label_island(self):
        mainland = shapely.box(0, 0, 3, 3)
        island = shapely.Point(4.5, 1.5).buffer(.7)
        surface = shapely.union_all((mainland, island))
        land = np.zeros((4, 6), dtype=bool)
        land[:3, :3] = True
        land[1, 4] = True
        values = np.full(land.shape, -1, dtype=np.int32)
        values[:3, :2] = 2
        values[:3, 2] = 3
        values[1, 4] = 8
        before = values.copy()
        # The smoothed working boundary erroneously continues across water,
        # dividing the otherwise single-province island in two.
        faces = (mainland.intersection(shapely.box(0, 0, 2, 3)),
                 mainland.intersection(shapely.box(2, 0, 3, 3)),
                 island.intersection(shapely.box(3, 0, 4.5, 3)),
                 island.intersection(shapely.box(4.5, 0, 6, 3)))
        result, labels = enforce_homogeneous_components(faces, (2, 3, 3, 8), values, land, surface)
        island_color = shapely.union_all([face for face, label in zip(result, labels, strict=True) if label == 8])
        self.assertLess(island.symmetric_difference(island_color).area, 1e-10)
        self.assertLess(surface.symmetric_difference(shapely.union_all(result)).area, 1e-10)
        self.assertEqual(set(labels), {2, 3, 8})
        self.assertAlmostEqual(sum(face.area for face in result), surface.area)
        np.testing.assert_array_equal(values, before)

    def test_uniform_continent_has_no_size_exception_and_lake_hole_stays_open(self):
        lake = shapely.box(4, 4, 6, 6)
        surface = shapely.box(0, 0, 10, 10).difference(lake)
        land = np.ones((10, 10), dtype=bool)
        land[4:6, 4:6] = False
        values = np.full(land.shape, 0, dtype=np.int32)
        values[~land] = 99
        faces = (surface.intersection(shapely.box(0, 0, 5, 10)),
                 surface.intersection(shapely.box(5, 0, 10, 10)))
        result, labels = enforce_homogeneous_components(faces, (0, 99), values, land, surface)
        self.assertEqual(set(labels), {0})
        self.assertLess(surface.symmetric_difference(shapely.union_all(result)).area, 1e-10)
        self.assertEqual(shapely.union_all(result).intersection(lake).area, 0)

    def test_component_without_native_centers_retains_its_existing_partition(self):
        fragment = shapely.box(2.8, .8, 3.2, 1.2)
        main = shapely.box(0, 0, 2, 2)
        surface = shapely.union_all((main, fragment))
        land = np.zeros((3, 5), dtype=bool)
        land[:2, :2] = True
        values = np.full(land.shape, 6, dtype=np.int32)
        faces = (main, fragment.intersection(shapely.box(2, 0, 3, 2)),
                 fragment.intersection(shapely.box(3, 0, 4, 2)))
        result, labels = enforce_homogeneous_components(faces, (6, 7, 8), values, land, surface)
        self.assertEqual(set(labels), {6, 7, 8})
        self.assertLess(surface.symmetric_difference(shapely.union_all(result)).area, 1e-10)

    def test_other_land_inside_a_component_bbox_does_not_change_its_native_labels(self):
        # The island in a lake lies inside the surrounding component's bbox,
        # but its native label must not make that surrounding land mixed.
        ring = shapely.box(0, 0, 7, 7).difference(shapely.box(2, 2, 5, 5))
        island = shapely.box(3, 3, 4, 4)
        surface = shapely.union_all((ring, island))
        land = np.ones((7, 7), dtype=bool)
        land[2:5, 2:5] = False
        land[3, 3] = True
        values = np.full(land.shape, 5, dtype=np.int32)
        values[3, 3] = 9
        faces = (ring.intersection(shapely.box(0, 0, 3.5, 7)),
                 ring.intersection(shapely.box(3.5, 0, 7, 7)), island)
        result, labels = enforce_homogeneous_components(faces, (5, 9, 9), values, land, surface)
        rings = shapely.union_all([face for face, label in zip(result, labels, strict=True) if label == 5])
        self.assertLess(ring.symmetric_difference(rings).area, 1e-10)
        self.assertEqual(set(labels), {5, 9})


if __name__ == "__main__":
    unittest.main()
