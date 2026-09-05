"""Exact spatial topology and seeded choices must survive acceleration."""
from collections import deque
import math
import unittest

import numpy as np

from world_atlas.core.society.spatial import connected_components, select_spaced_seeds
from world_atlas.core.society.transport import RoutingCache, _least_cost_path
from world_atlas.core.society.population import _proximity


def flood_oracle(mask):
    labels = np.zeros(mask.shape, dtype=np.int32)
    sizes = []
    for row, column in np.ndindex(mask.shape):
        if not mask[row, column] or labels[row, column]:
            continue
        identifier = len(sizes) + 1
        queue = deque([(row, column)])
        labels[row, column] = identifier
        size = 0
        while queue:
            r, c = queue.popleft()
            size += 1
            for y, x in ((r-1, c), (r+1, c), (r, (c-1) % mask.shape[1]), (r, (c+1) % mask.shape[1])):
                if 0 <= y < mask.shape[0] and mask[y, x] and not labels[y, x]:
                    labels[y, x] = identifier
                    queue.append((y, x))
        sizes.append(size)
    return labels, tuple(sizes)


class SpatialAccelerationTests(unittest.TestCase):
    def test_distance_transform_matches_iterative_coverage_exactly(self):
        rng = np.random.default_rng(39)
        for shape in ((1, 1), (9, 1), (1, 17), (23, 29)):
            for radius in (0, 1, 7, 34):
                for density in (0., .1, 1.):
                    mask = rng.random(shape) < density
                    frontier = mask.copy()
                    expected = mask.astype(float)
                    for distance in range(1, radius + 1):
                        padded = np.pad(frontier, ((1, 1), (0, 0)))
                        frontier = np.roll(frontier, 1, 1) | np.roll(frontier, -1, 1) | padded[:-2] | padded[2:]
                        expected = np.maximum(expected, frontier * (1.0 - distance / (radius + 1.0)))
                    np.testing.assert_array_equal(_proximity(mask, radius), expected)

    def test_components_preserve_wrapping_connectivity_sizes_and_order(self):
        rng = np.random.default_rng(971)
        for shape in ((1, 1), (1, 19), (19, 1), (17, 31), (31, 17)):
            for fraction in (0., .15, .5, .85, 1.):
                mask = rng.random(shape) < fraction
                expected, sizes = flood_oracle(mask)
                actual, actual_sizes = connected_components(mask)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual_sizes, sizes)
                self.assertFalse(actual.flags.writeable)

    def test_spatial_buckets_preserve_greedy_choices_and_ties(self):
        rng = np.random.default_rng(972)
        score = rng.integers(0, 6, (25, 37)).astype(float)
        valid = rng.random(score.shape) > .12
        for distance in (0., .5, 1., 3.3, 10., 42.):
            expected = []
            candidates = sorted(zip(*np.nonzero(valid)), key=lambda p: (-score[p], p))
            for r, c in candidates:
                if all(math.hypot(r-y, min(abs(c-x), score.shape[1]-abs(c-x))) >= distance for y,x in expected):
                    expected.append((r,c))
                    if len(expected) == 40:
                        break
            self.assertEqual(select_spaced_seeds(score, valid, count=40, minimum_distance=distance), tuple(expected))

    def test_route_cache_revalidates_entire_surface_and_direction(self):
        friction = np.ones((9, 13))
        valid = np.ones(friction.shape, dtype=bool)
        cache = RoutingCache()
        signature = cache.signature(friction, valid, friction)
        first = cache.route(signature, friction, valid, (4, 3), (4, 8), friction)
        self.assertEqual(cache.route(signature, friction, valid, (4, 3), (4, 8), friction), first)
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        valid[4, 5] = False
        changed = cache.signature(friction, valid, friction)
        self.assertNotEqual(changed, signature)
        second = cache.route(changed, friction, valid, (4, 3), (4, 8), friction)
        self.assertNotEqual(first, second)
        self.assertEqual((cache.hits, cache.misses), (1, 2))

    def test_route_diagonals_cannot_cross_blocked_corners(self):
        valid = np.zeros((5, 7), dtype=bool)
        valid[1, 1] = valid[2, 2] = True
        self.assertEqual(_least_cost_path(np.ones(valid.shape), valid, (1,1), (2,2)), ())


if __name__ == "__main__":
    unittest.main()
