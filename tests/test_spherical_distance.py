"""Distance acceleration must preserve the spherical D8 metric."""

import heapq
import math
import unittest

import numpy as np

from world_atlas.core.spherical_distance import distance_from_sources


def reference_distance(source, latitude, radius):
    height, width = source.shape
    distance = np.full(source.shape, np.inf)
    distance[source] = 0.0
    queue = [(0.0, int(r), int(c)) for r, c in np.argwhere(source)]
    heapq.heapify(queue)
    row_step = math.pi * radius / height
    while queue:
        value, row, column = heapq.heappop(queue)
        if value != distance[row, column]:
            continue
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            r, c = row + dr, (column + dc) % width
            if not 0 <= r < height:
                continue
            step = math.hypot(row_step * abs(dr), math.tau * radius / width * math.cos(math.radians(latitude[row])) * abs(dc))
            candidate = value + max(step, row_step * 0.035)
            if candidate < distance[r, c]:
                distance[r, c] = candidate
                heapq.heappush(queue, (candidate, r, c))
    return distance


class SphericalDistanceTests(unittest.TestCase):
    def test_accelerated_distance_matches_d8_paths_including_poles_and_seam(self):
        rng = np.random.default_rng(405)
        latitude = np.linspace(89.0, -89.0, 24)
        source = rng.random((24, 48)) < 0.015
        actual = distance_from_sources(source, latitude, 6400.0)
        np.testing.assert_allclose(actual, reference_distance(source, latitude, 6400.0), rtol=1e-13, atol=1e-10)
        self.assertTrue(np.all(actual[source] == 0.0))

    def test_equatorial_seam_is_an_edge_and_poles_do_not_wrap(self):
        source = np.zeros((8, 16), dtype=bool)
        source[3, 0] = True
        distance = distance_from_sources(source, np.zeros(8), 6400.0)
        self.assertEqual(distance[3, 1], distance[3, -1])
        self.assertGreater(distance[-1, 0], distance[0, 0])

    def test_empty_sources_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            distance_from_sources(np.zeros((8, 16), dtype=bool), np.zeros(8), 6400.0)
