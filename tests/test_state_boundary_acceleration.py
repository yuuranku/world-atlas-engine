"""Golden decision sequence captured from the published v1.1.0 algorithm."""
import hashlib
import unittest

import numpy as np

from world_atlas.core.society.state_formation import (
    ControlRegionEdge, ControlRegionGraph, settle_state_boundaries,
)


def cases():
    rng = np.random.default_rng(340971)
    for case in range(50):
        count = 18
        pairs = {(i, i + 1) for i in range(1, count)} if case % 2 else set()
        for _ in range(40):
            a, b = sorted(int(x) for x in rng.choice(np.arange(1, count + 1), 2, replace=False))
            pairs.add((a, b))
        edges = tuple(ControlRegionEdge(a, b, 3, 1., float(rng.integers(0, 9)), float(rng.random()), 0., float(rng.random())) for a, b in sorted(pairs))
        domain = np.ones(count + 1, np.int32)
        domain[0] = 0
        graph = ControlRegionGraph(domain, np.ones(count + 1), domain, edges)
        owners = rng.integers(1, 4, count + 1, dtype=np.int32)
        owners[:4] = [0, 1, 2, 3]
        yield graph, owners, np.arange(4, dtype=np.int32)


def decision_digest(function):
    digest = hashlib.sha256()
    for graph, owners, cores in cases():
        digest.update(function(graph, owners, core_region_by_state=cores).astype('<i4').tobytes())
    return digest.hexdigest()


class StateBoundaryAccelerationTests(unittest.TestCase):
    def test_golden_connected_and_disconnected_graph_decisions(self):
        self.assertEqual(decision_digest(settle_state_boundaries), 'c7f059cbeba478bbf00ec0836c0802ec2baffbebc1d70a6042cd1ba880673b26')
