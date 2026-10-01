"""Nominal outlines dissolve actual faces and retain actual local ownership."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import shapely

from test_world_identity import _society
from world_atlas.core.governance_render import write_governance_overlay
from world_atlas.core.society.model import GovernmentForm, TransportRoute


class GovernanceRenderingTests(unittest.TestCase):
    def test_nominal_union_retains_actual_faces_and_frontier_hole(self):
        society = _society()
        states = tuple(replace(state, population_min=8_000_000 if state.identifier == 1 else 30_000,
                               population_max=10_000_000 if state.identifier == 1 else 50_000)
                       for state in society.politics.states)
        society = replace(society, politics=replace(society.politics, states=states,
            government_forms=(GovernmentForm(1, 'bureaucratic-monarchy', '官僚君主制', 'fixture'),)),
            transport=replace(society.transport, routes=(TransportRoute('recorded-link', 'road', 'trunk',
                'east-port', 'west-river', ((1.5, 2.5), (5.5, 2.5))),)))
        grid = NS(shape=(6, 8), water=np.zeros((6, 8)), elevation=np.zeros((6, 8)), snow=np.zeros((6, 8)),
                  metadata={'worldProfile': {'technologyEra': 'ancient'}, 'planet': {'radiusKm': 6400},
                            'extents': {'west': -.1, 'east': .1, 'north': .1, 'south': -.1}},
                  content_digest=lambda: 'fixture-grid')
        hole = shapely.box(1, 1, 2, 2)
        first = shapely.difference(shapely.box(0, 0, 4, 6), hole)
        second = shapely.box(4, 0, 8, 6)
        before = society.politics.state_id.copy()
        with tempfile.TemporaryDirectory() as directory:
            governance, report, features = write_governance_overlay(Path(directory), grid, society, (first, second), [1, 2])
            self.assertEqual(report['compoundRealms'], 1)
            self.assertAlmostEqual(report['maximumAreaDriftNativeSquared'], 0)
            root = ET.parse(Path(directory)/'governance-overlay.svg').getroot()
            path = root.find('.//{http://www.w3.org/2000/svg}path')
            self.assertEqual(path.attrib['data-local-countries'], '2')
            self.assertEqual(path.attrib['d'].count('Z'), 2)  # Exterior plus the actual frontier hole.
            self.assertEqual(path.attrib['vector-effect'], 'non-scaling-stroke')
            self.assertEqual(json.loads((Path(directory)/'governance.json').read_text(encoding='utf-8'))['gridDigest'], 'fixture-grid')
            self.assertEqual(governance['relations'][0]['supportingRouteIds'], ['recorded-link'])
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0].layer, 'nominal-realms')
            self.assertTrue(shapely.equals(features[0].geometry, shapely.union_all([first, second]).boundary))
        np.testing.assert_array_equal(before, society.politics.state_id)


if __name__ == '__main__':
    unittest.main()
