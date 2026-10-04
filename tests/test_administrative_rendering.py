import unittest
from types import SimpleNamespace

import numpy as np
import shapely

from world_atlas.core.render import (
    _administrative_boundary_paths,
    _administrative_overview_features,
    _partition_boundary_overlay,
    _territorial_quality_metrics,
)
from world_atlas.core.cartographic_features import line_features
from world_atlas.core.cartographic_tiles import TileFeature


def _edge_set(paths):
    edges = set()
    for path in paths:
        points = np.asarray(path, dtype=np.float64)
        for start, end in zip(points[:-1], points[1:], strict=True):
            first = tuple(float(value) for value in start)
            second = tuple(float(value) for value in end)
            edges.add((first, second) if first < second else (second, first))
    return edges


class AdministrativeRenderingTests(unittest.TestCase):
    def test_overview_fills_and_borders_share_one_atomic_province_summary(self):
        faces = (
            shapely.Polygon(((0, 0), (2, 0), (2, 1.75), (2.2, 2), (0, 2))),
            shapely.Polygon(((0, 2), (2.2, 2), (2, 2.25), (2, 4), (0, 4))),
            shapely.Polygon(((2, 0), (4, 0), (4, 4), (2, 4),
                             (2, 2.25), (2.2, 2), (2, 1.75))),
        )
        labels = np.asarray((1, 2, 3), dtype=np.int32)
        parents = np.asarray((0, 1, 1, 2), dtype=np.int32)
        _, state_paths, province_paths = _administrative_boundary_paths(faces, labels, parents)
        features = []
        for identifier in (1, 2):
            geometry = shapely.union_all([face for face, label in zip(faces, labels, strict=True)
                                          if parents[label] == identifier])
            features.append(TileFeature(geometry, {'data-state': str(identifier)}, 'theme-fill',
                section='theme', theme='political'))
        for face, identifier in zip(faces, labels, strict=True):
            features.append(TileFeature(face, {'data-province': str(identifier)}, 'theme-fill',
                section='theme', theme='provinces'))
        for layer, paths in (('state-boundaries', state_paths), ('province-boundaries', province_paths)):
            features.extend(line_features(paths, layer, '#456', 1, clip='land'))
        # A coast cutting through the shared border must clip both maps and
        # their ink at the same physical point.
        land = shapely.box(.2, .3, 3.8, 3.7)
        summary = _administrative_overview_features(features, faces, labels, parents)
        states = {int(feature.attributes['data-state']): shapely.intersection(feature.geometry, land)
                  for feature in summary if feature.theme == 'political'}
        provinces = {int(feature.attributes['data-province']): shapely.intersection(feature.geometry, land)
                     for feature in summary if feature.theme == 'provinces'}
        ink = {feature.layer: shapely.intersection(feature.geometry, land)
               for feature in summary if feature.section == 'ink'}
        self.assertTrue(all(feature.attributes.get('clip') == 'land'
                            for feature in summary if feature.section == 'ink'))
        self.assertTrue(shapely.equals(states[1], shapely.union_all([provinces[1], provinces[2]])))
        self.assertTrue(shapely.equals(states[2], provinces[3]))
        self.assertTrue(shapely.equals(ink['state-boundaries'],
            shapely.intersection(states[1].boundary, states[2].boundary)))
        self.assertTrue(shapely.equals(ink['province-boundaries'],
            shapely.intersection(provinces[1].boundary, provinces[2].boundary)))
        self.assertTrue(shapely.equals(ink['state-boundaries'],
            shapely.intersection(shapely.MultiLineString(state_paths), land)))
        self.assertEqual(shapely.intersection(ink['state-boundaries'], ink['province-boundaries']).length, 0)
        for row in range(4):
            for col in range(4):
                point = shapely.Point(col + .5, row + .5)
                original = next(int(label) for face, label in zip(faces, labels, strict=True)
                                if face.covers(point))
                self.assertTrue(provinces[original].covers(point))

    def test_review_metrics_use_physical_frontiers_without_outlet_label_walls(self):
        shape = (16, 32)
        water = np.zeros(shape, dtype=np.uint8)
        water[:2] = 1
        elevation = np.where(water == 0, .24, 0.)
        grid = SimpleNamespace(
            shape=shape, water=water, elevation=elevation,
            river_order=np.zeros(shape, dtype=np.uint8),
            snow=np.zeros(shape, dtype=bool),
            metadata={"extents": {"west": -180, "east": 180, "north": 60, "south": -60},
                      "planet": {"radiusKm": 6400}},
        )
        states = np.where(water != 0, 0, np.where(np.indices(shape)[1] < 16, 1, 2))
        population = np.where(water == 0, 1., 0.)
        population /= population.sum()
        society = SimpleNamespace(
            transport=SimpleNamespace(routes=(), bridges=()),
            politics=SimpleNamespace(state_id=states),
            provinces=SimpleNamespace(province_id=states),
            population=SimpleNamespace(population_weight=population,
                                       population_min=1000000, population_max=2000000),
        )
        thematic = SimpleNamespace(
            drainage_basin=np.ones(shape, dtype=np.int32),
            climate=SimpleNamespace(annual_precipitation=np.full(shape, .5)),
            land_potential=np.full(shape, .5),
        )
        flat = _territorial_quality_metrics(grid, thematic, society)
        thematic.drainage_basin = states.copy()
        self.assertEqual(_territorial_quality_metrics(grid, thematic, society), flat)
        self.assertTrue(all(np.isfinite(value) for value in flat.values()))
        self.assertEqual(flat["bridgesOffRoadOrRiver"], 0)
        grid.river_order[2:, 16] = 4
        river = _territorial_quality_metrics(grid, thematic, society)
        self.assertGreater(river["stateBoundaryAlignment"], flat["stateBoundaryAlignment"])
        self.assertGreater(river["provinceBoundaryAlignment"], flat["provinceBoundaryAlignment"])

    def test_administrative_overlays_reject_a_detached_river_boundary_source(self):
        paths = (np.asarray(((1.0, 0.0), (1.0, 2.0))),)
        with self.assertRaises(TypeError):
            _partition_boundary_overlay(
                "province-boundaries",
                paths,
                color="#685f54",
                width=0.52,
                river_paths=paths,
            )

    def test_riverbank_country_border_uses_the_shared_coverage_edge(self):
        """A river cannot move a border away from the province coverage it colors."""

        faces = (
            shapely.box(0, 0, 1, 2),
            shapely.box(1, 0, 2, 2),
        )
        _state_ids, state_paths, province_paths = _administrative_boundary_paths(
            faces,
            np.asarray((1, 2), dtype=np.int32),
            np.asarray((0, 1, 2), dtype=np.int32),
        )

        self.assertEqual(
            _edge_set(state_paths),
            {((1.0, 0.0), (1.0, 2.0))},
        )
        self.assertEqual(_edge_set(province_paths), set())

    def test_t_junction_keeps_the_state_edge_and_internal_province_edge_complete(self):
        """Coverage arcs meet at the same T node instead of being buffer-clipped."""

        faces = (
            shapely.box(0, 0, 1, 1),
            shapely.box(0, 1, 1, 2),
            shapely.Polygon(((1, 0), (2, 0), (2, 2), (1, 2), (1, 1), (1, 0))),
        )
        _state_ids, state_paths, province_paths = _administrative_boundary_paths(
            faces,
            np.asarray((1, 2, 3), dtype=np.int32),
            np.asarray((0, 1, 1, 2), dtype=np.int32),
        )

        self.assertEqual(
            _edge_set(state_paths),
            {
                ((1.0, 0.0), (1.0, 1.0)),
                ((1.0, 1.0), (1.0, 2.0)),
            },
        )
        self.assertEqual(
            _edge_set(province_paths),
            {((0.0, 1.0), (1.0, 1.0))},
        )

    def test_internal_province_line_is_not_repeated_as_a_country_boundary(self):
        faces = (
            shapely.box(0, 0, 1, 2),
            shapely.box(1, 0, 2, 2),
        )
        _state_ids, state_paths, province_paths = _administrative_boundary_paths(
            faces,
            np.asarray((1, 2), dtype=np.int32),
            np.asarray((0, 1, 1), dtype=np.int32),
        )

        self.assertEqual(_edge_set(state_paths), set())
        self.assertEqual(
            _edge_set(province_paths),
            {((1.0, 0.0), (1.0, 2.0))},
        )
