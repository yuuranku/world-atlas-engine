import unittest
import re
import json
import xml.etree.ElementTree as ET
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import shapely

from world_atlas.core.render import (
    _bridge_overlay,
    _categorical_partition_paths,
    _mask_paths,
    _geometry_filled_paths,
    _scalar_zone_paths,
    _surface_outline_paths,
    _filled_mask_paths,
    _transport_overlay,
)
from world_atlas.core.cartographic_surface import continuous_land_surface
from world_atlas.core.cartographic_relief import physical_relief_paths
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.continuous_scalar import ContinuousScalarField
from world_atlas.core.river_courses import native_river_paths
from world_atlas.core.raster_topology import categorical_coverage


_spec = importlib.util.spec_from_file_location(
    "alignment_svg_decoder", Path(__file__).resolve().parents[1] / "scripts" / "reencode_globe_paths.py")
_svg_decoder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_svg_decoder)


def filled_geometry(paths):
    polygons = []
    for points, codes in paths:
        starts = np.flatnonzero(codes == 1)
        rings = [points[first:last] for first, last in
                 zip(starts, (*starts[1:], len(points)), strict=True)]
        polygons.append(shapely.Polygon(rings[0], rings[1:]))
    return shapely.union_all(polygons)


def physical_surface(land):
    field = PhysicalTerrainField(np.where(land, 1., -1.), land_mask=land,
                                 sea_level_m=0, elevation_scale_m=1000,
                                 elevation_exponent=1.06)
    return continuous_land_surface(SimpleNamespace(water=(~land).astype(np.uint8)),
                                   terrain_field=field), field


class RenderGridAlignmentTests(unittest.TestCase):
    def test_category_overlay_roundoff_cannot_cut_numeric_field_bands(self):
        rows, columns = np.indices((5, 5))
        values = .17*rows+.37*columns
        land = np.ones(values.shape, dtype=bool)
        frame = shapely.box(0, 0, 5, 5)
        # A category overlay's sub-ulp sliver must not move a visible shared
        # numeric edge onto a different point of the SVG coordinate grid.
        hole = ((1.1, 1.4), (2.3, .7), (1.7, np.nextafter(1.05, np.inf)))
        domain = shapely.Polygon(frame.exterior.coords, [hole])
        self.assertTrue(domain.is_valid)
        reference = _scalar_zone_paths(values, land, (.5, 1.), working_surface=frame)
        actual = _scalar_zone_paths(values, land, (.5, 1.), working_surface=domain)
        for clean, observed in zip(reference, actual, strict=True):
            self.assertTrue(filled_geometry(clean).symmetric_difference(
                filled_geometry(observed)).is_empty)

    def test_transport_overlay_handles_water_lanes_and_rails_alongside_curved_roads(self):
        shape = (12, 24)
        water = np.zeros(shape, dtype=np.uint8)
        water[6:] = 1
        grid = SimpleNamespace(shape=shape, water=water, elevation=np.zeros(shape))
        routes = (
            SimpleNamespace(identifier="road-one", mode="road", importance="trunk",
                            path=((2.5, 4.5), (8.5, 4.5), (8.5, 2.5), (14.5, 2.5))),
            SimpleNamespace(identifier="river-one", mode="river", importance="trunk",
                            path=((2.5, 4.5), (14.5, 4.5))),
            SimpleNamespace(identifier="rail-one", mode="rail", importance="trunk",
                            path=((2.5, 3.5), (14.5, 3.5))),
            SimpleNamespace(identifier="sea-one", mode="sea", importance="trunk",
                            path=((2.5, 8.5), (8.5, 8.5), (8.5, 10.5), (14.5, 10.5))),
        )
        prepared = SimpleNamespace(paths=tuple(
            (route.mode, route.importance, np.asarray(route.path))
            for route in routes if route.mode != "river"
        ))
        svg = _transport_overlay(prepared, technology_era="preindustrial")
        for mode in ("road", "rail", "sea"):
            self.assertIn(f'data-route-mode="{mode}"', svg)
        self.assertNotIn('data-route-mode="river"', svg)
        self.assertNotIn(" Q", svg, "the renderer must serialize the checked network without deforming it")

    def test_bridge_overlay_uses_the_shared_road_river_intersection(self):
        route = SimpleNamespace(identifier="road-one", mode="road",
                                path=((2.5, 2.5), (10.5, 9.5)))
        bridge = SimpleNamespace(identifier="bridge-one", route_identifier="road-one",
                                 column=5, row=6, importance="regional", river_order=2)
        road = shapely.LineString(route.path)
        river = shapely.LineString(((6.5, 0), (6.5, 12)))
        expected = shapely.intersection(road, river)
        positions = {bridge.identifier: tuple(expected.coords[0])}
        prepared = SimpleNamespace(bridges=(bridge,), positions=positions,
            tangents={bridge.identifier: (8., 7.)},
            spans={bridge.identifier: {"geometry": json.loads(shapely.to_geojson(
                road.intersection(river.buffer(.05))))}})
        svg = _bridge_overlay(prepared)
        transform = re.search(r'transform="translate\(([^ ]+) ([^)]+)\)', svg)
        published = np.array((float(transform[1]), float(transform[2])))
        self.assertLess(np.linalg.norm(published - positions["bridge-one"]), 1e-8,
                        "bridge paint must keep the precise shared route anchor at deep zoom")
        self.assertEqual((bridge.column, bridge.row), (5, 6))

    def test_bridge_deck_preserves_each_actual_confluence_arm(self):
        centre = np.array((5., 5.))
        deck = shapely.MultiLineString((
            ((4.8, 5.), (5., 5.), (5.1, 5.17)),
            ((5., 5.), (5., 4.8)),
        ))
        bridge = SimpleNamespace(identifier="fork-bridge", route_identifier="road",
                                 importance="regional", river_order=3)
        prepared = SimpleNamespace(bridges=(bridge,),
            positions={bridge.identifier: tuple(centre)},
            tangents={bridge.identifier: (1., 1.)},
            spans={bridge.identifier: {"geometry": json.loads(shapely.to_geojson(deck))}})
        root = ET.fromstring(_bridge_overlay(prepared))
        data = root.find(".//path[@class='bridge-physical-span']").attrib["d"]
        branches = []
        for coordinates, closed in _svg_decoder.linear_subpaths(data):
            self.assertFalse(closed)
            points = np.asarray(coordinates, dtype=float) / 100_000_000
            branches.append(points)
        published = shapely.MultiLineString(branches)
        self.assertLess(deck.hausdorff_distance(published), 1e-7)
        self.assertEqual(len(branches), 2)

    def test_uniform_numeric_ocean_depth_has_no_artificial_frame_class(self):
        depth = np.full((8, 16), .375)
        original = depth.copy()
        frame = shapely.box(0, 0, 16, 8)
        bands = _scalar_zone_paths(depth, np.ones(depth.shape, dtype=bool), (.25, .5),
                                   working_surface=frame)
        self.assertEqual([bool(band) for band in bands], [False, True, False])
        self.assertTrue(filled_geometry(bands[1]).equals(frame))
        np.testing.assert_array_equal(depth, original)

    def test_numeric_ocean_depth_is_periodic_and_uses_the_physical_water_clip(self):
        shape = (6, 10)
        depth = np.broadcast_to(.45 + .35*np.sin(np.arange(10)*2*np.pi/10), shape).copy()
        ocean = np.ones(shape, dtype=bool)
        ocean[2, 4] = False
        field = ContinuousScalarField(depth, ocean)
        samples = field.sample_rect((0., 10.), np.arange(6)+.5)
        np.testing.assert_allclose(samples[:, 0], samples[:, 1], atol=1e-12)
        native = field.sample_rect(np.arange(10)+.5, np.arange(6)+.5)
        np.testing.assert_allclose(native[ocean], depth[ocean], atol=1e-12)
        water_surface = shapely.box(0, 0, 10, 6).difference(shapely.box(4, 2, 5, 3))
        paths = _scalar_zone_paths(depth, ocean, (.25, .5, .75),
                                  working_surface=shapely.box(0, 0, 10, 6))
        working = shapely.union_all([filled_geometry(band) for band in paths])
        self.assertTrue(working.equals(shapely.box(0, 0, 10, 6)))
        coverage = working.intersection(water_surface)
        self.assertTrue(coverage.equals(water_surface))
        self.assertFalse(coverage.covers(shapely.Point(4.5, 2.5)), 'ocean color cannot cover physical land')

    def test_native_numeric_ocean_edge_observations_are_retained(self):
        depth = np.tile((.05, .1, .18, .3, .45, .65, .85, .95), (4, 1))
        original = depth.copy()
        field = ContinuousScalarField(depth, np.ones(depth.shape, dtype=bool))
        np.testing.assert_allclose(field.sample_rect(np.arange(8)+.5, np.arange(4)+.5), depth, atol=1e-12)
        np.testing.assert_array_equal(depth, original)

    def test_mask_fills_cover_full_outer_cells_without_a_texture_rim(self):
        for shape in ((4, 6), (1, 6), (4, 1), (1, 1)):
            with self.subTest(shape=shape):
                paths = _filled_mask_paths(np.ones(shape, dtype=bool))
                self.assertEqual(len(paths), 1)
                polygon = shapely.Polygon(paths[0][0])
                self.assertTrue(polygon.equals(shapely.box(0, 0, shape[1], shape[0])))
                self.assertFalse(_surface_outline_paths(paths, shape), "a full mask has no physical shoreline")

    def test_mask_padding_keeps_interior_shore_at_the_same_cell_boundary(self):
        mask = np.zeros((5, 6), dtype=bool)
        mask[:, :3] = True
        fills = _filled_mask_paths(mask)
        self.assertEqual(len(fills), 1)
        polygon = shapely.Polygon(fills[0][0])
        self.assertTrue(polygon.equals(shapely.box(0, 0, 3, 5)))
        outline = shapely.union_all([shapely.LineString(path) for path in _surface_outline_paths(fills, mask.shape)])
        self.assertTrue(outline.equals(shapely.LineString(((3, 0), (3, 5)))))
        self.assertEqual(outline.distance(shapely.Point(2.5, 2.5)), .5)

    def test_uniform_ground_keeps_the_world_frame_without_false_coasts(self):
        shape = (8, 16)
        surface, _field = physical_surface(np.ones(shape, dtype=bool))
        paths = _geometry_filled_paths(surface)
        self.assertTrue(surface.equals(shapely.box(0, 0, shape[1], shape[0])))
        self.assertFalse(_surface_outline_paths(paths, shape), "map corners must not become false coasts")

    def test_scalar_regions_reach_every_frame_edge_and_keep_their_inner_interval(self):
        values = np.full((12, 20), .8)
        values[4:8, 7:13] = .2
        for turn in range(4):
            with self.subTest(frame_direction=turn):
                original = values.copy()
                height, width = values.shape
                frame = shapely.box(0, 0, width, height)
                paths = _scalar_zone_paths(values, np.ones(values.shape, dtype=bool), (.5,),
                                          working_surface=frame)
                inner, outer = (filled_geometry(part) for part in paths)
                self.assertTrue(outer.is_valid)
                self.assertTrue(outer.boundary.covers(frame.boundary))
                self.assertEqual(len(outer.interiors), 1)
                self.assertLess(shapely.symmetric_difference(shapely.union_all((inner, outer)), frame).area, 1e-10)
                self.assertLess(inner.intersection(outer).area, 1e-12)
                for row, column in np.ndindex(values.shape):
                    expected = inner if values[row, column] < .5 else outer
                    self.assertTrue(expected.contains(shapely.Point(column+.5, row+.5)))
                np.testing.assert_array_equal(values, original)
            values = np.rot90(values)

    def test_periodic_ocean_band_keeps_both_longitude_frame_edges(self):
        values = np.full((20, 40), .2)
        values[3:17, :8] = .8
        values[3:17, -8:] = .8
        bands = _scalar_zone_paths(values, np.ones(values.shape, dtype=bool), (.5,),
                                  working_surface=shapely.box(0, 0, 40, 20))
        surface = filled_geometry(bands[1])
        self.assertTrue(surface.is_valid)
        for row in range(4, 16):
            for column in (.001, 39.999):
                self.assertTrue(surface.covers(shapely.Point(column, row+.5)),
                                "periodic ocean color must reach its physical frame")

    def test_isolated_land_sample_is_traced_from_the_ground_zero_contour(self):
        land = np.zeros((9, 9), dtype=bool)
        land[4, 4] = True
        original = land.copy()
        island, field = physical_surface(land)
        self.assertTrue(island.is_valid)
        self.assertEqual(island.geom_type, 'Polygon')
        self.assertTrue(island.contains(shapely.Point(4.5, 4.5)))
        self.assertFalse(island.covers(shapely.Point(3.5, 4.5)))
        points = np.asarray(island.exterior.coords)
        self.assertLess(np.abs(field.sample_points(points[:, 0], points[:, 1])).max(), .025,
                        "the silhouette must follow the source zero crossing")
        np.testing.assert_array_equal(land, original)

    def test_river_mouth_meets_visible_shore_without_moving_its_upstream_channel(self):
        shape = (5, 6)
        water = np.zeros(shape, dtype=np.uint8)
        water[:, 3:] = 1
        order = np.zeros(shape, dtype=np.uint8)
        order[2, 1:3] = 1
        downstream = np.full(shape, -1, dtype=np.int32)
        downstream[2, 1] = 2 * shape[1] + 2
        downstream[2, 2] = 2 * shape[1] + 3
        grid = SimpleNamespace(shape=shape, water=water, river_order=order, flow_to=downstream)
        land_surface, field = physical_surface(water == 0)
        paths = native_river_paths(grid, field)
        self.assertEqual(len(paths), 1)
        np.testing.assert_array_equal(paths[0][:-1], ((1.5, 2.5), (2.5, 2.5)))
        self.assertLess(land_surface.boundary.distance(shapely.Point(paths[0][-1])), 1e-8)

    def test_river_mouth_follows_native_outflow_instead_of_projecting_sideways(self):
        shape = (5, 6)
        water = np.zeros(shape, dtype=np.uint8)
        water[:, 3:] = 1
        order = np.zeros(shape, dtype=np.uint8)
        order[2, 1:3] = 1
        downstream = np.full(shape, -1, dtype=np.int32)
        downstream[2, 1] = 2 * shape[1] + 2
        downstream[2, 2] = 2 * shape[1] + 3
        grid = SimpleNamespace(shape=shape, water=water, river_order=order, flow_to=downstream)
        raw = np.where(water == 0, 10., -3.)
        field = PhysicalTerrainField(raw, land_mask=water == 0, sea_level_m=0,
                                    elevation_scale_m=1000, elevation_exponent=1)
        paths = native_river_paths(grid, field)
        self.assertEqual(paths[0][-1, 1], 2.5)
        self.assertGreater(paths[0][-1, 0], 3.)
        self.assertLess(abs(float(field.sample_points(*paths[0][-1]))), 1e-9)
        np.testing.assert_array_equal(paths[0][:-1], ((1.5, 2.5), (2.5, 2.5)))

    def test_visible_coast_uses_the_same_curved_land_edge_and_omits_map_frame(self):
        land = np.zeros((8, 10), dtype=bool)
        land[:, :4] = True
        land[2:5, 4:6] = True
        surface, _field = physical_surface(land)
        surfaces = _geometry_filled_paths(surface)
        coast = _surface_outline_paths(surfaces, land.shape)
        self.assertTrue(coast)
        for path in coast:
            for first, second in zip(path[:-1], path[1:]):
                for axis, edge in ((0, 0), (0, 10), (1, 0), (1, 8)):
                    self.assertFalse(np.isclose(first[axis], edge) and np.isclose(second[axis], edge))
                self.assertTrue(any(
                    np.array_equal(first, a) and np.array_equal(second, b)
                    for points, _ in surfaces for a, b in zip(points[:-1], points[1:])
                ), "coastal ink must exactly follow the visible land surface")

    def test_coastal_road_at_dry_cell_centers_stays_half_a_cell_inland(self):
        land = np.zeros((5, 6), dtype=bool)
        land[:, :3] = True
        coast = shapely.LineString(_mask_paths(land)[0])
        # Routes and settlement anchors already use column/row + 0.5.
        road = shapely.LineString(((2.5, 0.5), (2.5, 4.5)))

        self.assertEqual(coast.bounds, (3.0, 0.0, 3.0, 5.0))
        self.assertEqual(road.distance(coast), 0.5)

    def test_elevation_contours_and_fills_use_the_same_cell_centers(self):
        elevation = np.tile(np.array((0., 1., 2., 3., 4., 0.)), (4, 1))
        terrain = PhysicalTerrainField(elevation, land_mask=elevation > 0,
            sea_level_m=0., elevation_scale_m=10., elevation_exponent=1.)
        bands, contours, _levels = physical_relief_paths(
            terrain, terrain.palette_elevation(np.array((1.5, 2.5))))
        polygon = filled_geometry(bands[0]).difference(filled_geometry(bands[1]))

        # A unit slope through centres n+.5 crosses these cuts at x=2 and 3.
        # Latitude extends the outer samples through both polar half-cells.
        self.assertEqual((polygon.bounds[1], polygon.bounds[3]), (0., 4.))
        for expected_x in (2., 3.):
            self.assertTrue(any(np.allclose(path[:, 0], expected_x, atol=1e-12,
                                          rtol=0.) for path in contours))
        for path in contours:
            self.assertTrue(polygon.boundary.covers(shapely.LineString(path)))

    def test_categorical_fill_boundary_matches_coast_and_raster_image(self):
        categories = np.zeros((4, 6), dtype=np.uint8)
        categories[:, 3:] = 1
        surface = shapely.box(0, 0, 6, 4)
        partitions = _categorical_partition_paths(
            categories, np.ones(categories.shape, dtype=bool),
            category_count=2, land_surface=surface,
        )
        western = filled_geometry(partitions[0])
        eastern = filled_geometry(partitions[1])
        shared = western.boundary.intersection(eastern.boundary)

        self.assertTrue(shapely.union_all((western, eastern)).equals(surface))
        self.assertTrue(shared.equals(shapely.LineString(((3., 0.), (3., 4.)))))
        self.assertEqual(western.bounds, (0., 0., 3., 4.))
        self.assertEqual(eastern.bounds, (3., 0., 6., 4.))
        self.assertTrue(western.contains(shapely.Point(2.5, 1.5)))
        self.assertTrue(eastern.contains(shapely.Point(3.5, 1.5)))

    def test_sparse_category_identifiers_do_not_invent_intermediate_colors(self):
        categories = np.full((8, 12), 1, dtype=np.uint8)
        categories[:, 6:] = 5
        surface = shapely.box(0, 0, 12, 8)
        partitions = _categorical_partition_paths(
            categories, np.ones(categories.shape, dtype=bool),
            category_count=6, land_surface=surface,
        )
        self.assertEqual({index for index, paths in enumerate(partitions) if paths}, {1, 5})
        coverage = shapely.union_all([filled_geometry(paths) for paths in partitions])
        self.assertTrue(coverage.equals(surface))

    def test_thematic_categories_fill_continuous_coast_and_preserve_the_lake_hole(self):
        land = np.zeros((14, 18), dtype=bool)
        land[2:12, 2:16] = True
        land[5:9, 7:11] = False
        categories = np.zeros(land.shape, dtype=np.uint8)
        categories[land] = 1
        categories[:, 9:][land[:, 9:]] = 5
        grid = SimpleNamespace(
            water=(~land).astype(np.uint8), elevation=np.where(land, .05, 0),
            bathymetry_band=np.where(land, -1, 0), metadata={
                "planet": {"radiusKm": 6400}, "worldProfile": {"seed": 7},
                "bathymetry": {"levels": 8},
                "extents": {"west": -180, "east": 180, "north": 90, "south": -90},
            },
        )
        terrain_field = PhysicalTerrainField(np.where(land, 50., -20.), land_mask=land,
                                             sea_level_m=0, elevation_scale_m=1000,
                                             elevation_exponent=1.06)
        surface = continuous_land_surface(grid, terrain_field=terrain_field)
        partitions = _categorical_partition_paths(
            categories, land, category_count=6, land_surface=surface,
        )
        working = shapely.union_all([filled_geometry(paths) for paths in partitions])
        colored = working.intersection(surface)
        self.assertLess(shapely.area(shapely.symmetric_difference(colored, surface)), 1e-10,
                        "colored regions must reach exactly the displayed continuous shoreline")
        self.assertTrue(shapely.covers(surface, colored), "thematic color must not spill through the land clip")
        self.assertFalse(colored.covers(shapely.Point(8.5, 6.5)), "the inland lake must stay uncolored")
        self.assertEqual({index for index, paths in enumerate(partitions) if paths}, {1, 5})

    def test_continuous_partition_keeps_sample_ownership_and_the_inactive_hole(self):
        categories = np.tile(np.arange(1, 5, dtype=np.int16), (3, 1))
        active = np.ones(categories.shape, dtype=bool)
        active[1, 1] = False

        original = categories.copy()
        faces, labels = categorical_coverage(categories, active, category_count=5)

        coverage = shapely.union_all(faces)
        self.assertTrue(shapely.coverage_is_valid(faces))
        self.assertFalse(coverage.covers(shapely.Point(1.5, 1.5)))
        self.assertEqual(coverage.bounds, (0.0, 0.0, 4.0, 3.0))
        for row, column in np.argwhere(active):
            center = shapely.Point(column + 0.5, row + 0.5)
            owners = [
                label
                for face, label in zip(faces, labels, strict=True)
                if face.contains(center)
            ]
            self.assertEqual(owners, [categories[row, column]])
        np.testing.assert_array_equal(categories, original)


if __name__ == "__main__":
    unittest.main()
