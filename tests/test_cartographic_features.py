"""Far summaries retain delivered category ownership and shared coverage."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import shapely

from world_atlas.core.cartographic_features import (
    _protect_native_coverage, shared_display_coverage, line_features, overview_markup,
)
from world_atlas.core.cartographic_tiles import TileFeature, TileLevel, geometry_path_data, write_atlas_tiles
from world_atlas.core.render import _transport_overlay

spec = importlib.util.spec_from_file_location(
    "overview_svg_decoder", Path(__file__).resolve().parents[1] / "scripts" / "check_coastal_coverage.py"
)
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


class CartographicFeaturesTests(unittest.TestCase):
    def test_actual_transport_fragment_collapsing_to_one_integer_has_no_paint(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' /
                              'transport-collapsed-integer-line.json').read_text())
        points = np.asarray(fixture['points'])
        before = points.copy()
        self.assertNotEqual(points[0, 0], points[1, 0])
        geometry = shapely.LineString(points)
        self.assertEqual(geometry_path_data(geometry), '')
        self.assertEqual(line_features([points], fixture['layer'], '#8b5e3c', 1.15), [])
        features = []
        prepared = SimpleNamespace(paths=[(fixture['mode'], fixture['importance'], points)])
        markup = _transport_overlay(prepared, technology_era='medieval', tile_features=features)
        self.assertEqual(features, [])
        self.assertEqual(ET.fromstring(markup).findall('.//path'), [])
        np.testing.assert_array_equal(points, before)
        # The exporter keeps its strict rejection of an explicitly supplied
        # empty path. Filtering belongs to the source paint producer.
        malformed = TileFeature(geometry.envelope, {'fill': 'none', 'stroke': '#8b5e3c'},
                                fixture['layer'], path_data='')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'path_data must be a nonempty SVG'):
                write_atlas_tiles(directory, 128, 448,
                    [TileLevel('regional', 4, shapely.box(0, 0, 128, 448), [malformed])])

    def test_nonzero_integer_transport_keeps_every_surviving_node_and_both_styles(self):
        # This one output quantum is paintable. An exact collinear interior
        # model event and an adjacent source duplicate remain in the source.
        points = np.asarray([(105.99639198, 425.99639198),
                             (105.99639198, 425.99639198),
                             (105.99639199, 425.99639198),
                             (105.99639200, 425.99639198)])
        before = points.copy()
        expected = geometry_path_data(shapely.LineString(points))
        self.assertTrue(expected)
        features = line_features([points], 'transport-network', '#8b5e3c', 1.15)
        self.assertEqual(len(features), 1)
        np.testing.assert_array_equal(features[0].geometry.coords, points)
        transport = []
        markup = _transport_overlay(SimpleNamespace(paths=[('road', 'local', points)]),
                                    technology_era='medieval', tile_features=transport)
        self.assertEqual(len(transport), 2)
        for feature in transport:
            self.assertIsNone(feature.path_data)
            self.assertEqual(geometry_path_data(feature.geometry), expected)
        self.assertEqual([node.attrib['d'] for node in ET.fromstring(markup).findall('path')],
                         [expected, expected])
        self.assertEqual(expected.count('425.99639198'), 3,
                         'the source collinear model event must remain an encoded node')
        np.testing.assert_array_equal(points, before)

    def test_administrative_overview_fills_and_ink_keep_one_shared_boundary(self):
        edge = [(2, 0), (2, 1.75), (2.2, 2), (2, 2.25), (2, 4)]
        left = shapely.Polygon([(0, 0), *edge, (0, 4)])
        right = shapely.Polygon([*edge, (4, 4), (4, 0)])
        source_edge = shapely.LineString(edge)
        province_edge = [(0, 2), (1, 2.08), (2.2, 2)]
        lower = shapely.Polygon([(0, 0), *edge[:3], *province_edge[-2::-1]])
        upper = shapely.Polygon([*province_edge, *edge[3:], (0, 4)])
        # A valid coverage can retain every native centre while moving this
        # state edge .2 cell. The province T junction pins the same source
        # vertex, so separate coverage reduction also splits the hierarchy.
        reduced = shapely.coverage_simplify([left, right], .36,
                                            simplify_boundary=False)
        reduced_edge = reduced[0].boundary.intersection(reduced[1].boundary)
        self.assertAlmostEqual(reduced_edge.hausdorff_distance(source_edge), .2)
        xx, yy = np.meshgrid(np.arange(4) + .5, np.arange(4) + .5)
        for before, after in zip([left, right], reduced, strict=True):
            self.assertTrue(np.array_equal(shapely.intersects_xy(before, xx, yy),
                                          shapely.intersects_xy(after, xx, yy)))

        features = []
        for theme, faces in [('political', [left, right]),
                             ('provinces', [lower, upper, right])]:
            features.extend(TileFeature(face, {'fill': ['red', 'green', 'blue'][index],
                                               'data-category': str(index)},
                                'theme-fill', theme, 'theme')
                            for index, face in enumerate(faces))
        features += line_features([edge], 'state-boundaries', '#46413b', 1.4)
        features += line_features([province_edge], 'province-boundaries', '#685f54', .95)
        surface = shapely.box(0, 0, 4, 4)
        delivered = {}
        for theme in ('political', 'provinces'):
            markup = overview_markup(features, surface, 4, 4, section='theme', theme=theme)
            root = ET.fromstring('<svg>' + markup + '</svg>')
            delivered[theme] = [decoder.path_geometry(path.get('d'))
                                 for path in root.findall('.//path')]
        political_edge = delivered['political'][0].boundary.intersection(
            delivered['political'][1].boundary)
        province_state_edge = shapely.union_all(delivered['provinces'][:2]).boundary.intersection(
            delivered['provinces'][2].boundary)
        self.assertTrue(political_edge.equals(source_edge))
        self.assertTrue(province_state_edge.equals(source_edge))
        self.assertTrue(delivered['provinces'][0].boundary.intersection(
            delivered['provinces'][1].boundary).equals(shapely.LineString(province_edge)))

        markup = overview_markup(features, surface, 4, 4, section='ink')
        root = ET.fromstring('<svg>' + markup + '</svg>')
        for layer, points in [('state-boundaries', edge), ('province-boundaries', province_edge)]:
            path = root.find(f".//g[@data-tile-layer='{layer}']/path")
            self.assertEqual(path.get('d'), geometry_path_data(shapely.LineString(points)))

    def test_overview_retains_jointly_checked_road_river_crossing(self):
        river = shapely.LineString(((.5, 1), (1.5, 1.08), (2.5, 1)))
        road = shapely.LineString(((1.5, .5), (1.58, 1.08), (1.5, 1.5)))
        facility = river.intersection(road)
        summary_crossing = shapely.simplify(river, .12).intersection(
            shapely.simplify(road, .12))
        self.assertGreater(facility.distance(summary_crossing), .05,
                           "independent valid line simplification detaches the shared facility")
        features = [
            TileFeature(river, {"data-river-order": "4"}, "rivers"),
            TileFeature(road, {"data-route-importance": "trunk"}, "transport-network"),
        ]
        markup = overview_markup(features, shapely.box(0, 0, 3, 2), 3, 2, section="ink")
        root = ET.fromstring("<svg>" + markup + "</svg>")
        for feature in features:
            path = root.find(f".//g[@data-tile-layer='{feature.layer}']/path")
            self.assertEqual(path.attrib["d"], geometry_path_data(feature.geometry))

    def test_shared_noding_keeps_t_junction_coverage_and_natural_holes(self):
        left = shapely.Polygon([(0,0),(2,0),(2,1),(2,2),(0,2)])
        right = shapely.box(2,0,4,2)
        lake = shapely.box(.25,.25,.75,.75)
        original = [left.difference(lake), right]
        self.assertFalse(shapely.coverage_is_valid(original))
        result = shared_display_coverage(original)
        self.assertTrue(shapely.coverage_is_valid(result))
        self.assertTrue(shapely.union_all(result).equals(shapely.union_all(original)))
        self.assertFalse(shapely.union_all(result).covers(shapely.Point(.5,.5)))

    def test_noded_faces_keep_actual_later_category_paint_order(self):
        result = shared_display_coverage([shapely.box(0,0,3,2), shapely.box(2,0,4,2)])
        self.assertTrue(result[0].equals(shapely.box(0,0,2,2)))
        self.assertTrue(result[1].equals(shapely.box(2,0,4,2)))
        self.assertTrue(shapely.coverage_is_valid(result))

    def test_population_intersection_is_noded_once_on_the_delivery_grid(self):
        # Actual v97 population contours produced two intersection nodes
        # separated by one output quantum when floating noding was followed
        # by independently snapping polygonized category faces.
        fixture = json.loads((Path(__file__).parent / "fixtures" /
                              "population-grid-noding.json").read_text())
        source = shapely.from_wkb(fixture["regions_wkb"])
        before = [geometry.wkb for geometry in source]
        delivered = shapely.set_precision(source, 1e-8)
        self.assertTrue(np.all(shapely.is_valid(delivered)))
        self.assertFalse(shapely.coverage_is_valid(delivered))
        result = shared_display_coverage(source)
        self.assertEqual(len(result), len(source))
        self.assertTrue(np.all(shapely.is_valid(result)))
        self.assertTrue(shapely.coverage_is_valid(result))
        self.assertEqual(np.sum(shapely.get_num_coordinates(
            shapely.coverage_invalid_edges(result))), 0)
        self.assertTrue(shapely.symmetric_difference(
            shapely.union_all(delivered), shapely.union_all(result), grid_size=1e-8).is_empty)
        xmin, ymin, xmax, ymax = fixture["frame_bounds"]
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 42)[1:-1],
                             np.linspace(ymin, ymax, 42)[1:-1])
        expected = np.full(xx.shape, -1, dtype=np.int32)
        memberships = np.zeros(xx.shape, dtype=np.uint8)
        actual = np.full(xx.shape, -1, dtype=np.int32)
        for identifier, (original, geometry) in enumerate(zip(delivered, result, strict=True)):
            expected[shapely.intersects_xy(original, xx, yy)] = identifier
            owned = shapely.intersects_xy(geometry, xx, yy)
            memberships += owned
            actual[owned] = identifier
        self.assertTrue(np.all(memberships == 1))
        self.assertTrue(np.array_equal(expected, actual))
        self.assertEqual(before, [geometry.wkb for geometry in source])

    def test_raw_shared_t_node_is_rounded_before_deriving_category_faces(self):
        # Actual population support/density overlay: independently rounding
        # this almost collinear T node creates a three-edge transparent hole.
        fixture = json.loads((Path(__file__).parent / "fixtures" /
                              "population-shared-graph-precision.json").read_text())
        source = shapely.from_wkb(fixture["regions_wkb"])
        frame = shapely.box(*fixture["frame_bounds"])
        independently_delivered = [decoder.path_geometry(geometry_path_data(region))
                                  for region in source]
        rounded_union = shapely.union_all(independently_delivered)
        self.assertGreater(frame.difference(rounded_union).area, 0)
        result = shared_display_coverage(source)
        self.assertTrue(shapely.coverage_is_valid(result))
        delivered = [decoder.path_geometry(geometry_path_data(region)) for region in result]
        self.assertTrue(frame.difference(shapely.union_all(delivered)).is_empty)
        self.assertTrue(shapely.union_all(delivered).difference(frame).is_empty)
        # Every interior segment must have an exact opposite category edge;
        # a valid coverage alone would permit a small transparent hole.
        edges = {}
        for region in delivered:
            for part in shapely.get_parts(region):
                for ring in (part.exterior, *part.interiors):
                    points = np.rint(np.asarray(ring.coords)*1e8).astype(np.int64)
                    for first, last in zip(points[:-1], points[1:], strict=True):
                        key = tuple(sorted((tuple(first), tuple(last))))
                        edges[key] = edges.get(key, 0)+1
        for edge, count in edges.items():
            segment = shapely.LineString(np.asarray(edge)/1e8)
            self.assertEqual(count, 1 if frame.boundary.covers(segment) else 2)

    def test_population_source_exit_keeps_native_support_and_delivered_shared_nodes(self):
        from world_atlas.core.render import _population_zone_paths, _filled_path_data
        from world_atlas.core.society.population import POPULATION_DENSITY_THRESHOLDS
        fixture = json.loads((Path(__file__).parent / "fixtures" /
                              "population-native-source.json").read_text())
        density = np.asarray(fixture["density"])
        land = np.asarray(fixture["land"])
        height, width = density.shape
        frame = shapely.box(0, 0, width, height)
        paths = _population_zone_paths(density, land, working_surface=frame)
        delivered = [decoder.path_geometry(" ".join(_filled_path_data(points, codes)
                                                   for points, codes in band)) for band in paths]
        self.assertEqual(len(delivered), len(POPULATION_DENSITY_THRESHOLDS)+1)
        self.assertTrue(np.all(shapely.is_valid(delivered)))
        self.assertTrue(shapely.coverage_is_valid(delivered))
        self.assertTrue(shapely.union_all(delivered).equals(frame))
        xx, yy = np.meshgrid(np.arange(width)+.5, np.arange(height)+.5)
        expected = np.searchsorted(POPULATION_DENSITY_THRESHOLDS, density, side="right")
        expected = np.where(density > 0, expected, 0)
        counts = np.zeros(density.shape, dtype=np.uint8)
        for identifier, region in enumerate(delivered):
            owned = shapely.contains_xy(region, xx, yy)
            counts += owned
            self.assertTrue(np.all(expected[owned] == identifier))
        self.assertTrue(np.all(counts == 1))

    def test_already_canonical_coverage_retains_its_exact_source_and_small_hole(self):
        hole = [(1., 1.), (1.03125, 1.00143845), (1.00244937, 1.00011274)]
        source = np.asarray([shapely.Polygon([(0,0),(2,0),(2,2),(0,2)], [hole])], dtype=object)
        self.assertTrue(shapely.coverage_is_valid(source))
        coordinates = shapely.get_coordinates(source)
        self.assertTrue(np.array_equal(coordinates, np.rint(coordinates*1e8)/1e8))
        before = source[0].wkb
        result = shared_display_coverage(source)
        self.assertIs(result, source)
        self.assertEqual(result[0].wkb, before)
        self.assertEqual(len(result[0].interiors), 1)

    def test_numeric_coastal_working_domain_preserves_every_visible_contour(self):
        from world_atlas.core.render import (_scalar_zone_paths, _filled_path_data,
                                             _partition_overlay_svg_document)
        from types import SimpleNamespace
        height, width = 12, 20
        xx, yy = np.meshgrid(np.arange(width)+.5, np.arange(height)+.5)
        values = .45 + .35*np.sin(xx*.6)*np.cos(yy*.3)
        land = (xx >= 4) & (xx < 16)
        frame = shapely.box(0, 0, width, height)
        working = shapely.box(2, 0, 18, height)
        shore = shapely.Polygon([(4.15, 0), (15.85, 0), (15.7, 6),
                                 (15.85, 12), (4.15, 12), (4.3, 6)])
        shore = shore.difference(shapely.box(8.2, 4.2, 8.8, 4.8))
        full = _scalar_zone_paths(values, land, (.3, .6), working_surface=frame)
        bounded = _scalar_zone_paths(values, land, (.3, .6), working_surface=working)
        def decode(paths):
            return [decoder.path_geometry(" ".join(_filled_path_data(points, codes)
                        for points, codes in band)) for band in paths]
        original, delivered = decode(full), decode(bounded)
        self.assertTrue(shapely.coverage_is_valid(delivered))
        self.assertTrue(shapely.union_all(delivered).equals(working))
        for before, after in zip(original, delivered, strict=True):
            self.assertTrue(before.symmetric_difference(after).intersection(shore).is_empty)
        expected = np.searchsorted((.3, .6), values, side='right')
        counts = np.zeros(values.shape, dtype=np.uint8)
        for identifier, geometry in enumerate(delivered):
            owned = shapely.contains_xy(geometry, xx, yy) & land
            counts += owned
            self.assertTrue(np.all(expected[owned] == identifier))
        self.assertTrue(np.all(counts[land] == 1))
        svg = _partition_overlay_svg_document(SimpleNamespace(shape=values.shape), bounded,
            land_surface=shore, title='Continuous field', partition_id='field',
            data_attribute='band', zones=(('low', 'blue'), ('middle', 'green'), ('high', 'red')))
        root = ET.fromstring(svg)
        clips = root.findall('.//{*}clipPath')
        self.assertEqual(len(clips), 1)
        master = decoder.path_geometry(clips[0].find('{*}path').attrib['d'])
        self.assertTrue(shapely.union_all(delivered).intersection(master).equals(master))
        self.assertTrue(master.equals(shore))
        self.assertEqual(len(root.findall('.//{*}g[@clip-path="url(#land-silhouette-clip)"]')), 1)

    def test_native_guard_restores_shared_chords_crossing_observations(self):
        edge = [(2.4,0),(2.4,.4),(2.6,.5),(2.4,.6),(2.4,4)]
        left = shapely.Polygon([(0,0), *edge, (0,4)])
        right = shapely.Polygon([*edge, (4,4),(4,0)])
        original = np.asarray([left,right], dtype=object)
        summary = shapely.coverage_simplify(original,.36,simplify_boundary=False)
        station = shapely.Point(2.5,.5)
        self.assertTrue(left.covers(station))
        self.assertFalse(summary[0].covers(station))
        protected = _protect_native_coverage(original,summary,(4,4))
        xx,yy = np.meshgrid(np.arange(4)+.5,np.arange(4)+.5)
        for source, actual in zip(original,protected,strict=True):
            self.assertTrue(np.array_equal(shapely.intersects_xy(source,xx,yy),
                                          shapely.intersects_xy(actual,xx,yy)))
        self.assertTrue(shapely.coverage_is_valid(protected))
        self.assertTrue(shapely.union_all(protected).equals(shapely.box(0,0,4,4)))

    def test_empty_classes_are_absent_without_renumbering_or_mutating_detail(self):
        features = [TileFeature(shapely.box(0,0,2,2),{"fill":"red","clip":"land","data-band":"1"},
                                "theme-fill","potential","theme"),
                    TileFeature(shapely.box(2,0,4,2),{"fill":"blue","clip":"land","data-band":"5"},
                                "theme-fill","potential","theme"),
                    TileFeature(shapely.GeometryCollection(),{"fill":"green","data-band":"7"},
                                "theme-fill","potential","theme")]
        before = [(feature.geometry.wkb,dict(feature.attributes)) for feature in features]
        markup = overview_markup(features,shapely.box(0,0,4,2),4,2,section="theme",theme="potential")
        root = ET.fromstring(f"<svg>{markup}</svg>")
        paths = root.findall(".//path")
        self.assertEqual([path.get("data-band") for path in paths],["1","5"])
        actual = [decoder.path_geometry(path.get("d")) for path in paths]
        self.assertTrue(shapely.union_all(actual).equals(shapely.box(0,0,4,2)))
        self.assertEqual(before,[(feature.geometry.wkb,feature.attributes) for feature in features])

    def test_native_restored_hole_cannot_cross_neighbouring_summary_chord(self):
        # Reduced from a real vegetation face: preserving a hole's source
        # segment made it cross the outer ring's otherwise valid summary.
        fixture = json.loads((Path(__file__).parent / "fixtures" / "native-restored-hole.json").read_text())
        source = fixture["original"]
        middle = shapely.Polygon(source["exterior"], source["interiors"])
        frame = shapely.box(1735,667,1746,680)
        original = np.asarray([frame.difference(shapely.Polygon(middle.exterior)),
                               middle, shapely.Polygon(middle.interiors[0])], dtype=object)
        summary = shapely.coverage_simplify(original,fixture["tolerance"],simplify_boundary=False)
        result = _protect_native_coverage(original,summary,tuple(fixture["frameShape"]))
        self.assertTrue(np.all(shapely.is_valid(result)))
        self.assertTrue(shapely.coverage_is_valid(result))
        self.assertTrue(shapely.union_all(result).equals(frame))
        xx,yy = np.meshgrid(np.arange(1735,1746)+.5,np.arange(667,680)+.5)
        for before, after in zip(original,result,strict=True):
            self.assertTrue(np.array_equal(shapely.intersects_xy(before,xx,yy),
                                          shapely.intersects_xy(after,xx,yy)))


if __name__ == "__main__":
    unittest.main()
