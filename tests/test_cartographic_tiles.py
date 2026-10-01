"""Published tile geometry preserves shared topology, curves and native units."""

from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET

import shapely

from world_atlas.core.cartographic_tiles import (
    TileFeature, TileLevel, feature_markup, geometry_path_data, refresh_atlas_tile_themes, write_atlas_tiles,
)


spec = importlib.util.spec_from_file_location(
    "tile_coverage_decoder", Path(__file__).resolve().parents[1] / "scripts" / "check_coastal_coverage.py"
)
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def document(markup):
    return ET.fromstring(f"<svg>{markup}</svg>")


def line_geometry(data):
    """Read published M/l paths independently with exact decimal accumulation."""
    tokens = re.findall(r"[MLl]|[-+]?(?:\d*\.\d+|\d+\.?\d*)", data)
    paths, points, command, index = [], [], None, 0
    current = (Decimal(0), Decimal(0))
    while index < len(tokens):
        if tokens[index] in {"M", "L", "l"}:
            command = tokens[index]
            index += 1
            if command == "M" and points:
                paths.append(points)
                points = []
        x, y = Decimal(tokens[index]), Decimal(tokens[index + 1])
        if command == "l":
            x, y = x + current[0], y + current[1]
        current = (x, y)
        points.append((float(x), float(y)))
        index += 2
    if points:
        paths.append(points)
    return shapely.MultiLineString(paths)


class CartographicTilesTests(unittest.TestCase):
    def test_face_touching_tile_emits_neither_paint_nor_orphan_definition(self):
        pattern='<pattern id="wet" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M0,0h1"/></pattern>'
        feature=TileFeature(shapely.box(31,1,32,2),{"fill":"url(#wet)","clip":"land"},
            "geographic-textures",section="surface",definitions={"wet":pattern})
        with tempfile.TemporaryDirectory()as target:
            _,payloads=self.publish(target,64,32,shapely.box(0,0,64,32),[feature])
            self.assertEqual(len(document(payloads[0]["surface"]).findall('.//pattern')),1)
            second=document(payloads[1]["surface"])
            self.assertFalse(second.findall('.//pattern'))
            self.assertFalse([node for node in second.iter()if node.get("fill","").startswith("url(#wet")])

    def test_only_actual_paint_and_recursive_paint_dependencies_are_published(self):
        definitions={name:f'<pattern id="{name}" width="8" height="8"><path d="M0,0h1" fill="{fill}"/></pattern>'
                     for name,fill in (("visible","url(#dependency)"),("dependency","white"),("unused","blue"))}
        feature=TileFeature(shapely.box(1,1,2,2),{"fill":"url(#visible)","clip":"land"},
            "geographic-textures",section="surface",definitions=definitions)
        with tempfile.TemporaryDirectory()as target:
            _,payloads=self.publish(target,32,32,shapely.box(0,0,32,32),[feature])
            root=document(payloads[0]["surface"])
            self.assertEqual({node.get("id")for node in root.findall('.//pattern')},
                             {"visible-tile-detail-0-0","dependency-tile-detail-0-0"})
            self.assertEqual(root.find('.//pattern[@id="visible-tile-detail-0-0"]/path').get("fill"),
                             "url(#dependency-tile-detail-0-0)")

    def publish(self, target, width, height, land, features):
        manifest = write_atlas_tiles(target, width, height, [TileLevel("detail",4,land,features)])
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((Path(target) / "tiles" / "detail").glob("*.json"))]
        return manifest, payloads

    def test_full_world_manifest_partial_edge_and_empty_ocean_blocks(self):
        with tempfile.TemporaryDirectory() as target:
            manifest, payloads = self.publish(target, 70, 45, shapely.GeometryCollection(), [])
            self.assertEqual(manifest, {"width":70,"height":45,"tileSize":32,
                                       "columns":3,"rows":2,"themes":["none"],"levels":[{"id":"detail","minScale":4}],
                                       "stats":manifest["stats"]})
            self.assertEqual(len(payloads), 6)
            self.assertEqual(payloads[-1]["bounds"], [64,32,6,13])
            self.assertEqual(payloads[-1]["themes"], {"none":""})
            self.assertEqual(payloads[-1]["ink"], "")
            clips = document(payloads[-1]["surface"])
            water = coverage.path_geometry(clips.find(".//clipPath[@id='tile-water-detail-1-2']/path").attrib["d"])
            self.assertTrue(water.equals(shapely.box(64,32,70,45)))
            self.assertEqual(json.loads((Path(target) / "atlas-manifest.json").read_text()), manifest)
            sizes = [path.stat().st_size for path in (Path(target)/"tiles"/"detail").glob("*.json")]
            self.assertEqual(manifest["stats"],{"totalTiles":6,"maxTileBytes":max(sizes),"totalTileBytes":sum(sizes)})

    def test_shared_land_and_water_clips_reassemble_shore_and_lake_holes(self):
        shore = shapely.Polygon([(0,0),(64,0),(64,48),(45.12345,48),(32,39.54321),
                                 (28,54),(8,35),(0,40)], [[(29,20),(35,20),(35,28),(29,28)]])
        island = shapely.box(30,22,31,23)
        land = shapely.union_all([shore, island])
        feature = TileFeature(shapely.box(0,0,64,64), {"fill":"#abcd12"}, "political", "political", "theme")
        before = land.wkb
        with tempfile.TemporaryDirectory() as target:
            _, payloads = self.publish(target, 64, 64, land, [feature])
            decoded_land, decoded_water = [], []
            for payload in payloads:
                surface = document(payload["surface"])
                clips = surface.findall(".//clipPath")
                self.assertEqual(len(clips), 2)
                actual_land, actual_water = [coverage.path_geometry(clip.find("path").attrib["d"]) for clip in clips]
                self.assertLess(actual_land.intersection(actual_water).area, 1e-9)
                x,y,width,height = payload["bounds"]
                self.assertLess(actual_land.union(actual_water).symmetric_difference(shapely.box(x,y,x+width,y+height)).area, 1e-9)
                decoded_land.append(actual_land)
                decoded_water.append(actual_water)
                theme = document(payload["themes"]["political"])
                self.assertEqual(theme.find("g").attrib["clip-path"], f'url(#{clips[0].attrib["id"]})')
            self.assertLess(shapely.union_all(decoded_land).symmetric_difference(land).area, .0002)
            self.assertTrue(shapely.union_all(decoded_land).covers(shapely.Point(30.5,22.5)))
            self.assertFalse(shapely.union_all(decoded_land).covers(shapely.Point(33,25)))
            self.assertEqual(land.wkb, before)

    def test_categorical_partition_has_no_new_tile_edges_or_overlapping_colours(self):
        land = shapely.box(0,0,64,32)
        features = [TileFeature(shapely.box(0,0,31.12345,32), {"fill":"red","stroke":"black","data-state":"1"},
                                "states","political","theme"),
                    TileFeature(shapely.box(31.12345,0,64,32), {"fill":"blue","data-state":"2"},
                                "states","political","theme")]
        with tempfile.TemporaryDirectory() as target:
            _, payloads = self.publish(target, 64, 32, land, features)
            areas = {"1":[],"2":[]}
            for payload in payloads:
                for path in document(payload["themes"]["political"]).iter("path"):
                    self.assertEqual(path.attrib["stroke"], "none")
                    self.assertEqual(path.attrib["fill-rule"], "evenodd")
                    areas[path.attrib["data-state"]].append(coverage.path_geometry(path.attrib["d"]))
            first, second = (shapely.union_all(areas[state]) for state in ("1","2"))
            self.assertEqual(first.intersection(second).area, 0)
            self.assertLess(first.union(second).symmetric_difference(land).area, 1e-9)
            self.assertTrue(first.covers(shapely.Point(31.1,16)))
            self.assertTrue(second.covers(shapely.Point(31.2,16)))

    def test_country_face_touching_tile_corner_keeps_its_interior_city(self):
        # Reduced from the actual state 64 face at settlement-0104. Its
        # concave edge reaches the exact tile corner (352,416). Fast rectangle
        # clipping returned the opposite 4-cell sliver instead of this face's
        # 1019-cell interior, while reporting a valid output polygon.
        country = shapely.Polygon([(358,657),(381.598,419.088),(350.296,420.914),
                                   (352,416),(300.488,348.042)])
        centre = shapely.Point(345.5,422.5)
        rectangle = shapely.box(320,416,352,448)
        expected = country.intersection(rectangle)
        self.assertTrue(country.is_valid)
        self.assertTrue(expected.covers(centre))
        feature = TileFeature(country,{"fill":"#eb9269","data-state":"64"},
                              "states","political","theme")
        with tempfile.TemporaryDirectory() as target:
            self.publish(target,384,448,country,[feature])
            payload = json.loads((Path(target)/"tiles"/"detail"/"13-10.json").read_text())
            surface = document(payload["surface"])
            actual_land = coverage.path_geometry(surface.find(".//clipPath[@id='tile-land-detail-13-10']/path").attrib["d"])
            actual_country = coverage.path_geometry(document(payload["themes"]["political"]).find(".//path").attrib["d"])
            for actual in (actual_land,actual_country):
                self.assertTrue(actual.covers(centre))
                self.assertLess(actual.symmetric_difference(expected).area,1e-6)

    def test_exact_polygon_clip_does_not_draw_boundary_touching_line_remnants(self):
        face = shapely.MultiPolygon([shapely.box(1,1,4,4),shapely.box(32,8,36,16)])
        feature = TileFeature(face,{"fill":"red","stroke":"black"},"states","political","theme")
        with tempfile.TemporaryDirectory() as target:
            _,payloads = self.publish(target,64,32,shapely.box(0,0,64,32),[feature])
            first = document(payloads[0]["themes"]["political"]).findall(".//path")
            self.assertEqual(len(first),1)
            self.assertEqual(first[0].attrib["stroke"],"none")
            self.assertTrue(coverage.path_geometry(first[0].attrib["d"]).equals(shapely.box(1,1,4,4)))

    def test_lines_extend_beyond_tile_cut_without_changed_visible_route(self):
        source = shapely.LineString([(2,4),(30,6),(34,12),(62,20)])
        attributes = {"fill":"none","stroke":"#987654","stroke-width":"1.15",
                      "vector-effect":"non-scaling-stroke","data-route-importance":"trunk"}
        feature = TileFeature(source, attributes, "transport-network")
        before = (source.wkb, dict(attributes))
        with tempfile.TemporaryDirectory() as target:
            _, payloads = self.publish(target, 64, 32, shapely.box(0,0,64,32), [feature])
            visible = []
            for payload in payloads:
                markup = document(payload["ink"])
                self.assertEqual(markup.find("g").attrib["data-tile-layer"], "transport-network")
                line = markup.find(".//path")
                self.assertEqual(line.attrib["vector-effect"], "non-scaling-stroke")
                actual = line_geometry(line.attrib["d"])
                x,y,width,height = payload["bounds"]
                self.assertTrue(actual.bounds[2] > x+width if x == 0 else actual.bounds[0] < x)
                visible.append(actual.intersection(shapely.box(x,y,x+width,y+height)))
            self.assertLess(shapely.union_all(visible).hausdorff_distance(source), 1e-5)
            self.assertEqual((source.wkb, attributes), before)

    def test_exact_quadratic_path_and_native_river_face_survive_tile_seams(self):
        curve = "M30,4 Q32,7 34,4"
        road = TileFeature(shapely.box(30,4,34,7), {"fill":"none","stroke":"#999",
                           "vector-effect":"non-scaling-stroke"}, "transport-network", path_data=curve)
        river = TileFeature(shapely.box(31.999,2,32.001,14), {"fill":"#5799cf",
                            "data-reach-id":"river-0001","data-width-min-m":"37"}, "rivers")
        with tempfile.TemporaryDirectory() as target:
            _, payloads = self.publish(target, 64, 32, shapely.box(0,0,64,32), [road,river])
            channels = []
            for payload in payloads:
                paths = list(document(payload["ink"]).iter("path"))
                self.assertEqual(paths[0].attrib["d"], curve, "Q controls must be retained without sampled-line replacement")
                self.assertEqual(paths[1].attrib["data-width-min-m"], "37")
                channels.append(coverage.path_geometry(paths[1].attrib["d"]))
            self.assertLess(shapely.union_all(channels).symmetric_difference(river.geometry).area, 1e-9)

    def test_straight_curve_envelope_can_have_zero_height(self):
        envelope = shapely.box(2,4,62,4)
        self.assertFalse(envelope.is_valid)
        feature = TileFeature(envelope,{"fill":"none","stroke":"#abc"},"transport-network",
                              path_data="M2,4 L62,4")
        with tempfile.TemporaryDirectory() as target:
            _,payloads = self.publish(target,64,32,shapely.box(0,0,64,32),[feature])
            self.assertEqual(len(payloads),2)
            for payload in payloads:
                self.assertEqual(document(payload["ink"]).find(".//path").attrib["d"],feature.path_data)

    def test_paint_order_repeated_layers_ids_and_clip_references(self):
        features = [TileFeature(shapely.box(0,0,64,32), {"fill":"#aaa","id":"face","clip":"land"}, "elevation-bands", section="surface"),
                    TileFeature(shapely.box(0,0,64,32), {"fill":"url(#frontier-hatch)","aria-label":"海岸 & 湖泊"}, "snow", section="surface"),
                    TileFeature(shapely.box(0,0,64,32), {"fill":"#bbb"}, "elevation-bands", section="surface"),
                    TileFeature(shapely.LineString([(1,2),(63,2)]), {"id":"ink","clip":"water","stroke":"url(#face)"}, "coast")]
        with tempfile.TemporaryDirectory() as target:
            _, payloads = self.publish(target, 64, 32, shapely.box(0,0,64,32), features)
            ids = []
            for column,payload in enumerate(payloads):
                tree = document(payload["surface"] + payload["ink"])
                self.assertEqual([group.attrib["data-tile-layer"] for group in tree if group.tag == "g"],
                                 ["elevation-bands","snow","elevation-bands","coast"])
                face = tree.find(".//path[@id='face-tile-detail-0-%d']" % column)
                self.assertEqual(face.attrib["clip-path"], f"url(#tile-land-detail-0-{column})")
                ink = tree.find(".//path[@id='ink-tile-detail-0-%d']" % column)
                self.assertEqual(ink.attrib["clip-path"], f"url(#tile-water-detail-0-{column})")
                self.assertEqual(ink.attrib["stroke"], f"url(#face-tile-detail-0-{column})")
                self.assertEqual(tree.find(".//path[@aria-label]").attrib["aria-label"], "海岸 & 湖泊")
                self.assertIn('fill="url(#frontier-hatch)"', payload["surface"])
                ids.extend(node.attrib["id"] for node in tree.iter() if "id" in node.attrib)
            self.assertEqual(len(ids), len(set(ids)))

    def test_multi_geometry_uses_one_feature_id_and_relative_rounding_has_no_drift(self):
        polygon = shapely.MultiPolygon([shapely.box(0,0,1,1),shapely.box(2,0,3,1)])
        markup = feature_markup(TileFeature(polygon, {"id":"islands","fill":"#aaa"}, "islands"))
        self.assertEqual(len(document(markup).findall("path")), 1)
        points = [(123.123456 + index * .1234567, 9.987654 - index * .00023456) for index in range(1000)]
        data = geometry_path_data(shapely.LineString(points))
        decoded = line_geometry(data)
        self.assertEqual(tuple(decoded.geoms[0].coords[-1]), tuple(round(value,8) for value in points[-1]))

    def test_published_fine_bank_and_band_boundary_preserve_native_centre_sides(self):
        # Both the neighbouring colour partition and a narrow physical bank
        # must keep nearby points on the same side after SVG serialization.
        boundary = 814.50000032
        land = shapely.box(800,256,832,288)
        lower = shapely.box(800,256,boundary,288)
        upper = shapely.box(boundary,256,832,288)
        bank = shapely.box(boundary,259,boundary + .0000008,260)
        features = [
            TileFeature(lower,{"fill":"#abc","data-band":"lower"},"elevation-bands",section="surface"),
            TileFeature(upper,{"fill":"#bcd","data-band":"upper"},"elevation-bands",section="surface"),
            TileFeature(bank,{"fill":"#5799cf","data-reach-id":"fine-bank"},"rivers"),
        ]
        samples = [shapely.Point(814.5,259.5),shapely.Point(boundary + .00000032,259.5)]
        with tempfile.TemporaryDirectory() as target:
            self.publish(target,832,288,land,features)
            payload = json.loads((Path(target)/"tiles"/"detail"/"8-25.json").read_text())
            surface = document(payload["surface"])
            actual = {
                path.attrib["data-band"]:coverage.path_geometry(path.attrib["d"])
                for path in surface.iter("path") if "data-band" in path.attrib
            }
            actual_bank = coverage.path_geometry(document(payload["ink"]).find(".//path").attrib["d"])
            for source,displayed in ((lower,actual["lower"]),(upper,actual["upper"]),(bank,actual_bank)):
                self.assertEqual([source.covers(point) for point in samples],
                                 [displayed.covers(point) for point in samples])
            self.assertEqual(actual["lower"].intersection(actual["upper"]).area,0)
            self.assertTrue(actual["lower"].union(actual["upper"]).equals(land))

    def test_actual_world_band_edge_does_not_quantize_onto_lower_native_centre(self):
        # Captured from seed 627677218, row 259 / column 814: raw elevation
        # 627.7393188476562 m is below the 627.7402247986869 m band threshold.
        # The nearest source edge is only 1.6826788e-6 native cells away; its
        # lower endpoint used to round onto the centre at five decimals.
        centre = shapely.Point(814.5,259.5)
        first = (814.5000055232416,259.50000000000006)
        last = (814.5000000000001,259.4999982333395)
        previous = (814.25,259.4267139782595)
        land = shapely.box(814.25,259.25,815,259.6)
        upper = shapely.Polygon([(814.25,259.25),(815,259.25),(815,259.5),first,last,previous])
        lower = land.difference(upper)
        self.assertTrue(lower.covers(centre))
        self.assertFalse(upper.covers(centre))
        self.assertTrue(shapely.Polygon([(round(x,5),round(y,5)) for x,y in upper.exterior.coords]).covers(centre))
        features = [
            TileFeature(lower,{"fill":"#abc","data-band":"lower"},"elevation-bands",section="surface"),
            TileFeature(upper,{"fill":"#bcd","data-band":"upper"},"elevation-bands",section="surface"),
        ]
        with tempfile.TemporaryDirectory() as target:
            self.publish(target,832,288,land,features)
            payload = json.loads((Path(target)/"tiles"/"detail"/"8-25.json").read_text())
            bands = {path.attrib["data-band"]:coverage.path_geometry(path.attrib["d"])
                     for path in document(payload["surface"]).iter("path") if "data-band" in path.attrib}
            self.assertTrue(bands["lower"].covers(centre))
            self.assertFalse(bands["upper"].covers(centre))
            self.assertEqual(bands["lower"].intersection(bands["upper"]).area,0)

    def test_valid_polygon_collection_does_not_cancel_overlapping_land(self):
        collection = shapely.GeometryCollection([shapely.box(0,0,8,8),shapely.box(4,4,12,12)])
        self.assertTrue(collection.is_valid)
        markup = feature_markup(TileFeature(collection, {"fill":"#aaa"}, "land"))
        actual = coverage.path_geometry(document(markup).find("path").attrib["d"])
        self.assertTrue(actual.equals(shapely.union_all(collection.geoms)))
        self.assertTrue(actual.covers(shapely.Point(6,6)))

    def test_rejects_invalid_inputs_instead_of_repairing_or_replacing_geometry(self):
        land = shapely.box(0,0,32,32)
        with tempfile.TemporaryDirectory() as target:
            for feature in [TileFeature(land, {}, "terrain", theme="political"),
                            TileFeature(land, {"d":"M0,0Z"}, "terrain"),
                            TileFeature(land, {"clip":"other"}, "terrain"),
                            TileFeature(land, {}, "terrain", section="obsolete")]:
                with self.assertRaises(ValueError):
                    write_atlas_tiles(target,32,32,[TileLevel("detail",4,land,[feature])])

    def test_levels_have_distinct_payloads_and_ids_with_one_shared_theme_contract(self):
        land = shapely.box(0,0,32,32)
        feature = TileFeature(land,{"fill":"#abc","id":"state"},"states","political","theme")
        levels = [TileLevel("regional",4,land,[feature]),TileLevel("local",16,land,[feature]),
                  TileLevel("detail",64,land,[feature])]
        with tempfile.TemporaryDirectory() as target:
            manifest = write_atlas_tiles(target,32,32,levels)
            self.assertNotIn("detailMinScale",manifest)
            self.assertEqual(manifest["levels"],[{"id":"regional","minScale":4},
                                               {"id":"local","minScale":16},{"id":"detail","minScale":64}])
            ids = []
            for level in levels:
                payload = json.loads((Path(target)/"tiles"/level.id/"0-0.json").read_text())
                ids.extend(node.attrib["id"] for node in document(payload["surface"]+payload["themes"]["political"]).iter() if "id" in node.attrib)
            self.assertEqual(len(ids),len(set(ids)))
            for bad in ([levels[0],TileLevel("regional",16,land,[feature])],
                        [levels[0],TileLevel("detail",3,land,[feature])],
                        [levels[0],TileLevel("detail",16,land,[]) ]):
                with self.assertRaises(ValueError):
                    write_atlas_tiles(target,32,32,bad)

    def test_invalid_feature_reports_level_owner_layer_bounds_and_geos_reason(self):
        invalid = shapely.Polygon([(0,0),(8,8),(0,8),(8,0)])
        feature = TileFeature(invalid,{"fill":"#abc"},"states","political","theme")
        with tempfile.TemporaryDirectory() as target:
            with self.assertRaisesRegex(ValueError,
                r"level='detail' index=0 layer='states' theme='political' section='theme' is_valid_reason='Self-intersection.*bounds=\(0.0, 0.0, 8.0, 8.0\)"):
                write_atlas_tiles(target,32,32,[TileLevel("detail",4,shapely.box(0,0,32,32),[feature])])

    def test_single_oversized_json_fails_the_measured_utf8_budget(self):
        land = shapely.box(0,0,32,32)
        feature = TileFeature(land,{"fill":"#abc","aria-label":"海" * 700_000},"land")
        with tempfile.TemporaryDirectory() as target:
            with self.assertRaisesRegex(ValueError,r"tile JSON byte budget exceeded.*limit=2097152"):
                write_atlas_tiles(target,32,32,[TileLevel("detail",4,land,[feature])])

    def test_numeric_working_cover_fills_each_lod_shore_without_changing_other_payloads(self):
        # A valid overview shore can expand the fine coast between native
        # centres. Preclipping numeric paints to the fine coast exposed the
        # base terrain in this stripe at the regional and local scales.
        fine = shapely.box(0,0,20.2,32)
        expanded = shapely.box(0,0,20.4,32)
        old = TileFeature(fine,{"fill":"#abc","clip":"land","data-potential-band":"0"},
                          "theme-fill","potential","theme")
        categorical = TileFeature(shapely.box(0,0,32,32),{"fill":"#cde","data-state":"5"},
                                  "states","political","theme")
        ink = TileFeature(shapely.LineString([(0,7),(31,7)]),{"stroke":"#111","fill":"none"},"coast")
        working = TileFeature(shapely.box(0,0,32,32),dict(old.attributes),old.layer,old.theme,old.section)
        levels = [TileLevel("regional",4,expanded,[old,categorical,ink]),
                  TileLevel("local",16,expanded,[old,categorical,ink]),
                  TileLevel("detail",64,fine,[old,categorical,ink])]
        with tempfile.TemporaryDirectory() as directory:
            source, stage = Path(directory)/"source", Path(directory)/"stage"
            write_atlas_tiles(source,32,32,levels)
            original_manifest = (source/"atlas-manifest.json").read_bytes()
            result = refresh_atlas_tile_themes(source,stage,[working])
            self.assertEqual((source/"atlas-manifest.json").read_bytes(),original_manifest)
            sizes = []
            for level in levels:
                path = Path("tiles")/level.id/"0-0.json"
                original = json.loads((source/path).read_text())
                patched = json.loads((stage/path).read_text())
                self.assertEqual(patched["surface"],original["surface"])
                self.assertEqual(patched["ink"],original["ink"])
                self.assertEqual(patched["bounds"],original["bounds"])
                self.assertEqual(patched["themes"]["political"],original["themes"]["political"])
                self.assertEqual(patched["themes"]["none"],original["themes"]["none"])
                surface = document(patched["surface"])
                clip = coverage.path_geometry(surface.find(".//clipPath[@id='tile-land-%s-0-0']/path" % level.id).attrib["d"])
                theme = document(patched["themes"]["potential"])
                self.assertEqual(theme.find("g").attrib["clip-path"],f"url(#tile-land-{level.id}-0-0)")
                painted = coverage.path_geometry(theme.find(".//path").attrib["d"]).intersection(clip)
                self.assertTrue(painted.equals(clip))
                old_paint = coverage.path_geometry(document(original["themes"]["potential"]).find(".//path").attrib["d"]).intersection(clip)
                self.assertAlmostEqual(clip.difference(old_paint).area,6.4 if level.id != "detail" else 0)
                sizes.append((stage/path).stat().st_size)
            self.assertEqual(result["stats"],{"totalTiles":3,"maxTileBytes":max(sizes),"totalTileBytes":sum(sizes)})

    def test_numeric_tile_refresh_preserves_sparse_city_relief_without_theme_fields(self):
        land = shapely.box(0,0,64,32)
        original = TileFeature(land,{"fill":"#abc","clip":"land"},"theme-fill","potential","theme")
        replacement = TileFeature(land,{"fill":"#def","clip":"land"},"theme-fill","potential","theme")
        with tempfile.TemporaryDirectory() as directory:
            source, stage = Path(directory)/"source", Path(directory)/"stage"
            self.publish(source,64,32,land,[original])
            manifest_path=source/"atlas-manifest.json"
            manifest=json.loads(manifest_path.read_text())
            manifest["levels"].append({"id":"city-detail","minScale":128,"kind":"relief-overlay","publishedTiles":["0-1"]})
            manifest_path.write_text(json.dumps(manifest))
            city=source/"tiles/city-detail/0-1.json"
            city.parent.mkdir()
            encoded=b'{"bounds":[32,0,32,32],"surface":"<defs/>","ink":"<g/>"}'
            city.write_bytes(encoded)
            result=refresh_atlas_tile_themes(source,stage,[replacement])
            self.assertEqual((stage/"tiles/city-detail/0-1.json").read_bytes(),encoded)
            self.assertFalse((stage/"tiles/city-detail/0-0.json").exists())
            sizes=[path.stat().st_size for path in (stage/"tiles").rglob("*.json")]
            self.assertEqual(result["stats"],{"totalTiles":len(sizes),"maxTileBytes":max(sizes),"totalTileBytes":sum(sizes)})

    def test_numeric_tile_refresh_keeps_current_byte_budget_and_requires_a_stage(self):
        land = shapely.box(0,0,32,32)
        original = TileFeature(land,{"fill":"#abc","clip":"land"},"theme-fill","potential","theme")
        oversized = TileFeature(land,{"fill":"#abc","clip":"land","aria-label":"海"*700_000},
                                "theme-fill","potential","theme")
        with tempfile.TemporaryDirectory() as directory:
            source, stage = Path(directory)/"source", Path(directory)/"stage"
            self.publish(source,32,32,land,[original])
            before = (source/"tiles/detail/0-0.json").read_bytes()
            with self.assertRaisesRegex(ValueError,"separate staging directory"):
                refresh_atlas_tile_themes(source,source,[original])
            with self.assertRaisesRegex(ValueError,"tile JSON byte budget exceeded.*limit=2097152"):
                refresh_atlas_tile_themes(source,stage,[oversized])
            self.assertEqual((source/"tiles/detail/0-0.json").read_bytes(),before)

    def test_refreshed_administrative_ink_replaces_owned_groups_and_preserves_roads(self):
        land = shapely.box(0,0,64,32)
        paint = TileFeature(land,{"fill":"#abc","clip":"land"},"theme-fill","potential","theme")
        old = TileFeature(shapely.LineString([(2,0),(2,32)]),{"stroke":"#333"},"state-boundaries")
        road = TileFeature(shapely.LineString([(0,7),(64,7)]),{"stroke":"#999"},"transport-network")
        new = TileFeature(shapely.LineString([(40,0),(40,32)]),{"stroke":"#444","clip":"land"},"state-boundaries")
        with tempfile.TemporaryDirectory() as directory:
            source, stage = Path(directory)/"source", Path(directory)/"stage"
            self.publish(source,64,32,land,[paint,old,road])
            result = refresh_atlas_tile_themes(source,stage,[paint],ink_layers={"state-boundaries":[new]})
            for column in range(2):
                original=json.loads((source/f"tiles/detail/0-{column}.json").read_text())
                refreshed=json.loads((stage/f"tiles/detail/0-{column}.json").read_text())
                self.assertEqual(refreshed["surface"],original["surface"])
                tree=document(refreshed["ink"])
                old_road=document(original["ink"]).find("g[@data-tile-layer='transport-network']")
                self.assertEqual(ET.tostring(tree.find("g[@data-tile-layer='transport-network']")),ET.tostring(old_road))
                border=tree.find("g[@data-tile-layer='state-boundaries']")
                self.assertEqual(border is None,column==0)
                if column==1:
                    self.assertEqual(tree[0].attrib["data-tile-layer"],"state-boundaries")
                    self.assertEqual(border[0].attrib["clip-path"],"url(#tile-land-detail-0-1)")
                    self.assertEqual(border[0].attrib["d"],geometry_path_data(new.geometry))
            self.assertEqual(result["stats"]["totalTiles"],2)


    def test_same_class_fill_batch_retains_exact_rings_and_reduces_dom_nodes(self):
        land=shapely.box(0,0,32,32)
        faces=(shapely.box(1,1,7,7),shapely.Polygon([(9,2),(16,2),(16,9),(9,9)],
            [[(11,4),(14,4),(14,7),(11,7)]]))
        attributes={"fill":"#abc","fill-opacity":".7","clip":"land","data-vegetation-band":"3"}
        features=[TileFeature(face,attributes,"theme-fill","vegetation","theme")for face in faces]
        with tempfile.TemporaryDirectory() as target:
            _,payloads=self.publish(target,32,32,land,features)
            root=document(payloads[0]['themes']['vegetation']);paths=root.findall('.//path')
            self.assertEqual(len(paths),1)
            self.assertEqual(paths[0].get('clip-path'),'url(#tile-land-detail-0-0)')
            painted=coverage.path_geometry(paths[0].get('d'))
            self.assertTrue(painted.equals(shapely.union_all(faces)))
            self.assertFalse(painted.covers(shapely.Point(12,5)))

    def test_overlapping_translucent_fills_keep_separate_paint_operations(self):
        land=shapely.box(0,0,32,32)
        attributes={"fill":"#abc","fill-opacity":".7","clip":"land"}
        features=[TileFeature(face,attributes,"theme-fill","vegetation","theme")
            for face in (shapely.box(1,1,9,9),shapely.box(5,5,13,13))]
        with tempfile.TemporaryDirectory() as target:
            _,payloads=self.publish(target,32,32,land,features)
            self.assertEqual(len(document(payloads[0]['themes']['vegetation']).findall('.//path')),2)

if __name__ == "__main__":
    unittest.main()
