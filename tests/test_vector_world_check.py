"""Reject missing source evidence and detached or ambiguous delivered geometry."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import shapely


spec = importlib.util.spec_from_file_location("vector_world_check",
    Path(__file__).resolve().parents[1] / "scripts/check_vector_world.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def crossing_fixture():
    order = np.zeros((3, 3), dtype=np.uint8)
    order[:, 1] = 2
    flow = np.full((3, 3), -1, dtype=np.int32)
    flow[0, 1], flow[1, 1] = 4, 7
    grid = SimpleNamespace(shape=order.shape, river_order=order, flow_to=flow, water=np.zeros(order.shape, dtype=np.uint8))
    route = SimpleNamespace(identifier="road-1", mode="road", importance="trunk", path=((1., 1.5), (2., 1.5)))
    bridge = SimpleNamespace(identifier="bridge-1", row=1, column=1, route_identifier="road-1", river_order=2)
    society = SimpleNamespace(transport=SimpleNamespace(routes=(route,), bridges=(bridge,)))
    document = {"schema": "shared-road-river-crossings-v1",
        "endpointTouches": [], "bankBoundaryRoundoff": [],
        "channelGeometry": json.loads(shapely.to_geojson(shapely.box(1.4, .4, 1.6, 2.6))),
        "roadGeometry": json.loads(shapely.to_geojson(shapely.MultiLineString([route.path]))),
        "drawnTransportPaths":[{"mode":"road","importance":"trunk","geometry":json.loads(shapely.to_geojson(shapely.LineString(route.path)))}],
        "preparedRoadGeometry": [{"identifier": "road-1", "geometry":
                                  json.loads(shapely.to_geojson(shapely.MultiLineString([route.path])))}],
        "riverGeometry": json.loads(shapely.to_geojson(shapely.MultiLineString([[(1.5, .5), (1.5, 2.5)]]))),
        "checks": {"bridgesOffRoad": 0, "bridgesOffRiver": 0},
        "bridges": [{"identifier": "bridge-1", "routeIdentifier": "road-1", "position": [1.5, 1.5], "tangent": [1., 0.],
            "bankSpan": {"geometry": json.loads(shapely.to_geojson(shapely.LineString(((1.4, 1.5), (1.6, 1.5))))),
                         "bankPoints": [[1.4, 1.5], [1.6, 1.5]],
                         "bankPortals": [{"position": [1.4, 1.5], "sourceRoadIds": ["road-1"]},
                                         {"position": [1.6, 1.5], "sourceRoadIds": ["road-1"]}]}}],
        "crossings": [{"id": "bridge-1", "kind": "bridge", "sourceRoadIds": ["road-1"],
            "nativePoint": [1.5, 1.5], "displayPoint": [1.5, 1.5], "riverCell": [1, 1], "riverOrder": 2,
            "sourceBankTransitions":[{"routeIdentifier":"road-1","nativeDirections":[[-1.,0.],[1.,0.]],
                                      "bankPoints":[[1.4,1.5],[1.6,1.5]]}],
            "evidence": "recorded-bridge-and-native-flow-crossing"}]}
    return grid, society, document


class VectorWorldCheckTests(unittest.TestCase):
    def test_large_coordinate_banks_are_closed_by_one_shared_noding_operation(self):
        river = shapely.LineString(((916., 242.5), (915.5, 242.5), (914.5, 241.5),
                                    (913.5, 242.5), (912.5, 241.5), (912., 242.)))
        road = shapely.LineString(((913.5, 243.5), (913.5, 241.5)))
        point = shapely.Point(913.5, 242.5)
        distance = road.project(point)
        samples = (road.interpolate(distance - .025), road.interpolate(distance + .025))
        self.assertEqual(check._bank_count(river, point, samples), 2)
        for shift in ((-900., -240.), (170., 310.)):
            shifted_river = shapely.LineString(np.asarray(river.coords) + shift)
            shifted_point = shapely.Point(np.asarray(point.coords[0]) + shift)
            shifted_samples = tuple(shapely.Point(np.asarray(sample.coords[0]) + shift) for sample in samples)
            self.assertEqual(check._bank_count(shifted_river, shifted_point, shifted_samples), 2)

    def test_explicit_single_bank_city_endpoint_is_replayed(self):
        grid, society, document = crossing_fixture()
        route = SimpleNamespace(identifier="road-access", mode="road", path=((.5, .5), (1.5, .5)),
                                source_settlement_id="town-bank", target_settlement_id="town-river")
        society.transport.routes += (route,)
        document["preparedRoadGeometry"].append({"identifier": route.identifier, "geometry":
            json.loads(shapely.to_geojson(shapely.MultiLineString([route.path])))})
        society.settlements = (SimpleNamespace(identifier="town-bank", row=0, column=0),
                              SimpleNamespace(identifier="town-river", row=0, column=1))
        display_access = ((.5, .5), (1.3, .3), (1.5, .4), (1.5, .5))
        document["roadGeometry"] = json.loads(shapely.to_geojson(shapely.MultiLineString(
            [society.transport.routes[0].path, display_access])))
        document["endpointTouches"] = [{"kind": "same-bank-river-access", "nativePoint": [1.5, .5],
            "displayPoint": [1.5, .5], "sourceRoadIds": ["road-access"],
            "sourceSettlementIds": ["town-river"], "nativeBankCount": 1, "displayBankCount": 1,
            "bankPoints": [[1.5, .4]], "channelAccessGeometry": json.loads(shapely.to_geojson(
                shapely.LineString(((1.5, .4), (1.5, .5)))))}]
        result = check.check_crossings(document, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["checkedEndpointTouches"], 1)
        for field, value in (("sourceRoadIds", ["road-1"]), ("sourceSettlementIds", ["town-bank"]),
                             ("displayPoint", [1.5, .50001]), ("nativeBankCount", 0)):
            bad = copy.deepcopy(document)
            bad["endpointTouches"][0][field] = value
            result = check.check_crossings(bad, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["endpointTouchMismatches"], 1)

    def test_city_endpoint_with_two_incident_banks_needs_a_real_bridge(self):
        grid, society, document = crossing_fixture()
        west = SimpleNamespace(identifier="road-west", mode="road", path=((.5, 1.), (1.5, 1.5)),
                               source_settlement_id="west-town", target_settlement_id="river-town")
        east = SimpleNamespace(identifier="road-east", mode="road", path=((1.5, 1.5), (2.5, 1.)),
                               source_settlement_id="river-town", target_settlement_id="east-town")
        society.transport.routes = (west, east)
        society.settlements = (SimpleNamespace(identifier="west-town", row=0, column=0),
                              SimpleNamespace(identifier="river-town", row=1, column=1),
                              SimpleNamespace(identifier="east-town", row=0, column=2))
        roads = shapely.MultiLineString([west.path, east.path])
        rivers = shapely.from_geojson(json.dumps(document["riverGeometry"]))
        records = [{"kind":"same-bank-river-access", "nativePoint":[1.5,1.5], "displayPoint":[1.5,1.5],
                    "sourceRoadIds":["road-east","road-west"], "sourceSettlementIds":["river-town"],
                    "nativeBankCount":1,"displayBankCount":1}]
        licensed, errors = check.check_endpoint_touches(records, roads, rivers, rivers, society, 3)
        self.assertFalse(licensed)
        self.assertEqual((errors[0]["nativeBankCount"], errors[0]["displayBankCount"]), (2, 2))

    def test_missing_required_artifact_is_not_empty_success(self):
        with tempfile.TemporaryDirectory() as directory:
            world = Path(directory)
            for name in check.REQUIRED_FILES:
                path = world / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            (world / "thematic/wetland-support.npz").unlink()
            with self.assertRaisesRegex(ValueError, "missing required.*wetland-support"):
                check.required_files(world)

    def test_preserved_source_hash_is_measured_instead_of_trusting_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            original, world = Path(directory) / "original", Path(directory) / "rebuilt"
            for root in (original, world):
                (root / "grid").mkdir(parents=True)
                (root / "grid/world-grid.npz").write_bytes(b"accepted physical grid")
            (world / "regeneration.json").write_text(json.dumps({"schema": "accepted-world-v2",
                "status": "complete", "rebuildKind": "human-only-v1", "sourceWorld": str(original)}))
            (world / "grid/world-grid.npz").write_bytes(b"different physical grid")
            with self.assertRaisesRegex(ValueError, "preserved source changed: grid/world-grid.npz"):
                check.check_sources(world, SimpleNamespace())

    def test_continuous_support_threshold_and_source_replay_are_required(self):
        grid = SimpleNamespace(shape=(2, 3), water=np.array([[0, 0, 0], [0, 0, 1]], dtype=np.uint8))
        support = np.array([[.3, .6, 0.], [0., 0., 0.]], dtype=np.float32)
        mask = support >= .5
        thematic = SimpleNamespace(physiography=SimpleNamespace(wetland_support=support, wetland=mask))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wetland-support.npz"
            np.savez(path, support=support, mask=mask)
            metric = check.check_wetland_field(path, grid, thematic)
            self.assertEqual(metric["wetlandCenters"], 1)
            self.assertEqual(metric["fractionalSupportCenters"], 2)
            np.savez(path, support=support, mask=~mask)
            with self.assertRaisesRegex(ValueError, "mask differs"):
                check.check_wetland_field(path, grid, thematic)
            changed = support.copy()
            changed[0, 0] = .4
            np.savez(path, support=changed, mask=mask)
            with self.assertRaisesRegex(ValueError, "differs from physical"):
                check.check_wetland_field(path, grid, thematic)

    def test_native_centre_ownership_catches_land_theft_gaps_overlaps_and_water_paint(self):
        expected = np.array([[0, 1, 2, 0]])
        land = np.array([[True, True, True, False]])
        good = {0: shapely.box(0, 0, 1, 1), 1: shapely.box(1, 0, 2, 1), 2: shapely.box(2, 0, 3, 1)}
        metric = check.owner_metrics(expected, land, good)
        self.assertEqual((metric["checkedCenters"], metric["mismatchCount"], metric["waterPaintCount"]), (4, 0, 0))
        for replacement in (shapely.box(1, 0, 3, 1), shapely.box(1.6, 0, 2, 1), shapely.box(1, 0, 4, 1)):
            bad = {**good, 1: replacement}
            metric = check.owner_metrics(expected, land, bad)
            self.assertGreater(metric["mismatchCount"] + metric["waterPaintCount"], 0)

    def test_detail_tiles_check_all_native_centres_and_require_the_actual_file(self):
        fixture_spec = importlib.util.spec_from_file_location("vector_city_fixture", Path(__file__).with_name("test_city_coverage_check.py"))
        fixture_module = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixture_module)
        grid = SimpleNamespace(shape=(2, 10), water=np.zeros((2, 10), dtype=np.uint8))
        society = SimpleNamespace(politics=SimpleNamespace(state_id=np.ones(grid.shape, dtype=np.int16)),
                                  provinces=SimpleNamespace(province_id=np.ones(grid.shape, dtype=np.int16)))
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory)
            fixture_module.published_city_fixture(review)
            manifest = check._tiles.read_manifest(review)
            result = check.check_partitions(review, manifest, grid, society)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["countries"]["checkedCenters"], 20)
            path = review / "tiles/detail/0-0.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["themes"]["political"] = ('<g clip-path="url(#tile-land-detail-0-0)">'
                '<path data-state="1" d="M0,0h9v2h-9Z" fill="#abc"/>'
                '<path data-state="2" d="M9,0h1v2h-1Z" fill="#def"/></g>')
            path.write_text(json.dumps(document), encoding="utf-8")
            result = check.check_partitions(review, manifest, grid, society)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["countries"]["mismatchCount"], 2)
            self.assertEqual(result["provinces"]["mismatchCount"], 0)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                check.check_partitions(review, manifest, grid, society)

    def test_bridge_must_touch_both_delivered_lines_even_when_recorded_checks_are_zero(self):
        grid, society, document = crossing_fixture()
        self.assertEqual(check.check_crossings(document, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})["status"], "ok")
        for geometry in ("roadGeometry", "riverGeometry"):
            bad = copy.deepcopy(document)
            bad[geometry] = json.loads(shapely.to_geojson(shapely.MultiLineString([[(0., 0.), (.5, .5)]])))
            report = check.check_crossings(bad, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["mismatchCount"], 1)

    def test_missing_or_silently_generated_facility_identity_fails(self):
        grid, society, document = crossing_fixture()
        for collection, field in (("bridges", "identifier"), ("crossings", "id")):
            bad = copy.deepcopy(document)
            bad[collection][0][field] = "new-render-only-bridge"
            with self.assertRaisesRegex(ValueError, "identities differ"):
                check.check_crossings(bad, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})
        document["bridges"] = []
        with self.assertRaisesRegex(ValueError, "identities differ"):
            check.check_crossings(document, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})

    def test_source_crossing_cannot_borrow_an_unrelated_native_river(self):
        grid, society, document = crossing_fixture()
        document["crossings"][0]["nativePoint"] = [1., 1.5]
        report = check.check_crossings(document, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["maximumNativeRiverDistance"], .5)

    def test_every_source_road_in_shared_facility_proof_must_reach_native_point(self):
        grid, society, document = crossing_fixture()
        unrelated = SimpleNamespace(identifier="road-unrelated", mode="road", path=((.5, .5), (1., .5)))
        society.transport.routes += (unrelated,)
        document["preparedRoadGeometry"].append({"identifier": unrelated.identifier, "geometry":
            json.loads(shapely.to_geojson(shapely.MultiLineString([unrelated.path])))})
        document["crossings"][0]["sourceRoadIds"].append("road-unrelated")
        result = check.check_crossings(document, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["mismatchCount"], 1)
        self.assertGreater(result["maximumNativeRoadDistance"], .5)

    def test_unlicensed_crossings_and_roads_following_the_channel_fail(self):
        grid, society, document = crossing_fixture()
        for extra in ([[(.5, 1.), (2.5, 1.)]], [[(1.5, 1.5), (1.5, 2.)]]):
            bad = copy.deepcopy(document)
            bad["roadGeometry"] = json.loads(shapely.to_geojson(shapely.MultiLineString([society.transport.routes[0].path, *extra])))
            report = check.check_crossings(bad, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})
            self.assertEqual(report["status"], "failed")
            self.assertGreater(report["unlicensedCrossings"] + report["roadRiverOverlapLength"], 0)

    def test_parallel_water_road_fails_without_touching_the_centreline(self):
        grid, society, document = crossing_fixture()
        document["roadGeometry"] = json.loads(shapely.to_geojson(shapely.MultiLineString(
            [society.transport.routes[0].path, ((1.45,1.8),(1.45,2.4))])))
        report = check.check_crossings(document, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})
        self.assertEqual(report["unlicensedCrossings"], 0)
        self.assertEqual(report["roadRiverOverlapLength"], 0)
        self.assertEqual(report["status"], "failed")
        self.assertAlmostEqual(report["channelInk"]["unlicensedChannelRoadLength"], .6)

    def test_actual_bank_boundary_road_is_dry_bank_access(self):
        grid, society, document = crossing_fixture()
        document["roadGeometry"] = json.loads(shapely.to_geojson(shapely.MultiLineString(
            [society.transport.routes[0].path, ((1.4,1.8),(1.4,2.4))])))
        report = check.check_crossings(document, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["channelInk"]["unlicensedChannelRoadLength"], 0)

    def test_bank_limits_require_exact_original_edge_and_scale_ulp_bounds(self):
        channel = shapely.box(914.5,620.,915.5,622.)
        x = np.nextafter(914.5,np.inf)
        line = shapely.LineString(((x,620.25),(x,621.75)))
        edge = [[914.5,620.],[914.5,622.]]
        bound = 16*float(np.spacing(914.5))
        record = {"geometry":json.loads(shapely.to_geojson(line)),"bankSegment":edge,
                  "ulpBound":bound,"maxDistance":x-914.5}
        records,errors,maximum = check.check_bank_roundoff([record],channel,line)
        self.assertFalse(errors)
        self.assertEqual(maximum,x-914.5)
        unsafe,length,count = check._unsafe_water_segments(line,records)
        self.assertEqual((unsafe,length,count),(0.,1.5,1))
        for field,value in (("bankSegment",[[914.5,620.1],[914.5,621.9]]),
                            ("ulpBound",.0001),("maxDistance",0.)):
            bad = {**record,field:value}
            records,errors,_maximum = check.check_bank_roundoff([bad],channel,line)
            self.assertTrue(errors)
            self.assertFalse(records)
        cut = shapely.LineString(((914.5207,620.25),(914.5207,621.75)))
        self.assertEqual(check._unsafe_water_segments(cut,[record])[0],1.5)
        self.assertEqual(check._unsafe_water_segments(line,[])[0],1.5)

    def test_shore_certificates_cannot_cover_a_bent_water_chord(self):
        channel = shapely.box(914.5,620.,915.5,622.)
        bent = shapely.LineString(((914.5,620.25),(914.5207,621.),(914.5,621.75)))
        record = {"geometry":json.loads(shapely.to_geojson(bent)),
                  "bankSegment":[[914.5,620.],[914.5,622.]],
                  "ulpBound":16*float(np.spacing(914.5)),"maxDistance":0.}
        records,errors,_maximum = check.check_bank_roundoff([record],channel,bent)
        self.assertTrue(errors)
        self.assertFalse(records)

    def test_each_native_member_requires_its_source_bank_transition(self):
        grid,society,document = crossing_fixture()
        for corruption in ("same-source-bank","opposite-direction-order","single-bank-interior","missing-member"):
            bad = copy.deepcopy(document)
            members = bad["crossings"][0]["sourceBankTransitions"]
            if corruption=="same-source-bank":
                members[0]["nativeDirections"] = [[-1.,0.],[-1.,0.]]
            elif corruption=="opposite-direction-order":
                members[0]["nativeDirections"].reverse()
            elif corruption=="single-bank-interior":
                members[0]["nativeDirections"].pop()
                members[0]["bankPoints"].pop()
            else:
                members.clear()
            report = check.check_crossings(bad,grid,society,raw_elevation_m=np.ones(grid.shape),locations={})
            self.assertEqual(report["status"],"failed")
            self.assertGreater(report["nativeBankTransitions"]["mismatchCount"],0)
        bad = copy.deepcopy(document)
        del bad["crossings"][0]["sourceBankTransitions"]
        with self.assertRaises(KeyError):
            check.check_crossings(bad,grid,society,raw_elevation_m=np.ones(grid.shape),locations={})

    def test_shared_bridge_overall_two_banks_cannot_hide_one_member_same_bank(self):
        grid,society,document = crossing_fixture()
        other = SimpleNamespace(identifier="road-2",mode="road",path=((1.,1.5),(2.,1.5)))
        society.transport.routes += (other,)
        document["preparedRoadGeometry"].append({"identifier":"road-2","geometry":
            json.loads(shapely.to_geojson(shapely.MultiLineString([other.path])))})
        document["crossings"][0]["sourceRoadIds"].append("road-2")
        member = copy.deepcopy(document["crossings"][0]["sourceBankTransitions"][0])
        member["routeIdentifier"] = "road-2"
        document["crossings"][0]["sourceBankTransitions"].append(member)
        for portal in document["bridges"][0]["bankSpan"]["bankPortals"]:
            portal["sourceRoadIds"].append("road-2")
        report = check.check_crossings(document,grid,society,raw_elevation_m=np.ones(grid.shape),locations={})
        self.assertEqual(report["status"],"ok")
        member["bankPoints"] = [[1.4,1.5],[1.4,1.5]]
        report = check.check_crossings(document,grid,society,raw_elevation_m=np.ones(grid.shape),locations={})
        self.assertEqual(report["nativeBankTransitions"]["status"],"ok")
        self.assertEqual(report["channelInk"]["invalidSpans"],0)
        self.assertEqual(report["displayBankTransitions"]["mismatchCount"],1)
        self.assertEqual(report["status"],"failed")

    def test_native_overlap_replays_remote_entry_and_exit_banks(self):
        route = SimpleNamespace(identifier="overlap-road",path=((1.,0.),(2.,0.),(2.,3.),(3.,3.)))
        river = shapely.MultiLineString([((2.,-1.),(2.,4.))])
        records = [{"id":"bridge-overlap","sourceRoadIds":[route.identifier],"nativePoint":[2.,0.],
            "sourceBankTransitions":[{"routeIdentifier":route.identifier,
                                      "nativeDirections":[[-1.,0.],[1.,0.]],"bankPoints":[[1.9,0.],[2.1,0.]]}]}]
        report = check.check_native_bank_transitions(records,{route.identifier:route},20,river)
        self.assertEqual(report["status"],"ok")
        # The station's following road tangent lies on the river; the source
        # crossing must instead preserve the actual remote off-channel bank.
        records[0]["sourceBankTransitions"][0]["nativeDirections"][1] = [-1.,0.]
        report = check.check_native_bank_transitions(records,{route.identifier:route},20,river)
        self.assertEqual(report["status"],"failed")

    def test_endpoint_star_member_is_proven_as_one_native_bank(self):
        route = SimpleNamespace(identifier="endpoint-road",path=((1.,0.),(2.,0.)))
        river = shapely.MultiLineString([((2.,-1.),(2.,4.))])
        records = [{"id":"bridge-star","sourceRoadIds":[route.identifier],"nativePoint":[2.,0.],
            "sourceBankTransitions":[{"routeIdentifier":route.identifier,
                                      "nativeDirections":[[-1.,0.]],"bankPoints":[[1.9,0.]]}]}]
        report = check.check_native_bank_transitions(records,{route.identifier:route},20,river)
        self.assertEqual(report["status"],"ok")
        records[0]["sourceBankTransitions"][0]["nativeDirections"] = [[1.,0.]]
        self.assertEqual(check.check_native_bank_transitions(records,{route.identifier:route},20,river)["status"],"failed")

    def test_actual_tiny_display_bend_has_two_local_banks_before_the_next_bend(self):
        fixture = json.loads((Path(__file__).with_name("fixtures")/"display-bank-tiny-bend-v97.json").read_text())
        crossing = fixture["crossing"]
        point = shapely.Point(crossing["displayPoint"])
        member = crossing["sourceBankTransitions"][0]
        banks = np.asarray(member["bankPoints"])
        river = shapely.from_geojson(json.dumps(fixture["riverGeometry"]))
        vectors = banks-np.asarray(point.coords[0])
        samples = shapely.points(np.asarray(point.coords[0])+.025*vectors/np.linalg.norm(vectors,axis=1)[:,None])
        # This physical deck is shorter than .004 cells; a .1 audit circle
        # includes the next bend and joins its two banks around that bend.
        self.assertEqual(check._bank_count(river,point,samples),1)
        geometry = shapely.MultiLineString([[banks[0],point.coords[0],banks[1]]])
        document = {"crossings":[crossing],"preparedRoadGeometry":[{"identifier":member["routeIdentifier"],
            "geometry":json.loads(shapely.to_geojson(geometry))}],
            "bridges":[{"identifier":crossing["id"],"bankSpan":{"bankPortals":[
                {"position":bank.tolist(),"sourceRoadIds":[member["routeIdentifier"]]}for bank in banks]}}]}
        self.assertEqual(check.check_display_bank_transitions(document,river)["status"],"ok")

    def test_span_noding_residual_requires_whole_actual_water_edge_proof(self):
        original = shapely.LineString(((971.,651.),(972.,652.)))
        first = (971.0242470620292,651.0242470620293)
        last = (971.0242470520292,651.0242470520292)
        residual = shapely.LineString((first,last))
        unsafe,length,count = check._unsafe_segments_against_edges(residual,[original])
        self.assertEqual(unsafe,0)
        self.assertAlmostEqual(length,residual.length)
        self.assertEqual(count,1)
        moved = shapely.LineString(((first[0]+.0207,first[1]),(last[0]+.0207,last[1])))
        self.assertEqual(check._unsafe_segments_against_edges(moved,[original])[0],moved.length)

    def test_bank_portal_requires_its_own_licensed_named_road(self):
        grid, society, document = crossing_fixture()
        for corruption in ("unlicensed", "missing-terminal", "named-road-misses-portal"):
            bad = copy.deepcopy(document)
            payload = bad["bridges"][0]["bankSpan"]
            if corruption == "unlicensed":
                payload["bankPortals"][0]["sourceRoadIds"] = ["road-other"]
            elif corruption == "missing-terminal":
                payload["bankPortals"].pop()
            else:
                bad["preparedRoadGeometry"][0]["geometry"] = json.loads(shapely.to_geojson(
                    shapely.MultiLineString([((1.,1.5),(1.,0.),(2.,0.),(2.,1.5))])))
            report = check.check_crossings(bad, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})
            self.assertEqual(report["status"], "failed")
            self.assertGreater(report["channelInk"]["invalidSpans"], 0)

    def test_actual_channel_and_span_evidence_are_mandatory(self):
        grid, society, document = crossing_fixture()
        for field in ("channelGeometry", "preparedRoadGeometry"):
            bad = copy.deepcopy(document)
            del bad[field]
            with self.assertRaises(KeyError):
                check.check_crossings(bad, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})
        for field in ("geometry", "bankPoints", "bankPortals"):
            bad = copy.deepcopy(document)
            del bad["bridges"][0]["bankSpan"][field]
            with self.assertRaises(KeyError):
                check.check_crossings(bad, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})

    def test_channel_following_or_wide_permission_cannot_masquerade_as_a_deck(self):
        grid, society, document = crossing_fixture()
        for coordinates in (((1.4,1.5),(1.5,1.8),(1.6,1.5)),
                            ((1.5,.4),(1.5,1.5),(1.5,2.6)),
                            ((1.,1.5),(1.5,1.5),(2.,1.5))):
            bad = copy.deepcopy(document)
            bad["bridges"][0]["bankSpan"]["geometry"] = json.loads(shapely.to_geojson(
                shapely.LineString(coordinates)))
            report = check.check_crossings(bad, grid, society, raw_elevation_m=np.ones(grid.shape),locations={})
            self.assertEqual(report["status"], "failed")
            self.assertGreater(report["channelInk"]["invalidSpans"], 0)

    def test_confluence_bridge_can_have_three_straight_arms(self):
        _grid, _society, document = crossing_fixture()
        point = shapely.Point(0,0)
        rivers = shapely.MultiLineString([((0,0),(0,3)),((0,0),(-3,-3)),((0,0),(3,-3))])
        channel = rivers.buffer(.1)
        directions = check._bank_directions(rivers,point,shapely.STRtree(shapely.get_parts(rivers)))
        arms, roads, banks = [], [], []
        for direction in directions:
            ray = shapely.LineString((point.coords[0],direction*4))
            span = ray.intersection(channel)
            banks.append(list(span.coords[-1]))
            arms.append(span)
            roads.append(ray)
        network = shapely.MultiLineString(roads)
        geometry = shapely.line_merge(network.intersection(channel).difference(channel.boundary))
        document["channelGeometry"] = json.loads(shapely.to_geojson(channel))
        document["crossings"][0]["displayPoint"] = [0,0]
        document["bridges"][0]["position"] = [0,0]
        document["bridges"][0]["bankSpan"] = {"geometry":json.loads(shapely.to_geojson(geometry)),
            "bankPoints":banks,"bankPortals":[{"position":bank,"sourceRoadIds":["road-1"]}for bank in banks]}
        document["preparedRoadGeometry"][0]["geometry"] = json.loads(shapely.to_geojson(network))
        report = check.check_channel_ink(document,network,rivers,valid_bridges={"bridge-1"},valid_access_points=[])
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["invalidSpans"], 0)
        self.assertEqual(report["unlicensedChannelRoadLength"], 0)
        two_arm = shapely.line_merge(shapely.MultiLineString(roads[:2]).intersection(channel).difference(channel.boundary))
        reasons = check._channel_deck(two_arm, point, [shapely.Point(bank)for bank in banks[:2]],
                                      channel, network, directions, access=False)
        self.assertFalse(reasons)

    def test_delivered_roads_and_native_close_bridge_match_with_exact_cancelled_translations(self):
        _grid, _society, document = crossing_fixture()
        document["roadGeometry"] = json.loads(shapely.to_geojson(shapely.MultiLineString([((1,1),(2,2))])))
        document["preparedRoadGeometry"][0]["geometry"]=document["roadGeometry"]
        document['drawnTransportPaths'][0]['geometry']=document['roadGeometry']
        bridge = document["bridges"][0]
        bridge["tangent"] = [np.sqrt(.5),np.sqrt(.5)]
        bridge["bankSpan"]["geometry"] = json.loads(shapely.to_geojson(shapely.LineString(((1.4,1.4),(1.6,1.6)))))
        road = '<path data-route-mode="road" d="M1,1l1,1" fill="none" stroke="#123"/>'
        glyph = ('<g data-bridge-id="bridge-1" data-route-id="road-1" '
            'transform="translate(1.5 1.5)" data-map-x="1.5" data-map-y="1.5" data-base-angle="45">'
            '<g class="bridge-close" style="display:none" transform="translate(-1.5 -1.5)">'
            '<path d="M1.4,1.4L1.6,1.6" fill="none" stroke="#123" vector-effect="non-scaling-stroke"/>'
            '<path class="bridge-physical-span" d="M1.4,1.4L1.6,1.6" fill="none" stroke="#123" vector-effect="non-scaling-stroke"/>'
            '<path d="M1.4,1.4L1.6,1.6" fill="none" stroke="#123" vector-effect="non-scaling-stroke"/>'
            '</g></g>')
        scene = '<svg>'+road+glyph+'</svg>'
        tile = {"rectangle":shapely.box(0,0,3,3),"landClipId":"native-land","key":"detail/0-0",
                "payload":{"ink":'<g clip-path="url(#native-land)">'+road+'</g>'}}
        with patch.object(check._tiles,"read_tile",return_value=tile):
            result = check.check_transport_delivery(Path("unused"),{"rows":1,"columns":1},document,scene,source_routes=_society.transport.routes)
            self.assertEqual(result["status"],"ok")
            for corrupted in (scene.replace('l1,1','l1,.9'),
                              scene.replace('translate(1.5 1.5)','translate(1.5 1.5) rotate(45)'),
                              scene.replace('translate(-1.5 -1.5)','translate(-1.5 -1.49999999)'),
                              scene.replace('M1.4,1.4','M1.4,1.40000001'),
                              scene.replace('data-map-x="1.5"','data-map-x="1.6"'),
                              '<svg><g transform="translate(1 0)">'+road+glyph+'</g></svg>'):
                try:
                    result = check.check_transport_delivery(Path("unused"),{"rows":1,"columns":1},document,corrupted,source_routes=_society.transport.routes)
                    self.assertEqual(result["status"],"failed")
                except ValueError:
                    pass
        bad_tile = copy.deepcopy(tile)
        bad_tile["payload"]["ink"] = bad_tile["payload"]["ink"].replace('l1,1','l1,.9')
        with patch.object(check._tiles,"read_tile",return_value=bad_tile):
            result = check.check_transport_delivery(Path("unused"),{"rows":1,"columns":1},document,scene,source_routes=_society.transport.routes)
            self.assertEqual(result["status"],"failed")
            self.assertEqual(result["examples"][0]["artifact"],"detail/0-0")
        bad_tile["payload"]["ink"] = road
        with patch.object(check._tiles,"read_tile",return_value=bad_tile):
            with self.assertRaisesRegex(ValueError,"exact land clip"):
                check.check_transport_delivery(Path("unused"),{"rows":1,"columns":1},document,scene,source_routes=_society.transport.routes)

    def test_linear_svg_geometry_compares_exact_decimal_segments_not_point_count(self):
        first = check._line_signature(((1,1),(1.5,1.5),(2,2)))
        self.assertEqual(first,check._line_signature(((1,1),(2,2))))
        self.assertEqual(first,check._line_signature(((2,2),(1,1))))
        self.assertNotEqual(first,check._line_signature(((1,1),(1.5,1.500001),(2,2))))
        # Overlay intersection vertices carry double-precision roundoff. A
        # redundant station must not become a false quantized SVG corner.
        source = ((912.1,242.1),(912.3333333333334,242.3333333333333),(913.1,243.1))
        self.assertEqual(check._line_signature(source),check._line_signature((source[0],source[-1])))

    def test_zero_integer_ink_is_empty_but_a_single_integer_segment_is_retained(self):
        self.assertIsNone(check._line_signature(((105.99639198389184,425.9963919838926),
                                                (105.99639198389185,425.9963919838926))))
        self.assertIsNotNone(check._line_signature(((105.,425.),(105.00000001,425.))))

    def test_same_validated_deck_edge_roundoff_needs_no_second_overlay_subtraction(self):
        edge=shapely.LineString(((1030.,652.),(1030.2,652.1)))
        line=shapely.LineString(((1030.05,np.nextafter(652.025,np.inf)),
                                (1030.15,np.nextafter(652.075,np.inf))))
        remainder=line.difference(edge)
        self.assertGreater(remainder.length,.1)
        unsafe,certified,count=check._unsafe_segments_against_edges(remainder,[edge])
        self.assertEqual(unsafe,0)
        self.assertAlmostEqual(certified,remainder.length)
        self.assertEqual(count,1)
        real_water=shapely.LineString(np.asarray(line.coords)+(0.,1e-7))
        self.assertGreater(check._unsafe_segments_against_edges(real_water,[edge])[0],.1)

    def test_noded_shared_bank_fork_is_one_physical_facility_component(self):
        point = shapely.Point(243.5,387.5)
        fork = (243.4985148457,387.4999999967)
        banks = ((243.49,387.5),(243.49,387.4999999934),(243.51,387.5))
        pieces = [((243.51,387.5),point.coords[0],fork),(fork,banks[0]),(fork,banks[1]),
                  (fork,(fork[0]+1e-10,fork[1]+1e-10))]
        geometry = shapely.MultiLineString(pieces)
        channel = shapely.box(243.49,387.4,243.51,387.6)
        rivers = shapely.MultiLineString([((243.5,387.4),(243.5,387.6))])
        directions = check._bank_directions(rivers,point,shapely.STRtree(shapely.get_parts(rivers)))
        reasons = check._channel_deck(geometry,point,[shapely.Point(bank)for bank in banks],
                                      channel,geometry,directions,access=False)
        self.assertFalse(reasons)
        curved = shapely.MultiLineString([*pieces[:1],(fork,(243.493,387.501),banks[0]),*pieces[2:]])
        reasons = check._channel_deck(curved,point,[shapely.Point(bank)for bank in banks],
                                      channel,curved,directions,access=False)
        self.assertTrue(any("curved" in reason or "outside its straight" in reason for reason in reasons))
        disconnected = shapely.MultiLineString([*pieces,((243.501,387.55),(243.502,387.55))])
        reasons = check._channel_deck(disconnected,point,[shapely.Point(bank)for bank in banks],
                                      channel,disconnected,directions,access=False)
        self.assertIn("span contains a disconnected water component",reasons)

    def test_native_point_must_use_the_declared_flow_edge(self):
        grid,society,document = crossing_fixture()
        source = ((1.,.5),(2.,.5),(2.,1.5),(1.,1.5))
        society.transport.routes[0].path = source
        document["preparedRoadGeometry"][0]["geometry"] = json.loads(shapely.to_geojson(shapely.MultiLineString([source])))
        document["crossings"][0]["nativePoint"] = [1.5,.5]
        result = check.check_crossings(document,grid,society,raw_elevation_m=np.ones(grid.shape),locations={})
        self.assertEqual(result["status"],"failed")
        self.assertEqual(result["maximumNativeRoadDistance"],0)
        self.assertEqual(result["maximumNativeRiverDistance"],0)
        self.assertEqual(result["examples"][0]["declaredNativeFlowEdgeDistance"],1)

    def test_old_wetland_name_anchor_cannot_survive_new_physical_source(self):
        grid = SimpleNamespace(metadata={"societyGeneration": {"namingSeed": 17, "namingProfile": "procedural"}})
        feature = SimpleNamespace(identifier="wetland-01", feature_type="wetland", row=1, column=1,
                                  language_identifier=1, tier="secondary")
        moved = copy.copy(feature)
        moved.column = 2
        source = SimpleNamespace(relative_elevation_m=np.ones((3, 3)))
        society = SimpleNamespace(cultures=SimpleNamespace(), geographic_features=(feature,))
        with patch.object(check, "extract_geographic_features", return_value=(moved,)):
            with self.assertRaisesRegex(ValueError, "native physical-feature source replay"):
                check.check_names(grid, source, SimpleNamespace(), society, "<svg/>", [])


if __name__ == "__main__":
    unittest.main()
