"""Audit a completed vector world from its saved sources and delivered artifacts.

This command does not render, simplify, or repair geometry. Missing evidence
fails the check. Detail-tile partition verification is optional, and a skipped
partition check is never reported as a complete acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time

import numpy as np
import shapely
from shapely.ops import substring

from world_atlas.core.model import WorldGrid
from world_atlas.core.procedural_planet import load_surface_bundle
from world_atlas.core.river_network import hydrologic_outlet_targets
from world_atlas.core.society.fictional_names import procedural_name_lexicon
from world_atlas.core.society.names import extract_geographic_features
from world_atlas.core.society.storage import load_society
from world_atlas.core.society.world_identity import assign_world_identity, naming_audit
from world_atlas.core.thematic import derive_thematic_layers
from world_atlas.core.transport_geometry import native_river_geometry
from world_atlas.settings import load_world_settings


def _consumer(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_landforms = _consumer("check_landform_display")
_geography = _consumer("check_geographic_display")
_tiles = _landforms._tiles
TOLERANCE = 1e-7
SVG_COORDINATE_SCALE = 100_000_000
SOURCE_FILES = ("grid/world-grid.npz", "grid/world-grid.json", "source/physical-fields.npz",
                "source/physical-reference.png", "world-settings.json", "naming-exclusions.json")
REQUIRED_FILES = (*SOURCE_FILES, "worldgen.json", "regeneration.json", "source/provenance.json",
                  "source/physical-fields-verification.json", "society/society.json", "society/society.npz",
                  "thematic/wetland-support.npz", "review/society.json", "review/society.npz",
                  "review/atlas-manifest.json", "review/atlas-scene.svg", "review/landform-regions.svg",
                  "review/physical-surface.svgz", "review/elevation-contours.svgz",
                  "review/transport-crossings.json")
REPLAY_FIELDS = {"plate_id", "plate_velocity_east_cm_per_year", "plate_velocity_north_cm_per_year",
                 "boundary_class", "crust_kind", "ocean_age_myr", "continent_id", "land_mask",
                 "elevation", "bathymetry", "signed_height"}


def _sha256(path):
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256")
    return digest.hexdigest().upper()


def _json(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected an object")
    return document


def required_files(world):
    missing = [name for name in REQUIRED_FILES if not (world / name).is_file()]
    if missing:
        raise ValueError("missing required world evidence: " + ", ".join(missing))


def check_sources(world, grid):
    record = _json(world / "regeneration.json")
    if (record.get("schema") != "accepted-world-v2" or record.get("status") != "complete"
            or record.get("rebuildKind") != "human-only-v1"):
        raise ValueError("vector acceptance requires a completed human-only world rebuild")
    original = Path(record["sourceWorld"]).resolve()
    if original == world.resolve():
        raise ValueError("the preserved source world must be a separate world")
    hashes = {}
    for name in SOURCE_FILES:
        previous, current = _sha256(original / name), _sha256(world / name)
        # Travel capabilities are an explicit derived-generation input. The
        # accepted physical and human settings are checked against the grid
        # below; adding flight must not require rewriting physical metadata.
        if name != 'world-settings.json' and previous != current:
            raise ValueError(f"preserved source changed: {name}")
        hashes[name] = current
    for field, name in (("fieldSha256", "source/physical-fields.npz"),
                        ("sourceSha256", "source/physical-reference.png"),
                        ("settingsSha256", "world-settings.json")):
        if record.get(field, "").upper() != hashes[name]:
            raise ValueError(f"regeneration {field} does not match the preserved file")
    original_grid = WorldGrid.load(original / "grid")
    digest = grid.content_digest()
    if digest != record.get("gridDigest") or digest != original_grid.content_digest():
        raise ValueError("native physical grid digest changed")
    config = _json(world / "worldgen.json")
    if (config["source"]["fieldBundle"] != {"path": "source/physical-fields.npz", "sha256": hashes["source/physical-fields.npz"]}
            or config["source"]["path"] != "source/physical-reference.png"
            or config["source"]["sha256"].upper() != hashes["source/physical-reference.png"]):
        raise ValueError("worldgen source does not identify the saved accepted physical bundle")
    proof = _json(world / "source/physical-fields-verification.json")
    if (proof.get("schema") != "verified-physical-surface-replay-v1"
            or any(proof.get(key) is not True for key in
                   ("verifiedExact", "originalSourceUnchanged", "nativePaletteReconstructionExact"))
            or proof.get("verifiedBundleSha256", "").upper() != hashes["source/physical-fields.npz"]
            or proof.get("sourceBundleSha256", "").upper()
            != grid.metadata["proceduralPhysicalSource"]["fieldBundle"]["sha256"].upper()
            or proof.get("recipe") != _json(world / "source/provenance.json")["recipe"]
            or set(proof.get("comparisons", {})) != REPLAY_FIELDS
            or any(item.get("exact") is not True or item.get("different") != 0
                   for item in proof["comparisons"].values())):
        raise ValueError("saved physical verification does not prove the accepted raw bundle")
    source = load_surface_bundle(world / "source/physical-fields.npz")
    raw = source.relative_elevation_m
    land = grid.water == 0
    if (raw.shape != grid.shape or np.any(~np.isfinite(raw))
            or not np.array_equal(raw > 0, land) or not np.array_equal(source.land_mask, land)):
        raise ValueError("accepted raw metres, land mask and native grid disagree")
    settings = load_world_settings(world / "world-settings.json")
    request = grid.metadata["societyGeneration"]
    if (request["humanSeed"] != settings.human_seed or request["namingSeed"] != settings.naming_seed
            or any(request.get(key) != value for key, value in settings.society.items())
            or request.get("technologyEra") != settings.technology_era
            or any(grid.metadata["planet"].get(key) != value for key, value in settings.planet.items())
            or record.get("humanSeed") != settings.human_seed or record.get("namingSeed") != settings.naming_seed
            or record.get("terrainSeed") != source.diagnostics.get("seed")):
        raise ValueError("preserved generation seeds/profile differ from the native grid")
    return {"status": "ok", "sourceWorld": str(original), "gridDigest": digest,
            "unchangedSHA256": {key:value for key,value in hashes.items() if key!='world-settings.json'},
            "settingsSHA256":hashes['world-settings.json'],"travelCapabilities":list(settings.travel_capabilities),
            "rawLandSignMismatches": 0}, source


def check_wetland_field(path, grid, thematic):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"support", "mask"}:
            raise ValueError("wetland source artifact must contain exactly support and mask")
        support, mask = archive["support"], archive["mask"]
    if (support.shape != grid.shape or support.dtype != np.float32
            or mask.shape != grid.shape or mask.dtype != np.bool_
            or np.any(~np.isfinite(support)) or np.any((support < 0) | (support > 1))):
        raise ValueError("invalid native continuous wetland field")
    if not np.array_equal(mask, support >= .5):
        raise ValueError("wetland mask differs from its .5 continuous-field threshold")
    if (not np.array_equal(support, thematic.physiography.wetland_support)
            or not np.array_equal(mask, thematic.physiography.wetland)):
        raise ValueError("saved wetland field differs from physical/climate/drainage evidence")
    land = grid.water == 0
    if np.any(support[~land] != 0):
        raise ValueError("wetland support exists on water")
    fractional = int(np.count_nonzero((support > 0) & (support < 1)))
    if not fractional:
        raise ValueError("wetland field supplies no continuous transition evidence")
    count, land_count = int(mask.sum()), int(land.sum())
    return {"status": "ok", "threshold": .5, "checkedCenters": int(mask.size),
            "landCenters": land_count, "wetlandCenters": count,
            "landFraction": count / land_count, "fractionalSupportCenters": fractional,
            "uniqueSupportValues": int(np.unique(support).size), "sourceMismatchCount": 0,
            "artifactSHA256": _sha256(path)}


def owner_metrics(expected, land, regions, *, x=0, y=0):
    """Check each native centre once; a border shared by two owners is ambiguous."""
    rows, columns = np.indices(expected.shape)
    points = shapely.points(columns + x + .5, rows + y + .5)
    actual = np.full(expected.shape, -1, dtype=np.int32)
    coverage = np.zeros(expected.shape, dtype=np.int32)
    for owner, geometry in regions.items():
        occupied = shapely.covers(geometry, points)
        coverage += occupied
        actual[occupied] = owner
    mismatch = land & ((coverage != 1) | (actual != expected))
    water_paint = ~land & (coverage != 0)
    return {"checkedCenters": int(expected.size), "landCenters": int(land.sum()),
            "mismatchCount": int(mismatch.sum()), "waterPaintCount": int(water_paint.sum()),
            "ambiguousCenters": int(np.count_nonzero(coverage > 1)),
            "examples": [{"row": int(y + row), "column": int(x + column),
                          "expectedOwner": int(expected[row, column]),
                          "displayedOwner": int(actual[row, column]), "ownerCount": int(coverage[row, column])}
                         for row, column in np.argwhere(mismatch | water_paint)[:8]]}


def check_partitions(review, manifest, grid, society):
    if (manifest["height"], manifest["width"]) != grid.shape:
        raise ValueError("delivered tile dimensions differ from the native grid")
    if not {"political", "provinces"} <= set(manifest["themes"]):
        raise ValueError("delivered detail tiles omit country or province paint")
    result = {"status": "ok", "level": "detail", "checkedTiles": 0,
              "countries": {}, "provinces": {}}
    for name in ("countries", "provinces"):
        result[name] = {"checkedCenters": 0, "landCenters": 0, "mismatchCount": 0,
                        "waterPaintCount": 0, "ambiguousCenters": 0, "examples": []}
    for tile_row in range(manifest["rows"]):
        for tile_column in range(manifest["columns"]):
            tile = _tiles.read_tile(review, manifest, "detail", tile_row, tile_column)
            x, y, width, height = tile["payload"]["bounds"]
            land = grid.water[y:y + height, x:x + width] == 0
            for name, theme, source, attribute in (
                ("countries", "political", society.politics.state_id, "data-state"),
                ("provinces", "provinces", society.provinces.province_id, "data-province"),
            ):
                metric = owner_metrics(source[y:y + height, x:x + width], land,
                                       _tiles.tile_owner_regions(tile, theme, attribute), x=x, y=y)
                for key in ("checkedCenters", "landCenters", "mismatchCount", "waterPaintCount", "ambiguousCenters"):
                    result[name][key] += metric[key]
                result[name]["examples"].extend(metric["examples"][:8 - len(result[name]["examples"])])
            result["checkedTiles"] += 1
    for name in ("countries", "provinces"):
        metric = result[name]
        if metric["checkedCenters"] != grid.water.size:
            raise ValueError("detail tiles did not cover every native centre exactly once")
        if any(metric[key] for key in ("mismatchCount", "waterPaintCount", "ambiguousCenters")):
            result["status"] = "failed"
    return result


def _point(value, description):
    coordinates = np.asarray(value, dtype=float)
    if coordinates.shape != (2,) or np.any(~np.isfinite(coordinates)):
        raise ValueError(f"{description} requires two finite native coordinates")
    return shapely.Point(coordinates)


def _primitives(geometry):
    pending, result = [geometry], []
    while pending:
        part = pending.pop()
        if hasattr(part, "geoms"):
            pending.extend(part.geoms)
        elif not part.is_empty:
            result.append(part)
    return result


def _bank_faces(river, neighborhood):
    # Clip beyond the audit circle so one noding operation constructs every
    # bank/circle junction. Clipping to the circle first creates independent
    # floating-point endpoints and can falsely leave a bank open.
    local_river = shapely.intersection(river, neighborhood.buffer(.01))
    graph = shapely.node(shapely.union_all((neighborhood.boundary, local_river)))
    return shapely.get_parts(shapely.polygonize(shapely.get_parts(graph)))


def _bank_owners(faces, samples):
    return [next((index for index, face in enumerate(faces) if face.contains(sample)), None)
            for sample in samples]


def _bank_count(river, point, samples):
    owners = set(_bank_owners(_bank_faces(river,point.buffer(.1,quad_segs=8)),samples))
    owners.discard(None)
    return len(owners)


def _native_road_geometry(route, width):
    points = np.asarray(route.path)
    parts = np.split(points, np.flatnonzero(np.abs(np.diff(points[:, 0])) > width * .5) + 1)
    return shapely.MultiLineString([part for part in parts if len(part) > 1])


def _incident_angles(point, index):
    """The local ordered straight rays at an accepted river station."""
    angles = []
    for item in index.query(point.buffer(TOLERANCE), predicate="intersects"):
        line = index.geometries[item]
        station = line.project(point)
        for distance in (max(0., station - .0001), min(line.length, station + .0001)):
            sample = line.interpolate(distance)
            vector = np.asarray((sample.x - point.x, sample.y - point.y))
            if np.linalg.norm(vector) > 1e-10:
                angles.append(float(np.arctan2(vector[1], vector[0])))
    angles = sorted(set(round(angle, 10) for angle in angles))
    return angles


def _bank_directions(rivers, point, index):
    """Actual incident channel rays determine straight bank-sector decks."""
    angles = _incident_angles(point,index)
    if not angles:
        return np.empty((0,2))
    directions = []
    for first, last in zip(angles, (*angles[1:], angles[0] + 2 * np.pi), strict=True):
        angle = .5 * (first + last)
        directions.append((np.cos(angle), np.sin(angle)))
    return np.asarray(directions)


def _within_exact_edge_bound(point, edge, bound):
    """Bound distance to a closed original edge using exact binary64 values."""
    first, last = [[Fraction(float(value)) for value in pair] for pair in edge]
    point = [Fraction(float(value)) for value in point]
    vector = [last[index]-first[index] for index in range(2)]
    offset = [point[index]-first[index] for index in range(2)]
    squared = sum(value*value for value in vector)
    if not squared:
        return False
    projection = sum(vector[index]*offset[index] for index in range(2))
    limit = Fraction(float(bound))**2
    if projection <= 0:
        return sum(value*value for value in offset) <= limit
    if projection >= squared:
        return sum((point[index]-last[index])**2 for index in range(2)) <= limit
    cross = vector[0]*offset[1]-vector[1]*offset[0]
    return cross*cross <= limit*squared


def check_bank_roundoff(records, channel, water_road):
    """Recompute every bounded shore certificate against original ring edges.

    Distance to one closed straight edge is convex along a straight segment.
    Exact rational endpoint bounds therefore prove its whole bank limit.
    No physical water permission or fixed map-space buffer is added.
    """
    if not isinstance(records, list):
        raise ValueError("bank-boundary roundoff evidence must be an explicit list")
    edges = set()
    for ring in _primitives(channel.boundary):
        for first, last in zip(ring.coords[:-1], ring.coords[1:]):
            edges.add(tuple(sorted((tuple(first),tuple(last)))))
    certified, errors, maximum = [], [], 0.
    for index, record in enumerate(records):
        geometry = shapely.from_geojson(json.dumps(record["geometry"]))
        pair = np.asarray(geometry.coords) if geometry.geom_type == "LineString" else np.empty((0,2))
        bank = np.asarray(record["bankSegment"], dtype=float)
        if (pair.shape != (2,2) or bank.shape != (2,2) or not np.all(np.isfinite(pair))
                or not np.all(np.isfinite(bank)) or geometry.is_empty or not geometry.is_valid):
            errors.append({"index":index,"reason":"shore certificate must name two finite straight segments"})
            continue
        bound = 16*float(np.spacing(max(1.,float(np.max(np.abs(pair))))))
        original = tuple(sorted(map(tuple,bank))) in edges
        measured = float(shapely.distance(shapely.points(pair),shapely.LineString(bank)).max())
        exact = original and all(_within_exact_edge_bound(point,bank,bound) for point in pair)
        if (record["ulpBound"] != bound or record["maxDistance"] != measured or not exact
                or np.any(shapely.distance(shapely.points(pair),water_road)>bound)):
            errors.append({"index":index,"reason":"shore certificate lacks its exact original edge/ULP proof",
                           "originalEdge":original,"exactEndpointBounds":exact,"recomputedDistance":measured,
                           "recomputedUlpBound":bound})
        else:
            certified.append(record)
            maximum = max(maximum,measured)
    return certified, errors, maximum


def _unsafe_segments_against_edges(geometry, bank_edges):
    """Classify actual water segments without a second floating overlay.

    A union/difference of certified collinear clipped lines can reintroduce
    their original last-bit noding error. Test each actual segment directly
    against the independently validated original bank edges instead.
    """
    index = shapely.STRtree(bank_edges)
    unsafe, certified_length, certified_count = [], 0., 0
    for line in _primitives(geometry):
        if line.geom_type != "LineString":
            continue
        for first,last in zip(line.coords[:-1],line.coords[1:]):
            pair = np.asarray((first,last))
            bound = 16*float(np.spacing(max(1.,float(np.max(np.abs(pair))))))
            candidates = index.query(shapely.box(*(np.min(pair,axis=0)-bound),*(np.max(pair,axis=0)+bound)))
            segment = shapely.LineString(pair)
            if any(all(_within_exact_edge_bound(point,np.asarray(bank_edges[item].coords),bound)
                       for point in pair)for item in candidates):
                certified_length += segment.length
                certified_count += 1
            else:
                unsafe.append(segment)
    return sum(segment.length for segment in unsafe), certified_length, certified_count


def _unsafe_water_segments(geometry, records):
    return _unsafe_segments_against_edges(geometry,[shapely.LineString(record["bankSegment"])for record in records])


def check_native_bank_transitions(crossings, routes, width, native_river):
    """Replay each member's source bank walk, including channel overlaps."""
    index = shapely.STRtree(shapely.get_parts(native_river))
    by_route, errors, count = {}, [], 0
    for crossing in crossings:
        members = crossing["sourceBankTransitions"]
        if (not isinstance(members,list)
                or Counter(member["routeIdentifier"]for member in members)!=Counter(crossing["sourceRoadIds"])):
            errors.append({"identifier":crossing["id"],"reason":"source bank transitions must account for every member exactly once"})
            continue
        native = _point(crossing["nativePoint"],"source bank-transition point")
        for member in members:
            identifier = member["routeIdentifier"]
            directions = np.asarray(member["nativeDirections"],dtype=float)
            banks = member["bankPoints"]
            if (identifier not in routes or directions.ndim!=2 or directions.shape[1]!=2
                    or len(directions) not in {1,2} or len(banks)!=len(directions)
                    or not np.all(np.isfinite(directions))
                    or np.any(np.abs(np.linalg.norm(directions,axis=1)-1.)>1e-6)):
                errors.append({"identifier":crossing["id"],"routeIdentifier":identifier,"reason":"source bank transition has invalid native unit directions"})
                continue
            by_route.setdefault(identifier,[]).append((crossing["id"],native,directions))
            count += 1
    for identifier, records in by_route.items():
        original = _native_road_geometry(routes[identifier],width)
        for road in original.geoms:
            members = [(facility,point,directions)for facility,point,directions in records if point.distance(road)<=TOLERANCE]
            if not members:
                continue
            relevant = index.query(road,predicate="intersects")
            local = shapely.union_all(index.geometries[relevant])
            overlaps = [part for part in _primitives(shapely.line_merge(shapely.union_all(
                [part for part in _primitives(road.intersection(local))if part.geom_type=="LineString"])))
                if part.geom_type=="LineString"]
            handled = set()
            for overlap in overlaps:
                group = [(facility,point,directions)for facility,point,directions in members if point.distance(overlap)<=TOLERANCE]
                if not group:
                    continue
                group.sort(key=lambda item:road.project(item[1]))
                start,end = sorted(road.project(shapely.Point(value))for value in (overlap.coords[0],overlap.coords[-1]))
                neighborhood = substring(road,max(0.,start-.05),min(road.length,end+.05)).buffer(.1,quad_segs=8)
                faces = _bank_faces(local,neighborhood)
                entry = None if start<=TOLERANCE else _bank_owners(faces,[road.interpolate(start-.025)])[0]
                exit_owner = None if road.length-end<=TOLERANCE else _bank_owners(faces,[road.interpolate(end+.025)])[0]
                current, valid = entry, True
                for facility,point,directions in group:
                    handled.add(facility)
                    samples = shapely.points(np.asarray(point.coords[0])+.025*directions)
                    owners = _bank_owners(faces,samples)
                    station = road.project(point)
                    if len(owners)==1:
                        expected = exit_owner if station<=TOLERANCE else entry if road.length-station<=TOLERANCE else None
                        valid &= expected is not None and owners[0]==expected
                    else:
                        valid &= None not in owners and owners[0]!=owners[1] and owners[0]==current
                        current = owners[1]
                if entry is not None and exit_owner is not None:
                    valid &= current==exit_owner
                if not valid:
                    errors.append({"routeIdentifier":identifier,"identifiers":[item[0]for item in group],
                                   "reason":"native channel-overlap transitions do not replay the source entry-to-exit bank walk"})
            for facility,point,directions in members:
                if facility in handled:
                    continue
                station = float(road.project(point))
                source_samples = []
                if station>TOLERANCE:
                    source_samples.append(road.interpolate(max(0.,station-.025)))
                if road.length-station>TOLERANCE:
                    source_samples.append(road.interpolate(min(road.length,station+.025)))
                faces = _bank_faces(local,point.buffer(.1,quad_segs=8))
                expected = _bank_owners(faces,source_samples)
                observed = _bank_owners(faces,shapely.points(np.asarray(point.coords[0])+.025*directions))
                if (len(expected)!=len(directions) or None in expected or observed!=expected
                        or (len(expected)==2 and expected[0]==expected[1])):
                    errors.append({"identifier":facility,"routeIdentifier":identifier,
                                   "reason":"native bank directions do not replay the member's actual source approach"})
    return {"checkedMembers":count,"mismatchCount":len(errors),"examples":errors[:10],"status":"failed"if errors else"ok"}


def check_display_bank_transitions(document, rivers):
    named = {item["identifier"]:shapely.from_geojson(json.dumps(item["geometry"]))
             for item in document["preparedRoadGeometry"]}
    bridges = {item["identifier"]:item for item in document["bridges"]}
    index = shapely.STRtree(shapely.get_parts(rivers))
    errors, count = [], 0
    for crossing in document["crossings"]:
        point = _point(crossing["displayPoint"],"display bank-transition point")
        rays = _incident_angles(point,index)
        portals = bridges[crossing["id"]]["bankSpan"]["bankPortals"]
        for member in crossing["sourceBankTransitions"]:
            banks = [_point(value,"member display bank point")for value in member["bankPoints"]]
            identifier = member["routeIdentifier"]
            vectors = np.asarray([bank.coords[0]for bank in banks])-np.asarray(point.coords[0])
            lengths = np.linalg.norm(vectors,axis=1)
            # A fixed audit disk can contain the next bend of a tiny deck.
            # Its two sectors reconnect outside that bend and falsely look
            # like one bank. Classify the station's ordered incident rays.
            owners = []
            for vector,length in zip(vectors,lengths):
                angle = float(np.arctan2(vector[1],vector[0]))
                on_ray = any(abs((angle-ray+np.pi)%(2*np.pi)-np.pi)<=1e-10 for ray in rays)
                owners.append(None if not rays or length==0 or on_ray else (int(np.searchsorted(rays,angle,side="right"))-1)%len(rays))
            valid = (len(banks)in {1,2} and len(banks)==len(member["nativeDirections"])
                     and None not in owners and len(set(owners))==len(banks) and identifier in named)
            for bank in banks:
                valid &= any(bank.equals(_point(portal["position"],"member bank portal"))
                             and identifier in portal["sourceRoadIds"]for portal in portals)
                valid &= identifier in named and bank.distance(named[identifier])<=TOLERANCE
            if not valid:
                errors.append({"identifier":crossing["id"],"routeIdentifier":identifier,
                               "reason":"member does not reach its distinct source-required display banks"})
            count += 1
    return {"checkedMembers":count,"mismatchCount":len(errors),"examples":errors[:10],"status":"failed"if errors else"ok"}


def _channel_deck(geometry, point, bank_points, channel, roads, directions, *, access):
    """Replay one exact noded facility component and its station-bank arms."""
    import heapq

    lines = _primitives(geometry)
    reasons = []
    if (geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {"LineString", "MultiLineString"}
            or any(line.geom_type != "LineString" for line in lines)):
        return ["span is not nonempty valid line geometry"]
    if geometry.difference(channel).length > TOLERANCE:
        reasons.append("span extends outside the actual channel")
    if np.any(shapely.distance(shapely.points(shapely.get_coordinates(geometry)), roads) > TOLERANCE):
        reasons.append("span vertices are detached from the actual saved road")
    selected = min(range(len(lines)), key=lambda index: lines[index].distance(point))
    if lines[selected].distance(point) > TOLERANCE:
        return [*reasons, "facility anchor is not on its water component"]
    line = lines[selected]
    station = float(line.project(point))
    origin = tuple(line.interpolate(station).coords[0])
    pieces = [*lines[:selected], *lines[selected+1:]]
    pieces.extend(substring(line, first, last) for first, last in ((0.,station),(station,line.length)) if last>first)
    graph = {}
    for piece in pieces:
        coordinates = np.asarray(piece.coords)
        first, last = tuple(coordinates[0]), tuple(coordinates[-1])
        graph.setdefault(first, []).append((last, piece.length, coordinates))
        graph.setdefault(last, []).append((first, piece.length, coordinates[::-1]))
    distance, previous = {origin: 0.}, {}
    pending = [(0.,origin)]
    while pending:
        cost, node = heapq.heappop(pending)
        if cost != distance[node]:
            continue
        for neighbour, length, coordinates in graph.get(node, ()):
            candidate = cost+length
            if candidate < distance.get(neighbour,float("inf")):
                distance[neighbour] = candidate
                previous[neighbour] = (node,coordinates)
                heapq.heappush(pending,(candidate,neighbour))
    if len(distance) != len(graph):
        return [*reasons, "span contains a disconnected water component"]
    terminals, chords = [], []
    for node, entries in graph.items():
        terminal = shapely.Point(node)
        if len(entries) != 1 or terminal.distance(point) <= TOLERANCE:
            continue
        # A short leaf left by double-precision noding is not another bank.
        # Keep its exact geometry in the permission subtraction nevertheless.
        stem, before, current = 0., None, node
        while True:
            neighbour, length, _coordinates = next(entry for entry in graph[current] if entry[0] != before)
            stem += length
            before, current = current, neighbour
            if len(graph[current]) != 2 or current == origin:
                break
        if stem <= TOLERANCE and terminal.distance(channel.boundary) > TOLERANCE:
            continue
        terminals.append(terminal)
        if terminal.distance(channel.boundary) > TOLERANCE:
            reasons.append("water component has a terminal inside the actual channel")
        vector = np.asarray(node)-np.asarray(point.coords[0])
        if not any(np.dot(vector,direction)>0
                   and abs(vector[0]*direction[1]-vector[1]*direction[0])<=TOLERANCE for direction in directions):
            reasons.append("bank terminal does not follow an actual channel bank-sector bisector")
        chord = shapely.LineString((point.coords[0],node))
        chords.append(chord)
        arm, current = [], node
        while current != origin:
            before, coordinates = previous[current]
            arm.extend(coordinates)
            current = before
        arm = shapely.points(np.asarray(arm))
        if (np.any(shapely.distance(arm,chord)>TOLERANCE)
                or abs(distance[node]+point.distance(shapely.Point(origin))-chord.length)>TOLERANCE):
            reasons.append("station-bank arm is curved or follows the channel")
    for line in pieces:
        for first,last in zip(line.coords[:-1],line.coords[1:]):
            segment = shapely.LineString((first,last))
            if segment.length>TOLERANCE and not any(
                shapely.Point(first).distance(chord)<=TOLERANCE and shapely.Point(last).distance(chord)<=TOLERANCE
                for chord in chords):
                reasons.append("water component contains a segment outside its straight station-bank arms")
    minimum = 1 if access else 2
    if len(terminals)!=len(bank_points) or len(terminals)<minimum:
        reasons.append("span bank-terminal count differs from declared physical banks")
    for bank in bank_points:
        if (bank.distance(channel.boundary)>TOLERANCE
                or not any(bank.distance(terminal)<=TOLERANCE for terminal in terminals)):
            reasons.append("declared bank point lacks an exact channel-boundary terminal")
    if len(bank_points) and len({tuple(bank.coords[0])for bank in bank_points})!=len(bank_points):
        reasons.append("duplicate bank points")
    return reasons


def check_channel_ink(document, roads, rivers, *, valid_bridges, valid_access_points):
    channel = shapely.from_geojson(json.dumps(document["channelGeometry"]))
    if (channel.geom_type not in {"Polygon", "MultiPolygon"} or channel.is_empty or not channel.is_valid):
        raise ValueError("actual rendered channel evidence must be nonempty valid polygon geometry")
    river_index = shapely.STRtree(shapely.get_parts(rivers))
    source_members = {record["id"]: set(record["sourceRoadIds"]) for record in document["crossings"]}
    named_roads = {record["identifier"]: shapely.from_geojson(json.dumps(record["geometry"]))
                   for record in document["preparedRoadGeometry"]}
    allowed, errors = [], []
    for bridge in document["bridges"]:
        point = _point(bridge["position"], "bridge channel position")
        payload = bridge["bankSpan"]
        span = shapely.from_geojson(json.dumps(payload["geometry"]))
        banks = [_point(value, "bridge bank point") for value in payload["bankPoints"]]
        reasons = _channel_deck(span, point, banks, channel, roads,
                                _bank_directions(rivers, point, river_index), access=False)
        portals = payload["bankPortals"]
        if len(portals) != len(banks):
            reasons.append("bank portals do not account for every physical bank terminal")
        positions = []
        for portal in portals:
            position = _point(portal["position"], "source bank portal")
            members = portal["sourceRoadIds"]
            if (not members or len(set(members)) != len(members)
                    or not set(members) <= source_members.get(bridge["identifier"], set())
                    or not any(position.equals(bank) for bank in banks)
                    or any(position.equals(previous) for previous in positions)):
                reasons.append("physical bank portal lacks a distinct licensed source road")
            if any(member not in named_roads or position.distance(named_roads[member]) > TOLERANCE
                   for member in members):
                reasons.append("declared source road does not reach its physical bank portal")
            positions.append(position)
        if bridge["identifier"] not in valid_bridges:
            reasons.append("bridge lacks validated source facility evidence")
        if reasons:
            errors.append({"identifier": bridge["identifier"], "reasons": reasons})
        else:
            allowed.append(span)
    for index, record in enumerate(document["endpointTouches"]):
        point = _point(record["displayPoint"], "city channel access point")
        access = shapely.from_geojson(json.dumps(record["channelAccessGeometry"]))
        banks = [_point(value, "city channel bank point") for value in record["bankPoints"]]
        reasons = _channel_deck(access, point, banks, channel, roads,
                                _bank_directions(rivers, point, river_index), access=True)
        if not any(point.equals(valid) for valid in valid_access_points):
            reasons.append("channel access lacks validated exact source-city/single-bank evidence")
        if reasons:
            errors.append({"endpointTouchIndex": index, "reasons": reasons})
        else:
            allowed.append(access)
    # A path exactly on the physical bank belongs to the dry bank. It is not
    # an unlicensed route through the channel's interior.
    water_road = shapely.intersection(roads, channel).difference(channel.boundary)
    valid_certificates, roundoff_errors, maximum_roundoff = check_bank_roundoff(
        document["bankBoundaryRoundoff"],channel,water_road)
    errors.extend({"bankBoundaryRoundoff":error}for error in roundoff_errors)
    permitted = shapely.union_all(allowed)
    extra = water_road.difference(permitted)
    bank_unsafe_length, roundoff_length, roundoff_count = _unsafe_water_segments(extra,valid_certificates)
    bank_edges=[shapely.LineString(record["bankSegment"])for record in valid_certificates]
    permission_edges=[shapely.LineString((first,last))for line in _primitives(permitted)
                      if line.geom_type=="LineString"for first,last in zip(line.coords[:-1],line.coords[1:])]
    # GEOS may also leave a last-bit subline on an independently validated
    # straight physical deck. Classify each real remainder against its
    # original bank or licensed span edge once; never subtract certificates.
    unsafe_length, _, certified_count = _unsafe_segments_against_edges(extra,bank_edges+permission_edges)
    unused = permitted.difference(water_road)
    actual_water_edges = [shapely.LineString((first,last))for line in _primitives(water_road)
                          if line.geom_type=="LineString"for first,last in zip(line.coords[:-1],line.coords[1:])]
    outside_length,noding_length,noding_count = _unsafe_segments_against_edges(unused,actual_water_edges)
    return {"status": "failed" if errors or unsafe_length > TOLERANCE or outside_length > TOLERANCE else "ok",
            "checkedBridgeSpans": len(document["bridges"]), "checkedSameBankAccessSpans": len(document["endpointTouches"]),
            "invalidSpans": len(errors), "examples": errors[:10], "actualChannelRoadLength": float(water_road.length),
            "unlicensedChannelRoadLength": float(unsafe_length), "permissionOutsideActualWaterRoadLength":outside_length,
            "rawSpanWaterDifferenceLength":float(unused.length),"certifiedSpanNodingResidualLength":noding_length,
            "certifiedSpanNodingResidualSegments":noding_count,
            "certifiedWaterPermissionNodingResidualLength":float(bank_unsafe_length-unsafe_length),
            "certifiedWaterPermissionNodingResidualSegments":certified_count-roundoff_count,
            "checkedBankRoundoffSegments":len(document["bankBoundaryRoundoff"]),
            "certifiedBankRoundoffLength":roundoff_length,"certifiedActualBankSegments":roundoff_count,
            "maximumBankRoundoffDistance":maximum_roundoff,
            "toleranceNativeCells": TOLERANCE, "widePermissionBuffers": 0}


def check_endpoint_touches(records, roads, rivers, native_river, society, width):
    """Replay explicit single-bank city access from exact recorded endpoints."""
    if not isinstance(records, list):
        raise ValueError("endpoint-touch evidence must be a list, including when empty")
    routes = tuple(route for route in society.transport.routes if route.mode in {"road", "rail"})
    settlements = {item.identifier: item for item in society.settlements} if records else {}
    lines = shapely.get_parts(roads)
    line_index = shapely.STRtree(lines)
    licensed, errors = [], []
    seen = set()
    for record in records:
        point = _point(record["nativePoint"], "endpoint touch native point")
        display = _point(record["displayPoint"], "endpoint touch display point")
        if tuple(point.coords[0]) in seen:
            raise ValueError("duplicate endpoint-touch evidence")
        seen.add(tuple(point.coords[0]))
        entries, samples = [], []
        for route in routes:
            for coordinates, identifier, first in (
                (route.path[0], getattr(route, "source_settlement_id", None), True),
                (route.path[-1], getattr(route, "target_settlement_id", None), False),
            ):
                if identifier is None or point.distance(shapely.Point(coordinates)) > TOLERANCE:
                    continue
                entries.append((route.identifier, identifier))
                parts = np.split(np.asarray(route.path),
                    np.flatnonzero(np.abs(np.diff(np.asarray(route.path)[:, 0])) > width * .5) + 1)
                part = parts[0] if first else parts[-1]
                if len(part) < 2:
                    raise ValueError("endpoint-touch native road has no incident segment")
                line = shapely.LineString(part)
                samples.append(line.interpolate(min(.025, line.length) if first else max(0., line.length - .025)))
        source_roads = sorted({route for route, _settlement in entries})
        source_settlements = sorted({settlement for _route, settlement in entries})
        source_centers_valid = bool(entries) and all(identifier in settlements
            and point.distance(shapely.Point(settlements[identifier].column + .5,
                                            settlements[identifier].row + .5)) <= TOLERANCE
            for identifier in source_settlements)
        display_samples = []
        for index in line_index.query(point.buffer(TOLERANCE), predicate="intersects"):
            line = lines[index]
            distance = line.project(point)
            if distance <= TOLERANCE:
                display_samples.append(line.interpolate(min(.025, line.length)))
            elif line.length - distance <= TOLERANCE:
                display_samples.append(line.interpolate(max(0., line.length - .025)))
            else:
                display_samples.extend((line.interpolate(max(0., distance - .025)),
                                        line.interpolate(min(line.length, distance + .025))))
        native_banks = _bank_count(native_river, point, samples)
        display_banks = _bank_count(rivers, point, display_samples)
        valid = (record.get("kind") == "same-bank-river-access" and point.equals(display)
                 and source_centers_valid and record.get("sourceRoadIds") == source_roads
                 and record.get("sourceSettlementIds") == source_settlements
                 and native_banks <= 1 and display_banks == 1
                 and record.get("nativeBankCount") == native_banks
                 and record.get("displayBankCount") == display_banks
                 and all(point.distance(geometry) <= TOLERANCE for geometry in (roads, rivers, native_river)))
        if valid:
            licensed.append(point)
        else:
            errors.append({"nativePoint": list(point.coords[0]), "sourceEndpointsValid": source_centers_valid,
                           "nativeBankCount": native_banks, "displayBankCount": display_banks})
    return licensed, errors


def check_crossings(document, grid, society, *, raw_elevation_m, locations):
    if document.get("schema") != "shared-road-river-crossings-v1":
        raise ValueError("missing shared road/river geometry evidence")
    if document.get("checks") != {"bridgesOffRoad": 0, "bridgesOffRiver": 0}:
        raise ValueError("saved renderer crossing checks are missing or report detached facilities")
    roads = shapely.from_geojson(json.dumps(document["roadGeometry"]))
    rivers = shapely.from_geojson(json.dumps(document["riverGeometry"]))
    if any(geometry.geom_type != "MultiLineString" or geometry.is_empty or not geometry.is_valid
           for geometry in (roads, rivers)):
        raise ValueError("saved road and river evidence must be nonempty valid MultiLineStrings")
    expected = {bridge.identifier: bridge for bridge in society.transport.bridges}
    if not expected:
        raise ValueError("rebuilt world has no saved bridge facilities to verify")
    bridges, crossings = document["bridges"], document["crossings"]
    if (not isinstance(bridges, list) or not isinstance(crossings, list)
            or Counter(item["identifier"] for item in bridges) != Counter(expected.keys())
            or Counter(item["id"] for item in crossings) != Counter(expected.keys())):
        raise ValueError("delivered bridge/crossing identities differ from saved society facilities")
    crossing_by_id = {item["id"]: item for item in crossings}
    routes = {route.identifier: route for route in society.transport.routes if route.mode in {"road", "rail"}}
    prepared = document["preparedRoadGeometry"]
    if (not isinstance(prepared, list)
            or Counter(item["identifier"] for item in prepared) != Counter(routes.keys())):
        raise ValueError("named bank-approach road identities differ from saved society routes")
    for item in prepared:
        geometry = shapely.from_geojson(json.dumps(item["geometry"]))
        if geometry.geom_type != "MultiLineString" or geometry.is_empty or not geometry.is_valid:
            raise ValueError("named bank approaches require nonempty valid MultiLineString geometry")
        route=routes[item['identifier']]
        source=list(route.path)
        for index,identifier in ((0,getattr(route,'source_settlement_id',None)),(-1,getattr(route,'target_settlement_id',None))):
            if identifier in locations:
                source[index]=tuple(reversed(locations[identifier]))
        if (shapely.Point(geometry.geoms[0].coords[0]).distance(shapely.Point(source[0])) > TOLERANCE
                or shapely.Point(geometry.geoms[-1].coords[-1]).distance(shapely.Point(source[-1])) > TOLERANCE):
            raise ValueError("a named bank approach moved its accepted source route endpoint")
    native_river = native_river_geometry(grid, raw_elevation_m=raw_elevation_m)
    outlets = hydrologic_outlet_targets(grid,raw_elevation_m)
    errors = []
    distances = {"maximumRoadDistance": 0., "maximumRiverDistance": 0., "maximumNativeRoadDistance": 0.,
                 "maximumNativeRiverDistance": 0.}
    for item in bridges:
        bridge = expected[item["identifier"]]
        crossing = crossing_by_id[bridge.identifier]
        point = _point(item["position"], "bridge position")
        display = _point(crossing["displayPoint"], "crossing display point")
        native = _point(crossing["nativePoint"], "crossing native point")
        tangent = np.asarray(item["tangent"], dtype=float)
        valid_tangent = tangent.shape == (2,) and np.all(np.isfinite(tangent)) and abs(np.linalg.norm(tangent) - 1.) <= 1e-6
        row, column = crossing["riverCell"]
        sources = crossing["sourceRoadIds"]
        valid_source = (crossing.get("kind") == "bridge" and sources and len(set(sources)) == len(sources)
                        and bridge.route_identifier in sources and all(identifier in routes for identifier in sources)
                        and item["routeIdentifier"] == bridge.route_identifier
                        and crossing.get("evidence") == "recorded-bridge-and-native-flow-crossing"
                        and type(row) is int and type(column) is int
                        and (row, column) == (bridge.row, bridge.column)
                        and 0 <= row < grid.shape[0] and 0 <= column < grid.shape[1]
                        and crossing["riverOrder"] == bridge.river_order == int(grid.river_order[row, column]))
        native_edge_distance = float("inf")
        if valid_source:
            source_cell = row*grid.shape[1]+column
            target = outlets.get(source_cell,int(grid.flow_to[row,column]))
            if 0<=target<grid.water.size:
                target_row,target_column = divmod(target,grid.shape[1])
                edge = shapely.LineString(((column+.5,row+.5),(target_column+.5,target_row+.5)))
                native_edge_distance = native.distance(edge)
                valid_source = native_edge_distance<=TOLERANCE
        route = routes.get(bridge.route_identifier)
        native_road = shapely.MultiLineString([]) if route is None else _native_road_geometry(route, grid.shape[1])
        if native_road.is_empty:
            raise ValueError(f"{bridge.identifier}: source facility lacks a native road segment")
        local = {"maximumRoadDistance": max(point.distance(roads), display.distance(roads)),
                 "maximumRiverDistance": max(point.distance(rivers), display.distance(rivers)),
                 "maximumNativeRoadDistance": max([native.distance(native_road),
                     *(native.distance(_native_road_geometry(routes[identifier], grid.shape[1]))
                       for identifier in sources if identifier in routes)]),
                 "maximumNativeRiverDistance": native.distance(native_river)}
        for key, value in local.items():
            distances[key] = max(distances[key], float(value))
        if (not valid_source or not valid_tangent or point.distance(display) > TOLERANCE
                or any(not np.isfinite(value) or value > TOLERANCE for value in local.values())):
            errors.append({"identifier": bridge.identifier, "sourceEvidenceValid": bool(valid_source),
                           "tangentValid": bool(valid_tangent), "positionDifference": float(point.distance(display)),
                           "declaredNativeFlowEdgeDistance": float(native_edge_distance) if np.isfinite(native_edge_distance) else None,
                           **local})
    endpoint_points, endpoint_errors = check_endpoint_touches(document["endpointTouches"], roads, rivers,
                                                              native_river, society, grid.shape[1])
    native_banks = check_native_bank_transitions(crossings,routes,grid.shape[1],native_river)
    display_banks = check_display_bank_transitions(document,rivers)
    licensed = shapely.MultiPoint([item["position"] for item in bridges] + endpoint_points)
    intersections = shapely.intersection(roads, rivers)
    primitives = _primitives(intersections)
    unexplained = [part for part in primitives if part.geom_type == "Point" and part.distance(licensed) > 1e-6]
    overlap = sum(part.length for part in primitives if part.geom_type == "LineString")
    channel = check_channel_ink(document, roads, rivers,
        valid_bridges=set(expected) - {error["identifier"] for error in errors}, valid_access_points=endpoint_points)
    return {"status": "failed" if errors or endpoint_errors or unexplained or overlap > TOLERANCE
            or channel["status"] == "failed" or native_banks["status"] == "failed" or display_banks["status"] == "failed" else "ok", "checkedBridges": len(bridges),
            "checkedCrossings": len(crossings), "toleranceNativeCells": TOLERANCE,
            "mismatchCount": len(errors), "examples": errors[:10], "unlicensedCrossings": len(unexplained),
            "checkedEndpointTouches": len(document["endpointTouches"]), "endpointTouchMismatches": len(endpoint_errors),
            "nativeBankTransitions":native_banks,"displayBankTransitions":display_banks,
            "endpointTouchExamples": endpoint_errors[:10],
            "channelInk": channel,
            "checkedNamedBankApproachRoads": len(prepared),
            "roadRiverOverlapLength": float(overlap), **distances}


def check_names(grid, source, thematic, society, scene, forbidden):
    request = grid.metadata["societyGeneration"]
    if request.get("namingProfile") != "procedural":
        raise ValueError("geographic source replay requires the saved procedural naming profile")
    expected = extract_geographic_features(grid, thematic, society.cultures,
        procedural_name_lexicon(request["namingSeed"]), raw_elevation_m=source.relative_elevation_m)
    signature = lambda item: (item.identifier, item.feature_type, item.row, item.column, item.language_identifier, item.tier)
    authored = Counter(signature(item) for item in society.geographic_features)
    replayed = Counter(signature(item) for item in expected)
    if not authored or authored != replayed:
        raise ValueError("saved geographic identities differ from native physical-feature source replay")
    audit = naming_audit(society, forbidden)
    canonical = naming_audit(assign_world_identity(society, seed=request["namingSeed"], forbidden=forbidden), forbidden)
    if audit != canonical or audit["oldNameMatches"] or audit["duplicateNames"]:
        raise ValueError("saved names fail canonical seed replay, exclusions or uniqueness")
    display = _geography.check_scene(grid, source.relative_elevation_m, society, scene)
    return {"status": display["status"], "sourceReplayFeatures": len(expected),
            "geographicTypeCounts": dict(Counter(item.feature_type for item in expected)),
            "canonicalNameDigest": audit["nameDigest"], "unsourcedGeographicNames": 0,
            "delivered": display}


def _line_signature(points):
    """Canonical integer SVG geometry, retaining every non-collinear turn."""
    vertices = []
    for coordinates in np.asarray(points):
        point = tuple(round(float(value) * SVG_COORDINATE_SCALE) for value in coordinates)
        if vertices and vertices[-1] == point:
            continue
        while len(vertices) > 1:
            before, centre = vertices[-2:]
            first = (centre[0]-before[0], centre[1]-before[1])
            last = (point[0]-centre[0], point[1]-centre[1])
            if first[0]*last[1]-first[1]*last[0] or first[0]*last[0]+first[1]*last[1] < 0:
                break
            vertices.pop()
        vertices.append(point)
    if len(vertices)<2:
        return None
    forward = tuple(vertices)
    return min(forward, forward[::-1])


def _road_signatures(document, *, land_clip=None):
    signatures = []
    def visit(node, *, clips=(), transformed=False, definitions=False):
        tag = _tiles.tag(node)
        definitions = definitions or tag in {"defs", "clipPath", "pattern"}
        if definitions:
            return
        attributes = _tiles._attributes(node)
        transformed = transformed or bool(attributes.get("transform"))
        reference = attributes.get("clip-path")
        if reference and reference != "none":
            match = re.fullmatch(r"url\(#([^)]*)\)", reference)
            if not match:
                raise ValueError("delivered road references an unsupported clip")
            clips = (*clips, match[1])
        if attributes.get("data-route-mode") in {"road", "rail"} and attributes.get("data-route-casing") != "true":
            if (tag != "path" or transformed or attributes.get("fill") != "none"
                    or attributes.get("stroke", "none") == "none"
                    or float(attributes.get("stroke-opacity", "1")) <= 0
                    or (land_clip is not None and clips != (land_clip,))):
                raise ValueError("delivered road lacks unchanged native line geometry and its exact land clip")
            for points in _landforms._physical._line_points(attributes["d"]):
                if len(points) < 2:
                    raise ValueError("delivered road path has no segment")
                signature=_line_signature(points)
                if signature is not None:
                    signatures.append(signature)
        for child in node:
            visit(child, clips=clips, transformed=transformed, definitions=definitions)
    visit(document)
    return Counter(signatures)


def _bridge_span_matches(node, expected):
    """Check the sole native-coordinate deck and cancelling translations."""
    number = r"([-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?)"
    match = re.fullmatch(rf"translate\({number}[ ,]+{number}\)", node.get("transform", ""))
    if not match:
        return ["bridge glyph lacks its sole native translation"]
    x, y = map(float, match.groups())
    reasons = []
    position = np.asarray(expected["position"])
    metadata = np.asarray([float(node.get(key, "nan")) for key in ("data-map-x", "data-map-y", "data-base-angle")])
    expected_angle = float(np.rad2deg(np.arctan2(expected["tangent"][1],expected["tangent"][0])))
    if (np.linalg.norm(np.asarray((x,y))-position) > TOLERANCE
            or not np.isfinite((x,y)).all() or not np.isfinite(metadata).all()
            or np.max(np.abs(metadata[:2]-(x,y))) > TOLERANCE
            or abs((metadata[2]-expected_angle+180)%360-180) > TOLERANCE
            or node.get("data-route-id") != expected["routeIdentifier"]):
        reasons.append("bridge glyph moved its saved facility or source route identity")
    close = [child for child in node.iter() if "bridge-close" in child.get("class", "").split()]
    if len(close) != 1:
        return [*reasons, "bridge glyph lacks its single physical close deck"]
    cancellation=re.fullmatch(rf"translate\({number}[ ,]+{number}\)",close[0].get("transform",""))
    if not cancellation or tuple(map(float,cancellation.groups()))!=(-x,-y):
        reasons.append("physical close bridge translations do not exactly cancel")
    if any(_tiles._attributes(child).get("transform") for child in close[0].iter() if child is not close[0]):
        reasons.append("physical close bridge deck has an unrecorded nested transform")
    paths = [child for child in close[0].iter() if _tiles.tag(child) == "path"]
    if sum("bridge-physical-span" in child.get("class", "").split() for child in paths) != 1:
        reasons.append("bridge glyph lacks its unique explicit physical span")
    span = shapely.from_geojson(json.dumps(expected["bankSpan"]["geometry"]))
    target=Counter(signature for line in _primitives(span)
                   if (signature:=_line_signature(line.coords))is not None)
    if len(paths)!=3:
        reasons.append("close bridge lacks its three physical deck strokes")
    for path in paths:
        if (path.get("transform") or path.get("fill") != "none" or path.get("stroke", "none") == "none"
                or path.get("vector-effect")!="non-scaling-stroke"):
            reasons.append("physical bridge deck applies an unrecorded transform or fill")
            continue
        actual=Counter(signature for points in _landforms._physical._line_points(path.attrib["d"])
                       if (signature:=_line_signature(points))is not None)
        if actual!=target:
            reasons.append("close bridge stroke differs from its integer native bank span")
    if not paths:
        reasons.append("close bridge has no painted deck")
    return reasons


def _joined_line_signatures(signatures):
    lines=[np.asarray(signature,dtype=float)/SVG_COORDINATE_SCALE for signature in signatures if signature is not None]
    if not lines:
        return Counter()
    merged=shapely.line_merge(shapely.MultiLineString(lines))
    return Counter(signature for line in _primitives(merged)
                   if (signature:=_line_signature(line.coords))is not None)


def check_transport_delivery(review, manifest, document, scene, *, source_routes):
    """Compare delivered SVG roads and close bridge decks with saved evidence."""
    roads = shapely.from_geojson(json.dumps(document["roadGeometry"]))
    road_parts = shapely.get_parts(roads)
    signatures = [_line_signature(line.coords) for line in road_parts]
    expected = Counter(signature for signature in signatures if signature is not None)
    rail_signatures={_line_signature(line.coords) for item in document['drawnTransportPaths']
                     if item['mode']=='rail'
                     for line in _primitives(shapely.from_geojson(json.dumps(item['geometry'])))}
    scene_document = _tiles.ET.fromstring(scene)
    actual = _road_signatures(scene_document)
    errors = []
    overview_source=[_line_signature(line.coords)for item in document["drawnTransportPaths"]
                     if item['mode']in {'road','rail'} and item['importance']=='trunk'
                     for line in _primitives(shapely.from_geojson(json.dumps(item["geometry"])))]
    overview_expected=_joined_line_signatures(overview_source)
    overview_actual=_joined_line_signatures(actual.elements())
    if overview_actual!=overview_expected or actual-expected:
        errors.append({"artifact":"atlas-scene.svg","missingTrunkComponents":sum((overview_expected-overview_actual).values()),
                       "unexpectedTrunkComponents":sum((overview_actual-overview_expected).values()),
                       "unexpectedRoadPaths":sum((actual-expected).values())})
    expected_bridges = {item["identifier"]: item for item in document["bridges"]}
    bridge_nodes = [node for node in scene_document.iter() if node.get("data-bridge-id")]
    if Counter(node.get("data-bridge-id") for node in bridge_nodes) != Counter(expected_bridges.keys()):
        raise ValueError("delivered scene bridge identities differ from validated source facilities")
    for node in bridge_nodes:
        reasons = _bridge_span_matches(node, expected_bridges[node.get("data-bridge-id")])
        if reasons:
            errors.append({"bridge": node.get("data-bridge-id"), "reasons": reasons})
    def check_ancestors(node, transformed=False):
        if node.get("data-bridge-id") and transformed:
            errors.append({"bridge": node.get("data-bridge-id"), "reasons": ["bridge layer applies an unrecorded ancestor transform"]})
        transformed = transformed or bool(node.get("transform"))
        for child in node:
            check_ancestors(child,transformed)
    check_ancestors(scene_document)
    # Exact road polylines are clipped at the tile frame with a half-cell
    # stroke halo; no vertex simplification is permitted. Dashed rail paths
    # retain their authored phase. Check independent source intersections.
    envelopes = shapely.envelope(road_parts)
    index = shapely.STRtree(envelopes)
    checked = 0
    for row in range(manifest["rows"]):
        for column in range(manifest["columns"]):
            tile = _tiles.read_tile(review,manifest,"detail",row,column)
            west,north,east,south=tile['rectangle'].bounds
            frame=shapely.box(west-.5,north-.5,east+.5,south+.5)
            wanted=Counter()
            for item in index.query(tile['rectangle']):
                if signatures[int(item)] in rail_signatures:
                    wanted.update((signatures[int(item)],))
                    continue
                points=[(round(float(x)*SVG_COORDINATE_SCALE)/SVG_COORDINATE_SCALE,
                         round(float(y)*SVG_COORDINATE_SCALE)/SVG_COORDINATE_SCALE)
                        for x,y in road_parts[int(item)].coords]
                clipped=shapely.LineString(points).intersection(frame)
                wanted.update(signature for part in _primitives(clipped)
                              if (signature:=_line_signature(part.coords))is not None)
            actual = _road_signatures(_tiles.tile_document(tile["payload"]["ink"]), land_clip=tile["landClipId"])
            if actual != wanted:
                errors.append({"artifact": tile["key"], "missingRoadPaths": sum((wanted-actual).values()),
                               "unexpectedRoadPaths": sum((actual-wanted).values())})
            checked += 1
    return {"status": "failed" if errors else "ok", "checkedSceneRoadPaths": sum(expected.values()),
            "checkedDetailTiles": checked, "checkedCloseBridgeDecks": len(bridge_nodes),
            "geometryMismatches": len(errors), "examples": errors[:10],"checkedOverviewSourceTrunkRoutes":len(overview_source),
            "linearSVGCoordinateScale": SVG_COORDINATE_SCALE, "bridgeCoordinateTolerance": TOLERANCE}


def check_world(world: Path, *, skip_partitions=False):
    started = time.monotonic()
    world = world.resolve()
    required_files(world)
    grid = WorldGrid.load(world / "grid")
    fingerprints, source = check_sources(world, grid)
    society = load_society(world / "society", expected_grid_digest=grid.content_digest())
    for name in ("society.json", "society.npz"):
        if _sha256(world / "society" / name) != _sha256(world / "review" / name):
            raise ValueError(f"delivered society differs from saved canonical society: {name}")
    from world_atlas.core.ecological_sources import derive_ecological_sources
    thematic = derive_thematic_layers(grid, ecological_sources=derive_ecological_sources(grid, source))
    crossing_document = _json(world / "review/transport-crossings.json")
    scene = (world / "review/atlas-scene.svg").read_text(encoding="utf-8")
    locations=_json(world/'review/interface-presentation.json')['settlementLocations']
    if set(locations)!= {town.identifier for town in society.settlements}:
        raise ValueError('displayed settlement anchors must cover the actual world')
    for town in society.settlements:
        row,column=locations[town.identifier]
        if not town.row<=row<=town.row+1 or not town.column<=column<=town.column+1:
            raise ValueError('displayed settlement anchor left its accepted native cell')
    checks = {"sources": fingerprints,
              "wetlandField": check_wetland_field(world / "thematic/wetland-support.npz", grid, thematic),
              "crossings": check_crossings(crossing_document, grid, society,
                                           raw_elevation_m=source.relative_elevation_m,locations=locations)}
    checks["landformPaint"] = _landforms.check_static(world / "review", grid_path=world / "grid",
                                                     physical_source_path=world / "source/physical-fields.npz")
    checks["geographicNames"] = check_names(grid, source, thematic, society,
        scene,
        _json(world / "naming-exclusions.json")["forbidden"])
    manifest = _tiles.read_manifest(world / "review")
    checks["transportDelivery"] = check_transport_delivery(world/"review",manifest,crossing_document,scene,
                                                          source_routes=society.transport.routes)
    checks["partitions"] = ({"status": "skipped", "reason": "explicit --skip-partitions"} if skip_partitions
                            else check_partitions(world / "review", manifest, grid, society))
    failures = [name for name, result in checks.items() if result["status"] == "failed"]
    return {"schema": "delivered-vector-world-check-v1", "status": "failed" if failures else ("partial" if skip_partitions else "ok"),
            "world": str(world), "failures": failures, "checks": checks,
            "elapsedSeconds": round(time.monotonic() - started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("--skip-partitions", action="store_true", help="Skip costly detail paint verification; result is partial")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = check_world(args.world, skip_partitions=args.skip_partitions)
    except (ValueError, KeyError, TypeError, OSError) as error:
        report = {"schema": "delivered-vector-world-check-v1", "status": "failed",
                  "world": str(args.world.resolve()), "failures": [str(error)]}
    destination = args.report or args.world / "review/vector-world-check.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "failures": report["failures"], "report": str(destination)}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
