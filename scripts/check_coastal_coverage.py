"""Check full-quality SVG exports cover their authoritative physical shore.

This deliberately reads the delivered SVG paths instead of reproducing the
renderer. Interactive tile geometry is checked by check_atlas_tiles.py, and
actual zoom performance is checked by check_review_browser.cjs.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import shapely

from world_atlas.core.svg_artifacts import read_svgz

TOKEN = re.compile(r"[MLHVZmlhvz]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
NS = {"s": "http://www.w3.org/2000/svg"}
THEMES = ("climate", "biome", "watersheds", "land-potential", "habitability", "vegetation-cover", "civilizations",
          "languages", "religions", "political", "provinces", "population")
GLOBE_THEMES = ("terrain", "climate", "biome", "watershed", "potential", "habitability", "vegetation",
                "population", "civilizations", "languages", "religions", "political", "provinces")


def globe_payload(path: Path):
    content = path.read_text(encoding="utf-8")
    assignment = re.match(r"\s*window\.WorldAtlasGlobe\s*=\s*", content)
    if assignment is None:
        raise ValueError("globe-data.js must assign WorldAtlasGlobe")
    value, end = json.JSONDecoder().raw_decode(content, assignment.end())
    if not isinstance(value, dict) or content[end:].strip() != ";":
        raise ValueError("globe-data.js must contain one JSON object assignment")
    return value


def path_geometry(data: str):
    """Decode standard polygon path coordinates, with even-odd holes."""
    if re.sub(r"[\s,]+", "", TOKEN.sub("", data)):
        raise ValueError("Coverage check requires polygonal SVG path commands")
    tokens = TOKEN.findall(data)
    rings, ring, index, command = [], [], 0, None
    # Relative coordinates are authored decimal deltas. Accumulate them
    # exactly before Shapely converts to doubles, so shared SVG vertices do
    # not separate by floating-point ulps and invent invalid touching rings.
    current, start = (Decimal(0), Decimal(0)), None
    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
            if command.upper() == "Z":
                if len(ring) >= 3:
                    rings.append(ring + [ring[0]])
                if start is not None:
                    current = start
                ring = []
                start = None
                command = None
                continue
        if command in ("M", "m", "L", "l"):
            if command.upper() == "M" and ring:
                rings.append(ring + [ring[0]])
                ring = []
            point = (Decimal(tokens[index]), Decimal(tokens[index + 1]))
            if command.islower():
                point = (point[0] + current[0], point[1] + current[1])
            current = point
            if command.upper() == "M":
                start = current
            ring.append(current)
            index += 2
            command = "l" if command.islower() else "L"
        elif command in ("H", "h", "V", "v") and ring:
            value = Decimal(tokens[index]); index += 1
            axis = 0 if command.upper() == "H" else 1
            if command.islower():
                value += current[axis]
            current = (value, current[1]) if axis == 0 else (current[0], value)
            ring.append(current)
        else:
            raise ValueError(f"Unsupported SVG command {command!r}")
    if ring:
        rings.append(ring + [ring[0]])
    if not rings:
        return shapely.GeometryCollection()
    # SVG evenodd counts ring crossings, including nested islands and rings
    # that share an edge or touch another edge at a vertex. Build_area applies
    # its own hole inference and can omit such real painted faces. Node only
    # the exact linework, then classify every face by the original ring parity.
    linework = shapely.node(shapely.MultiLineString(rings))
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(linework)))
    if not len(faces):
        return shapely.GeometryCollection()
    polygons = np.asarray([shapely.Polygon(ring) for ring in rings], dtype=object)
    shapely.prepare(polygons)
    index = shapely.STRtree(polygons)
    points = shapely.point_on_surface(faces)
    candidates = index.query(points)
    # Prepared polygons must be the first predicate operand. Querying points
    # with "within" repeatedly scans the fine continental ring for every lake.
    inside = shapely.contains(polygons[candidates[1]], points[candidates[0]])
    counts = np.bincount(candidates[0][inside], minlength=len(faces))
    return shapely.union_all(faces[counts % 2 == 1])


class SurfacePaths(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.paths, self.elements = [], {}, {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        owner = attributes.get("id") or (self.stack[-1][1] if self.stack else None)
        if attributes.get("id"):
            self.elements.setdefault(attributes["id"], []).append((tag, attributes))
        if tag == "path" and attributes.get("d"):
            self.paths.setdefault(owner, []).append(attributes)
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append((tag, owner))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)


def union_paths(paths, *, paint=False):
    geometries = []
    for attributes in paths:
        geometry = path_geometry(attributes["d"])
        if paint and attributes.get("stroke", "none") != "none":
            geometry = geometry.buffer(float(attributes.get("stroke-width", "0")) / 2)
        geometries.append(geometry)
    return shapely.union_all(geometries)


def coverage_metrics(land, fills, *, sample=None):
    missing = shapely.difference(land, fills)
    pieces = [part for part in shapely.get_parts(missing) if part.area > 1e-6]
    pieces.sort(key=lambda part: part.area, reverse=True)
    result = {
        "uncoveredLandArea": float(missing.area),
        "uncoveredLandPercent": float(missing.area / land.area * 100),
        "uncoveredParts": len(pieces),
        "paintOutsideLandAreaBeforeClip": float(fills.difference(land).area),
        "largestGaps": [
            {"area": float(part.area), "bounds": list(part.bounds),
             "samplePoint": list(part.representative_point().coords[0])}
            for part in pieces[:5]
        ],
    }
    if sample is not None:
        local_land = land.intersection(sample)
        local_missing = missing.intersection(sample)
        result["reportedRegion"] = {
            "bounds": list(sample.bounds),
            "landArea": float(local_land.area),
            "uncoveredLandArea": float(local_missing.area),
        }
    return result


def settlement_display_location(land, lakes, identifier, name, x, y):
    point = shapely.Point(x, y)
    components = [part for part in shapely.get_parts(land) if part.covers(point)]
    return {
        "id": identifier, "name": name, "point": [x, y],
        "displayCenterOnVisibleLand": bool(land.covers(point)),
        "displayCenterCoveredByLake": bool(lakes.covers(point)),
        "distanceToVisibleLand": float(land.distance(point)),
        "visibleLandConnectedComponentArea": float(sum(part.area for part in components)),
    }


def published_surfaces(review: Path):
    parser = SurfacePaths()
    parser.feed(read_svgz(review / "physical-surface.svgz"))
    definitions = parser.elements.get("land-silhouette-clip", [])
    if len(definitions) != 1 or definitions[0][0] != "clippath":
        raise ValueError("Published atlas must define exactly one land-silhouette-clip")
    if definitions[0][1].get("clippathunits") != "userSpaceOnUse":
        raise ValueError("Published land clip must use native world coordinates")
    clip_paths = parser.paths["land-silhouette-clip"]
    if not clip_paths or any(path.get("clip-rule") != "evenodd" for path in clip_paths):
        raise ValueError("Published land clip must use explicit evenodd paths")
    physical_land = union_paths(clip_paths)
    lake_paths = [path for path in parser.paths.get("lakes", [])
                  if path.get("fill", "none") != "none"]
    lakes = union_paths(lake_paths)
    return parser, physical_land, lakes, physical_land.difference(lakes)


def clip_contract(review: Path, parser: SurfacePaths):
    globe = globe_payload(review / "globe-data.js")
    if not isinstance(globe.get("ink"), str):
        raise ValueError("Globe requires one published shared ink SVG")
    ink = ET.fromstring(globe["ink"])
    if ink.tag != "{http://www.w3.org/2000/svg}svg" or ink.findall(".//s:g[@data-partition='mutually-exclusive']", NS):
        raise ValueError("Globe shared ink must be SVG without thematic partitions")
    surface = ET.fromstring(globe["surface"])
    definitions = surface.findall(".//s:clipPath[@id='land-silhouette-clip']", NS)
    if len(definitions) != 1 or definitions[0].get("clipPathUnits") != "userSpaceOnUse":
        raise ValueError("Globe surface must define one native-coordinate physical land clip")
    globe_paths = [path.attrib for path in definitions[0].findall("s:path", NS)]
    flat_paths = parser.paths["land-silhouette-clip"]
    identity = lambda paths: [(path["d"], path.get("clip-rule")) for path in paths]
    if identity(globe_paths) != identity(flat_paths):
        raise ValueError("Globe and physical SVG export must use exactly the same physical land clip")
    if not isinstance(globe.get("textures"), dict) or set(globe["textures"]) != set(GLOBE_THEMES):
        raise ValueError("Globe must publish its thirteen current thematic document URLs")
    for theme in GLOBE_THEMES:
        filename = globe["textures"][theme]
        if filename != f"globe-theme-{theme}.svg":
            raise ValueError(f"{theme}: globe must load its own explicit thematic SVG filename")
        root = ET.parse(review / filename).getroot()
        partitions = root.findall(".//s:g[@data-partition='mutually-exclusive']", NS)
        if theme == "terrain":
            if partitions:
                raise ValueError("Globe terrain must not repeat a categorical partition")
            continue
        if len(partitions) != 1:
            raise ValueError(f"{theme}: globe must contain one published thematic partition")
        parents = {child: parent for parent in root.iter() for child in parent}
        node, clipped = partitions[0], False
        while node is not None:
            clipped |= node.get("clip-path") == "url(#land-silhouette-clip)"
            node = parents.get(node)
        if not clipped:
            raise ValueError(f"{theme}: globe thematic paint lacks the physical land clip")
        if root.findall(".//s:clipPath[@id='land-silhouette-clip']", NS):
            raise ValueError(f"{theme}: globe overlay duplicates the authoritative physical land clip")
    return {"globeThematicPartitions": len(GLOBE_THEMES) - 1, "globeLazyDocuments": len(GLOBE_THEMES),
            "sharedLandClipIdentical": True, "sharedGlobeInkSeparate": True}


def native_shoreline_metrics(physical_land, water):
    """Compare delivered shore paint to source centres without reconstructing it."""
    if water.ndim != 2 or min(water.shape) < 1:
        raise ValueError("Native shoreline check requires a nonempty water raster")
    shapely.prepare(physical_land)
    height, width = water.shape
    dry_as_water = wet_as_land = 0
    examples = []
    for first in range(0, height, 32):
        last = min(height, first + 32)
        x, y = np.meshgrid(np.arange(width) + .5, np.arange(first, last) + .5)
        displayed = shapely.covers(physical_land, shapely.points(x, y))
        expected = water[first:last] == 0
        dry_as_water += int(np.count_nonzero(expected & ~displayed))
        wet_as_land += int(np.count_nonzero(~expected & displayed))
        for row, column in np.argwhere(expected != displayed)[:max(0, 10 - len(examples))]:
            examples.append({"row": int(first + row), "column": int(column),
                             "expectedLand": bool(expected[row, column]),
                             "displayedLand": bool(displayed[row, column])})
    return {"checkedNativeCenters": int(water.size),
            "landCentersDisplayedAsWater": dry_as_water,
            "waterCentersDisplayedAsLand": wet_as_land,
            "mismatchCount": dry_as_water + wet_as_land,
            "examples": examples}


def check_review(review: Path, *, grid: Path | None = None):
    required = ("physical-surface.svgz", "globe-data.js", *(f"{theme}.svg" for theme in THEMES))
    missing = [name for name in required if not (review / name).is_file()]
    if missing:
        raise ValueError("Missing required published vector layers: " + ", ".join(missing))
    parser, physical_land, lakes, land = published_surfaces(review)
    clips = clip_contract(review, parser)
    sample = shapely.box(1095, 480, 1145, 510)
    layers = {}
    for theme in THEMES:
        root = ET.parse(review / f"{theme}.svg").getroot()
        partition = root.find(".//s:g[@data-partition='mutually-exclusive']", NS)
        if partition is None:
            raise ValueError(f"{theme}: missing published thematic partition")
        fills = [path.attrib for path in partition.findall(".//s:path", NS)
                 if path.get("fill", "none") != "none"]
        for frontier in root.findall(".//s:g[@id='political-frontier']", NS):
            fills.extend(path.attrib for path in frontier.findall(".//s:path", NS)
                         if path.get("fill", "none") != "none")
        raw = union_paths(fills)
        painted = raw if all(path.get("stroke", "none") == "none" or float(path.get("stroke-width", "0")) == 0
                             for path in fills) else union_paths(fills, paint=True)
        visible = raw.intersection(land)
        layers[theme] = coverage_metrics(land, visible, sample=sample)
        layers[theme]["paintOutsideLandAreaBeforeClip"] = float(raw.difference(land).area)
        layers[theme]["visiblePaintOutsideLandArea"] = float(visible.difference(land).area)
        layers[theme]["uncoveredLandAreaIncludingStroke"] = float(land.difference(painted.intersection(land)).area)
    valid = all(layer["uncoveredLandArea"] < .01 for layer in layers.values())
    native = None
    if grid is not None:
        with np.load(grid / "world-grid.npz", allow_pickle=False) as source:
            native = native_shoreline_metrics(physical_land, source["water"])
        valid &= native["mismatchCount"] == 0
    return {
        "schema": "full-quality-export-coverage-v1",
        "status": "ok" if valid else "failed",
        "visibleLandArea": float(land.area),
        "visibleLandUnderLakeFillArea": float(physical_land.intersection(lakes).area),
        "areaUnits": "native grid cells squared",
        "criterion": "Full-quality exported thematic fills intersected with their shared physical shore must cover all visible land; interactive tiles have separate acceptance.",
        "clipContract": clips,
        "nativeShoreline": native,
        "settlements": [
            settlement_display_location(land, lakes, identifier, name, x, y)
            for identifier, name, x, y in (("settlement-0150", "宁泽", 1114.5, 492.5),
                                          ("settlement-0262", "临泽", 1112.5, 498.5))
        ],
        "themes": layers,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("review", type=Path)
    cli.add_argument("--report", type=Path)
    cli.add_argument("--grid", type=Path, help="Canonical saved grid directory for an exhaustive shoreline classification check")
    args = cli.parse_args()
    report = check_review(args.review, grid=args.grid)
    destination = args.report or args.review / "coastal-coverage-check.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(destination),
                      "nativeShoreline": report["nativeShoreline"],
                      "uncoveredLandArea": {key: round(value["uncoveredLandArea"], 3)
                                            for key, value in report["themes"].items()}}))
    raise SystemExit(0 if report["status"] == "ok" else 1)
