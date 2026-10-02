"""Compare delivered scalar paint to every canonical land observation.

The consumer reads full-quality SVG exports and actual detail tile payloads.
It does not import the renderer, interpolate fields or reconstruct contours.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import shapely


def _consumer(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_svg = _consumer("scalar_svg_decoder", "check_coastal_coverage.py")
_tiles = _consumer("scalar_tile_decoder", "check_atlas_tiles.py")
SPECS = {
    "potential": {"field": "land_potential", "export": "land-potential.svg",
                  "attribute": "data-potential-band", "thresholds": (.2, .35, .5, .65, .8)},
    "habitability": {"field": "habitability", "export": "habitability.svg",
                     "attribute": "data-habitability-band", "thresholds": (.2, .35, .5, .65, .8)},
    "vegetation": {"field": "vegetation_cover", "export": "vegetation-cover.svg",
                   "attribute": "data-vegetation-band", "thresholds": tuple(i / 10 for i in range(1, 10))},
    "population": {"field": "population_density", "export": "population.svg",
                   "attribute": "data-population-band", "thresholds": (0., .1, .5, 1., 2., 5.)},
}


def expected_classes(theme: str, values: np.ndarray) -> np.ndarray:
    """Classify numeric observations by the published absolute legend units."""
    values = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{theme}: source observations must be finite and nonnegative")
    if theme != "population" and np.any(values > 1):
        raise ValueError(f"{theme}: source observations must be fractions within 0..1")
    thresholds = np.asarray(SPECS[theme]["thresholds"], dtype=np.float64)
    if theme == "population":
        # The actual surface uses log(1+density), preserving observations
        # while tracing the same absolute people/km² legend boundaries.
        result = np.digitize(np.log1p(values), np.log1p(thresholds)).astype(np.int16)
        result[values == 0] = 0
        return result
    return np.digitize(values, thresholds).astype(np.int16)


def export_regions(path: Path, attribute: str, *, physical_land, dimensions):
    """Decode the export's actual owned clip, checked against physical authority."""
    root = ET.parse(path).getroot()
    if (root.tag != "{http://www.w3.org/2000/svg}svg"
            or root.get("data-surface-contract") != "shared-land-clip"
            or [float(value) for value in root.get("viewBox", "").split()]
            != [0., 0., float(dimensions[1]), float(dimensions[0])]):
        raise ValueError(f"{path.name}: scalar export must declare its native shared-land-clip contract")
    parents = {child: parent for parent in root.iter() for child in parent}
    definitions = root.findall(".//s:clipPath", _svg.NS)
    if (len(definitions) != 1
            or definitions[0].get("id") != "land-silhouette-clip"
            or definitions[0].get("clipPathUnits") != "userSpaceOnUse"
            or parents[definitions[0]].tag != "{http://www.w3.org/2000/svg}defs"
            or parents.get(parents[definitions[0]]) is not root
            or parents[definitions[0]].get("transform") or parents[definitions[0]].get("style")
            or definitions[0].get("transform") or definitions[0].get("clip-path")):
        raise ValueError(f"{path.name}: export must own one native land-silhouette-clip definition")
    if sum(node.get("id") == "land-silhouette-clip" for node in root.iter()) != 1:
        raise ValueError(f"{path.name}: owned physical clip ID must be unique")
    clip_paths = list(definitions[0])
    if not clip_paths or any(
            node.tag != "{http://www.w3.org/2000/svg}path"
            or node.get("clip-rule") != "evenodd" or not node.get("d")
            or node.get("transform") or node.get("clip-path") or node.get("style")
            for node in clip_paths):
        raise ValueError(f"{path.name}: owned physical clip must use explicit evenodd native paths")
    owned_land = _svg.union_paths([node.attrib for node in clip_paths])
    if not owned_land.is_valid or not owned_land.equals(physical_land):
        raise ValueError(f"{path.name}: owned physical clip differs from the published land authority")
    partitions = root.findall(".//s:g[@data-partition='mutually-exclusive']", _svg.NS)
    if len(partitions) != 1:
        raise ValueError(f"{path.name}: scalar export must contain exactly one categorical partition")
    regions = []

    def attributes(node):
        result = dict(node.attrib)
        for rule in result.get("style", "").split(";"):
            if ":" in rule:
                name, value = rule.split(":", 1)
                result[name.strip()] = value.strip()
        return result

    def clip_depth(node, inherited):
        values = attributes(node)
        if values.get("transform"):
            raise ValueError(f"{path.name}: unexpected transform on native scalar paint")
        reference = values.get("clip-path")
        if reference and reference != "none":
            if reference != "url(#land-silhouette-clip)":
                raise ValueError(f"{path.name}: scalar paint uses a foreign physical clip")
            if inherited:
                raise ValueError(f"{path.name}: scalar paint uses a secondary physical clip")
            return 1
        return inherited

    ancestors, ancestor = [], parents.get(partitions[0])
    while ancestor is not None:
        if _tiles.tag(ancestor) in {"defs", "clipPath", "pattern", "mask"}:
            raise ValueError(f"{path.name}: scalar partition cannot be a definition")
        ancestors.append(ancestor)
        ancestor = parents.get(ancestor)
    inherited_clip = 0
    inherited_fill, inherited_opacity = "black", 1.
    for ancestor in reversed(ancestors):
        inherited_clip = clip_depth(ancestor, inherited_clip)
        values = attributes(ancestor)
        inherited_fill = values.get("fill", inherited_fill)
        inherited_opacity *= float(values.get("opacity", "1")) * float(values.get("fill-opacity", "1"))
        if values.get("display") == "none" or values.get("visibility") == "hidden" or "hidden" in values:
            inherited_opacity = 0.

    def visit(node, owner=None, fill="black", opacity=1., clip=0):
        if _tiles.tag(node) in {"defs", "clipPath", "pattern", "mask"}:
            return
        if not node.tag.startswith("{http://www.w3.org/2000/svg}"):
            raise ValueError(f"{path.name}: scalar paint must use the SVG namespace")
        if _tiles.tag(node) not in {"g", "path", "title", "desc"}:
            raise ValueError(f"{path.name}: scalar partition contains unsupported visible geometry")
        values = attributes(node)
        clip = clip_depth(node, clip)
        value = node.get(attribute)
        if value is not None and value != "partition":
            if not value.isdigit():
                raise ValueError(f"{path.name}: scalar class must be a nonnegative integer")
            owner = int(value)
        fill = values.get("fill", fill)
        opacity *= float(values.get("opacity", "1")) * float(values.get("fill-opacity", "1"))
        if values.get("display") == "none" or values.get("visibility") == "hidden" or "hidden" in values:
            return
        if _tiles.tag(node) == "path" and fill != "none" and opacity > 0:
            if not clip:
                raise ValueError(f"{path.name}: scalar paint lacks its owned physical clip")
            if owner is None or values.get("fill-rule") != "evenodd":
                raise ValueError(f"{path.name}: scalar paint requires its numeric class and explicit evenodd fill")
            if values.get("stroke", "none") != "none" and float(values.get("stroke-width", "0")) > 0:
                raise ValueError(f"{path.name}: scalar area cannot depend on an outline covering a gap")
            regions.append((owner, _svg.path_geometry(node.get("d", "")).intersection(owned_land)))
        for child in node:
            visit(child, owner, fill, opacity, clip)

    visit(partitions[0], fill=inherited_fill, opacity=inherited_opacity, clip=inherited_clip)
    return regions


def classify_paint(regions, points, maximum_class: int):
    """Query actual polygons in SVG paint order, including shared boundaries."""
    painted = np.full(len(points), -1, dtype=np.int16)
    grouped = {}
    for owner, geometry in regions:
        if type(owner) is not int or not 0 <= owner <= maximum_class:
            raise ValueError("Delivered scalar class lies outside its absolute legend")
        if geometry.is_empty:
            continue
        if not geometry.is_valid:
            raise ValueError("Delivered scalar paint is invalid; no repair is applied by acceptance")
        shapely.prepare(geometry)
        painted[shapely.covers(geometry, points)] = owner
        grouped.setdefault(owner, []).append(geometry)
    classes = {owner: shapely.union_all(parts) for owner, parts in grouped.items()}
    return painted, classes


def scalar_metrics(theme: str, values, land, regions, *, offset=(0, 0)):
    values = np.asarray(values)
    rows, columns = np.nonzero(np.asarray(land, dtype=bool))
    expected = expected_classes(theme, values)[rows, columns]
    points = shapely.points(columns + .5 + offset[0], rows + .5 + offset[1])
    actual, classes = classify_paint(regions, points, len(SPECS[theme]["thresholds"]))
    wrong = actual != expected
    missing = actual < 0
    expected_counts = {str(int(owner)): int(count) for owner, count in zip(*np.unique(expected, return_counts=True), strict=True)}
    painted_counts = {str(int(owner)): int(count) for owner, count in zip(*np.unique(actual, return_counts=True), strict=True)}
    overlap = sum(first.intersection(second).area for index, first in enumerate(classes.values())
                  for second in list(classes.values())[index + 1:])
    return {
        "checkedLandCenters": int(len(points)), "wrongClassCenters": int(np.count_nonzero(wrong & ~missing)),
        "unpaintedLandCenters": int(np.count_nonzero(missing)), "mismatchCount": int(np.count_nonzero(wrong)),
        "expectedClassLandCells": expected_counts, "paintedClassLandCells": painted_counts,
        "classOverlapArea": float(overlap),
        "examples": [{"column": int(columns[index] + offset[0]), "row": int(rows[index] + offset[1]),
                      "sourceValue": float(values[rows[index], columns[index]]),
                      "expectedClass": int(expected[index]), "paintedClass": int(actual[index])}
                     for index in np.flatnonzero(wrong)[:10]],
    }, shapely.union_all(list(classes.values()))


def _status(metrics):
    return "ok" if metrics["mismatchCount"] == 0 and metrics["classOverlapArea"] < .01 and metrics["uncoveredLandArea"] < .01 else "failed"


def _check_exports(review, water, fields, land):
    results = {}
    for theme, spec in SPECS.items():
        path = review / spec["export"]
        regions = export_regions(path, spec["attribute"], physical_land=land, dimensions=water.shape)
        metrics, paint = scalar_metrics(theme, fields[spec["field"]], water == 0, regions)
        metrics.update(uncoveredLandArea=float(land.difference(paint).area), exportBytes=path.stat().st_size)
        metrics["status"] = _status(metrics)
        results[theme] = metrics
        print(json.dumps({"consumer": "full-quality", "theme": theme, "status": metrics["status"],
                          "mismatchCount": metrics["mismatchCount"]}), flush=True)
    return results


def _check_detail(review, manifest, water, fields):
    results = {theme: {"checkedLandCenters": 0, "wrongClassCenters": 0, "unpaintedLandCenters": 0,
                       "mismatchCount": 0, "uncoveredLandArea": 0., "classOverlapArea": 0., "examples": []}
               for theme in SPECS}
    checked = 0
    for row in range(manifest["rows"]):
        for column in range(manifest["columns"]):
            tile = _tiles.read_tile(review, manifest, "detail", row, column)
            x, y, width, height = tile["payload"]["bounds"]
            local_land = water[y:y + height, x:x + width] == 0
            for theme, spec in SPECS.items():
                regions = _tiles.painted_regions(tile["payload"]["themes"][theme], tile,
                                                   owner_attribute=spec["attribute"])
                metrics, paint = scalar_metrics(theme, fields[spec["field"]][y:y + height, x:x + width],
                                                 local_land, regions, offset=(x, y))
                metrics["uncoveredLandArea"] = float(tile["land"].difference(paint).area)
                target = results[theme]
                for name in ("checkedLandCenters", "wrongClassCenters", "unpaintedLandCenters", "mismatchCount",
                             "uncoveredLandArea", "classOverlapArea"):
                    target[name] += metrics[name]
                for example in metrics["examples"][:max(0, 10 - len(target["examples"]))]:
                    target["examples"].append({"tile": tile["key"], **example})
            checked += 1
        if (row + 1) % 4 == 0 or row + 1 == manifest["rows"]:
            print(json.dumps({"consumer": "actual-detail-tiles", "checkedTiles": checked,
                              "totalTiles": manifest["rows"] * manifest["columns"],
                              "mismatchCount": {theme: value["mismatchCount"] for theme, value in results.items()}}), flush=True)
    for theme, metrics in results.items():
        metrics["status"] = _status(metrics)
        print(json.dumps({"consumer": "actual-detail-tiles", "theme": theme, "status": metrics["status"],
                          "mismatchCount": metrics["mismatchCount"]}), flush=True)
    return {"checkedTiles": checked, "themes": results}


def check_review(world: Path, review: Path):
    started = time.perf_counter()
    manifest = _tiles.read_manifest(review)
    if not set(SPECS) <= set(manifest["themes"]):
        raise ValueError("Published tiles must expose all four numeric themes")
    with np.load(world / "grid/world-grid.npz", allow_pickle=False) as source:
        water = source["water"]
    if water.shape != (manifest["height"], manifest["width"]):
        raise ValueError("Canonical grid dimensions differ from published scalar tiles")
    with np.load(world / "thematic/human-capacity.npz", allow_pickle=False) as source:
        fields = {spec["field"]: source[spec["field"]] for spec in SPECS.values()}
    for theme, spec in SPECS.items():
        values = fields[spec["field"]]
        if values.shape != water.shape or np.any(values[water != 0] != 0):
            raise ValueError(f"{theme}: saved numeric field must align with land and be zero on water")
        expected_classes(theme, values)
    _parser, _physical, _lakes, land = _svg.published_surfaces(review)
    exports = _check_exports(review, water, fields, land)
    detail = _check_detail(review, manifest, water, fields)
    passed = all(metrics["status"] == "ok" for metrics in exports.values()) and all(
        metrics["status"] == "ok" for metrics in detail["themes"].values())
    return {
        "schema": "published-scalar-capacity-v1", "status": "ok" if passed else "failed",
        "source": "thematic/human-capacity.npz", "sourceNativeLandCenters": int(np.count_nonzero(water == 0)),
        "criterion": "Each full-quality export and actual detail tile must paint the source numeric class at every native land centre; total missing/overlap area below .01 native cell squared per theme.",
        "legend": {theme: {"field": spec["field"], "thresholds": list(spec["thresholds"]),
                           "units": "people/km²; contour surface log(1+density)" if theme == "population" else "fraction",
                           "zeroPopulationClass": 0 if theme == "population" else None}
                   for theme, spec in SPECS.items()},
        "fullQualityExports": exports, "detail": detail, "elapsedSeconds": round(time.perf_counter() - started, 3),
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("world", type=Path)
    cli.add_argument("--review", type=Path, required=True)
    cli.add_argument("--report", type=Path)
    args = cli.parse_args()
    report = check_review(args.world, args.review)
    destination = args.report or args.review / "scalar-capacity-check.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(destination),
                      "elapsedSeconds": report["elapsedSeconds"]}), flush=True)
    raise SystemExit(0 if report["status"] == "ok" else 1)
