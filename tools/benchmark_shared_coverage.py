"""Bounded comparison of common-graph dissolves and real coastal clipping.

This measures shared-display operations, not full-world generation. A raw
population regression fixture exercises real source noding, and a generated
grid measures a larger common graph. Delivered biome paint is used separately
to determine whether independent coast clipping satisfies the union contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import types
import xml.etree.ElementTree as ET

import numpy as np
import shapely
from shapely.affinity import translate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from world_atlas.core.cartographic_features import shared_display_coverage

    baseline_source = subprocess.check_output(
        ["git", "show", "HEAD:src/world_atlas/core/cartographic_features.py"],
        cwd=root, text=True,
    )
    baseline = types.ModuleType("world_atlas.core._coverage_baseline")
    exec(compile(baseline_source, "baseline_cartographic_features.py", "exec"), baseline.__dict__)
    spec = importlib.util.spec_from_file_location(
        "coverage_svg_decoder", root / "scripts" / "check_coastal_coverage.py")
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)

    asset = args.review / "biome.svg"
    svg = ET.parse(asset).getroot()
    # The bounded coastal region retains islands, disconnected paint, holes,
    # and category T nodes from the actual successful world.
    bounds = (750, 250, 1050, 550)
    frame = shapely.box(*bounds)
    paths = svg.findall(".//{*}path[@fill]")
    delivered = np.asarray([
        shapely.intersection(decoder.path_geometry(path.attrib["d"]), frame)
        for path in paths
    ], dtype=object)
    fixture = json.loads((root / "tests" / "fixtures" /
                          "population-grid-noding.json").read_text())
    fixture_bounds = fixture["frame_bounds"]
    yy, xx = np.mgrid[:96, :128]
    boxes = shapely.box(xx.ravel(), yy.ravel(), xx.ravel()+1, yy.ravel()+1)
    labels = ((xx//4 + yy//3) % 6).ravel()
    grid_source = np.asarray([
        translate(shapely.coverage_union_all(boxes[labels == owner]),
                  xoff=2e-10, yoff=2e-10)
        for owner in range(6)
    ], dtype=object)
    workloads = (
        ("actual-population-grid-noding", shapely.from_wkb(fixture["regions_wkb"]),
         np.meshgrid(np.linspace(fixture_bounds[0], fixture_bounds[2], 42)[1:-1],
                     np.linspace(fixture_bounds[1], fixture_bounds[3], 42)[1:-1])),
        ("128x96-common-grid", grid_source, (xx+.5, yy+.5)),
    )
    comparisons = []
    for workload, source, (x, y) in workloads:
        before = shapely.to_wkb(source).tolist()
        runs = []
        outputs = []
        for name, operation in (("baseline", baseline.shared_display_coverage),
                                ("coverage-dissolve", shared_display_coverage)):
            started = time.perf_counter()
            result = operation(source)
            seconds = time.perf_counter() - started
            outputs.append(result)
            runs.append({"name": name, "seconds": seconds,
                         "coverageValid": bool(shapely.coverage_is_valid(result)),
                         "geometriesValid": bool(np.all(shapely.is_valid(result))),
                         "coordinateCount": int(np.sum(shapely.get_num_coordinates(result)))})
        original, improved = outputs
        differences = shapely.symmetric_difference(original, improved, grid_size=1e-8)
        native_mismatches = 0
        for first, last in zip(original, improved, strict=True):
            native_mismatches += int(np.count_nonzero(
                shapely.intersects_xy(first, x, y) != shapely.intersects_xy(last, x, y)))
        comparisons.append({
            "workload": workload,
            "sourceCoordinateCount": int(np.sum(shapely.get_num_coordinates(source))),
            "sourceCoverageValid": bool(shapely.coverage_is_valid(source)),
            "runs": runs, "speedup": runs[0]["seconds"] / runs[1]["seconds"],
            "symmetricDifferenceEmpty": bool(np.all(shapely.is_empty(differences))),
            "nativeWitnesses": int(x.size), "nativeMembershipMismatches": native_mismatches,
            "sourceBytesUnchanged": before == shapely.to_wkb(source).tolist(),
        })
    # The independently clipped coastal faces are examined separately. This
    # is evidence about their contract, not an attempt to fix shared nodes.
    coast_path = svg.find(".//{*}clipPath/{*}path")
    coast = shapely.intersection(decoder.path_geometry(coast_path.attrib["d"]), frame)
    from world_atlas.core.coastal_partition import clip_partition_to_surface
    clipped, _ = clip_partition_to_surface(delivered, np.arange(len(delivered)), coast)
    report = {
        "asset": str(asset.resolve()), "bounds": bounds,
        "comparisons": comparisons,
        "independentlyCoastClippedCoverageValid": bool(shapely.coverage_is_valid(clipped)),
        "independentlyCoastClippedDifferenceArea": float(shapely.symmetric_difference(
            coast, shapely.union_all(clipped)).area),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
