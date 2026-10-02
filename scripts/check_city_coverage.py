"""Check published political paint owns every canonical settlement centre.

Read delivered tile geometry and the saved society arrays independently of the
renderer. A presentation change must not put a city in a different country or
province, leave it unpainted, or place it on an ambiguous displayed border.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import shapely


_spec = importlib.util.spec_from_file_location(
    "published_tile_coverage", Path(__file__).with_name("check_atlas_tiles.py"),
)
_tiles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tiles)


def owner_metrics(settlements, source_owners, regions):
    labels = tuple(sorted(regions))
    index = shapely.STRtree([regions[label] for label in labels])
    mismatches = []
    for settlement in settlements:
        row, column = settlement["row"], settlement["column"]
        point = shapely.Point(column + .5, row + .5)
        expected = int(source_owners[row, column])
        displayed = sorted(labels[item] for item in index.query(point, predicate="intersects"))
        if displayed == [expected]:
            continue
        region = regions.get(expected)
        mismatches.append({
            "id": settlement["identifier"], "name": settlement["name"],
            "point": list(point.coords[0]), "expectedOwnerId": expected,
            "displayedOwnerIds": displayed,
            "distanceToExpectedRegion": None if region is None else float(region.distance(point)),
        })
    return {"checkedCities": len(settlements), "mismatchCount": len(mismatches),
            "mismatches": mismatches}


def overview_regions(review, manifest, theme, attribute):
    tile = _tiles.read_overview(review,manifest,theme)
    return _tiles.tile_owner_regions(tile,theme,attribute)


def check_review(review: Path):
    society = json.loads((review / "society.json").read_text(encoding="utf-8"))
    manifest = _tiles.read_manifest(review)
    if not {"political", "provinces"} <= set(manifest["themes"]):
        raise ValueError("Actual atlas tiles must publish political and provinces themes")
    by_tile = {}
    for settlement in society["settlements"]:
        row, column = settlement["row"], settlement["column"]
        if not (0 <= row < manifest["height"] and 0 <= column < manifest["width"]):
            raise ValueError("Canonical city centre is outside the published tile world")
        key = (row // manifest["tileSize"], column // manifest["tileSize"])
        by_tile.setdefault(key, []).append(settlement)
    views = {}
    # City refinement adds physical relief above the existing detail surface;
    # the political and provincial partitions remain that same detail tile.
    base_levels = _tiles.base_levels(manifest)
    with np.load(review / "society.npz", allow_pickle=False) as source:
        owners = {name:source[name] for name in ("state_id", "province_id")}
        if any(array.shape != (manifest["height"], manifest["width"]) for array in owners.values()):
            raise ValueError("Canonical administrative arrays do not match actual atlas tile dimensions")
        for level in base_levels:
            view = {name:{"checkedCities":0,"mismatchCount":0,"mismatches":[]}
                    for name in ("countries","provinces")}
            for (row,column), settlements in by_tile.items():
                tile = _tiles.read_tile(review,manifest,level["id"],row,column)
                for name,theme,array_name,attribute in (
                    ("countries","political","state_id","data-state"),
                    ("provinces","provinces","province_id","data-province"),
                ):
                    metrics = owner_metrics(settlements,owners[array_name],
                                            _tiles.tile_owner_regions(tile,theme,attribute))
                    for field in ("checkedCities","mismatchCount"):
                        view[name][field] += metrics[field]
                    view[name]["mismatches"].extend(metrics["mismatches"])
            views[level["id"]] = view
        views["overview"] = {
            name:owner_metrics(society["settlements"],owners[array_name],
                               overview_regions(review,manifest,theme,attribute))
            for name,theme,array_name,attribute in (
                ("countries","political","state_id","data-state"),
                ("provinces","provinces","province_id","data-province"),
            )
        }
    summaries = {}
    for name in ("countries","provinces"):
        mismatches = [{**failure,"view":view} for view,metrics in views.items()
                      for failure in metrics[name]["mismatches"]]
        summaries[name] = {"checkedCities":len(society["settlements"]),
                           "checkedCityViews":sum(metrics[name]["checkedCities"] for metrics in views.values()),
                           "mismatchCount":len(mismatches),"mismatches":mismatches}
    countries,provinces = summaries["countries"],summaries["provinces"]
    return {
        "schema": "published-city-ownership-v4",
        "status": "ok" if countries["mismatchCount"] == provinces["mismatchCount"] == 0 else "failed",
        "gridDigest": society["gridDigest"],
        "consumer": "Actual overview-political.svg/overview-provinces.svg and the manifest tile containing each city at every detail level",
        "criterion": "At every detail level and in overview, every canonical city centre has exactly its canonical owner in actual displayed paint under that view's authoritative physical land clip.",
        "checkedCityTiles": len(by_tile)*len(base_levels),
        "cityReliefOwnership": "The sparse physical relief overlay does not replace thematic paint; city-detail retains the checked detail country and province partitions.",
        "countries": countries, "provinces": provinces,"views":views,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("review", type=Path)
    cli.add_argument("--report", type=Path)
    args = cli.parse_args()
    report = check_review(args.review)
    destination = args.report or args.review / "city-coverage-check.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(destination),
                      "checkedCities": report["countries"]["checkedCities"],
                      "countryMismatches": report["countries"]["mismatchCount"],
                      "provinceMismatches": report["provinces"]["mismatchCount"]}))
    raise SystemExit(0 if report["status"] == "ok" else 1)
