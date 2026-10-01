"""Check exported city geometry with Shapely, independent of the JS clipper."""
import json
from pathlib import Path
import sys

import shapely

source = Path(sys.argv[1])
items = json.loads(source.read_text(encoding="utf-8"))
failures = []
checked = 0
polygon = lambda points: shapely.Polygon([(p["column"], p["row"]) for p in points])
for city in items:
    compounds = {item["compoundId"]: polygon(item["compoundPoints"]) for item in city["landmarks"]}
    for landmark in city["landmarks"]:
        checked += 1
        if not compounds[landmark["compoundId"]].buffer(1e-8).covers(polygon(landmark["points"])):
            failures.append([city["id"], "landmark outside its reserved compound", landmark["kind"]])
    for plaza in city["plazas"]:
        if plaza.get("precinct"):
            interior = polygon(plaza["points"]).buffer(-1e-8)
            for wall in city["walls"]:
                line = shapely.LineString([(p["column"], p["row"]) for p in wall])
                if interior.intersects(line):
                    failures.append([city["id"], "city wall cuts through palace"])
    if city.get("harbor"):
        reserve = polygon(city["harbor"]["reservedLand"])
        for landmark in city["landmarks"]:
            if reserve.intersection(polygon(landmark["points"])).area > 1e-12:
                failures.append([city["id"], "civic building occupies reserved harbor", landmark["kind"]])
        for building in city["buildings"]:
            if reserve.intersection(polygon(building)).area > 1e-12:
                failures.append([city["id"], "house occupies reserved harbor"])
report = {"checkedCities": len(items), "checkedLandmarks": checked, "failures": failures}
source.with_name("clearance-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report))
if failures:
    raise SystemExit(1)
