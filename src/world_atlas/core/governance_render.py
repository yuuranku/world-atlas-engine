"""Nominal realms reuse the same actual province coverage as local government."""

import html
import json
from pathlib import Path

import numpy as np
import shapely

from .society.governance import derive_governance, write_governance
from .cartographic_tiles import TileFeature


def write_governance_overlay(directory: Path, grid, society, faces, province_labels):
    """Dissolve existing visible province faces; never independently smooth them."""
    from .render import _geometry_polygons
    from .svg_paths import integer_subpath_data

    governance = derive_governance(grid, society)
    write_governance(directory, grid, society, governance)
    country = np.zeros(len(society.provinces.provinces)+1, dtype=int)
    for province in society.provinces.provinces:
        country[province.identifier] = province.state_identifier
    nominal = np.zeros(len(society.politics.states)+1, dtype=int)
    for record in governance["countries"]:
        nominal[record["id"]] = record["nominalRealmId"]
    labels = nominal[country[np.asarray(province_labels, dtype=int)]]
    members = {}
    states = {state.identifier: state for state in society.politics.states}
    for record in governance["countries"]:
        members.setdefault(record["nominalRealmId"], []).append(record["id"])
    groups = []
    features = []
    area_drift = 0.0
    for realm, countries in sorted(members.items()):
        if len(countries) < 2:
            continue  # Independent countries already have an actual border.
        selected = [face for face, owner in zip(faces, labels, strict=True) if owner == realm]
        geometry = shapely.union_all(selected)
        features.append(TileFeature(geometry.boundary, {
            "fill": "none", "stroke": "#73583f", "stroke-width": "2.2",
            "stroke-dasharray": "10 5", "stroke-linejoin": "round",
            "vector-effect": "non-scaling-stroke", "clip": "land",
            "data-nominal-realm": str(realm), "data-local-countries": str(len(countries)),
            "aria-label": states[realm].name + " · 名义势力范围",
            "data-screen-stroke": "2.2", "data-screen-dash": "10 5",
        }, "nominal-realms"))
        area_drift = max(area_drift, abs(float(geometry.area)-sum(float(face.area) for face in selected)))
        rings = []
        for polygon in _geometry_polygons(geometry):
            for ring in (polygon.exterior, *polygon.interiors):
                coords = np.asarray(ring.coords, dtype=float)
                delivered = np.rint(coords*1e8).astype(np.int64)
                rings.append(integer_subpath_data([tuple(point) for point in delivered], closed=True))
        groups.append(f'<path data-nominal-realm="{realm}" data-local-countries="{len(countries)}" d="{"".join(rings)}"><title>{html.escape(states[realm].name)} · 名义势力范围</title></path>')
    if area_drift > .01:
        raise ValueError("nominal realm dissolve changed actual local territory")
    height, width = grid.shape
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><g id="nominal-realm-geometries" fill="none" stroke="#73583f" stroke-width="2.2" stroke-dasharray="10 5" stroke-linejoin="round" vector-effect="non-scaling-stroke">{"".join(groups)}</g></svg>'
    # Path-level vector effect is needed because the attribute is not inherited.
    svg = svg.replace('<path ', '<path vector-effect="non-scaling-stroke" ')
    encoded = svg.encode("utf-8")
    if len(encoded) > 10*1024**2:
        raise ValueError("nominal realm overlay exceeded its document budget")
    (directory/"governance-overlay.svg").write_bytes(encoded)
    report = {"status": "ok", "schema": "world-atlas-governance-render-check-v1", "era": governance["era"],
              "actualCountries": len(states), "nominalRelations": len(governance["relations"]),
              "compoundRealms": len(groups), "maximumAreaDriftNativeSquared": area_drift,
              "source": "same visible province faces as actual fills and borders", "bytes": len(encoded)}
    (directory/"governance-render-check.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return governance, report, tuple(features)
