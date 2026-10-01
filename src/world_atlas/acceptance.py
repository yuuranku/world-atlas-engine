"""Check actual published geography, identities and transport before handoff."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from world_atlas.core.model import WorldGrid
from world_atlas.core.procedural_planet import load_surface_bundle
from world_atlas.core.society.storage import load_society
from world_atlas.core.society.administrative_audit import audit_administrative_topology
from world_atlas.core.society.transport import _path_cells
from world_atlas.core.society.world_identity import assign_world_identity, naming_audit
from world_atlas.core.tectonic_review import derive_tectonic_review


def _transport_surface_violations(grid: WorldGrid, routes) -> dict[str, list[dict]]:
    """Return any route drawn on the wrong physical surface.

    Sea lanes may begin and end at ports on land, but their intervening cells
    must remain water.  Roads and rails are independent overland systems and
    must never contain a water cell; bridge eligibility is checked during
    transport generation rather than waived by this release gate.
    """

    violations = {"sea": [], "road": [], "rail": []}
    for route in routes:
        if route.mode not in violations:
            continue
        cells = _path_cells(route.path, grid.shape)
        route_cells = cells[1:-1] if route.mode == "sea" else cells
        invalid = [
            cell
            for cell in route_cells
            if (grid.water[cell] == 0) == (route.mode == "sea")
        ]
        if invalid:
            violations[route.mode].append(
                {"route": route.identifier, "count": len(invalid), "examples": invalid[:3]}
            )
    return violations


def _settlement_surface_violations(grid: WorldGrid, settlements) -> dict[str, list[dict]]:
    """Return settlement anchors that conflict with physical source layers."""

    water: list[dict] = []
    river: list[dict] = []
    for settlement in settlements:
        cell = (settlement.row, settlement.column)
        record = {
            "settlement": settlement.identifier,
            "row": settlement.row,
            "column": settlement.column,
            "siteType": settlement.site_type,
        }
        if int(grid.water[cell]) != 0:
            water.append(record)
        order = int(grid.river_order[cell])
        if order > 0:
            river.append({**record, "riverOrder": order})
    return {"water": water, "river": river}


def verify_release(output: Path) -> dict:
    grid = WorldGrid.load(output / "grid")
    society = load_society(output / "review", expected_grid_digest=grid.content_digest())
    forbidden = json.loads((output / "naming-exclusions.json").read_text(encoding="utf-8"))["forbidden"]
    seed = grid.metadata["societyGeneration"]["namingSeed"]
    audit = naming_audit(society, forbidden)
    renamed = assign_world_identity(society, seed=seed, forbidden=forbidden)
    assert audit == naming_audit(renamed, forbidden), "canonical names must be idempotent"
    with np.load(output / "source/physical-fields.npz", allow_pickle=False) as source:
        coastline_errors = int(np.count_nonzero(source["land_mask"] != (grid.water == 0)))
        tectonics = derive_tectonic_review(
            grid, physical_source=load_surface_bundle(output / "source/physical-fields.npz")
        )
        plate_errors = int(np.count_nonzero(source["plate_id"] != tectonics.plate_id))
    svg = ET.parse(output / "review/tectonic.svg")
    plate_names = [node.text for node in svg.iter() if node.attrib.get("class") == "plate-name"]
    assert plate_names == [plate.name for plate in tectonics.plates], "published plate names differ from model"
    extra_names = plate_names + [grid.metadata["worldProfile"]["name"]]
    extra_matches = sorted({(name, old) for name in extra_names for old in forbidden
                            if old == name or (len(old) >= 2 and old in name)})
    transport_violations = _transport_surface_violations(grid, society.transport.routes)
    settlement_violations = _settlement_surface_violations(grid, society.settlements)
    sea_crossings = transport_violations["sea"]
    road_crossings = transport_violations["road"]
    rail_crossings = transport_violations["rail"]
    land = grid.water == 0
    provenance = json.loads((output / "source/provenance.json").read_text(encoding="utf-8"))
    polar = provenance["recipe"]
    latitude = 90.0 - (np.arange(land.shape[0]) + .5) * 180.0 / land.shape[0]
    polar_projection_rows = ((latitude >= 58.0) & polar["north_polar_continent"]) | ((latitude <= -58.0) & polar["south_polar_continent"])
    polar_errors = []
    for name, row in (("north", 0), ("south", -1)):
        if not np.all(land[row] == polar[f"{name}_polar_continent"]):
            polar_errors.append(name)
    unpartitioned = int(np.count_nonzero((society.politics.state_id > 0) & (society.provinces.province_id == 0)))
    administrative_topology = audit_administrative_topology(grid, society)
    (output / "review/administrative-topology.json").write_text(
        json.dumps(administrative_topology, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record = {"worldName": grid.metadata["worldProfile"]["name"], "counts": audit["counts"],
        "oldNameMatches": audit["oldNameMatches"] + extra_matches, "duplicateNames": audit["duplicateNames"],
        "coastlineCellMismatches": coastline_errors, "plateCellMismatches": plate_errors,
        "plateNames": plate_names, "leftRightLandCells": int(land[~polar_projection_rows][:, [0, -1]].sum()),
        "polarProjectionEdgeLandCells": int(land[polar_projection_rows][:, [0, -1]].sum()), "polarChoiceMismatches": polar_errors,
        "unpartitionedStateCells": unpartitioned, "seaRoutesAcrossLand": sea_crossings,
        "roadsAcrossWater": road_crossings, "railsAcrossWater": rail_crossings,
        "settlementsOnWater": settlement_violations["water"],
        "settlementsOnRiver": settlement_violations["river"],
        "nameReplayIdentical": True,
        "administrativeTopologyStatus": administrative_topology["status"],
        "administrativeTopologySummary": {
            layer: administrative_topology[layer]["summary"] for layer in ("states", "provinces")
        },
        "htmlSha256": hashlib.sha256((output / "review/index.html").read_bytes()).hexdigest()}
    (output / "review/acceptance-check.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert not record["oldNameMatches"] and not record["duplicateNames"]
    assert not coastline_errors and not plate_errors and not unpartitioned and not record["leftRightLandCells"]
    assert not sea_crossings and not road_crossings and not rail_crossings, (
        "transport must remain on its navigable surface"
    )
    assert not settlement_violations["water"] and not settlement_violations["river"], (
        "settlement anchors must remain on buildable land outside represented river channels"
    )
    assert not polar_errors, "polar land must match each independent choice"
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(verify_release(parser.parse_args().output), ensure_ascii=False, indent=2))
