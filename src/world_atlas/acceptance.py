"""Check actual published geography, identities and transport before handoff."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from world_atlas.core.model import WorldGrid
from world_atlas.core.society.storage import load_society
from world_atlas.core.society.transport import _path_cells
from world_atlas.core.society.world_identity import assign_world_identity, naming_audit
from world_atlas.core.tectonic_review import derive_tectonic_review


def verify_release(output: Path) -> dict:
    grid = WorldGrid.load(output / "grid")
    society = load_society(output / "review", expected_grid_digest=grid.content_digest())
    forbidden = json.loads((output / "naming-exclusions.json").read_text(encoding="utf-8"))["forbidden"]
    seed = grid.metadata["societyGeneration"]["seed"]
    audit = naming_audit(society, forbidden)
    renamed = assign_world_identity(society, seed=seed, forbidden=forbidden)
    assert audit == naming_audit(renamed, forbidden), "canonical names must be idempotent"
    with np.load(output / "source/physical-fields.npz", allow_pickle=False) as source:
        coastline_errors = int(np.count_nonzero(source["land_mask"] != (grid.water == 0)))
        tectonics = derive_tectonic_review(grid)
        plate_errors = int(np.count_nonzero(source["plate_id"] != tectonics.plate_id))
    svg = ET.parse(output / "review/tectonic.svg")
    plate_names = [node.text for node in svg.iter() if node.attrib.get("class") == "plate-name"]
    assert plate_names == [plate.name for plate in tectonics.plates], "published plate names differ from model"
    extra_names = plate_names + [grid.metadata["worldProfile"]["name"]]
    extra_matches = sorted({(name, old) for name in extra_names for old in forbidden
                            if old == name or (len(old) >= 2 and old in name)})
    sea_crossings, road_crossings = [], []
    for route in society.transport.routes:
        if route.mode not in {"sea", "road"}:
            continue
        cells = _path_cells(route.path, grid.shape)
        # Ports may terminate on their own land cell; no intermediate land is
        # permitted on shipping paths. Rivers have land cells plus bridge data.
        invalid = [cell for cell in (cells[1:-1] if route.mode == "sea" else cells)
                   if (grid.water[cell] == 0) == (route.mode == "sea")]
        if invalid:
            (sea_crossings if route.mode == "sea" else road_crossings).append(
                {"route": route.identifier, "count": len(invalid), "examples": invalid[:3]})
    land = grid.water == 0
    unpartitioned = int(np.count_nonzero((society.politics.state_id > 0) & (society.provinces.province_id == 0)))
    record = {"worldName": grid.metadata["worldProfile"]["name"], "counts": audit["counts"],
        "oldNameMatches": audit["oldNameMatches"] + extra_matches, "duplicateNames": audit["duplicateNames"],
        "coastlineCellMismatches": coastline_errors, "plateCellMismatches": plate_errors,
        "plateNames": plate_names, "leftRightLandCells": int(land[:, [0, -1]].sum()),
        "unpartitionedStateCells": unpartitioned, "seaRoutesAcrossLand": sea_crossings,
        "roadsAcrossWater": road_crossings, "nameReplayIdentical": True,
        "htmlSha256": hashlib.sha256((output / "review/index.html").read_bytes()).hexdigest()}
    (output / "review/acceptance-check.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert not record["oldNameMatches"] and not record["duplicateNames"]
    assert not coastline_errors and not plate_errors and not unpartitioned and not record["leftRightLandCells"]
    assert not sea_crossings and not road_crossings, "transport must remain on its navigable surface"
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(verify_release(parser.parse_args().output), ensure_ascii=False, indent=2))
