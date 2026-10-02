"""Append supported landforms to a saved world without rewriting its arrays."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from world_atlas.core.model import WorldGrid
from world_atlas.core.presentation import society_content_digest
from world_atlas.core.procedural_planet import load_surface_bundle
from world_atlas.core.society.fictional_names import procedural_name_lexicon
from world_atlas.core.society.geographic_features import enrich_saved_geographic_features
from world_atlas.core.society.storage import load_society
from world_atlas.core.society.world_identity import naming_audit
from world_atlas.core.thematic import derive_thematic_layers
from world_atlas.core.ecological_sources import derive_ecological_sources


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def enrich_world(world: Path) -> dict:
    world=world.resolve(strict=True)
    started=time.monotonic()
    frozen_names=("grid/world-grid.npz","grid/world-grid.json","source/physical-fields.npz",
                  "thematic/human-capacity.npz","society/society.npz","world-settings.json","naming-exclusions.json")
    frozen={name:sha256(world/name) for name in frozen_names}
    grid=WorldGrid.load(world/"grid")
    original=load_society(world/"society",expected_grid_digest=grid.content_digest())
    record=json.loads((world/"regeneration.json").read_text(encoding="utf8"))
    if frozen["source/physical-fields.npz"].lower()!=record["fieldSha256"].lower():
        raise ValueError("saved landform source is not the accepted physical field")
    repair_path=world/"repair-check.json"
    repair=json.loads(repair_path.read_text(encoding="utf8"))
    if repair["repairedSocietyDigest"]!=society_content_digest(original):
        raise ValueError("saved society differs from its completed repair evidence")
    physical=load_surface_bundle(world/"source/physical-fields.npz")
    request=grid.metadata["societyGeneration"]
    lexicon=procedural_name_lexicon(int(request["namingSeed"]),forbidden=request["forbiddenNames"])
    enriched,evidence=enrich_saved_geographic_features(grid,original,thematic=derive_thematic_layers(
        grid,ecological_sources=derive_ecological_sources(grid,physical)),
        raw_elevation_m=physical.relative_elevation_m,lexicon=lexicon)
    if enriched.geographic_features == original.geographic_features and "geographyEnrichment" in repair:
        # A replay verifies support without replacing the original 335-record
        # enrichment evidence with a misleading zero-change second baseline.
        if any(sha256(world/name)!=value for name,value in frozen.items()):
            raise AssertionError("landform support verification changed a frozen input")
        return repair["geographyEnrichment"]
    metadata_path=world/"society/society.json"
    metadata=json.loads(metadata_path.read_text(encoding="utf8"))
    prior_records=metadata["geographicFeatures"]
    updated_records=[asdict(item) for item in enriched.geographic_features]
    if updated_records[:len(prior_records)]!=prior_records:
        raise AssertionError("geographic enrichment changed a prior saved record")
    metadata["geographicFeatures"]=updated_records
    new_digest=society_content_digest(enriched)
    name_digest=naming_audit(enriched,request["forbiddenNames"])["nameDigest"]
    evidence.update(status="ok",sourceSocietyDigest=society_content_digest(original),
                    enrichedSocietyDigest=new_digest,unchangedFileSha256=frozen,
                    nameDigest=name_digest,
                    elapsedSeconds=round(time.monotonic()-started,3))
    repair.update(repairedSocietyDigest=new_digest,geographyEnrichment=evidence)
    record["nameDigest"]=name_digest
    # Only named-record metadata and its evidence change. Recompressing the
    # unchanged NPZ would destroy a useful byte-for-byte provenance guarantee.
    for path,document in ((metadata_path,metadata),(repair_path,repair),(world/"regeneration.json",record)):
        temporary=path.with_name(path.name+".landform-writing")
        temporary.write_text(json.dumps(document,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
        temporary.replace(path)
    loaded=load_society(world/"society",expected_grid_digest=grid.content_digest())
    if society_content_digest(loaded)!=new_digest:
        raise AssertionError("saved landform metadata differs from the enriched model")
    if any(sha256(world/name)!=value for name,value in frozen.items()):
        raise AssertionError("landform enrichment changed a frozen physical/human array or setting")
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world",type=Path)
    args=parser.parse_args()
    report=enrich_world(args.world)
    print(json.dumps({key:value for key,value in report.items() if key not in ("addedFeatures","support","unchangedFileSha256")},ensure_ascii=True))


if __name__=="__main__":
    main()
