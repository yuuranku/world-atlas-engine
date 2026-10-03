"""Regenerate downstream maps from a verified, already accepted terrain bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import time

import numpy as np

from world_atlas.inputs import attach_world_metadata
from world_atlas.core.baseline import compute_input_fingerprint, load_baseline
from world_atlas.core.config import load_world_config
from world_atlas.core.model import WorldGrid
from world_atlas.core.procedural_planet import PlanetRecipe, load_surface_bundle
from world_atlas.core.render import render_review, _society_generation_request
from world_atlas.core.thematic import derive_thematic_layers
from world_atlas.core.ecological_sources import derive_ecological_sources
from world_atlas.core.society.pipeline import derive_society_layers
from world_atlas.core.society.storage import load_society, save_society
from world_atlas.core.society.world_identity import collect_proper_names, naming_audit
from world_atlas.settings import WorldSettings, load_world_settings
from world_atlas.timing import elapsed_seconds, measure_stage
from world_atlas import __version__


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _reusable_physical_contours(config_path, bundle, config, settings, physical_source):
    """Choose only unmodified extraction records for the same physical source.

    The extractor still verifies its entire field binding when it opens the
    copy. This check never rebinds a manifest or accepts changed source code.
    """
    planet = config["planet"]
    if planet != {**planet, **settings.planet}:
        return None
    from world_atlas.core.physical_contour_stage import current_height_graphs
    identity = {
        "rawElevationSha256": hashlib.sha256(
            np.ascontiguousarray(physical_source.relative_elevation_m).tobytes()).hexdigest(),
        "physicalDiagnostics": dict(physical_source.diagnostics),
    }
    for directory in dict.fromkeys((config_path.parent / "physical-contours",
                                   bundle.parent / "physical-contours")):
        if not current_height_graphs(directory):
            continue
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["binding"]["physicalInput"] == identity:
            return directory
    return None


def prepare_regeneration(
    config_path: Path,
    provenance_path: Path,
    output: Path,
    exclusions: list[Path],
    *,
    settings: WorldSettings,
) -> dict:
    """Validate first; copy inputs exactly into a fresh, self-contained run."""
    config_path, provenance_path, output = config_path.resolve(), provenance_path.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"new world output already exists: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = (config_path.parent / config["source"]["path"]).resolve()
    if sha256(source) != config["source"]["sha256"].upper():
        raise ValueError("accepted terrain image hash mismatch")
    field_record = config["source"]["fieldBundle"]
    bundle = (config_path.parent / field_record["path"]).resolve()
    if sha256(bundle) != field_record["sha256"].upper():
        raise ValueError("accepted physical field hash mismatch")
    physical_source = load_surface_bundle(bundle)
    contour_checkpoint = _reusable_physical_contours(
        config_path, bundle, config, settings, physical_source)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    forbidden = {"埃瑞诺", "中洛", "中洛天", "米斯拉"}
    inventory = []
    for path in exclusions:
        document = json.loads(path.read_text(encoding="utf-8"))
        forbidden.update(collect_proper_names(document))
        forbidden.update(document.get("forbidden", ()))
        inventory.append({"file": path.name, "sha256": sha256(path)})
    (output / "source").mkdir(parents=True)
    shutil.copy2(source, output / "source/physical-reference.png")
    shutil.copy2(bundle, output / "source/physical-fields.npz")
    shutil.copy2(provenance_path, output / "source/provenance.json")
    source_checkpoint = bundle.parent / "physical-grid"
    if source_checkpoint.is_dir():
        shutil.copytree(source_checkpoint, output / "source/physical-grid")
    if contour_checkpoint is not None:
        shutil.copytree(contour_checkpoint, output / "physical-contours")
    config["source"]["path"] = "source/physical-reference.png"
    config["source"]["fieldBundle"]["path"] = "source/physical-fields.npz"
    config["output"]["directory"] = "."
    config["planet"].update(settings.planet)
    write_json(output / "worldgen.json", config)
    write_json(output / "world-settings.json", settings.document())
    write_json(output / "naming-exclusions.json", {"forbidden": sorted(forbidden), "sources": inventory})
    return {
        "config": output / "worldgen.json",
        "provenance": provenance,
        "forbidden": tuple(sorted(forbidden)),
        "settings": settings,
        "physicalContoursCheckpoint": {
            "state": "hit" if contour_checkpoint is not None else "miss",
            "source": str(contour_checkpoint) if contour_checkpoint is not None else None,
        },
    }


def _load_verified_physical_checkpoint(config_path: Path, checkpoint: Path) -> WorldGrid | None:
    """Load a terrain-owned grid only when every physical input still agrees.

    The cache is deliberately not a broad memoization layer: its fingerprint
    includes the source bytes, surface bundle, normalized physical settings,
    importer identity, and cell-truth source digest. A missing, malformed, or
    stale checkpoint simply yields ``None`` and forces the canonical rebuild.
    """

    manifest_path = checkpoint / "checkpoint.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "world-atlas-physical-checkpoint-v1":
            return None
        grid = WorldGrid.load(checkpoint)
        config = load_world_config(config_path)
        fingerprint = compute_input_fingerprint(config.source.path, config)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if (
        manifest.get("inputFingerprint") != fingerprint
        or grid.metadata.get("inputFingerprint") != fingerprint
        or manifest.get("gridDigest") != grid.content_digest()
    ):
        return None
    return grid


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_accepted_world(
    config_path: Path,
    provenance_path: Path,
    output: Path,
    exclusions: list[Path],
    *,
    settings: WorldSettings,
) -> dict:
    started = time.monotonic()
    prepared = prepare_regeneration(
        config_path,
        provenance_path,
        output,
        exclusions,
        settings=settings,
    )
    provenance = prepared["provenance"]
    recipe = PlanetRecipe(**provenance["recipe"])
    output = output.resolve()
    write_json(output / "regeneration.json", {"schema": "accepted-world-v2",
        "engineVersion": __version__,
        "humanSeed": settings.human_seed, "namingSeed": settings.naming_seed,
        "terrainSeed": recipe.seed, "terrainSchema": provenance["schema"], "status": "building",
        "sourceSha256": sha256(output / "source/physical-reference.png"),
        "fieldSha256": sha256(output / "source/physical-fields.npz"),
        "settingsSha256": sha256(output / "world-settings.json"),
        "physicalContoursCheckpoint": prepared["physicalContoursCheckpoint"]})
    print(f"Accepted terrain verified; {len(prepared['forbidden'])} old names excluded. Building climate and hydrology.", flush=True)
    checkpoint = output / "source/physical-grid"
    with measure_stage(output, "climate-hydrology"):
        physical_grid = _load_verified_physical_checkpoint(prepared["config"], checkpoint)
        checkpoint_state = "hit" if physical_grid is not None else "miss"
        if physical_grid is None:
            physical_grid = load_baseline(prepared["config"])
            physical_grid.save(checkpoint)
            write_json(checkpoint / "checkpoint.json", {
                "schema": "world-atlas-physical-checkpoint-v1",
                "inputFingerprint": physical_grid.metadata["inputFingerprint"],
                "gridDigest": physical_grid.content_digest(),
            })
        grid = attach_world_metadata(physical_grid, recipe, provenance,
            bundle_path=output / "source/physical-fields.npz", bundle_sha256=sha256(output / "source/physical-fields.npz"),
            settings=settings, forbidden_names=prepared["forbidden"])
        grid.save(output / "grid")
    record = json.loads((output / "regeneration.json").read_text(encoding="utf-8"))
    record["physicalCheckpoint"] = {
        "state": checkpoint_state,
        "source": "source/physical-grid",
    }
    write_json(output / "regeneration.json", record)
    print(f"Physical grid saved ({time.monotonic() - started:.0f}s). Building society and review layers.", flush=True)
    return publish_accepted_world(output, started=started)


def publish_accepted_world(output: Path, *, started: float | None = None) -> dict:
    """Publish downstream layers from the validated physical checkpoint."""
    started = time.monotonic() if started is None else started
    output = output.resolve()
    record = json.loads((output / "regeneration.json").read_text(encoding="utf-8"))
    if record["engineVersion"] != __version__:
        raise ValueError("checkpoint engine version differs from the running engine")
    if record["status"] != "building":
        raise ValueError("only an unfinished build can be published")
    for key, relative in (
        ("sourceSha256", "source/physical-reference.png"),
        ("fieldSha256", "source/physical-fields.npz"),
        ("settingsSha256", "world-settings.json"),
    ):
        if sha256(output / relative) != record[key]:
            raise ValueError("accepted input changed after the physical checkpoint")
    forbidden = json.loads((output / "naming-exclusions.json").read_text(encoding="utf-8"))["forbidden"]
    grid = WorldGrid.load(output / "grid")
    request = grid.metadata["societyGeneration"]
    if (
        request["humanSeed"] != record["humanSeed"]
        or request["namingSeed"] != record["namingSeed"]
        or sorted(request["forbiddenNames"]) != sorted(forbidden)
    ):
        raise ValueError("human generation contract changed after the physical checkpoint")
    name_source, generation_request = _society_generation_request(grid)
    raw_source = load_surface_bundle(output / "source/physical-fields.npz")
    with measure_stage(output, "society"):
        society = derive_society_layers(grid, derive_thematic_layers(grid,
                                        ecological_sources=derive_ecological_sources(grid, raw_source)), name_source,
                                        raw_elevation_m=raw_source.relative_elevation_m,
                                        **generation_request)
        save_society(society, output / "society", grid_digest=grid.content_digest())
    logging.info("Society checkpoint saved; rendering all map views")
    return finish_accepted_world(output, started=started)


def finish_accepted_world(output: Path, *, started: float | None = None) -> dict:
    """Render the verified society checkpoint, without rerunning simulation."""
    started = time.monotonic() if started is None else started
    output = output.resolve()
    record = json.loads((output / "regeneration.json").read_text(encoding="utf-8"))
    if record["engineVersion"] != __version__:
        raise ValueError("checkpoint engine version differs from the running engine")
    if record["status"] != "building":
        raise ValueError("only an unfinished build can be published")
    grid = WorldGrid.load(output / "grid")
    bundle_path = output / "source/physical-fields.npz"
    if sha256(bundle_path) != record["fieldSha256"]:
        raise ValueError("accepted physical source changed before rendering")
    physical_source = load_surface_bundle(bundle_path)
    society = load_society(output / "society", expected_grid_digest=grid.content_digest())
    forbidden = grid.metadata["societyGeneration"]["forbiddenNames"]
    with measure_stage(output, "render"):
        settings = load_world_settings(output/'world-settings.json')
        render_review(grid, output / "review", society=society, physical_source=physical_source,
                      travel_capabilities=settings.travel_capabilities)
    society = load_society(output / "review", expected_grid_digest=grid.content_digest())
    audit = naming_audit(society, forbidden)
    write_json(output / "review/naming-audit.json", audit)
    record = {**record, "schema": "accepted-world-v2", "status": "complete", "worldName": grid.metadata["worldProfile"]["name"],
        "physicalCheckpoint": record.get("physicalCheckpoint"),
        "engineVersion": __version__,
        "humanSeed": record["humanSeed"], "namingSeed": record["namingSeed"],
        "terrainSeed": record["terrainSeed"], "gridDigest": grid.content_digest(),
        "sourceSha256": sha256(output / "source/physical-reference.png"),
        "fieldSha256": sha256(output / "source/physical-fields.npz"),
        "settingsSha256": sha256(output / "world-settings.json"),
        "nameDigest": audit["nameDigest"], "excludedNameCount": len(forbidden),
        "elapsedSeconds": elapsed_seconds(output)}
    write_json(output / "regeneration.json", record)
    write_json(output / "review/world.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    parser.add_argument("--legacy-artifacts", type=Path)
    args = parser.parse_args()
    exclusions = list(args.exclude)
    if args.legacy_artifacts:
        exclusions.extend(sorted(args.legacy_artifacts.glob("worldgrid*/review/society.json")))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from world_atlas.settings import load_world_settings
    build_accepted_world(
        args.config,
        args.provenance,
        args.output,
        exclusions,
        settings=load_world_settings(args.settings),
    )


if __name__ == "__main__":
    main()
