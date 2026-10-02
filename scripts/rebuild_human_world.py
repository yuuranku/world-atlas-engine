"""Rebuild all human geography into a fresh world, preserving its physical grid.

The raw physical bundle and its verification proof are required inputs. This
entry point never regenerates physical geography and never renders a review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import shutil
import time

import numpy as np

from world_atlas import __version__
from world_atlas.acceptance import _settlement_surface_violations, _transport_surface_violations
from world_atlas.checks import array_digest
from world_atlas.core.model import WorldGrid
from world_atlas.core.procedural_planet import load_surface_bundle
from world_atlas.core.render import _society_generation_request
from world_atlas.core.society.administrative_audit import audit_administrative_topology
from world_atlas.core.society.pipeline import derive_society_layers
from world_atlas.core.society.population import population_density
from world_atlas.core.society.storage import load_society, save_society
from world_atlas.core.society.world_identity import assign_world_identity, naming_audit
from world_atlas.core.thematic import derive_thematic_layers
from world_atlas.core.ecological_sources import derive_ecological_sources
from world_atlas.core.vegetation import derive_vegetation_cover
from world_atlas.rebuild import sha256, write_json
from world_atlas.settings import load_world_settings
from world_atlas.timing import measure_stage


_REPLAY_FIELDS = {
    "plate_id", "plate_velocity_east_cm_per_year", "plate_velocity_north_cm_per_year",
    "boundary_class", "crust_kind", "ocean_age_myr", "continent_id", "land_mask",
    "elevation", "bathymetry", "signed_height",
}


def validate_inputs(world: Path, physical_source: Path, physical_proof: Path, output: Path) -> dict:
    """Read and validate every input before creating the new output directory."""
    if output.exists():
        raise FileExistsError(f"human rebuild requires a fresh output: {output}")
    if output.is_relative_to(world):
        raise ValueError("human rebuild output must be outside the saved world")
    grid = WorldGrid.load(world / "grid")
    settings = load_world_settings(world / "world-settings.json")
    config = json.loads((world / "worldgen.json").read_text(encoding="utf-8"))
    provenance = json.loads((world / "source/provenance.json").read_text(encoding="utf-8"))
    proof = json.loads(physical_proof.read_text(encoding="utf-8"))
    source = load_surface_bundle(physical_source)
    bundle_sha = sha256(physical_source)
    old_bundle = (world / config["source"]["fieldBundle"]["path"]).resolve()
    reference = (world / config["source"]["path"]).resolve()
    if not old_bundle.is_relative_to(world) or not reference.is_relative_to(world):
        raise ValueError("saved world physical inputs must be self-contained")
    old_sha = sha256(old_bundle)
    if old_sha != config["source"]["fieldBundle"]["sha256"].upper():
        raise ValueError("saved physical bundle differs from worldgen.json")
    if sha256(reference) != config["source"]["sha256"].upper():
        raise ValueError("saved physical reference differs from worldgen.json")
    if (proof.get("schema") != "verified-physical-surface-replay-v1"
            or proof.get("verifiedExact") is not True
            or proof.get("originalSourceUnchanged") is not True
            or proof.get("nativePaletteReconstructionExact") is not True
            or proof.get("verifiedBundleSha256", "").upper() != bundle_sha
            or proof.get("sourceBundleSha256", "").upper()
            != grid.metadata["proceduralPhysicalSource"]["fieldBundle"]["sha256"].upper()
            or proof.get("recipe") != provenance["recipe"]
            or set(proof.get("comparisons", {})) != _REPLAY_FIELDS
            or any(value.get("exact") is not True or value.get("different") != 0
                   for value in proof.get("comparisons", {}).values())):
        raise ValueError("physical replay proof does not verify this saved world and raw bundle")
    land = grid.water == 0
    ground = np.asarray(source.relative_elevation_m)
    if (ground.shape != grid.shape or np.any(~np.isfinite(ground))
            or not np.array_equal(ground > 0, land)
            or not np.array_equal(source.land_mask, land)
            or source.diagnostics.get("seed") != provenance["recipe"]["seed"]):
        raise ValueError("verified raw ground does not align with the saved physical grid")
    request = grid.metadata["societyGeneration"]
    if (request["humanSeed"] != settings.human_seed or request["namingSeed"] != settings.naming_seed
            or request["technologyEra"] != settings.technology_era
            or request["namingProfile"] != "procedural"
            or any(request.get(key) != value for key, value in settings.society.items())
            or any(grid.metadata["planet"].get(key) != value for key, value in settings.planet.items())):
        raise ValueError("saved grid generation profile differs from the preserved world settings")
    if settings.society["stateCount"] != 80:
        raise ValueError("this human rebuild requires the existing 80-country world profile")
    exclusions = json.loads((world / "naming-exclusions.json").read_text(encoding="utf-8"))
    if sorted(exclusions["forbidden"]) != sorted(request["forbiddenNames"]):
        raise ValueError("saved naming exclusions differ from the grid generation profile")
    original = load_society(world / "society", expected_grid_digest=grid.content_digest())
    names, society_request = _society_generation_request(grid)
    return dict(grid=grid, settings=settings, config=config, provenance=provenance,
                proof=proof, bundle_sha=bundle_sha, old_bundle_sha=old_sha, reference=reference,
                exclusions=exclusions, original=original, names=names, society_request=society_request,
                raw_elevation_m=ground)


def human_checks(grid, thematic, society, prepared: dict) -> tuple[dict, dict, dict]:
    """Check generated data, rather than accepting requested counts as evidence."""
    land = grid.water == 0
    weights = society.population.population_weight
    forbidden = prepared["exclusions"]["forbidden"]
    audit = naming_audit(society, forbidden)
    replay = assign_world_identity(society, seed=prepared["settings"].naming_seed, forbidden=forbidden)
    failures = []
    if naming_audit(replay, forbidden) != audit or audit["oldNameMatches"] or audit["duplicateNames"]:
        failures.append("canonical naming replay, exclusions or uniqueness failed")
    requested = prepared["settings"].society
    counts = dict(countries=len(society.politics.states), provinces=len(society.provinces.provinces),
                  settlements=len(society.settlements), civilizations=len(society.cultures.civilizations),
                  languages=len(society.cultures.languages), religions=len(society.religions.religions),
                  routes=len(society.transport.routes), bridges=len(society.transport.bridges))
    if counts["countries"] != 80 or not 400 <= counts["provinces"] <= 500:
        failures.append("actual administration must contain 80 countries and 400–500 provinces")
    for actual, requested_key in (("civilizations", "civilizationCount"), ("languages", "languageCount"), ("religions", "religionCount")):
        if counts[actual] != requested[requested_key]:
            failures.append(f"actual {actual} differs from the preserved generation profile")
    normalized_total = float(np.sum(weights, dtype=np.float64))
    if abs(normalized_total - 1.0) > 1e-6 or np.any(weights[~land] != 0):
        failures.append("population must retain normalized headcounts only on land")
    if (society.population.population_min != requested["populationMin"]
            or society.population.population_max != requested["populationMax"]):
        failures.append("population total range differs from the preserved world settings")
    for name in ("land_potential", "habitability"):
        if np.any(getattr(thematic, name)[~land] != 0):
            failures.append(f"{name} has support on water")
    settlements = _settlement_surface_violations(grid, society.settlements)
    routes = _transport_surface_violations(grid, society.transport.routes)
    if any(settlements.values()) or any(routes.values()):
        failures.append("settlements or transport violate canonical physical surfaces")
    state_id, province_id = society.politics.state_id, society.provinces.province_id
    parent = np.zeros(int(province_id.max(initial=0)) + 1, dtype=np.int32)
    for province in society.provinces.provinces:
        parent[province.identifier] = province.state_identifier
    ownership = {
        "unpartitionedCountryCells": int(np.count_nonzero((state_id > 0) & (province_id <= 0))),
        "provinceParentMismatchCells": int(np.count_nonzero((province_id > 0) & (parent[np.maximum(province_id, 0)] != state_id))),
        "countryCellsOnWater": int(np.count_nonzero((state_id > 0) & ~land)),
        "provinceCellsOnWater": int(np.count_nonzero((province_id > 0) & ~land)),
    }
    if any(ownership.values()):
        failures.append("administrative coverage or province parents differ from canonical ownership")
    topology = audit_administrative_topology(grid, society)
    if topology["status"] != "ok":
        failures.append("administrative topology requires review")
    field_bands = lambda values: {str(int(value)): int(count) for value, count in zip(*np.unique(values[land], return_counts=True), strict=True)}
    old_audit = naming_audit(prepared["original"], forbidden)
    old_profiles = {item["civilizationId"]: item["key"] for item in old_audit["profiles"]}
    new_profiles = {item["civilizationId"]: item["key"] for item in audit["profiles"]}
    report = {
        "schema": "world-atlas-human-rebuild-check-v1", "status": "failed" if failures else "ok",
        "failures": failures, "counts": counts, "countryTarget": 80, "provinceRange": [400, 500],
        "humanSeed": prepared["settings"].human_seed, "namingSeed": prepared["settings"].naming_seed,
        "namingProfile": grid.metadata["societyGeneration"]["namingProfile"],
        "selectedNamingProfilesUnchanged": old_profiles == new_profiles,
        "namingProfileChanges": [{"civilizationId": identifier, "previous": old_profiles.get(identifier),
                                  "current": new_profiles.get(identifier)}
                                 for identifier in sorted(old_profiles.keys() | new_profiles.keys())
                                 if old_profiles.get(identifier) != new_profiles.get(identifier)],
        "previousNamingProfiles": old_audit["profiles"], "namingProfiles": audit["profiles"],
        "populationRange": [society.population.population_min, society.population.population_max],
        "populationMidpoint": (society.population.population_min + society.population.population_max) / 2,
        "populationWeightSum": normalized_total,
        "populationWeightChangedCells": int(np.count_nonzero(weights != prepared["original"].population.population_weight)),
        "populationBandChangedCells": int(np.count_nonzero(society.population.population_band != prepared["original"].population.population_band)),
        "frontierPopulationShare": float(np.sum(weights[land & (state_id == 0)], dtype=np.float64)),
        "populationBandsLandCells": field_bands(society.population.population_band),
        "agriculturalBandsLandCells": field_bands(thematic.land_potential_band),
        "habitabilityBandsLandCells": field_bands(thematic.habitability_band),
        "settlementTiers": dict(Counter(item.tier for item in society.settlements)),
        "settlementSiteTypes": dict(Counter(item.site_type for item in society.settlements)),
        "ownership": ownership, "settlementSurfaceViolations": settlements, "routeSurfaceViolations": routes,
    }
    return report, audit, topology


def rebuild_human_world(world: Path, physical_source: Path, physical_proof: Path, output: Path) -> dict:
    world, physical_source, physical_proof, output = (path.resolve() for path in (world, physical_source, physical_proof, output))
    started = time.perf_counter()
    prepared = validate_inputs(world, physical_source, physical_proof, output)
    grid = prepared["grid"]
    before_digest = grid.content_digest()
    before_arrays = array_digest(world / "grid/world-grid.npz")
    before_grid_sha = {name: sha256(world / "grid" / name) for name in ("world-grid.npz", "world-grid.json")}
    output.mkdir(parents=True)
    (output / "source").mkdir()
    shutil.copytree(world / "grid", output / "grid")
    shutil.copy2(prepared["reference"], output / "source/physical-reference.png")
    shutil.copy2(physical_source, output / "source/physical-fields.npz")
    shutil.copy2(physical_proof, output / "source/physical-fields-verification.json")
    shutil.copy2(world / "source/provenance.json", output / "source/provenance.json")
    for name in ("world-settings.json", "naming-exclusions.json"):
        shutil.copy2(world / name, output / name)
    config = prepared["config"]
    config["source"]["path"] = "source/physical-reference.png"
    config["source"]["fieldBundle"] = {"path": "source/physical-fields.npz", "sha256": prepared["bundle_sha"]}
    config["output"]["directory"] = "."
    write_json(output / "worldgen.json", config)
    record = {
        "schema": "accepted-world-v2", "status": "building", "rebuildKind": "human-only-v1",
        "engineVersion": __version__, "sourceWorld": str(world),
        "humanSeed": prepared["settings"].human_seed, "namingSeed": prepared["settings"].naming_seed,
        "terrainSeed": prepared["provenance"]["recipe"]["seed"], "terrainSchema": prepared["provenance"]["schema"],
        "gridDigest": before_digest, "sourceSha256": sha256(output / "source/physical-reference.png"),
        "fieldSha256": prepared["bundle_sha"], "settingsSha256": sha256(output / "world-settings.json"),
        "physicalCheckpoint": {"state": "preserved", "source": "grid"},
        "physicalVerification": "source/physical-fields-verification.json",
    }
    write_json(output / "regeneration.json", record)
    try:
        logging.info("Physical grid and verified raw source preserved; deriving all thematic and human layers")
        with measure_stage(output, "human-cascade"):
            ecological_sources=derive_ecological_sources(
                grid,load_surface_bundle(output/'source/physical-fields.npz'))
            thematic = derive_thematic_layers(grid,ecological_sources=ecological_sources)
            society = derive_society_layers(grid, thematic, prepared["names"],
                                            raw_elevation_m=prepared["raw_elevation_m"],
                                            **prepared["society_request"])
            save_society(society, output / "society", grid_digest=before_digest)
        (output / "thematic").mkdir()
        np.savez_compressed(output / "thematic/wetland-support.npz",
                            support=thematic.physiography.wetland_support,
                            mask=thematic.physiography.wetland)
        np.savez_compressed(output / "thematic/human-capacity.npz",
                            land_potential=thematic.land_potential, land_potential_band=thematic.land_potential_band,
                            habitability=thematic.habitability, habitability_band=thematic.habitability_band,
                            vegetation_cover=derive_vegetation_cover(grid,thematic.climate,
                                ecological_sources=ecological_sources).fraction,
                            population_density=population_density(grid, society.population))
        checks, names, topology = human_checks(grid, thematic, society, prepared)
        unchanged = (WorldGrid.load(output / "grid").content_digest() == before_digest
                     and grid.content_digest() == before_digest
                     and array_digest(output / "grid/world-grid.npz") == before_arrays
                     and all(sha256(world / "grid" / name) == before_grid_sha[name]
                             and sha256(output / "grid" / name) == before_grid_sha[name] for name in before_grid_sha)
                     and sha256(physical_source) == prepared["bundle_sha"]
                     and sha256(output / "source/physical-fields.npz") == prepared["bundle_sha"])
        checks.update(physicalGridSemanticUnchanged=unchanged, gridDigest=before_digest,
                      gridArrayDigest=before_arrays, physicalFieldSha256=prepared["bundle_sha"],
                      oldPhysicalFieldSha256=prepared["old_bundle_sha"], rawLandSignMismatches=0,
                      metadataPhysicalOrigin=dict(grid.metadata["proceduralPhysicalSource"]["fieldBundle"]),
                      elapsedSeconds=round(time.perf_counter() - started, 3))
        if not unchanged:
            checks["status"] = "failed"
            checks["failures"].append("physical grid or explicit source changed during the human rebuild")
        write_json(output / "society/human-rebuild-checks.json", checks)
        write_json(output / "society/naming-audit.json", names)
        write_json(output / "society/administrative-topology.json", topology)
        record.update(humanStatus=checks["status"], nameDigest=names["nameDigest"],
                      excludedNameCount=len(prepared["exclusions"]["forbidden"]))
        write_json(output / "regeneration.json", record)
        if checks["status"] != "ok":
            raise ValueError("human cascade checks failed: " + "; ".join(checks["failures"]))
        logging.info("Human cascade ready: %d countries, %d provinces; review rendering remains separate", checks["counts"]["countries"], checks["counts"]["provinces"])
        return checks
    except BaseException:
        record["status"] = "failed"
        record["humanStatus"] = "failed"
        write_json(output / "regeneration.json", record)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path, help="Existing saved world; read-only")
    parser.add_argument("--physical-source", type=Path, required=True, help="Verified v3 raw-ground bundle")
    parser.add_argument("--physical-proof", type=Path, required=True, help="Exact replay proof for the supplied bundle")
    parser.add_argument("--output", type=Path, required=True, help="Fresh human-world destination")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    checks = rebuild_human_world(args.world, args.physical_source, args.physical_proof, args.output)
    print(json.dumps(checks, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
