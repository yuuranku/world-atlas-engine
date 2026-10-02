"""Independently check saved administrative population after a border repair.

This consumer reads both complete world bundles and writes only its report. It
does not call the administrative repair or its raster aggregation helpers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from world_atlas.core.model import WorldGrid
from world_atlas.core.presentation import society_content_digest
from world_atlas.core.society.storage import load_society


def check_population_records(grid, society) -> dict:
    """Recompute original country rounding and spherical province density."""
    population = society.population
    weights = population.population_weight
    states = society.politics.state_id
    provinces = society.provinces.province_id
    if weights.shape != grid.shape or states.shape != grid.shape or provinces.shape != grid.shape:
        raise ValueError("Saved administrative population must match the grid")

    state_checks = []
    for record in society.politics.states:
        # Use the original per-country float64 sum, independently of the
        # repair's grouped bincount and without renormalising controlled land.
        share = float(weights[states == record.identifier].sum(dtype=np.float64))
        lower = max(1_000, int(round(share * population.population_min / 10_000)) * 10_000)
        upper = max(lower + 1_000, int(round(share * population.population_max / 10_000)) * 10_000)
        saved = [record.population_min, record.population_max]
        state_checks.append({
            "id": record.identifier, "name": record.name, "populationWeight": share,
            "expectedRange": [lower, upper], "savedRange": saved,
            "matches": saved == [lower, upper],
        })

    # Integrate each spherical latitude strip directly. This deliberately
    # avoids the production cell_areas_km2 and population_density helpers.
    extents = grid.metadata["extents"]
    radius = float(grid.metadata["planet"]["radiusKm"])
    north, south = float(extents["north"]), float(extents["south"])
    east, west = float(extents["east"]), float(extents["west"])
    latitude = np.linspace(north, south, grid.shape[0] + 1)
    longitude_width = math.radians(east - west) / grid.shape[1]
    strip_areas = radius ** 2 * longitude_width * np.abs(np.diff(np.sin(np.radians(latitude))))
    if np.any(~np.isfinite(strip_areas)) or np.any(strip_areas <= 0):
        raise ValueError("The saved sphere must have positive cell areas")
    areas = np.broadcast_to(strip_areas[:, None], grid.shape)
    controlled = states > 0
    midpoint = (population.population_min + population.population_max) / 2.0
    controlled_area = float(areas[controlled].sum(dtype=np.float64))
    if controlled_area <= 0:
        raise ValueError("Administrative summaries require controlled territory")
    controlled_density = float(weights[controlled].sum(dtype=np.float64) * midpoint / controlled_area)
    province_checks = []
    for record in society.provinces.provinces:
        cells = provinces == record.identifier
        count = int(np.count_nonzero(cells))
        area = float(areas[cells].sum(dtype=np.float64))
        if area <= 0:
            raise ValueError("Every saved province must have positive area")
        density = float(weights[cells].sum(dtype=np.float64) * midpoint / area)
        ratio = density / max(controlled_density, 1.0e-12)
        category = "dense" if ratio >= 1.35 else "settled" if ratio >= .62 else "sparse"
        province_checks.append({
            "id": record.identifier, "name": record.name, "areaKm2": area,
            "expectedAreaCells": count, "savedAreaCells": record.area_cells,
            "densityPersonsPerKm2": density, "relativeDensity": ratio,
            "expectedDensityClass": category, "savedDensityClass": record.population_density_class,
            "matches": count == record.area_cells and category == record.population_density_class,
        })
    return {
        "statesChecked": len(state_checks), "stateChecks": state_checks,
        "statePopulationMismatches": [row for row in state_checks if not row["matches"]],
        "provincesChecked": len(province_checks), "provinceChecks": province_checks,
        "provinceAreaOrDensityMismatches": [row for row in province_checks if not row["matches"]],
        "sphericalArea": {
            "radiusKm": radius, "formula": "R^2 * deltaLongitudeRadians * abs(sin(north)-sin(south))",
            "controlledAreaKm2": controlled_area, "controlledDensityPersonsPerKm2": controlled_density,
        },
    }


def _file_hashes(world: Path) -> dict:
    return {name: sha256((world / name).read_bytes()).hexdigest() for name in (
        "grid/world-grid.json", "grid/world-grid.npz", "society/society.json", "society/society.npz",
    )}


def check_worlds(source: Path, target: Path) -> dict:
    source, target = source.resolve(), target.resolve()
    initial_hashes = [_file_hashes(path) for path in (source, target)]
    source_grid, target_grid = [WorldGrid.load(path / "grid") for path in (source, target)]
    source_digest, target_digest = source_grid.content_digest(), target_grid.content_digest()
    before, after = [load_society(path / "society", expected_grid_digest=digest)
                     for path, digest in ((source, source_digest), (target, target_digest))]
    if source_grid.shape != target_grid.shape:
        raise ValueError("Border refinement must retain the physical grid shape")
    result = check_population_records(target_grid, after)
    prior_states = {record.identifier: record for record in before.politics.states}
    interval_changes = []
    for record in after.politics.states:
        prior = prior_states.get(record.identifier)
        if prior is not None and (prior.population_min, prior.population_max) != (record.population_min, record.population_max):
            interval_changes.append({
                "id": record.identifier, "name": record.name,
                "sourceRange": [prior.population_min, prior.population_max],
                "targetRange": [record.population_min, record.population_max],
                "delta": [record.population_min - prior.population_min, record.population_max - prior.population_max],
            })
    def identity(records):
        return [(record.identifier, record.name, record.core_settlement_id) for record in records]
    unchanged = {
        "gridDigest": source_digest == target_digest,
        "populationWeight": np.array_equal(before.population.population_weight, after.population.population_weight),
        "populationBand": np.array_equal(before.population.population_band, after.population.population_band),
        "populationRange": (before.population.population_min, before.population.population_max) ==
                           (after.population.population_min, after.population.population_max),
        "settlementRecords": before.settlements == after.settlements,
        "stateIdentityAndCapitals": identity(before.politics.states) == identity(after.politics.states),
        "provinceIdentityAndCapitals": identity(before.provinces.provinces) == identity(after.provinces.provinces),
        "provinceParentCountries": [(p.identifier, p.state_identifier) for p in before.provinces.provinces] ==
                                    [(p.identifier, p.state_identifier) for p in after.provinces.provinces],
    }
    city_owner_changes = []
    for city in after.settlements:
        row, column = city.row, city.column
        old = [int(before.politics.state_id[row, column]), int(before.provinces.province_id[row, column])]
        new = [int(after.politics.state_id[row, column]), int(after.provinces.province_id[row, column])]
        if old != new:
            city_owner_changes.append({"id": city.identifier, "sourceOwners": old, "targetOwners": new})
    final_hashes = [_file_hashes(path) for path in (source, target)]
    stable = initial_hashes == final_hashes
    passed = (all(unchanged.values()) and stable and not city_owner_changes
              and not result["statePopulationMismatches"] and not result["provinceAreaOrDensityMismatches"])
    return {
        "schema": "world-atlas-administrative-population-check-v1", "passed": passed,
        "checkedAtUtc": datetime.now(timezone.utc).isoformat(),
        "source": {"directory": str(source), "gridDigest": source_digest,
                   "societyDigest": society_content_digest(before), "fileSha256": initial_hashes[0]},
        "target": {"directory": str(target), "gridDigest": target_digest,
                   "societyDigest": society_content_digest(after), "fileSha256": initial_hashes[1]},
        "inputFilesStableDuringCheck": stable, "unchanged": unchanged,
        "settlementsChecked": len(after.settlements), "cityOwnershipMismatches": city_owner_changes,
        "nativeStateChanges": int(np.count_nonzero(before.politics.state_id != after.politics.state_id)),
        "nativeProvinceChanges": int(np.count_nonzero(before.provinces.province_id != after.provinces.province_id)),
        "populationRange": [after.population.population_min, after.population.population_max],
        "populationWeightSum": float(after.population.population_weight.sum(dtype=np.float64)),
        "countriesWithUpdatedPopulationIntervals": len(interval_changes), "populationIntervalChanges": interval_changes,
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Complete world before administrative refinement")
    parser.add_argument("--target", required=True, type=Path, help="Complete world after administrative refinement")
    parser.add_argument("--report", required=True, type=Path, help="Report outside both read-only world directories")
    args = parser.parse_args()
    output = args.report.resolve()
    if any(output.is_relative_to(world.resolve()) for world in (args.source, args.target)):
        parser.error("--report must be outside both read-only world directories")
    report = check_worlds(args.source, args.target)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"], "report": str(output),
        "gridDigest": report["target"]["gridDigest"], "societyDigest": report["target"]["societyDigest"],
        "statesChecked": report["statesChecked"], "provincesChecked": report["provincesChecked"],
        "statePopulationMismatchCount": len(report["statePopulationMismatches"]),
        "provinceMismatchCount": len(report["provinceAreaOrDensityMismatches"]),
        "countriesWithUpdatedPopulationIntervals": report["countriesWithUpdatedPopulationIntervals"],
    }, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
