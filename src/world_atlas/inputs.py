"""Generate and independently deploy the seeded fictional world."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from world_atlas.core.baseline import load_baseline
from world_atlas.core.model import WorldGrid
from world_atlas.core.procedural_planet import (
    PlanetRecipe,
    generate_planet_surface,
    save_palette_source,
    save_surface_bundle,
)
from world_atlas.core.render import render_review
from world_atlas.core.society.world_identity import NameRegistry
from world_atlas.settings import WorldSettings


SEED = 2116268501
LAND_PALETTE = (
    (0x69, 0xBD, 0xA9),
    (0x8F, 0xD2, 0xA4),
    (0xB6, 0xE2, 0xA1),
    (0xD7, 0xEF, 0x9F),
    (0xEE, 0xF8, 0xA8),
    (0xFB, 0xF8, 0xB0),
    (0xFE, 0xEB, 0x9F),
    (0xFE, 0xD4, 0x83),
    (0xFD, 0xB7, 0x6A),
    (0xF9, 0x94, 0x56),
    (0xF0, 0x70, 0x4A),
    (0xE0, 0x50, 0x4A),
)
BATHYMETRY_PALETTE = (
    (0x9F, 0xB8, 0xE7),
    (0x99, 0xB3, 0xE4),
    (0x91, 0xAD, 0xE1),
    (0x99, 0xAF, 0xD1),
    (0x8C, 0xA5, 0xCB),
    (0x7C, 0x98, 0xC3),
    (0x6B, 0x8B, 0xBC),
    (0x67, 0x87, 0xB8),
)


@dataclass(frozen=True, slots=True)
class PreparedWorldInputs:
    source_path: Path
    bundle_path: Path
    provenance_path: Path
    config_path: Path
    diagnostics: Mapping[str, object]


def default_recipe(*, width: int = 2176, height: int = 1088) -> PlanetRecipe:
    return PlanetRecipe(
        seed=SEED,
        width=width,
        height=height,
        plate_count=14,
        continent_count=8,
        land_fraction=0.35,
        tectonic_activity=0.34,
        mountain_density=0.82,
        coastline_detail=0.28,
        north_polar_continent=False,
        south_polar_continent=False,
    )


def _hex_palette(palette: tuple[tuple[int, int, int], ...]) -> list[str]:
    return ["#" + "".join(f"{channel:02x}" for channel in colour) for colour in palette]


def _relative_path(path: Path, start: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), start=start.resolve())).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _config_document(
    recipe: PlanetRecipe,
    *,
    source_path: Path,
    source_sha256: str,
    bundle_path: Path,
    bundle_sha256: str,
    config_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    return {
        "format": "eirenor-worldgen",
        "schemaVersion": 2,
        "source": {
            "path": _relative_path(source_path, config_path.parent),
            "sha256": source_sha256,
            "registration": "procedural-equirectangular",
            "fieldBundle": {
                "path": _relative_path(bundle_path, config_path.parent),
                "sha256": bundle_sha256,
            },
        },
        "planet": {
            "radiusKm": 6400.0,
            "gravityMps2": 9.80665,
            "rotationPeriodHours": 26.0,
            "rotationDirection": "retrograde",
            "axialTiltDegrees": 18.0,
            "orbitalPeriodDays": 320.0,
            "orbitalEccentricity": 0.025,
            "longitudeOfPeriapsisDegrees": 0.0,
            "solarConstantWm2": 1620.0,
            "surfacePressurePa": 101325.0,
            "oceanHeatCapacityFactor": 3.4,
        },
        "grid": {
            "width": recipe.width,
            "height": recipe.height,
            "boardWidthPx": recipe.width,
            "boardHeightPx": recipe.height,
            "paddingPx": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "samplingStepPx": 1,
            "longitudeExtent": [-180.0, 180.0],
            "latitudeExtent": [-90.0, 90.0],
        },
        "palettes": {
            "land": _hex_palette(LAND_PALETTE),
            "bathymetry": _hex_palette(BATHYMETRY_PALETTE),
            "levels": {"elevation": 16, "bathymetry": 8},
        },
        "elevationRegularization": {"sigmaGridUnits": 1.5, "passes": 2},
        "hydrology": {
            "streamBurnDepth": 0.035,
            "streamThresholdFraction": 0.000055,
            "tributaryThresholdFraction": 0.0012,
            "mainstemThresholdFraction": 0.012,
            "minimumHeadwaterLength": 6,
            "tieEpsilon": 0.000000000001,
            "minimumStreamSpanFactor": 0.05,
            "closureBudgetFraction": 0.01,
            "closureMaxChainFraction": 0.012,
            "mfdExponent": 1.1,
            "valleyWindowFraction": 0.02,
            "valleyDepthFraction": 0.01,
            "valleySupportThreshold": 0.09,
            "valleySupportDistanceFactor": 1.0,
            "parallelSearchRadiusFactor": 0.5,
            "parallelPriorityWeight": 0.35,
            "runoffWeightFloor": 0.28,
            "snowmeltRunoffBonus": 0.7,
            "headwaterUplandQuantile": 0.68,
            "lowlandHeadwaterFlowMultiplier": 1.4,
            "majorRiversPerContinent": 2,
            "majorRiverTributaries": 3,
            "continentMinimumLandFraction": 0.01,
            "majorRiverMinimumSpanFactor": 0.35,
            "lakeOutlets": [],
            "inlandSinkCells": [],
        },
        "snowline": {
            "equatorThreshold": 0.93,
            "poleThreshold": 0.025,
            "exponent": 1.1,
            "maxOffset": 0.12,
            "polarFullSnowLatitude": 80.0,
        },
        "output": {
            "directory": _relative_path(output_directory, config_path.parent)
        },
        "budget": {"maxCells": recipe.width * recipe.height},
    }


def prepare_world_inputs(
    recipe: PlanetRecipe,
    *,
    source_directory: str | Path,
    config_path: str | Path,
    output_directory: str | Path,
) -> PreparedWorldInputs:
    source_directory = Path(source_directory).resolve()
    config_path = Path(config_path).resolve()
    output_directory = Path(output_directory).resolve()
    source_directory.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    surface = generate_planet_surface(recipe)
    source_path = source_directory / "physical-reference.procedural.png"
    save_palette_source(
        surface,
        source_path,
        land_palette=LAND_PALETTE,
        bathymetry_palette=BATHYMETRY_PALETTE,
    )
    bundle_path = source_directory / "physical-fields.npz"
    save_surface_bundle(surface, bundle_path)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest().upper()
    bundle_sha256 = hashlib.sha256(bundle_path.read_bytes()).hexdigest().upper()
    diagnostics = dict(surface.diagnostics)
    provenance = {
        "format": "fictional-world-physical-source",
        "schema": "procedural-planet-v15",
        "engineVersion": version('world-atlas-engine'),
        **diagnostics,
        "recipe": asdict(recipe),
        "sourceFile": source_path.name,
        "sourceSha256": source_sha256,
        "fieldBundle": {
            "file": bundle_path.name,
            "sha256": bundle_sha256,
            "schema": "procedural-surface-bundle-v2",
        },
    }
    provenance_path = source_directory / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            _config_document(
                recipe,
                source_path=source_path,
                source_sha256=source_sha256,
                bundle_path=bundle_path,
                bundle_sha256=bundle_sha256,
                config_path=config_path,
                output_directory=output_directory,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return PreparedWorldInputs(
        source_path=source_path,
        bundle_path=bundle_path,
        provenance_path=provenance_path,
        config_path=config_path,
        diagnostics=diagnostics,
    )


def attach_world_metadata(
    grid: WorldGrid,
    recipe: PlanetRecipe,
    diagnostics: Mapping[str, object],
    *,
    bundle_path: str | Path,
    bundle_sha256: str,
    settings: WorldSettings,
    forbidden_names: tuple[str, ...] = (),
) -> WorldGrid:
    naming_seed = settings.naming_seed
    world_name = NameRegistry(naming_seed, forbidden_names).name("world", 9)
    metadata = dict(grid.metadata)
    coordinate_reference_system = dict(metadata["coordinateReferenceSystem"])
    coordinate_reference_system["referenceBody"] = world_name
    metadata["coordinateReferenceSystem"] = coordinate_reference_system
    metadata["worldProfile"] = {
        "name": world_name,
        "seed": naming_seed,
        "starClass": "G2黄色恒星",
        "satellites": "多颗小卫星",
        "tides": "偏弱",
        "plateRegime": f"{recipe.plate_count}块、连续运动学板块系统",
        "landFraction": float(diagnostics["landFraction"]),
        "continentLayout": (
            f"{recipe.continent_count}个大陆地壳核心；"
            f"{diagnostics['worldFamily']}"
        ),
        "climateBias": "偏干，草原、荒漠与内流盆地偏多",
        "technology": "preindustrial",
        "populationPattern": "稀疏定居与广阔边疆",
        "transportSemantics": ["官道", "驿道", "商路", "可通航河道", "海路"],
    }
    metadata["proceduralPhysicalSource"] = {
        "method": diagnostics["method"],
        "plateCount": recipe.plate_count,
        "continentCount": recipe.continent_count,
        "landFraction": float(diagnostics["landFraction"]),
        "continentAreas": dict(diagnostics["continentAreas"]),
        "worldFamily": diagnostics["worldFamily"],
        "morphologyAxes": dict(diagnostics["morphologyAxes"]),
        "classificationMetrics": dict(diagnostics["classificationMetrics"]),
        "terrainSystemMetrics": dict(diagnostics["terrainSystemMetrics"]),
        "mapSeam": dict(diagnostics["mapSeam"]),
        "fieldBundle": {
            "path": str(Path(bundle_path).resolve()),
            "sha256": bundle_sha256,
            "schema": "procedural-surface-bundle-v2",
        },
    }
    metadata["societyGeneration"] = {
        "namingProfile": "procedural",
        "humanSeed": settings.human_seed,
        "namingSeed": settings.naming_seed,
        **dict(settings.society),
    }
    if forbidden_names:
        metadata["societyGeneration"]["forbiddenNames"] = tuple(sorted(set(forbidden_names)))
    return replace(grid, metadata=metadata)

