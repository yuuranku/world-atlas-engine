"""Import the authored PNG into the canonical WorldGrid cell arrays."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import WorldConfig, load_world_config, physical_payload
from .model import WorldGrid
from .polar import apply_polar_sea_ice
from .procedural_planet import load_surface_bundle


BASELINE_IMPORTER_VERSION = "worldgen-baseline-importer-v6"
_SEASON_IDS = ("vernal", "june", "autumnal", "december")
_SEASON_SOLAR_LONGITUDES = (0.0, 90.0, 180.0, 270.0)
_RELATIVE_PRECIPITATION_MAXIMUM = 1.35

# Keep this list explicit and relative to the project root.  These are the local
# Python sources that can change the canonical cell truth; render.py is
# intentionally absent because it only consumes an already-built WorldGrid.
CELL_TRUTH_SOURCE_FILES = (
    "tectonic_foundation.py",
    "physical/build_source.py",
    "physical/hydrology.py",
    "physical/island_generation.py",
    "physical/procedural_landmass.py",
    "physical/physiographic_provinces.py",
    "physical/terrain_refinement.py",
    "physical/climate.py",
    "physical/planetary_grid.py",
    "physical/polar_terrain.py",
    "core/baseline.py",
    "core/config.py",
    "core/model.py",
    "core/polar.py",
)


class BaselineImportError(RuntimeError):
    """Raised when the authored baseline cannot be imported safely."""


def _seasonal_river_strength(
    seasonal_runoff: np.ndarray,
    terrain_mask: np.ndarray,
    downstream_index: np.ndarray,
    stream_order: np.ndarray,
) -> np.ndarray:
    """Quantize four seasonal river states on the canonical D8 network."""

    runoff = np.asarray(seasonal_runoff, dtype=np.float64)
    terrain = np.asarray(terrain_mask)
    downstream = np.asarray(downstream_index)
    order = np.asarray(stream_order)
    if runoff.ndim != 3 or runoff.shape[0] != 4:
        raise BaselineImportError("seasonal runoff must have shape (4, H, W)")
    shape = runoff.shape[1:]
    if any(value.shape != shape for value in (terrain, downstream, order)):
        raise BaselineImportError("seasonal river inputs must share one grid shape")
    if terrain.dtype != np.dtype(bool):
        raise BaselineImportError("seasonal river terrain mask must be boolean")
    if not np.all(np.isfinite(runoff)) or np.any(runoff < 0.0):
        raise BaselineImportError("seasonal runoff must be finite and non-negative")

    hydrology = _load_hydrology_core()
    accumulated = np.stack(
        [
            hydrology.accumulate_d8_runoff(season, terrain, downstream)
            for season in runoff
        ]
    )
    stream = order > 0
    result = np.zeros(runoff.shape, dtype=np.uint8)
    if not np.any(stream):
        return result
    mean_discharge = np.mean(accumulated, axis=0)
    reference = max(float(np.quantile(mean_discharge[stream], 0.98)), 1.0e-9)
    log_reference = max(float(np.log1p(reference)), 1.0e-9)
    width = shape[1]
    for season_index in range(4):
        discharge = accumulated[season_index]
        absolute = np.clip(np.log1p(discharge) / log_reference, 0.0, 1.0)
        relative = np.divide(
            discharge,
            mean_discharge,
            out=np.zeros_like(discharge),
            where=mean_discharge > 1.0e-12,
        )
        visible = stream & (
            (order >= 3)
            | ((relative >= 0.42) & (absolute >= 0.025))
        )
        strength = np.rint(
            255.0
            * np.clip(
                0.68 * absolute + 0.32 * np.clip(relative / 1.75, 0.0, 1.0),
                0.0,
                1.0,
            )
        ).astype(np.uint8)
        strength[~visible] = 0
        strength[(order >= 3) & stream] = np.maximum(
            strength[(order >= 3) & stream], 48
        )

        # Any visible reach keeps every canonical downstream reach visible.
        active = strength > 0
        for start in np.flatnonzero(active.ravel()):
            current = int(start)
            for _ in range(stream.size):
                target = int(downstream.flat[current])
                if target < 0 or not stream.flat[target]:
                    break
                if strength.flat[target] == 0:
                    strength.flat[target] = max(24, int(strength.flat[current]))
                current = target
            else:  # pragma: no cover - canonical D8 is already acyclic
                raise BaselineImportError("seasonal river closure found a D8 cycle")
        result[season_index] = strength
    return result


def _rgb_hex(value: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02x}" for channel in value)


def _cell_truth_source_paths() -> tuple[tuple[str, Path], ...]:
    project_root = Path(__file__).resolve().parents[1]
    return tuple(
        (relative_path, project_root / Path(relative_path))
        for relative_path in CELL_TRUTH_SOURCE_FILES
    )


def cell_truth_source_digest() -> str:
    """Return a stable digest of every local source that can change cell truth."""

    digest = hashlib.sha256()
    for relative_path, source_path in _cell_truth_source_paths():
        try:
            source_bytes = source_path.read_bytes()
        except OSError as error:
            raise BaselineImportError(
                f"unable to read cell-truth source dependency: {source_path}"
            ) from error
        digest.update(b"cell-truth-source\0")
        digest.update(relative_path.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(source_bytes).hexdigest().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_and_validate_source(source_path: Path, config: WorldConfig) -> bytes:
    try:
        source_bytes = source_path.read_bytes()
    except OSError as error:
        raise BaselineImportError(f"unable to read baseline source: {source_path}") from error
    actual_hash = hashlib.sha256(source_bytes).hexdigest().upper()
    if actual_hash != str(config.source.sha256).upper():
        raise BaselineImportError(
            "baseline source SHA-256 does not match worldgen.json: "
            f"expected {config.source.sha256}, got {actual_hash}"
        )
    return source_bytes


def compute_input_fingerprint(
    source_path: str | Path,
    config: WorldConfig,
    importer_version: str = BASELINE_IMPORTER_VERSION,
) -> str:
    """Hash source bytes, physical config, importer identity, and cell-truth code."""

    source = Path(source_path).resolve()
    source_bytes = _read_and_validate_source(source, config)
    digest = hashlib.sha256()
    digest.update(b"source-bytes\0")
    digest.update(source_bytes)
    if config.source.field_bundle_path is not None:
        try:
            bundle_bytes = config.source.field_bundle_path.read_bytes()
        except OSError as error:
            raise BaselineImportError(
                f"unable to read procedural field bundle: {config.source.field_bundle_path}"
            ) from error
        actual_bundle_hash = hashlib.sha256(bundle_bytes).hexdigest().upper()
        if actual_bundle_hash != str(config.source.field_bundle_sha256).upper():
            raise BaselineImportError(
                "procedural field bundle SHA-256 does not match worldgen.json"
            )
        digest.update(b"procedural-field-bundle\0")
        digest.update(bundle_bytes)
    digest.update(b"normalized-config\0")
    digest.update(
        json.dumps(
            physical_payload(config),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"importer-version\0")
    digest.update(str(importer_version).encode("utf-8"))
    digest.update(b"cell-truth-source-digest\0")
    digest.update(cell_truth_source_digest().encode("ascii"))
    return digest.hexdigest()


def load_baseline(
    config: WorldConfig | str | Path,
) -> WorldGrid:
    """Load and classify the configured PNG, returning one immutable WorldGrid."""

    active_config = load_world_config(config) if isinstance(config, (str, Path)) else config
    source_path = active_config.source.path.resolve()
    # Fingerprint and source declaration validation happen before the
    # importer opens and classifies the expensive raster.
    before_fingerprint = compute_input_fingerprint(
        source_path,
        active_config,
        BASELINE_IMPORTER_VERSION,
    )
    try:
        fields = _prepare_physical_fields(source_path, active_config)
    except BaselineImportError:
        raise
    except Exception as error:
        raise BaselineImportError(f"unable to prepare baseline fields: {error}") from error
    after_fingerprint = compute_input_fingerprint(
        source_path,
        active_config,
        BASELINE_IMPORTER_VERSION,
    )
    if after_fingerprint != before_fingerprint:
        raise BaselineImportError(
            "baseline input fingerprint changed during field import"
        )
    return _world_grid_from_physical_fields(fields, active_config, before_fingerprint)


def _world_grid_from_physical_fields(
    fields: Any,
    config: WorldConfig,
    input_fingerprint: str,
) -> WorldGrid:
    """Convert ``PhysicalFields`` once at the importer boundary.

    The old builder is allowed to supply cell truth here only.  Geometry and
    serialized vector layers are intentionally not read by this adapter.
    """

    expected_shape = (config.grid.height, config.grid.width)
    continuous_elevation = _validated_array(
        fields.continuous_elevation,
        "continuous_elevation",
        expected_shape,
        kind="floating",
        minimum=0.0,
        maximum=1.0,
    )
    elevation_ordinal = _validated_array(
        fields.elevation_ordinal,
        "elevation_ordinal",
        expected_shape,
        kind="integer",
        minimum=-1,
        maximum=min(config.palettes.elevation_levels - 1, np.iinfo(np.uint8).max),
    )
    bathymetry = _validated_array(
        fields.bathymetry,
        "bathymetry",
        expected_shape,
        kind="integer",
        minimum=-1,
        maximum=min(config.palettes.bathymetry_levels - 1, np.iinfo(np.int8).max),
    )
    downstream_index = _validated_array(
        fields.hydrology.downstream_index,
        "hydrology.downstream_index",
        expected_shape,
        kind="integer",
        minimum=-1,
        maximum=min(int(np.prod(expected_shape, dtype=np.int64)) - 1, np.iinfo(np.int32).max),
    )
    accumulation = _validated_array(
        fields.hydrology.accumulation,
        "hydrology.accumulation",
        expected_shape,
        kind="floating",
        minimum=0.0,
        maximum=float(np.finfo(np.float32).max),
    )
    stream_order = _validated_array(
        fields.hydrology.stream_order,
        "hydrology.stream_order",
        expected_shape,
        kind="integer",
        minimum=0,
        maximum=np.iinfo(np.uint8).max,
    )
    perennial_snow_mask = _validated_array(
        fields.perennial_snow_mask,
        "perennial_snow_mask",
        expected_shape,
        kind="boolean",
    )
    ocean_mask = _validated_array(
        fields.ocean_mask,
        "ocean_mask",
        expected_shape,
        kind="boolean",
    )
    lake_mask = _validated_array(
        fields.lake_mask,
        "lake_mask",
        expected_shape,
        kind="boolean",
    )
    inland_sea_mask = _validated_array(
        fields.inland_sea_mask,
        "inland_sea_mask",
        expected_shape,
        kind="boolean",
    )
    land_mask = _validated_array(
        fields.land_mask,
        "land_mask",
        expected_shape,
        kind="boolean",
    )
    north_polar_land_mask = _validated_array(
        fields.generated_north_polar_land_mask,
        "generated_north_polar_land_mask",
        expected_shape,
        kind="boolean",
    )
    south_polar_land_mask = _validated_array(
        fields.generated_south_polar_land_mask,
        "generated_south_polar_land_mask",
        expected_shape,
        kind="boolean",
    )
    if np.any(north_polar_land_mask & south_polar_land_mask):
        raise BaselineImportError("generated polar continent masks must be disjoint")
    if np.any((north_polar_land_mask | south_polar_land_mask) & ~land_mask):
        raise BaselineImportError("generated polar continent masks must be canonical land")
    if np.any(inland_sea_mask & ~lake_mask):
        raise BaselineImportError("inland_sea_mask must be a subset of lake_mask")
    if (
        np.any(ocean_mask & lake_mask)
        or np.any(ocean_mask & land_mask)
        or np.any(lake_mask & land_mask)
        or np.any(~(ocean_mask | lake_mask | land_mask))
    ):
        raise BaselineImportError("prepared surface masks are not a complete disjoint partition")
    latitude_north = config.grid.latitude_extent[1]
    latitude_south = config.grid.latitude_extent[0]
    latitude_rows = latitude_north - (
        np.arange(expected_shape[0], dtype=np.float64) + 0.5
    ) * ((latitude_north - latitude_south) / expected_shape[0])
    polar_circle_latitude = 90.0 - config.planet.axial_tilt_degrees
    polar_latitude_mask = np.abs(latitude_rows)[:, None] >= polar_circle_latitude
    if np.any(lake_mask & polar_latitude_mask):
        raise BaselineImportError("polar regions must not contain lakes")
    arrays = {
        "elevation": continuous_elevation.astype(np.float32),
        # Water has no authored land ordinal and arrives as -1; represent it
        # with the lowest valid display band rather than narrowing -1 to 255.
        "elevation_band": np.maximum(elevation_ordinal, 0).astype(np.uint8),
        "bathymetry_band": bathymetry.astype(np.int8),
        "flow_to": downstream_index.astype(np.int32),
        "discharge": accumulation.astype(np.float32),
        "river_order": stream_order.astype(np.uint8),
        "snow": perennial_snow_mask.copy(),
    }
    for name, array in arrays.items():
        if array.shape != expected_shape:
            raise BaselineImportError(f"prepared field {name} has shape {array.shape}")
    water = np.zeros(expected_shape, dtype=np.uint8)
    water[ocean_mask] = 1
    water[lake_mask] = 2
    water[inland_sea_mask] = 3
    arrays["water"] = water

    hydroclimate = fields.hydroclimate
    seasonal_shape = (4, *expected_shape)
    seasonal_precipitation = _validated_seasonal_array(
        hydroclimate.seasonal_precipitation,
        "hydroclimate.seasonal_precipitation",
        seasonal_shape,
        minimum=0.0,
        maximum=_RELATIVE_PRECIPITATION_MAXIMUM,
    )
    seasonal_wind_east = _validated_seasonal_array(
        hydroclimate.seasonal_wind_east,
        "hydroclimate.seasonal_wind_east",
        seasonal_shape,
        minimum=-1.0,
        maximum=1.0,
    )
    seasonal_wind_north = _validated_seasonal_array(
        hydroclimate.seasonal_wind_north,
        "hydroclimate.seasonal_wind_north",
        seasonal_shape,
        minimum=-1.0,
        maximum=1.0,
    )
    seasonal_runoff = _validated_seasonal_array(
        hydroclimate.seasonal_runoff_yield,
        "hydroclimate.seasonal_runoff_yield",
        seasonal_shape,
        minimum=0.0,
    )
    arrays["seasonal_precipitation"] = np.rint(
        np.clip(
            seasonal_precipitation / _RELATIVE_PRECIPITATION_MAXIMUM,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(np.uint8)
    arrays["seasonal_wind_east"] = np.rint(
        np.clip(seasonal_wind_east, -1.0, 1.0) * 127.0
    ).astype(np.int8)
    arrays["seasonal_wind_north"] = np.rint(
        np.clip(seasonal_wind_north, -1.0, 1.0) * 127.0
    ).astype(np.int8)
    arrays["seasonal_river_strength"] = _seasonal_river_strength(
        seasonal_runoff,
        land_mask | lake_mask,
        downstream_index,
        stream_order,
    )

    extents = {
        "west": config.grid.longitude_extent[0],
        "south": config.grid.latitude_extent[0],
        "east": config.grid.longitude_extent[1],
        "north": config.grid.latitude_extent[1],
    }
    polar_regions = apply_polar_sea_ice(
        arrays,
        extents,
        north_polar_land_mask,
        south_polar_land_mask,
    )

    left, right, top, bottom = config.grid.padding_px
    try:
        source_artifact = Path(
            os.path.relpath(config.source.path, start=config.path.parent)
        ).as_posix()
    except ValueError:
        source_artifact = config.source.path.name
    metadata = {
        "format": "eirenor-world-grid",
        "schema": "world-grid-v3",
        "inputFingerprint": input_fingerprint,
        "source": {
            "path": source_artifact,
            "sha256": config.source.sha256,
            "registration": config.source.registration,
        },
        "planet": {
            "radiusKm": config.planet.radius_km,
            "gravityMps2": config.planet.gravity_mps2,
            "rotationPeriodHours": config.planet.rotation_period_hours,
            "rotationDirection": config.planet.rotation_direction,
            "axialTiltDegrees": config.planet.axial_tilt_degrees,
            "orbitalPeriodDays": config.planet.orbital_period_days,
            "orbitalEccentricity": config.planet.orbital_eccentricity,
            "longitudeOfPeriapsisDegrees": (
                config.planet.longitude_of_periapsis_degrees
            ),
            "solarConstantWm2": config.planet.solar_constant_wm2,
            "surfacePressurePa": config.planet.surface_pressure_pa,
            "oceanHeatCapacityFactor": config.planet.ocean_heat_capacity_factor,
        },
        "extents": extents,
        "coordinateReferenceSystem": {
            "contractId": "EIR-GEOG-1",
            "referenceBody": "eirenor",
            "referenceSurface": "sphere",
            "latitudeType": "planetocentric",
            "longitudeConvention": "east-positive",
            "primeMeridianLongitude": 0.0,
            "storageGridMapping": "plate-carree",
            "reviewProjection": "equirectangular",
            "centralMeridianLongitude": 0.0,
            "standardParallelLatitude": 0.0,
            "readerProjection": "equal-earth",
            "polarViews": ["north-polar-equal-area", "south-polar-equal-area"],
        },
        "polarRegions": {
            "arctic": {
                "surface": (
                    "continental-land"
                    if bool(polar_regions.north_land.any())
                    else "source-defined-terrain"
                ),
                "cover": (
                    "grounded-ice-sheet-and-polar-biomes"
                    if bool(polar_regions.north_land.any())
                    else "latitude-derived-sea-ice"
                ),
                "generatedLandCells": int(
                    np.count_nonzero(polar_regions.north_land)
                ),
                "medianCoastLatitude": float(
                    np.median(polar_regions.north_coast_latitude)
                ),
                "medianOceanIceEdgeLatitude": float(
                    np.median(polar_regions.north_ice_edge_latitude)
                ),
            },
            "antarctic": {
                "surface": (
                    "continental-land"
                    if bool(polar_regions.south_land.any())
                    else "source-defined-terrain"
                ),
                "cover": (
                    "grounded-ice-sheet-and-polar-biomes"
                    if bool(polar_regions.south_land.any())
                    else "latitude-derived-sea-ice"
                ),
                "generatedLandCells": int(
                    np.count_nonzero(polar_regions.south_land)
                ),
                "medianCoastLatitude": float(
                    np.median(polar_regions.south_coast_latitude)
                ),
                "medianOceanIceEdgeLatitude": float(
                    np.median(polar_regions.south_ice_edge_latitude)
                ),
            },
            "generationStage": "before-terrain-refinement-hydrology-snow-and-climate",
            "societyPolicy": "uninhabitable-climate-and-biome-cells-stop-before-population",
        },
        "grid": {
            "width": config.grid.width,
            "height": config.grid.height,
            "boardWidthPx": config.grid.board_width_px,
            "boardHeightPx": config.grid.board_height_px,
            "paddingPx": {"left": left, "right": right, "top": top, "bottom": bottom},
            "samplingStepPx": config.grid.sampling_step_px,
        },
        "waterCodes": {"land": 0, "ocean": 1, "lake": 2, "inlandSea": 3},
        "elevation": {"units": "relative-palette-position", "levels": config.palettes.elevation_levels},
        "bathymetry": {
            "units": "relative-palette-position",
            "levels": config.palettes.bathymetry_levels,
        },
        "climate": {
            "seasons": list(_SEASON_IDS),
            "solarLongitudeDegrees": list(_SEASON_SOLAR_LONGITUDES),
            "precipitationQuantization": (
                "relative-linear-uint8; 0..255 maps to 0..1.35 model units"
            ),
            "windQuantization": (
                "relative-signed-int8; -127..127 maps to -1..1 components"
            ),
            "riverQuantization": (
                "relative-linear-uint8 on canonical D8 stream network"
            ),
            "model": str(hydroclimate.model_name),
        },
        "classification": {
            "landPalette": [_rgb_hex(value) for value in config.palettes.land],
            "bathymetryPalette": [
                _rgb_hex(value) for value in config.palettes.bathymetry
            ],
        },
        "provenance": {
            "importerVersion": BASELINE_IMPORTER_VERSION,
            "sourceFieldVersion": fields.physical_version,
        },
    }
    return WorldGrid(metadata=metadata, **arrays)


def _prepare_physical_fields(source_path: Path, config: WorldConfig) -> Any:
    """Call the old PNG classifier only at this temporary importer boundary."""

    builder = _load_field_builder()
    from_context = builder.BuildContext.reviewed()
    from PIL import Image

    with Image.open(source_path) as image:
        source_dimensions = tuple(int(value) for value in image.size)
    context = replace(
        from_context,
        source_name=source_path.name,
        source_artifact=str(config.source.path),
        source_dimensions=source_dimensions,
        board_dimensions=(config.grid.board_width_px, config.grid.board_height_px),
        padding=config.grid.padding_px,
        sampling_step_px=config.grid.sampling_step_px,
        grid_dimensions=(config.grid.width, config.grid.height),
        longitude_extent=config.grid.longitude_extent,
        latitude_extent=config.grid.latitude_extent,
        complete_globe=(
            config.source.registration == "procedural-equirectangular"
        ),
        elevation_levels=config.palettes.elevation_levels,
        bathymetry_levels=config.palettes.bathymetry_levels,
        land_palette=config.palettes.land,
        bathymetry_palette=config.palettes.bathymetry,
        elevation_regularization_sigma_grid_units=(
            config.elevation_regularization.sigma_grid_units
        ),
        elevation_regularization_passes=config.elevation_regularization.passes,
        hydrology_config=builder.HydrologyConfig(
            stream_burn_depth=config.hydrology.stream_burn_depth,
            stream_threshold_fraction=config.hydrology.stream_threshold_fraction,
            tributary_threshold_fraction=config.hydrology.tributary_threshold_fraction,
            mainstem_threshold_fraction=config.hydrology.mainstem_threshold_fraction,
            minimum_headwater_length=config.hydrology.minimum_headwater_length,
            tie_epsilon=config.hydrology.tie_epsilon,
            minimum_stream_span_factor=config.hydrology.minimum_stream_span_factor,
            closure_budget_fraction=config.hydrology.closure_budget_fraction,
            closure_max_chain_fraction=config.hydrology.closure_max_chain_fraction,
            mfd_exponent=config.hydrology.mfd_exponent,
            valley_window_fraction=config.hydrology.valley_window_fraction,
            valley_depth_fraction=config.hydrology.valley_depth_fraction,
            valley_support_threshold=config.hydrology.valley_support_threshold,
            valley_support_distance_factor=config.hydrology.valley_support_distance_factor,
            parallel_search_radius_factor=config.hydrology.parallel_search_radius_factor,
            parallel_priority_weight=config.hydrology.parallel_priority_weight,
            runoff_weight_floor=config.hydrology.runoff_weight_floor,
            snowmelt_runoff_bonus=config.hydrology.snowmelt_runoff_bonus,
            headwater_upland_quantile=config.hydrology.headwater_upland_quantile,
            lowland_headwater_flow_multiplier=(
                config.hydrology.lowland_headwater_flow_multiplier
            ),
            major_rivers_per_continent=(
                config.hydrology.major_rivers_per_continent
            ),
            major_river_tributaries=config.hydrology.major_river_tributaries,
            continent_minimum_land_fraction=(
                config.hydrology.continent_minimum_land_fraction
            ),
            major_river_minimum_span_factor=(
                config.hydrology.major_river_minimum_span_factor
            ),
            lake_outlets=(
                {
                    anchor: outlet
                    for anchor, outlet in (config.hydrology.lake_outlets or ())
                }
                if config.hydrology.lake_outlets
                else None
            ),
            inland_sink_cells=config.hydrology.inland_sink_cells,
        ),
        snowline_config=builder.SnowlineConfig(
            equator_threshold=config.snowline.equator_threshold,
            pole_threshold=config.snowline.pole_threshold,
            exponent=config.snowline.exponent,
            max_offset=config.snowline.max_offset,
            polar_full_snow_latitude=config.snowline.polar_full_snow_latitude,
        ),
        planet=builder.PlanetConfig(
            radius_km=config.planet.radius_km,
            gravity_mps2=config.planet.gravity_mps2,
            rotation_period_hours=config.planet.rotation_period_hours,
            rotation_direction=config.planet.rotation_direction,
            axial_tilt_degrees=config.planet.axial_tilt_degrees,
            orbital_period_days=config.planet.orbital_period_days,
            orbital_eccentricity=config.planet.orbital_eccentricity,
            longitude_of_periapsis_degrees=(
                config.planet.longitude_of_periapsis_degrees
            ),
            solar_constant_wm2=config.planet.solar_constant_wm2,
            surface_pressure_pa=config.planet.surface_pressure_pa,
            ocean_heat_capacity_factor=config.planet.ocean_heat_capacity_factor,
        ),
    )
    procedural_surface = (
        load_surface_bundle(config.source.field_bundle_path)
        if config.source.field_bundle_path is not None
        else None
    )
    return builder.prepare_physical_fields(
        source_path,
        context=context,
        procedural_surface=procedural_surface,
    )


def _validated_array(
    value: Any,
    name: str,
    expected_shape: tuple[int, int],
    *,
    kind: str,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
) -> np.ndarray:
    """Validate a PhysicalFields array before any narrowing cast occurs."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise BaselineImportError(f"{name} must be a NumPy array") from error
    if array.shape != expected_shape:
        raise BaselineImportError(
            f"{name} has shape {array.shape}; expected {expected_shape}"
        )
    if kind == "boolean":
        if array.dtype != np.dtype(bool):
            raise BaselineImportError(f"{name} must have boolean dtype")
    elif kind == "integer":
        if array.dtype.kind not in "iu":
            raise BaselineImportError(f"{name} must have integer dtype")
    elif kind == "floating":
        if array.dtype.kind != "f":
            raise BaselineImportError(f"{name} must have floating-point dtype")
    else:  # pragma: no cover - internal call-site invariant
        raise BaselineImportError(f"unknown validation kind: {kind}")
    if array.dtype.kind != "b" and not np.all(np.isfinite(array)):
        raise BaselineImportError(f"{name} contains non-finite values")
    if minimum is not None and np.any(array < minimum):
        raise BaselineImportError(f"{name} contains values below {minimum}")
    if maximum is not None and np.any(array > maximum):
        raise BaselineImportError(f"{name} contains values above {maximum}")
    return array


def _validated_seasonal_array(
    value: Any,
    name: str,
    expected_shape: tuple[int, int, int],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise BaselineImportError(f"{name} must be a NumPy array") from error
    if array.shape != expected_shape or array.dtype.kind != "f":
        raise BaselineImportError(
            f"{name} must be a floating array with shape {expected_shape}"
        )
    if not np.all(np.isfinite(array)):
        raise BaselineImportError(f"{name} contains non-finite values")
    if minimum is not None and np.any(array < minimum):
        raise BaselineImportError(f"{name} contains values below {minimum}")
    if maximum is not None and np.any(array > maximum):
        raise BaselineImportError(f"{name} contains values above {maximum}")
    return array


def _load_field_builder() -> Any:
    from world_atlas.physical import build_source
    return build_source


def _load_hydrology_core() -> Any:
    from world_atlas.physical import hydrology
    return hydrology


__all__ = [
    "BASELINE_IMPORTER_VERSION",
    "BaselineImportError",
    "CELL_TRUTH_SOURCE_FILES",
    "cell_truth_source_digest",
    "compute_input_fingerprint",
    "load_baseline",
    "physical_payload",
]
