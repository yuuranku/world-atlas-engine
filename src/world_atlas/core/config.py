"""Strict, small configuration for the first WorldGrid baseline slice."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


class WorldConfigError(ValueError):
    """Raised when a WorldGrid configuration is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class SourceConfig:
    path: Path
    sha256: str
    registration: str
    field_bundle_path: Path | None
    field_bundle_sha256: str | None


@dataclass(frozen=True, slots=True)
class PlanetConfig:
    radius_km: float
    gravity_mps2: float
    rotation_period_hours: float
    rotation_direction: str
    axial_tilt_degrees: float
    orbital_period_days: float
    orbital_eccentricity: float
    longitude_of_periapsis_degrees: float
    solar_constant_wm2: float
    surface_pressure_pa: float
    ocean_heat_capacity_factor: float


@dataclass(frozen=True, slots=True)
class GridConfig:
    width: int
    height: int
    board_width_px: int
    board_height_px: int
    padding_px: tuple[int, int, int, int]
    sampling_step_px: int
    longitude_extent: tuple[float, float]
    latitude_extent: tuple[float, float]

    @property
    def cell_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class PaletteConfig:
    land: tuple[tuple[int, int, int], ...]
    bathymetry: tuple[tuple[int, int, int], ...]
    elevation_levels: int
    bathymetry_levels: int


@dataclass(frozen=True, slots=True)
class ElevationRegularizationConfig:
    sigma_grid_units: float
    passes: int


@dataclass(frozen=True, slots=True)
class HydrologyConfig:
    stream_burn_depth: float
    stream_threshold_fraction: float
    tributary_threshold_fraction: float
    mainstem_threshold_fraction: float
    minimum_headwater_length: int
    tie_epsilon: float
    minimum_stream_span_factor: float
    closure_budget_fraction: float
    closure_max_chain_fraction: float
    mfd_exponent: float
    valley_window_fraction: float
    valley_depth_fraction: float
    valley_support_threshold: float
    valley_support_distance_factor: float
    parallel_search_radius_factor: float
    parallel_priority_weight: float
    runoff_weight_floor: float
    snowmelt_runoff_bonus: float
    headwater_upland_quantile: float
    lowland_headwater_flow_multiplier: float
    major_rivers_per_continent: int
    major_river_tributaries: int
    continent_minimum_land_fraction: float
    major_river_minimum_span_factor: float
    lake_outlets: tuple[tuple[tuple[int, int], tuple[int, int]], ...] | None = None
    inland_sink_cells: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class SnowlineConfig:
    equator_threshold: float
    pole_threshold: float
    exponent: float
    max_offset: float
    polar_full_snow_latitude: float


@dataclass(frozen=True, slots=True)
class OutputConfig:
    directory: Path


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    max_cells: int


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """The complete current configuration; there is intentionally no v1 fallback."""

    path: Path
    format: str
    schema_version: int
    source: SourceConfig
    planet: PlanetConfig
    grid: GridConfig
    palettes: PaletteConfig
    elevation_regularization: ElevationRegularizationConfig
    hydrology: HydrologyConfig
    snowline: SnowlineConfig
    output: OutputConfig
    budget: BudgetConfig

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: str | Path = ".",
        path: str | Path | None = None,
    ) -> "WorldConfig":
        root = _strict_mapping(
            value,
            "worldgen",
            {
                "format",
                "schemaVersion",
                "source",
                "planet",
                "grid",
                "palettes",
                "elevationRegularization",
                "hydrology",
                "snowline",
                "output",
                "budget",
            },
        )
        if root["format"] != "eirenor-worldgen":
            raise WorldConfigError("format must be 'eirenor-worldgen'")
        schema_version = _integer(root["schemaVersion"], "schemaVersion", minimum=1)
        if schema_version != 2:
            raise WorldConfigError("schemaVersion must be 2")
        base = Path(base_dir).resolve()

        source_value = _strict_mapping(
            root["source"],
            "source",
            {"path", "sha256", "registration", "fieldBundle"},
        )
        source_path_value = _string(source_value["path"], "source.path")
        source_path = Path(source_path_value)
        if not source_path.is_absolute():
            source_path = base / source_path
        registration = _string(source_value["registration"], "source.registration")
        bundle_path: Path | None = None
        bundle_sha256: str | None = None
        if "fieldBundle" in source_value:
            bundle_value = _strict_mapping(
                source_value["fieldBundle"],
                "source.fieldBundle",
                {"path", "sha256"},
            )
            bundle_path = Path(
                _string(bundle_value["path"], "source.fieldBundle.path")
            )
            if not bundle_path.is_absolute():
                bundle_path = base / bundle_path
            bundle_path = bundle_path.resolve()
            bundle_sha256 = _hash(
                bundle_value["sha256"], "source.fieldBundle.sha256"
            )
        if registration == "procedural-equirectangular" and bundle_path is None:
            raise WorldConfigError(
                "procedural-equirectangular sources require source.fieldBundle"
            )
        source = SourceConfig(
            path=source_path.resolve(),
            sha256=_hash(source_value["sha256"], "source.sha256"),
            registration=registration,
            field_bundle_path=bundle_path,
            field_bundle_sha256=bundle_sha256,
        )

        planet_value = _strict_mapping(
            root["planet"],
            "planet",
            {
                "radiusKm",
                "gravityMps2",
                "rotationPeriodHours",
                "rotationDirection",
                "axialTiltDegrees",
                "orbitalPeriodDays",
                "orbitalEccentricity",
                "longitudeOfPeriapsisDegrees",
                "solarConstantWm2",
                "surfacePressurePa",
                "oceanHeatCapacityFactor",
            },
        )
        rotation_direction = _string(
            planet_value["rotationDirection"], "planet.rotationDirection"
        )
        if rotation_direction not in {"prograde", "retrograde"}:
            raise WorldConfigError(
                "planet.rotationDirection must be 'prograde' or 'retrograde'"
            )
        orbital_eccentricity = _nonnegative_number(
            planet_value["orbitalEccentricity"], "planet.orbitalEccentricity"
        )
        if orbital_eccentricity >= 1.0:
            raise WorldConfigError("planet.orbitalEccentricity must be below 1")
        planet = PlanetConfig(
            radius_km=_positive_number(planet_value["radiusKm"], "planet.radiusKm"),
            gravity_mps2=_positive_number(
                planet_value["gravityMps2"], "planet.gravityMps2"
            ),
            rotation_period_hours=_positive_number(
                planet_value["rotationPeriodHours"], "planet.rotationPeriodHours"
            ),
            rotation_direction=rotation_direction,
            axial_tilt_degrees=_bounded_number(
                planet_value["axialTiltDegrees"],
                "planet.axialTiltDegrees",
                0.0,
                90.0,
            ),
            orbital_period_days=_positive_number(
                planet_value["orbitalPeriodDays"], "planet.orbitalPeriodDays"
            ),
            orbital_eccentricity=orbital_eccentricity,
            longitude_of_periapsis_degrees=_bounded_number(
                planet_value["longitudeOfPeriapsisDegrees"],
                "planet.longitudeOfPeriapsisDegrees",
                0.0,
                360.0,
            ),
            solar_constant_wm2=_positive_number(
                planet_value["solarConstantWm2"], "planet.solarConstantWm2"
            ),
            surface_pressure_pa=_positive_number(
                planet_value["surfacePressurePa"], "planet.surfacePressurePa"
            ),
            ocean_heat_capacity_factor=_positive_number(
                planet_value["oceanHeatCapacityFactor"],
                "planet.oceanHeatCapacityFactor",
            ),
        )

        grid_value = _strict_mapping(
            root["grid"],
            "grid",
            {
                "width",
                "height",
                "boardWidthPx",
                "boardHeightPx",
                "paddingPx",
                "samplingStepPx",
                "longitudeExtent",
                "latitudeExtent",
            },
        )
        width = _integer(grid_value["width"], "grid.width", minimum=2)
        height = _integer(grid_value["height"], "grid.height", minimum=2)
        board_width = _integer(grid_value["boardWidthPx"], "grid.boardWidthPx", minimum=1)
        board_height = _integer(grid_value["boardHeightPx"], "grid.boardHeightPx", minimum=1)
        padding_value = _strict_mapping(
            grid_value["paddingPx"],
            "grid.paddingPx",
            {"left", "right", "top", "bottom"},
        )
        padding = tuple(
            _integer(padding_value[key], f"grid.paddingPx.{key}", minimum=0)
            for key in ("left", "right", "top", "bottom")
        )
        step = _integer(grid_value["samplingStepPx"], "grid.samplingStepPx", minimum=1)
        if board_width % step or board_height % step:
            raise WorldConfigError("grid samplingStepPx must divide the authoring board")
        if width != board_width // step or height != board_height // step:
            raise WorldConfigError(
                "grid dimensions must equal board dimensions divided by samplingStepPx"
            )
        longitude_extent = _extent(grid_value["longitudeExtent"], "grid.longitudeExtent", -180.0, 180.0)
        latitude_extent = _extent(grid_value["latitudeExtent"], "grid.latitudeExtent", -90.0, 90.0)
        grid = GridConfig(
            width=width,
            height=height,
            board_width_px=board_width,
            board_height_px=board_height,
            padding_px=padding,
            sampling_step_px=step,
            longitude_extent=longitude_extent,
            latitude_extent=latitude_extent,
        )

        palettes_value = _strict_mapping(
            root["palettes"],
            "palettes",
            {"land", "bathymetry", "levels"},
        )
        levels_value = _strict_mapping(
            palettes_value["levels"],
            "palettes.levels",
            {"elevation", "bathymetry"},
        )
        palettes = PaletteConfig(
            land=_palette(palettes_value["land"], "palettes.land"),
            bathymetry=_palette(palettes_value["bathymetry"], "palettes.bathymetry"),
            elevation_levels=_integer(
                levels_value["elevation"], "palettes.levels.elevation", minimum=1
            ),
            bathymetry_levels=_integer(
                levels_value["bathymetry"], "palettes.levels.bathymetry", minimum=1
            ),
        )
        if palettes.elevation_levels < 2 or palettes.bathymetry_levels < 2:
            raise WorldConfigError("palette levels must be at least 2")

        elevation_value = _strict_mapping(
            root["elevationRegularization"],
            "elevationRegularization",
            {"sigmaGridUnits", "passes"},
        )
        elevation_regularization = ElevationRegularizationConfig(
            sigma_grid_units=_nonnegative_number(
                elevation_value["sigmaGridUnits"],
                "elevationRegularization.sigmaGridUnits",
            ),
            passes=_integer(elevation_value["passes"], "elevationRegularization.passes", minimum=0),
        )

        hydrology_value = _strict_mapping(
            root["hydrology"],
            "hydrology",
            {
                "streamBurnDepth",
                "streamThresholdFraction",
                "tributaryThresholdFraction",
                "mainstemThresholdFraction",
                "minimumHeadwaterLength",
                "tieEpsilon",
                "minimumStreamSpanFactor",
                "closureBudgetFraction",
                "closureMaxChainFraction",
                "mfdExponent",
                "valleyWindowFraction",
                "valleyDepthFraction",
                "valleySupportThreshold",
                "valleySupportDistanceFactor",
                "parallelSearchRadiusFactor",
                "parallelPriorityWeight",
                "runoffWeightFloor",
                "snowmeltRunoffBonus",
                "headwaterUplandQuantile",
                "lowlandHeadwaterFlowMultiplier",
                "majorRiversPerContinent",
                "majorRiverTributaries",
                "continentMinimumLandFraction",
                "majorRiverMinimumSpanFactor",
                "lakeOutlets",
                "inlandSinkCells",
            },
        )
        hydrology = HydrologyConfig(
            stream_burn_depth=_nonnegative_number(
                hydrology_value["streamBurnDepth"], "hydrology.streamBurnDepth"
            ),
            stream_threshold_fraction=_nonnegative_number(
                hydrology_value["streamThresholdFraction"],
                "hydrology.streamThresholdFraction",
            ),
            tributary_threshold_fraction=_nonnegative_number(
                hydrology_value["tributaryThresholdFraction"],
                "hydrology.tributaryThresholdFraction",
            ),
            mainstem_threshold_fraction=_nonnegative_number(
                hydrology_value["mainstemThresholdFraction"],
                "hydrology.mainstemThresholdFraction",
            ),
            minimum_headwater_length=_integer(
                hydrology_value["minimumHeadwaterLength"],
                "hydrology.minimumHeadwaterLength",
                minimum=1,
            ),
            tie_epsilon=_positive_number(
                hydrology_value["tieEpsilon"], "hydrology.tieEpsilon"
            ),
            minimum_stream_span_factor=_nonnegative_number(
                hydrology_value["minimumStreamSpanFactor"],
                "hydrology.minimumStreamSpanFactor",
            ),
            closure_budget_fraction=_bounded_number(
                hydrology_value["closureBudgetFraction"],
                "hydrology.closureBudgetFraction",
                0.0,
                1.0,
            ),
            closure_max_chain_fraction=_bounded_number(
                hydrology_value["closureMaxChainFraction"],
                "hydrology.closureMaxChainFraction",
                0.0,
                1.0,
            ),
            mfd_exponent=_positive_number(
                hydrology_value["mfdExponent"], "hydrology.mfdExponent"
            ),
            valley_window_fraction=_bounded_number(
                hydrology_value["valleyWindowFraction"],
                "hydrology.valleyWindowFraction",
                0.0,
                1.0,
            ),
            valley_depth_fraction=_bounded_number(
                hydrology_value["valleyDepthFraction"],
                "hydrology.valleyDepthFraction",
                0.0,
                1.0,
            ),
            valley_support_threshold=_bounded_number(
                hydrology_value["valleySupportThreshold"],
                "hydrology.valleySupportThreshold",
                0.0,
                1.0,
            ),
            valley_support_distance_factor=_nonnegative_number(
                hydrology_value["valleySupportDistanceFactor"],
                "hydrology.valleySupportDistanceFactor",
            ),
            parallel_search_radius_factor=_nonnegative_number(
                hydrology_value["parallelSearchRadiusFactor"],
                "hydrology.parallelSearchRadiusFactor",
            ),
            parallel_priority_weight=_nonnegative_number(
                hydrology_value["parallelPriorityWeight"],
                "hydrology.parallelPriorityWeight",
            ),
            runoff_weight_floor=_bounded_number(
                hydrology_value["runoffWeightFloor"],
                "hydrology.runoffWeightFloor",
                0.000000000001,
                1.0,
            ),
            snowmelt_runoff_bonus=_nonnegative_number(
                hydrology_value["snowmeltRunoffBonus"],
                "hydrology.snowmeltRunoffBonus",
            ),
            headwater_upland_quantile=_bounded_number(
                hydrology_value["headwaterUplandQuantile"],
                "hydrology.headwaterUplandQuantile",
                0.0,
                1.0,
            ),
            lowland_headwater_flow_multiplier=_bounded_number(
                hydrology_value["lowlandHeadwaterFlowMultiplier"],
                "hydrology.lowlandHeadwaterFlowMultiplier",
                1.0,
                float("inf"),
            ),
            major_rivers_per_continent=_integer(
                hydrology_value["majorRiversPerContinent"],
                "hydrology.majorRiversPerContinent",
                minimum=0,
            ),
            major_river_tributaries=_integer(
                hydrology_value["majorRiverTributaries"],
                "hydrology.majorRiverTributaries",
                minimum=0,
            ),
            continent_minimum_land_fraction=_bounded_number(
                hydrology_value["continentMinimumLandFraction"],
                "hydrology.continentMinimumLandFraction",
                0.0,
                1.0,
            ),
            major_river_minimum_span_factor=_nonnegative_number(
                hydrology_value["majorRiverMinimumSpanFactor"],
                "hydrology.majorRiverMinimumSpanFactor",
            ),
            lake_outlets=_lake_outlets(
                hydrology_value["lakeOutlets"],
                "hydrology.lakeOutlets",
                grid,
            ),
            inland_sink_cells=_cell_list(
                hydrology_value["inlandSinkCells"],
                "hydrology.inlandSinkCells",
                grid,
            ),
        )

        snowline_value = _strict_mapping(
            root["snowline"],
            "snowline",
            {
                "equatorThreshold",
                "poleThreshold",
                "exponent",
                "maxOffset",
                "polarFullSnowLatitude",
            },
        )
        snowline = SnowlineConfig(
            equator_threshold=_bounded_number(
                snowline_value["equatorThreshold"], "snowline.equatorThreshold", 0.0, 1.0
            ),
            pole_threshold=_bounded_number(
                snowline_value["poleThreshold"], "snowline.poleThreshold", 0.0, 1.0
            ),
            exponent=_positive_number(snowline_value["exponent"], "snowline.exponent"),
            max_offset=_nonnegative_number(snowline_value["maxOffset"], "snowline.maxOffset"),
            polar_full_snow_latitude=_bounded_number(
                snowline_value["polarFullSnowLatitude"],
                "snowline.polarFullSnowLatitude",
                0.0,
                90.0,
            ),
        )

        output_value = _strict_mapping(
            root["output"],
            "output",
            {"directory"},
        )
        output_directory_value = _string(output_value["directory"], "output.directory")
        output_directory = Path(output_directory_value)
        if not output_directory.is_absolute():
            output_directory = base / output_directory
        output = OutputConfig(directory=output_directory.resolve())

        budget_value = _strict_mapping(root["budget"], "budget", {"maxCells"})
        budget = BudgetConfig(
            max_cells=_integer(budget_value["maxCells"], "budget.maxCells", minimum=grid.cell_count)
        )
        return cls(
            path=Path(path).resolve() if path is not None else (base / "worldgen.json").resolve(),
            format="eirenor-worldgen",
            schema_version=2,
            source=source,
            planet=planet,
            grid=grid,
            palettes=palettes,
            elevation_regularization=elevation_regularization,
            hydrology=hydrology,
            snowline=snowline,
            output=output,
            budget=budget,
        )


def physical_payload(config: WorldConfig) -> dict[str, Any]:
    """Return only configuration values that can change cell-level truth.

    Source bytes, local cell-truth source, and importer identity are hashed
    separately.  Deliberately
    omit source location/provenance, output settings, and QA budget so moving
    a project or changing delivery policy cannot invalidate a physical grid.
    """

    left, right, top, bottom = config.grid.padding_px
    return {
        "sourceRegistration": config.source.registration,
        "fieldBundleSha256": config.source.field_bundle_sha256,
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
        "grid": {
            "width": config.grid.width,
            "height": config.grid.height,
            "boardWidthPx": config.grid.board_width_px,
            "boardHeightPx": config.grid.board_height_px,
            "paddingPx": {
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
            },
            "samplingStepPx": config.grid.sampling_step_px,
            "longitudeExtent": list(config.grid.longitude_extent),
            "latitudeExtent": list(config.grid.latitude_extent),
        },
        "palettes": {
            "land": [_rgb_hex(value) for value in config.palettes.land],
            "bathymetry": [_rgb_hex(value) for value in config.palettes.bathymetry],
            "levels": {
                "elevation": config.palettes.elevation_levels,
                "bathymetry": config.palettes.bathymetry_levels,
            },
        },
        "elevationRegularization": {
            "sigmaGridUnits": config.elevation_regularization.sigma_grid_units,
            "passes": config.elevation_regularization.passes,
        },
        "hydrology": {
            "streamBurnDepth": config.hydrology.stream_burn_depth,
            "streamThresholdFraction": config.hydrology.stream_threshold_fraction,
            "tributaryThresholdFraction": config.hydrology.tributary_threshold_fraction,
            "mainstemThresholdFraction": config.hydrology.mainstem_threshold_fraction,
            "minimumHeadwaterLength": config.hydrology.minimum_headwater_length,
            "tieEpsilon": config.hydrology.tie_epsilon,
            "minimumStreamSpanFactor": config.hydrology.minimum_stream_span_factor,
            "closureBudgetFraction": config.hydrology.closure_budget_fraction,
            "closureMaxChainFraction": config.hydrology.closure_max_chain_fraction,
            "mfdExponent": config.hydrology.mfd_exponent,
            "valleyWindowFraction": config.hydrology.valley_window_fraction,
            "valleyDepthFraction": config.hydrology.valley_depth_fraction,
            "valleySupportThreshold": config.hydrology.valley_support_threshold,
            "valleySupportDistanceFactor": config.hydrology.valley_support_distance_factor,
            "parallelSearchRadiusFactor": config.hydrology.parallel_search_radius_factor,
            "parallelPriorityWeight": config.hydrology.parallel_priority_weight,
            "runoffWeightFloor": config.hydrology.runoff_weight_floor,
            "snowmeltRunoffBonus": config.hydrology.snowmelt_runoff_bonus,
            "headwaterUplandQuantile": config.hydrology.headwater_upland_quantile,
            "lowlandHeadwaterFlowMultiplier": (
                config.hydrology.lowland_headwater_flow_multiplier
            ),
            "majorRiversPerContinent": config.hydrology.major_rivers_per_continent,
            "majorRiverTributaries": config.hydrology.major_river_tributaries,
            "continentMinimumLandFraction": (
                config.hydrology.continent_minimum_land_fraction
            ),
            "majorRiverMinimumSpanFactor": (
                config.hydrology.major_river_minimum_span_factor
            ),
            "lakeOutlets": [
                {"anchor": list(anchor), "outlet": list(outlet)}
                for anchor, outlet in (config.hydrology.lake_outlets or ())
            ],
            "inlandSinkCells": [
                list(cell) for cell in config.hydrology.inland_sink_cells
            ],
        },
        "snowline": {
            "equatorThreshold": config.snowline.equator_threshold,
            "poleThreshold": config.snowline.pole_threshold,
            "exponent": config.snowline.exponent,
            "maxOffset": config.snowline.max_offset,
            "polarFullSnowLatitude": config.snowline.polar_full_snow_latitude,
        },
    }


def load_world_config(path: str | Path) -> WorldConfig:
    """Load the one current WorldGrid configuration document."""

    config_path = Path(path).resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WorldConfigError(f"unable to read worldgen config: {config_path}") from error
    if not isinstance(value, Mapping):
        raise WorldConfigError("worldgen config must be an object")
    return WorldConfig.from_mapping(value, base_dir=config_path.parent, path=config_path)


def _strict_mapping(value: Any, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorldConfigError(f"{name} must be an object")
    actual = set(value)
    missing = keys - actual
    extra = actual - keys
    if missing or extra:
        fragments: list[str] = []
        if missing:
            fragments.append(f"missing {sorted(missing)}")
        if extra:
            fragments.append(f"unexpected {sorted(extra)}")
        raise WorldConfigError(f"{name}: " + "; ".join(fragments))
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorldConfigError(f"{name} must be a non-empty string")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorldConfigError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise WorldConfigError(f"{name} must be finite")
    return number


def _positive_number(value: Any, name: str) -> float:
    number = _number(value, name)
    if number <= 0:
        raise WorldConfigError(f"{name} must be positive")
    return number


def _nonnegative_number(value: Any, name: str) -> float:
    number = _number(value, name)
    if number < 0:
        raise WorldConfigError(f"{name} must be non-negative")
    return number


def _bounded_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    number = _number(value, name)
    if not minimum <= number <= maximum:
        raise WorldConfigError(f"{name} must be within [{minimum}, {maximum}]")
    return number


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorldConfigError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise WorldConfigError(f"{name} must be at least {minimum}")
    return value


def _hash(value: Any, name: str) -> str:
    result = _string(value, name).upper()
    if re.fullmatch(r"[0-9A-F]{64}", result) is None:
        raise WorldConfigError(f"{name} must be a SHA-256 hex digest")
    return result


def _extent(value: Any, name: str, minimum: float, maximum: float) -> tuple[float, float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence) or len(value) != 2:
        raise WorldConfigError(f"{name} must contain exactly two numbers")
    start = _bounded_number(value[0], f"{name}[0]", minimum, maximum)
    end = _bounded_number(value[1], f"{name}[1]", minimum, maximum)
    if not start < end:
        raise WorldConfigError(f"{name} must be strictly increasing")
    return (start, end)


def _cell(value: Any, name: str, grid: GridConfig) -> tuple[int, int]:
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise WorldConfigError(f"{name} must contain [row, column]")
    row = _integer(value[0], f"{name}[0]")
    column = _integer(value[1], f"{name}[1]")
    if not (0 <= row < grid.height and 0 <= column < grid.width):
        raise WorldConfigError(
            f"{name} must be inside grid bounds {grid.height}x{grid.width}"
        )
    return (row, column)


def _cell_list(value: Any, name: str, grid: GridConfig) -> tuple[tuple[int, int], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WorldConfigError(f"{name} must be a list of cells")
    cells = tuple(_cell(raw, f"{name}[{index}]", grid) for index, raw in enumerate(value))
    if len(set(cells)) != len(cells):
        raise WorldConfigError(f"{name} must not contain duplicate cells")
    return cells


def _lake_outlets(
    value: Any,
    name: str,
    grid: GridConfig,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...] | None:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WorldConfigError(f"{name} must be a list of anchor/outlet pairs")
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for index, raw in enumerate(value):
        item_name = f"{name}[{index}]"
        if isinstance(raw, Mapping):
            item = _strict_mapping(raw, item_name, {"anchor", "outlet"})
            anchor = _cell(item["anchor"], f"{item_name}.anchor", grid)
            outlet = _cell(item["outlet"], f"{item_name}.outlet", grid)
        elif (
            not isinstance(raw, (str, bytes, bytearray))
            and isinstance(raw, Sequence)
            and len(raw) == 2
        ):
            anchor = _cell(raw[0], f"{item_name}[0]", grid)
            outlet = _cell(raw[1], f"{item_name}[1]", grid)
        else:
            raise WorldConfigError(f"{item_name} must contain anchor and outlet cells")
        pairs.append((anchor, outlet))
    anchors = [anchor for anchor, _ in pairs]
    if len(set(anchors)) != len(anchors):
        raise WorldConfigError(f"{name} must not contain duplicate anchors")
    return tuple(sorted(pairs)) or None


def _palette(value: Any, name: str) -> tuple[tuple[int, int, int], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence) or len(value) < 2:
        raise WorldConfigError(f"{name} must contain at least two RGB colours")
    parsed: list[tuple[int, int, int]] = []
    for index, raw in enumerate(value):
        colour = _string(raw, f"{name}[{index}]")
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", colour) is None:
            raise WorldConfigError(f"{name}[{index}] must be #RRGGBB")
        parsed.append(tuple(int(colour[offset : offset + 2], 16) for offset in (1, 3, 5)))
    if len(set(parsed)) != len(parsed):
        raise WorldConfigError(f"{name} must not contain duplicate colours")
    return tuple(parsed)


def _rgb_hex(value: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02x}" for channel in value)


__all__ = [
    "BudgetConfig",
    "ElevationRegularizationConfig",
    "GridConfig",
    "HydrologyConfig",
    "OutputConfig",
    "PaletteConfig",
    "PlanetConfig",
    "SnowlineConfig",
    "SourceConfig",
    "WorldConfig",
    "WorldConfigError",
    "physical_payload",
    "load_world_config",
]
