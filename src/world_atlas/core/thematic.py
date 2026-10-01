"""Deterministic thematic fields derived from the canonical WorldGrid.

These arrays are review products, not a second world model.  Climate, biomes,
drainage basins and land potential therefore share the same elevation,
hydrology and four-season precipitation inputs as the physical map.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .model import WorldGrid
from .polar import polar_continent_mask
from .suitability import continuous_proximity, relative_land_slope, smooth_transition
from .wetlands import derive_wetland_support, WETLAND_SUPPORT_THRESHOLD
from .continuous_ecology import ContinuousEcologyField, FreshwaterCorridors


LAND_POTENTIAL_THRESHOLDS = (.2,.35,.5,.65,.8)
HABITABILITY_THRESHOLDS = (.2,.35,.5,.65,.8)


BIOME_ICE = 0
BIOME_TUNDRA_ALPINE = 1
BIOME_BOREAL_FOREST = 2
BIOME_TEMPERATE_MIXED_FOREST = 3
BIOME_TEMPERATE_GRASSLAND = 4
BIOME_MEDITERRANEAN_SCRUB = 5
BIOME_HUMID_SUBTROPICAL = 6
BIOME_TROPICAL_RAINFOREST = 7
BIOME_TROPICAL_SEASONAL_FOREST = 8
BIOME_SAVANNA_DRY_GRASSLAND = 9
BIOME_DESERT_SCRUB = 10
BIOME_POLAR_ICE_DESERT = 11
BIOME_POLAR_COASTAL_TUNDRA = 12
BIOME_COUNT = 13


# Köppen-Geiger uses water as an unclassified surface.  Land codes are kept
# contiguous so a renderer can build one deterministic vector partition per
# class without translating an ad-hoc set of strings at draw time.
KOPPEN_OCEAN = -1
KOPPEN_EF = 0
KOPPEN_ET = 1
KOPPEN_AF = 2
KOPPEN_AM = 3
KOPPEN_AW = 4
KOPPEN_AS = 5
KOPPEN_BWH = 6
KOPPEN_BWK = 7
KOPPEN_BSH = 8
KOPPEN_BSK = 9
KOPPEN_CSA = 10
KOPPEN_CSB = 11
KOPPEN_CSC = 12
KOPPEN_CWA = 13
KOPPEN_CWB = 14
KOPPEN_CWC = 15
KOPPEN_CFA = 16
KOPPEN_CFB = 17
KOPPEN_CFC = 18
KOPPEN_DSA = 19
KOPPEN_DSB = 20
KOPPEN_DSC = 21
KOPPEN_DSD = 22
KOPPEN_DWA = 23
KOPPEN_DWB = 24
KOPPEN_DWC = 25
KOPPEN_DWD = 26
KOPPEN_DFA = 27
KOPPEN_DFB = 28
KOPPEN_DFC = 29
KOPPEN_DFD = 30

# The physical hydrology model emits four seasonal states.  These weights
# place the equinox/solstice samples at March, June, September, and December,
# interpolate the two months between each sample, and preserve each cell's
# annual amount exactly.
_MONTHLY_SEASON_WEIGHTS = np.asarray(
    (
        (1.0 / 3.0, 0.0, 0.0, 2.0 / 3.0),
        (2.0 / 3.0, 0.0, 0.0, 1.0 / 3.0),
        (1.0, 0.0, 0.0, 0.0),
        (2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0),
        (1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0),
        (0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 2.0 / 3.0, 1.0 / 3.0),
        (0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float32,
)
# The seasonal hydroclimate is intentionally expressed in a compact relative
# range (most viable land is near 0.08 model units).  8,400 mm per unit maps
# that established wetness distribution to an Earth-like land mean while
# retaining its relative, deterministic spatial pattern.
_KOPPEN_ANNUAL_MM_PER_MODEL_UNIT = 8400.0
_KOPPEN_MONTHLY_MM_PER_MODEL_UNIT = _KOPPEN_ANNUAL_MM_PER_MODEL_UNIT / 12.0


@dataclass(frozen=True, slots=True)
class KoppenClass:
    """A stable display record for one estimated Köppen-Geiger class."""

    code: int
    symbol: str
    label: str
    color: str


KOPPEN_LEGEND = (
    KoppenClass(KOPPEN_EF, "EF", "冰原气候", "#f4f6f3"),
    KoppenClass(KOPPEN_ET, "ET", "苔原气候", "#dbe1da"),
    KoppenClass(KOPPEN_AF, "Af", "热带雨林气候", "#237a56"),
    KoppenClass(KOPPEN_AM, "Am", "热带季风气候", "#4c9b5f"),
    KoppenClass(KOPPEN_AW, "Aw", "热带干湿季草原气候", "#91a94e"),
    KoppenClass(KOPPEN_AS, "As", "热带夏干草原气候", "#b1af57"),
    KoppenClass(KOPPEN_BWH, "BWh", "热带沙漠气候", "#dfb866"),
    KoppenClass(KOPPEN_BWK, "BWk", "温带沙漠气候", "#d9c27b"),
    KoppenClass(KOPPEN_BSH, "BSh", "热带半干旱气候", "#c9a65e"),
    KoppenClass(KOPPEN_BSK, "BSk", "温带半干旱气候", "#b6ad75"),
    KoppenClass(KOPPEN_CSA, "Csa", "地中海夏热气候", "#d69c5c"),
    KoppenClass(KOPPEN_CSB, "Csb", "地中海暖夏气候", "#d7b36b"),
    KoppenClass(KOPPEN_CSC, "Csc", "地中海凉夏气候", "#c8bf85"),
    KoppenClass(KOPPEN_CWA, "Cwa", "温带冬干夏热气候", "#9bb560"),
    KoppenClass(KOPPEN_CWB, "Cwb", "温带冬干暖夏气候", "#9bbd78"),
    KoppenClass(KOPPEN_CWC, "Cwc", "温带冬干凉夏气候", "#a6bc91"),
    KoppenClass(KOPPEN_CFA, "Cfa", "温带湿润夏热气候", "#69af75"),
    KoppenClass(KOPPEN_CFB, "Cfb", "温带海洋性气候", "#82b994"),
    KoppenClass(KOPPEN_CFC, "Cfc", "温带海洋性凉夏气候", "#a7c5ad"),
    KoppenClass(KOPPEN_DSA, "Dsa", "寒带夏干夏热气候", "#a186a8"),
    KoppenClass(KOPPEN_DSB, "Dsb", "寒带夏干暖夏气候", "#9c9fbe"),
    KoppenClass(KOPPEN_DSC, "Dsc", "寒带夏干凉夏气候", "#90aac3"),
    KoppenClass(KOPPEN_DSD, "Dsd", "寒带夏干严寒气候", "#7d94b6"),
    KoppenClass(KOPPEN_DWA, "Dwa", "寒带冬干夏热气候", "#a77f9c"),
    KoppenClass(KOPPEN_DWB, "Dwb", "寒带冬干暖夏气候", "#8d9ebd"),
    KoppenClass(KOPPEN_DWC, "Dwc", "寒带冬干凉夏气候", "#7eabd0"),
    KoppenClass(KOPPEN_DWD, "Dwd", "寒带冬干严寒气候", "#6f8eb7"),
    KoppenClass(KOPPEN_DFA, "Dfa", "寒带湿润夏热气候", "#9d829f"),
    KoppenClass(KOPPEN_DFB, "Dfb", "寒带湿润暖夏气候", "#849fba"),
    KoppenClass(KOPPEN_DFC, "Dfc", "寒带湿润凉夏气候", "#78acc0"),
    KoppenClass(KOPPEN_DFD, "Dfd", "寒带湿润严寒气候", "#6d8aa8"),
)
KOPPEN_CLASS_COUNT = len(KOPPEN_LEGEND)
KOPPEN_BY_CODE: Mapping[int, KoppenClass] = MappingProxyType(
    {entry.code: entry for entry in KOPPEN_LEGEND}
)
KOPPEN_CODE_BY_SYMBOL: Mapping[str, int] = MappingProxyType(
    {entry.symbol: entry.code for entry in KOPPEN_LEGEND}
)


@dataclass(frozen=True, slots=True)
class MonthlyClimateEstimate:
    """Metric monthly inputs used by the estimated Köppen-Geiger classifier.

    This explicit product is useful for diagnostics, but the regular thematic
    pipeline deliberately streams the same twelve months and retains only
    summary rasters.  Full 12-month fields are therefore opt-in rather than a
    permanent cost of every generated world.
    """

    temperature_c: np.ndarray
    precipitation_mm: np.ndarray

    def __post_init__(self) -> None:
        temperature = np.asarray(self.temperature_c)
        precipitation = np.asarray(self.precipitation_mm)
        if (
            temperature.ndim != 3
            or temperature.shape[0] != 12
            or temperature.dtype != np.dtype(np.float32)
        ):
            raise ValueError("temperature_c must be a float32 array with twelve months")
        if precipitation.shape != temperature.shape or precipitation.dtype != np.dtype(
            np.float32
        ):
            raise ValueError(
                "precipitation_mm must match temperature_c and use float32 values"
            )
        if not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(precipitation)):
            raise ValueError("monthly climate estimates must be finite")
        if np.any(precipitation < 0.0):
            raise ValueError("monthly precipitation cannot be negative")
        object.__setattr__(self, "temperature_c", _immutable(temperature))
        object.__setattr__(self, "precipitation_mm", _immutable(precipitation))


@dataclass(frozen=True, slots=True)
class PhysiographicFeatures:
    """Cartographically useful physical masks derived from one WorldGrid."""

    desert: np.ndarray
    wetland: np.ndarray
    wetland_support: np.ndarray

    def __post_init__(self) -> None:
        shape = np.asarray(self.desert).shape
        for name in ("desert", "wetland"):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype != np.dtype(bool):
                raise ValueError(f"{name} must have shape {shape} and dtype bool")
            object.__setattr__(self, name, _immutable(value))
        support = np.asarray(self.wetland_support)
        if (support.shape != shape or support.dtype != np.dtype(np.float32)
                or np.any(~np.isfinite(support)) or np.any((support < 0) | (support > 1))):
            raise ValueError("wetland_support must be aligned finite float32 evidence within 0..1")
        if not np.array_equal(self.wetland, support >= WETLAND_SUPPORT_THRESHOLD):
            raise ValueError("wetland must use the shared wetland support threshold")
        object.__setattr__(self, "wetland_support", _immutable(support))


@dataclass(frozen=True, slots=True)
class ClimateDrivers:
    """Continuous climate drivers and an estimated Köppen-Geiger partition.

    ``temperature`` and the precipitation fields retain the existing
    normalized hydrology model so downstream carrying-capacity calculations
    continue to share one physical basis.  Metric values are a calibrated
    diagnostic interpretation used only for Köppen-Geiger thresholds; they
    are deliberately labelled *estimated* rather than presented as a second
    atmospheric simulation.
    """

    temperature: np.ndarray
    seasonal_precipitation: np.ndarray
    annual_precipitation: np.ndarray
    precipitation_range: np.ndarray
    mean_annual_temperature_c: np.ndarray
    annual_precipitation_mm: np.ndarray
    coldest_month_temperature_c: np.ndarray
    warmest_month_temperature_c: np.ndarray
    koppen_code: np.ndarray

    def __post_init__(self) -> None:
        shape = np.asarray(self.temperature).shape
        expected = {
            "temperature": (shape, np.dtype(np.float64)),
            "seasonal_precipitation": ((4, *shape), np.dtype(np.float64)),
            "annual_precipitation": (shape, np.dtype(np.float64)),
            "precipitation_range": (shape, np.dtype(np.float64)),
            "mean_annual_temperature_c": (shape, np.dtype(np.float32)),
            "annual_precipitation_mm": (shape, np.dtype(np.float32)),
            "coldest_month_temperature_c": (shape, np.dtype(np.float32)),
            "warmest_month_temperature_c": (shape, np.dtype(np.float32)),
            "koppen_code": (shape, np.dtype(np.int8)),
        }
        for name, (expected_shape, expected_dtype) in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != expected_shape or value.dtype != expected_dtype:
                raise ValueError(
                    f"{name} must have shape {expected_shape} and dtype {expected_dtype}"
                )
            object.__setattr__(self, name, _immutable(value))

        metric_fields = (
            self.mean_annual_temperature_c,
            self.annual_precipitation_mm,
            self.coldest_month_temperature_c,
            self.warmest_month_temperature_c,
        )
        if any(not np.all(np.isfinite(value)) for value in metric_fields):
            raise ValueError("metric climate summaries must be finite")
        if np.any(self.annual_precipitation_mm < 0.0):
            raise ValueError("annual_precipitation_mm cannot be negative")
        if np.any(
            self.coldest_month_temperature_c > self.warmest_month_temperature_c
        ):
            raise ValueError("coldest_month_temperature_c cannot exceed warmest month")
        if np.any(
            (self.koppen_code < KOPPEN_OCEAN)
            | (self.koppen_code >= KOPPEN_CLASS_COUNT)
        ):
            raise ValueError("koppen_code contains an unknown formal class code")


@dataclass(frozen=True, slots=True)
class ThematicLayers:
    """Shared physical themes, agriculture capacity and residential suitability."""

    climate: ClimateDrivers
    biome_zone: np.ndarray
    drainage_basin: np.ndarray
    major_basin_rank: np.ndarray
    land_potential: np.ndarray
    land_potential_band: np.ndarray
    habitability: np.ndarray
    habitability_band: np.ndarray
    physiography: PhysiographicFeatures

    def __post_init__(self) -> None:
        shape = self.climate.temperature.shape
        expected = {
            "biome_zone": np.dtype(np.int8),
            "drainage_basin": np.dtype(np.int32),
            "major_basin_rank": np.dtype(np.int16),
            "land_potential": np.dtype(np.float32),
            "land_potential_band": np.dtype(np.uint8),
            "habitability": np.dtype(np.float32),
            "habitability_band": np.dtype(np.uint8),
        }
        for name, dtype in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
            object.__setattr__(self, name, _immutable(value))
        for name in ("land_potential","habitability"):
            value=getattr(self,name)
            if np.any(~np.isfinite(value)) or np.any((value<0)|(value>1)):
                raise ValueError(f"{name} must contain finite capacity scores in [0, 1]")
        if not isinstance(self.physiography, PhysiographicFeatures):
            raise TypeError("physiography must be a PhysiographicFeatures instance")
        if np.asarray(self.physiography.desert).shape != shape:
            raise ValueError("physiography masks must match climate dimensions")


def _immutable(value: np.ndarray) -> np.ndarray:
    if value.flags.c_contiguous and not value.flags.writeable:
        return value
    result = np.ascontiguousarray(value).copy()
    result.setflags(write=False)
    return result


def smooth_field(field: np.ndarray, *, radius: int, passes: int = 2) -> np.ndarray:
    """Box-smooth a world field with wrapped longitude and clamped poles."""

    result = np.asarray(field, dtype=np.float64)
    radius = int(radius)
    if radius <= 0 or passes <= 0:
        return result.copy()
    kernel_width = radius * 2 + 1
    height = result.shape[0]
    for _ in range(passes):
        wrapped = np.concatenate(
            (result[:, -radius:], result, result[:, :radius]),
            axis=1,
        )
        horizontal_sum = np.pad(
            np.cumsum(wrapped, axis=1, dtype=np.float64),
            ((0, 0), (1, 0)),
        )
        horizontal = (
            horizontal_sum[:, kernel_width:] - horizontal_sum[:, :-kernel_width]
        ) / kernel_width
        padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
        vertical_sum = np.pad(
            np.cumsum(padded, axis=0, dtype=np.float64),
            ((1, 0), (0, 0)),
        )
        result = (
            vertical_sum[kernel_width:] - vertical_sum[:-kernel_width]
        ) / kernel_width
        if result.shape[0] != height:
            raise RuntimeError("smoothed climate field changed shape")
    return result


def regularize_partition(
    zones: np.ndarray,
    *,
    category_count: int,
    radius: int,
    protected_categories: tuple[int, ...] = (-1, 0),
) -> np.ndarray:
    """Remove categorical slivers without changing protected water/ice cells."""

    values = np.asarray(zones)
    if radius <= 0:
        return values.copy()
    best_zone = values.copy()
    best_support = np.full(values.shape, -1.0, dtype=np.float64)
    original_support = np.zeros(values.shape, dtype=np.float64)
    protected = np.isin(values, protected_categories)
    for zone in range(category_count):
        if zone in protected_categories:
            continue
        support = smooth_field(values == zone, radius=radius, passes=1)
        original_support[values == zone] = support[values == zone]
        stronger = support > best_support
        best_zone[stronger] = zone
        best_support[stronger] = support[stronger]
    cleaned = values.copy()
    replace = ~protected & (best_support > original_support + 1.0e-12)
    cleaned[replace] = best_zone[replace]
    return cleaned


def _latitude_field(grid: WorldGrid) -> np.ndarray:
    """Return one EIR-GEOG-1 latitude value per raster row."""

    height, _width = grid.shape
    extents = grid.metadata["extents"]
    north = float(extents["north"])
    south = float(extents["south"])
    return north - (np.arange(height, dtype=np.float64) + 0.5) * (
        (north - south) / height
    )


def _derive_continuous_climate_fields(
    grid: WorldGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep the established normalized climate drivers in one shared helper."""

    latitude_field = _latitude_field(grid)[:, None]
    climate_scale = min(grid.shape)
    relief_radius = min(12, climate_scale // 36)
    rain_radius = min(16, climate_scale // 28)
    temperature = np.clip(
        np.cos(np.radians(np.abs(latitude_field))) ** 1.25
        - 0.48 * smooth_field(grid.elevation, radius=relief_radius),
        0.0,
        1.0,
    )
    seasonal_rain = np.stack(
        [
            smooth_field(season.astype(np.float64) / 255.0, radius=rain_radius)
            for season in grid.seasonal_precipitation
        ]
    )
    annual_rain = seasonal_rain.mean(axis=0)
    rain_range = seasonal_rain.max(axis=0) - seasonal_rain.min(axis=0)
    return temperature, seasonal_rain, annual_rain, rain_range


def _metric_temperature_parameters(
    grid: WorldGrid,
    normalized_temperature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate mean temperature and annual range from the physical grid.

    The canonical grid intentionally uses relative elevation and normalized
    hydrology.  This calibration gives the Köppen thresholds a stable metric
    interpretation while preserving those canonical continuous fields for all
    downstream world calculations.  A maritime smoothing term moderates the
    seasonal range near ocean margins; it is not a second climate simulation.
    """

    latitude = _latitude_field(grid)[:, None]
    absolute_latitude = np.abs(latitude)
    ocean = (grid.water == 1) | (grid.water == 3)
    maritime_radius = max(2, min(64, min(grid.shape) // 18))
    maritime = np.clip(
        smooth_field(ocean, radius=maritime_radius, passes=1), 0.0, 1.0
    )
    continentality = 1.0 - maritime

    # ``temperature`` already encodes latitude and relief.  Mapping [0, 1]
    # to [-18, 32] °C gives ordinary lowland tropical / temperate / polar
    # thresholds without inventing a metre-scale elevation model.
    mean_temperature = (-18.0 + 50.0 * normalized_temperature).astype(
        np.float32
    )
    latitude_seasonality = np.power(
        np.sin(np.radians(absolute_latitude)),
        0.82,
    )
    # Insolation's annual phase disappears at zero axial tilt.  The relative
    # temperature field has no separate orbital solver, so scale only its
    # calendar excursion by the physically monotone sine of the configured
    # tilt; this is a calibrated Köppen input, not a full atmosphere model.
    # The Earth-reference denominator preserves the established 23.44° range
    # while still allowing higher-obliquity worlds to express larger seasons.
    tilt = float(grid.metadata.get("planet", {}).get("axialTiltDegrees", 23.44))
    if not np.isfinite(tilt) or not 0.0 <= tilt <= 90.0:
        raise ValueError("planet axialTiltDegrees must be finite and within [0, 90]")
    earth_reference = np.sin(np.radians(23.44))
    tilt_forcing = np.sin(np.radians(tilt)) / max(earth_reference, 1.0e-12)
    annual_range = (
        tilt_forcing
        * (1.5 + 19.0 * latitude_seasonality * (0.35 + 0.65 * continentality))
    ).astype(np.float32)
    return mean_temperature, annual_range, latitude


def _monthly_temperature(
    mean_temperature: np.ndarray,
    annual_range: np.ndarray,
    latitude: np.ndarray,
    snow_land: np.ndarray,
    month_index: int,
) -> np.ndarray:
    """Return one synthetic calendar month from the four physical seasons."""

    phase = np.sin(2.0 * np.pi * (float(month_index) - 2.0) / 12.0)
    hemisphere = np.where(latitude >= 0.0, 1.0, -1.0)
    month = np.asarray(
        mean_temperature + annual_range * phase * hemisphere,
        dtype=np.float32,
    )
    # The base grid's snow field is the canonical perennial-snow decision.
    # Cap its warmest synthetic month so named ice sheets and high peaks do
    # not receive an implausible warm-temperate Köppen label.  The cap itself
    # falls through the high polar latitudes, allowing polar ice caps to be
    # EF while isolated alpine snow remains ET.
    snow_cap = 5.0 - 0.45 * np.maximum(np.abs(latitude) - 60.0, 0.0)
    np.minimum(month, snow_cap, out=month, where=snow_land)
    return month


def _monthly_precipitation(
    seasonal_rain: np.ndarray,
    month_index: int,
) -> np.ndarray:
    """Interpolate one monthly total from four physical precipitation states."""

    weights = _MONTHLY_SEASON_WEIGHTS[month_index]
    month = (
        weights[0] * seasonal_rain[0]
        + weights[1] * seasonal_rain[1]
        + weights[2] * seasonal_rain[2]
        + weights[3] * seasonal_rain[3]
    )
    return np.asarray(
        month * _KOPPEN_MONTHLY_MM_PER_MODEL_UNIT,
        dtype=np.float32,
    )


@dataclass(frozen=True, slots=True)
class _KoppenMonthlyStatistics:
    """Only the monthly extrema and seasonal aggregates Köppen requires."""

    coldest_month_c: np.ndarray
    warmest_month_c: np.ndarray
    months_above_10c: np.ndarray
    annual_precipitation_mm: np.ndarray
    driest_month_mm: np.ndarray
    summer_precipitation_mm: np.ndarray
    summer_driest_month_mm: np.ndarray
    summer_wettest_month_mm: np.ndarray
    winter_driest_month_mm: np.ndarray
    winter_wettest_month_mm: np.ndarray


def _coerce_latitude(
    latitude_degrees: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Accept a row field or cell field and return a read-only cell field."""

    latitude = np.asarray(latitude_degrees, dtype=np.float32)
    if latitude.shape == (shape[0],):
        latitude = latitude[:, None]
    try:
        return np.broadcast_to(latitude, shape)
    except ValueError as error:
        raise ValueError(
            "latitude_degrees must have one value per row or one value per cell"
        ) from error


def _validate_monthly_climate(
    temperature_c: np.ndarray,
    precipitation_mm: np.ndarray,
    land_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperature = np.asarray(temperature_c, dtype=np.float32)
    precipitation = np.asarray(precipitation_mm, dtype=np.float32)
    land = np.asarray(land_mask, dtype=bool)
    if temperature.ndim != 3 or temperature.shape[0] != 12:
        raise ValueError("temperature_c must provide exactly twelve monthly rasters")
    if precipitation.shape != temperature.shape:
        raise ValueError("precipitation_mm must match temperature_c")
    if land.shape != temperature.shape[1:]:
        raise ValueError("land_mask must match one monthly climate raster")
    invalid_temperature = ~np.isfinite(temperature)
    if np.any(invalid_temperature & land[None, :, :]):
        raise ValueError("land monthly temperatures must be finite")
    invalid_precipitation = (~np.isfinite(precipitation)) | (precipitation < 0.0)
    if np.any(invalid_precipitation & land[None, :, :]):
        raise ValueError("land monthly precipitation must be finite and non-negative")
    return temperature, precipitation, land


def _summarize_monthly_climate(
    monthly_values: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    latitude_degrees: np.ndarray,
    shape: tuple[int, int],
) -> _KoppenMonthlyStatistics:
    """Stream twelve months into the bounded statistics used by Köppen."""

    latitude = _coerce_latitude(latitude_degrees, shape)
    northern = latitude >= 0.0
    coldest = np.full(shape, np.inf, dtype=np.float32)
    warmest = np.full(shape, -np.inf, dtype=np.float32)
    warm_months = np.zeros(shape, dtype=np.uint8)
    annual = np.zeros(shape, dtype=np.float32)
    driest = np.full(shape, np.inf, dtype=np.float32)
    summer_total = np.zeros(shape, dtype=np.float32)
    summer_driest = np.full(shape, np.inf, dtype=np.float32)
    summer_wettest = np.full(shape, -np.inf, dtype=np.float32)
    winter_driest = np.full(shape, np.inf, dtype=np.float32)
    winter_wettest = np.full(shape, -np.inf, dtype=np.float32)

    seen = 0
    for month_index, (temperature, precipitation) in enumerate(monthly_values):
        if month_index >= 12:
            raise ValueError("monthly climate requires exactly twelve months")
        temperature_array = np.asarray(temperature, dtype=np.float32)
        precipitation_array = np.asarray(precipitation, dtype=np.float32)
        if temperature_array.shape != shape or precipitation_array.shape != shape:
            raise ValueError("monthly climate raster shape changed during iteration")
        np.minimum(coldest, temperature_array, out=coldest)
        np.maximum(warmest, temperature_array, out=warmest)
        warm_months += (temperature_array > 10.0).astype(np.uint8)
        annual += precipitation_array
        np.minimum(driest, precipitation_array, out=driest)

        # Köppen's summer/winter predicates use Apr–Sep in the north and the
        # complementary Oct–Mar period in the south.
        north_summer_month = 3 <= month_index <= 8
        summer = northern if north_summer_month else ~northern
        winter = ~summer
        np.add(summer_total, precipitation_array, out=summer_total, where=summer)
        np.minimum(
            summer_driest,
            precipitation_array,
            out=summer_driest,
            where=summer,
        )
        np.maximum(
            summer_wettest,
            precipitation_array,
            out=summer_wettest,
            where=summer,
        )
        np.minimum(
            winter_driest,
            precipitation_array,
            out=winter_driest,
            where=winter,
        )
        np.maximum(
            winter_wettest,
            precipitation_array,
            out=winter_wettest,
            where=winter,
        )
        seen += 1
    if seen != 12:
        raise ValueError("monthly climate requires exactly twelve months")
    return _KoppenMonthlyStatistics(
        coldest_month_c=coldest,
        warmest_month_c=warmest,
        months_above_10c=warm_months,
        annual_precipitation_mm=annual,
        driest_month_mm=driest,
        summer_precipitation_mm=summer_total,
        summer_driest_month_mm=summer_driest,
        summer_wettest_month_mm=summer_wettest,
        winter_driest_month_mm=winter_driest,
        winter_wettest_month_mm=winter_wettest,
    )


def _classify_koppen_statistics(
    statistics: _KoppenMonthlyStatistics,
    *,
    mean_annual_temperature_c: np.ndarray,
    land_mask: np.ndarray,
) -> np.ndarray:
    """Apply standard Köppen-Geiger threshold rules to monthly statistics."""

    land = np.asarray(land_mask, dtype=bool)
    mean_temperature = np.asarray(mean_annual_temperature_c, dtype=np.float32)
    if mean_temperature.shape != land.shape:
        raise ValueError("mean_annual_temperature_c must match land_mask")
    codes = np.full(land.shape, KOPPEN_OCEAN, dtype=np.int8)
    coldest = statistics.coldest_month_c
    warmest = statistics.warmest_month_c
    annual = statistics.annual_precipitation_mm

    polar = land & (warmest < 10.0)
    codes[polar & (warmest < 0.0)] = KOPPEN_EF
    codes[polar & (warmest >= 0.0)] = KOPPEN_ET

    growing = land & ~polar
    season = np.full(land.shape, 2, dtype=np.int8)  # f by default
    summer_dry = (statistics.summer_driest_month_mm < 40.0) & (
        statistics.summer_driest_month_mm
        < statistics.winter_wettest_month_mm / 3.0
    )
    winter_dry = statistics.winter_driest_month_mm < (
        statistics.summer_wettest_month_mm / 10.0
    )
    # Beck/Peel resolve cells satisfying both dry-season predicates by their
    # half-year totals; a single dry summer month cannot override a dry winter.
    both_dry = summer_dry & winter_dry
    winter_wetter = statistics.summer_precipitation_mm < annual * 0.5
    season[summer_dry & (~both_dry | winter_wetter)] = 0
    season[winter_dry & (~both_dry | ~winter_wetter)] = 1

    third = np.full(land.shape, 2, dtype=np.int8)  # cool summer c
    third[warmest >= 22.0] = 0  # a
    third[(warmest < 22.0) & (statistics.months_above_10c >= 4)] = 1  # b

    temperate = growing & (coldest > 0.0) & (coldest < 18.0)
    temperate_code = (KOPPEN_CSA + season * 3 + np.minimum(third, 2)).astype(
        np.int8
    )
    codes[temperate] = temperate_code[temperate]

    continental = growing & (coldest <= 0.0)
    continental_third = third.copy()
    continental_third[(continental) & (coldest <= -38.0) & (third == 2)] = 3
    continental_code = (
        KOPPEN_DSA + season * 4 + continental_third
    ).astype(np.int8)
    codes[continental] = continental_code[continental]

    tropical = growing & (coldest >= 18.0)
    tropical_code = np.full(land.shape, KOPPEN_AW, dtype=np.int8)
    tropical_code[statistics.driest_month_mm >= 60.0] = KOPPEN_AF
    monsoon_threshold = 100.0 - annual / 25.0
    monsoon = (statistics.driest_month_mm < 60.0) & (
        statistics.driest_month_mm >= monsoon_threshold
    )
    tropical_code[monsoon] = KOPPEN_AM
    # A dry summer rather than a dry winter gives the uncommon As subtype.
    tropical_code[
        (~monsoon)
        & (statistics.driest_month_mm < 60.0)
        & (statistics.summer_driest_month_mm < statistics.winter_driest_month_mm)
    ] = KOPPEN_AS
    codes[tropical] = tropical_code[tropical]

    summer_share = np.divide(
        statistics.summer_precipitation_mm,
        annual,
        out=np.zeros_like(annual),
        where=annual > 0.0,
    )
    arid_threshold = 20.0 * mean_temperature + 140.0
    arid_threshold = np.where(summer_share >= 0.70, arid_threshold + 140.0, arid_threshold)
    arid_threshold = np.where(summer_share <= 0.30, arid_threshold - 140.0, arid_threshold)
    arid_threshold = np.maximum(arid_threshold, 0.0)
    # Aridity takes precedence over A/C/D/E, including cold dry highlands.
    arid = land & (annual < arid_threshold)
    desert = arid & (annual < arid_threshold * 0.50)
    hot = mean_temperature >= 18.0
    codes[arid & desert & hot] = KOPPEN_BWH
    codes[arid & desert & ~hot] = KOPPEN_BWK
    codes[arid & ~desert & hot] = KOPPEN_BSH
    codes[arid & ~desert & ~hot] = KOPPEN_BSK

    if np.any(land & (codes == KOPPEN_OCEAN)):
        raise ValueError("Köppen rules did not classify every land cell")
    return _immutable(codes)


def classify_koppen_geiger(
    temperature_c: np.ndarray,
    precipitation_mm: np.ndarray,
    *,
    latitude_degrees: np.ndarray,
    land_mask: np.ndarray,
) -> np.ndarray:
    """Classify twelve metric monthly fields with the Köppen-Geiger rules.

    This public function is deterministic and independent of the generator,
    which makes reference fixtures and future external climate solvers easy to
    verify against the same map legend.
    """

    temperature, precipitation, land = _validate_monthly_climate(
        temperature_c,
        precipitation_mm,
        land_mask,
    )
    latitude = _coerce_latitude(latitude_degrees, land.shape)
    statistics = _summarize_monthly_climate(
        tuple((temperature[index], precipitation[index]) for index in range(12)),
        latitude_degrees=latitude,
        shape=land.shape,
    )
    annual_mean_temperature = temperature.mean(axis=0, dtype=np.float32)
    return _classify_koppen_statistics(
        statistics,
        mean_annual_temperature_c=annual_mean_temperature,
        land_mask=land,
    )


def derive_monthly_climate_estimate(grid: WorldGrid) -> MonthlyClimateEstimate:
    """Materialize a diagnostic twelve-month metric climate estimate.

    Ordinary world generation should use :func:`derive_climate_drivers`;
    that path streams these same values and retains only the compact metrics
    required by the map and simulations.
    """

    normalized_temperature, seasonal_rain, _annual, _range = (
        _derive_continuous_climate_fields(grid)
    )
    mean_temperature, annual_range, latitude = _metric_temperature_parameters(
        grid,
        normalized_temperature,
    )
    shape = grid.shape
    snow_land = grid.snow & (grid.water == 0)
    monthly_temperature = np.empty((12, *shape), dtype=np.float32)
    monthly_precipitation = np.empty((12, *shape), dtype=np.float32)
    for month_index in range(12):
        monthly_temperature[month_index] = _monthly_temperature(
            mean_temperature,
            annual_range,
            latitude,
            snow_land,
            month_index,
        )
        monthly_precipitation[month_index] = _monthly_precipitation(
            seasonal_rain,
            month_index,
        )
    return MonthlyClimateEstimate(
        temperature_c=monthly_temperature,
        precipitation_mm=monthly_precipitation,
    )


def derive_climate_drivers(grid: WorldGrid) -> ClimateDrivers:
    """Derive continuous climate fields plus a streamed Köppen-Geiger estimate."""

    temperature, seasonal_rain, annual_rain, rain_range = (
        _derive_continuous_climate_fields(grid)
    )
    mean_temperature, annual_range, latitude = _metric_temperature_parameters(
        grid,
        temperature,
    )
    snow_land = grid.snow & (grid.water == 0)
    statistics = _summarize_monthly_climate(
        (
            (
                _monthly_temperature(
                    mean_temperature,
                    annual_range,
                    latitude,
                    snow_land,
                    month_index,
                ),
                _monthly_precipitation(seasonal_rain, month_index),
            )
            for month_index in range(12)
        ),
        latitude_degrees=latitude,
        shape=grid.shape,
    )
    koppen = _classify_koppen_statistics(
        statistics,
        mean_annual_temperature_c=mean_temperature,
        land_mask=grid.water == 0,
    )
    return ClimateDrivers(
        temperature=temperature.astype(np.float64),
        seasonal_precipitation=seasonal_rain.astype(np.float64),
        annual_precipitation=annual_rain.astype(np.float64),
        precipitation_range=rain_range.astype(np.float64),
        mean_annual_temperature_c=mean_temperature.astype(np.float32),
        annual_precipitation_mm=statistics.annual_precipitation_mm.astype(np.float32),
        coldest_month_temperature_c=statistics.coldest_month_c.astype(np.float32),
        warmest_month_temperature_c=statistics.warmest_month_c.astype(np.float32),
        koppen_code=koppen,
    )


def derive_biome_zones(grid: WorldGrid, climate: ClimateDrivers) -> np.ndarray:
    """Classify adjacent, non-overlapping vegetation regions from climate."""

    temperature = climate.temperature
    annual = climate.annual_precipitation
    seasonal_min = climate.seasonal_precipitation.min(axis=0)
    zones = np.full(grid.shape, BIOME_TEMPERATE_MIXED_FOREST, dtype=np.int8)

    zones[temperature < 0.38] = BIOME_BOREAL_FOREST
    zones[temperature < 0.22] = BIOME_TUNDRA_ALPINE
    zones[(temperature >= 0.38) & (annual < 0.085)] = BIOME_TEMPERATE_GRASSLAND
    zones[(temperature >= 0.62) & (annual >= 0.10)] = BIOME_HUMID_SUBTROPICAL
    zones[temperature >= 0.76] = BIOME_TROPICAL_SEASONAL_FOREST
    zones[(temperature >= 0.76) & (annual < 0.10)] = BIOME_SAVANNA_DRY_GRASSLAND
    zones[
        (temperature >= 0.76)
        & (annual >= 0.16)
        & (seasonal_min >= 0.055)
    ] = BIOME_TROPICAL_RAINFOREST
    zones[(temperature >= 0.64) & (annual < 0.05)] = BIOME_SAVANNA_DRY_GRASSLAND
    zones[
        (temperature >= 0.22) & (temperature < 0.64) & (annual < 0.055)
    ] = BIOME_TEMPERATE_GRASSLAND
    zones[annual < 0.032] = BIOME_DESERT_SCRUB

    height = grid.shape[0]
    extents = grid.metadata["extents"]
    north = float(extents["north"])
    south = float(extents["south"])
    latitude = north - (np.arange(height, dtype=np.float64) + 0.5) * (
        (north - south) / height
    )
    northern = latitude[:, None] >= 0.0
    summer_rain = np.where(
        northern,
        climate.seasonal_precipitation[1],
        climate.seasonal_precipitation[3],
    )
    winter_rain = np.where(
        northern,
        climate.seasonal_precipitation[3],
        climate.seasonal_precipitation[1],
    )
    mediterranean = (
        (temperature >= 0.45)
        & (temperature < 0.76)
        & (annual >= 0.065)
        & (summer_rain + 0.005 < winter_rain * 0.90)
    )
    zones[mediterranean] = BIOME_MEDITERRANEAN_SCRUB
    zones[grid.snow & (grid.water == 0)] = BIOME_ICE
    zones[grid.water != 0] = -1
    cleaned = regularize_partition(
        zones,
        category_count=BIOME_COUNT,
        radius=min(5, min(grid.shape) // 180),
    )
    polar_land = polar_continent_mask(grid)
    if bool(polar_land.any()):
        ocean_nearby = grid.water == 1
        for _ in range(max(2, min(8, min(grid.shape) // 120))):
            vertical = np.pad(ocean_nearby, ((1, 1), (0, 0)), mode="constant")
            ocean_nearby = (
                ocean_nearby
                | vertical[:-2]
                | vertical[2:]
                | np.roll(ocean_nearby, 1, axis=1)
                | np.roll(ocean_nearby, -1, axis=1)
            )
        coastal_tundra = (
            polar_land
            & ocean_nearby
            & ~grid.snow
            & (temperature >= 0.08)
            & (annual >= 0.035)
            & (grid.elevation <= 0.30)
        )
        cleaned[polar_land] = BIOME_POLAR_ICE_DESERT
        cleaned[coastal_tundra] = BIOME_POLAR_COASTAL_TUNDRA
    return _immutable(cleaned.astype(np.int8))


def derive_drainage_basins(grid: WorldGrid) -> np.ndarray:
    """Assign every land cell to its terminal D8 outlet using pointer jumping."""

    cell_count = int(np.prod(grid.shape, dtype=np.int64))
    indices = np.arange(cell_count, dtype=np.int64)
    flow = grid.flow_to.reshape(-1).astype(np.int64, copy=False)
    water = grid.water.reshape(-1) != 0
    parent = indices.copy()
    valid = (flow >= 0) & ~water
    parent[valid] = flow[valid]
    parent[water] = indices[water]

    maximum_steps = max(2, int(np.ceil(np.log2(max(2, cell_count)))) + 2)
    for _ in range(maximum_steps):
        next_parent = parent[parent]
        if np.array_equal(next_parent, parent):
            parent = next_parent
            break
        parent = next_parent

    # A functional-graph cycle does not converge to one shared terminal under
    # pointer jumping.  Validate the original edge relation after compression.
    if np.any(valid & (parent != parent[np.maximum(flow, 0)])):
        raise ValueError("flow_to contains a drainage cycle")

    basins = parent.astype(np.int32)
    basins[water] = -1
    return _immutable(basins.reshape(grid.shape))


def rank_major_basins(
    grid: WorldGrid,
    basins: np.ndarray,
    *,
    maximum: int = 24,
) -> np.ndarray:
    """Rank the largest outlet basins; zero denotes all minor basins."""

    if isinstance(maximum, bool) or int(maximum) < 1:
        raise ValueError("maximum must be a positive integer")
    basin_values = np.asarray(basins, dtype=np.int32)
    if basin_values.shape != grid.shape:
        raise ValueError("basins must match the WorldGrid shape")
    land_values = basin_values[grid.water == 0]
    identifiers, counts = np.unique(land_values[land_values >= 0], return_counts=True)
    order = np.lexsort((identifiers, -counts))[: int(maximum)]
    ranked = np.zeros(grid.shape, dtype=np.int16)
    ranked[grid.water != 0] = -1
    for rank, index in enumerate(order, start=1):
        ranked[basin_values == identifiers[index]] = rank
    return _immutable(ranked)


def _relief_score(elevation: np.ndarray, land: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(elevation, dtype=np.float64)
    gradient = relative_land_slope(values, land)
    score = (1.0 - np.clip(gradient / 0.08, 0.0, 1.0)) * (
        1.0 - np.clip((values - 0.62) / 0.30, 0.0, 1.0)
    )
    return score, gradient


def derive_physiographic_features(
    grid: WorldGrid,
    climate: ClimateDrivers,
) -> PhysiographicFeatures:
    """Derive compact masks for physical symbols and region naming.

    The masks deliberately describe candidates rather than inventing named
    places.  The label layer can select spatially separated wetland groups
    and desert regions while every candidate remains traceable to the
    same elevation, hydrography, and Köppen field used elsewhere.
    """

    land = grid.water == 0
    empty = np.zeros(grid.shape, dtype=bool)
    if not bool(land.any()):
        return PhysiographicFeatures(
            desert=empty,
            wetland=empty,
            wetland_support=np.zeros(grid.shape, dtype=np.float32),
        )
    desert = land & np.isin(
        climate.koppen_code,
        (KOPPEN_BWH, KOPPEN_BWK),
    )

    wetland_support = derive_wetland_support(grid, climate)
    return PhysiographicFeatures(
        desert=desert.astype(bool),
        wetland=wetland_support >= WETLAND_SUPPORT_THRESHOLD,
        wetland_support=wetland_support,
    )


def derive_land_potential_field(
    grid: WorldGrid,
    climate: ClimateDrivers,
    *, ecological_sources:FreshwaterCorridors,
) -> ContinuousEcologyField:
    """Score agricultural capacity before the population model is applied.

    Climate and water determine the resource ceiling.  Terrain is a carrying-
    capacity constraint rather than another interchangeable bonus: steep or
    high ground remains marginal even when it is wet, while flat lower river
    valleys and lake shores receive a modest alluvial-access advantage.
    """

    heat = smooth_transition(climate.temperature,.10,.46)
    heat *= 1.0-.35*smooth_transition(climate.temperature,.88,1.0)
    reliability = 1.0 - np.clip(climate.precipitation_range / 0.45, 0.0, 1.0)
    _relief, gradient = _relief_score(grid.elevation, grid.water == 0)
    scale = min(grid.shape)
    freshwater_radius = max(4, min(16, scale // 64))
    land = grid.water == 0
    # Rain-fed and irrigated opportunity are distinct water supplies. A dry
    # climate cannot erase a supplied valley at a single rainfall threshold.
    rainfed = -np.expm1(-np.maximum(climate.annual_precipitation,0.0)/.08)
    rainfed *= .65+.35*reliability
    wet_excess = 1.0-.45*smooth_transition(climate.annual_precipitation,.26,.50)

    # Relative slope breakpoints mirror the ordering of established land-
    # evaluation classes without pretending that this normalized elevation
    # grid is a metre-based DEM.
    slope_capacity = np.interp(
        gradient,
        (0.0, 0.018, 0.035, 0.060, 0.095, 0.14),
        (1.0, 0.98, 0.84, 0.58, 0.28, 0.08),
    )
    highland = np.clip((grid.elevation - 0.42) / 0.43, 0.0, 1.0)
    elevation_capacity = 1.0 - 0.72 * np.power(highland, 1.25)
    terrain_capacity = slope_capacity * elevation_capacity

    lower_land = np.clip((0.66 - grid.elevation) / 0.28, 0.0, 1.0)
    flat_land = np.clip((slope_capacity - 0.25) / 0.75, 0.0, 1.0)
    ceiling = heat*wet_excess*(.15+.85*terrain_capacity)
    ceiling += .10*heat*flat_land*lower_land*(1.0-ceiling)
    ceiling *= .65+.35*reliability
    background = heat*rainfed*wet_excess*(.15+.85*terrain_capacity)
    background = np.minimum(background,ceiling)
    extents = grid.metadata["extents"]
    north = float(extents["north"])
    south = float(extents["south"])
    latitude = north - (np.arange(grid.shape[0], dtype=np.float64) + 0.5) * (
        north - south
    ) / grid.shape[0]
    tilt = float(grid.metadata.get("planet", {}).get("axialTiltDegrees", 23.44))
    polar_circle = 90.0 - float(np.clip(tilt, 0.0, 45.0))
    # Agricultural seasons shorten well before the polar circle.  This is a
    # capacity ceiling, not snow camouflage: low relief by a southern coast
    # cannot remain highly productive merely because it is ice-free.
    subpolar_start = min(52.0, polar_circle - 8.0)
    polar_growing_season = np.clip(
        (polar_circle - np.abs(latitude)) / max(polar_circle - subpolar_start, 1.0),
        0.0,
        1.0,
    )[:, None]
    seasonal = 0.10 + 0.90 * np.power(polar_growing_season, 1.35)
    background *= seasonal;ceiling *= seasonal
    inactive = grid.snow | ~land | polar_continent_mask(grid)
    background[inactive]=0.0;ceiling[inactive]=0.0
    sources=ecological_sources
    return ContinuousEcologyField(np.clip(background,0.,1.),np.clip(ceiling,0.,1.),land,sources,
        supply_capacity=.72,river_radii=[min(freshwater_radius,2.+1.25*(order-1)) for order in sources.river_orders],
        lake_radius=max(2,freshwater_radius//2))


def derive_land_potential(grid:WorldGrid,climate:ClimateDrivers,*,ecological_sources:FreshwaterCorridors)->tuple[np.ndarray,np.ndarray]:
    potential = derive_land_potential_field(grid,climate,ecological_sources=ecological_sources).native
    bands = np.digitize(potential, LAND_POTENTIAL_THRESHOLDS).astype(np.uint8)
    bands[grid.water != 0] = 0
    return _immutable(potential), _immutable(bands)


def derive_habitability_field(grid:WorldGrid,climate:ClimateDrivers,*,ecological_sources:FreshwaterCorridors)->ContinuousEcologyField:
    """Residential comfort and reliable drinking water, separate from farming."""
    scale=min(grid.shape)
    radius=max(4,min(12,scale//96))
    maritime=continuous_proximity((grid.water==1)|(grid.water==3),max(2,radius//2))
    reliability=1.0-np.clip(climate.precipitation_range/.45,0.0,1.0)
    rain_access=-np.expm1(-np.maximum(climate.annual_precipitation,0.0)/.035)
    rain_access*=.70+.30*reliability
    thermal=smooth_transition(climate.temperature,.10,.42)
    thermal*=1.0-.35*smooth_transition(climate.temperature,.86,1.0)
    _,gradient=_relief_score(grid.elevation, grid.water == 0)
    terrain=(1.0-.70*smooth_transition(gradient,.02,.14))
    terrain*=1.0-.55*smooth_transition(grid.elevation,.42,.90)
    ceiling=thermal*(.25+.75*terrain)*(.94+.06*maritime)
    background=ceiling*(.15+.85*rain_access)
    allowed=(grid.water==0)&~grid.snow&~polar_continent_mask(grid)
    background[~allowed]=0.0;ceiling[~allowed]=0.0
    sources=ecological_sources
    return ContinuousEcologyField(np.clip(background,0.,1.),np.clip(ceiling,0.,1.),grid.water==0,sources,
        supply_capacity=.90,river_radii=[min(radius,2.+.75*(order-1)) for order in sources.river_orders],
        lake_radius=max(2,radius//2))


def derive_habitability(grid:WorldGrid,climate:ClimateDrivers,*,ecological_sources:FreshwaterCorridors)->tuple[np.ndarray,np.ndarray]:
    value=derive_habitability_field(grid,climate,ecological_sources=ecological_sources).native
    bands=np.digitize(value,HABITABILITY_THRESHOLDS).astype(np.uint8)
    allowed=(grid.water==0)&~grid.snow&~polar_continent_mask(grid)
    bands[~allowed]=0
    return _immutable(value),_immutable(bands)


def derive_thematic_layers(grid:WorldGrid,*,ecological_sources:FreshwaterCorridors)->ThematicLayers:
    """Derive all review themes once from one immutable WorldGrid."""

    climate = derive_climate_drivers(grid)
    biomes = derive_biome_zones(grid, climate)
    basins = derive_drainage_basins(grid)
    ranks = rank_major_basins(grid, basins)
    potential, potential_bands = derive_land_potential(grid,climate,ecological_sources=ecological_sources)
    habitability, habitability_bands = derive_habitability(grid,climate,ecological_sources=ecological_sources)
    physiography = derive_physiographic_features(grid, climate)
    return ThematicLayers(
        climate=climate,
        biome_zone=biomes,
        drainage_basin=basins,
        major_basin_rank=ranks,
        land_potential=potential,
        land_potential_band=potential_bands,
        habitability=habitability,
        habitability_band=habitability_bands,
        physiography=physiography,
    )
