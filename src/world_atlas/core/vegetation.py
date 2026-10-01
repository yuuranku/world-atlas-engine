"""Estimated annual vegetated ground fraction from the shared physical world."""

from dataclasses import dataclass

import numpy as np

from .model import WorldGrid
from .suitability import relative_land_slope, smooth_transition
from .thematic import ClimateDrivers
from .continuous_ecology import ContinuousEcologyField, FreshwaterCorridors


VEGETATION_THRESHOLDS = tuple(index / 10.0 for index in range(1, 10))


@dataclass(frozen=True, slots=True)
class VegetationCover:
    """Modelled ground cover by all living plants, rather than forest cover."""

    fraction: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.fraction, dtype=np.float32)
        if value.ndim != 2 or not np.all(np.isfinite(value)):
            raise ValueError("vegetation cover must be a finite two-dimensional field")
        if np.any((value < 0.0) | (value > 1.0)):
            raise ValueError("vegetation fraction must lie within 0..1")
        value = value.copy()
        value.setflags(write=False)
        object.__setattr__(self, "fraction", value)


def derive_vegetation_field(grid:WorldGrid,climate:ClimateDrivers,*,ecological_sources:FreshwaterCorridors)->ContinuousEcologyField:
    """Estimate cover with continuous water, growing-season and terrain limits.

    This is a deterministic ecological estimate for a fictional world. The
    available climate lacks the radiation, humidity and wind observations
    required for reference evapotranspiration; the temperature-based water
    demand below is a model response, not a FAO ET calculation or measured
    satellite vegetation index. Rivers improve local moisture, not rainfall
    across entire drainage basins. Native water and permanent snow are bare.
    """
    land = np.asarray(grid.water) == 0
    temperature = np.asarray(climate.mean_annual_temperature_c, dtype=np.float64)
    annual_rain = np.asarray(climate.annual_precipitation_mm, dtype=np.float64)
    seasonal_rain = np.asarray(climate.seasonal_precipitation, dtype=np.float64)
    if temperature.shape != grid.shape or annual_rain.shape != grid.shape:
        raise ValueError("vegetation climate must match the physical grid")
    demand = 200.0 + 45.0 * np.maximum(temperature + 5.0, 0.0)
    moisture = 1.0 - np.exp(-1.9 * np.maximum(annual_rain, 0.0) / demand)
    seasonal_mean = seasonal_rain.mean(axis=0)
    reliability = np.divide(seasonal_rain.min(axis=0), seasonal_mean,
                            out=np.zeros(grid.shape), where=seasonal_mean > 0.0)
    moisture *= 0.75 + 0.25 * np.sqrt(np.clip(reliability, 0.0, 1.0))
    slope = relative_land_slope(grid.elevation, land)
    growing = smooth_transition(temperature, -8.0, 12.0)
    growing *= smooth_transition(climate.warmest_month_temperature_c, -1.0, 8.0)
    exposed_rock = 1.0 - 0.45 * smooth_transition(grid.elevation, 0.72, 0.98)
    ceiling = np.clip(.98*growing*exposed_rock*(1.-.35*smooth_transition(slope,.04,.14)),0.,1.)
    background = ceiling*moisture
    background[~land|grid.snow]=0.;ceiling[~land|grid.snow]=0.
    sources=ecological_sources
    return ContinuousEcologyField(background,ceiling,land,sources,supply_capacity=.78,
        river_radii=[min(2.,1.+.35*(order-1)) for order in sources.river_orders],lake_radius=2.)


def derive_vegetation_cover(grid:WorldGrid,climate:ClimateDrivers,*,ecological_sources:FreshwaterCorridors)->VegetationCover:
    return VegetationCover(derive_vegetation_field(grid,climate,ecological_sources=ecological_sources).native)
