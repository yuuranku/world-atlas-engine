"""Deterministic physical planet generation from explicit crust and motion.

The fixed spherical reference grid owns plate topology, continental crust,
oceanic age and boundary kinematics.  Those continuous fields are projected
to the requested output resolution before the final sea-level cut, so output
pixels gain detail without becoming square copies of reference cells.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
from world_atlas.core.geomorphology import relax_hillslopes

from world_atlas.core.planet_morphology import (
    WorldMorphology,
    classify_world_morphology,
    sample_world_morphology,
    vars_from_slots,
)


from world_atlas.physical.boundary_chains import merge_boundary_segments, smooth_boundary_chains  # noqa: E402
from world_atlas.physical.hydrology import HydrologyConfig, compute_hydrology  # noqa: E402
from world_atlas.physical.planetary_grid import (  # noqa: E402
    LatLonGrid,
    build_lat_lon_grid,
    lon_lat_to_unit_vector,
    unit_vector_to_lon_lat,
)
from world_atlas.physical.tectonic_potential import build_tectonic_potential  # noqa: E402
from world_atlas.physical.tectonic_relief import (  # noqa: E402
    CRUST_CONTINENTAL,
    CRUST_OCEANIC,
    CRUST_TRANSITIONAL,
    CrustFields,
    TectonicProfileConfig,
    classify_crust,
    compute_ocean_age_myr,
    ocean_depth_from_age,
)
from world_atlas.physical.tectonics import (  # noqa: E402
    BoundaryCorridor,
    PlateDefinition,
    PlateFields,
    build_plate_fields,
    resolve_plate_definitions,
)


BOUNDARY_INTERIOR = 0
BOUNDARY_CONVERGENT = 1
BOUNDARY_DIVERGENT = 2
BOUNDARY_TRANSFORM = 3

_REFERENCE_HEIGHT = 360
_REFERENCE_WIDTH = 720
_PLANET_RADIUS_KM = 6400.0
_STAGE_SALTS = {
    "morphology": 0x13198A2E,
    "plates": 0x243F6A88,
    "crust": 0xB7E15162,
    "coast": 0xA4093822,
    "orogeny": 0x299F31D0,
    "relief": 0x082EFA98,
}


@dataclass(frozen=True, slots=True)
class PlanetRecipe:
    seed: int
    width: int
    height: int
    plate_count: int
    continent_count: int
    land_fraction: float
    tectonic_activity: float
    mountain_density: float
    coastline_detail: float
    north_polar_continent: bool
    south_polar_continent: bool

    def __post_init__(self) -> None:
        for name in ("north_polar_continent", "south_polar_continent"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.width < 32 or self.height < 16:
            raise ValueError("procedural planet grid is too small")
        if self.plate_count < 4:
            raise ValueError("procedural planet needs at least four plates")
        if not 3 <= self.continent_count <= 12:
            raise ValueError("continent_count must be between three and twelve")
        if not 0.18 <= self.land_fraction <= 0.55:
            raise ValueError("land_fraction must be between 0.18 and 0.55")
        for name, value in (
            ("tectonic_activity", self.tectonic_activity),
            ("mountain_density", self.mountain_density),
            ("coastline_detail", self.coastline_detail),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True, slots=True)
class ProceduralSurface:
    plate_id: np.ndarray
    plate_velocity_east_cm_per_year: np.ndarray
    plate_velocity_north_cm_per_year: np.ndarray
    boundary_class: np.ndarray
    crust_kind: np.ndarray
    ocean_age_myr: np.ndarray
    continent_id: np.ndarray
    land_mask: np.ndarray
    elevation: np.ndarray
    bathymetry: np.ndarray
    signed_height: np.ndarray
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        shape = np.asarray(self.land_mask).shape
        if len(shape) != 2:
            raise ValueError("procedural surface arrays must be two-dimensional")
        arrays = {
            "plate_id": np.asarray(self.plate_id, dtype=np.int16),
            "plate_velocity_east_cm_per_year": np.asarray(
                self.plate_velocity_east_cm_per_year, dtype=np.float32
            ),
            "plate_velocity_north_cm_per_year": np.asarray(
                self.plate_velocity_north_cm_per_year, dtype=np.float32
            ),
            "boundary_class": np.asarray(self.boundary_class, dtype=np.int8),
            "crust_kind": np.asarray(self.crust_kind, dtype=np.uint8),
            "ocean_age_myr": np.asarray(self.ocean_age_myr, dtype=np.float32),
            "continent_id": np.asarray(self.continent_id, dtype=np.int16),
            "land_mask": np.asarray(self.land_mask, dtype=bool),
            "elevation": np.asarray(self.elevation, dtype=np.float32),
            "bathymetry": np.asarray(self.bathymetry, dtype=np.float32),
            "signed_height": np.asarray(self.signed_height, dtype=np.float32),
        }
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("procedural surface arrays must share one shape")
        for name, array in arrays.items():
            immutable = np.array(array, copy=True)
            immutable.setflags(write=False)
            object.__setattr__(self, name, immutable)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def _derive_seed(root_seed: int, stage: str) -> int:
    value = (int(root_seed) ^ _STAGE_SALTS[stage]) & 0xFFFFFFFF
    value = (value + 0x9E3779B9) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x85EBCA6B) & 0xFFFFFFFF
    value ^= value >> 13
    value = (value * 0xC2B2AE35) & 0xFFFFFFFF
    value ^= value >> 16
    return value


def _normalise(values: np.ndarray) -> np.ndarray:
    low, high = np.quantile(values.astype(np.float64), (0.02, 0.98))
    if high <= low + 1.0e-12:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _polar_crust_bias(field: np.ndarray, latitude: np.ndarray, recipe: PlanetRecipe) -> np.ndarray:
    """Set each polar crust domain before sea level, preserving local relief.

    A spherical cap has no longitude seam. The transition blends into the
    existing tectonic field; it does not stamp a post-render ice continent.
    """
    latitude = np.asarray(latitude).reshape(-1, 1)
    transition = np.clip((np.abs(latitude) - 58.0) / 26.0, 0.0, 1.0)
    weight = transition * transition * (3.0 - 2.0 * transition)
    positive = np.where(latitude >= 0, recipe.north_polar_continent, recipe.south_polar_continent)
    span = float(np.ptp(field)) + 1.0
    return (field + np.where(positive, 1.0, -1.0) * span * weight).astype(np.float32)


def _box_blur_axis(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(values, dtype=np.float32)
    padding = [(0, 0), (0, 0)]
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="wrap" if axis == 1 else "edge")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
    leading = [(0, 0), (0, 0)]
    leading[axis] = (1, 0)
    cumulative = np.pad(cumulative, leading, mode="constant")
    window = 2 * radius + 1
    if axis == 1:
        result = cumulative[:, window:] - cumulative[:, :-window]
    else:
        result = cumulative[window:, :] - cumulative[:-window, :]
    return (result / window).astype(np.float32)


def _smooth_field(values: np.ndarray, radius: int, passes: int = 1) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    for _ in range(passes):
        result = _box_blur_axis(result, radius, axis=1)
        result = _box_blur_axis(result, radius, axis=0)
    return result


def _sphere_vectors(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    longitude = (
        (np.arange(width, dtype=np.float32)[None, :] + 0.5)
        * (math.tau / width)
        - math.pi
    )
    latitude = (
        math.pi / 2.0
        - (np.arange(height, dtype=np.float32)[:, None] + 0.5)
        * (math.pi / height)
    )
    cosine = np.cos(latitude)
    x = (cosine * np.cos(longitude)).astype(np.float32)
    y = (cosine * np.sin(longitude)).astype(np.float32)
    z = np.broadcast_to(np.sin(latitude), (height, width)).astype(np.float32)
    return x, y, z


def _spherical_fbm(height: int, width: int, seed: int, *, detail: float = 1.0) -> np.ndarray:
    x, y, z = _sphere_vectors(height, width)
    rng = np.random.Generator(np.random.PCG64(int(seed) & 0xFFFFFFFFFFFFFFFF))
    result = np.zeros((height, width), dtype=np.float32)
    amplitude_total = 0.0
    for frequency, amplitude, wave_count in (
        (1.35, 0.46, 5),
        (2.7, 0.27, 5),
        (5.4, 0.16, 4),
        (10.8, 0.08 * detail, 4),
        (21.6, 0.03 * detail, 3),
        (43.2, 0.016 * detail, 3),
        (76.0, 0.008 * detail, 2),
    ):
        for _ in range(wave_count):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            phase = float(rng.uniform(-math.pi, math.pi))
            projection = x * direction[0] + y * direction[1] + z * direction[2]
            result += (amplitude / wave_count) * np.sin(
                math.pi * frequency * projection + phase
            )
        amplitude_total += amplitude
    result /= max(amplitude_total, 1.0e-9)
    result -= float(np.mean(result))
    scale = float(np.quantile(np.abs(result), 0.98))
    return np.clip(result / max(scale, 1.0e-9), -1.0, 1.0).astype(np.float32)


def _spherical_band_noise(
    height: int,
    width: int,
    seed: int,
    bands: Sequence[tuple[float, float, int]],
) -> np.ndarray:
    """Return deterministic, seam-free relief carried by explicit scale bands."""

    x, y, z = _sphere_vectors(height, width)
    rng = np.random.Generator(np.random.PCG64(int(seed) & 0xFFFFFFFFFFFFFFFF))
    result = np.zeros((height, width), dtype=np.float32)
    amplitude_total = 0.0
    for frequency, amplitude, wave_count in bands:
        if frequency <= 0.0 or amplitude <= 0.0 or wave_count <= 0:
            raise ValueError("spherical noise bands must be positive")
        for _ in range(wave_count):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            phase = float(rng.uniform(-math.pi, math.pi))
            projection = x * direction[0] + y * direction[1] + z * direction[2]
            result += (amplitude / wave_count) * np.sin(
                math.pi * frequency * projection + phase
            )
        amplitude_total += amplitude
    result /= max(amplitude_total, 1.0e-9)
    result -= float(np.mean(result))
    scale = float(np.quantile(np.abs(result), 0.985))
    return np.clip(result / max(scale, 1.0e-9), -1.0, 1.0).astype(np.float32)


def _full_resolution_relief_detail(
    relative_height_m: np.ndarray,
    provisional_land: np.ndarray,
    low_relief_support: np.ndarray,
    seed: int,
    coastline_detail: float,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Sculpt 60--900 km landforms on the final synthesis grid.

    Plate kinematics still own the large-scale uplift. These scale-banded
    fields only roughen existing ranges, dissect continental interiors and
    articulate the sea-level transition. They therefore cannot create a
    mountain system in an unrelated ocean or turn global noise into geology.
    """

    relative = np.asarray(relative_height_m, dtype=np.float32)
    land = np.asarray(provisional_land, dtype=bool)
    quiet_surface = np.asarray(low_relief_support, dtype=np.float32)
    if relative.shape != land.shape or quiet_surface.shape != relative.shape:
        raise ValueError("full-resolution relief fields must share one shape")
    height, width = relative.shape
    subregional = _spherical_band_noise(
        height,
        width,
        seed ^ 0xDB4F0B9175AE2165,
        ((17.0, 0.25, 6), (29.0, 0.31, 6), (47.0, 0.27, 6), (71.0, 0.17, 5)),
    )
    regional = _spherical_band_noise(
        height,
        width,
        seed ^ 0x94D049BB133111EB,
        (
            (38.0, 0.34, 5),
            (61.0, 0.27, 5),
            (97.0, 0.21, 5),
            (143.0, 0.12, 4),
            (193.0, 0.06, 4),
        ),
    )
    local = _spherical_band_noise(
        height,
        width,
        seed ^ 0x2545F4914F6CDD1D,
        (
            (113.0, 0.30, 5),
            (181.0, 0.25, 5),
            (263.0, 0.22, 5),
            (367.0, 0.15, 4),
            (487.0, 0.08, 4),
        ),
    )
    coastal_micro = _spherical_band_noise(
        height,
        width,
        seed ^ 0x369DEA0F31A53F85,
        (
            (191.0, 0.34, 5),
            (283.0, 0.29, 5),
            (397.0, 0.23, 5),
            (541.0, 0.14, 4),
        ),
    )

    positive_relief = np.maximum(relative, 0.0)
    land_relief = positive_relief[land]
    mountain_floor = float(np.quantile(land_relief, 0.67))
    mountain_ceiling = float(np.quantile(land_relief, 0.975))
    mountain_support = np.power(
        np.clip(
            (positive_relief - mountain_floor)
            / max(mountain_ceiling - mountain_floor, 1.0),
            0.0,
            1.0,
        ),
        0.72,
    )
    upland_support = np.clip(positive_relief / 1250.0, 0.0, 1.0)
    interior_support = land.astype(np.float32) * (0.40 + 0.60 * upland_support)
    regional_floor = _smooth_field(
        relative, radius=max(2, int(round(width / 180.0))), passes=2,
    )
    # Sedimentary plains and plateau interiors keep their low-relief fabric.
    # A later orogen can cross a plain, so substantial inherited relief locally
    # releases this constraint instead of flattening an active mountain belt.
    quiet_surface = np.clip(quiet_surface, 0.0, 1.0) * (
        1.0 - np.clip(np.abs(relative - regional_floor) / 900.0, 0.0, 1.0)
    )
    fabric_strength = 1.0 - 0.82 * quiet_surface
    interior_support *= fabric_strength
    coast_width_m = 260.0 + 210.0 * coastline_detail
    coastal_support = np.exp(
        -0.5 * np.square(relative / coast_width_m)
    ).astype(np.float32)

    # Zero crossings of a band-limited field form branching, broken ridges.
    # Confining them to inherited uplift turns a smooth Gaussian tube into a
    # mountain system with parallel crests, passes and eroded flanks.
    ridge_signal = _normalise(1.0 - np.abs(local)) - 0.47
    detail = (
        interior_support
        * (
            400.0 * subregional
            + (320.0 + 160.0 * coastline_detail) * regional
            + 125.0 * local
        )
    )
    detail += (
        land.astype(np.float32)
        * mountain_support
        * fabric_strength
        * (1150.0 * ridge_signal + 360.0 * regional + 140.0 * local)
    )
    # Sea-level detail is deliberately weaker than mountain detail. It adds
    # capes, coves, skerries and drowned valleys without dissolving continents.
    detail += coastal_support * (
        (135.0 + 140.0 * coastline_detail) * regional
        + (82.0 + 98.0 * coastline_detail) * local
    )
    detail = _smooth_field(detail.astype(np.float32), radius=2, passes=1)
    # Preserve one final shoreline-only scale band after the general terrain
    # smoothing pass.  Because its support follows the provisional zero
    # contour, it cuts coves and capes into existing margins rather than
    # scattering unrelated islands across the ocean.
    detail += coastal_support * (
        (92.0 + 165.0 * coastline_detail) * coastal_micro
    )
    return detail, {
        "grid": {"height": height, "width": width},
        "scaleBandsKm": {
            "subregional": [560.0, 2360.0],
            "regional": [205.0, 1050.0],
            "local": [80.0, 355.0],
        },
        "maximumDetailMeters": float(np.max(np.abs(detail))),
        "mountainSupportFraction": float(np.mean(mountain_support[land] >= 0.10)),
        "coastalSupportFraction": float(np.mean(coastal_support >= 0.25)),
        "lowReliefFabricLandFraction": float(np.mean(quiet_surface[land] >= 0.45)),
    }


def _visible_ocean_bathymetry(
    land_mask: np.ndarray,
    ocean_depth_m: np.ndarray,
    crust_kind: np.ndarray,
    ocean_age_myr: np.ndarray,
    distance_to_ridge_km: np.ndarray,
    distance_to_convergence_km: np.ndarray,
    boundary_class: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Render a geologically legible ocean floor without exposing plate cells.

    Ocean-crust age owns the basin-scale depth and therefore creates a broad
    ridge swell instead of a thin boundary stroke.  Only oceanic convergence
    adds a narrow trench.  Low-amplitude band-limited abyssal texture supplies
    local detail, while the raw tectonic potential is deliberately excluded
    from the visible deep-ocean field because it contains the complete plate
    graph, including transforms that should not read as equal-strength lines.
    """

    land = np.asarray(land_mask, dtype=bool)
    depth_m = np.asarray(ocean_depth_m, dtype=np.float32)
    kind = np.asarray(crust_kind, dtype=np.uint8)
    age = np.asarray(ocean_age_myr, dtype=np.float32)
    ridge_distance = np.asarray(distance_to_ridge_km, dtype=np.float32)
    convergence_distance = np.asarray(distance_to_convergence_km, dtype=np.float32)
    classes = np.asarray(boundary_class, dtype=np.int8)
    shape = land.shape
    if any(
        value.shape != shape
        for value in (
            depth_m,
            kind,
            age,
            ridge_distance,
            convergence_distance,
            classes,
        )
    ):
        raise ValueError("visible ocean-floor fields must share one shape")

    ocean = ~land
    ocean_scale = max(float(np.quantile(depth_m[ocean], 0.985)), 1.0)
    raw_relative_depth = np.clip(
        np.power(depth_m / ocean_scale, 0.78),
        0.0,
        1.0,
    ).astype(np.float32)
    age_fraction = np.clip(np.nan_to_num(age, nan=0.0) / 190.0, 0.0, 1.0)
    thermal_depth = np.sqrt(age_fraction).astype(np.float32)
    # The extra swell is broad (hundreds of kilometres) and subordinate to
    # thermal subsidence.  It prevents the youngest crust from becoming a
    # razor-thin contour even where the classified ridge cells are sparse.
    ridge_swell = np.exp(
        -0.5 * np.square(ridge_distance / 720.0)
    ).astype(np.float32)
    deep_floor = (
        0.29
        + 0.51 * thermal_depth
        - 0.055 * ridge_swell
    )

    abyssal_texture = _spherical_band_noise(
        shape[0],
        shape[1],
        seed ^ 0xA24BAED4963EE407,
        (
            (19.0, 0.18, 5),
            (31.0, 0.22, 5),
            (49.0, 0.22, 5),
            (73.0, 0.17, 5),
            (109.0, 0.13, 4),
            (157.0, 0.08, 4),
        ),
    )
    deep_floor += (0.012 + 0.013 * thermal_depth) * abyssal_texture

    # A real trench belongs only to a convergent ocean margin.  Segment its
    # visible depth along strike so it reads as a chain of basins rather than
    # another unbroken vector line.
    trench_profile = np.exp(
        -0.5 * np.square(convergence_distance / 115.0)
    ).astype(np.float32)
    trench_segmentation = _normalise(
        _smooth_field(
            _spherical_fbm(
                shape[0],
                shape[1],
                seed ^ 0x9FB21C651E98DF25,
                detail=0.42,
            ),
            radius=max(2, int(round(shape[1] / 180.0))),
            passes=2,
        )
    )
    trench_strength = 0.055 + 0.080 * np.power(trench_segmentation, 1.6)
    arc_support = np.clip(
        _smooth_field(
            land.astype(np.float32),
            radius=max(2, int(round(shape[1] / 70.0))),
            passes=2,
        ) / 0.16,
        0.0,
        1.0,
    )
    deep_floor += trench_profile * trench_strength * arc_support

    # Build a continuous shelf around the *actual* shoreline, including
    # volcanic islands.  This avoids a categorical ring at the edge of the
    # continental-crust mask.
    shelf_steps = max(6, int(round(shape[1] / 105.0)))
    distance_cells = np.full(shape, shelf_steps + 1, dtype=np.int16)
    reached = land.copy()
    distance_cells[land] = 0
    for step in range(1, shelf_steps + 1):
        expanded = _dilate(reached, 1)
        frontier = expanded & ~reached
        distance_cells[frontier] = step
        reached = expanded
    shelf_progress = np.clip(
        (distance_cells.astype(np.float32) - 0.5) / max(shelf_steps - 1.0, 1.0),
        0.0,
        1.0,
    )
    shelf_progress = shelf_progress * shelf_progress * (3.0 - 2.0 * shelf_progress)
    shelf_depth = 0.018 + 0.33 * raw_relative_depth
    oceanic_support = _smooth_field(
        np.where(kind == CRUST_OCEANIC, 1.0, 0.30).astype(np.float32),
        radius=max(2, int(round(shape[1] / 360.0))),
        passes=2,
    )
    deep_blend = np.clip(shelf_progress * oceanic_support, 0.0, 1.0)
    bathymetry = (
        shelf_depth * (1.0 - deep_blend)
        + deep_floor * deep_blend
    ).astype(np.float32)
    bathymetry[land] = 0.0

    # One light pass removes colour-scale pinholes without discarding the
    # multi-scale abyssal fabric added above.
    smoothed = _smooth_field(
        bathymetry,
        radius=max(1, int(round(shape[1] / 1080.0))),
        passes=1,
    )
    bathymetry[ocean] = np.clip(
        0.82 * bathymetry[ocean] + 0.18 * smoothed[ocean],
        0.015,
        1.0,
    )

    gradient = np.hypot(
        np.gradient(bathymetry.astype(np.float64), axis=0),
        np.gradient(bathymetry.astype(np.float64), axis=1),
    )
    corridor_passes = max(2, int(round(shape[1] / 900.0)))
    boundary_corridor = _dilate(classes != BOUNDARY_INTERIOR, corridor_passes)
    open_ocean = ocean & ~_dilate(
        classes != BOUNDARY_INTERIOR,
        max(3, int(round(shape[1] / 180.0))),
    )
    open_gradient = max(float(np.mean(gradient[open_ocean])), 1.0e-9)

    def gradient_ratio(boundary_value: int) -> float:
        corridor = ocean & _dilate(classes == boundary_value, corridor_passes)
        if not np.any(corridor):
            return 1.0
        return float(np.mean(gradient[corridor]) / open_gradient)

    residual = np.abs(
        bathymetry
        - _smooth_field(bathymetry, radius=2, passes=2)
    )
    deep_ocean = ocean & (kind == CRUST_OCEANIC) & ~boundary_corridor
    abyssal_plain_fraction = (
        float(np.mean(residual[deep_ocean] < 0.025))
        if np.any(deep_ocean)
        else 0.0
    )
    return bathymetry, {
        "visibleReliefModel": "thermal-age-ridge-swell-subduction-trench",
        "ridgeSwellHalfWidthKm": 720.0,
        "trenchHalfWidthKm": 115.0,
        "trenchArcSupport": "convergent-oceanic-margin-near-exposed-crust",
        "transformGradientRatio": gradient_ratio(BOUNDARY_TRANSFORM),
        "divergentGradientRatio": gradient_ratio(BOUNDARY_DIVERGENT),
        "convergentGradientRatio": gradient_ratio(BOUNDARY_CONVERGENT),
        "abyssalPlainFraction": abyssal_plain_fraction,
        "shelfTransitionCells": shelf_steps,
    }


def _evolve_coastal_margins(relative_m, low_relief_support, boundary_class,
                           velocity_east, velocity_north, seed, coastline_detail):
    """Offset whole coastal sectors while transporting their continuous relief.

    The plate-motion field bounds structural slip. Varying affine segments
    represent rift/transfer accommodation, not a calibrated history. Sampling
    the original height field moves the coast and local hills together; no
    triangle-shaped holes are cut into a finished terrain image.
    """
    from world_atlas.core.coastal_margins import margin_motion, motion_budget_km
    relative = np.asarray(relative_m, dtype=np.float32)
    land = relative > 0
    height, width = land.shape
    coast = land & ~(np.roll(land,1,0) & np.roll(land,-1,0)
                     & np.roll(land,1,1) & np.roll(land,-1,1))
    coast[:max(1,height//12)] = False
    coast[-max(1,height//12):] = False
    rows, cols = np.where(coast)
    row_offsets = np.zeros(land.shape,dtype=np.float32)
    col_offsets = np.zeros_like(row_offsets)
    weights = np.zeros_like(row_offsets)
    if not len(rows):
        return row_offsets, {'modifiedShoreSegments':0}
    lat = np.pi/2-(rows+.5)*np.pi/height
    lon = (cols+.5)*np.pi*2/width-np.pi
    vectors = np.column_stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)))
    rng = np.random.Generator(np.random.PCG64(seed ^ 0x6A09E667F3BCC909))
    row_km = np.pi*_PLANET_RADIUS_KM/height
    gradient_row, gradient_col = np.gradient(_smooth_field(relative,max(1,round(width/270)),passes=2))
    reach = max(1,round(300/row_km))
    convergent = _dilate(boundary_class == BOUNDARY_CONVERGENT,reach)
    divergent = _dilate(boundary_class == BOUNDARY_DIVERGENT,reach)
    transform = _dilate(boundary_class == BOUNDARY_TRANSFORM,reach)
    available = np.ones(len(rows),dtype=bool)
    scores = rng.random(len(rows))*np.sqrt(np.cos(lat))
    counts = {kind:0 for kind in ('rift','active','passive','transform','sedimentary')}
    slips = []
    for ordinal in range(480):
        if not available.any():
            break
        selected = int(np.argmax(np.where(available,scores,-1)))
        row,col = int(rows[selected]),int(cols[selected])
        available &= (vectors @ vectors[selected]) < np.cos(350/_PLANET_RADIUS_KM)
        quiet = float(low_relief_support[row,col])
        if quiet > .62 and not (convergent[row,col] or divergent[row,col] or transform[row,col]):
            counts['sedimentary'] += 1
            continue
        family = ('rift' if divergent[row,col] else 'active' if convergent[row,col]
                  else 'transform' if transform[row,col] else 'passive')
        cos_lat = max(float(np.cos(lat[selected])),.15)
        col_km = 2*np.pi*_PLANET_RADIUS_KM/width*cos_lat
        north = -float(gradient_row[row,col])/row_km
        east = float(gradient_col[row,col])/col_km
        norm = np.hypot(north,east)
        if norm < 1e-8:
            continue
        north,east = north/norm,east/norm
        speed = np.hypot(float(velocity_east[row,col]),float(velocity_north[row,col]))
        # cm/yr * Myr = 10 km; bounded because this is margin-scale reactivation.
        slip = .40*float(motion_budget_km(speed,rng.uniform(1.5,4.5)))
        length = float(rng.uniform(250,750))
        influence = max(360.,slip*4.)
        radius = max(length,influence)*1.5
        rr = np.arange(max(0,row-int(radius/row_km)-2),min(height,row+int(radius/row_km)+3))
        cc = np.arange(col-int(radius/col_km)-2,col+int(radius/col_km)+3) % width
        east_km = ((cc[None,:]-col+width/2)%width-width/2)*col_km
        north_km = (row-rr[:,None])*row_km
        inward = north_km*north+east_km*east
        along = north_km*east-east_km*north
        normal,tangent,support = margin_motion(along,inward,length,influence,slip,
                                              seed=int(rng.integers(0,2**63)),family=family)
        patch_index = np.ix_(rr,cc)
        # Average overlapping motions; never accumulate them into a folded map.
        row_offsets[patch_index] += (normal*north+tangent*east)/row_km
        col_offsets[patch_index] += (-normal*east+tangent*north)/col_km
        weights[patch_index] += support
        counts[family] += 1
        slips.append(slip)
    denominator = np.maximum(weights,1.)
    row_offsets /= denominator
    col_offsets /= denominator
    evolved = _warp_periodic_field(relative,row_offsets,col_offsets)
    return evolved-relative, {
        'model':'full-resolution-finite-structural-margin-displacement',
        'modifiedShoreSegments':len(slips),'families':counts,
        'lengthBandsKm':{'coastalSectorHalfLength':[250,750]},
        'slipRangeKm':[min(slips,default=0),max(slips,default=0)],
        'grid':{'height':height,'width':width},
        'maximumCoordinateDisplacementCells':float(np.max(np.hypot(row_offsets,col_offsets))),
    }


def _fluvial_dissection(
    relative_height_m: np.ndarray,
    land_mask: np.ndarray,
    latitude_degrees: np.ndarray,
    low_relief_support: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Incise a full-resolution drainage graph into the generated relief."""

    relative = np.asarray(relative_height_m, dtype=np.float32)
    land = np.asarray(land_mask, dtype=bool)
    land_heights = np.maximum(relative[land], 0.0)
    elevation_scale = max(float(np.quantile(land_heights, 0.997)), 1.0)
    normalized = np.zeros(relative.shape, dtype=np.float64)
    normalized[land] = np.clip(land_heights / elevation_scale, 0.0, 1.0)
    hydrology = compute_hydrology(
        normalized,
        land,
        ~land,
        config=HydrologyConfig(
            stream_burn_depth=0.0,
            stream_threshold_fraction=0.00008,
            tributary_threshold_fraction=0.0012,
            mainstem_threshold_fraction=0.012,
            minimum_headwater_length=6,
            minimum_stream_span_factor=0.04,
            closure_budget_fraction=0.01,
            closure_max_chain_fraction=0.012,
            mfd_exponent=1.1,
            valley_window_fraction=0.02,
            valley_depth_fraction=0.01,
            valley_support_threshold=0.09,
            valley_support_distance_factor=1.0,
            parallel_search_radius_factor=0.5,
            parallel_priority_weight=0.35,
            runoff_weight_floor=0.28,
            snowmelt_runoff_bonus=0.70,
            headwater_upland_quantile=0.68,
            lowland_headwater_flow_multiplier=1.4,
            major_rivers_per_continent=2,
            major_river_tributaries=3,
            continent_minimum_land_fraction=0.01,
            major_river_minimum_span_factor=0.35,
        ),
        latitude_degrees=np.asarray(latitude_degrees, dtype=np.float64),
    )
    accumulation = np.asarray(hydrology.mfd_accumulation, dtype=np.float64)
    maximum_accumulation = max(float(np.max(accumulation)), 1.0)
    flow_strength = np.log1p(accumulation) / math.log1p(maximum_accumulation)
    channel_strength = np.zeros(relative.shape, dtype=np.float32)
    channel_strength[hydrology.stream_mask] = np.asarray(
        0.22 + 0.78 * np.power(flow_strength[hydrology.stream_mask], 1.35),
        dtype=np.float32,
    )
    corridor_radius = max(1, int(round(relative.shape[1] / 1440.0)))
    valley_corridor = np.maximum(
        channel_strength,
        0.68
        * _smooth_field(
            channel_strength,
            radius=corridor_radius,
            passes=1,
        ),
    )
    # The map's displayed streams are only major rivers.  Minor tributaries
    # also dissect foothills and uplands, so erosion uses the complete MFD
    # accumulation instead of restricting every valley to the display mask.
    local_floor = _smooth_field(
        np.maximum(relative, 0.0),
        radius=max(2, int(round(relative.shape[1] / 360.0))),
        passes=2,
    )
    local_relief = np.abs(relative - local_floor)
    tributary_flow = np.power(
        np.clip(np.log1p(np.maximum(accumulation - 2.0, 0.0)) / 5.0, 0.0, 1.0),
        1.30,
    )
    # Absolute altitude is not local ruggedness: a 2,400 m plateau can have
    # gentler tributaries than a 500 m dissected foothill belt.
    erodibility = np.clip(local_relief / 220.0, 0.0, 1.0)
    quiet_surface = np.clip(low_relief_support, 0.0, 1.0) * (
        1.0 - np.clip(local_relief / 450.0, 0.0, 1.0)
    )
    minor_valleys = (
        (25.0 + 380.0 * erodibility + 0.32 * local_relief)
        * tributary_flow
        * land
        * (1.0 - 0.80 * quiet_surface)
    )
    requested_incision = (
        (85.0 + 590.0 * np.power(flow_strength, 1.45)) * valley_corridor
        + minor_valleys
    )
    available_relief = np.maximum(relative - 12.0, 0.0)
    incision = np.minimum(requested_incision, 0.46 * available_relief)
    dissected = relative.copy()
    dissected[land] -= incision[land].astype(np.float32)
    return dissected, {
        "model": "full-resolution-priority-flood-d8-mfd",
        "drainageReachCount": len(hydrology.network),
        "streamCellCount": int(np.count_nonzero(hydrology.stream_mask)),
        "carvedValleyLandFraction": float(np.mean(incision[land] >= 20.0)),
        "maximumIncisionMeters": float(np.max(incision, initial=0.0)),
        "minorValleyLandFraction": float(np.mean(minor_valleys[land] >= 25.0)),
    }


def _weighted_level(
    field: np.ndarray,
    latitude_weight: np.ndarray,
    target_fraction: float,
) -> float:
    lower = float(np.min(field))
    upper = float(np.max(field))
    total_weight = float(np.sum(latitude_weight)) * field.shape[1]
    for _ in range(44):
        midpoint = 0.5 * (lower + upper)
        fraction = float(np.sum((field > midpoint) * latitude_weight)) / total_weight
        if fraction > target_fraction:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _plate_definitions(
    count: int,
    morphology: WorldMorphology,
    seed: int,
) -> tuple[PlateDefinition, ...]:
    major_count = max(4, min(count, round(count * (1.0 - 0.46 * morphology.microplate_share))))
    generated = resolve_plate_definitions(major_count, (), seed)
    majors: list[PlateDefinition] = []
    for ordinal, value in enumerate(generated, 1):
        majors.append(
            PlateDefinition(
                f"plate-major-{ordinal:02d}",
                value.anchor_lon,
                value.anchor_lat,
                value.euler_pole_lon,
                value.euler_pole_lat,
                value.angular_speed_deg_per_myr,
                True,
            )
        )
    if major_count == count:
        return tuple(majors)

    rng = np.random.Generator(np.random.PCG64(seed ^ 0x9E3779B97F4A7C15))
    result = list(majors)
    for ordinal in range(count - major_count):
        parent = majors[ordinal % len(majors)]
        parent_vector = np.asarray(
            lon_lat_to_unit_vector(parent.anchor_lon, parent.anchor_lat),
            dtype=np.float64,
        )
        tangent = rng.normal(size=3)
        tangent -= float(np.dot(tangent, parent_vector)) * parent_vector
        tangent /= np.linalg.norm(tangent)
        separation = math.radians(
            float(rng.uniform(7.0, 18.0))
            * (1.18 - 0.42 * morphology.microplate_share)
        )
        vector = math.cos(separation) * parent_vector + math.sin(separation) * tangent
        longitude, latitude = unit_vector_to_lon_lat(vector)
        result.append(
            PlateDefinition(
                f"plate-micro-{ordinal + 1:02d}",
                float(longitude),
                float(latitude),
                (
                    parent.euler_pole_lon
                    + float(rng.uniform(-22.0, 22.0))
                    + 180.0
                )
                % 360.0
                - 180.0,
                float(np.clip(parent.euler_pole_lat + rng.uniform(-16.0, 16.0), -89.0, 89.0)),
                parent.angular_speed_deg_per_myr * float(rng.uniform(0.72, 1.34)),
                False,
            )
        )
    return tuple(result)


def _advect_plate_definitions(
    definitions: Sequence[PlateDefinition],
    lookback_myr: float,
) -> tuple[PlateDefinition, ...]:
    """Move plate anchors backward along their Euler rotations."""

    result: list[PlateDefinition] = []
    for definition in definitions:
        anchor = np.asarray(
            lon_lat_to_unit_vector(definition.anchor_lon, definition.anchor_lat),
            dtype=np.float64,
        )
        pole = np.asarray(
            lon_lat_to_unit_vector(
                definition.euler_pole_lon,
                definition.euler_pole_lat,
            ),
            dtype=np.float64,
        )
        angle = -math.radians(
            definition.angular_speed_deg_per_myr * float(lookback_myr)
        )
        rotated = (
            anchor * math.cos(angle)
            + np.cross(pole, anchor) * math.sin(angle)
            + pole * float(np.dot(pole, anchor)) * (1.0 - math.cos(angle))
        )
        rotated /= np.linalg.norm(rotated)
        longitude, latitude = unit_vector_to_lon_lat(rotated)
        result.append(
            PlateDefinition(
                definition.plate_id,
                float(longitude),
                float(latitude),
                definition.euler_pole_lon,
                definition.euler_pole_lat,
                definition.angular_speed_deg_per_myr,
                definition.major,
            )
        )
    return tuple(result)


def _curved_boundary_corridors(
    plate_fields: PlateFields,
    morphology: WorldMorphology,
    seed: int,
) -> tuple[BoundaryCorridor, ...]:
    """Turn first-pass geodesic borders into gently sinuous plate corridors."""

    chains = merge_boundary_segments(plate_fields.boundaries, _PLANET_RADIUS_KM)
    rng = np.random.Generator(np.random.PCG64(seed ^ 0x9B05688C2B3E6C1F))
    corridors: list[BoundaryCorridor] = []
    for chain_ordinal, chain in enumerate(chains):
        if chain.length_km < 520.0 or len(chain.points_lon_lat) < 3:
            continue
        original = tuple(chain.points_lon_lat)
        unique_count = len(original) - 1 if chain.closed else len(original)
        control_count = min(13, max(4, int(math.ceil(chain.length_km / 1450.0)) + 2))
        indices = np.linspace(0, unique_count - 1, control_count, dtype=np.int64)
        indices = np.unique(indices)
        vectors = np.asarray(
            [lon_lat_to_unit_vector(*original[int(index)]) for index in indices],
            dtype=np.float64,
        )
        phase = float(rng.uniform(-math.pi, math.pi))
        cycles = 1.0 + float(rng.integers(1, 4))
        maximum_offset = math.radians(
            1.35 + 2.25 * morphology.crust_fragmentation
        )
        controls: list[tuple[float, float]] = []
        for index, vector in enumerate(vectors):
            if not chain.closed and index in (0, len(vectors) - 1):
                displaced = vector
            else:
                previous = vectors[(index - 1) % len(vectors)]
                following = vectors[(index + 1) % len(vectors)]
                tangent = following - previous
                tangent -= float(np.dot(tangent, vector)) * vector
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm <= 1.0e-9:
                    displaced = vector
                else:
                    tangent /= tangent_norm
                    lateral = np.cross(vector, tangent)
                    lateral /= np.linalg.norm(lateral)
                    progress = index / max(len(vectors) - 1, 1)
                    envelope = 1.0 if chain.closed else math.sin(math.pi * progress)
                    offset = (
                        maximum_offset
                        * envelope
                        * math.sin(phase + math.tau * cycles * progress)
                    )
                    displaced = math.cos(offset) * vector + math.sin(offset) * lateral
                    displaced /= np.linalg.norm(displaced)
            longitude, latitude = unit_vector_to_lon_lat(displaced)
            controls.append((float(longitude), float(latitude)))
        if chain.closed:
            controls.append(controls[0])
        try:
            corridors.append(
                BoundaryCorridor(
                    f"generated-corridor-{chain_ordinal:03d}",
                    chain.plate_ids,
                    tuple(controls),
                    190.0 + 135.0 * morphology.crust_fragmentation,
                    0.34 + 0.20 * morphology.crust_fragmentation,
                )
            )
        except ValueError:
            continue
    return tuple(corridors)


def _select_crust_roots(
    grid: LatLonGrid,
    count: int,
    morphology: WorldMorphology,
    seed: int,
) -> tuple[int, ...]:
    vectors = np.asarray(grid.unit_vectors, dtype=np.float64).reshape((-1, 3))
    latitudes = np.abs(np.asarray(grid.latitude_degrees, dtype=np.float64).reshape(-1)) / 90.0
    signed_latitudes = np.asarray(grid.latitude_degrees, dtype=np.float64).reshape(-1) / 90.0
    candidates = np.flatnonzero(latitudes <= 0.86)
    rng = np.random.Generator(np.random.PCG64(seed))
    desired_abs_latitude = 0.20 + 0.45 * max(morphology.latitude_bias, 0.0)
    latitude_score = np.exp(-np.square((latitudes[candidates] - desired_abs_latitude) / 0.30))
    hemisphere_sign = -1.0 if rng.random() < 0.5 else 1.0
    hemisphere_score = 0.5 + 0.5 * (
        1.0
        + hemisphere_sign * signed_latitudes[candidates] * morphology.hemisphere_asymmetry
    )
    # Sample an area-weighted continental province field, not a farthest-point
    # packing. Dispersal is not equal spacing: neighbouring crustal fragments
    # and large empty basins must coexist even near minimum aggregation.
    province = _spherical_band_noise(
        *grid.shape, seed ^ 0x8CB92BA72F3D8DD7,
        ((1.3, 0.65, 6), (2.7, 0.35, 6)),
    ).reshape(-1)[candidates]
    density = (
        np.cos(latitudes[candidates] * math.pi / 2.0)
        * (0.06 + 0.94 * latitude_score)
        * hemisphere_score
        * np.exp(1.35 * province)
    )
    first = int(rng.choice(candidates, p=density / np.sum(density)))
    selected = [first]
    cluster = vectors[first]
    for _ in range(1, count):
        similarities = vectors[candidates] @ vectors[np.asarray(selected)].T
        nearest_angle = np.arccos(np.clip(np.max(similarities, axis=1), -1.0, 1.0))
        cluster_angle = np.arccos(np.clip(vectors[candidates] @ cluster, -1.0, 1.0))
        cluster_radius = math.radians(28.0 + 72.0 * (1.0 - morphology.continental_aggregation))
        clustered = np.exp(-0.5 * np.square(cluster_angle / max(cluster_radius, 1.0e-6)))
        exclusion = math.radians(22.0 + 10.0 * (1.0 - morphology.continental_aggregation))
        score = density * ((1.0 - morphology.continental_aggregation) + 3.0 * morphology.continental_aggregation * clustered)
        score *= np.clip((nearest_angle - exclusion) / math.radians(12.0), 0.0, 1.0)
        selected.append(int(rng.choice(candidates, p=score / np.sum(score))))
        cluster = np.sum(vectors[np.asarray(selected)], axis=0)
        cluster /= np.linalg.norm(cluster)
    return tuple(selected)


def _tangent_direction(vector: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    direction = rng.normal(size=3)
    direction -= float(np.dot(direction, vector)) * vector
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        direction = np.cross(vector, np.array([0.0, 0.0, 1.0]))
        norm = float(np.linalg.norm(direction))
    return direction / norm


def _grow_continental_crust(
    grid: LatLonGrid,
    roots: Sequence[int],
    plate_grid: np.ndarray,
    boundary_class: np.ndarray,
    morphology: WorldMorphology,
    target_fraction: float,
    seed: int,
) -> np.ndarray:
    """Grow connected continental terranes over a spherical tectonic cost field."""

    root_cells = tuple(int(value) for value in roots)
    if not root_cells:
        raise ValueError("continental crust needs at least one root")
    rng = np.random.Generator(np.random.PCG64(seed ^ 0x6A09E667F3BCC909))
    height, width = grid.shape
    cell_count = grid.cell_count
    # Accretion follows regional crustal fabric. A weak, nearly constant
    # speed field produces travel-time balls (rounded continents), even when
    # the nuclei are branched. Resolve promontories and failed-rift recesses
    # in the growth field itself, before adding any coastal detail.
    texture = _spherical_band_noise(
        height, width, seed ^ 0xBB67AE8584CAA73B,
        ((3.5, 0.40, 6), (7.5, 0.35, 6), (15.0, 0.25, 6)),
    ).reshape(-1)
    accretion_cost = 0.16 + np.exp(2.8 * texture)
    fabric_angle = 1.1 * _spherical_band_noise(
        height, width, seed ^ 0xB5AD4ECEDA1CE2A9,
        ((3.0, 0.65, 6), (6.0, 0.35, 6)),
    ).reshape(-1)
    plate_flat = np.asarray(plate_grid).reshape(-1)
    boundary_flat = np.asarray(boundary_class, dtype=np.int8).reshape(-1)
    latitude_weight = np.cos(np.radians(grid.latitude_degrees[:, 0]))
    cell_weight = np.repeat(latitude_weight, width)
    absolute_latitude = np.abs(np.asarray(grid.latitude_degrees).reshape(-1)) / 90.0
    total_area = float(np.sum(cell_weight))
    target_area = total_area * float(target_fraction)

    quota_weights = rng.lognormal(mean=0.0, sigma=0.42, size=len(root_cells))
    quota_weights /= float(np.sum(quota_weights))
    quotas = target_area * quota_weights
    owner_area = np.zeros(len(root_cells), dtype=np.float64)
    owner = np.full(cell_count, -1, dtype=np.int16)
    costs = np.full((len(root_cells), cell_count), np.inf, dtype=np.float32)
    frontiers: list[list[tuple[float, int]]] = [[] for _ in root_cells]
    preferred_angles = rng.uniform(-math.pi, math.pi, size=len(root_cells))
    anisotropy = rng.uniform(0.28, 1.0, size=len(root_cells))

    for owner_index, root in enumerate(root_cells):
        if owner[root] >= 0:
            continue
        owner[root] = owner_index
        costs[owner_index, root] = 0.0
        owner_area[owner_index] += cell_weight[root]
        heapq.heappush(frontiers[owner_index], (0.0, root))

    row_step = math.pi * _PLANET_RADIUS_KM / height
    longitude_steps = (
        math.tau
        * _PLANET_RADIUS_KM
        / width
        * np.cos(np.radians(grid.latitude_degrees[:, 0]))
    )

    def expand(owner_index: int, cell: int, current_cost: float) -> None:
        row, column = divmod(cell, width)
        for delta_row, delta_column in (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ):
            next_row = row + delta_row
            if not 0 <= next_row < height:
                continue
            next_column = (column + delta_column) % width
            neighbour = next_row * width + next_column
            if owner[neighbour] >= 0:
                continue
            physical_x = delta_column * float(longitude_steps[row])
            physical_y = -delta_row * row_step
            step_length = max(math.hypot(physical_x, physical_y), row_step * 0.035)
            edge_angle = math.atan2(physical_y, physical_x)
            cross_axis = math.sin(edge_angle - preferred_angles[owner_index] - float(fabric_angle[neighbour]))
            # A smooth tensor norm has no |sin| cusp that imprints flat facets.
            directional_cost = math.sqrt(1.0 + 2.2 * anisotropy[owner_index] * cross_axis * cross_axis)
            cell_cost = float(accretion_cost[neighbour])
            polar_penalty = (
                1.25
                - 0.82 * max(morphology.latitude_bias, 0.0)
            ) * float(absolute_latitude[neighbour] ** 4)
            cell_cost += polar_penalty
            boundary_value = int(boundary_flat[neighbour])
            if boundary_value == BOUNDARY_DIVERGENT:
                cell_cost += 1.25 + 1.15 * morphology.ocean_basin_openness
            elif boundary_value == BOUNDARY_TRANSFORM:
                cell_cost += 0.28
            elif boundary_value == BOUNDARY_CONVERGENT:
                cell_cost -= 0.12 * morphology.continental_aggregation
            if plate_flat[neighbour] != plate_flat[cell]:
                cell_cost += 0.18 + 0.22 * morphology.ocean_basin_openness
            candidate = current_cost + step_length * directional_cost * max(cell_cost, 0.20)
            if candidate < float(costs[owner_index, neighbour]):
                costs[owner_index, neighbour] = candidate
                heapq.heappush(frontiers[owner_index], (candidate, int(neighbour)))

    # Continental roots are accreted belts of connected crustal nuclei, not
    # single points whose travel-time balls inevitably become oval continents.
    # Asymmetric arms bend with the crustal fabric and avoid spreading seams.
    nucleus_cells: list[list[int]] = [[root] for root in root_cells]
    for owner_index, root in enumerate(root_cells):
        core_radius_km = row_step * math.sqrt(float(quotas[owner_index]))
        arm_length = float(np.clip(0.32 * core_radius_km, 600.0, 2200.0))
        first_arm: list[int] = []
        for arm in range(3):
            cell = root if arm < 2 or not first_arm else first_arm[len(first_arm) // 2]
            heading = preferred_angles[owner_index] + (math.pi if arm == 1 else 0.0)
            if arm == 2:
                heading += float(rng.choice((-1.0, 1.0))) * float(rng.uniform(0.8, 1.4))
            length = arm_length * float(rng.uniform(0.48, 1.25))
            curvature = float(rng.uniform(-0.85, 0.85))
            travelled = 0.0
            while travelled < length:
                row, column = divmod(cell, width)
                direction = heading + curvature * travelled / max(length, 1.0)
                options = []
                for dr, dc in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                    nr, nc = row + dr, (column + dc) % width
                    if not 1 <= nr < height - 1:
                        continue
                    neighbour = nr * width + nc
                    if owner[neighbour] >= 0:
                        continue
                    east, north = dc * float(longitude_steps[row]), -dr * row_step
                    step = max(math.hypot(east, north), row_step * 0.035)
                    alignment = (east * math.cos(direction) + north * math.sin(direction)) / step
                    score = alignment - 0.16 * float(texture[neighbour])
                    if boundary_flat[neighbour] == BOUNDARY_DIVERGENT:
                        score -= 1.2
                    if plate_flat[neighbour] != plate_flat[cell]:
                        score -= 0.25
                    options.append((score, neighbour, step))
                if not options:
                    break
                score, cell, step = max(options)
                if score < 0.18:
                    break
                travelled += step
                owner[cell] = owner_index
                costs[owner_index, cell] = 0.0
                owner_area[owner_index] += cell_weight[cell]
                nucleus_cells[owner_index].append(cell)
                if arm == 0:
                    first_arm.append(cell)
        for nucleus in nucleus_cells[owner_index]:
            expand(owner_index, nucleus, 0.0)

    occupied_area = float(np.sum(cell_weight[owner >= 0]))
    active = set(range(len(root_cells)))
    while occupied_area < target_area and active:
        progressed = False
        for owner_index in tuple(sorted(active)):
            if owner_area[owner_index] >= quotas[owner_index]:
                active.discard(owner_index)
                continue
            frontier = frontiers[owner_index]
            while frontier:
                current_cost, cell = heapq.heappop(frontier)
                if owner[cell] >= 0:
                    continue
                owner[cell] = owner_index
                weight = float(cell_weight[cell])
                owner_area[owner_index] += weight
                occupied_area += weight
                expand(owner_index, cell, current_cost)
                progressed = True
                break
            if not frontier and owner_area[owner_index] < quotas[owner_index]:
                active.discard(owner_index)
            if occupied_area >= target_area:
                break
        if not progressed:
            break
    return owner.reshape(grid.shape) >= 0


def _coastal_rift_field(
    continental_mask: np.ndarray,
    inside_distance_km: np.ndarray,
    seed: int,
    count: int,
) -> tuple[np.ndarray, int]:
    """Trace a few drowned rifts from existing coasts into continental crust."""

    land = np.asarray(continental_mask, dtype=bool)
    height, width = land.shape
    ocean_neighbour = np.zeros(land.shape, dtype=bool)
    for delta_row, delta_column in (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ):
        shifted = np.roll(land, (delta_row, delta_column), axis=(0, 1))
        if delta_row < 0:
            shifted[delta_row:] = True
        elif delta_row > 0:
            shifted[:delta_row] = True
        ocean_neighbour |= ~shifted
    coast = land & ocean_neighbour
    latitude_margin = max(3, int(round(height * 0.08)))
    coast[:latitude_margin] = False
    coast[-latitude_margin:] = False
    candidates = np.flatnonzero(coast.reshape(-1))
    if not len(candidates) or count <= 0:
        return np.zeros(land.shape, dtype=np.float32), 0

    rng = np.random.Generator(np.random.PCG64(seed ^ 0xA54FF53A5F1D36F1))
    selected: list[tuple[int, int]] = []
    rift = np.zeros(land.shape, dtype=bool)
    for _ in range(count):
        scores = rng.random(len(candidates))
        for row, column in selected:
            candidate_rows, candidate_columns = np.divmod(candidates, width)
            column_distance = np.minimum(
                np.abs(candidate_columns - column),
                width - np.abs(candidate_columns - column),
            )
            separation = np.hypot(candidate_rows - row, column_distance)
            scores *= np.clip(separation / max(height * 0.18, 1.0), 0.0, 1.0)
        start = int(candidates[int(np.argmax(scores))])
        row, column = divmod(start, width)
        selected.append((row, column))
        previous_step = (0, 0)
        path_length = int(rng.integers(14, 35))
        for ordinal in range(path_length):
            rift[row, column] = True
            options: list[tuple[float, int, int, int, int]] = []
            for delta_row, delta_column in (
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1),
            ):
                next_row = row + delta_row
                if not 0 <= next_row < height:
                    continue
                next_column = (column + delta_column) % width
                if not land[next_row, next_column] or rift[next_row, next_column]:
                    continue
                continuity = (
                    delta_row * previous_step[0]
                    + delta_column * previous_step[1]
                )
                score = (
                    float(inside_distance_km[next_row, next_column])
                    + 46.0 * continuity
                    + float(rng.uniform(-95.0, 95.0))
                    - 16.0 * ordinal
                )
                options.append((score, next_row, next_column, delta_row, delta_column))
            if not options:
                break
            _, row, column, step_row, step_column = max(options)
            previous_step = (step_row, step_column)
    widened = _dilate(rift, 1).astype(np.float32)
    return _smooth_field(widened, radius=1, passes=2), len(selected)


def _crust_accommodation_footprint(along_km, across_km, length_km, half_width_km, seed):
    """Broad pre-erosion crustal accommodation; not the final shoreline.

    Preserve the accepted world's basal crust and inland orogen placement.
    The final shore is evolved separately on the full-resolution surface.
    """
    rng = np.random.Generator(np.random.PCG64(int(seed) & 0xFFFFFFFFFFFFFFFF))
    t = np.asarray(along_km) / length_km
    bend = float(rng.choice((-1.0, 1.0)))
    frequency = float(rng.uniform(1.1, 1.8))
    axis = half_width_km * bend * (1.10 * np.sin(math.pi * t * frequency) + 0.65 * t * t)
    displacement = np.asarray(across_km) - axis
    tip = np.sqrt(np.clip(1.04 - t, 0.0, 1.0))
    phase_left, phase_right = rng.uniform(-math.pi, math.pi, 2)
    left = half_width_km * tip * (0.88 + 0.26 * np.sin(7.0 * t + phase_left))
    right = half_width_km * tip * (0.94 + 0.30 * np.sin(5.0 * t + phase_right))
    bank = np.where(displacement < 0.0, left, right)
    boundary = np.minimum(bank - np.abs(displacement), (1.04 - t) * length_km)
    boundary = np.minimum(boundary, (t + 1.0) * length_km)
    edge = max(half_width_km * 0.14, 20.0)
    footprint = np.clip((boundary + edge) / (2.0 * edge), 0.0, 1.0)
    footprint = footprint * footprint * (3.0 - 2.0 * footprint)
    footprint[t >= 1.04] = 0.0
    return footprint.astype(np.float32)


def _coastal_margin_field(
    grid: LatLonGrid,
    continental_mask: np.ndarray,
    inside_distance_km: np.ndarray,
    boundary_class: np.ndarray,
    morphology: WorldMorphology,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Carve large bays and build peninsulas from measured crustal margins."""

    land = np.asarray(continental_mask, dtype=bool)
    height, width = land.shape
    distance_outside_km = _distance_from_sources(land, grid)
    coast = land & ~(
        np.roll(land, 1, axis=0)
        & np.roll(land, -1, axis=0)
        & np.roll(land, 1, axis=1)
        & np.roll(land, -1, axis=1)
    )
    coast[0] = False
    coast[-1] = False
    candidates = np.flatnonzero(coast.reshape(-1))
    if not len(candidates):
        return np.zeros(grid.shape, dtype=np.float32), 0

    rng = np.random.Generator(np.random.PCG64(seed ^ 0x510E527FADE682D1))
    candidate_rows, candidate_columns = np.divmod(candidates, width)
    selected: list[tuple[int, int]] = []
    vectors = np.asarray(grid.unit_vectors, dtype=np.float64)
    result = np.zeros(grid.shape, dtype=np.float64)
    active_margin = _dilate(
        np.asarray(boundary_class) != BOUNDARY_INTERIOR,
        4,
    )
    # Margin-scale deformation must survive the final sea-level cut.  A pair
    # of spherical fields corrugates each tectonic embayment or promontory at
    # hundreds-to-thousands of kilometres without turning it into pixel noise.
    margin_shape = _smooth_field(
        _spherical_band_noise(
            height,
            width,
            seed ^ 0xC6BC279692B5CC83,
            ((13.0, 0.42, 5), (23.0, 0.34, 5), (37.0, 0.24, 5)),
        ),
        radius=2,
        passes=1,
    ).astype(np.float64)
    margin_fabric = _spherical_band_noise(
        height,
        width,
        seed ^ 0xD3833E804F4C574B,
        ((31.0, 0.42, 5), (53.0, 0.34, 5), (79.0, 0.24, 4)),
    ).astype(np.float64)
    feature_count = 20 + int(round(22.0 * morphology.crust_fragmentation))
    row_step_km = math.pi * _PLANET_RADIUS_KM / height

    for ordinal in range(feature_count):
        scores = 0.35 + 0.65 * rng.random(len(candidates))
        if selected:
            nearest = np.full(len(candidates), np.inf, dtype=np.float64)
            for selected_row, selected_column in selected:
                column_delta = np.minimum(
                    np.abs(candidate_columns - selected_column),
                    width - np.abs(candidate_columns - selected_column),
                )
                nearest = np.minimum(
                    nearest,
                    np.hypot(candidate_rows - selected_row, column_delta),
                )
            scores *= np.clip(nearest / max(height * 0.13, 1.0), 0.0, 1.0)
        flat = int(candidates[int(np.argmax(scores))])
        row, column = divmod(flat, width)
        selected.append((row, column))

        north = max(0, row - 1)
        south = min(height - 1, row + 1)
        gradient_row = float(
            inside_distance_km[south, column]
            - inside_distance_km[north, column]
        )
        gradient_column = float(
            inside_distance_km[row, (column + 1) % width]
            - inside_distance_km[row, (column - 1) % width]
        )
        latitude = math.radians(float(grid.latitude_degrees[row, column]))
        longitude_step_km = math.tau * _PLANET_RADIUS_KM / width * max(math.cos(latitude), 0.05)
        gradient_north = -gradient_row / (2.0 * row_step_km)
        gradient_east = gradient_column / (2.0 * longitude_step_km)
        gradient_norm = max(math.hypot(gradient_north, gradient_east), 1.0e-9)
        inward_north = gradient_north / gradient_norm
        inward_east = gradient_east / gradient_norm
        is_active = bool(active_margin[row, column])
        family_selector = int(rng.integers(0, 5))
        if is_active and family_selector in (1, 4):
            family = "tectonic-cape"
        elif family_selector in (0, 1):
            family = "embayment"
        elif family_selector == 2:
            family = "rift-inlet"
        else:
            family = "peninsula"
        make_gulf = family in ("embayment", "rift-inlet")
        centre = vectors[row, column]
        east_axis = np.array([-centre[1], centre[0], 0.0])
        east_axis /= np.linalg.norm(east_axis)
        north_axis = np.cross(centre, east_axis)
        forward = vectors @ centre
        east_km = _PLANET_RADIUS_KM * np.arctan2(vectors @ east_axis, forward)
        north_km = _PLANET_RADIUS_KM * np.arctan2(vectors @ north_axis, forward)
        along = east_km * inward_east + north_km * inward_north
        across = -east_km * inward_north + north_km * inward_east
        if family == "embayment":
            major_axis = float(rng.uniform(760.0, 1850.0))
            minor_axis = float(rng.uniform(310.0, 760.0))
            amplitude = -float(rng.uniform(0.88, 1.58))
        elif family == "rift-inlet":
            major_axis = float(rng.uniform(980.0, 2180.0))
            minor_axis = float(rng.uniform(150.0, 390.0))
            amplitude = -float(rng.uniform(0.76, 1.36))
        elif family == "tectonic-cape":
            major_axis = float(rng.uniform(470.0, 980.0))
            minor_axis = float(rng.uniform(180.0, 430.0))
            amplitude = float(rng.uniform(0.42, 0.82))
        else:
            major_axis = float(rng.uniform(650.0, 1460.0))
            minor_axis = float(rng.uniform(190.0, 520.0))
            amplitude = float(rng.uniform(0.50, 1.02))

        footprint = _crust_accommodation_footprint(
            (along if make_gulf else -along) + 95.0 * margin_shape,
            across + minor_axis * (0.30 * margin_shape + 0.16 * margin_fabric),
            major_axis, minor_axis, int(rng.integers(0, 2**63)),
        )
        footprint[forward < 0.0] = 0.0
        if not make_gulf:
            anchor_width_km = 220.0 if family == "tectonic-cape" else 260.0
            footprint *= np.exp(
                -0.5 * np.square(distance_outside_km / anchor_width_km)
            )
        if make_gulf:
            # This is a change to the signed crustal distance, not a constant
            # depression painted over it. Cancel the positive inland distance
            # inside a gulf's core so its intended long inlet can actually
            # cross sea level; the tapered footprint still retains headlands.
            drowning_depth = np.maximum(
                abs(amplitude), inside_distance_km / 680.0 + 0.42,
            )
            result -= drowning_depth * footprint
        else:
            result += amplitude * footprint

    return _smooth_field(result.astype(np.float32), radius=1, passes=1), len(selected)


def _transport_continental_crust(
    grid: LatLonGrid,
    continental_mask: np.ndarray,
    plate_labels: np.ndarray,
    definitions: Sequence[PlateDefinition],
    duration_myr: float,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Transport crustal fragments by their parent plate's Euler rotation.

    This is a finite kinematic reconstruction, not a mantle/convection
    simulation. Splitting occurs at shared plate boundaries before rotation:
    divergent pieces leave matching opposing margins; convergent pieces
    overlap. No independent oval masks are introduced at the new coast.
    """
    vectors = np.asarray(grid.unit_vectors, dtype=np.float64)
    rows, columns = np.indices(grid.shape, dtype=np.float32)
    coverage = np.zeros(grid.shape, dtype=np.float32)
    for definition in definitions:
        fragment = continental_mask & (plate_labels == definition.plate_id)
        if not np.any(fragment):
            continue
        pole = np.asarray(lon_lat_to_unit_vector(definition.euler_pole_lon, definition.euler_pole_lat))
        angle = -math.radians(definition.angular_speed_deg_per_myr * duration_myr)
        source = (
            vectors * math.cos(angle)
            + np.cross(pole, vectors) * math.sin(angle)
            + (vectors @ pole)[..., None] * pole * (1.0 - math.cos(angle))
        )
        longitude, latitude = unit_vector_to_lon_lat(source)
        source_rows = (90.0 - latitude) * grid.shape[0] / 180.0 - 0.5
        source_columns = (longitude + 180.0) * grid.shape[1] / 360.0 - 0.5
        coverage += _warp_periodic_field(fragment.astype(np.float32), source_rows - rows, source_columns - columns)
    moved = coverage >= 0.5
    weight = np.cos(np.radians(grid.latitude_degrees))
    total = float(np.sum(weight))
    old_area = float(np.sum(weight * continental_mask))
    return moved, {
        'model': 'spherical-euler-fragment-transport',
        'durationMyr': float(duration_myr),
        # Vacated area includes translation as well as true rift opening.
        'vacatedAreaFraction': float(np.sum(weight * (continental_mask & ~moved)) / total),
        'overlapAreaFraction': float(np.sum(weight * (coverage > 1.5)) / total),
        'coverageAreaRelativeError': abs(float(np.sum(weight * coverage)) - old_area) / max(old_area, 1.0),
    }


def _continental_crust(
    grid: LatLonGrid,
    plate_fields: PlateFields,
    boundary_class: np.ndarray,
    morphology: WorldMorphology,
    recipe: PlanetRecipe,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, Mapping[str, object]]:
    vectors = np.asarray(grid.unit_vectors, dtype=np.float64).reshape((-1, 3))
    divergent_distance = _distance_from_sources(boundary_class == BOUNDARY_DIVERGENT, grid)
    transport_duration = 32.0 + 28.0 * morphology.crust_fragmentation
    ancestral_definitions = _advect_plate_definitions(plate_fields.plates, transport_duration)
    ancestral_initial = build_plate_fields(
        grid, plate_count=len(ancestral_definitions), anchors=ancestral_definitions,
        stage_seed=seed, radius_km=_PLANET_RADIUS_KM, boundary_corridors=(),
    )
    ancestral_plates = build_plate_fields(
        grid, plate_count=len(ancestral_definitions), anchors=ancestral_definitions,
        stage_seed=seed, radius_km=_PLANET_RADIUS_KM,
        boundary_corridors=_curved_boundary_corridors(ancestral_initial, morphology, seed),
    )
    ancestral_boundaries = _boundary_class_field(ancestral_plates)
    # Accreted crust must be allowed across the future rift. Pre-clearing a
    # wide ocean around every divergent boundary leaves little shared crust
    # to tear apart when the finite motion below actually starts.
    roots = _select_crust_roots(grid, recipe.continent_count, morphology, seed)
    rng = np.random.Generator(np.random.PCG64(seed ^ 0xA24BAED4963EE407))
    crust_fraction = min(recipe.land_fraction + 0.075, 0.62)
    continental_mask = _grow_continental_crust(
        grid,
        roots,
        ancestral_plates.plate_grid,
        ancestral_boundaries,
        morphology,
        crust_fraction,
        seed,
    )

    # Detached continental fragments now come from the same transported
    # crust, not extra radial islands stamped around its perimeter. Oceanic
    # island arcs and hotspot chains are generated by the relief stage.
    continental_mask, transport_metrics = _transport_continental_crust(
        grid, continental_mask, ancestral_plates.plate_grid, ancestral_plates.plates,
        transport_duration,
    )
    inside_distance = _distance_from_sources(~continental_mask, grid)
    outside_distance = _distance_from_sources(continental_mask, grid)
    potential_grid = (inside_distance - outside_distance) / 680.0
    rift_field, coastal_rift_count = _coastal_rift_field(
        continental_mask,
        inside_distance,
        seed,
        8 + int(round(12.0 * morphology.crust_fragmentation)),
    )
    potential_grid -= (
        0.96 + 0.72 * morphology.crust_fragmentation
    ) * rift_field
    margin_field, macro_margin_feature_count = _coastal_margin_field(
        grid,
        continental_mask,
        inside_distance,
        boundary_class,
        morphology,
        seed,
    )
    potential_grid += margin_field
    texture = _spherical_fbm(
        grid.shape[0],
        grid.shape[1],
        seed ^ 0xC13FA9A902A6328F,
        detail=0.55 + morphology.crust_fragmentation,
    )
    plate_ids = tuple(value.plate_id for value in plate_fields.plates)
    rng_plate = np.random.Generator(np.random.PCG64(seed ^ 0x91E10DA5C79E7B1D))
    plate_bias_by_id = {
        plate_id: float(rng_plate.normal(0.0, 0.08)) for plate_id in plate_ids
    }
    plate_bias = np.asarray(
        [plate_bias_by_id[str(value)] for value in plate_fields.plate_grid.reshape(-1)],
        dtype=np.float32,
    ).reshape(grid.shape)
    plate_bias = _smooth_field(plate_bias, 3, passes=3)
    potential_grid += (0.17 + 0.20 * morphology.crust_fragmentation) * texture
    potential_grid += plate_bias

    # Bend the whole crustal probability field before the sea-level cut.
    # Warping a continuous field preserves connected peninsulas and rifted
    # fragments while avoiding circular unions of individual source lobes.
    crust_warp_rows = (1.8 + 2.4 * morphology.crust_fragmentation) * _smooth_field(
        _spherical_fbm(
            grid.shape[0],
            grid.shape[1],
            seed ^ 0xD6E8FEB86659FD93,
            detail=0.30,
        ),
        radius=5,
        passes=2,
    )
    crust_warp_columns = (3.0 + 3.8 * morphology.crust_fragmentation) * _smooth_field(
        _spherical_fbm(
            grid.shape[0],
            grid.shape[1],
            seed ^ 0xA5A3564E27F8862F,
            detail=0.30,
        ),
        radius=5,
        passes=2,
    )
    potential_grid = _warp_periodic_field(
        potential_grid,
        crust_warp_rows,
        crust_warp_columns,
    )
    potential_grid += (0.06 + 0.07 * morphology.crust_fragmentation) * texture

    # Mature spreading centres open ocean basins.  Continental crust may cross
    # an occasional young rift, but it should not sit astride whole ridge
    # systems as if the plate diagram itself were land.
    convergent_distance = _distance_from_sources(
        boundary_class == BOUNDARY_CONVERGENT,
        grid,
    )
    potential_grid -= (0.64 + 0.42 * morphology.ocean_basin_openness) * np.exp(
        -0.5 * np.square(divergent_distance / 390.0)
    )
    potential_grid += (0.07 + 0.10 * morphology.continental_aggregation) * np.exp(
        -0.5 * np.square(convergent_distance / 285.0)
    )

    breakup_strength = math.exp(
        -0.5 * (min(abs(morphology.cycle_position - 0.50), 1.0 - abs(morphology.cycle_position - 0.50)) / 0.15) ** 2
    )
    if breakup_strength > 0.16 and roots:
        reference = vectors[int(roots[0])]
        for _ in range(1 + int(round(2.0 * morphology.crust_fragmentation))):
            normal = _tangent_direction(reference, rng)
            distance = np.arcsin(np.clip(np.abs(vectors @ normal), 0.0, 1.0))
            width = math.radians(float(rng.uniform(2.2, 5.2)))
            potential_grid -= (
                0.42
                * breakup_strength
                * np.exp(-0.5 * np.square(distance / width)).reshape(grid.shape)
            )

    # Reserve one genuine oceanic longitude corridor for the rectangular map
    # seam.  Pick the weakest existing meridian, so this widens a basin rather
    # than slicing an established continental core.
    basin_column = int(np.argmin(np.max(potential_grid, axis=0)))
    columns = np.arange(grid.shape[1], dtype=np.float64)
    basin_distance = np.minimum(
        np.mod(columns - basin_column, grid.shape[1]),
        np.mod(basin_column - columns, grid.shape[1]),
    )
    basin_sigma = 13.0 + 6.0 * morphology.ocean_basin_openness
    potential_grid -= 5.0 * np.exp(-0.5 * np.square(basin_distance / basin_sigma))[None, :]

    # A landmass that reaches the mathematical pole must close around every
    # longitude.  Partial polar strips are projection artefacts, so ordinary
    # seeds keep a polar ocean; only a strongly pole-biased morphology relaxes
    # this cost enough for a genuine cap to form.
    absolute_latitude_grid = np.abs(grid.latitude_degrees) / 90.0
    polar_excess = np.clip((absolute_latitude_grid - 0.76) / 0.24, 0.0, 1.0)
    polar_strength = 3.8 * (1.0 - 0.82 * max(morphology.latitude_bias, 0.0))
    potential_grid -= polar_strength * np.power(polar_excess, 2.2)
    potential_grid = _polar_crust_bias(potential_grid, grid.latitude_degrees[:, 0], recipe)

    latitude_weight = np.cos(np.radians(grid.latitude_degrees[:, 0]))[:, None]
    field = potential_grid.astype(np.float32)
    level = _weighted_level(field, latitude_weight, recipe.land_fraction)
    # Sea-level crust, thermal ocean age and relief must share the same
    # geometry. The uncut accretion mask predates rifting and subsidence;
    # using it here leaves newly opened seas classified as dry continent.
    return field > level, field, float(level), {
        "macroMarginFeatureCount": macro_margin_feature_count,
        "coastalRiftCount": coastal_rift_count,
        "crustTransport": transport_metrics,
    }


def _boundary_class_field(plate_fields: PlateFields) -> np.ndarray:
    result = np.zeros(plate_fields.plate_grid.shape, dtype=np.int8)
    priority = np.zeros(result.shape, dtype=np.int8)
    class_values = {
        "convergent": (BOUNDARY_CONVERGENT, 3),
        "oblique-convergent": (BOUNDARY_CONVERGENT, 3),
        "divergent": (BOUNDARY_DIVERGENT, 2),
        "oblique-divergent": (BOUNDARY_DIVERGENT, 2),
        "transform": (BOUNDARY_TRANSFORM, 1),
        "stable": (BOUNDARY_INTERIOR, 0),
    }
    flat_result = result.reshape(-1)
    flat_priority = priority.reshape(-1)
    for boundary in plate_fields.boundaries:
        value, rank = class_values[boundary.classification]
        for cell in boundary.cell_indices:
            if rank > flat_priority[cell]:
                flat_result[cell] = value
                flat_priority[cell] = rank
    return result


def _distance_from_sources(mask: np.ndarray, grid: LatLonGrid) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if source.shape != grid.shape:
        raise ValueError("source mask must match the reference grid")
    sources = np.flatnonzero(source.reshape(-1))
    if not len(sources):
        raise ValueError("distance source mask is empty")
    distance = np.full(grid.cell_count, np.inf, dtype=np.float64)
    distance[sources] = 0.0
    queue = [(0.0, int(cell)) for cell in sources]
    heapq.heapify(queue)
    row_step = math.pi * _PLANET_RADIUS_KM / grid.shape[0]
    longitude_step = (
        math.tau
        * _PLANET_RADIUS_KM
        / grid.shape[1]
        * np.cos(np.radians(grid.latitude_degrees[:, 0]))
    )
    while queue:
        current_distance, cell = heapq.heappop(queue)
        if current_distance != distance[cell]:
            continue
        row, column = divmod(cell, grid.shape[1])
        for delta_row, delta_column in (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ):
            neighbour_row = row + delta_row
            if not 0 <= neighbour_row < grid.shape[0]:
                continue
            neighbour_column = (column + delta_column) % grid.shape[1]
            neighbour = neighbour_row * grid.shape[1] + neighbour_column
            north_south = row_step * abs(delta_row)
            east_west = float(longitude_step[row]) * abs(delta_column)
            step = math.hypot(north_south, east_west)
            candidate = current_distance + max(step, row_step * 0.035)
            if candidate < distance[neighbour]:
                distance[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return distance.reshape(grid.shape)


def _build_crust_fields(
    grid: LatLonGrid,
    plate_fields: PlateFields,
    continental_mask: np.ndarray,
    boundary_class: np.ndarray,
    morphology: WorldMorphology,
) -> tuple[CrustFields, np.ndarray, np.ndarray, float]:
    distance_to_continent = _distance_from_sources(continental_mask, grid)
    shelf_width = 105.0 + 185.0 * (1.0 - morphology.ocean_basin_openness)
    kind = classify_crust(
        continental_mask,
        ~continental_mask,
        distance_to_continent,
        shelf_width,
    )
    oceanic = kind == CRUST_OCEANIC
    ridge = oceanic & (boundary_class == BOUNDARY_DIVERGENT)
    if not np.any(ridge):
        ridge = oceanic & (boundary_class != BOUNDARY_INTERIOR)
    if not np.any(ridge):
        ridge = oceanic.copy()
    distance_to_ridge = _distance_from_sources(ridge, grid)
    half_spreading_rate = 1.8 + 2.9 * morphology.ocean_basin_openness
    age = compute_ocean_age_myr(distance_to_ridge, half_spreading_rate, oceanic)
    age[oceanic] = np.minimum(age[oceanic], 190.0)
    return CrustFields(kind, age), distance_to_continent, distance_to_ridge, shelf_width


def _hotspot_uplift(
    grid: LatLonGrid,
    plate_fields: PlateFields,
    crust: CrustFields,
    morphology: WorldMorphology,
    seed: int,
) -> np.ndarray:
    if morphology.hotspot_activity < 0.12:
        return np.zeros(grid.shape, dtype=np.float64)
    oceanic = crust.kind.reshape(-1) == CRUST_OCEANIC
    candidates = np.flatnonzero(oceanic)
    if not len(candidates):
        return np.zeros(grid.shape, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    vectors = np.asarray(grid.unit_vectors, dtype=np.float64).reshape((-1, 3))
    velocities = np.asarray(
        plate_fields.velocity_vectors_cm_per_year,
        dtype=np.float64,
    ).reshape((-1, 3))
    uplift = np.zeros(grid.cell_count, dtype=np.float64)
    candidate_weight = np.cos(np.radians(grid.latitude_degrees.reshape(-1)[candidates]))
    candidate_weight /= np.sum(candidate_weight)
    chain_count = 1 + int(round(3.0 * morphology.hotspot_activity))
    for _ in range(chain_count):
        start_index = int(rng.choice(candidates, p=candidate_weight))
        start = vectors[start_index]
        direction = velocities[start_index].copy()
        direction -= float(np.dot(direction, start)) * start
        if np.linalg.norm(direction) <= 1.0e-9:
            direction = _tangent_direction(start, rng)
        else:
            direction /= np.linalg.norm(direction)
        island_count = 5 + int(round(7.0 * morphology.hotspot_activity))
        for ordinal in range(island_count):
            age = ordinal / max(island_count - 1, 1)
            angle = math.radians(ordinal * float(rng.uniform(1.4, 2.3)))
            centre = math.cos(angle) * start - math.sin(angle) * direction
            centre /= np.linalg.norm(centre)
            radius = math.radians(0.75 + 0.95 * (1.0 - age))
            separation = np.arccos(np.clip(vectors @ centre, -1.0, 1.0))
            amplitude = (3550.0 + 1150.0 * morphology.hotspot_activity) * (1.0 - 0.58 * age)
            # These profiles are complete edifice heights above the floor,
            # not successive lava thicknesses. Summing overlapping age
            # samples built 17 km piles and spurious polar continents.
            uplift = np.maximum(uplift, amplitude * np.exp(-0.5 * np.square(separation / radius)))
    uplift[~oceanic] = 0.0
    return uplift.reshape(grid.shape)


def _warp_periodic_field(
    values: np.ndarray,
    row_offset: np.ndarray,
    column_offset: np.ndarray,
) -> np.ndarray:
    """Sample a latitude-clamped, longitude-periodic field at displaced cells."""

    source = np.asarray(values, dtype=np.float32)
    if source.shape != np.asarray(row_offset).shape or source.shape != np.asarray(column_offset).shape:
        raise ValueError("warp offsets must match the source field")
    height, width = source.shape
    rows, columns = np.indices(source.shape, dtype=np.float32)
    sample_rows = np.clip(rows + row_offset, 0.0, height - 1.0)
    sample_columns = np.mod(columns + column_offset, width)
    row0 = np.floor(sample_rows).astype(np.int32)
    row1 = np.minimum(row0 + 1, height - 1)
    column_floor = np.floor(sample_columns)
    column0 = column_floor.astype(np.int32) % width
    column1 = (column0 + 1) % width
    row_fraction = sample_rows - row0
    column_fraction = sample_columns - column_floor
    top = source[row0, column0] * (1.0 - column_fraction) + source[row0, column1] * column_fraction
    bottom = source[row1, column0] * (1.0 - column_fraction) + source[row1, column1] * column_fraction
    return (top * (1.0 - row_fraction) + bottom * row_fraction).astype(np.float32)


def _continental_relief_provinces(
    grid: LatLonGrid,
    continental_mask: np.ndarray,
    boundary_class: np.ndarray,
    land_affinity: np.ndarray,
    seed: int,
    province_count: int,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    """Build morphologically distinct continental-interior landform provinces.

    These are lithospheric provinces rather than pixel noise: centres are
    selected in old plate interiors and kept apart on the sphere.  Plateaus use
    irregular, flat-topped blocks with escarpments; basins use broad subdued
    floors; plains suppress inherited relief; shields remain rolling multi-lobed
    uplands. Active margins remain responsible for the young, highest ranges.
    """

    active = np.asarray(boundary_class) != BOUNDARY_INTERIOR
    distance_to_active = _distance_from_sources(active, grid)
    distance_to_margin = _distance_from_sources(
        ~np.asarray(continental_mask, dtype=bool), grid
    )
    candidates = np.flatnonzero(
        np.asarray(continental_mask, dtype=bool).reshape(-1)
        & (distance_to_active.reshape(-1) >= 340.0)
        & (distance_to_margin.reshape(-1) >= 190.0)
        & (np.abs(grid.latitude_degrees).reshape(-1) <= 74.0)
    )
    if not len(candidates):
        return (
            np.zeros(grid.shape, dtype=np.float64),
            np.zeros(grid.shape, dtype=np.float64),
            {
            "count": 0,
            "shieldCount": 0,
            "plateauCount": 0,
            "basinCount": 0,
            "plainCount": 0,
            "landformFamilyCount": 0,
            "plateauHeightRangeMeters": 0.0,
            "basinDepthRangeMeters": 0.0,
            },
        )

    rng = np.random.Generator(np.random.PCG64(seed ^ 0x3C6EF372FE94F82B))
    vectors = np.asarray(grid.unit_vectors, dtype=np.float64).reshape((-1, 3))
    candidate_vectors = vectors[candidates]
    selected: list[int] = []
    nearest_separation = np.full(len(candidates), 2.0, dtype=np.float64)
    result = np.zeros(grid.cell_count, dtype=np.float64)
    plain_support = np.zeros(grid.cell_count, dtype=np.float64)
    shape_texture = 2.0 * _normalise(
        _smooth_field(
            _spherical_fbm(
                grid.shape[0],
                grid.shape[1],
                seed ^ 0xA54FF53A5F1D36F1,
                detail=0.52,
            ),
            radius=3,
            passes=2,
        )
    ).reshape(-1) - 1.0
    floor_texture = _normalise(
        _smooth_field(
            _spherical_fbm(
                grid.shape[0],
                grid.shape[1],
                seed ^ 0x243F6A8885A308D3,
                detail=0.24,
            ),
            radius=7,
            passes=3,
        )
    ).reshape(-1)
    type_counts = {"shield": 0, "plateau": 0, "basin": 0, "plain": 0}
    plateau_heights: list[float] = []
    basin_depths: list[float] = []
    family_order = np.asarray(("plateau", "plain", "shield", "basin"), dtype=object)
    rng.shuffle(family_order)
    target = min(max(8, int(province_count)), len(candidates))
    for ordinal in range(target):
        interior_score = np.clip(
            distance_to_margin.reshape(-1)[candidates] / 1200.0,
            0.0,
            1.0,
        )
        if selected:
            similarity = candidate_vectors @ vectors[np.asarray(selected)].T
            nearest_separation = np.minimum(
                nearest_separation,
                1.0 - np.max(similarity, axis=1),
            )
        score = (
            0.62 * nearest_separation
            + 0.25 * interior_score
            + 0.08 * (0.5 + 0.5 * shape_texture[candidates])
            + 0.05 * rng.random(len(candidates))
        )
        if selected:
            score[
                np.max(
                    candidate_vectors @ vectors[np.asarray(selected)].T,
                    axis=1,
                )
                > math.cos(math.radians(6.5))
            ] = -math.inf
        centre_index = int(candidates[int(np.argmax(score))])
        selected.append(centre_index)
        centre = vectors[centre_index]
        axis_a = _tangent_direction(centre, rng)
        axis_b = np.cross(centre, axis_a)
        axis_b /= np.linalg.norm(axis_b)
        # atan2 retains the centre-facing component.  arcsin(dot(tangent))
        # aliases every local feature at the antipode and used to create
        # unexplained matching plateaus/basins on the far side of the planet.
        facing = vectors @ centre
        along_a = np.arctan2(vectors @ axis_a, facing) * _PLANET_RADIUS_KM
        along_b = np.arctan2(vectors @ axis_b, facing) * _PLANET_RADIUS_KM

        kind = str(family_order[ordinal % len(family_order)])
        family_ordinal = type_counts[kind]
        if kind == "plateau":
            height_bands = ((720.0, 980.0), (1350.0, 1720.0), (2050.0, 2650.0))
            low, high = height_bands[family_ordinal % len(height_bands)]
            amplitude = float(rng.uniform(low, high))
            major_axis = float(rng.uniform(720.0, 1880.0))
            minor_axis = float(rng.uniform(360.0, 1080.0))
            plateau_heights.append(amplitude)
        elif kind == "basin":
            depth_bands = ((260.0, 430.0), (690.0, 940.0), (1120.0, 1540.0))
            low, high = depth_bands[family_ordinal % len(depth_bands)]
            amplitude = -float(rng.uniform(low, high))
            major_axis = float(rng.uniform(820.0, 2060.0))
            minor_axis = float(rng.uniform(460.0, 1260.0))
            basin_depths.append(abs(amplitude))
        elif kind == "plain":
            amplitude = float(rng.uniform(-210.0, 130.0))
            major_axis = float(rng.uniform(980.0, 2350.0))
            minor_axis = float(rng.uniform(520.0, 1420.0))
        else:
            amplitude = float(rng.uniform(380.0, 920.0))
            major_axis = float(rng.uniform(880.0, 1940.0))
            minor_axis = float(rng.uniform(470.0, 1180.0))
        type_counts[kind] += 1
        exponent = float(rng.uniform(1.55, 2.35))
        radius = np.power(
            np.power(np.abs(along_a / major_axis), exponent)
            + np.power(np.abs(along_b / minor_axis), exponent),
            1.0 / exponent,
        )
        angle = np.arctan2(along_b / minor_axis, along_a / major_axis)
        harmonic = int(rng.integers(2, 6))
        phase = float(rng.uniform(-math.pi, math.pi))
        boundary = (
            1.0
            - radius
            + 0.18 * shape_texture
            + 0.075 * np.sin(harmonic * angle + phase)
            + 0.045 * np.sin((harmonic + 3) * angle - 0.7 * phase)
        )
        transition = float(rng.uniform(0.16, 0.27))
        footprint = np.clip(
            (boundary + transition) / (2.0 * transition),
            0.0,
            1.0,
        )
        footprint = footprint * footprint * (3.0 - 2.0 * footprint)

        if kind == "plateau":
            flat_top = 0.88 + 0.12 * floor_texture
            escarpment = np.exp(-0.5 * np.square(boundary / 0.13))
            result += amplitude * footprint * flat_top
            result += (0.08 * amplitude) * escarpment
            plain_support = np.maximum(
                plain_support, 0.78 * footprint * (1.0 - escarpment),
            )
        elif kind == "basin":
            floor = 0.86 + 0.14 * floor_texture
            rim = np.exp(-0.5 * np.square(boundary / 0.18))
            result += amplitude * footprint * floor
            result += (115.0 + 0.10 * abs(amplitude)) * rim
            plain_support = np.maximum(
                plain_support, 0.95 * footprint * (1.0 - rim),
            )
        elif kind == "plain":
            plain_support = np.maximum(plain_support, footprint)
            result += amplitude * footprint * (0.92 + 0.08 * floor_texture)
        else:
            rolling = np.exp(-0.5 * np.square(radius))
            result += amplitude * rolling * (0.58 + 0.42 * floor_texture)

    result = result.reshape(grid.shape)
    result *= np.power(np.clip(land_affinity, 0.0, 1.0), 1.45)
    plain_support = plain_support.reshape(grid.shape)
    plain_support *= np.power(np.clip(land_affinity, 0.0, 1.0), 1.55)
    return result, plain_support, {
        "count": len(selected),
        "shieldCount": type_counts["shield"],
        "plateauCount": type_counts["plateau"],
        "basinCount": type_counts["basin"],
        "plainCount": type_counts["plain"],
        "landformFamilyCount": sum(value > 0 for value in type_counts.values()),
        "plateauHeightRangeMeters": float(
            max(plateau_heights) - min(plateau_heights)
            if len(plateau_heights) > 1 else 0.0
        ),
        "basinDepthRangeMeters": float(
            max(basin_depths) - min(basin_depths)
            if len(basin_depths) > 1 else 0.0
        ),
        "shapeModel": "warped-flat-top-escarpment-basin-plain-shield",
    }


def _reactivated_suture_sources(
    grid: LatLonGrid,
    terrane_fields: PlateFields,
    continental_mask: np.ndarray,
    seed: int,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, object]]:
    """Select inherited orogens and assign distinct erosional-age profiles."""

    chains = merge_boundary_segments(terrane_fields.boundaries, _PLANET_RADIUS_KM)
    segment_lookup = {
        segment.boundary_id: segment for segment in terrane_fields.boundaries
    }
    continental_flat = np.asarray(continental_mask, dtype=bool).reshape(-1)
    vectors = np.asarray(grid.unit_vectors, dtype=np.float64).reshape((-1, 3))
    rng = np.random.Generator(np.random.PCG64(seed ^ 0x452821E638D01377))
    candidates: list[tuple[object, np.ndarray, np.ndarray, float, float]] = []
    for chain in chains:
        source_cells = np.unique(
            np.fromiter(
                (
                    cell
                    for source_id in chain.source_segment_ids
                    for cell in segment_lookup[source_id].cell_indices
                ),
                dtype=np.int64,
            )
        )
        land_cells = source_cells[continental_flat[source_cells]]
        if len(land_cells) < 10 or chain.length_km < 720.0:
            continue
        centroid = np.mean(vectors[land_cells], axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= 1.0e-9:
            continue
        centroid /= centroid_norm
        continental_overlap = float(len(land_cells) / max(len(source_cells), 1))
        base_score = (
            0.48 * min(math.log1p(len(land_cells)) / math.log(180.0), 1.0)
            + 0.27 * min(chain.length_km / 5200.0, 1.0)
            + 0.16 * continental_overlap
            + 0.09 * float(rng.random())
        )
        candidates.append(
            (chain, source_cells, centroid, base_score, continental_overlap)
        )

    target_count = min(max(18, terrane_fields.plate_count // 2 + 3), len(candidates))
    selected: list[int] = []
    while len(selected) < target_count:
        best_index = -1
        best_score = -math.inf
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in selected:
                continue
            score = float(candidate[3])
            if selected:
                nearest_similarity = max(
                    float(candidate[2] @ candidates[index][2])
                    for index in selected
                )
                separation_km = math.acos(
                    float(np.clip(nearest_similarity, -1.0, 1.0))
                ) * _PLANET_RADIUS_KM
                score += 0.52 * min(separation_km / 4200.0, 1.0)
            else:
                score += 0.52
            if score > best_score:
                best_score = score
                best_index = candidate_index
        if best_index < 0:
            break
        selected.append(best_index)

    profile_names = ("young-fold", "mature-fold", "old-rounded")
    profile_sources = {
        name: np.zeros(grid.cell_count, dtype=bool) for name in profile_names
    }
    profile_counts = {name: 0 for name in profile_names}
    profile_offset = int(rng.integers(0, len(profile_names)))
    selected_lengths: list[float] = []
    selected_overlaps: list[float] = []
    for selected_ordinal, selected_index in enumerate(selected):
        chain, _source_cells, _centroid, _score, overlap = candidates[selected_index]
        segment_count = len(chain.source_segment_ids)
        chain_rng = np.random.Generator(
            np.random.PCG64(seed ^ int(chain.chain_id[-16:], 16))
        )
        land_positions = [
            position
            for position, source_id in enumerate(chain.source_segment_ids)
            if any(
                continental_flat[cell]
                for cell in segment_lookup[source_id].cell_indices
            )
        ]
        centre = int(chain_rng.choice(land_positions))
        target_length = float(chain_rng.uniform(1400.0, 3200.0))
        arc_positions = {centre}
        arc_length = float(
            segment_lookup[chain.source_segment_ids[centre]].length_km
        )
        left = centre - 1
        right = centre + 1
        choose_left = bool(chain_rng.integers(0, 2))
        while arc_length < target_length and len(arc_positions) < segment_count:
            if chain.closed:
                position = (left if choose_left else right) % segment_count
            else:
                available_left = left >= 0
                available_right = right < segment_count
                if not available_left and not available_right:
                    break
                position = left if choose_left and available_left else right
                if position >= segment_count:
                    position = left
            if position not in arc_positions:
                arc_positions.add(position)
                arc_length += float(
                    segment_lookup[
                        chain.source_segment_ids[position]
                    ].length_km
                )
            if choose_left:
                left -= 1
            else:
                right += 1
            choose_left = not choose_left
        arc_source_cells = np.unique(
            np.fromiter(
                (
                    cell
                    for position in sorted(arc_positions)
                    for cell in segment_lookup[
                        chain.source_segment_ids[position]
                    ].cell_indices
                ),
                dtype=np.int64,
            )
        )
        profile_name = profile_names[
            (selected_ordinal + profile_offset) % len(profile_names)
        ]
        profile_sources[profile_name][arc_source_cells] = True
        profile_counts[profile_name] += 1
        selected_lengths.append(arc_length)
        selected_overlaps.append(float(overlap))
    profile_widths_km = {
        "young-fold": 150.0,
        "mature-fold": 295.0,
        "old-rounded": 560.0,
    }
    return {
        name: values.reshape(grid.shape)
        for name, values in profile_sources.items()
    }, {
        "candidateChainCount": len(candidates),
        "selectedChainCount": len(selected),
        "selectedArcLengthKm": float(sum(selected_lengths)),
        "medianArcLengthKm": float(
            np.median(selected_lengths) if selected_lengths else 0.0
        ),
        "medianContinentalOverlap": float(
            np.median(selected_overlaps) if selected_overlaps else 0.0
        ),
        "profileFamilyCount": sum(value > 0 for value in profile_counts.values()),
        "profileChainCounts": profile_counts,
        "profileWidthsKm": profile_widths_km,
        "profileWidthRangeKm": float(
            max(profile_widths_km.values()) - min(profile_widths_km.values())
        ),
    }


def _crust_base_relief(continentality: np.ndarray) -> np.ndarray:
    # Saturate the inland base to avoid radial continental domes, but do not
    # flatten a rifted ocean basin back to a 760 m shelf. Its subsidence must
    # survive later uplift and sea-level selection, including the map seam.
    continentality = np.asarray(continentality, dtype=np.float64)
    return (
        760.0 * np.tanh(continentality / 0.34)
        # Zero extra slope at sea level retains fine coastal relief; the
        # continental slope steepens offshore before reaching the abyss.
        - 5000.0 * (1.0 - np.exp(-0.5 * np.square(np.minimum(continentality, 0.0) / 1.15)))
    )


def _reference_relief(
    grid: LatLonGrid,
    plate_fields: PlateFields,
    crust: CrustFields,
    continental_mask: np.ndarray,
    crust_potential: np.ndarray,
    crust_level: float,
    boundary_class: np.ndarray,
    ancient_terrane_fields: PlateFields,
    paleo_boundaries: Sequence[tuple[float, np.ndarray]],
    distance_to_continent: np.ndarray,
    shelf_width: float,
    morphology: WorldMorphology,
    recipe: PlanetRecipe,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    chains = merge_boundary_segments(plate_fields.boundaries, _PLANET_RADIUS_KM)
    chains = smooth_boundary_chains(
        chains,
        _PLANET_RADIUS_KM,
        smoothing_passes=3,
        max_displacement_km=105.0,
    )
    activity = 0.72 + 0.72 * recipe.tectonic_activity
    mountain = 0.76 + 0.72 * recipe.mountain_density
    config = TectonicProfileConfig(
        distance_scale_km=235.0,
        reference_speed_cm_per_year=1.0,
        max_speed_multiplier=3.2,
        ridge_uplift_m=1050.0 * activity,
        rift_shoulder_uplift_m=720.0 * activity,
        trench_depth_m=2450.0 * activity,
        arc_uplift_m=4200.0 * activity * (0.55 + 0.85 * morphology.island_arc_activity),
        collision_uplift_m=3850.0 * activity * mountain,
        basin_depth_m=620.0 * activity,
        transform_valley_depth_m=280.0 * activity,
    )
    potential = build_tectonic_potential(
        grid,
        plate_grid=plate_fields.plate_grid,
        boundaries=plate_fields.boundaries,
        smoothed_chains=chains,
        crust_fields=crust,
        plate_definitions={value.plate_id: value for value in plate_fields.plates},
        radius_km=_PLANET_RADIUS_KM,
        profile_config=config,
        gaussian_cutoff_sigma=3.25,
    )

    texture = _spherical_fbm(grid.shape[0], grid.shape[1], seed, detail=0.55)
    continental = crust.kind == CRUST_CONTINENTAL
    transitional = crust.kind == CRUST_TRANSITIONAL
    oceanic = crust.kind == CRUST_OCEANIC

    # The plate/crust mask is causal data, not a visible cut-out.  Turn the
    # continuous continental potential into relief so the shoreline crosses
    # plate cells organically instead of tracing their raster outline.
    continentality = np.asarray(crust_potential, dtype=np.float64) - crust_level
    transition_width = 0.12 + 0.05 * morphology.crust_fragmentation
    land_affinity = 1.0 / (1.0 + np.exp(-continentality / transition_width))
    # Continentality locates the shoreline; it must not turn every continent
    # into a radial dome whose centre is automatically a high plateau.  Once
    # inland, the base surface saturates into subdued cratonic relief and the
    # tectonic histories below become the actual source of major high ground.
    cratonic_texture = _smooth_field(
        _spherical_fbm(
            grid.shape[0],
            grid.shape[1],
            seed ^ 0x1F83D9ABFB41BD6B,
            detail=0.30,
        ),
        radius=3,
        passes=2,
    )
    signed = (
        _crust_base_relief(continentality)
        + 240.0 * texture
        + 330.0 * cratonic_texture * np.power(land_affinity, 1.25)
    )
    (
        province_relief,
        continental_plain_support,
        province_diagnostics,
    ) = _continental_relief_provinces(
        grid,
        continental_mask,
        boundary_class,
        land_affinity,
        seed,
        22 + 2 * recipe.continent_count + recipe.plate_count // 2,
    )
    # Plains are depositional/erosional surfaces, not low Gaussian hills.  They
    # subdue only the old cratonic fabric; later sutures and active ranges can
    # still cross them and create foreland relief.
    signed *= 1.0 - 0.50 * continental_plain_support
    signed += province_relief

    # Modern plates are too few to explain the interior fabric of old
    # continents.  A denser, subordinate terrane mosaic represents accreted
    # crustal blocks whose sutures survive as broken old ranges, escarpments
    # and basin rims.  It does not alter present plate ownership.
    ancient_terrane_boundary = _boundary_class_field(ancient_terrane_fields)
    terrane_suture = np.asarray(ancient_terrane_boundary) != BOUNDARY_INTERIOR
    terrane_distance = _distance_from_sources(terrane_suture, grid)
    terrane_corridor = np.exp(
        -0.5 * np.square(terrane_distance / 128.0)
    )
    terrane_texture = _normalise(
        _smooth_field(
            _spherical_fbm(
                grid.shape[0],
                grid.shape[1],
                seed ^ 0x9E3779B97F4A7C15,
                detail=0.82,
            ),
            radius=2,
            passes=1,
        )
    )
    terrane_modulation = np.power(
        np.clip((terrane_texture - 0.38) / 0.62, 0.0, 1.0),
        1.22,
    )
    old_suture_uplift = (
        (650.0 + 850.0 * recipe.mountain_density)
        * terrane_corridor
        * (0.24 + 0.76 * terrane_modulation)
        * np.power(land_affinity, 1.48)
    )
    signed += old_suture_uplift

    # Some complete inherited sutures are compressed again during later plate
    # cycles. Selecting whole chains avoids the equal-strength Voronoi web that
    # results from raising every terrane boundary at once.
    reactivated_profiles, reactivated_diagnostics = _reactivated_suture_sources(
        grid,
        ancient_terrane_fields,
        continental_mask,
        seed,
    )
    reactivated_union = np.zeros(grid.shape, dtype=bool)
    profile_specs = (
        (
            "young-fold", 150.0, 540.0,
            2050.0 + 2200.0 * recipe.mountain_density,
            610.0 + 650.0 * recipe.mountain_density,
            ((41.0, 0.34, 5), (67.0, 0.28, 5), (107.0, 0.23, 5), (163.0, 0.15, 4)),
            1.48,
        ),
        (
            "mature-fold", 295.0, 780.0,
            1210.0 + 1390.0 * recipe.mountain_density,
            740.0 + 690.0 * recipe.mountain_density,
            ((23.0, 0.38, 5), (43.0, 0.31, 5), (73.0, 0.21, 5), (113.0, 0.10, 4)),
            1.20,
        ),
        (
            "old-rounded", 560.0, 1160.0,
            360.0 + 540.0 * recipe.mountain_density,
            590.0 + 440.0 * recipe.mountain_density,
            ((11.0, 0.45, 5), (21.0, 0.32, 5), (37.0, 0.16, 5), (59.0, 0.07, 4)),
            0.86,
        ),
    )
    for profile_ordinal, (
        profile_name,
        spine_width_km,
        shoulder_width_km,
        spine_amplitude_m,
        shoulder_amplitude_m,
        fold_bands,
        crest_exponent,
    ) in enumerate(profile_specs):
        sources = np.asarray(reactivated_profiles[profile_name], dtype=bool)
        reactivated_union |= sources
        if not np.any(sources):
            continue
        profile_distance = _distance_from_sources(sources, grid)
        profile_texture = _normalise(
            _smooth_field(
                _spherical_fbm(
                    grid.shape[0],
                    grid.shape[1],
                    seed ^ (0x243F6A88 + 0x9E3779B9 * profile_ordinal),
                    detail=0.22 + 0.22 * profile_ordinal,
                ),
                radius=3 + 3 * profile_ordinal,
                passes=2,
            )
        )
        fold_field = _spherical_band_noise(
            grid.shape[0],
            grid.shape[1],
            seed ^ (0x13198A2E + 0x85EBCA77 * profile_ordinal),
            fold_bands,
        )
        fold_crests = np.power(
            _normalise(1.0 - np.abs(fold_field)),
            crest_exponent,
        )
        spine = np.exp(-0.5 * np.square(profile_distance / spine_width_km))
        shoulder = np.exp(-0.5 * np.square(profile_distance / shoulder_width_km))
        # Mountain systems keep a broad, low shoulder but their high crests
        # occur as offset subranges separated by passes.  This mirrors real
        # orogens more closely than one equally high Gaussian tube.
        segment_threshold = 0.43 - 0.06 * profile_ordinal
        segment_gate = np.power(
            np.clip(
                (profile_texture - segment_threshold)
                / max(0.93 - segment_threshold, 0.1),
                0.0,
                1.0,
            ),
            1.65 - 0.30 * profile_ordinal,
        )
        offset_km = 175.0 + 95.0 * profile_ordinal
        offset_width_km = 92.0 + 105.0 * profile_ordinal
        subranges = np.exp(
            -0.5 * np.square((profile_distance - offset_km) / offset_width_km)
        )
        crest_corridor = 0.40 * spine + 0.60 * subranges
        crest_strength = (
            0.24 + 0.98 * fold_crests
            if profile_name == "young-fold"
            else 0.42 + 0.62 * fold_crests
            if profile_name == "mature-fold"
            else 0.72 + 0.20 * fold_crests
        )
        signed += (
            np.power(land_affinity, 1.68)
            * (
                shoulder_amplitude_m
                * shoulder
                * (0.48 + 0.52 * np.power(profile_texture, 0.75))
                + spine_amplitude_m
                * crest_corridor
                * crest_strength
                * (0.08 + 0.92 * segment_gate)
            )
        )

    terrane_rift = np.asarray(ancient_terrane_boundary) == BOUNDARY_DIVERGENT
    if np.any(terrane_rift):
        terrane_rift_distance = _distance_from_sources(terrane_rift, grid)
        signed -= (
            210.0
            * np.exp(-0.5 * np.square(terrane_rift_distance / 115.0))
            * np.power(land_affinity, 1.65)
            * (0.35 + 0.65 * terrane_modulation)
        )

    # Ocean-floor age only deepens established oceanic basins.  Near shelves
    # the influence fades continuously, avoiding a second artificial coast.
    ocean_depth = np.zeros(grid.shape, dtype=np.float64)
    ocean_depth[oceanic] = np.asarray(
        ocean_depth_from_age(
            crust.ocean_age_myr[oceanic],
            -2250.0,
            -5750.0,
            62.0,
        ),
        dtype=np.float64,
    )
    shelf_progress = np.clip(distance_to_continent / max(shelf_width * 2.4, 1.0), 0.0, 1.0)
    ocean_blend = np.clip((0.45 - land_affinity) / 0.45, 0.0, 1.0)
    ocean_target = ocean_depth * (0.34 + 0.66 * shelf_progress)
    signed = signed * (1.0 - 0.22 * ocean_blend) + ocean_target * (0.22 * ocean_blend)

    # Convergence may build exposed ranges; spreading ridges generally remain
    # submarine.  This suppresses the previous luminous plate-diagram bands
    # while retaining trenches, continental collisions and occasional arcs.
    warp_rows = 7.2 * _smooth_field(
        _spherical_fbm(grid.shape[0], grid.shape[1], seed ^ 0x8CB92BA72F3D8DD7, detail=0.88),
        radius=1,
        passes=1,
    )
    warp_columns = 10.8 * _smooth_field(
        _spherical_fbm(grid.shape[0], grid.shape[1], seed ^ 0x4CF5AD432745937F, detail=0.88),
        radius=1,
        passes=1,
    )
    tectonic_delta = _warp_periodic_field(
        potential.elevation_delta_m,
        warp_rows,
        warp_columns,
    ).astype(np.float64)
    positive_delta = _smooth_field(
        np.maximum(tectonic_delta, 0.0).astype(np.float32),
        radius=2,
        passes=1,
    ).astype(np.float64)
    tectonic_uplift_ceiling_m = 5650.0 + 950.0 * recipe.tectonic_activity
    positive_delta = tectonic_uplift_ceiling_m * np.tanh(
        positive_delta / tectonic_uplift_ceiling_m
    )
    active_segment_field = _normalise(
        _smooth_field(
            _spherical_fbm(
                grid.shape[0],
                grid.shape[1],
                seed ^ 0x94D049BB133111EB,
                detail=0.34,
            ),
            radius=6,
            passes=2,
        )
    )
    active_segment_gate = np.power(
        np.clip((active_segment_field - 0.40) / 0.52, 0.0, 1.0),
        1.65,
    )
    fold_field = _spherical_band_noise(
        grid.shape[0],
        grid.shape[1],
        seed ^ 0xD1B54A32D192ED03,
        (
            (31.0, 0.34, 5),
            (53.0, 0.28, 5),
            (83.0, 0.22, 5),
            (127.0, 0.16, 4),
        ),
    )
    fold_crests = _normalise(1.0 - np.abs(fold_field))
    broad_positive_delta = _smooth_field(
        positive_delta.astype(np.float32),
        radius=6,
        passes=2,
    ).astype(np.float64)
    positive_delta = (
        0.34 * broad_positive_delta
        + 0.66
        * positive_delta
        * (0.06 + 0.94 * active_segment_gate)
        * (0.20 + 0.92 * np.power(fold_crests, 1.35))
    )
    tectonic_subsidence_ceiling_m = 6200.0
    negative_delta = -tectonic_subsidence_ceiling_m * np.tanh(
        np.maximum(-tectonic_delta, 0.0) / tectonic_subsidence_ceiling_m
    )
    positive_weight = 0.035 + 0.965 * np.power(land_affinity, 1.65)
    negative_weight = 0.32 + 0.68 * (1.0 - land_affinity)
    signed += positive_delta * positive_weight + negative_delta * negative_weight

    # A world-scale map needs more than the currently active ranges.  Advected
    # plate anchors provide an earlier kinematic snapshot; its convergent seams
    # survive as lower, broader and more discontinuous palaeo-orogens.
    paleo_fraction: dict[str, float] = {}
    for snapshot_ordinal, (lookback_myr, paleo_boundary_class) in enumerate(
        paleo_boundaries
    ):
        paleo_convergent = (
            np.asarray(paleo_boundary_class) == BOUNDARY_CONVERGENT
        )
        paleo_fraction[str(int(round(lookback_myr)))] = float(
            np.mean(paleo_convergent)
        )
        if not np.any(paleo_convergent):
            continue
        paleo_distance = _distance_from_sources(paleo_convergent, grid)
        corridor_width = 260.0 + 1.15 * lookback_myr
        paleo_corridor = np.exp(
            -0.5 * np.square(paleo_distance / corridor_width)
        )
        paleo_texture = _normalise(
            _smooth_field(
                np.roll(texture, 43 + 47 * snapshot_ordinal, axis=1),
                radius=3 + snapshot_ordinal,
                passes=2,
            )
        )
        paleo_modulation = 0.12 + 0.88 * np.power(
            paleo_texture,
            1.75 + 0.18 * snapshot_ordinal,
        )
        # Young sutures retain a narrow, recognisable spine; older ones erode
        # into lower, wider uplands.  Sampling several geological epochs below
        # therefore adds continental-scale variety without inventing isolated
        # mountains unrelated to plate history.
        age_decay = math.exp(-lookback_myr / 285.0)
        history_weight = 1.0 / (1.0 + 0.10 * snapshot_ordinal)
        signed += (
            (900.0 + 1425.0 * morphology.continental_aggregation)
            * age_decay
            * history_weight
            * paleo_corridor
            * paleo_modulation
            * np.power(land_affinity, 1.35)
        )

    # Ocean-ocean convergence exposes only intermittent volcanic summits.  A
    # modulated corridor creates an island arc rather than another continuous
    # plate-coloured ribbon.
    arc_seed = oceanic & (boundary_class == BOUNDARY_CONVERGENT)
    if np.any(arc_seed):
        arc_corridor = _smooth_field(_dilate(arc_seed, 3).astype(np.float32), 2, passes=1)
        arc_signal = _normalise(
            _spherical_fbm(
                grid.shape[0],
                grid.shape[1],
                seed ^ 0x5BE0CD19137E2179,
                detail=1.18,
            )
        )
        arc_modulation = np.power(
            np.clip((arc_signal - 0.69) / 0.31, 0.0, 1.0),
            2.5,
        )
        signed += (
            positive_delta
            * arc_corridor
            * arc_modulation
            * (1.0 - land_affinity)
            * (0.12 + 0.24 * morphology.island_arc_activity)
        )
        signed += (
            (2350.0 + 3050.0 * morphology.island_arc_activity)
            * np.power(np.clip(arc_corridor, 0.0, 1.0), 1.4)
            * arc_modulation
            * np.power(1.0 - land_affinity, 1.2)
            * np.clip(
                (78.0 - np.abs(grid.latitude_degrees)) / 12.0,
                0.0,
                1.0,
            )
        )

    hotspot = _hotspot_uplift(
        grid,
        plate_fields,
        crust,
        morphology,
        seed ^ 0xDB4F0B9175AE2165,
    )
    signed += hotspot * (0.58 + 0.30 * morphology.hotspot_activity)
    diagnostics = {
        "boundaryChainCount": len(chains),
        "tectonicPotential": dict(potential.diagnostics),
        "shelfWidthKm": shelf_width,
        "boundaryWarpCells": {"latitude": 7.2, "longitude": 10.8},
        "paleoConvergentFractions": paleo_fraction,
        "continentalReliefProvinces": dict(province_diagnostics),
        "ancientTerraneCount": int(ancient_terrane_fields.plate_count),
        "ancientSutureFraction": float(np.mean(terrane_suture)),
        "reactivatedSutureFraction": float(np.mean(reactivated_union)),
        "reactivatedSutures": dict(reactivated_diagnostics),
        "tectonicSaturationMeters": {
            "uplift": tectonic_uplift_ceiling_m,
            "subsidence": tectonic_subsidence_ceiling_m,
        },
    }
    return signed, continental_plain_support, diagnostics


def _resize_periodic_float(values: np.ndarray, height: int, width: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if source.shape == (height, width):
        return source.copy()
    source_height, source_width = source.shape
    x = (np.arange(width, dtype=np.float64) + 0.5) * source_width / width - 0.5
    x0 = np.floor(x).astype(np.int64) % source_width
    x1 = (x0 + 1) % source_width
    fraction_x = (x - np.floor(x)).astype(np.float32)
    horizontal = source[:, x0] * (1.0 - fraction_x) + source[:, x1] * fraction_x
    y = np.clip(
        (np.arange(height, dtype=np.float64) + 0.5) * source_height / height - 0.5,
        0.0,
        source_height - 1.0,
    )
    y0 = np.floor(y).astype(np.int64)
    y1 = np.minimum(y0 + 1, source_height - 1)
    fraction_y = (y - y0).astype(np.float32)[:, None]
    return (
        horizontal[y0] * (1.0 - fraction_y)
        + horizontal[y1] * fraction_y
    ).astype(np.float32)


def _resize_nearest(values: np.ndarray, height: int, width: int) -> np.ndarray:
    source = np.asarray(values)
    if source.shape == (height, width):
        return source.copy()
    rows = np.clip(
        ((np.arange(height) + 0.5) * source.shape[0] / height).astype(np.int64),
        0,
        source.shape[0] - 1,
    )
    columns = (
        ((np.arange(width) + 0.5) * source.shape[1] / width).astype(np.int64)
        % source.shape[1]
    )
    return source[np.ix_(rows, columns)]


def _ocean_map_seam(land: np.ndarray) -> tuple[int, int]:
    """Return the centre and width of the widest all-ocean meridian corridor."""

    occupied = np.any(np.asarray(land, dtype=bool), axis=0)
    ocean_columns = ~occupied
    width = len(ocean_columns)
    doubled = np.concatenate((ocean_columns, ocean_columns))
    best_start = -1
    best_length = 0
    run_start = -1
    for index, is_ocean in enumerate(doubled):
        if is_ocean and run_start < 0:
            run_start = index
        if run_start >= 0 and (not is_ocean or index == len(doubled) - 1):
            run_end = index if not is_ocean else index + 1
            run_length = min(run_end - run_start, width)
            if run_start < width and run_length > best_length:
                best_start = run_start
                best_length = run_length
            run_start = -1
    minimum_width = max(2, int(math.ceil(width * 0.025)))
    if best_start < 0 or best_length < minimum_width:
        raise ValueError(
            "generated planet has no sufficiently wide complete ocean corridor for the map seam"
        )
    return (best_start + best_length // 2) % width, best_length


def _remap_plate_ids(labels: np.ndarray) -> np.ndarray:
    result = np.zeros(labels.shape, dtype=np.int16)
    for ordinal, plate_id in enumerate(sorted(str(value) for value in np.unique(labels))):
        result[labels == plate_id] = ordinal
    return result


def _continent_owners(
    land: np.ndarray,
    height: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    x, y, z = _sphere_vectors(*land.shape)
    flat_land = np.flatnonzero(land)
    if flat_land.size < count:
        raise ValueError("not enough land cells for the requested continent count")
    land_height = height.reshape(-1)[flat_land]
    candidates = flat_land[land_height >= float(np.quantile(land_height, 0.46))]
    rng = np.random.Generator(np.random.PCG64(seed))
    first = int(candidates[int(rng.integers(0, len(candidates)))])
    selected = [first]
    flat_x, flat_y, flat_z = x.reshape(-1), y.reshape(-1), z.reshape(-1)
    nearest_similarity = (
        flat_x[candidates] * flat_x[first]
        + flat_y[candidates] * flat_y[first]
        + flat_z[candidates] * flat_z[first]
    )
    for _ in range(1, count):
        chosen = int(candidates[int(np.argmax(1.0 - nearest_similarity))])
        selected.append(chosen)
        similarity = (
            flat_x[candidates] * flat_x[chosen]
            + flat_y[candidates] * flat_y[chosen]
            + flat_z[candidates] * flat_z[chosen]
        )
        nearest_similarity = np.maximum(nearest_similarity, similarity)
    best = np.full(land.shape, -2.0, dtype=np.float32)
    owner = np.zeros(land.shape, dtype=np.int16)
    for identifier, cell in enumerate(selected, 1):
        similarity = x * flat_x[cell] + y * flat_y[cell] + z * flat_z[cell]
        wins = similarity > best
        best[wins] = similarity[wins]
        owner[wins] = identifier
    return np.where(land, owner, 0).astype(np.int16)


def _coastline_scale_profile(land: np.ndarray) -> dict[str, float]:
    profile: dict[str, float] = {}
    for block_size in (1, 4, 8):
        cropped_height = land.shape[0] // block_size * block_size
        cropped_width = land.shape[1] // block_size * block_size
        reduced = (
            land[:cropped_height, :cropped_width]
            .reshape(
                cropped_height // block_size,
                block_size,
                cropped_width // block_size,
                block_size,
            )
            .mean(axis=(1, 3))
            > 0.5
        )
        interior = reduced.copy()
        for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            interior &= np.roll(reduced, (row_delta, column_delta), axis=(0, 1))
        coastline = reduced & ~interior
        profile[str(block_size)] = float(
            np.count_nonzero(coastline)
            / max(math.sqrt(float(np.count_nonzero(reduced))), 1.0)
        )
    return profile


def _component_areas(mask: np.ndarray, latitude_weights: np.ndarray) -> list[float]:
    height, width = mask.shape
    remaining = np.asarray(mask, dtype=bool).copy()
    areas: list[float] = []
    while np.any(remaining):
        start = int(np.flatnonzero(remaining.reshape(-1))[0])
        stack = [start]
        remaining.reshape(-1)[start] = False
        area = 0.0
        while stack:
            cell = stack.pop()
            row, column = divmod(cell, width)
            area += float(latitude_weights[row])
            for neighbour_row, neighbour_column in (
                (row - 1, column),
                (row + 1, column),
                (row, (column - 1) % width),
                (row, (column + 1) % width),
            ):
                if 0 <= neighbour_row < height and remaining[neighbour_row, neighbour_column]:
                    remaining[neighbour_row, neighbour_column] = False
                    stack.append(neighbour_row * width + neighbour_column)
        areas.append(area)
    return sorted(areas, reverse=True)


def _dilate(mask: np.ndarray, passes: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(passes):
        result |= (
            np.roll(result, 1, axis=0)
            | np.roll(result, -1, axis=0)
            | np.roll(result, 1, axis=1)
            | np.roll(result, -1, axis=1)
        )
        result[0] &= result[1]
        result[-1] &= result[-2]
    return result


def _classification_metrics(
    land: np.ndarray,
    crust_kind: np.ndarray,
    boundary_class: np.ndarray,
    latitude_degrees: np.ndarray,
) -> dict[str, float]:
    weights = np.cos(np.radians(latitude_degrees))
    land_areas = _component_areas(land, weights)
    total_land = max(sum(land_areas), 1.0e-9)
    land_shares = [value / total_land for value in land_areas]
    ocean_areas = _component_areas(~land, weights)
    total_ocean = max(sum(ocean_areas), 1.0e-9)

    interior = land.copy()
    for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= np.roll(land, (row_delta, column_delta), axis=(0, 1))
    coast = land & ~interior
    plate_boundary = boundary_class != BOUNDARY_INTERIOR
    active_coast = coast & _dilate(plate_boundary, 3)
    internal = land & plate_boundary
    internal_count = max(int(np.count_nonzero(internal)), 1)
    latitude_grid = np.broadcast_to(np.abs(latitude_degrees)[:, None] / 90.0, land.shape)
    area_grid = np.broadcast_to(weights[:, None], land.shape)
    small_island_share = sum(value for value in land_shares if value < 0.035)
    return {
        "largest_landmass_share": float(land_shares[0] if land_shares else 0.0),
        "second_landmass_share": float(land_shares[1] if len(land_shares) > 1 else 0.0),
        "land_component_count": float(len(land_areas)),
        "island_area_share": float(small_island_share),
        "internal_divergent_fraction": float(
            np.count_nonzero(internal & (boundary_class == BOUNDARY_DIVERGENT)) / internal_count
        ),
        "internal_convergent_fraction": float(
            np.count_nonzero(internal & (boundary_class == BOUNDARY_CONVERGENT)) / internal_count
        ),
        "active_margin_fraction": float(
            np.count_nonzero(active_coast) / max(int(np.count_nonzero(coast)), 1)
        ),
        "enclosed_sea_fraction": float(sum(ocean_areas[1:]) / total_ocean),
        "shelf_fraction": float(
            np.count_nonzero((crust_kind == CRUST_TRANSITIONAL) & ~land)
            / max(int(np.count_nonzero(~land)), 1)
        ),
        "mean_absolute_land_latitude": float(
            np.sum(latitude_grid * area_grid * land)
            / max(float(np.sum(area_grid * land)), 1.0e-9)
        ),
    }


def _terrain_system_metrics(
    land: np.ndarray,
    elevation: np.ndarray,
    continent_id: np.ndarray,
) -> dict[str, object]:
    """Measure whether relief reads as a planet, not one local terrain tile."""

    height, _width = land.shape
    latitude_weights = np.cos(
        np.radians(90.0 - (np.arange(height, dtype=np.float64) + 0.5) * 180.0 / height)
    )
    total_land = max(sum(_component_areas(land, latitude_weights)), 1.0e-9)

    def systems(threshold: float, minimum_share: float) -> int:
        areas = _component_areas(
            land & (elevation >= threshold),
            latitude_weights,
        )
        return sum(area / total_land >= minimum_share for area in areas)

    upland = land & (elevation >= 0.30)
    occupied_continents = 0
    for identifier in np.unique(continent_id[land]):
        continent = continent_id == identifier
        continent_area = int(np.count_nonzero(continent))
        if continent_area and np.count_nonzero(upland & continent) / continent_area >= 0.008:
            occupied_continents += 1

    # Connected-component counts merge an entire branching cordillera into a
    # single object.  For a world map the more useful question is how many
    # independently readable relief centres exist after regional smoothing.
    # Greedy spherical non-maximum suppression keeps the measurement stable
    # across export resolutions and across the equirectangular seam.
    smoothing_radius = max(2, int(round(land.shape[1] / 240.0)))
    regional_relief = _smooth_field(
        (np.asarray(elevation, dtype=np.float32) * land).astype(np.float32),
        radius=smoothing_radius,
        passes=2,
    )
    peak_mask = land & (regional_relief >= 0.42)
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            peak_mask &= regional_relief >= np.roll(
                np.roll(regional_relief, row_offset, axis=0),
                column_offset,
                axis=1,
            )
    peak_rows, peak_columns = np.where(peak_mask)
    peak_values = regional_relief[peak_rows, peak_columns]
    peak_order = np.lexsort((peak_columns, peak_rows, -peak_values))
    peak_latitude = (
        math.pi / 2.0
        - (peak_rows.astype(np.float64) + 0.5) * math.pi / land.shape[0]
    )
    peak_longitude = (
        (peak_columns.astype(np.float64) + 0.5) * 2.0 * math.pi / land.shape[1]
        - math.pi
    )
    peak_vectors = np.column_stack(
        (
            np.cos(peak_latitude) * np.cos(peak_longitude),
            np.cos(peak_latitude) * np.sin(peak_longitude),
            np.sin(peak_latitude),
        )
    )
    separation_cosine = math.cos(700.0 / _PLANET_RADIUS_KM)
    selected_peaks: list[int] = []
    for peak_index in peak_order:
        candidate_index = int(peak_index)
        if selected_peaks and float(
            np.max(
                peak_vectors[candidate_index]
                @ peak_vectors[np.asarray(selected_peaks, dtype=np.int64)].T
            )
        ) > separation_cosine:
            continue
        selected_peaks.append(candidate_index)
    orographic_continents = {
        int(continent_id[peak_rows[index], peak_columns[index]])
        for index in selected_peaks
        if int(continent_id[peak_rows[index], peak_columns[index]]) > 0
    }
    return {
        "uplandSystemCount": systems(0.30, 0.0015),
        "highlandSystemCount": systems(0.55, 0.0008),
        "alpineSystemCount": systems(0.72, 0.0004),
        "uplandContinentCount": occupied_continents,
        "uplandLandFraction": float(np.mean(elevation[land] >= 0.30)),
        "majorOrographicCentreCount": len(selected_peaks),
        "orographicContinentCount": len(orographic_continents),
        "orographicCentreMinimumSpacingKm": 700.0,
        "orographicCentreElevationThreshold": 0.42,
    }


def generate_planet_surface(recipe: PlanetRecipe) -> ProceduralSurface:
    """Generate one reproducible physical planet from explicit causal fields."""

    stage_seeds = {
        stage: _derive_seed(recipe.seed, stage) for stage in _STAGE_SALTS
    }
    morphology = sample_world_morphology(stage_seeds["morphology"])
    reference_height = min(_REFERENCE_HEIGHT, recipe.height)
    reference_width = min(_REFERENCE_WIDTH, recipe.width)
    reference_grid = build_lat_lon_grid(reference_height, reference_width)
    definitions = _plate_definitions(recipe.plate_count, morphology, stage_seeds["plates"])
    initial_plate_fields = build_plate_fields(
        reference_grid,
        plate_count=recipe.plate_count,
        anchors=definitions,
        stage_seed=stage_seeds["plates"],
        radius_km=_PLANET_RADIUS_KM,
        boundary_corridors=(),
    )
    boundary_corridors = _curved_boundary_corridors(
        initial_plate_fields,
        morphology,
        stage_seeds["plates"],
    )
    plate_fields = build_plate_fields(
        reference_grid,
        plate_count=recipe.plate_count,
        anchors=definitions,
        stage_seed=stage_seeds["plates"],
        radius_km=_PLANET_RADIUS_KM,
        boundary_corridors=boundary_corridors,
    )
    reference_boundary = _boundary_class_field(plate_fields)
    # A world map needs the accumulated relief of more than one plate cycle.
    # These snapshots are deterministic states of the same moving plates, not
    # independent noise layers.  Five epochs yield young ranges, mature fold
    # belts and subdued ancient sutures at visibly different scales.
    lookback_epochs = (
        30.0 + 18.0 * morphology.continental_aggregation,
        64.0 + 24.0 * morphology.continental_aggregation,
        104.0 + 30.0 * morphology.continental_aggregation,
        152.0 + 36.0 * morphology.continental_aggregation,
        214.0 + 44.0 * morphology.continental_aggregation,
    )
    paleo_boundaries: list[tuple[float, np.ndarray]] = []
    for epoch_ordinal, lookback_myr in enumerate(lookback_epochs):
        paleo_definitions = _advect_plate_definitions(definitions, lookback_myr)
        paleo_fields = build_plate_fields(
            reference_grid,
            plate_count=recipe.plate_count,
            anchors=paleo_definitions,
            stage_seed=(
                stage_seeds["plates"]
                ^ (0x510E527F + 0x9E3779B9 * epoch_ordinal)
            ),
            radius_km=_PLANET_RADIUS_KM,
            boundary_corridors=(),
        )
        paleo_boundaries.append(
            (lookback_myr, _boundary_class_field(paleo_fields))
        )
    ancient_terrane_count = max(36, recipe.plate_count * 3)
    ancient_terrane_definitions = _plate_definitions(
        ancient_terrane_count,
        morphology,
        stage_seeds["orogeny"] ^ 0xBB67AE85,
    )
    initial_terrane_fields = build_plate_fields(
        reference_grid,
        plate_count=ancient_terrane_count,
        anchors=ancient_terrane_definitions,
        stage_seed=stage_seeds["orogeny"] ^ 0x3C6EF372,
        radius_km=_PLANET_RADIUS_KM,
        boundary_corridors=(),
    )
    terrane_corridors = _curved_boundary_corridors(
        initial_terrane_fields,
        morphology,
        stage_seeds["orogeny"] ^ 0xA54FF53A,
    )
    ancient_terrane_fields = build_plate_fields(
        reference_grid,
        plate_count=ancient_terrane_count,
        anchors=ancient_terrane_definitions,
        stage_seed=stage_seeds["orogeny"] ^ 0x3C6EF372,
        radius_km=_PLANET_RADIUS_KM,
        boundary_corridors=terrane_corridors,
    )
    (
        continental_mask,
        crust_potential,
        crust_level,
        crust_structure_metrics,
    ) = _continental_crust(
        reference_grid,
        plate_fields,
        reference_boundary,
        morphology,
        recipe,
        stage_seeds["crust"],
    )
    (
        crust,
        distance_to_continent,
        distance_to_ridge,
        shelf_width,
    ) = _build_crust_fields(
        reference_grid,
        plate_fields,
        continental_mask,
        reference_boundary,
        morphology,
    )
    convergence_source = (
        (reference_boundary == BOUNDARY_CONVERGENT)
        & (crust.kind == CRUST_OCEANIC)
    )
    if np.any(convergence_source):
        distance_to_oceanic_convergence = _distance_from_sources(
            convergence_source,
            reference_grid,
        )
    else:
        distance_to_oceanic_convergence = np.full(
            reference_grid.shape,
            1.0e6,
            dtype=np.float64,
        )
    reference_signed, reference_low_relief, relief_diagnostics = _reference_relief(
        reference_grid,
        plate_fields,
        crust,
        continental_mask,
        crust_potential,
        crust_level,
        reference_boundary,
        ancient_terrane_fields,
        tuple(paleo_boundaries),
        distance_to_continent,
        shelf_width,
        morphology,
        recipe,
        stage_seeds["orogeny"],
    )

    terrain_warp_rows = 4.8 * _smooth_field(
        _spherical_fbm(
            reference_height,
            reference_width,
            stage_seeds["relief"] ^ 0xC6BC279692B5CC83,
            detail=0.45,
        ),
        radius=4,
        passes=2,
    )
    terrain_warp_columns = 7.2 * _smooth_field(
        _spherical_fbm(
            reference_height,
            reference_width,
            stage_seeds["relief"] ^ 0xD3833E804F4C574B,
            detail=0.45,
        ),
        radius=4,
        passes=2,
    )
    reference_signed = _warp_periodic_field(
        reference_signed,
        terrain_warp_rows,
        terrain_warp_columns,
    )
    low_relief_support = _resize_periodic_float(
        _warp_periodic_field(
            reference_low_relief.astype(np.float32),
            terrain_warp_rows,
            terrain_warp_columns,
        ),
        recipe.height,
        recipe.width,
    )

    signed_m = _resize_periodic_float(reference_signed, recipe.height, recipe.width)
    mature_spreading_distance = _distance_from_sources(
        reference_boundary == BOUNDARY_DIVERGENT,
        reference_grid,
    )
    mature_spreading_exclusion = _resize_periodic_float(
        np.exp(
            -0.5 * np.square(mature_spreading_distance / 185.0)
        ).astype(np.float32),
        recipe.height,
        recipe.width,
    )
    signed_m -= (
        760.0 + 660.0 * morphology.ocean_basin_openness
    ) * mature_spreading_exclusion
    full_detail = _spherical_fbm(
        recipe.height,
        recipe.width,
        stage_seeds["coast"],
        detail=0.55 + recipe.coastline_detail,
    )
    signed_m += (300.0 + 680.0 * recipe.coastline_detail) * full_detail
    latitude = (
        math.pi / 2.0
        - (np.arange(recipe.height, dtype=np.float64)[:, None] + 0.5)
        * (math.pi / recipe.height)
    )
    latitude_weight = np.cos(latitude)
    provisional_sea_level_m = _weighted_level(
        signed_m,
        latitude_weight,
        recipe.land_fraction,
    )
    provisional_relative_m = signed_m - provisional_sea_level_m
    relaxed_relative_m, hillslope_diagnostics = relax_hillslopes(
        provisional_relative_m,
        provisional_relative_m > 0.0,
        low_relief_support,
        np.degrees(latitude[:, 0]),
    )
    signed_m += relaxed_relative_m - provisional_relative_m
    provisional_relative_m = relaxed_relative_m
    full_resolution_detail, detail_diagnostics = _full_resolution_relief_detail(
        provisional_relative_m,
        provisional_relative_m > 0.0,
        low_relief_support,
        stage_seeds["relief"],
        recipe.coastline_detail,
    )
    signed_m += full_resolution_detail
    coast_reference_level = _weighted_level(signed_m, latitude_weight, recipe.land_fraction)
    coast_velocity = np.asarray(plate_fields.velocity_vectors_cm_per_year)
    ref_lon = np.radians(reference_grid.longitude_degrees)
    ref_lat = np.radians(reference_grid.latitude_degrees)
    ref_east = np.stack((-np.sin(ref_lon),np.cos(ref_lon),np.zeros_like(ref_lon)),axis=-1)
    ref_north = np.stack((-np.sin(ref_lat)*np.cos(ref_lon),-np.sin(ref_lat)*np.sin(ref_lon),np.cos(ref_lat)),axis=-1)
    coast_delta, coast_evolution = _evolve_coastal_margins(
        signed_m-coast_reference_level, low_relief_support,
        _resize_nearest(reference_boundary,recipe.height,recipe.width),
        _resize_periodic_float(np.sum(coast_velocity*ref_east,axis=-1),recipe.height,recipe.width),
        _resize_periodic_float(np.sum(coast_velocity*ref_north,axis=-1),recipe.height,recipe.width),
        stage_seeds['coast'], recipe.coastline_detail,
    )
    signed_m += coast_delta
    sea_level_m = _weighted_level(
        signed_m,
        latitude_weight,
        recipe.land_fraction,
    )
    relative_m = signed_m - sea_level_m
    land = relative_m > 0.0
    relative_m, geomorphic_evolution = _fluvial_dissection(
        relative_m,
        land,
        np.degrees(latitude[:, 0]),
        low_relief_support,
    )

    # Pole-spanning land necessarily touches every longitude in Plate Carree;
    # choose the rectangular seam using the inhabited/non-polar continents.
    seam_land = land.copy()
    polar_rows = np.abs(np.degrees(latitude[:, 0])) >= 58.0
    selected_polar_rows = polar_rows & np.where(latitude[:, 0] >= 0,
        recipe.north_polar_continent, recipe.south_polar_continent)
    seam_land[selected_polar_rows] = False
    seam_column, seam_width = _ocean_map_seam(seam_land)
    longitude_roll = -seam_column
    signed_m = np.roll(signed_m, longitude_roll, axis=1)
    relative_m = np.roll(relative_m, longitude_roll, axis=1)
    land = np.roll(land, longitude_roll, axis=1)

    reference_plate_ids = _remap_plate_ids(plate_fields.plate_grid)
    plate_id = _resize_nearest(reference_plate_ids, recipe.height, recipe.width).astype(np.int16)
    longitude_radians = np.radians(reference_grid.longitude_degrees)
    latitude_radians = np.radians(reference_grid.latitude_degrees)
    east_vectors = np.stack(
        (
            -np.sin(longitude_radians),
            np.cos(longitude_radians),
            np.zeros(reference_grid.shape, dtype=np.float64),
        ),
        axis=-1,
    )
    north_vectors = np.stack(
        (
            -np.sin(latitude_radians) * np.cos(longitude_radians),
            -np.sin(latitude_radians) * np.sin(longitude_radians),
            np.cos(latitude_radians),
        ),
        axis=-1,
    )
    reference_velocity = np.asarray(
        plate_fields.velocity_vectors_cm_per_year,
        dtype=np.float64,
    )
    velocity_east = _resize_periodic_float(
        np.sum(reference_velocity * east_vectors, axis=-1).astype(np.float32),
        recipe.height,
        recipe.width,
    )
    velocity_north = _resize_periodic_float(
        np.sum(reference_velocity * north_vectors, axis=-1).astype(np.float32),
        recipe.height,
        recipe.width,
    )
    boundary_class = _resize_nearest(reference_boundary, recipe.height, recipe.width).astype(np.int8)
    crust_kind = _resize_nearest(crust.kind, recipe.height, recipe.width).astype(np.uint8)
    age_source = _warp_periodic_field(
        _smooth_field(
            np.nan_to_num(crust.ocean_age_myr, nan=0.0).astype(np.float32),
            radius=2,
            passes=2,
        ),
        terrain_warp_rows,
        terrain_warp_columns,
    )
    ocean_age = _resize_periodic_float(age_source, recipe.height, recipe.width)
    ridge_distance = _resize_periodic_float(
        _warp_periodic_field(
            distance_to_ridge.astype(np.float32),
            terrain_warp_rows,
            terrain_warp_columns,
        ),
        recipe.height,
        recipe.width,
    )
    convergence_distance = _resize_periodic_float(
        _warp_periodic_field(
            distance_to_oceanic_convergence.astype(np.float32),
            terrain_warp_rows,
            terrain_warp_columns,
        ),
        recipe.height,
        recipe.width,
    )
    ocean_age[crust_kind != CRUST_OCEANIC] = np.nan
    plate_id = np.roll(plate_id, longitude_roll, axis=1)
    velocity_east = np.roll(velocity_east, longitude_roll, axis=1)
    velocity_north = np.roll(velocity_north, longitude_roll, axis=1)
    boundary_class = np.roll(boundary_class, longitude_roll, axis=1)
    crust_kind = np.roll(crust_kind, longitude_roll, axis=1)
    ocean_age = np.roll(ocean_age, longitude_roll, axis=1)
    ridge_distance = np.roll(ridge_distance, longitude_roll, axis=1)
    convergence_distance = np.roll(convergence_distance, longitude_roll, axis=1)

    land_height_m = np.maximum(relative_m, 0.0)
    ocean_depth_m = np.maximum(-relative_m, 0.0)
    land_scale = max(float(np.quantile(land_height_m[land], 0.997)), 1.0)
    elevation = np.zeros(land.shape, dtype=np.float32)
    elevation[land] = np.clip(
        0.02 + 0.98 * np.power(land_height_m[land] / land_scale, 1.06),
        0.015,
        1.0,
    )
    bathymetry, ocean_floor_metrics = _visible_ocean_bathymetry(
        land,
        ocean_depth_m,
        crust_kind,
        ocean_age,
        ridge_distance,
        convergence_distance,
        boundary_class,
        stage_seeds["relief"],
    )
    signed_height = np.where(land, elevation, -bathymetry).astype(np.float32)
    continent_id = _continent_owners(
        land,
        relative_m,
        recipe.continent_count,
        stage_seeds["crust"],
    )

    analysis_land = _resize_nearest(
        land,
        reference_height,
        reference_width,
    ).astype(bool)
    metrics = _classification_metrics(
        analysis_land,
        crust.kind,
        reference_boundary,
        reference_grid.latitude_degrees[:, 0],
    )
    continent_areas = {
        str(identifier): int(np.count_nonzero(continent_id == identifier))
        for identifier in range(1, recipe.continent_count + 1)
    }
    fill_ratios: list[float] = []
    for identifier in range(1, recipe.continent_count + 1):
        continent = continent_id == identifier
        rows, columns = np.where(continent)
        bounding_area = (
            (int(rows.max()) - int(rows.min()) + 1)
            * (int(columns.max()) - int(columns.min()) + 1)
        )
        fill_ratios.append(float(np.count_nonzero(continent)) / bounding_area)
    convergent_land = land & (boundary_class == BOUNDARY_CONVERGENT)
    quiet_land = land & (boundary_class == BOUNDARY_INTERIOR)
    diagnostics = {
        "seed": recipe.seed,
        "polarContinents": {"north": recipe.north_polar_continent, "south": recipe.south_polar_continent},
        "stageSeeds": stage_seeds,
        "plateCount": recipe.plate_count,
        "effectivePlateCount": int(len(np.unique(plate_id))),
        "continentCount": recipe.continent_count,
        "seaLevelMeters": sea_level_m,
        "mapSeam": {
            "sourceColumn": seam_column,
            "sourceLongitudeDegrees": (
                (seam_column + 0.5) * 360.0 / recipe.width - 180.0
            ),
            "oceanCorridorWidthColumns": seam_width,
            "longitudeRollColumns": longitude_roll,
        },
        "landFraction": float(
            np.sum(land * latitude_weight)
            / (np.sum(latitude_weight) * recipe.width)
        ),
        "rasterCellLandFraction": float(np.mean(land)),
        "continentAreas": continent_areas,
        "continentBoundingFillMedian": float(np.median(fill_ratios)),
        "coastlineScaleProfile": _coastline_scale_profile(land),
        "tectonicUpliftRatio": float(
            np.mean(elevation[convergent_land])
            / max(float(np.mean(elevation[quiet_land])), 1.0e-6)
        ) if np.any(convergent_land) and np.any(quiet_land) else 1.0,
        "boundaryFractions": {
            "convergent": float(np.mean(boundary_class == BOUNDARY_CONVERGENT)),
            "divergent": float(np.mean(boundary_class == BOUNDARY_DIVERGENT)),
            "transform": float(np.mean(boundary_class == BOUNDARY_TRANSFORM)),
        },
        "referenceGrid": {"height": reference_height, "width": reference_width},
        "terrainSynthesisGrid": {"height": recipe.height, "width": recipe.width},
        "plateDiagnostics": dict(plate_fields.diagnostics),
        "authoredBoundaryCorridorCount": len(boundary_corridors),
        "crustPotentialRange": [float(np.min(crust_potential)), float(np.max(crust_potential))],
        "crustStructureMetrics": dict(crust_structure_metrics),
        "reliefDiagnostics": dict(relief_diagnostics),
        "fullResolutionRelief": dict(detail_diagnostics),
        "coastalEvolution": dict(coast_evolution),
        "geomorphicEvolution": dict(geomorphic_evolution),
        "hillslopeEvolution": dict(hillslope_diagnostics),
        "morphologyAxes": vars_from_slots(morphology),
        "classificationMetrics": metrics,
        "terrainSystemMetrics": _terrain_system_metrics(
            land,
            elevation,
            continent_id,
        ),
        "oceanFloorMetrics": dict(ocean_floor_metrics),
        "worldFamily": classify_world_morphology(metrics),
        "landModel": "seeded-continental-crust-growth",
        "reliefModel": "boundary-profile-plus-ocean-age",
        "mountainLandFraction": float(np.mean(elevation[land] >= 0.72)),
        "method": "explicit-crust-kinematics-v15",
    }
    return ProceduralSurface(
        plate_id=plate_id,
        plate_velocity_east_cm_per_year=velocity_east,
        plate_velocity_north_cm_per_year=velocity_north,
        boundary_class=boundary_class,
        crust_kind=crust_kind,
        ocean_age_myr=ocean_age,
        continent_id=continent_id,
        land_mask=land,
        elevation=elevation,
        bathymetry=bathymetry,
        signed_height=signed_height,
        diagnostics=diagnostics,
    )


def save_palette_source(
    surface: ProceduralSurface,
    path: str | Path,
    *,
    land_palette: Sequence[Sequence[int]],
    bathymetry_palette: Sequence[Sequence[int]],
) -> Path:
    """Quantize one generated surface into the established physical palette."""

    land_colours = np.asarray(land_palette, dtype=np.uint8)
    water_colours = np.asarray(bathymetry_palette, dtype=np.uint8)
    if land_colours.ndim != 2 or land_colours.shape[1] != 3 or len(land_colours) < 2:
        raise ValueError("land_palette must contain at least two RGB colours")
    if water_colours.ndim != 2 or water_colours.shape[1] != 3 or len(water_colours) < 2:
        raise ValueError("bathymetry_palette must contain at least two RGB colours")
    pixels = np.empty((*surface.land_mask.shape, 3), dtype=np.uint8)
    land_index = np.clip(
        np.floor(surface.elevation * len(land_colours)),
        0,
        len(land_colours) - 1,
    ).astype(np.int16)
    water_index = np.clip(
        np.floor(surface.bathymetry * len(water_colours)),
        0,
        len(water_colours) - 1,
    ).astype(np.int16)
    pixels[surface.land_mask] = land_colours[land_index[surface.land_mask]]
    pixels[~surface.land_mask] = water_colours[water_index[~surface.land_mask]]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(output, optimize=True)
    return output


def save_surface_bundle(surface: ProceduralSurface, path: str | Path) -> Path:
    """Persist the causal physical fields consumed by downstream map stages."""

    if not isinstance(surface, ProceduralSurface):
        raise TypeError("surface must be a ProceduralSurface")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    diagnostics = json.dumps(
        dict(surface.diagnostics),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                plate_id=surface.plate_id,
                plate_velocity_east_cm_per_year=(
                    surface.plate_velocity_east_cm_per_year
                ),
                plate_velocity_north_cm_per_year=(
                    surface.plate_velocity_north_cm_per_year
                ),
                boundary_class=surface.boundary_class,
                crust_kind=surface.crust_kind,
                ocean_age_myr=surface.ocean_age_myr,
                continent_id=surface.continent_id,
                land_mask=surface.land_mask,
                elevation=surface.elevation,
                bathymetry=surface.bathymetry,
                signed_height=surface.signed_height,
                diagnostics_utf8=np.frombuffer(diagnostics, dtype=np.uint8),
            )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def load_surface_bundle(path: str | Path) -> ProceduralSurface:
    """Load one causal bundle without pickle or inferred reconstruction."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        expected = {
            "plate_id",
            "plate_velocity_east_cm_per_year",
            "plate_velocity_north_cm_per_year",
            "boundary_class",
            "crust_kind",
            "ocean_age_myr",
            "continent_id",
            "land_mask",
            "elevation",
            "bathymetry",
            "signed_height",
            "diagnostics_utf8",
        }
        if set(archive.files) != expected:
            raise ValueError("surface bundle fields do not match the current schema")
        diagnostics = json.loads(
            bytes(np.asarray(archive["diagnostics_utf8"], dtype=np.uint8)).decode("utf-8")
        )
        return ProceduralSurface(
            plate_id=archive["plate_id"],
            plate_velocity_east_cm_per_year=archive[
                "plate_velocity_east_cm_per_year"
            ],
            plate_velocity_north_cm_per_year=archive[
                "plate_velocity_north_cm_per_year"
            ],
            boundary_class=archive["boundary_class"],
            crust_kind=archive["crust_kind"],
            ocean_age_myr=archive["ocean_age_myr"],
            continent_id=archive["continent_id"],
            land_mask=archive["land_mask"],
            elevation=archive["elevation"],
            bathymetry=archive["bathymetry"],
            signed_height=archive["signed_height"],
            diagnostics=diagnostics,
        )


__all__ = [
    "BOUNDARY_CONVERGENT",
    "BOUNDARY_DIVERGENT",
    "BOUNDARY_INTERIOR",
    "BOUNDARY_TRANSFORM",
    "CRUST_CONTINENTAL",
    "CRUST_OCEANIC",
    "CRUST_TRANSITIONAL",
    "PlanetRecipe",
    "ProceduralSurface",
    "generate_planet_surface",
    "load_surface_bundle",
    "save_palette_source",
    "save_surface_bundle",
]
