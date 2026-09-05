"""Generate physical polar continents before climate and biome derivation.

The authored reference omits both polar caps. This module builds each missing
continent in a local polar plane and samples it on the canonical global
latitude/longitude raster. Coastlines therefore arise from two-dimensional
continental masses, bays and islands rather than a decorative latitude curve
or repeated fjord cut-outs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_atlas.physical.procedural_landmass import (
    ellipse_margin,
    fractal_plane_noise,
    lattice_noise,
)


@dataclass(frozen=True, slots=True)
class PolarMorphologyProfile:
    """Coast grammar measured from the authored non-polar continents."""

    coastline_density: float
    medium_scale_retention: float
    broad_scale_retention: float
    fragmented_coast_share: float

    def as_record(self) -> dict[str, float]:
        return {
            "coastlineDensity": self.coastline_density,
            "mediumScaleRetention": self.medium_scale_retention,
            "broadScaleRetention": self.broad_scale_retention,
            "fragmentedCoastShare": self.fragmented_coast_share,
        }


@dataclass(frozen=True, slots=True)
class PolarTerrainField:
    """Generated land, relief and continental shelf for both poles."""

    north_land: np.ndarray
    south_land: np.ndarray
    land_mask: np.ndarray
    relative_elevation: np.ndarray
    continental_shelf: np.ndarray
    shelf_bathymetry: np.ndarray
    north_coast_latitude: np.ndarray
    south_coast_latitude: np.ndarray
    morphology_profile: PolarMorphologyProfile

    @property
    def diagnostics(self) -> dict[str, object]:
        return {
            "algorithm": "deterministic-polar-plane-continent-field-v2",
            "northLandCells": int(np.count_nonzero(self.north_land)),
            "southLandCells": int(np.count_nonzero(self.south_land)),
            "shelfCells": int(np.count_nonzero(self.continental_shelf)),
            "northMedianCoastLatitude": float(np.median(self.north_coast_latitude)),
            "southMedianCoastLatitude": float(np.median(self.south_coast_latitude)),
            "physicalChainPosition": "before-terrain-refinement-climate-and-biome",
            "hydrologyPolicy": "excluded-before-hydrology",
            "referenceMorphology": self.morphology_profile.as_record(),
        }


def _coordinates(
    shape: tuple[int, int],
    longitude_extent: tuple[float, float],
    latitude_extent: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if tuple(float(value) for value in longitude_extent) != (-180.0, 180.0):
        raise ValueError("polar terrain requires a global -180..180 longitude extent")
    if tuple(float(value) for value in latitude_extent) != (-90.0, 90.0):
        raise ValueError("polar terrain requires a global -90..90 latitude extent")
    height, width = shape
    longitude = -180.0 + (np.arange(width, dtype=np.float64) + 0.5) * 360.0 / width
    latitude = 90.0 - (np.arange(height, dtype=np.float64) + 0.5) * 180.0 / height
    return latitude[:, None], longitude[None, :]


def _boundary_count(mask: np.ndarray) -> int:
    land = np.asarray(mask, dtype=bool)
    padded = np.pad(land, ((1, 1), (0, 0)), mode="constant")
    north = padded[:-2]
    south = padded[2:]
    west = np.roll(land, 1, axis=1)
    east = np.roll(land, -1, axis=1)
    return int(np.count_nonzero(land & ~(north & south & west & east)))


def _coarsen_majority(mask: np.ndarray, factor: int) -> np.ndarray:
    height = mask.shape[0] // factor
    width = mask.shape[1] // factor
    cropped = mask[: height * factor, : width * factor]
    return (
        cropped.reshape(height, factor, width, factor).mean(axis=(1, 3))
        >= 0.5
    )


def _measure_reference_morphology(reference_land_mask: np.ndarray) -> PolarMorphologyProfile:
    land = np.asarray(reference_land_mask, dtype=bool)
    if land.ndim != 2 or min(land.shape) < 8:
        raise ValueError("reference land morphology requires a global two-dimensional mask")
    if not bool(land.any()):
        return PolarMorphologyProfile(
            coastline_density=0.08,
            medium_scale_retention=0.72,
            broad_scale_retention=0.58,
            fragmented_coast_share=0.04,
        )
    # The source intentionally has no polar basemap.  Measure only the central
    # 76% of rows so the blank authoring margins do not masquerade as a coast.
    margin = max(1, int(round(land.shape[0] * 0.12)))
    authored = land[margin:-margin]
    if not bool(authored.any()):
        authored = land
    fine_boundary = max(1, _boundary_count(authored))
    medium = _coarsen_majority(authored, 4)
    broad = _coarsen_majority(authored, 8)
    medium_retention = np.clip(
        _boundary_count(medium) * 4.0 / fine_boundary, 0.0, 1.0
    )
    broad_retention = np.clip(
        _boundary_count(broad) * 8.0 / fine_boundary, 0.0, 1.0
    )
    numeric = authored.astype(np.uint8)
    neighbours = (
        np.roll(numeric, 1, axis=0)
        + np.roll(numeric, -1, axis=0)
        + np.roll(numeric, 1, axis=1)
        + np.roll(numeric, -1, axis=1)
    )
    fragmented = float(np.mean(neighbours[authored] <= 2))
    return PolarMorphologyProfile(
        coastline_density=float(fine_boundary / max(1, np.count_nonzero(authored))),
        medium_scale_retention=float(medium_retention),
        broad_scale_retention=float(broad_retention),
        fragmented_coast_share=fragmented,
    )


def _continent_field(
    radius: np.ndarray,
    longitude: np.ndarray,
    profile: PolarMorphologyProfile,
    *,
    north: bool,
) -> np.ndarray:
    """Synthesize one landmass from the authored world's coast spectrum."""

    angle = np.deg2rad(longitude)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    if north:
        center_x, center_y = (-0.8, 0.7)
        radius_x, radius_y, rotation = (16.3, 12.4, 19.0)
        coast_seed, island_seed = (431, 577)
    else:
        center_x, center_y = (0.9, -0.5)
        radius_x, radius_y, rotation = (13.2, 16.7, -27.0)
        coast_seed, island_seed = (863, 997)
    field = ellipse_margin(
        x,
        y,
        center_x,
        center_y,
        radius_x,
        radius_y,
        rotation,
    )

    # Coarse and medium terms establish continental lobes and open bays.  Fine
    # terms inherit their amplitude from how much shoreline survives at 4x and
    # 8x downsampling in the authored reference.
    broad = lattice_noise(x, y, spacing=10.5, seed=coast_seed)
    medium = lattice_noise(x, y, spacing=5.2, seed=coast_seed + 37)
    fine = fractal_plane_noise(x, y, seed=coast_seed + 83)
    micro = (
        0.48 * lattice_noise(x, y, spacing=1.25, seed=coast_seed + 137)
        + 0.32 * lattice_noise(x, y, spacing=0.58, seed=coast_seed + 191)
        + 0.20 * lattice_noise(x, y, spacing=0.29, seed=coast_seed + 251)
    )
    broad_gain = 0.58 + 0.34 * (1.0 - profile.broad_scale_retention)
    medium_gain = 0.28 + 0.30 * (1.0 - profile.medium_scale_retention)
    fine_gain = 2.0 * (
        0.16 + 1.30 * np.clip(profile.coastline_density, 0.0, 0.16)
    )
    micro_gain = 3.5 * (
        0.10 + 1.45 * np.clip(profile.coastline_density, 0.0, 0.16)
    )
    coast_envelope = 0.28 + 0.72 * np.exp(-((field / 0.68) ** 2))
    field += (
        broad_gain * broad
        + medium_gain * medium
        + fine_gain * fine
        + micro_gain * micro
    ) * coast_envelope

    # Independent annular noise produces natural islands and broken peninsulas
    # without fixed counts, spacing or geometry.  North and south use separate
    # seeds and differently oriented continental axes.
    island_noise = (
        0.63 * lattice_noise(x, y, spacing=4.3, seed=island_seed)
        + 0.27 * lattice_noise(x, y, spacing=2.0, seed=island_seed + 41)
        + 0.10 * lattice_noise(x, y, spacing=0.9, seed=island_seed + 97)
    )
    island_threshold = (
        0.52
        - 0.55 * np.clip(profile.fragmented_coast_share, 0.0, 0.18)
        + 0.021 * np.abs(radius - (19.5 if north else 20.0))
    )
    island_field = island_noise - island_threshold
    island_annulus = (radius >= 15.0) & (radius <= 22.6)
    combined = np.maximum(field, np.where(island_annulus, island_field, -np.inf))
    # Leave a real open-ocean belt between the generated poles and the
    # authored continents.  The cap remains poleward of the axial-tilt-derived
    # polar circle, so every generated cell receives the polar-only biome and
    # non-human policy instead of leaking into ordinary continental systems.
    outer_falloff = np.clip((radius - 20.3) / 2.4, 0.0, None)
    combined -= 2.7 * outer_falloff * outer_falloff
    return np.where(radius <= 23.2, combined, -np.inf)


def _coast_latitudes(
    land: np.ndarray,
    radius: np.ndarray,
    *,
    north: bool,
) -> np.ndarray:
    radial_grid = np.broadcast_to(radius, land.shape)
    outer_radius = np.max(np.where(land, radial_grid, -np.inf), axis=0)
    if not bool(np.all(np.isfinite(outer_radius))):
        raise ValueError("each pole-spanning continent must cover every longitude")
    return 90.0 - outer_radius if north else -90.0 + outer_radius


def _fill_enclosed_polar_water(land: np.ndarray, *, north: bool) -> np.ndarray:
    """Fill closed polar depressions while preserving ocean-connected bays."""

    result = np.asarray(land, dtype=bool).copy()
    occupied_rows = np.flatnonzero(np.any(result, axis=1))
    if occupied_rows.size == 0:
        return result
    if north:
        band_start = 0
        band_stop = min(result.shape[0], int(occupied_rows.max()) + 2)
        band = result[band_start:band_stop].copy()
        seed_row = -1
    else:
        band_start = max(0, int(occupied_rows.min()) - 1)
        band_stop = result.shape[0]
        band = result[band_start:band_stop].copy()
        seed_row = 0
    water = ~band
    exterior = np.zeros_like(water)
    exterior[seed_row, :] = water[seed_row, :]
    while True:
        reached = np.roll(exterior, 1, axis=1) | np.roll(exterior, -1, axis=1)
        reached[1:, :] |= exterior[:-1, :]
        reached[:-1, :] |= exterior[1:, :]
        reached &= water
        expanded = exterior | reached
        if np.array_equal(expanded, exterior):
            break
        exterior = expanded
    band |= water & ~exterior
    result[band_start:band_stop] = band
    return result


def _polar_relief(
    land: np.ndarray,
    strength: np.ndarray,
    radius: np.ndarray,
    longitude: np.ndarray,
    *,
    north: bool,
) -> np.ndarray:
    angle = np.deg2rad(longitude)
    polar_x = radius * np.cos(angle)
    polar_y = radius * np.sin(angle)
    interior = np.clip(strength, 0.0, 1.0)
    texture = fractal_plane_noise(
        polar_x, polar_y, seed=887 if north else 1217
    )

    if north:
        ridge_one = np.exp(-((polar_y - 0.38 * polar_x + 1.5) / 1.35) ** 2) * np.exp(
            -((polar_x + 1.5) / 13.0) ** 2
        )
        ridge_two = np.exp(-((polar_x + 0.52 * polar_y - 2.0) / 1.55) ** 2) * np.exp(
            -((polar_y - 1.0) / 11.0) ** 2
        )
        massif = np.exp(
            -0.5 * (((polar_x + 6.0) / 3.0) ** 2 + ((polar_y + 4.5) / 3.8) ** 2)
        )
        basin = np.exp(
            -0.5 * (((polar_x - 4.2) / 4.8) ** 2 + ((polar_y - 5.0) / 4.0) ** 2)
        )
    else:
        ridge_one = np.exp(-((polar_y + 0.46 * polar_x - 1.0) / 1.25) ** 2) * np.exp(
            -((polar_x - 1.0) / 13.5) ** 2
        )
        ridge_two = np.exp(-((polar_x - 0.44 * polar_y + 2.3) / 1.45) ** 2) * np.exp(
            -((polar_y + 0.5) / 10.5) ** 2
        )
        massif = np.exp(
            -0.5 * (((polar_x - 6.8) / 3.4) ** 2 + ((polar_y + 3.6) / 3.3) ** 2)
        )
        basin = np.exp(
            -0.5 * (((polar_x + 4.5) / 4.6) ** 2 + ((polar_y - 4.8) / 4.2) ** 2)
        )

    coastal_plain = 0.055 + 0.12 * np.power(interior, 0.72)
    mountain_envelope = np.clip(interior * 3.6, 0.0, 1.0)
    relief = (
        coastal_plain
        + 0.22 * texture * mountain_envelope
        + 0.34 * ridge_one * mountain_envelope
        + 0.25 * ridge_two * mountain_envelope
        + 0.23 * massif * mountain_envelope
        - 0.15 * basin * mountain_envelope
    )
    relief = np.clip(relief, 0.035, 0.88)
    relief[~land] = 0.0
    return relief


def generate_polar_terrain(
    shape: tuple[int, int],
    longitude_extent: tuple[float, float],
    latitude_extent: tuple[float, float],
    reference_land_mask: np.ndarray,
) -> PolarTerrainField:
    """Generate two pole-spanning continents from authored coast morphology."""

    if len(shape) != 2 or min(shape) < 8:
        raise ValueError("polar terrain requires a two-dimensional global raster")
    latitude, longitude = _coordinates(shape, longitude_extent, latitude_extent)
    if np.asarray(reference_land_mask).shape != shape:
        raise ValueError("reference land mask and polar terrain shape must match")
    morphology = _measure_reference_morphology(reference_land_mask)
    north_radius = 90.0 - latitude
    south_radius = 90.0 + latitude
    north_strength = _continent_field(
        north_radius, longitude, morphology, north=True
    )
    south_strength = _continent_field(
        south_radius, longitude, morphology, north=False
    )
    north_land = north_strength > 0.0
    south_land = south_strength > 0.0

    # A continent containing a pole occupies the entire projection edge in
    # Plate Carree because all longitudes converge at one physical point.
    north_land[0, :] = True
    south_land[-1, :] = True
    north_land = _fill_enclosed_polar_water(north_land, north=True)
    south_land = _fill_enclosed_polar_water(south_land, north=False)
    north_strength[0, :] = np.maximum(north_strength[0, :], 0.55)
    south_strength[-1, :] = np.maximum(south_strength[-1, :], 0.55)
    land = north_land | south_land

    elevation = np.zeros(shape, dtype=np.float64)
    north_relief = _polar_relief(
        north_land,
        north_strength,
        north_radius,
        longitude,
        north=True,
    )
    south_relief = _polar_relief(
        south_land,
        south_strength,
        south_radius,
        longitude,
        north=False,
    )
    elevation[north_land] = north_relief[north_land]
    elevation[south_land] = south_relief[south_land]

    shelf_limit = -0.34
    north_shelf = (~land) & (north_strength > shelf_limit)
    south_shelf = (~land) & (south_strength > shelf_limit)
    shelf = north_shelf | south_shelf
    shelf_strength = np.where(north_shelf, north_strength, south_strength)
    shelf_bathymetry = np.ones(shape, dtype=np.float64)
    shelf_bathymetry[shelf] = np.clip(
        0.035 + 0.265 * (-shelf_strength[shelf] / abs(shelf_limit)),
        0.035,
        0.30,
    )

    return PolarTerrainField(
        north_land=north_land.astype(bool),
        south_land=south_land.astype(bool),
        land_mask=land.astype(bool),
        relative_elevation=elevation,
        continental_shelf=shelf.astype(bool),
        shelf_bathymetry=shelf_bathymetry,
        north_coast_latitude=_coast_latitudes(
            north_land, north_radius, north=True
        ).astype(np.float64),
        south_coast_latitude=_coast_latitudes(
            south_land, south_radius, north=False
        ).astype(np.float64),
        morphology_profile=morphology,
    )


__all__ = [
    "PolarMorphologyProfile",
    "PolarTerrainField",
    "generate_polar_terrain",
]
