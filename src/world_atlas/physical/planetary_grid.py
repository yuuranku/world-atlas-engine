"""Parameterized spherical geometry and radiation primitives.

The authored map remains plate carrée, while distances, directions, areas,
and radiation are evaluated on the unit sphere.  This module is deliberately
limited to that shared geometry layer; terrain, plates, hydrology, and
rendering belong to later pipeline stages.

Ordinary grid cells stop short of both poles.  The adjacency graph adds one
explicit node for each pole and joins it to the complete adjacent latitude
row, so a polar cap is one spherical neighbourhood rather than a row of
unrelated duplicate points.  Longitude cell neighbours wrap across the
``-180/180`` seam.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class PlanetConfigError(ValueError):
    """Raised when a planetary configuration is incomplete or non-physical."""


@dataclass(frozen=True, slots=True)
class PlanetConfig:
    """Required planet-scale parameters for spherical calculations."""

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

    def __post_init__(self) -> None:
        numeric_fields = (
            "radius_km",
            "gravity_mps2",
            "rotation_period_hours",
            "axial_tilt_degrees",
            "orbital_period_days",
            "orbital_eccentricity",
            "longitude_of_periapsis_degrees",
            "solar_constant_wm2",
            "surface_pressure_pa",
            "ocean_heat_capacity_factor",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if isinstance(value, (bool, np.bool_)):
                raise PlanetConfigError(f"{field_name} must be a finite number")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise PlanetConfigError(
                    f"{field_name} must be a finite number"
                ) from error
            if not math.isfinite(numeric):
                raise PlanetConfigError(f"{field_name} must be a finite number")
            object.__setattr__(self, field_name, numeric)

        if not isinstance(self.rotation_direction, str) or self.rotation_direction not in {
            "prograde",
            "retrograde",
        }:
            raise PlanetConfigError(
                "rotation_direction must be 'prograde' or 'retrograde'"
            )
        for field_name in (
            "radius_km",
            "gravity_mps2",
            "rotation_period_hours",
            "orbital_period_days",
            "solar_constant_wm2",
            "surface_pressure_pa",
            "ocean_heat_capacity_factor",
        ):
            if getattr(self, field_name) <= 0.0:
                raise PlanetConfigError(f"{field_name} must be positive")
        if not 0.0 <= self.axial_tilt_degrees <= 90.0:
            raise PlanetConfigError("axial_tilt_degrees must be in [0, 90]")
        if not 0.0 <= self.orbital_eccentricity < 1.0:
            raise PlanetConfigError("orbital_eccentricity must be in [0, 1)")

    @property
    def rotation_sign(self) -> float:
        """Signed rotation direction: +1 prograde, -1 retrograde."""

        return 1.0 if self.rotation_direction == "prograde" else -1.0

    @property
    def angular_velocity_rad_s(self) -> float:
        """Signed angular velocity in radians per second."""

        return self.rotation_sign * 2.0 * math.pi / (
            self.rotation_period_hours * 3600.0
        )


def _as_float_array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes, bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_lon_lat(
    longitude_degrees: Any,
    latitude_degrees: Any,
) -> tuple[np.ndarray, np.ndarray]:
    longitude = _as_float_array(longitude_degrees, "longitude_degrees")
    latitude = _as_float_array(latitude_degrees, "latitude_degrees")
    try:
        longitude, latitude = np.broadcast_arrays(longitude, latitude)
    except ValueError as error:
        raise ValueError("longitude and latitude cannot be broadcast together") from error
    if np.any(latitude < -90.0) or np.any(latitude > 90.0):
        raise ValueError("latitude_degrees must be in [-90, 90]")
    longitude = ((longitude + 180.0) % 360.0) - 180.0
    return longitude, latitude


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def lon_lat_to_unit_vector(
    longitude_degrees: Any,
    latitude_degrees: Any,
) -> np.ndarray:
    """Convert plate-carrée longitude/latitude to unit vectors.

    Inputs may be scalars or broadcastable arrays.  The final vector axis is
    ``(x, y, z)`` in an origin-centred coordinate frame.
    """

    longitude, latitude = _validate_lon_lat(longitude_degrees, latitude_degrees)
    longitude_radians = np.deg2rad(longitude)
    latitude_radians = np.deg2rad(latitude)
    cos_latitude = np.cos(latitude_radians)
    return np.stack(
        (
            cos_latitude * np.cos(longitude_radians),
            cos_latitude * np.sin(longitude_radians),
            np.sin(latitude_radians),
        ),
        axis=-1,
    )


def _normalise_vectors(vectors: Any) -> np.ndarray:
    try:
        array = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("vectors must be numeric") from error
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError("vectors must have a final dimension of length 3")
    if not np.all(np.isfinite(array)):
        raise ValueError("vectors contains non-finite values")
    norm = np.linalg.norm(array, axis=-1)
    if np.any(norm <= np.finfo(np.float64).tiny):
        raise ValueError("vectors contains a zero-length vector")
    return array / norm[..., None]


def unit_vector_to_lon_lat(
    vectors: Any,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert vectors to canonical ``[-180, 180)`` longitude and latitude."""

    normalized = _normalise_vectors(vectors)
    longitude = np.rad2deg(np.arctan2(normalized[..., 1], normalized[..., 0]))
    longitude = ((longitude + 180.0) % 360.0) - 180.0
    latitude = np.rad2deg(np.arcsin(np.clip(normalized[..., 2], -1.0, 1.0)))
    # Longitude is undefined at the poles; use one stable serialized value.
    pole = np.isclose(np.abs(normalized[..., 2]), 1.0, atol=1.0e-12)
    longitude = np.where(pole, 0.0, longitude)
    return _scalar_or_array(longitude), _scalar_or_array(latitude)


def _point_to_unit_vector(point: Any, name: str) -> np.ndarray:
    array = _as_float_array(point, name)
    if array.ndim == 0 or array.shape[-1] not in (2, 3):
        raise ValueError(f"{name} must have a final dimension of length 2 or 3")
    if array.shape[-1] == 2:
        return lon_lat_to_unit_vector(array[..., 0], array[..., 1])
    return _normalise_vectors(array)


def great_circle_distance_km(
    point_a: Any,
    point_b: Any,
    planet: PlanetConfig,
) -> float | np.ndarray:
    """Return the shortest spherical distance between two points in km.

    Points are normally ``(longitude, latitude)`` in degrees.  Unit vectors
    are also accepted when they are already the caller's working geometry.
    """

    if not isinstance(planet, PlanetConfig):
        raise TypeError("planet must be a PlanetConfig instance")
    vectors_a = _point_to_unit_vector(point_a, "point_a")
    vectors_b = _point_to_unit_vector(point_b, "point_b")
    try:
        vectors_a, vectors_b = np.broadcast_arrays(vectors_a, vectors_b)
    except ValueError as error:
        raise ValueError("point_a and point_b cannot be broadcast together") from error
    dot = np.sum(vectors_a * vectors_b, axis=-1)
    cross_norm = np.linalg.norm(np.cross(vectors_a, vectors_b), axis=-1)
    # atan2 retains small-angle precision better than acos(dot), while also
    # giving the correct pi value for antipodal points.
    angle = np.arctan2(cross_norm, dot)
    return _scalar_or_array(angle * planet.radius_km)


def local_east_north_basis(
    longitude_degrees: Any,
    latitude_degrees: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal local east and north tangent vectors."""

    longitude, latitude = _validate_lon_lat(longitude_degrees, latitude_degrees)
    lon = np.deg2rad(longitude)
    lat = np.deg2rad(latitude)
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
    north = np.stack(
        (
            -np.sin(lat) * np.cos(lon),
            -np.sin(lat) * np.sin(lon),
            np.cos(lat),
        ),
        axis=-1,
    )
    return east, north


@dataclass(frozen=True, slots=True)
class LatLonGrid:
    """Regular plate-carrée cells and their explicit spherical adjacency."""

    latitudes: np.ndarray
    longitudes: np.ndarray
    latitude_degrees: np.ndarray
    longitude_degrees: np.ndarray
    unit_vectors: np.ndarray
    neighbors: tuple[tuple[int, ...], ...]
    north_pole_index: int
    south_pole_index: int
    node_latitudes: np.ndarray
    node_longitudes: np.ndarray
    node_vectors: np.ndarray
    latitude_edges: np.ndarray
    longitude_edges: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.latitude_degrees.shape

    @property
    def cell_count(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def node_count(self) -> int:
        return len(self.neighbors)

    @property
    def pole_indices(self) -> tuple[int, int]:
        return self.north_pole_index, self.south_pole_index

    def cell_index(self, row: int, column: int) -> int:
        if not isinstance(row, (int, np.integer)) or not isinstance(
            column, (int, np.integer)
        ):
            raise TypeError("row and column must be integers")
        row = int(row)
        column = int(column)
        if not (0 <= row < self.shape[0] and 0 <= column < self.shape[1]):
            raise IndexError("cell coordinate is outside the grid")
        return row * self.shape[1] + column

    def row_column(self, index: int) -> tuple[int, int]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("index must be an integer")
        index = int(index)
        if not 0 <= index < self.cell_count:
            raise IndexError("index is not an ordinary grid cell")
        return divmod(index, self.shape[1])

    @property
    def cell_neighbors(self) -> np.ndarray:
        result = np.empty(self.shape, dtype=object)
        for row in range(self.shape[0]):
            for column in range(self.shape[1]):
                result[row, column] = self.neighbors[self.cell_index(row, column)]
        return result


def _freeze_array(value: Any) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def build_lat_lon_grid(latitude_count: int, longitude_count: int) -> LatLonGrid:
    """Build a regular grid with seam wrapping and one node per pole."""

    if isinstance(latitude_count, (bool, np.bool_)) or not isinstance(
        latitude_count, (int, np.integer)
    ):
        raise TypeError("latitude_count must be a positive integer")
    if isinstance(longitude_count, (bool, np.bool_)) or not isinstance(
        longitude_count, (int, np.integer)
    ):
        raise TypeError("longitude_count must be a positive integer")
    n_lat = int(latitude_count)
    n_lon = int(longitude_count)
    if n_lat <= 0 or n_lon <= 0:
        raise ValueError("latitude_count and longitude_count must be positive")

    # The authored rasters use y-south coordinates: row zero is the north
    # edge.  Keep the latitude axis and every derived array in that order.
    latitude_edges = np.linspace(90.0, -90.0, n_lat + 1, dtype=np.float64)
    latitude_degrees = (latitude_edges[:-1] + latitude_edges[1:]) * 0.5
    longitude_edges = np.linspace(-180.0, 180.0, n_lon + 1, dtype=np.float64)
    # Cell centers, not longitude boundaries.  The first and last centers
    # still connect through the explicit date-line seam in the graph.
    longitude_degrees = (longitude_edges[:-1] + longitude_edges[1:]) * 0.5
    lat_grid, lon_grid = np.meshgrid(
        latitude_degrees, longitude_degrees, indexing="ij"
    )
    vectors = lon_lat_to_unit_vector(lon_grid, lat_grid)

    cell_count = n_lat * n_lon
    north_pole_index = cell_count
    south_pole_index = cell_count + 1
    neighbour_sets: list[set[int]] = [set() for _ in range(cell_count + 2)]

    def connect(first: int, second: int) -> None:
        if first == second:
            return
        neighbour_sets[first].add(second)
        neighbour_sets[second].add(first)

    for row in range(n_lat):
        for column in range(n_lon):
            index = row * n_lon + column
            # The modulo is the plate-carrée date-line seam.
            connect(index, row * n_lon + ((column - 1) % n_lon))
            connect(index, row * n_lon + ((column + 1) % n_lon))
            if row > 0:
                connect(index, (row - 1) * n_lon + column)
            else:
                connect(index, north_pole_index)
            if row + 1 < n_lat:
                connect(index, (row + 1) * n_lon + column)
            else:
                connect(index, south_pole_index)

    neighbours = tuple(tuple(sorted(values)) for values in neighbour_sets)
    node_latitudes = np.concatenate((latitude_degrees.repeat(n_lon), [90.0, -90.0]))
    node_longitudes = np.concatenate((np.tile(longitude_degrees, n_lat), [0.0, 0.0]))
    node_vectors = np.concatenate(
        (
            vectors.reshape((-1, 3)),
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        ),
        axis=0,
    )
    return LatLonGrid(
        latitudes=_freeze_array(latitude_degrees),
        longitudes=_freeze_array(longitude_degrees),
        latitude_degrees=_freeze_array(lat_grid),
        longitude_degrees=_freeze_array(lon_grid),
        unit_vectors=_freeze_array(vectors),
        neighbors=neighbours,
        north_pole_index=north_pole_index,
        south_pole_index=south_pole_index,
        node_latitudes=_freeze_array(node_latitudes),
        node_longitudes=_freeze_array(node_longitudes),
        node_vectors=_freeze_array(node_vectors),
        latitude_edges=_freeze_array(latitude_edges),
        longitude_edges=_freeze_array(longitude_edges),
    )


def cell_area_weights(grid: LatLonGrid) -> np.ndarray:
    """Return exact dimensionless spherical solid angles for every grid cell."""

    if not isinstance(grid, LatLonGrid):
        raise TypeError("grid must be a LatLonGrid instance")
    latitude_band = np.abs(
        np.diff(np.sin(np.deg2rad(grid.latitude_edges)))
    )
    longitude_band = np.abs(np.diff(np.deg2rad(grid.longitude_edges)))
    return _freeze_array(latitude_band[:, None] * longitude_band[None, :])


def cell_area_m2(grid: LatLonGrid, radius_km: Any) -> np.ndarray:
    """Return exact spherical cell areas in square metres for ``radius_km``."""

    if isinstance(radius_km, (bool, np.bool_)):
        raise TypeError("radius_km must be a finite positive scalar")
    try:
        raw_radius = np.asarray(radius_km)
    except (TypeError, ValueError) as error:
        raise TypeError("radius_km must be a finite positive scalar") from error
    if raw_radius.ndim != 0 or raw_radius.dtype.kind not in {"i", "u", "f"}:
        raise TypeError("radius_km must be a finite positive scalar")
    try:
        radius_value = float(raw_radius)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("radius_km must be a finite positive scalar") from error
    if not math.isfinite(radius_value) or radius_value <= 0.0:
        raise ValueError("radius_km must be a finite positive scalar")
    radius_m = radius_value * 1000.0
    return _freeze_array(cell_area_weights(grid) * radius_m**2)


def _month_geometry(
    month_index: Any,
    planet: PlanetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return solar declination radians and inverse-square distance factor."""

    month = _as_float_array(month_index, "month_index")
    # Months are a normalized orbital phase, not a Gregorian day count.  The
    # periapsis longitude places that phase relative to the seasonal equinox;
    # no Earth orbital-period constant is embedded here.
    phase = (np.mod(month, 12.0) + 0.5) / 12.0
    periapsis = math.radians(planet.longitude_of_periapsis_degrees)
    mean_anomaly = 2.0 * math.pi * phase
    eccentricity = planet.orbital_eccentricity
    if eccentricity == 0.0:
        true_anomaly = mean_anomaly
    else:
        eccentric_anomaly = mean_anomaly.copy()
        for _ in range(8):
            eccentric_anomaly -= (
                eccentric_anomaly
                - eccentricity * np.sin(eccentric_anomaly)
                - mean_anomaly
            ) / (1.0 - eccentricity * np.cos(eccentric_anomaly))
        true_anomaly = 2.0 * np.arctan2(
            math.sqrt(1.0 + eccentricity) * np.sin(eccentric_anomaly / 2.0),
            math.sqrt(1.0 - eccentricity) * np.cos(eccentric_anomaly / 2.0),
        )
    solar_longitude = true_anomaly - periapsis
    declination = np.arcsin(
        np.clip(
            math.sin(math.radians(planet.axial_tilt_degrees))
            * np.sin(solar_longitude),
            -1.0,
            1.0,
        )
    )
    radius_ratio = (1.0 - eccentricity * eccentricity) / (
        1.0 + eccentricity * np.cos(true_anomaly)
    )
    distance_factor = 1.0 / np.square(radius_ratio)
    return declination, distance_factor


def solar_declination(
    month_index: Any,
    planet: PlanetConfig,
) -> float | np.ndarray:
    """Return solar declination in degrees for a zero-based month index."""

    if not isinstance(planet, PlanetConfig):
        raise TypeError("planet must be a PlanetConfig instance")
    declination, _ = _month_geometry(month_index, planet)
    return _scalar_or_array(np.rad2deg(declination))


def daily_mean_insolation(
    latitude_degrees: Any,
    month_index: Any,
    planet: PlanetConfig,
) -> float | np.ndarray:
    """Return daily-mean top-of-atmosphere insolation in W/m².

    Polar day and polar night are separate branches; ordinary ``acos`` input
    clipping is never used to turn an all-night month into daylight.
    """

    if not isinstance(planet, PlanetConfig):
        raise TypeError("planet must be a PlanetConfig instance")
    latitude = _as_float_array(latitude_degrees, "latitude_degrees")
    if np.any(latitude < -90.0) or np.any(latitude > 90.0):
        raise ValueError("latitude_degrees must be in [-90, 90]")
    declination, distance_factor = _month_geometry(month_index, planet)
    return _daily_mean_insolation_from_geometry(
        latitude,
        declination,
        distance_factor,
        planet,
        broadcast_error="latitude and month cannot be broadcast together",
    )


def _daily_mean_insolation_from_geometry(
    latitude: np.ndarray,
    declination: np.ndarray,
    distance_factor: np.ndarray,
    planet: PlanetConfig,
    *,
    broadcast_error: str,
) -> float | np.ndarray:
    try:
        latitude, declination, distance_factor = np.broadcast_arrays(
            latitude, declination, distance_factor
        )
    except ValueError as error:
        raise ValueError(broadcast_error) from error

    phi = np.deg2rad(latitude)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_delta = np.sin(declination)
    cos_delta = np.cos(declination)
    pole = np.isclose(np.abs(latitude), 90.0, atol=1.0e-12)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        hour_angle_argument = -np.tan(phi) * np.tan(declination)
    polar_day = (~pole) & (hour_angle_argument <= -1.0)
    polar_night = (~pole) & (hour_angle_argument >= 1.0)
    ordinary = ~(polar_day | polar_night | pole)
    hour_angle = np.zeros_like(latitude, dtype=np.float64)
    hour_angle[polar_day] = math.pi
    hour_angle[ordinary] = np.arccos(
        np.clip(hour_angle_argument[ordinary], -1.0, 1.0)
    )
    pole_day = pole & (sin_phi * sin_delta > 0.0)
    hour_angle[pole_day] = math.pi
    bracket = hour_angle * sin_phi * sin_delta + cos_phi * cos_delta * np.sin(
        hour_angle
    )
    result = planet.solar_constant_wm2 * distance_factor * bracket / math.pi
    result = np.where(polar_night, 0.0, result)
    result = np.where(pole & ~pole_day, 0.0, result)
    return _scalar_or_array(np.maximum(result, 0.0))


def daily_mean_insolation_at_solar_longitude(
    latitude_degrees: Any,
    solar_longitude_degrees: Any,
    planet: PlanetConfig,
) -> float | np.ndarray:
    """Return daily-mean insolation for an exact seasonal solar longitude.

    Solar longitude is measured from the northern vernal equinox: ``0`` is
    the March equinox, ``90`` the northern June solstice, ``180`` the
    September equinox, and ``270`` the northern December solstice.  Orbital
    eccentricity and the configured periapsis longitude affect only the
    inverse-square distance term; the requested seasonal longitude remains
    exact rather than being approximated by a calendar month.
    """

    if not isinstance(planet, PlanetConfig):
        raise TypeError("planet must be a PlanetConfig instance")
    latitude = _as_float_array(latitude_degrees, "latitude_degrees")
    if np.any(latitude < -90.0) or np.any(latitude > 90.0):
        raise ValueError("latitude_degrees must be in [-90, 90]")
    solar_longitude = _as_float_array(
        solar_longitude_degrees, "solar_longitude_degrees"
    )
    seasonal_angle = np.deg2rad(np.mod(solar_longitude, 360.0))
    declination = np.arcsin(
        np.clip(
            math.sin(math.radians(planet.axial_tilt_degrees))
            * np.sin(seasonal_angle),
            -1.0,
            1.0,
        )
    )
    true_anomaly = seasonal_angle + math.radians(
        planet.longitude_of_periapsis_degrees
    )
    eccentricity = planet.orbital_eccentricity
    radius_ratio = (1.0 - eccentricity * eccentricity) / (
        1.0 + eccentricity * np.cos(true_anomaly)
    )
    distance_factor = 1.0 / np.square(radius_ratio)
    return _daily_mean_insolation_from_geometry(
        latitude,
        declination,
        distance_factor,
        planet,
        broadcast_error=(
            "latitude and solar_longitude_degrees cannot be broadcast together"
        ),
    )


def coriolis_parameter(
    latitude_degrees: Any,
    planet: PlanetConfig,
) -> float | np.ndarray:
    """Return signed Coriolis parameter ``f = 2 Ω sin(latitude)`` in s⁻¹."""

    if not isinstance(planet, PlanetConfig):
        raise TypeError("planet must be a PlanetConfig instance")
    latitude = _as_float_array(latitude_degrees, "latitude_degrees")
    if np.any(latitude < -90.0) or np.any(latitude > 90.0):
        raise ValueError("latitude_degrees must be in [-90, 90]")
    result = 2.0 * planet.angular_velocity_rad_s * np.sin(np.deg2rad(latitude))
    return _scalar_or_array(result)


def derive_stage_seed(
    master_seed: str,
    stage_name: str,
    input_hashes: Sequence[str] = (),
) -> int:
    """Derive a stable unsigned 64-bit seed from stage provenance.

    Components are length-prefixed before hashing to avoid delimiter
    collisions.  No random generator is created or touched.
    """

    if not isinstance(master_seed, str) or not master_seed:
        raise ValueError("master_seed must be a non-empty string")
    if not isinstance(stage_name, str) or not stage_name:
        raise ValueError("stage_name must be a non-empty string")
    if isinstance(input_hashes, (str, bytes)):
        raise TypeError("input_hashes must be a sequence of strings")
    digest_material = bytearray()
    for part in (master_seed, stage_name, *input_hashes):
        if not isinstance(part, str) or not part:
            raise ValueError("seed components must be non-empty strings")
        encoded = part.encode("utf-8")
        digest_material.extend(len(encoded).to_bytes(8, "big", signed=False))
        digest_material.extend(encoded)
    digest = hashlib.sha256(digest_material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


__all__ = [
    "LatLonGrid",
    "PlanetConfig",
    "PlanetConfigError",
    "build_lat_lon_grid",
    "cell_area_m2",
    "cell_area_weights",
    "coriolis_parameter",
    "daily_mean_insolation",
    "daily_mean_insolation_at_solar_longitude",
    "derive_stage_seed",
    "great_circle_distance_km",
    "local_east_north_basis",
    "lon_lat_to_unit_vector",
    "solar_declination",
    "unit_vector_to_lon_lat",
]
