"""Ground-scaled river channels on the canonical planetary drainage graph."""

from __future__ import annotations

import math

import numpy as np
import shapely

from world_atlas.physical.hydrology import accumulate_d8_runoff

from .model import WorldGrid


# The source's discharge is accumulated land-cell equivalents, not m³/s.
# Downstream hydraulic geometry supports a power law, but its coefficients
# depend on the river: https://pubs.usgs.gov/publication/pp252 . Until measured
# discharge/channel observations exist, this explicit drainage-area estimate
# gives 50 m at 10,000 km² and 500 m at 1,000,000 km². It is not an assertion
# of measured bankfull width. No world-maximum normalization or screen units.
WIDTH_METRES_PER_SQRT_CATCHMENT_KM2 = 0.5
RIVER_WIDTH_MODEL = "estimated-0.5-metres-times-sqrt-canonical-catchment-km2"


def _planet_geometry(grid: WorldGrid) -> tuple[float, float, float, float]:
    """Return radius metres, cell angular spans, and north edge radians."""

    height, width = grid.shape
    radius = float(grid.metadata["planet"]["radiusKm"]) * 1000.0
    extents = grid.metadata["extents"]
    west, east, north, south = (
        float(extents[key]) for key in ("west", "east", "north", "south")
    )
    if (
        not all(math.isfinite(value) for value in (radius, west, east, north, south))
        or radius <= 0 or not 0 < east - west <= 360
        or not -90 <= south < north <= 90
        or height < 1 or width < 1
    ):
        raise ValueError("river channels require valid planetary radius and geographic extents")
    return (
        radius, math.radians(east - west) / width,
        math.radians(north - south) / height, math.radians(north),
    )


def river_width_field(grid: WorldGrid) -> np.ndarray:
    """Estimate channel width in metres from exact upstream land area.

    Spherical cell areas remove the latitude and resolution dependence of
    the stored cell-count accumulation. They are added on the unchanged D8
    graph, so the width grows downstream and includes both tributaries at a
    confluence. Non-channel cells are zero. Physical source arrays are never
    modified. This returns an estimated reference channel, not seasonal flow.
    """

    radius, longitude_step, latitude_step, north = _planet_geometry(grid)
    latitude_edges = north - np.arange(grid.shape[0] + 1) * latitude_step
    row_area_km2 = (radius / 1000.0) ** 2 * longitude_step * (
        np.sin(latitude_edges[:-1]) - np.sin(latitude_edges[1:])
    )
    land = grid.water == 0
    source = np.where(land, row_area_km2[:, None], 0.0)
    catchment_km2 = accumulate_d8_runoff(source, land, grid.flow_to)
    np.sqrt(catchment_km2, out=catchment_km2)
    catchment_km2 *= WIDTH_METRES_PER_SQRT_CATCHMENT_KM2
    catchment_km2[(grid.river_order == 0) | ~land] = 0.0
    return catchment_km2


def _points(points: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or len(result) < 2:
        raise ValueError(f"{name} must contain at least two planar points")
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} must contain finite coordinates")
    if np.any(np.linalg.norm(np.diff(result, axis=0), axis=1) == 0):
        raise ValueError(f"{name} cannot contain consecutive repeated points")
    return result


def river_width_profile(
    grid: WorldGrid,
    source_points: np.ndarray,
    curve_points: np.ndarray,
    widths_m: np.ndarray,
) -> np.ndarray:
    """Transfer the canonical reach widths onto its display curve.

    Widths are sampled at source drainage cells, then interpolated by the
    nearest position along that original reach. Curved points are never used
    to sample the raster: they may lie between channel cells. A snapped mouth
    outside the last dry channel retains its upstream width. Shared reach
    endpoints retain exactly the same canonical junction width.
    """

    source = _points(source_points, "source_points")
    curve = _points(curve_points, "curve_points")
    field = np.asarray(widths_m)
    if field.shape != grid.shape:
        raise ValueError("widths_m must match the grid")
    if not np.allclose(curve[[0, -1]], source[[0, -1]], rtol=0, atol=1e-9):
        raise ValueError("display curves must preserve their canonical reach endpoints")
    columns = np.clip(np.floor(source[:, 0]).astype(np.int64), 0, grid.shape[1] - 1)
    rows = np.clip(np.floor(source[:, 1]).astype(np.int64), 0, grid.shape[0] - 1)
    sampled = field[rows, columns].astype(np.float64)
    if not bool(np.isfinite(sampled).all()) or np.any(sampled < 0):
        raise ValueError("reach widths must be finite and non-negative")
    if np.any(sampled[:-1] <= 0):
        raise ValueError("source reach must follow canonical river cells")
    # The last point can be a fractional shoreline anchor in a wet cell or
    # an unrelated dry cell. Only native channel centers supply a new width.
    last_is_native = bool(np.allclose(source[-1] % 1, 0.5, rtol=0, atol=1e-9))
    if sampled[-1] <= 0 or not last_is_native:
        sampled[-1] = sampled[-2]
    source_distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(source, axis=0), axis=1))]
    positions = shapely.line_locate_point(shapely.LineString(source), shapely.points(curve))
    positions = np.maximum.accumulate(positions)
    positions[0], positions[-1] = 0.0, source_distance[-1]
    return np.interp(positions, source_distance, sampled)


def river_channel_surface(
    grid: WorldGrid,
    curve_points: np.ndarray,
    profile_m: np.ndarray,
) -> shapely.Geometry:
    """Build a variable-width channel polygon in native map coordinates.

    Width is measured normal to the channel in local metres, including the
    longitude cosine correction. Rendering this filled surface makes ground
    width grow naturally with zoom. Overview minimum ink belongs to a separate
    centerline renderer. Round end caps overlap at tributary junctions; the
    caller clips the resulting geometry to the shared land silhouette.
    """

    curve = _points(curve_points, "curve_points")
    widths = np.asarray(profile_m, dtype=np.float64)
    if widths.shape != (len(curve),) or not bool(np.isfinite(widths).all()) or np.any(widths <= 0):
        raise ValueError("profile_m must contain one finite positive width per curve point")
    radius, longitude_step, latitude_step, north = _planet_geometry(grid)
    latitude = north - curve[:, 1] * latitude_step
    # A pole has no longitude metric. A frame endpoint uses the adjacent
    # native row's metric rather than dividing by zero at the projection edge.
    latitude = np.clip(latitude, -math.pi / 2 + latitude_step / 2, math.pi / 2 - latitude_step / 2)
    x_metres = radius * longitude_step * np.cos(latitude)
    y_metres = radius * latitude_step
    tangent = np.gradient(curve, axis=0)
    tangent[:, 0] *= x_metres
    tangent[:, 1] *= y_metres
    length = np.linalg.norm(tangent, axis=1)
    if np.any(length <= 0):
        raise ValueError("river display curves cannot reverse exactly at a point")
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / length[:, None]
    offset = normal * widths[:, None] / 2
    offset[:, 0] /= x_metres
    offset[:, 1] /= y_metres
    strip = shapely.Polygon(np.vstack((curve + offset, (curve - offset)[::-1])))
    angles = np.linspace(0.0, math.tau, 24, endpoint=False)
    unit_cap = np.column_stack((np.cos(angles), np.sin(angles)))
    caps = [
        shapely.Polygon(curve[index] + unit_cap * (
            widths[index] / (2 * x_metres[index]), widths[index] / (2 * y_metres),
        ))
        for index in (0, -1)
    ]
    if not shapely.is_valid(strip):
        strip = shapely.make_valid(strip)
    return shapely.union_all((strip, *caps))
