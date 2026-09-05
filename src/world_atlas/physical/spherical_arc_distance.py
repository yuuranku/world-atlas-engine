"""Signed distances and nearest-neighbour queries for spherical minor arcs."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from world_atlas.physical.planetary_grid import lon_lat_to_unit_vector


_ARC_ANGLE_TOLERANCE = 1.0e-12
_NORMAL_TOLERANCE = 1.0e-8
_PROJECTION_EPSILON = 1.0e-14
_SIGN_EPSILON = 1.0e-14
_ZERO_ANGLE_EPSILON = 1.0e-14


def _numeric_array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes, bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalise_rows(value: Any, name: str) -> np.ndarray:
    array = _numeric_array(value, name)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have a final dimension of length 3")
    if array.size == 0:
        return np.array(array, dtype=np.float64, copy=True)

    # Scaling before the norm keeps otherwise valid very large or subnormal
    # vectors from overflowing or underflowing during normalization.
    scale = np.max(np.abs(array), axis=-1)
    if np.any(scale == 0.0):
        raise ValueError(f"{name} contains a zero-length vector")
    scaled = array / scale[..., None]
    norm = np.linalg.norm(scaled, axis=-1)
    if np.any(~np.isfinite(norm)) or np.any(norm == 0.0):
        raise ValueError(f"{name} contains an invalid vector")
    return scaled / norm[..., None]


def _lon_lat_pair(value: Any, name: str) -> tuple[float, float]:
    array = _numeric_array(value, name)
    if array.shape != (2,):
        raise ValueError(f"{name} must be a longitude/latitude pair")
    longitude, latitude = (float(array[0]), float(array[1]))
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"{name} longitude must be in [-180, 180]")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"{name} latitude must be in [-90, 90]")
    return longitude, latitude


def _normalise_normal(value: Any) -> tuple[float, float, float]:
    array = _numeric_array(value, "normal_a_to_b")
    if array.shape != (3,):
        raise ValueError("normal_a_to_b must be a three-dimensional vector")
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        raise ValueError("normal_a_to_b must be non-zero")
    scaled = array / scale
    norm = float(np.linalg.norm(scaled))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("normal_a_to_b must be a finite non-zero vector")
    normalized = scaled / norm
    return tuple(float(component) for component in normalized)


def _validate_radius(radius_km: Any) -> float:
    if isinstance(radius_km, (bool, np.bool_)):
        raise TypeError("radius_km must be a positive finite number")
    try:
        value = np.asarray(radius_km, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise TypeError("radius_km must be a positive finite number") from error
    if value.ndim != 0:
        raise TypeError("radius_km must be a scalar")
    radius = float(value)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_km must be a positive finite number")
    return radius


def _validate_max_distance(max_distance_km: Any) -> float:
    if isinstance(max_distance_km, (bool, np.bool_)):
        raise TypeError("max_distance_km must be a non-negative number")
    try:
        value = np.asarray(max_distance_km, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise TypeError("max_distance_km must be a non-negative number") from error
    if value.ndim != 0:
        raise TypeError("max_distance_km must be a scalar")
    distance = float(value)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("max_distance_km must be a finite non-negative number")
    return distance


def _validate_chunk_size(chunk_size: Any) -> int:
    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(
        chunk_size, (int, np.integer)
    ):
        raise TypeError("chunk_size must be a positive integer")
    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return chunk


@dataclasses.dataclass(frozen=True, slots=True)
class OrientedSphericalArc:
    """A non-degenerate minor great-circle arc with an oriented side normal."""

    arc_id: str
    start_lon_lat: tuple[float, float]
    end_lon_lat: tuple[float, float]
    normal_a_to_b: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.arc_id, str) or not self.arc_id.strip():
            raise ValueError("arc_id must be a non-empty string")
        if self.arc_id.strip() != self.arc_id:
            raise ValueError("arc_id must be trimmed")
        start = _lon_lat_pair(self.start_lon_lat, "start_lon_lat")
        end = _lon_lat_pair(self.end_lon_lat, "end_lon_lat")
        normal_tuple = _normalise_normal(self.normal_a_to_b)

        start_vector = lon_lat_to_unit_vector(*start)
        end_vector = lon_lat_to_unit_vector(*end)
        great_circle_cross = np.cross(start_vector, end_vector)
        cross_norm = float(np.linalg.norm(great_circle_cross))
        central_angle = math.atan2(
            cross_norm,
            float(np.clip(np.dot(start_vector, end_vector), -1.0, 1.0)),
        )
        if (
            central_angle <= _ARC_ANGLE_TOLERANCE
            or math.pi - central_angle <= _ARC_ANGLE_TOLERANCE
        ):
            raise ValueError("arc endpoints must define a non-degenerate minor arc")

        midpoint_sum = start_vector + end_vector
        midpoint = midpoint_sum / np.linalg.norm(midpoint_sum)
        great_circle_normal = great_circle_cross / cross_norm
        midpoint_tangent = np.cross(great_circle_normal, midpoint)
        midpoint_tangent /= np.linalg.norm(midpoint_tangent)
        normal_vector = np.asarray(normal_tuple, dtype=np.float64)
        if abs(float(np.dot(normal_vector, midpoint))) > _NORMAL_TOLERANCE:
            raise ValueError("normal_a_to_b must lie in the midpoint tangent plane")
        if abs(float(np.dot(normal_vector, midpoint_tangent))) > _NORMAL_TOLERANCE:
            raise ValueError("normal_a_to_b must be perpendicular to the arc")

        object.__setattr__(self, "start_lon_lat", start)
        object.__setattr__(self, "end_lon_lat", end)
        object.__setattr__(self, "normal_a_to_b", normal_tuple)


@dataclasses.dataclass(frozen=True, slots=True)
class NearestArcResult:
    """Read-only nearest-arc fields with the query shape (without vector axis)."""

    signed_distance_km: np.ndarray
    absolute_distance_km: np.ndarray
    arc_index: np.ndarray

    def __post_init__(self) -> None:
        signed = np.array(self.signed_distance_km, dtype=np.float64, copy=True)
        absolute = np.array(self.absolute_distance_km, dtype=np.float64, copy=True)
        raw_indices = np.asarray(self.arc_index)
        if raw_indices.dtype.kind not in {"i", "u"}:
            raise ValueError("NearestArcResult arc_index must be an integer array")
        maximum_index = np.iinfo(np.int64).max
        if raw_indices.dtype.kind == "u":
            if np.any(raw_indices > np.uint64(maximum_index)):
                raise ValueError("NearestArcResult arc_index must be -1 or non-negative")
        elif np.any(raw_indices < -1) or np.any(raw_indices > maximum_index):
            raise ValueError("NearestArcResult arc_index must be -1 or non-negative")
        indices = np.array(raw_indices, dtype=np.int64, copy=True)
        if signed.shape != absolute.shape or signed.shape != indices.shape:
            raise ValueError("NearestArcResult arrays must have equal shapes")
        if np.any(np.isnan(signed)) or np.any(np.isnan(absolute)):
            raise ValueError("NearestArcResult distances cannot contain NaN")
        if np.any(absolute < 0.0):
            raise ValueError("NearestArcResult absolute distances must be non-negative")

        miss = indices == -1
        hit = indices >= 0
        if np.any(~(miss | hit)):
            raise ValueError("NearestArcResult arc_index must be -1 or non-negative")
        if np.any(miss & ~(np.isposinf(signed) & np.isposinf(absolute))):
            raise ValueError("NearestArcResult misses must use positive infinity distances")
        if np.any(hit & ~(np.isfinite(signed) & np.isfinite(absolute))):
            raise ValueError("NearestArcResult hits must use finite distances")
        if np.any(
            hit
            & ~np.isclose(
                np.abs(signed),
                absolute,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ):
            raise ValueError("NearestArcResult signed and absolute distances disagree")
        signed.setflags(write=False)
        absolute.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "signed_distance_km", signed)
        object.__setattr__(self, "absolute_distance_km", absolute)
        object.__setattr__(self, "arc_index", indices)


def _arc_geometry(
    arc: OrientedSphericalArc,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    if not isinstance(arc, OrientedSphericalArc):
        raise TypeError("arc must be an OrientedSphericalArc")
    start = lon_lat_to_unit_vector(*arc.start_lon_lat)
    end = lon_lat_to_unit_vector(*arc.end_lon_lat)
    cross = np.cross(start, end)
    cross_norm = float(np.linalg.norm(cross))
    pole = cross / cross_norm
    angle = math.atan2(
        cross_norm,
        float(np.clip(np.dot(start, end), -1.0, 1.0)),
    )
    normal = np.asarray(arc.normal_a_to_b, dtype=np.float64)
    return start, end, pole, angle, normal


def _great_circle_angles(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    cross_norm = np.linalg.norm(np.cross(first, second), axis=-1)
    dot = np.sum(first * second, axis=-1)
    return np.arctan2(cross_norm, np.clip(dot, -1.0, 1.0))


def _signed_distance_for_normalized_vectors(
    vectors: np.ndarray,
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray],
    radius_km: float,
) -> np.ndarray:
    start, end, pole, arc_angle, normal = geometry
    pole_component = np.sum(vectors * pole, axis=-1)
    projection = vectors - pole_component[..., None] * pole
    projection_norm = np.linalg.norm(projection, axis=-1)
    safe_projection_norm = np.where(
        projection_norm > _PROJECTION_EPSILON,
        projection_norm,
        1.0,
    )
    projected = projection / safe_projection_norm[..., None]
    start_projection_angle = _great_circle_angles(start, projected)
    projection_end_angle = _great_circle_angles(projected, end)
    projection_inside = (
        (projection_norm > _PROJECTION_EPSILON)
        & (
            start_projection_angle + projection_end_angle
            <= arc_angle + _ARC_ANGLE_TOLERANCE
        )
    )

    cross_track_angle = np.arctan2(np.abs(pole_component), projection_norm)
    start_cap_angle = _great_circle_angles(vectors, start)
    end_cap_angle = _great_circle_angles(vectors, end)
    endpoint_angle = np.minimum(start_cap_angle, end_cap_angle)

    # The validated normal is parallel to the great-circle pole.  Its dot
    # product with the pole selects which side is positive, while a near-zero
    # component deliberately defaults to positive zero rather than -0.0.
    normal_pole_orientation = float(np.dot(normal, pole))
    oriented_pole_component = pole_component * (
        1.0 if normal_pole_orientation >= 0.0 else -1.0
    )
    interior_sign = np.where(oriented_pole_component < -_SIGN_EPSILON, -1.0, 1.0)
    interior_signed_angle = interior_sign * cross_track_angle

    normal_component = np.sum(vectors * normal, axis=-1)
    cap_sign = np.where(normal_component < -_SIGN_EPSILON, -1.0, 1.0)
    cap_signed_angle = cap_sign * endpoint_angle
    signed_angle = np.where(
        projection_inside,
        interior_signed_angle,
        cap_signed_angle,
    )
    signed_angle = np.where(
        np.abs(signed_angle) <= _ZERO_ANGLE_EPSILON,
        0.0,
        signed_angle,
    )
    return np.asarray(signed_angle * radius_km, dtype=np.float64)


def signed_distance_to_arc(
    unit_vectors: Any,
    arc: OrientedSphericalArc,
    radius_km: Any,
) -> np.ndarray:
    """Return signed geodesic distance in km from points to a minor arc."""

    radius = _validate_radius(radius_km)
    geometry = _arc_geometry(arc)
    vectors = _normalise_rows(unit_vectors, "unit_vectors")
    result = _signed_distance_for_normalized_vectors(vectors, geometry, radius)
    return np.asarray(result, dtype=np.float64).reshape(vectors.shape[:-1])


def nearest_oriented_arc(
    unit_vectors: Any,
    arcs: Any,
    radius_km: Any,
    max_distance_km: Any,
    chunk_size: Any,
) -> NearestArcResult:
    """Return the closest oriented arc using bounded NumPy work chunks."""

    radius = _validate_radius(radius_km)
    max_distance = _validate_max_distance(max_distance_km)
    chunk = _validate_chunk_size(chunk_size)
    vectors = _normalise_rows(unit_vectors, "unit_vectors")
    if isinstance(arcs, (str, bytes)):
        raise TypeError("arcs must be a sequence of OrientedSphericalArc values")
    try:
        arc_values = tuple(arcs)
    except TypeError as error:
        raise TypeError("arcs must be a sequence of OrientedSphericalArc values") from error

    prepared = []
    seen_ids: set[str] = set()
    for index, arc in enumerate(arc_values):
        geometry = _arc_geometry(arc)
        if arc.arc_id in seen_ids:
            raise ValueError("arcs must not contain duplicate arc_id values")
        seen_ids.add(arc.arc_id)
        prepared.append((geometry, index))
    prepared.sort(key=lambda item: (arc_values[item[1]].arc_id, item[1]))

    query_shape = vectors.shape[:-1]
    flat_vectors = vectors.reshape((-1, 3))
    flat_count = flat_vectors.shape[0]
    output_signed = np.empty(flat_count, dtype=np.float64)
    output_absolute = np.empty(flat_count, dtype=np.float64)
    output_index = np.empty(flat_count, dtype=np.int64)

    for start_index in range(0, flat_count, chunk):
        stop_index = min(start_index + chunk, flat_count)
        vector_chunk = flat_vectors[start_index:stop_index]
        best_signed = np.full(vector_chunk.shape[0], np.inf, dtype=np.float64)
        best_absolute = np.full(vector_chunk.shape[0], np.inf, dtype=np.float64)
        best_index = np.full(vector_chunk.shape[0], -1, dtype=np.int64)

        # Arcs are visited in arc_id order, so strict improvement preserves
        # the required lexicographic tie-break without an arcs-sized matrix.
        for geometry, original_index in prepared:
            candidate_signed = _signed_distance_for_normalized_vectors(
                vector_chunk,
                geometry,
                radius,
            )
            candidate_absolute = np.abs(candidate_signed)
            better = candidate_absolute < best_absolute
            best_signed[better] = candidate_signed[better]
            best_absolute[better] = candidate_absolute[better]
            best_index[better] = original_index

        hit = best_absolute <= max_distance
        output_signed[start_index:stop_index] = np.where(hit, best_signed, np.inf)
        output_absolute[start_index:stop_index] = np.where(
            hit,
            best_absolute,
            np.inf,
        )
        output_index[start_index:stop_index] = np.where(hit, best_index, -1)

    return NearestArcResult(
        output_signed.reshape(query_shape),
        output_absolute.reshape(query_shape),
        output_index.reshape(query_shape),
    )


__all__ = [
    "NearestArcResult",
    "OrientedSphericalArc",
    "nearest_oriented_arc",
    "signed_distance_to_arc",
]
