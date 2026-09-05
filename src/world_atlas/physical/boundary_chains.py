"""Spherical boundary-chain assembly and bounded geometry smoothing.

The tectonic stage emits one segment per shared raster edge.  This module
turns those edges into deterministic chains without treating longitude as a
flat x coordinate.

Only the topology is merged here.  Smoothing never inserts or removes a
point: it applies a small three-point weighted average on the unit sphere and
limits every point's total movement from its original position.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from dataclasses import dataclass
from typing import Any, Iterable

from world_atlas.physical.planetary_grid import lon_lat_to_unit_vector, unit_vector_to_lon_lat
from world_atlas.physical.tectonics import BoundarySegment


_NODE_DECIMALS = 12
_POLE_TOLERANCE = 1.0e-10
_VECTOR_EPSILON = 1.0e-15


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_number(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_number(value: Any, name: str) -> float:
    result = _number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _point(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a longitude/latitude pair")
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a longitude/latitude pair") from error
    if len(values) != 2:
        raise ValueError(f"{name} must contain longitude and latitude")
    longitude = _number(values[0], f"{name}.longitude")
    latitude = _number(values[1], f"{name}.latitude")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"{name}.latitude must be in [-90, 90]")
    longitude = ((longitude + 180.0) % 360.0) - 180.0
    if abs(abs(latitude) - 90.0) <= _POLE_TOLERANCE:
        return (0.0, 90.0 if latitude >= 0.0 else -90.0)
    return (longitude, latitude)


def _vector(point: tuple[float, float]) -> tuple[float, float, float]:
    converted = lon_lat_to_unit_vector(point[0], point[1])
    return tuple(float(value) for value in converted)


def _normalise(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= _VECTOR_EPSILON:
        raise ValueError("unit vector cannot have zero length")
    return tuple(value / length for value in vector)


def _point_from_vector(vector: tuple[float, float, float]) -> tuple[float, float]:
    normalised = _normalise(vector)
    longitude, latitude = unit_vector_to_lon_lat(normalised)
    return (float(longitude), float(latitude))


def _canonical_node(point: Any) -> tuple[tuple[float, float, float, float], tuple[float, float, float], tuple[float, float]]:
    canonical = _point(point, "point")
    vector = _normalise(_vector(canonical))
    if abs(abs(vector[2]) - 1.0) <= _POLE_TOLERANCE:
        sign = 1.0 if vector[2] >= 0.0 else -1.0
        pole_vector = (0.0, 0.0, sign)
        return (1.0, sign, 0.0, 0.0), pole_vector, (0.0, 90.0 if sign > 0.0 else -90.0)
    key = (0.0,) + tuple(round(value, _NODE_DECIMALS) for value in vector)
    return key, vector, _point_from_vector(vector)


def _dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(first, second))


def _cross(first: tuple[float, float, float], second: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _angle(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    cross_length = math.sqrt(sum(value * value for value in _cross(first, second)))
    return math.atan2(cross_length, max(-1.0, min(1.0, _dot(first, second))))


def _unit_sum(values: Iterable[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    total = tuple(sum(value[index] for value in values) for index in range(3))
    length = math.sqrt(sum(value * value for value in total))
    if length <= _VECTOR_EPSILON:
        return None
    return tuple(value / length for value in total)


def _slerp(first: tuple[float, float, float], second: tuple[float, float, float], fraction: float) -> tuple[float, float, float]:
    fraction = max(0.0, min(1.0, fraction))
    first = _normalise(first)
    second = _normalise(second)
    dot = max(-1.0, min(1.0, _dot(first, second)))
    angle = math.acos(dot)
    if angle <= 1.0e-12:
        return _normalise(tuple((1.0 - fraction) * a + fraction * b for a, b in zip(first, second)))
    if math.pi - angle <= 1.0e-10:
        # Pick a deterministic tangent for the antipodal case.
        basis = min(((abs(first[index]), index) for index in range(3)), key=lambda item: item[0])[1]
        axis_values = [0.0, 0.0, 0.0]
        axis_values[basis] = 1.0
        axis = _normalise(_cross(first, tuple(axis_values)))
        return _normalise(tuple(math.cos(math.pi * fraction) * first[index] + math.sin(math.pi * fraction) * axis[index] for index in range(3)))
    sine = math.sin(angle)
    first_weight = math.sin((1.0 - fraction) * angle) / sine
    second_weight = math.sin(fraction * angle) / sine
    return _normalise(tuple(first_weight * a + second_weight * b for a, b in zip(first, second)))


def _clamp_from_original(
    original: tuple[float, float, float],
    candidate: tuple[float, float, float],
    maximum_angle: float,
) -> tuple[float, float, float]:
    original = _normalise(original)
    candidate = _normalise(candidate)
    if maximum_angle <= 0.0:
        return original
    displacement = _angle(original, candidate)
    if displacement <= maximum_angle + 1.0e-12:
        return candidate
    return _slerp(original, candidate, maximum_angle / displacement)


def _path_length(points: tuple[tuple[float, float], ...], radius_km: float) -> float:
    length = 0.0
    for first, second in zip(points, points[1:]):
        length += _angle(_vector(first), _vector(second)) * radius_km
    return length


def _validate_plate_pair(value: Any, name: str) -> tuple[str, str]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain two plate IDs")
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must contain two plate IDs") from error
    if len(values) != 2 or any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{name} must contain two non-empty plate IDs")
    if values[0] == values[1]:
        raise ValueError(f"{name} must contain two distinct plate IDs")
    return tuple(sorted(values))  # type: ignore[return-value]


def _validate_source_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class BoundaryChain:
    """One maximal chain of shared boundary segments.

    ``points_lon_lat`` contains one endpoint per segment plus the final
    endpoint.  A closed loop therefore repeats its first point at the end.
    ``source_segment_ids`` is the exact, ordered, lossless source ledger.
    """

    chain_id: str
    plate_ids: tuple[str, str]
    points_lon_lat: tuple[tuple[float, float], ...]
    source_segment_ids: tuple[str, ...]
    closed: bool
    length_km: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_id", _validate_source_id(self.chain_id, "chain_id"))
        object.__setattr__(self, "plate_ids", _validate_plate_pair(self.plate_ids, "plate_ids"))
        if not isinstance(self.closed, bool):
            raise TypeError("closed must be a boolean")
        try:
            raw_points = tuple(self.points_lon_lat)
        except TypeError as error:
            raise TypeError("points_lon_lat must be a sequence") from error
        if len(raw_points) < 2:
            raise ValueError("points_lon_lat must contain at least two points")
        canonical_points = tuple(_point(value, f"points_lon_lat[{index}]") for index, value in enumerate(raw_points))
        try:
            source_ids = tuple(self.source_segment_ids)
        except TypeError as error:
            raise TypeError("source_segment_ids must be a sequence") from error
        if len(source_ids) != len(canonical_points) - 1:
            raise ValueError("source_segment_ids must contain one ID per segment")
        if any(not isinstance(value, str) or not value.strip() for value in source_ids):
            raise ValueError("source_segment_ids must contain non-empty strings")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source_segment_ids must be unique within a chain")
        if self.closed and _canonical_node(canonical_points[0])[0] != _canonical_node(canonical_points[-1])[0]:
            raise ValueError("closed chains must repeat their first point at the end")
        length = _non_negative_number(self.length_km, "length_km")
        object.__setattr__(self, "points_lon_lat", canonical_points)
        object.__setattr__(self, "source_segment_ids", source_ids)
        object.__setattr__(self, "length_km", length)


@dataclass(frozen=True, slots=True)
class _Edge:
    source_id: str
    plate_ids: tuple[str, str]
    start_key: tuple[float, float, float, float]
    end_key: tuple[float, float, float, float]
    start_point: tuple[float, float]
    end_point: tuple[float, float]


def _read_segment(segment: Any) -> _Edge:
    if not isinstance(segment, BoundarySegment):
        raise TypeError("segments must contain tectonics.BoundarySegment values")
    try:
        source_id = _validate_source_id(getattr(segment, "boundary_id"), "boundary_id")
        plate_ids = _validate_plate_pair(getattr(segment, "plate_ids"), "plate_ids")
        start_key, _start_vector, start_point = _canonical_node(getattr(segment, "start_lon_lat"))
        end_key, _end_vector, end_point = _canonical_node(getattr(segment, "end_lon_lat"))
    except AttributeError as error:
        raise TypeError("each segment must expose boundary_id, plate_ids, start_lon_lat, and end_lon_lat") from error
    if start_key == end_key:
        raise ValueError("a boundary segment must have distinct endpoints")
    return _Edge(source_id, plate_ids, start_key, end_key, start_point, end_point)


def _edge_sort_key(edge: _Edge, node: tuple[float, float, float, float]) -> tuple[str, tuple[float, float, float, float], tuple[float, float, float, float]]:
    other = edge.end_key if edge.start_key == node else edge.start_key
    return edge.source_id, other, edge.start_key if edge.start_key <= edge.end_key else edge.end_key


def _chain_id(plate_ids: tuple[str, str], source_ids: tuple[str, ...]) -> str:
    payload = "\0".join((*plate_ids, *source_ids)).encode("utf-8")
    return "chain-" + hashlib.sha256(payload).hexdigest()[:20]


def _walk_chain(
    edges: tuple[_Edge, ...],
    incidents: dict[tuple[float, float, float, float], tuple[int, ...]],
    node_points: dict[tuple[float, float, float, float], tuple[float, float]],
    start_node: tuple[float, float, float, float],
    first_edge_index: int,
    visited: set[int],
    radius_km: float,
) -> BoundaryChain:
    points = [node_points[start_node]]
    source_ids: list[str] = []
    current_node = start_node
    edge_index = first_edge_index
    while True:
        if edge_index in visited:
            raise RuntimeError("boundary traversal attempted to consume a segment twice")
        visited.add(edge_index)
        edge = edges[edge_index]
        next_node = edge.end_key if edge.start_key == current_node else edge.start_key
        source_ids.append(edge.source_id)
        points.append(node_points[next_node])
        if next_node == start_node:
            break
        if len(incidents[next_node]) != 2:
            break
        candidates = [candidate for candidate in incidents[next_node] if candidate not in visited]
        if not candidates:
            break
        current_node = next_node
        edge_index = min(candidates, key=lambda index: _edge_sort_key(edges[index], current_node))

    is_closed = next_node == start_node and len(incidents[start_node]) == 2
    points_tuple = tuple(points)
    source_tuple = tuple(source_ids)
    if not is_closed and source_tuple > tuple(reversed(source_tuple)):
        points_tuple = tuple(reversed(points_tuple))
        source_tuple = tuple(reversed(source_tuple))
    return BoundaryChain(
        _chain_id(edges[first_edge_index].plate_ids, source_tuple),
        edges[first_edge_index].plate_ids,
        points_tuple,
        source_tuple,
        is_closed,
        _path_length(points_tuple, radius_km),
    )


def merge_boundary_segments(segments: Iterable[BoundarySegment], radius_km: float) -> tuple[BoundaryChain, ...]:
    """Merge each shared plate-pair edge exactly once into deterministic chains."""

    radius = _positive_number(radius_km, "radius_km")
    if isinstance(segments, (str, bytes)) or segments is None:
        raise TypeError("segments must be an iterable of boundary segment objects")
    try:
        values = tuple(segments)
    except TypeError as error:
        raise TypeError("segments must be an iterable of boundary segment objects") from error
    edges_by_pair: dict[tuple[str, str], list[_Edge]] = {}
    seen_ids: set[str] = set()
    for value in values:
        edge = _read_segment(value)
        if edge.source_id in seen_ids:
            raise ValueError("boundary_id values must be unique")
        seen_ids.add(edge.source_id)
        edges_by_pair.setdefault(edge.plate_ids, []).append(edge)

    chains: list[BoundaryChain] = []
    for pair in sorted(edges_by_pair):
        ordered_edges = tuple(sorted(edges_by_pair[pair], key=lambda edge: (edge.source_id, edge.start_key, edge.end_key)))
        node_points: dict[tuple[float, float, float, float], tuple[float, float]] = {}
        incident_lists: dict[tuple[float, float, float, float], list[int]] = {}
        for index, edge in enumerate(ordered_edges):
            node_points[edge.start_key] = edge.start_point
            node_points[edge.end_key] = edge.end_point
            incident_lists.setdefault(edge.start_key, []).append(index)
            incident_lists.setdefault(edge.end_key, []).append(index)
        incidents = {node: tuple(sorted(indexes, key=lambda index: _edge_sort_key(ordered_edges[index], node))) for node, indexes in incident_lists.items()}
        visited: set[int] = set()
        start_nodes = tuple(sorted(node for node, indexes in incidents.items() if len(indexes) != 2))
        for start_node in start_nodes:
            for edge_index in incidents[start_node]:
                if edge_index not in visited:
                    chains.append(_walk_chain(ordered_edges, incidents, node_points, start_node, edge_index, visited, radius))
        for start_node in sorted(incidents):
            if any(edge_index not in visited for edge_index in incidents[start_node]):
                edge_index = min(
                    (candidate for candidate in incidents[start_node] if candidate not in visited),
                    key=lambda index: _edge_sort_key(ordered_edges[index], start_node),
                )
                chains.append(_walk_chain(ordered_edges, incidents, node_points, start_node, edge_index, visited, radius))
        if len(visited) != len(ordered_edges):
            raise RuntimeError("boundary traversal did not consume every source segment")

    chains.sort(key=lambda chain: (chain.plate_ids, chain.source_segment_ids, chain.chain_id))
    return tuple(chains)


def _validate_passes(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("smoothing_passes must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("smoothing_passes must be a non-negative integer")
    return result


def smooth_boundary_chains(
    chains: Iterable[BoundaryChain],
    radius_km: float,
    smoothing_passes: int,
    max_displacement_km: float,
) -> tuple[BoundaryChain, ...]:
    """Smooth chain interiors on the unit sphere with bounded displacement."""

    radius = _positive_number(radius_km, "radius_km")
    passes = _validate_passes(smoothing_passes)
    maximum_displacement = _non_negative_number(max_displacement_km, "max_displacement_km")
    if isinstance(chains, (str, bytes)) or chains is None:
        raise TypeError("chains must be an iterable of BoundaryChain values")
    try:
        values = tuple(chains)
    except TypeError as error:
        raise TypeError("chains must be an iterable of BoundaryChain values") from error
    if any(not isinstance(chain, BoundaryChain) for chain in values):
        raise TypeError("chains must contain only BoundaryChain values")
    if len({chain.chain_id for chain in values}) != len(values):
        raise ValueError("chain_id values must be unique")
    if passes == 0 or not values:
        return tuple(values)

    maximum_angle = maximum_displacement / radius
    smoothed: list[BoundaryChain] = []
    for chain in values:
        original = tuple(_vector(point) for point in chain.points_lon_lat)
        current = list(original)
        point_count = len(current)
        if chain.closed:
            unique_count = point_count - 1
            update_indices = range(unique_count)
        else:
            update_indices = range(1, point_count - 1)
        fixed = {index for index, point in enumerate(original) if abs(abs(point[2]) - 1.0) <= _POLE_TOLERANCE}
        if not chain.closed:
            fixed.update((0, point_count - 1))
        else:
            # A closed chain still has a duplicated serialized endpoint.
            if 0 in fixed:
                fixed.update((0, point_count - 1))

        for _pass in range(passes):
            updated = list(current)
            for index in update_indices:
                if index in fixed:
                    continue
                if chain.closed:
                    previous = current[(index - 1) % (point_count - 1)]
                    following = current[(index + 1) % (point_count - 1)]
                else:
                    previous = current[index - 1]
                    following = current[index + 1]
                current_point = current[index]
                candidate = _unit_sum(
                    (
                        previous,
                        tuple(2.0 * value for value in current_point),
                        following,
                    )
                )
                if candidate is not None:
                    updated[index] = _clamp_from_original(original[index], candidate, maximum_angle)
            if chain.closed:
                updated[-1] = updated[0]
            current = updated

        points = tuple(_point_from_vector(value) for value in current)
        smoothed.append(
            BoundaryChain(
                chain.chain_id,
                chain.plate_ids,
                points,
                chain.source_segment_ids,
                chain.closed,
                _path_length(points, radius),
            )
        )
    return tuple(smoothed)


__all__ = ["BoundaryChain", "merge_boundary_segments", "smooth_boundary_chains"]
