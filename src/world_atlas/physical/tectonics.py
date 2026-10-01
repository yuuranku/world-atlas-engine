"""Deterministic spherical plate topology and Euler-pole motion.

Task 3 intentionally stops at topology and kinematics.  The module has no
manifest, DEM, corridor, planar-geometry, or implicit planetary defaults.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from world_atlas.physical.planetary_grid import LatLonGrid, lon_lat_to_unit_vector, unit_vector_to_lon_lat
from world_atlas.physical.spherical_arc_distance import OrientedSphericalArc, nearest_oriented_arc

_EPS = 1.0e-12
_VECTOR_EPS = 1.0e-14
_CLASSIFICATION_THRESHOLD = 0.1
_CORRIDOR_CHUNK_SIZE = 65536
def _float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result
def _coordinate(value: Any, name: str, *, latitude: bool = False) -> float:
    result = _float(value, name)
    if latitude:
        if not -90.0 <= result <= 90.0:
            raise ValueError(f"{name} must be in [-90, 90]")
    elif not -180.0 <= result <= 180.0:
        raise ValueError(f"{name} must be in [-180, 180]")
    if not latitude and math.isclose(result, 180.0, abs_tol=_EPS):
        return -180.0
    return result
def _seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError("stage_seed must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("stage_seed must be non-negative")
    return result
def _hash(*parts: Any) -> bytes:
    payload = bytearray()
    for part in parts:
        encoded = str(part).encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    return hashlib.sha256(payload).digest()
def _vectors(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if array.size == 0:
        if array.ndim == 1:
            return np.empty((0, 3), dtype=np.float64)
        if array.ndim == 2 and array.shape[-1] == 3:
            return array
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have final dimension 3")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    norm = np.linalg.norm(array, axis=-1)
    if np.any(norm <= _VECTOR_EPS):
        raise ValueError(f"{name} contains a zero-length vector")
    return array / norm[..., None]
def _frozen_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result
@dataclass(frozen=True, slots=True)
class PlateDefinition:
    """Stable plate identity and its spherical Euler motion parameters."""

    plate_id: str
    anchor_lon: float
    anchor_lat: float
    euler_pole_lon: float
    euler_pole_lat: float
    angular_speed_deg_per_myr: float
    major: bool

    def __post_init__(self) -> None:
        if not isinstance(self.plate_id, str) or not self.plate_id.strip():
            raise ValueError("plate_id must be a non-empty string")
        if not isinstance(self.major, (bool, np.bool_)):
            raise TypeError("major must be a boolean")
        object.__setattr__(self, "anchor_lon", _coordinate(self.anchor_lon, "anchor_lon"))
        object.__setattr__(self, "anchor_lat", _coordinate(self.anchor_lat, "anchor_lat", latitude=True))
        object.__setattr__(self, "euler_pole_lon", _coordinate(self.euler_pole_lon, "euler_pole_lon"))
        object.__setattr__(self, "euler_pole_lat", _coordinate(self.euler_pole_lat, "euler_pole_lat", latitude=True))
        object.__setattr__(self, "angular_speed_deg_per_myr", _float(self.angular_speed_deg_per_myr, "angular_speed_deg_per_myr"))
        object.__setattr__(self, "major", bool(self.major))
@dataclass(frozen=True, slots=True)
class BoundaryCorridor:
    corridor_id: str
    plate_ids: tuple[str, str]
    controls_lon_lat: tuple[tuple[float, float], ...]
    influence_width_km: float
    strength: float

    def __post_init__(self) -> None:
        if not isinstance(self.corridor_id, str) or not self.corridor_id.strip():
            raise ValueError("corridor_id must be a non-empty string")
        object.__setattr__(self, "corridor_id", self.corridor_id.strip())
        if isinstance(self.plate_ids, (str, bytes)):
            raise TypeError("plate_ids must contain two IDs")
        try:
            pair = tuple(self.plate_ids)
        except TypeError as error:
            raise TypeError("plate_ids must contain two IDs") from error
        if len(pair) != 2 or any(not isinstance(value, str) or not value.strip() for value in pair):
            raise ValueError("plate_ids must contain two non-empty IDs")
        if any(value.strip() != value for value in pair):
            raise ValueError("plate_ids must contain trimmed IDs")
        if pair[0] == pair[1]:
            raise ValueError("plate_ids must contain two different IDs")
        if pair != tuple(sorted(pair)):
            raise ValueError("plate_ids must be in canonical order")
        if isinstance(self.controls_lon_lat, (str, bytes)):
            raise TypeError("controls_lon_lat must be a sequence of points")
        try:
            raw_controls = tuple(self.controls_lon_lat)
        except TypeError as error:
            raise TypeError("controls_lon_lat must be a sequence of points") from error
        if len(raw_controls) < 2:
            raise ValueError("controls_lon_lat must contain at least two points")
        controls: list[tuple[float, float]] = []
        for point in raw_controls:
            if isinstance(point, (str, bytes)):
                raise TypeError("each corridor control must be a longitude/latitude pair")
            try:
                pair_values = tuple(point)
            except TypeError as error:
                raise TypeError("each corridor control must be a longitude/latitude pair") from error
            if len(pair_values) != 2:
                raise ValueError("each corridor control must be a longitude/latitude pair")
            controls.append((_coordinate(pair_values[0], "corridor longitude"), _coordinate(pair_values[1], "corridor latitude", latitude=True)))
        for first, second in zip(controls, controls[1:]):
            first_vector = lon_lat_to_unit_vector(*first)
            second_vector = lon_lat_to_unit_vector(*second)
            angle = math.atan2(float(np.linalg.norm(np.cross(first_vector, second_vector))), float(np.dot(first_vector, second_vector)))
            if angle <= _EPS or math.pi - angle <= _EPS:
                raise ValueError("consecutive corridor controls must define a minor non-degenerate arc")
        width = _float(self.influence_width_km, "influence_width_km")
        strength = _float(self.strength, "strength")
        if width <= 0.0 or strength <= 0.0:
            raise ValueError("corridor width and strength must be positive")
        object.__setattr__(self, "plate_ids", (pair[0], pair[1]))
        object.__setattr__(self, "controls_lon_lat", tuple(controls))
        object.__setattr__(self, "influence_width_km", width)
        object.__setattr__(self, "strength", strength)
@dataclass(frozen=True, slots=True)
class PolePlateAssignments:
    """IDs assigned to north and south mathematical poles."""

    north: str
    south: str

    def __post_init__(self) -> None:
        for name in ("north", "south"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} pole plate ID must be a non-empty string")
@dataclass(frozen=True, slots=True)
class TripleJunction:
    junction_id: str
    longitude_degrees: float
    latitude_degrees: float
    boundary_ids: tuple[str, ...]
    plate_ids: tuple[str, ...]

    @property
    def branch_count(self) -> int:
        return len(self.boundary_ids)
@dataclass(frozen=True, slots=True)
class BoundarySegment:
    boundary_id: str
    plate_ids: tuple[str, str]
    start_lon_lat: tuple[float, float]
    end_lon_lat: tuple[float, float]
    midpoint_lon_lat: tuple[float, float]
    length_km: float
    cell_indices: tuple[int, int]
    boundary_normal: tuple[float, float, float]
    boundary_tangent: tuple[float, float, float]
    relative_velocity_cm_per_year: tuple[float, float, float]
    normal_velocity_cm_per_year: float
    tangent_velocity_cm_per_year: float
    classification: str
@dataclass(frozen=True, slots=True)
class PlateFields:
    plate_grid: np.ndarray
    pole_plate_assignments: PolePlateAssignments
    plates: tuple[PlateDefinition, ...]
    core_vectors: np.ndarray
    boundaries: tuple[BoundarySegment, ...]
    triple_junctions: tuple[TripleJunction, ...]
    velocity_vectors_cm_per_year: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        grid = _frozen_array(self.plate_grid, dtype=object)
        cores = _frozen_array(self.core_vectors, dtype=np.float64)
        velocities = _frozen_array(self.velocity_vectors_cm_per_year, dtype=np.float64)
        if grid.ndim != 2:
            raise ValueError("plate_grid must be two-dimensional")
        if cores.ndim != 2 or cores.shape[-1] != 3:
            raise ValueError("core_vectors must have shape (plate_count, 3)")
        if velocities.shape != grid.shape + (3,):
            raise ValueError("velocity_vectors_cm_per_year shape mismatch")
        object.__setattr__(self, "plate_grid", grid)
        object.__setattr__(self, "core_vectors", cores)
        object.__setattr__(self, "velocity_vectors_cm_per_year", velocities)
        object.__setattr__(self, "plates", tuple(self.plates))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "triple_junctions", tuple(self.triple_junctions))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
    @property
    def plate_count(self) -> int:
        return len(self.plates)
def _validate_grid(grid: Any) -> LatLonGrid:
    if not isinstance(grid, LatLonGrid):
        raise TypeError("grid must be a LatLonGrid instance")
    return grid
def _validate_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("plate IDs must be a sequence")
    result = tuple(values)
    if not result or any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError("plate IDs must be non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("plate IDs must be unique")
    return result
def _validate_corridors(
    values: Sequence[BoundaryCorridor],
    plate_ids: Sequence[str],
) -> tuple[BoundaryCorridor, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("boundary_corridors must be a sequence of BoundaryCorridor values")
    try:
        result = tuple(values)
    except TypeError as error:
        raise TypeError("boundary_corridors must be a sequence of BoundaryCorridor values") from error
    if any(not isinstance(value, BoundaryCorridor) for value in result):
        raise TypeError("boundary_corridors must contain only BoundaryCorridor values")
    corridor_ids = [value.corridor_id for value in result]
    if len(set(corridor_ids)) != len(corridor_ids):
        raise ValueError("boundary corridor IDs must be unique")
    allowed = set(plate_ids)
    if any(set(value.plate_ids) - allowed for value in result):
        raise ValueError("boundary corridor plate IDs must reference known plates")
    return tuple(sorted(result, key=lambda value: value.corridor_id))
def _validate_anchors(values: Sequence[PlateDefinition]) -> tuple[PlateDefinition, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("anchors must be a sequence of PlateDefinition values")
    result = tuple(values)
    if any(not isinstance(value, PlateDefinition) for value in result):
        raise TypeError("anchors must contain only PlateDefinition values")
    ids = [value.plate_id for value in result]
    if len(set(ids)) != len(ids):
        raise ValueError("anchor plate IDs must be unique")
    if result:
        vectors = np.asarray([lon_lat_to_unit_vector(value.anchor_lon, value.anchor_lat) for value in result])
        dots = vectors @ vectors.T
        if np.any(dots[np.triu_indices(len(result), 1)] >= 1.0 - 1.0e-12):
            raise ValueError("anchor cores must be unique")
    return result
def fibonacci_sphere_candidates(candidate_count: int, stage_seed: int) -> np.ndarray:
    """Return equal-area Fibonacci candidates with a deterministic phase."""

    if isinstance(candidate_count, (bool, np.bool_)) or not isinstance(candidate_count, (int, np.integer)):
        raise TypeError("candidate_count must be a positive integer")
    count = int(candidate_count)
    if count <= 0:
        raise ValueError("candidate_count must be a positive integer")
    seed = _seed(stage_seed)
    digest = _hash(seed, "fibonacci")
    phase = int.from_bytes(digest[:8], "big") / 2.0**64
    shift = (int.from_bytes(digest[8:16], "big") / 2.0**64 - 0.5) * 0.45
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * np.clip((index + 0.5 + shift) / count, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    longitude = 2.0 * math.pi * ((index / ((1.0 + math.sqrt(5.0)) / 2.0) + phase) % 1.0)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return _frozen_array(np.stack((radial * np.cos(longitude), radial * np.sin(longitude), z), axis=-1), dtype=np.float64)
def select_maximin_cores(candidates: Any, existing_vectors: Any, additional_count: int) -> np.ndarray:
    """Select candidates by spherical angular maximin distance."""

    if isinstance(additional_count, (bool, np.bool_)) or not isinstance(additional_count, (int, np.integer)):
        raise TypeError("additional_count must be a non-negative integer")
    count = int(additional_count)
    if count < 0:
        raise ValueError("additional_count must be non-negative")
    candidate_vectors = _vectors(candidates, "candidates")
    if candidate_vectors.ndim != 2:
        raise ValueError("candidates must have shape (n, 3)")
    existing_array = np.asarray(existing_vectors, dtype=np.float64)
    existing = _vectors(existing_array, "existing_vectors") if existing_array.size else np.empty((0, 3))
    if existing.ndim != 2:
        raise ValueError("existing_vectors must have shape (n, 3)")
    if count > len(candidate_vectors):
        raise ValueError("additional_count exceeds candidate count")
    available = np.ones(len(candidate_vectors), dtype=bool)
    reference = existing.copy()
    selected: list[np.ndarray] = []
    for _ in range(count):
        if len(reference):
            score = np.min(np.arccos(np.clip(candidate_vectors @ reference.T, -1.0, 1.0)), axis=1)
        else:
            score = np.full(len(candidate_vectors), math.inf)
        score[~available] = -math.inf
        index = int(np.argmax(score))
        available[index] = False
        selected.append(candidate_vectors[index].copy())
        reference = np.vstack((reference, candidate_vectors[index]))
    return _frozen_array(np.vstack(selected), dtype=np.float64) if selected else np.empty((0, 3))
def _generated_motion(seed: int, ordinal: int, vector: np.ndarray) -> tuple[float, float, float]:
    digest = _hash(seed, "generated-motion", ordinal)
    lon, _ = unit_vector_to_lon_lat(vector)
    pole_lon = (float(lon) + 90.0 + (int.from_bytes(digest[:4], "big") / 2.0**32 - 0.5) * 60.0 + 180.0) % 360.0 - 180.0
    pole_lat = (int.from_bytes(digest[4:8], "big") / 2.0**32 - 0.5) * 120.0
    speed = 0.15 + int.from_bytes(digest[8:12], "big") / 2.0**32 * 1.35
    return float(pole_lon), float(pole_lat), float(-speed if digest[12] & 1 else speed)
def resolve_plate_definitions(plate_count: int, anchors: Sequence[PlateDefinition], stage_seed: int) -> tuple[PlateDefinition, ...]:
    if isinstance(plate_count, (bool, np.bool_)) or not isinstance(plate_count, (int, np.integer)):
        raise TypeError("plate_count must be a positive integer")
    count = int(plate_count)
    if count <= 0:
        raise ValueError("plate_count must be positive")
    seed = _seed(stage_seed)
    anchor_values = _validate_anchors(anchors)
    if len(anchor_values) > count:
        raise ValueError("plate_count cannot be smaller than anchor count")
    if len(anchor_values) == count:
        return tuple(sorted(anchor_values, key=lambda value: value.plate_id))
    existing = np.asarray([lon_lat_to_unit_vector(value.anchor_lon, value.anchor_lat) for value in anchor_values], dtype=np.float64)
    candidates = fibonacci_sphere_candidates(max(64, 32 * count), seed)
    selected = select_maximin_cores(candidates, existing, count - len(anchor_values))
    result = list(anchor_values)
    used = {value.plate_id for value in result}
    for ordinal, vector in enumerate(selected, 1):
        plate_id = f"plate-auto-{ordinal:03d}"
        while plate_id in used:
            ordinal += 1
            plate_id = f"plate-auto-{ordinal:03d}"
        used.add(plate_id)
        lon, lat = unit_vector_to_lon_lat(vector)
        pole_lon, pole_lat, speed = _generated_motion(seed, ordinal, vector)
        result.append(PlateDefinition(plate_id, float(lon), float(lat), pole_lon, pole_lat, speed, False))
    return tuple(sorted(result, key=lambda value: value.plate_id))
def _corridor_arcs(corridor: BoundaryCorridor, cores: Mapping[str, np.ndarray]) -> tuple[OrientedSphericalArc, ...]:
    first_core = cores[corridor.plate_ids[0]]
    second_core = cores[corridor.plate_ids[1]]
    arcs: list[OrientedSphericalArc] = []
    for index, (start, end) in enumerate(zip(corridor.controls_lon_lat, corridor.controls_lon_lat[1:])):
        start_vector = lon_lat_to_unit_vector(*start)
        end_vector = lon_lat_to_unit_vector(*end)
        cross = np.cross(start_vector, end_vector)
        cross_norm = float(np.linalg.norm(cross))
        midpoint = start_vector + end_vector
        midpoint_norm = float(np.linalg.norm(midpoint))
        if cross_norm <= _VECTOR_EPS or midpoint_norm <= _VECTOR_EPS:
            raise ValueError("corridor control arc is degenerate")
        midpoint /= midpoint_norm
        cross /= cross_norm
        projected_a = first_core - float(np.dot(first_core, midpoint)) * midpoint
        projected_b = second_core - float(np.dot(second_core, midpoint)) * midpoint
        witness = projected_b - projected_a
        witness_norm = float(np.linalg.norm(witness))
        orientation = float(np.dot(cross, witness))
        if witness_norm <= _VECTOR_EPS or abs(orientation) <= _VECTOR_EPS:
            raise ValueError("corridor arc has a degenerate A-to-B side witness")
        normal = cross if orientation > 0.0 else -cross
        arcs.append(OrientedSphericalArc(
            f"{corridor.corridor_id}-{index:04d}",
            start,
            end,
            tuple(float(value) for value in normal),
        ))
    return tuple(arcs)


def _core_domain_corridor_gate(
    base_scores: np.ndarray,
    cores: np.ndarray,
    radius_km: float,
) -> np.ndarray:
    """Keep a displaced border out of every plate's generating core domain.

    A corridor is a local deformation of a pre-existing Voronoi boundary, not
    a second plate generator.  Without this constraint a strong, curved arc
    can cross the interior of a nearby microplate, sever its anchor from most
    of its area, and leave a one-cell ``core`` behind.  The safe domain is
    expressed as a fraction of each core's nearest-neighbour separation, so it
    remains spherical and scales with the actual plate configuration instead
    of with output resolution.
    """

    if len(cores) <= 1:
        return np.ones(base_scores.shape[0], dtype=np.float64)
    pairwise = np.clip(cores @ cores.T, -1.0, 1.0)
    np.fill_diagonal(pairwise, -1.0)
    nearest_spacing_km = radius_km * np.arccos(np.max(pairwise, axis=1))
    nearest_index = np.argmax(base_scores, axis=1)
    distance_to_nearest_core_km = radius_km * np.arccos(
        np.clip(np.max(base_scores, axis=1), -1.0, 1.0)
    )
    spacing = nearest_spacing_km[nearest_index]
    # Boundary ownership is allowed to move only through the outer part of
    # each Voronoi province.  A short feather avoids introducing a synthetic
    # circular rim around the protected core.
    start = 0.32 * spacing
    end = 0.46 * spacing
    return np.clip(
        (distance_to_nearest_core_km - start) / np.maximum(end - start, _EPS),
        0.0,
        1.0,
    )


def _apply_corridor_scores(
    scores: np.ndarray,
    cell_vectors: np.ndarray,
    cores: np.ndarray,
    plate_ids: tuple[str, ...],
    corridors: tuple[BoundaryCorridor, ...],
    radius_km: float,
) -> np.ndarray:
    if not corridors:
        return scores
    original = np.array(scores, dtype=np.float64, copy=True)
    result = np.array(scores, dtype=np.float64, copy=True)
    core_domain_gate = _core_domain_corridor_gate(original, cores, radius_km)
    core_map = {plate_id: cores[index] for index, plate_id in enumerate(plate_ids)}
    indices = {plate_id: index for index, plate_id in enumerate(plate_ids)}
    for corridor in corridors:
        arcs = _corridor_arcs(corridor, core_map)
        nearest = nearest_oriented_arc(
            cell_vectors,
            arcs,
            radius_km,
            max_distance_km=4.0 * corridor.influence_width_km,
            chunk_size=_CORRIDOR_CHUNK_SIZE,
        )
        signed = np.asarray(nearest.signed_distance_km, dtype=np.float64).reshape(-1)
        finite = np.isfinite(signed)
        absolute = np.where(finite, np.abs(signed), 0.0)
        weight = np.where(
            finite,
            np.exp(-0.5 * np.square(absolute / corridor.influence_width_km)),
            0.0,
        )
        weight *= core_domain_gate
        side = np.zeros_like(signed)
        side[finite] = np.tanh(signed[finite] / (0.35 * corridor.influence_width_km))
        first_index = indices[corridor.plate_ids[0]]
        second_index = indices[corridor.plate_ids[1]]
        base_difference = original[:, first_index] - original[:, second_index]
        result[:, first_index] += (
            -0.5 * base_difference * weight
            + corridor.strength * weight
            - corridor.strength * weight * side
        )
        result[:, second_index] += (
            0.5 * base_difference * weight
            + corridor.strength * weight
            + corridor.strength * weight * side
        )
    return result
def assign_plate_grid(
    grid: LatLonGrid,
    core_vectors: Any,
    plate_ids: Sequence[str],
    boundary_corridors: Sequence[BoundaryCorridor],
    radius_km: float,
) -> np.ndarray:
    """Assign every ordinary grid cell to its nearest spherical core."""

    checked = _validate_grid(grid)
    ids = _validate_ids(plate_ids)
    cores = _vectors(core_vectors, "core_vectors")
    if cores.ndim != 2 or len(cores) != len(ids):
        raise ValueError("core_vectors must have one vector per plate ID")
    if len(cores) > 1:
        dots = cores @ cores.T
        if np.any(dots[np.triu_indices(len(cores), 1)] >= 1.0 - 1.0e-12):
            raise ValueError("core_vectors must be unique")
    radius = _float(radius_km, "radius_km")
    if radius <= 0.0:
        raise ValueError("radius_km must be positive")
    corridors = _validate_corridors(boundary_corridors, ids)
    scores = checked.unit_vectors.reshape((-1, 3)) @ cores.T
    scores = _apply_corridor_scores(scores, checked.unit_vectors.reshape((-1, 3)), cores, ids, corridors, radius)
    nearest = np.argmax(scores, axis=1)
    # A boundary corridor may bend ownership near a tiny plate, but it cannot
    # dislodge the plate from its own generating site. Pinning each nearest
    # core cell is the spherical Voronoi invariant that also gives orphan
    # cleanup one stable cell to protect.
    cell_vectors = checked.unit_vectors.reshape((-1, 3))
    closest_cells = np.argmax(cell_vectors @ cores.T, axis=0)
    nearest[closest_cells] = np.arange(len(ids), dtype=nearest.dtype)
    labels = np.asarray([ids[int(index)] for index in nearest], dtype=object).reshape(checked.shape)
    flat_labels = labels.reshape(-1)
    missing = tuple(plate_id for plate_id in ids if not np.any(flat_labels == plate_id))
    if missing:
        raise ValueError(
            "corridor assignment left no grid cells for plate(s): "
            + ", ".join(missing)
        )
    mismatches = tuple(
        ids[index]
        for index, cell in enumerate(closest_cells)
        if flat_labels[int(cell)] != ids[index]
    )
    if mismatches:
        raise ValueError(
            "corridor assignment moved the nearest core cell away from its plate: "
            + ", ".join(mismatches)
        )
    return _frozen_array(labels, dtype=object)

def _components(grid: LatLonGrid, labels: np.ndarray, label: str) -> list[set[int]]:
    remaining = {index for index, value in enumerate(labels.reshape(-1)) if value == label}
    result: list[set[int]] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbour in grid.neighbors[current]:
                if neighbour < grid.cell_count and neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        result.append(component)
    return result

def _canonical_point(point: tuple[float, float]) -> tuple[float, float]:
    longitude, latitude = point
    if math.isclose(latitude, 90.0, abs_tol=_EPS):
        return 0.0, 90.0
    if math.isclose(latitude, -90.0, abs_tol=_EPS):
        return 0.0, -90.0
    longitude = ((longitude + 180.0) % 360.0) - 180.0
    return float(longitude), float(latitude)

def _edge_endpoints(grid: LatLonGrid, first: int, second: int) -> tuple[tuple[float, float], tuple[float, float]]:
    row_a, column_a = grid.row_column(first)
    row_b, column_b = grid.row_column(second)
    n_lon = grid.shape[1]
    if row_a == row_b:
        if {column_a, column_b} == {0, n_lon - 1}:
            longitude = -180.0
        elif abs(column_a - column_b) == 1:
            longitude = float(grid.longitude_edges[max(column_a, column_b)])
        else:
            raise ValueError("cells are not horizontally adjacent")
        return (_canonical_point((longitude, float(grid.latitude_edges[row_a]))),
                _canonical_point((longitude, float(grid.latitude_edges[row_a + 1]))))
    if column_a == column_b and abs(row_a - row_b) == 1:
        latitude = float(grid.latitude_edges[max(row_a, row_b)])
        return (_canonical_point((float(grid.longitude_edges[column_a]), latitude)),
                _canonical_point((float(grid.longitude_edges[column_a + 1]), latitude)))
    raise ValueError("cells are not a shared grid edge")

def _edge_length_unit(grid: LatLonGrid, first: int, second: int) -> float:
    start, end = _edge_endpoints(grid, first, second)
    a = lon_lat_to_unit_vector(*start)
    b = lon_lat_to_unit_vector(*end)
    return float(math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))

def clean_orphan_components(
    grid: LatLonGrid,
    plate_grid: Any,
    plate_ids: Sequence[str],
    protected_cells: Mapping[str, Iterable[int]] | None = None,
) -> tuple[np.ndarray, int]:
    """Merge disconnected plate fragments while retaining each declared root."""

    checked = _validate_grid(grid)
    ids = _validate_ids(plate_ids)
    labels = np.asarray(plate_grid, dtype=object)
    if labels.shape != checked.shape:
        raise ValueError("plate_grid shape must match grid.shape")
    if any(not isinstance(value, str) or not value.strip() for value in labels.reshape(-1)):
        raise ValueError("plate_grid must contain non-empty IDs")
    if not set(np.unique(labels)).issubset(set(ids)):
        raise ValueError("plate_grid contains an unknown plate ID")
    result = labels.reshape(-1).copy()
    merge_count = 0
    for _ in range(checked.cell_count + 1):
        changed = False
        for label in ids:
            components = _components(checked, result, label)
            if len(components) <= 1:
                continue
            protected = set(protected_cells.get(label, ())) if protected_cells else set()
            protected_components = [component for component in components if component & protected]
            if protected_components:
                unowned = sorted(cell for cell in protected if result[cell] != label)
                if unowned:
                    raise ValueError(
                        f"protected core cell for {label!r} is not owned by its plate"
                    )
                if len(protected_components) != 1:
                    raise ValueError(
                        f"protected core cells for {label!r} occupy disconnected components"
                    )
                # Plate identity is rooted at its generating core.  A larger
                # disconnected patch is an assignment artefact, not a reason
                # to discard the core and silently relabel the plate.
                primary = protected_components[0]
            else:
                primary = min(components, key=lambda item: (-len(item), min(item)))
            for orphan in [component for component in components if component is not primary]:
                shared: dict[str, float] = {}
                for current in sorted(orphan):
                    for neighbour in checked.neighbors[current]:
                        if neighbour >= checked.cell_count:
                            continue
                        neighbour_label = str(result[neighbour])
                        if neighbour_label != label and neighbour_label in ids:
                            shared[neighbour_label] = shared.get(neighbour_label, 0.0) + _edge_length_unit(checked, current, neighbour)
                if not shared:
                    raise ValueError(f"orphan component for {label!r} has no adjacent plate")
                target = min(shared, key=lambda item: (-shared[item], item))
                for current in orphan:
                    result[current] = target
                merge_count += 1
                changed = True
        if not changed:
            break
    return _frozen_array(result.reshape(checked.shape), dtype=object), merge_count


def _validate_plate_topology(
    grid: LatLonGrid,
    plate_grid: np.ndarray,
    plate_ids: Sequence[str],
    protected_cells: Mapping[str, Iterable[int]],
) -> dict[str, int]:
    """Check the causal plate invariant after corridor deformation and repair."""

    flat = np.asarray(plate_grid, dtype=object).reshape(-1)
    component_counts: list[int] = []
    protected_count = 0
    for plate_id in plate_ids:
        components = _components(grid, flat, plate_id)
        component_counts.append(len(components))
        for cell in protected_cells.get(plate_id, ()):
            protected_count += 1
            if cell < 0 or cell >= grid.cell_count or flat[cell] != plate_id:
                raise ValueError(f"plate core ownership lost for {plate_id!r}")
        if len(components) != 1:
            raise ValueError(f"plate {plate_id!r} remains disconnected after topology repair")
    return {
        "connected_plate_count": int(len(plate_ids)),
        "maximum_component_count": int(max(component_counts, default=0)),
        "protected_core_count": int(protected_count),
    }

def _unique_edges(grid: LatLonGrid) -> Iterable[tuple[int, int]]:
    for first in range(grid.cell_count):
        for second in grid.neighbors[first]:
            if second < grid.cell_count and first < second:
                yield first, second

def _midpoint(start: tuple[float, float], end: tuple[float, float]) -> np.ndarray:
    first = lon_lat_to_unit_vector(*start)
    second = lon_lat_to_unit_vector(*end)
    value = first + second
    if np.linalg.norm(value) <= _VECTOR_EPS:
        value = first
    return value / np.linalg.norm(value)

def _node_key(point: tuple[float, float]) -> tuple[str, float | str, float | None]:
    longitude, latitude = point
    if math.isclose(latitude, 90.0, abs_tol=_EPS):
        return "pole", "north", None
    if math.isclose(latitude, -90.0, abs_tol=_EPS):
        return "pole", "south", None
    return "vertex", round(((longitude + 180.0) % 360.0) - 180.0, 12), round(latitude, 12)

def classify_boundary_motion(
    normal_velocity_cm_per_year: Any,
    tangent_velocity_cm_per_year: Any,
    *,
    threshold_cm_per_year: float = _CLASSIFICATION_THRESHOLD,
) -> str:
    normal = _float(normal_velocity_cm_per_year, "normal_velocity_cm_per_year")
    tangent = _float(tangent_velocity_cm_per_year, "tangent_velocity_cm_per_year")
    threshold = _float(threshold_cm_per_year, "threshold_cm_per_year")
    if threshold <= 0.0:
        raise ValueError("threshold_cm_per_year must be positive")
    normal_active = abs(normal) >= threshold
    tangent_active = abs(tangent) >= threshold
    if not normal_active and not tangent_active:
        return "stable"
    if not normal_active:
        return "transform"
    if tangent_active:
        return "oblique-divergent" if normal > 0.0 else "oblique-convergent"
    return "divergent" if normal > 0.0 else "convergent"

def plate_velocity_cm_per_year(plate: PlateDefinition, location: Any, radius_km: float) -> np.ndarray:
    """Evaluate ``v = omega x r`` in cm/year for an explicit world radius."""

    if not isinstance(plate, PlateDefinition):
        raise TypeError("plate must be a PlateDefinition")
    radius = _float(radius_km, "radius_km")
    if radius <= 0.0:
        raise ValueError("radius_km must be positive")
    values = np.asarray(location, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] not in (2, 3):
        raise ValueError("location must have final dimension 2 or 3")
    if not np.all(np.isfinite(values)):
        raise ValueError("location contains non-finite values")
    vectors = lon_lat_to_unit_vector(values[..., 0], values[..., 1]) if values.shape[-1] == 2 else _vectors(values, "location")
    pole = lon_lat_to_unit_vector(plate.euler_pole_lon, plate.euler_pole_lat)
    omega = pole * math.radians(plate.angular_speed_deg_per_myr)
    return np.cross(omega, vectors) * radius * 0.1

def extract_shared_boundaries(
    grid: LatLonGrid,
    plate_grid: Any,
    *,
    plates: Mapping[str, PlateDefinition],
    radius_km: float,
    pole_plate_assignments: PolePlateAssignments,
    threshold_cm_per_year: float = _CLASSIFICATION_THRESHOLD,
) -> tuple[tuple[BoundarySegment, ...], tuple[TripleJunction, ...], int]:
    """Extract unique unlike-neighbour edges and exact three-way junctions."""

    checked = _validate_grid(grid)
    labels = np.asarray(plate_grid, dtype=object)
    if labels.shape != checked.shape:
        raise ValueError("plate_grid shape must match grid.shape")
    if not isinstance(plates, Mapping) or not plates:
        raise TypeError("plates must be a non-empty ID-to-PlateDefinition mapping")
    if any(not isinstance(key, str) or not isinstance(value, PlateDefinition) for key, value in plates.items()):
        raise TypeError("plates must map string IDs to PlateDefinition values")
    if any(key != value.plate_id for key, value in plates.items()):
        raise ValueError("plate mapping keys must equal PlateDefinition.plate_id")
    if any(not isinstance(value, str) or not value.strip() for value in labels.reshape(-1)):
        raise ValueError("plate_grid must contain non-empty string IDs")
    unique_ids = set(np.unique(labels))
    if unique_ids != set(plates):
        raise ValueError("plates must match all plate_grid IDs")
    if not isinstance(pole_plate_assignments, PolePlateAssignments):
        raise TypeError("pole_plate_assignments must be PolePlateAssignments")
    if pole_plate_assignments.north not in plates or pole_plate_assignments.south not in plates:
        raise ValueError("pole assignments must reference known plates")
    radius = _float(radius_km, "radius_km")
    if radius <= 0.0:
        raise ValueError("radius_km must be positive")
    threshold = _float(threshold_cm_per_year, "threshold_cm_per_year")
    if threshold <= 0.0:
        raise ValueError("threshold_cm_per_year must be positive")

    flat = labels.reshape(-1)
    segments: list[BoundarySegment] = []
    endpoints: dict[tuple[str, float | str, float | None], list[str]] = {}
    for first, second in _unique_edges(checked):
        first_id = str(flat[first])
        second_id = str(flat[second])
        if first_id == second_id:
            continue
        pair = tuple(sorted((first_id, second_id)))
        a_cell, b_cell = (first, second) if first_id == pair[0] else (second, first)
        start, end = _edge_endpoints(checked, first, second)
        midpoint_vector = _midpoint(start, end)
        midpoint_lon, midpoint_lat = unit_vector_to_lon_lat(midpoint_vector)
        midpoint = (float(midpoint_lon), float(midpoint_lat))
        node_vectors = checked.node_vectors
        a_vector = node_vectors[a_cell]
        b_vector = node_vectors[b_cell]
        normal = b_vector - float(np.dot(b_vector, midpoint_vector)) * midpoint_vector
        if np.linalg.norm(normal) <= _VECTOR_EPS:
            normal = b_vector - a_vector
            normal -= float(np.dot(normal, midpoint_vector)) * midpoint_vector
        if np.linalg.norm(normal) <= _VECTOR_EPS:
            normal = np.cross(midpoint_vector, np.array([0.0, 0.0, 1.0]))
        normal /= np.linalg.norm(normal)
        tangent = np.cross(midpoint_vector, normal)
        tangent /= np.linalg.norm(tangent)
        velocity_a = plate_velocity_cm_per_year(plates[pair[0]], midpoint, radius)
        velocity_b = plate_velocity_cm_per_year(plates[pair[1]], midpoint, radius)
        relative = np.asarray(velocity_b - velocity_a, dtype=np.float64)
        normal_value = float(np.dot(relative, normal))
        tangent_value = float(np.dot(relative, tangent))
        classification = classify_boundary_motion(normal_value, tangent_value, threshold_cm_per_year=threshold)
        first_vector = lon_lat_to_unit_vector(*start)
        second_vector = lon_lat_to_unit_vector(*end)
        length_angle = float(math.atan2(np.linalg.norm(np.cross(first_vector, second_vector)), np.dot(first_vector, second_vector)))
        boundary_id = "boundary-" + hashlib.sha256(f"{pair[0]}\0{pair[1]}\0{min(first, second)}\0{max(first, second)}".encode()).hexdigest()[:20]
        segment = BoundarySegment(
            boundary_id,
            (pair[0], pair[1]),
            start,
            end,
            midpoint,
            length_angle * radius,
            (a_cell, b_cell),
            tuple(float(value) for value in normal),
            tuple(float(value) for value in tangent),
            tuple(float(value) for value in relative),
            normal_value,
            tangent_value,
            classification,
        )
        segments.append(segment)
        endpoints.setdefault(_node_key(start), []).append(segment.boundary_id)
        endpoints.setdefault(_node_key(end), []).append(segment.boundary_id)

    segments.sort(key=lambda segment: segment.boundary_id)
    junctions: list[TripleJunction] = []
    rejected = 0
    by_id = {segment.boundary_id: segment for segment in segments}
    for node, incident in sorted(endpoints.items(), key=lambda item: item[0]):
        boundary_ids = tuple(sorted(set(incident)))
        plate_ids = tuple(sorted({plate_id for boundary_id in boundary_ids for plate_id in by_id[boundary_id].plate_ids}))
        if len(boundary_ids) == 3 and len(plate_ids) == 3:
            if node[0] == "pole":
                longitude, latitude = 0.0, (90.0 if node[1] == "north" else -90.0)
            else:
                longitude, latitude = float(node[1]), float(node[2])
            junction_id = "junction-" + hashlib.sha256(repr(node).encode()).hexdigest()[:20]
            junctions.append(TripleJunction(junction_id, longitude, latitude, boundary_ids, plate_ids))
        elif len(boundary_ids) >= 4:
            rejected += 1
    junctions.sort(key=lambda junction: junction.junction_id)
    return tuple(segments), tuple(junctions), rejected

def build_plate_fields(
    grid: LatLonGrid,
    *,
    plate_count: int,
    anchors: Sequence[PlateDefinition],
    stage_seed: int,
    radius_km: float,
    boundary_corridors: Sequence[BoundaryCorridor],
    velocity_threshold_cm_per_year: float = _CLASSIFICATION_THRESHOLD,
) -> PlateFields:
    """Build one deterministic Task 3 plate topology and motion result."""

    checked = _validate_grid(grid)
    if isinstance(plate_count, (bool, np.bool_)) or not isinstance(plate_count, (int, np.integer)):
        raise TypeError("plate_count must be a positive integer")
    count = int(plate_count)
    if count <= 0:
        raise ValueError("plate_count must be positive")
    if count > checked.cell_count:
        raise ValueError("plate_count cannot exceed grid cell count")
    seed = _seed(stage_seed)
    radius = _float(radius_km, "radius_km")
    if radius <= 0.0:
        raise ValueError("radius_km must be positive")
    threshold = _float(velocity_threshold_cm_per_year, "velocity_threshold_cm_per_year")
    if threshold <= 0.0:
        raise ValueError("velocity_threshold_cm_per_year must be positive")

    anchor_values = _validate_anchors(anchors)
    definitions = resolve_plate_definitions(count, anchor_values, seed)
    ids = tuple(value.plate_id for value in definitions)
    corridors = _validate_corridors(boundary_corridors, ids)
    cores = np.asarray([lon_lat_to_unit_vector(value.anchor_lon, value.anchor_lat) for value in definitions], dtype=np.float64)
    labels = np.asarray(assign_plate_grid(checked, cores, ids, corridors, radius), dtype=object)
    protected: dict[str, set[int]] = {}
    flat = labels.reshape(-1).copy()
    cell_vectors = checked.unit_vectors.reshape((-1, 3))
    for definition in definitions:
        cell = int(np.argmax(cell_vectors @ lon_lat_to_unit_vector(definition.anchor_lon, definition.anchor_lat)))
        protected[definition.plate_id] = {cell}
    labels, orphan_merge_count = clean_orphan_components(
        checked,
        flat.reshape(checked.shape),
        ids,
        protected_cells=protected,
    )
    topology_diagnostics = _validate_plate_topology(
        checked,
        labels,
        ids,
        protected,
    )
    pole_vectors = np.asarray(
        [lon_lat_to_unit_vector(0.0, 90.0), lon_lat_to_unit_vector(0.0, -90.0)],
        dtype=np.float64,
    )
    pole_scores = _apply_corridor_scores(
        pole_vectors @ cores.T,
        pole_vectors,
        cores,
        ids,
        corridors,
        radius,
    )
    poles = PolePlateAssignments(ids[int(np.argmax(pole_scores[0]))], ids[int(np.argmax(pole_scores[1]))])
    definition_map = {value.plate_id: value for value in definitions}
    boundaries, junctions, rejected_junction_count = extract_shared_boundaries(
        checked,
        labels,
        plates=definition_map,
        radius_km=radius,
        pole_plate_assignments=poles,
        threshold_cm_per_year=threshold,
    )
    velocity_grid = np.zeros(checked.shape + (3,), dtype=np.float64)
    for plate_id, definition in definition_map.items():
        indices = np.flatnonzero(labels.reshape(-1) == plate_id)
        if len(indices):
            velocity_grid.reshape((-1, 3))[indices] = plate_velocity_cm_per_year(definition, cell_vectors[indices], radius)
    anchor_ids = tuple(value.plate_id for value in anchor_values)
    anchor_id_set = set(anchor_ids)
    diagnostics = {
            "plate_count": count,
            "cell_count": checked.cell_count,
            "anchor_ids": anchor_ids,
            "generated_ids": tuple(value.plate_id for value in definitions if value.plate_id not in anchor_id_set),
            "stage_seed": seed,
            "boundary_count": len(boundaries),
            "triple_junction_count": len(junctions),
            "rejected_junction_count": rejected_junction_count,
            "orphan_merge_count": orphan_merge_count,
            "corridor_count": len(corridors),
            "corridor_ids": tuple(value.corridor_id for value in corridors),
            "topology": topology_diagnostics,
        }
    return PlateFields(labels, poles, definitions, cores, boundaries, junctions, velocity_grid, diagnostics)


__all__ = [
    "BoundarySegment",
    "BoundaryCorridor",
    "PlateDefinition",
    "PlateFields",
    "PolePlateAssignments",
    "TripleJunction",
    "assign_plate_grid",
    "build_plate_fields",
    "clean_orphan_components",
    "classify_boundary_motion",
    "extract_shared_boundaries",
    "fibonacci_sphere_candidates",
    "plate_velocity_cm_per_year",
    "resolve_plate_definitions",
    "select_maximin_cores",
]
