"""Project spherical boundary relief profiles onto an ordinary latitude grid.

This stage only constructs a tectonic elevation correction.  It does not read,
write, or modify a DEM; applying a correction remains the responsibility of
``metric_dem``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from world_atlas.physical.boundary_chains import BoundaryChain
from world_atlas.physical.planetary_grid import LatLonGrid, lon_lat_to_unit_vector
from world_atlas.physical.spherical_arc_distance import OrientedSphericalArc, signed_distance_to_arc
from world_atlas.physical.tectonic_relief import (
    CrustFields,
    ProfileFeature,
    TectonicProfileConfig,
    build_boundary_profile,
    choose_subduction_side,
    evaluate_boundary_profile,
)
from world_atlas.physical.tectonics import BoundarySegment, PlateDefinition, plate_velocity_cm_per_year


_REQUIRED_DIAGNOSTIC_KEYS = frozenset(
    {
        "chain_count",
        "source_segment_count",
        "influenced_cell_count",
        "max_abs_raw_delta_m",
        "gaussian_cutoff_sigma",
    }
)
_MAX_UINT16 = np.iinfo(np.uint16).max
_VECTOR_EPSILON = 1.0e-14
_ANGLE_EPSILON = 1.0e-12


def _real_array(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric array") from error
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must contain real numeric values")
    try:
        result = np.array(value, dtype=dtype, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must contain real numeric values") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _validate_output_diagnostics(value: Any) -> MappingProxyType:
    if not isinstance(value, Mapping):
        raise TypeError("diagnostics must be a mapping")
    missing = _REQUIRED_DIAGNOSTIC_KEYS.difference(value.keys())
    if missing:
        raise ValueError(
            "diagnostics is missing required keys: " + ", ".join(sorted(missing))
        )
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class TectonicPotential:
    """Read-only tectonic elevation correction and its influence ledger."""

    elevation_delta_m: np.ndarray
    corridor_mask: np.ndarray
    influence_count: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        delta = _real_array(
            self.elevation_delta_m,
            name="elevation_delta_m",
            dtype=np.float64,
        )
        if delta.ndim != 2:
            raise ValueError("elevation_delta_m must be a two-dimensional array")

        try:
            raw_mask = np.asarray(self.corridor_mask)
        except (TypeError, ValueError) as error:
            raise TypeError("corridor_mask must be a boolean array") from error
        if raw_mask.ndim != 2 or raw_mask.dtype.kind != "b":
            raise TypeError("corridor_mask must be a two-dimensional boolean array")
        if raw_mask.shape != delta.shape:
            raise ValueError("corridor_mask shape must match elevation_delta_m")
        mask = np.array(raw_mask, dtype=bool, copy=True)

        try:
            raw_count = np.asarray(self.influence_count)
        except (TypeError, ValueError) as error:
            raise TypeError("influence_count must be an integer array") from error
        if raw_count.ndim != 2 or raw_count.dtype.kind not in {"i", "u"}:
            raise TypeError("influence_count must be a two-dimensional integer array")
        if raw_count.shape != delta.shape:
            raise ValueError("influence_count shape must match elevation_delta_m")
        if raw_count.dtype.kind == "i" and np.any(raw_count < 0):
            raise ValueError("influence_count must be non-negative")
        if np.any(raw_count > _MAX_UINT16):
            raise ValueError("influence_count exceeds uint16 capacity")
        count = np.array(raw_count, dtype=np.uint16, copy=True)

        expected_mask = count > 0
        if not np.array_equal(mask, expected_mask):
            raise ValueError("corridor_mask must be true exactly where influence_count > 0")
        if np.any(delta[~mask] != 0.0):
            raise ValueError("elevation_delta_m must be exactly zero outside corridor_mask")

        diagnostics = _validate_output_diagnostics(self.diagnostics)
        delta.setflags(write=False)
        mask.setflags(write=False)
        count.setflags(write=False)
        object.__setattr__(self, "elevation_delta_m", delta)
        object.__setattr__(self, "corridor_mask", mask)
        object.__setattr__(self, "influence_count", count)
        object.__setattr__(self, "diagnostics", diagnostics)


def _finite_scalar(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite number")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite number") from error
    if raw.ndim != 0 or raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _iter_values(value: Any, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or value is None:
        raise TypeError(f"{name} must be an iterable")
    try:
        result = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable") from error
    return result


def _validate_plate_definitions(value: Any) -> dict[str, PlateDefinition]:
    if not isinstance(value, Mapping):
        raise TypeError("plate_definitions must be a mapping")
    result: dict[str, PlateDefinition] = {}
    for key, definition in value.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError("plate_definitions keys must be non-empty strings")
        if not isinstance(definition, PlateDefinition):
            raise TypeError("plate_definitions values must be PlateDefinition values")
        if key != definition.plate_id:
            raise ValueError("plate_definitions keys must equal PlateDefinition.plate_id")
        if key in result:
            raise ValueError("plate_definitions keys must be unique")
        result[key] = definition
    return result


def _validate_boundary_segment(segment: Any, *, cell_count: int) -> BoundarySegment:
    if not isinstance(segment, BoundarySegment):
        raise TypeError("boundaries must contain BoundarySegment values")
    boundary_id = segment.boundary_id
    if not isinstance(boundary_id, str) or not boundary_id.strip():
        raise ValueError("boundary_id must be a non-empty string")
    plate_ids = segment.plate_ids
    if not isinstance(plate_ids, tuple) or len(plate_ids) != 2:
        raise ValueError("BoundarySegment.plate_ids must be a two-item tuple")
    if any(not isinstance(value, str) or not value.strip() for value in plate_ids):
        raise ValueError("BoundarySegment.plate_ids must contain non-empty strings")
    if plate_ids[0] == plate_ids[1]:
        raise ValueError("BoundarySegment.plate_ids must contain distinct IDs")
    try:
        cells = segment.cell_indices
    except AttributeError as error:
        raise TypeError("BoundarySegment must provide cell_indices") from error
    if not isinstance(cells, tuple) or len(cells) != 2:
        raise ValueError("BoundarySegment.cell_indices must be a two-item tuple")
    checked_cells: list[int] = []
    for cell in cells:
        if isinstance(cell, (bool, np.bool_)) or not isinstance(cell, (int, np.integer)):
            raise TypeError("BoundarySegment.cell_indices must contain integer cell indices")
        index = int(cell)
        if not 0 <= index < cell_count:
            raise ValueError("BoundarySegment.cell_indices must reference ordinary grid cells")
        checked_cells.append(index)
    if checked_cells[0] == checked_cells[1]:
        raise ValueError("BoundarySegment.cell_indices must reference two distinct cells")
    # The segment is deliberately used in its authored A/B order.  Validate
    # the finite normal now so an invalid orientation cannot be hidden by a
    # later profile calculation.
    try:
        normal = np.asarray(segment.boundary_normal, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("BoundarySegment.boundary_normal must be numeric") from error
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("BoundarySegment.boundary_normal must be a finite 3-vector")
    if np.linalg.norm(normal) <= _VECTOR_EPSILON:
        raise ValueError("BoundarySegment.boundary_normal must be non-zero")
    return segment


def _validate_boundaries(value: Any, *, cell_count: int) -> tuple[BoundarySegment, ...]:
    values = _iter_values(value, name="boundaries")
    result = tuple(
        _validate_boundary_segment(segment, cell_count=cell_count) for segment in values
    )
    ids = [segment.boundary_id for segment in result]
    if len(set(ids)) != len(ids):
        raise ValueError("boundary_id values must be unique")
    return result


def _validate_plate_grid(
    value: Any,
    *,
    grid: LatLonGrid,
    plate_ids: Mapping[str, PlateDefinition],
) -> np.ndarray:
    try:
        labels = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError("plate_grid must be a two-dimensional string array") from error
    if labels.ndim != 2 or labels.shape != grid.shape:
        raise ValueError("plate_grid shape must match grid.shape")
    flat = labels.reshape(-1)
    for label in flat:
        if not isinstance(label, str) or not label.strip():
            raise TypeError("plate_grid must contain non-empty string IDs")
        if label not in plate_ids:
            raise ValueError("plate_grid contains an unknown plate ID")
    return labels


def _validate_chains(value: Any) -> tuple[BoundaryChain, ...]:
    values = _iter_values(value, name="smoothed_chains")
    result = tuple(values)
    if any(not isinstance(chain, BoundaryChain) for chain in result):
        raise TypeError("smoothed_chains must contain BoundaryChain values")
    ids = [chain.chain_id for chain in result]
    if len(set(ids)) != len(ids):
        raise ValueError("chain_id values must be unique")
    return result


def _validate_ledger(
    boundaries: tuple[BoundarySegment, ...],
    chains: tuple[BoundaryChain, ...],
) -> dict[str, BoundarySegment]:
    segments = {segment.boundary_id: segment for segment in boundaries}
    seen: set[str] = set()
    for chain in chains:
        if len(chain.points_lon_lat) != len(chain.source_segment_ids) + 1:
            raise ValueError("each chain must have one smoothed edge per source segment")
        for source_id in chain.source_segment_ids:
            if source_id in seen:
                raise ValueError("source segment ledger contains a duplicate")
            if source_id not in segments:
                raise ValueError("source segment ledger contains an unknown boundary ID")
            seen.add(source_id)
            segment = segments[source_id]
            if set(chain.plate_ids) != set(segment.plate_ids):
                raise ValueError("chain plate_ids must match each source boundary pair")
    expected = set(segments)
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        detail = []
        if missing:
            detail.append("missing=" + repr(missing))
        if extra:
            detail.append("unknown=" + repr(extra))
        raise ValueError("source segment ledger must use every boundary exactly once (" + ", ".join(detail) + ")")
    return segments


def _arc_orientation(
    start_lon_lat: tuple[float, float],
    end_lon_lat: tuple[float, float],
    raw_normal: Any,
) -> tuple[float, float, float]:
    start = np.asarray(lon_lat_to_unit_vector(*start_lon_lat), dtype=np.float64)
    end = np.asarray(lon_lat_to_unit_vector(*end_lon_lat), dtype=np.float64)
    cross = np.cross(start, end)
    norm = float(np.linalg.norm(cross))
    angle = math.atan2(norm, float(np.clip(np.dot(start, end), -1.0, 1.0)))
    if norm <= _VECTOR_EPSILON or angle <= _ANGLE_EPSILON or math.pi - angle <= _ANGLE_EPSILON:
        raise ValueError("smoothed chain edge must define a non-degenerate minor spherical arc")
    normal = np.asarray(raw_normal, dtype=np.float64)
    dot = float(np.dot(cross, normal))
    if abs(dot) <= _VECTOR_EPSILON:
        raise ValueError("raw boundary normal cannot orient the smoothed arc")
    if dot < 0.0:
        cross = -cross
    return tuple(float(value) for value in cross / norm)


def _build_arc(
    chain: BoundaryChain,
    edge_index: int,
    segment: BoundarySegment,
) -> OrientedSphericalArc:
    start = chain.points_lon_lat[edge_index]
    end = chain.points_lon_lat[edge_index + 1]
    normal = _arc_orientation(start, end, segment.boundary_normal)
    return OrientedSphericalArc(
        segment.boundary_id,
        start,
        end,
        normal,
    )


def _feature_cutoff(features: tuple[ProfileFeature, ...], sigma: float) -> float:
    if not features:
        return 0.0
    return max(
        abs(feature.center_km) + sigma * feature.width_km for feature in features
    )


def _arc_cap_candidates(
    grid: LatLonGrid,
    midpoint: np.ndarray,
    cap_angle: float,
) -> np.ndarray:
    """Return ordinary cell indices inside a midpoint spherical cap.

    The latitude rows are solved analytically for their longitude interval;
    no plate-carrée rectangle is used.  This is a broad cap filter only;
    exact minor-arc distance is evaluated after this function.
    """

    if cap_angle >= math.pi - _ANGLE_EPSILON:
        return np.arange(grid.cell_count, dtype=np.int64)
    cap_cosine = math.cos(max(0.0, cap_angle))
    midpoint = np.asarray(midpoint, dtype=np.float64)
    midpoint /= np.linalg.norm(midpoint)
    center_lon = math.atan2(float(midpoint[1]), float(midpoint[0]))
    center_lat = math.asin(float(np.clip(midpoint[2], -1.0, 1.0)))
    ordinary_lats = np.asarray(grid.latitudes, dtype=np.float64)
    ordinary_lons = np.asarray(grid.longitudes, dtype=np.float64)
    longitudes = np.deg2rad(ordinary_lons)
    columns = np.arange(ordinary_lons.size, dtype=np.int64)
    result: list[np.ndarray] = []
    sin_center = math.sin(center_lat)
    cos_center = math.cos(center_lat)
    for row, latitude in enumerate(ordinary_lats):
        row_lat = math.radians(float(latitude))
        sine_term = math.sin(row_lat) * sin_center
        cosine_term = math.cos(row_lat) * cos_center
        if cosine_term <= _VECTOR_EPSILON:
            if sine_term + cosine_term >= cap_cosine - _ANGLE_EPSILON:
                selected_columns = columns
            else:
                selected_columns = columns[:0]
        elif sine_term + cosine_term < cap_cosine - _ANGLE_EPSILON:
            selected_columns = columns[:0]
        elif sine_term - cosine_term >= cap_cosine - _ANGLE_EPSILON:
            selected_columns = columns
        else:
            ratio = (cap_cosine - sine_term) / cosine_term
            half_width = math.acos(float(np.clip(ratio, -1.0, 1.0)))
            delta = np.abs((longitudes - center_lon + math.pi) % (2.0 * math.pi) - math.pi)
            selected_columns = columns[delta <= half_width + _ANGLE_EPSILON]
        if selected_columns.size:
            result.append(row * ordinary_lons.size + selected_columns)
    if not result:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(result).astype(np.int64, copy=False)


def _chain_profile(
    segment: BoundarySegment,
    *,
    crust: CrustFields,
    plates: Mapping[str, PlateDefinition],
    radius_km: float,
    config: TectonicProfileConfig,
) -> tuple[ProfileFeature, ...]:
    plate_a, plate_b = segment.plate_ids
    if plate_a not in plates or plate_b not in plates:
        raise ValueError("plate_definitions is missing a boundary plate ID")
    cell_a, cell_b = segment.cell_indices
    crust_a = crust.kind.reshape(-1)[cell_a]
    crust_b = crust.kind.reshape(-1)[cell_b]
    age_a = crust.ocean_age_myr.reshape(-1)[cell_a]
    age_b = crust.ocean_age_myr.reshape(-1)[cell_b]
    normal = np.array(segment.boundary_normal, dtype=np.float64, copy=True)
    normal /= np.linalg.norm(normal)
    midpoint = segment.midpoint_lon_lat
    velocity_a = plate_velocity_cm_per_year(plates[plate_a], midpoint, radius_km)
    velocity_b = plate_velocity_cm_per_year(plates[plate_b], midpoint, radius_km)
    drive_a = max(float(np.dot(velocity_a, normal)), 0.0)
    drive_b = max(float(-np.dot(velocity_b, normal)), 0.0)
    decision = choose_subduction_side(
        segment.classification,
        plate_a,
        crust_a,
        age_a,
        drive_a,
        plate_b,
        crust_b,
        age_b,
        drive_b,
    )
    return build_boundary_profile(
        segment.classification,
        segment.plate_ids,
        crust_a,
        crust_b,
        decision,
        segment.normal_velocity_cm_per_year,
        segment.tangent_velocity_cm_per_year,
        0.0,
        config,
    )


def build_tectonic_potential(
    grid: LatLonGrid,
    *,
    plate_grid: Any,
    boundaries: Iterable[BoundarySegment],
    smoothed_chains: Iterable[BoundaryChain],
    crust_fields: CrustFields,
    plate_definitions: Mapping[str, PlateDefinition],
    radius_km: float,
    profile_config: TectonicProfileConfig,
    gaussian_cutoff_sigma: float,
) -> TectonicPotential:
    """Build a deterministic spherical, finite-support tectonic delta."""

    if not isinstance(grid, LatLonGrid):
        raise TypeError("grid must be a LatLonGrid instance")
    if not isinstance(crust_fields, CrustFields):
        raise TypeError("crust_fields must be a CrustFields instance")
    if crust_fields.kind.shape != grid.shape:
        raise ValueError("crust_fields shape must match grid.shape")
    if not isinstance(profile_config, TectonicProfileConfig):
        raise TypeError("profile_config must be a TectonicProfileConfig")
    radius = _finite_scalar(radius_km, name="radius_km", positive=True)
    sigma = _finite_scalar(
        gaussian_cutoff_sigma,
        name="gaussian_cutoff_sigma",
        positive=True,
    )
    plates = _validate_plate_definitions(plate_definitions)
    labels = _validate_plate_grid(plate_grid, grid=grid, plate_ids=plates)
    flat_labels = labels.reshape(-1)
    segments = _validate_boundaries(boundaries, cell_count=grid.cell_count)
    for segment in segments:
        if any(plate_id not in plates for plate_id in segment.plate_ids):
            raise ValueError("plate_definitions is missing a boundary plate ID")
        cell_a, cell_b = segment.cell_indices
        if flat_labels[cell_a] != segment.plate_ids[0] or flat_labels[cell_b] != segment.plate_ids[1]:
            raise ValueError("BoundarySegment.cell_indices must follow plate_ids A/B order in plate_grid")
    chains = _validate_chains(smoothed_chains)
    segment_by_id = _validate_ledger(segments, chains)

    # Profiles are constructed once per source segment.  Their source-side
    # crust samples are intentionally taken from the ordinary A/B cells in
    # the BoundarySegment ledger, never inferred from smoothed geometry.
    profiles: dict[str, tuple[ProfileFeature, ...]] = {}
    cutoffs_km: dict[str, float] = {}
    for source_id in sorted(segment_by_id):
        feature_values = _chain_profile(
            segment_by_id[source_id],
            crust=crust_fields,
            plates=plates,
            radius_km=radius,
            config=profile_config,
        )
        profiles[source_id] = feature_values
        cutoff = _feature_cutoff(feature_values, sigma)
        if not math.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("profile feature cutoff must be finite and non-negative")
        cutoffs_km[source_id] = cutoff

    ordinary_vectors = np.asarray(grid.unit_vectors, dtype=np.float64).reshape((-1, 3))
    ordinary_crust = np.asarray(crust_fields.kind, dtype=np.uint8).reshape(-1)
    flat_raw_delta = np.zeros(grid.cell_count, dtype=np.float64)
    flat_count = np.zeros(grid.cell_count, dtype=np.uint64)

    # Chain-local nearest-arc ledgers prevent adjacent smoothed segments from
    # stacking into a seam wall.  These fixed-size vectors are updated only at
    # cap-filtered candidates; no cells-by-arcs matrix is allocated.
    best_absolute = np.full(grid.cell_count, np.inf, dtype=np.float64)
    best_signed = np.zeros(grid.cell_count, dtype=np.float64)
    best_source_ordinal = np.full(grid.cell_count, -1, dtype=np.int64)
    touched = np.empty(grid.cell_count, dtype=np.int64)
    ordered_chains = sorted(
        chains,
        key=lambda chain: (chain.chain_id, chain.source_segment_ids),
    )
    for chain in ordered_chains:
        ordered_edges = sorted(
            enumerate(chain.source_segment_ids),
            key=lambda item: (item[1], item[0]),
        )
        source_ids = tuple(source_id for _edge_index, source_id in ordered_edges)
        touched_count = 0
        for source_ordinal, (edge_index, source_id) in enumerate(ordered_edges):
            features = profiles[source_id]
            if not features:
                continue
            cutoff_km = cutoffs_km[source_id]
            if cutoff_km <= 0.0:
                continue
            segment = segment_by_id[source_id]
            arc = _build_arc(chain, edge_index, segment)
            start = np.asarray(lon_lat_to_unit_vector(*arc.start_lon_lat), dtype=np.float64)
            end = np.asarray(lon_lat_to_unit_vector(*arc.end_lon_lat), dtype=np.float64)
            arc_midpoint = start + end
            midpoint_norm = float(np.linalg.norm(arc_midpoint))
            if midpoint_norm <= _VECTOR_EPSILON:
                raise ValueError("smoothed chain edge midpoint is undefined")
            arc_midpoint /= midpoint_norm
            arc_angle = math.atan2(
                float(np.linalg.norm(np.cross(start, end))),
                float(np.clip(np.dot(start, end), -1.0, 1.0)),
            )
            cap_angle = min(math.pi, 0.5 * arc_angle + cutoff_km / radius)
            candidates = _arc_cap_candidates(grid, arc_midpoint, cap_angle)
            if candidates.size == 0:
                continue
            distances = np.asarray(
                signed_distance_to_arc(ordinary_vectors[candidates], arc, radius),
                dtype=np.float64,
            )
            absolute = np.abs(distances)
            within = absolute <= cutoff_km + 1.0e-9
            if not np.any(within):
                continue
            valid_candidates = candidates[within]
            valid_distances = distances[within]
            valid_absolute = np.abs(valid_distances)
            prior = best_absolute[valid_candidates]
            unseen = np.isinf(prior)
            better = unseen | (valid_absolute < prior - 1.0e-10)
            if not np.any(better):
                continue
            update_candidates = valid_candidates[better]
            update_distances = valid_distances[better]
            update_unseen = unseen[better]
            if np.any(update_unseen):
                new_indices = update_candidates[update_unseen]
                next_touched = touched_count + new_indices.size
                touched[touched_count:next_touched] = new_indices
                touched_count = next_touched
            best_absolute[update_candidates] = np.abs(update_distances)
            best_signed[update_candidates] = update_distances
            best_source_ordinal[update_candidates] = source_ordinal

        if touched_count == 0:
            continue
        selected_indices = np.fromiter(
            sorted(touched[:touched_count].tolist()),
            dtype=np.int64,
            count=touched_count,
        )
        selected_distances = best_signed[selected_indices]
        selected_source_ordinals = best_source_ordinal[selected_indices]
        # The source/profile tuple can vary by selected arc, so evaluate by
        # source group while retaining a single chain contribution per cell.
        chain_values = np.zeros(selected_indices.size, dtype=np.float64)
        for source_ordinal in np.unique(selected_source_ordinals):
            positions = np.flatnonzero(selected_source_ordinals == source_ordinal)
            source_id = source_ids[int(source_ordinal)]
            chain_values[positions] = np.asarray(
                evaluate_boundary_profile(
                    profiles[source_id],
                    selected_distances[positions],
                    ordinary_crust[selected_indices[positions]],
                ),
                dtype=np.float64,
            )
        if np.any(~np.isfinite(chain_values)):
            raise ValueError("tectonic profile evaluation produced non-finite values")
        contributes = chain_values != 0.0
        if np.any(contributes):
            contributing_indices = selected_indices[contributes]
            contributing_values = chain_values[contributes]
            flat_raw_delta[contributing_indices] += contributing_values
            flat_count[contributing_indices] += 1
        # Reset only cells touched by this chain.  This preserves sparse-cap
        # work even when production chains outnumber grid rows.
        touched_indices = touched[:touched_count]
        best_absolute[touched_indices] = np.inf
        best_source_ordinal[touched_indices] = -1

    if np.any(flat_count > _MAX_UINT16):
        raise ValueError("influence_count exceeds uint16 capacity")
    if np.any(~np.isfinite(flat_raw_delta)):
        raise ValueError("tectonic raw delta is not finite")
    mask = flat_count > 0
    normalized = np.array(flat_raw_delta, dtype=np.float64, copy=True)
    normalized[mask] /= np.sqrt(flat_count[mask].astype(np.float64))
    normalized[~mask] = 0.0
    if np.any(~np.isfinite(normalized)):
        raise ValueError("tectonic elevation delta is not finite")

    diagnostics = {
        "chain_count": len(chains),
        "source_segment_count": len(segments),
        "influenced_cell_count": int(np.count_nonzero(mask)),
        "max_abs_raw_delta_m": float(np.max(np.abs(flat_raw_delta))) if flat_raw_delta.size else 0.0,
        "gaussian_cutoff_sigma": sigma,
        "overlap_normalization": "sum-over-sqrt-chain-count",
    }
    return TectonicPotential(
        normalized.reshape(grid.shape),
        mask.reshape(grid.shape),
        flat_count.astype(np.uint16, copy=False).reshape(grid.shape),
        diagnostics,
    )


__all__ = ["TectonicPotential", "build_tectonic_potential"]
