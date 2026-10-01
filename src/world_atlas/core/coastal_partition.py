"""Complete thematic coverages before clipping them to the physical shoreline."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
import shapely


@dataclass(frozen=True)
class CoastalPartition:
    """Working paint and its exact visible topology under the physical clip."""

    faces: tuple
    labels: np.ndarray
    visible_faces: tuple
    visible_labels: np.ndarray


def _polygon_parts(geometry):
    """Keep polygons even when an overlay also returns lines or nested parts."""

    if geometry.geom_type == "Polygon":
        if not geometry.is_empty:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _polygon_parts(part)


def extend_coastal_partition(values, land, *, margin, wrap_longitude):
    """Extend nearest land labels across a display-only coastal working band.

    A partition smoother may move a shoreline inward. Its working-domain
    edge therefore has to lie beyond the physical shoreline, rather than
    trying to make an independent thematic coastline fit afterwards.
    """
    values = np.asarray(values)
    land = np.asarray(land, dtype=bool)
    if values.ndim != 2 or values.shape != land.shape:
        raise ValueError("partition labels and land mask must share a 2D shape")
    if not np.any(land) or margin <= 0:
        raise ValueError("coastal partition needs land and a positive working margin")
    padding = int(np.ceil(margin)) if wrap_longitude else 0
    working_land = np.pad(land, ((0, 0), (padding, padding)), mode="wrap") if padding else land
    distance, indices = ndimage.distance_transform_edt(~working_land, return_indices=True)
    if padding:
        distance = distance[:, padding:-padding]
        indices = indices[:, :, padding:-padding]
    nearest_rows = indices[0]
    nearest_columns = (indices[1] - padding) % land.shape[1]
    extended = values[nearest_rows, nearest_columns]
    return extended, distance <= margin


def clip_partition_to_surface(faces, labels, land_surface):
    """Intersect a complete working coverage with the one physical surface.

    Missing coverage is a rendering error. Clipping outside colors alone is
    insufficient: the resulting faces must also cover every visible land area.
    """
    clipped_faces, clipped_labels = [], []
    shapely.prepare(land_surface)
    for face, label in zip(faces, labels, strict=True):
        if shapely.covers(land_surface, face):
            clipped = face
        elif not shapely.intersects(land_surface, face):
            continue
        else:
            # A tiny coastal category need not overlay every vertex of a
            # continent. The rectangular clip retains the exact shore, then
            # the ordinary polygon overlay supplies the final shared edge.
            local_surface = shapely.clip_by_rect(land_surface, *face.bounds)
            if not shapely.is_valid(local_surface):
                local_surface = shapely.make_valid(local_surface)
            clipped = shapely.intersection(face, local_surface)
        for part in _polygon_parts(clipped):
            clipped_faces.append(part)
            clipped_labels.append(int(label))
    coverage = shapely.union_all(clipped_faces)
    uncovered = shapely.area(shapely.difference(land_surface, coverage))
    outside = shapely.area(shapely.difference(coverage, land_surface))
    tolerance = max(1.0e-9, float(shapely.area(land_surface)) * 1.0e-11)
    if uncovered > tolerance or outside > tolerance:
        raise ValueError(f"thematic coverage misses its physical surface: uncovered={uncovered:g}, outside={outside:g}")
    return tuple(clipped_faces), np.asarray(clipped_labels, dtype=np.int32)


def enforce_homogeneous_components(faces, labels, values, land_mask, land_surface):
    """Keep a physical land component with one native label wholly that label.

    Working-band extension and shared-boundary smoothing can move a mainland
    category into a nearby island. A component whose actual dry cell centers
    have only one category has no native internal boundary to smooth. Its
    entire authoritative surface therefore retains that category, regardless
    of component size. Components containing multiple categories or no native
    centers retain their existing partition.
    """

    values = np.asarray(values)
    land = np.asarray(land_mask, dtype=bool)
    if values.ndim != 2 or values.shape != land.shape or values.dtype.kind not in "iu":
        raise ValueError("native categorical labels and land mask must share a 2D shape")
    height, width = land.shape
    fixed_components, fixed_labels = [], []
    for component in shapely.get_parts(land_surface):
        if component.geom_type != "Polygon" or component.is_empty:
            continue
        west, north, east, south = component.bounds
        first_column = max(0, int(np.ceil(west - .5)))
        last_column = min(width, int(np.floor(east - .5)) + 1)
        first_row = max(0, int(np.ceil(north - .5)))
        last_row = min(height, int(np.floor(south - .5)) + 1)
        if first_column >= last_column or first_row >= last_row:
            continue
        rows, columns = np.nonzero(land[first_row:last_row, first_column:last_column])
        rows += first_row
        columns += first_column
        if len(rows) == 0:
            continue
        # Prepare the actual component before testing centers, and crop its
        # raster candidates. A continental bbox does not imply slow polygon
        # tests at every point or include nearby islands with other labels.
        shapely.prepare(component)
        inside = shapely.contains_xy(component, columns + .5, rows + .5)
        native_labels = np.unique(values[rows[inside], columns[inside]])
        if len(native_labels) == 1:
            # Already uniform working paint needs no fine coast copied into
            # its theme asset. Only a real foreign-label intersection needs
            # the component's exact geometry to correct its ownership.
            native_label = int(native_labels[0])
            foreign = [face for face, label in zip(faces, labels, strict=True)
                       if int(label) != native_label and shapely.intersects(component, face)]
            if not any(shapely.area(shapely.intersection(component, face)) > 1e-12
                       for face in foreign):
                continue
            fixed_components.append(component)
            fixed_labels.append(native_label)

    retained_faces, retained_labels = [], []
    if fixed_components:
        fixed_surface = shapely.union_all(fixed_components)
        shapely.prepare(fixed_surface)
        for face, label in zip(faces, labels, strict=True):
            if not shapely.intersects(fixed_surface, face):
                remaining = face
            elif shapely.covers(fixed_surface, face):
                continue
            else:
                local_fixed = shapely.clip_by_rect(fixed_surface, *face.bounds)
                if not shapely.is_valid(local_fixed):
                    local_fixed = shapely.make_valid(local_fixed)
                remaining = shapely.difference(face, local_fixed)
            for part in _polygon_parts(remaining):
                retained_faces.append(part)
                retained_labels.append(int(label))
        retained_faces.extend(fixed_components)
        retained_labels.extend(fixed_labels)
    else:
        retained_faces.extend(faces)
        retained_labels.extend(int(label) for label in labels)

    # Combine fragments by category before the final physical-surface check.
    # Deducting all fixed components first prevents overlap with old colors.
    groups = {}
    for face, label in zip(retained_faces, retained_labels, strict=True):
        groups.setdefault(label, []).append(face)
    merged_faces, merged_labels = [], []
    for label, group in groups.items():
        for part in _polygon_parts(shapely.union_all(group)):
            merged_faces.append(part)
            merged_labels.append(label)
    return tuple(merged_faces), np.asarray(merged_labels, dtype=np.int32)
