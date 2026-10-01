"""Native raster topology and shared continuous categorical coverages."""

import numpy as np
from scipy import ndimage
import shapely


_SUBDIVISIONS = 4
_TILE_CELLS = 32
_DELIVERY_PRECISION = 1.0e-8


def _cardinal_weights(coordinates):
    """Four-sample Keys cubic convolution, exactly interpolating observations."""
    native = np.asarray(coordinates, dtype=np.float64) - .5
    first = np.floor(native).astype(np.int64)
    fraction = native - first
    weights = np.stack((
        -.5*fraction + fraction*fraction - .5*fraction**3,
        1.0 - 2.5*fraction*fraction + 1.5*fraction**3,
        .5*fraction + 2.0*fraction*fraction - 1.5*fraction**3,
        -.5*fraction*fraction + .5*fraction**3,
    ))
    return first[None, :] + np.arange(-1, 3)[:, None], weights


def _indicator_scores(categories, x, y):
    """Interpolate separate indicators; category identifiers are never numbers.

    Only labels in the compact four-by-four sample support are allocated.
    Longitude is periodic and latitude repeats the outermost observation.
    """
    height, width = categories.shape
    columns, horizontal_weights = _cardinal_weights(x)
    rows, vertical_weights = _cardinal_weights(y)
    source_rows = np.arange(int(rows.min()), int(rows.max()) + 1)
    source_columns = np.arange(int(columns.min()), int(columns.max()) + 1)
    observations = categories[np.clip(source_rows, 0, height-1)[:, None],
                              source_columns[None, :] % width]
    identifiers = np.unique(observations)
    horizontal_indices = columns - source_columns[0]
    vertical_indices = rows - source_rows[0]
    scores = np.empty((len(identifiers), len(y), len(x)), dtype=np.float64)
    for index, identifier in enumerate(identifiers):
        indicator = observations == identifier
        horizontal = sum(indicator[:, horizontal_indices[offset]]
                         * horizontal_weights[offset] for offset in range(4))
        scores[index] = sum(horizontal[vertical_indices[offset]]
                            * vertical_weights[offset, :, None] for offset in range(4))
    return identifiers, scores


def _triangle_boundaries(points, scores):
    """Trace the shared upper envelope of categorical scores in triangles.

    Two-owner intersections use score differences, rather than midpoints of
    classified pixels. Three-owner intersections solve the same affine score
    equalities for all owners, producing one identical junction coordinate.
    """
    edge_first = np.array((0, 1, 2))
    edge_last = np.array((1, 2, 0))
    # A material can win between vertices without winning a triangle corner.
    # Keep all locally present competitors and clip each equal-score segment
    # by every competitor. Corner-winner-only marching leaves broken edges at
    # precisely the crowded junctions found in actual administrative maps.
    possible = np.max(scores, axis=2) > 0
    segments = []
    for first_owner in range(scores.shape[1]):
        for last_owner in range(first_owner+1, scores.shape[1]):
            difference = scores[:, first_owner] - scores[:, last_owner]
            selected = (possible[:, first_owner] & possible[:, last_owner]
                        & (difference.min(axis=1) < 0) & (difference.max(axis=1) >= 0))
            if not bool(selected.any()):
                continue
            ids = np.flatnonzero(selected)
            difference = difference[ids]
            crossings = ((difference[:, edge_first] >= 0)
                         != (difference[:, edge_last] >= 0))
            triangle_rows, triangle_edges = np.nonzero(crossings)
            starts, ends = edge_first[triangle_edges], edge_last[triangle_edges]
            triangle_points = points[ids]
            first_points = triangle_points[triangle_rows, starts]
            last_points = triangle_points[triangle_rows, ends]
            reverse = ((first_points[:, 0] > last_points[:, 0])
                       | ((first_points[:, 0] == last_points[:, 0])
                          & (first_points[:, 1] > last_points[:, 1])))
            # Both incident triangles must evaluate an edge in the same
            # direction. Reversed floating arithmetic can round an exact
            # half-grid intersection to two distinct delivery coordinates.
            starts, ends = np.where(reverse, ends, starts), np.where(reverse, starts, ends)
            fraction = (difference[triangle_rows, starts]
                        / (difference[triangle_rows, starts] - difference[triangle_rows, ends]))
            intersections = (triangle_points[triangle_rows, starts] * (1-fraction[:, None])
                             + triangle_points[triangle_rows, ends] * fraction[:, None]).reshape(-1, 2, 2)
            barycentric = np.zeros((len(triangle_rows), 3), dtype=np.float64)
            barycentric[np.arange(len(triangle_rows)), starts] = 1-fraction
            barycentric[np.arange(len(triangle_rows)), ends] = fraction
            barycentric = barycentric.reshape(-1, 2, 3)
            values = np.einsum("nkj,nij->nik", scores[ids], barycentric)
            restrictions = values[:, :, first_owner, None] - values
            change = restrictions[:, 1] - restrictions[:, 0]
            roots = np.divide(-restrictions[:, 0], change,
                              out=np.zeros_like(change), where=np.abs(change) > 1e-14)
            lower = np.maximum(0., np.max(np.where(change > 1e-14, roots, -np.inf), axis=1))
            upper = np.minimum(1., np.min(np.where(change < -1e-14, roots, np.inf), axis=1))
            blocked = np.any((np.abs(change) <= 1e-14) & (restrictions[:, 0] < -1e-14), axis=1)
            keep = (upper > lower) & ~blocked
            fractions = np.stack((lower[keep], upper[keep]), axis=1)
            intersections = (intersections[keep, :1] * (1-fractions[:, :, None])
                             + intersections[keep, 1:] * fractions[:, :, None])
            intersections = np.round(intersections, 8)
            nonzero = np.any(intersections[:, 0] != intersections[:, 1], axis=1)
            segments.append(intersections[nonzero])
    return np.concatenate(segments) if segments else np.empty((0, 2, 2))


def _tile_boundaries(categories, west, north, east, south):
    x = np.arange(west*_SUBDIVISIONS, east*_SUBDIVISIONS+1)/_SUBDIVISIONS
    y = np.arange(north*_SUBDIVISIONS, south*_SUBDIVISIONS+1)/_SUBDIVISIONS
    identifiers, sampled = _indicator_scores(categories, x, y)
    if len(identifiers) == 1:
        return np.empty((0, 2, 2))
    winners = np.argmax(sampled, axis=0)
    mixed = ((winners[:-1, :-1] != winners[:-1, 1:])
             | (winners[:-1, :-1] != winners[1:, 1:])
             | (winners[:-1, :-1] != winners[1:, :-1]))
    row, column = np.nonzero(mixed)
    if not len(row):
        return np.empty((0, 2, 2))
    corner_rows = np.stack((row, row, row+1, row+1), axis=1)
    corner_columns = np.stack((column, column+1, column+1, column), axis=1)
    corners = np.stack((x[corner_columns], y[corner_rows]), axis=2)
    corner_scores = sampled[:, corner_rows, corner_columns].transpose(1, 0, 2)
    quad_points = np.concatenate((corners, corners.mean(axis=1)[:, None]), axis=1)
    quad_scores = np.concatenate((corner_scores, corner_scores.mean(axis=2)[:, :, None]), axis=2)
    # A symmetric fan avoids imposing one preferred diagonal on every quad.
    triangles = np.array(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
    points = quad_points[:, triangles].reshape(-1, 3, 2)
    scores = quad_scores[:, :, triangles].transpose(0, 2, 1, 3).reshape(-1, len(identifiers), 3)
    return _triangle_boundaries(points, scores)


def _face_owner(categories, face):
    """Evaluate the same sampled, piecewise affine ownership reconstruction."""
    point = face.representative_point()
    west = np.floor(point.x*_SUBDIVISIONS)/_SUBDIVISIONS
    north = np.floor(point.y*_SUBDIVISIONS)/_SUBDIVISIONS
    step = 1/_SUBDIVISIONS
    identifiers, values = _indicator_scores(categories, (west, west+step), (north, north+step))
    corners = values[:, (0, 0, 1, 1), (0, 1, 1, 0)]
    centre = corners.mean(axis=1)
    sx, sy = (point.x-west)/step, (point.y-north)/step
    distances = (sy, 1-sx, 1-sy, sx)
    edge = int(np.argmin(distances))
    first, last = ((0, 1), (1, 2), (2, 3), (3, 0))[edge]
    centre_weight = 2*distances[edge]
    along = (sx, sy, 1-sx, 1-sy)[edge]
    last_weight = along-centre_weight*.5
    first_weight = 1-centre_weight-last_weight
    scores = (corners[:, first]*first_weight + corners[:, last]*last_weight
              + centre*centre_weight)
    return int(identifiers[np.argmax(scores)])


def categorical_coverage(values, active_mask, *, category_count):
    """Reconstruct one exact shared coverage from categorical observations.

    The native raster defines observations at cell centres, not square polygon
    outlines. Compact cubic indicator fields compete between observations;
    subcell contour intersections build a single planar boundary graph. Its
    faces supply both fills and border ink. No polygon rounding, noise, SVG
    curves, topology repair, or external smoothing changes the result.

    Memory depends on a tile's locally present labels, never on the global
    number of provinces. Inactive observations form one excluded material;
    callers with a physical shoreline extend their working paint before this
    operation and clip the finished coverage to that authoritative surface.
    """
    values = np.asarray(values)
    active = np.asarray(active_mask, dtype=bool)
    if (values.ndim != 2 or min(values.shape) < 1 or values.shape != active.shape
            or values.dtype.kind not in "iu"):
        raise ValueError("categorical observations require matching nonempty integer rasters")
    if isinstance(category_count, bool) or int(category_count) != category_count or category_count < 1:
        raise ValueError("category_count must be a positive integer")
    if np.any(values[active] < 0) or np.any(values[active] >= category_count):
        raise ValueError("categorical observation exceeds category_count")
    if not bool(active.any()):
        return (), np.empty(0, dtype=np.int32)
    categories = np.where(active, values, -1).astype(np.int64)
    height, width = categories.shape
    segments = []
    for north in range(0, height, _TILE_CELLS):
        for west in range(0, width, _TILE_CELLS):
            boundary = _tile_boundaries(categories, west, north,
                                        min(west+_TILE_CELLS, width), min(north+_TILE_CELLS, height))
            if len(boundary):
                segments.append(boundary)
    frame = shapely.box(0, 0, width, height)
    if not segments:
        return (frame,), np.array((int(categories[0, 0]),), dtype=np.int32)
    lines = shapely.linestrings(np.concatenate(segments))
    linework = shapely.node(shapely.MultiLineString([*lines, frame.boundary]))
    polygons, cuts, dangles, invalid = shapely.polygonize_full(shapely.get_parts(linework))
    if not cuts.is_empty or not dangles.is_empty or not invalid.is_empty:
        raise ValueError("categorical contour graph contains unresolved shared boundaries")
    faces = shapely.get_parts(polygons)
    labels = np.array([_face_owner(categories, face) for face in faces], dtype=np.int32)
    kept = labels >= 0
    # Shared delivery coordinates are fixed once. Remove GEOS's attached grid
    # model afterwards: overlays must retain the physical sea-level contour's
    # full precision instead of snapping that independent authoritative field.
    result = tuple(shapely.set_precision(
        shapely.set_precision(faces[kept], _DELIVERY_PRECISION, mode="pointwise"), 0.0))
    labels = labels[kept]
    if not bool(shapely.coverage_is_valid(result)) or not bool(np.all(shapely.is_valid(result))):
        raise ValueError("categorical contour graph must produce an exact valid coverage")
    if set(labels) != set(values[active]):
        missing = sorted(set(values[active]) - set(labels))
        raise ValueError(f"categorical contour reconstruction lost native materials: {missing}")
    return result, labels


def periodic_component_labels(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    """Label a longitude-periodic raster with 4- or 8-neighbour connectivity."""

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    if not count:
        return labels.astype(np.int32, copy=False), 0

    parent = np.arange(count + 1, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        left = find(first)
        right = find(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    height = labels.shape[0]
    row_offsets = (0,) if connectivity == 4 else (-1, 0, 1)
    for row in range(height):
        left = int(labels[row, 0])
        if not left:
            continue
        for offset in row_offsets:
            other_row = row + offset
            if 0 <= other_row < height:
                right = int(labels[other_row, -1])
                if right:
                    union(left, right)

    roots = np.arange(count + 1, dtype=np.int32)
    for label in range(1, count + 1):
        roots[label] = find(label)
    canonical = roots[labels]
    # Renumbering is not required for the evidence below, but it makes the
    # reported component count precise after seam unions.
    present = np.unique(canonical[canonical > 0])
    remap = np.zeros(count + 1, dtype=np.int32)
    remap[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    return remap[canonical], int(len(present))

