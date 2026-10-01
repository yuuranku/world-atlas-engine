"""Adaptively encode known branches of an unchanged implicit model.

The model owns branch discovery and root isolation. This module only asks
that model for three true curve points between already shared branch ports;
it never invents a sampled scalar field or decides hidden topology.
"""

import numpy as np
import shapely


def polygon_path(polygon):
    """Encode a shared polygon and all its holes without changing vertices."""
    rings = [np.asarray(ring.coords) for ring in (polygon.exterior, *polygon.interiors)]
    codes = []
    for ring in rings:
        commands = np.full(len(ring), 2, np.uint8)
        commands[0], commands[-1] = 1, 79
        codes.append(commands)
    return np.concatenate(rings), np.concatenate(codes)


def adaptive_curve_paths(starts, ends, evaluate_sections, *, relative_error=.02):
    """Encode true branch sections using their normal distance to the chord.

    ``evaluate_sections(starts, ends, owner)`` returns quarter, middle and
    three-quarter points, in branch order. ``owner`` identifies the original
    branch and lets its solver retain the same root bracket after subdivision.
    Shared first/last ports remain byte-identical. Unresolved curvature fails
    rather than being replaced by a linear mesh edge.
    """
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    if (starts.ndim != 2 or starts.shape[1:] != (2,) or ends.shape != starts.shape
            or not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends))
            or not np.isfinite(relative_error) or relative_error <= 0):
        raise ValueError("implicit curves require matching finite branch ports and positive error")
    originals = np.stack((starts, ends), axis=1)
    starts, ends = starts.copy(), ends.copy()
    owner = np.arange(len(starts))
    begin, finish = np.zeros(len(starts)), np.ones(len(starts))
    accepted = []
    for _ in range(40):
        if not len(starts):
            break
        length = np.linalg.norm(ends-starts, axis=1)
        distinct = length > 8*np.finfo(float).eps*np.maximum(1., np.max(abs(starts), axis=1))
        accepted.extend(zip(owner[~distinct], begin[~distinct], starts[~distinct], strict=True))
        starts, ends, owner, begin, finish = (
            item[distinct] for item in (starts, ends, owner, begin, finish))
        if not len(starts):
            break
        samples = np.asarray(evaluate_sections(starts, ends, owner), dtype=float)
        if samples.shape != (len(starts), 3, 2) or not np.all(np.isfinite(samples)):
            raise ValueError("implicit model sections must return three finite true curve points")
        delta = ends-starts
        squared = np.sum(delta*delta, axis=1)
        projection = np.clip(np.sum((samples-starts[:, None, :])*delta[:, None, :], axis=2)
                             / squared[:, None], 0., 1.)
        feet = starts[:, None, :]+delta[:, None, :]*projection[:, :, None]
        error = np.linalg.norm(samples-feet, axis=2).max(axis=1)/np.sqrt(squared)
        subdivide = error > relative_error
        accepted.extend(zip(owner[~subdivide], begin[~subdivide], starts[~subdivide], strict=True))
        middle = samples[subdivide, 1]
        midpoint = (begin[subdivide]+finish[subdivide])*.5
        starts, ends = np.concatenate((starts[subdivide], middle)), np.concatenate((middle, ends[subdivide]))
        owner = np.tile(owner[subdivide], 2)
        begin, finish = np.concatenate((begin[subdivide], midpoint)), np.concatenate((midpoint, finish[subdivide]))
    else:
        if len(starts):
            raise ValueError("implicit curve curvature must converge")
    accepted.sort(key=lambda item: (int(item[0]), float(item[1])))
    chains = [[] for _ in range(len(originals))]
    for identifier, _position, point in accepted:
        chains[int(identifier)].append(point)
    return [np.vstack((*chain, segment[1])) for chain, segment in zip(chains, originals, strict=True)
            if np.any(segment[0] != segment[1])]


def level_bands(field, levels, paths):
    """Node shared true curves once and classify faces on their source model."""
    levels = np.asarray(levels, dtype=float)
    if (levels.ndim != 1 or not np.all(np.isfinite(levels))
            or np.any(np.diff(levels) <= 0) or len(paths) != len(levels)):
        raise ValueError("implicit bands require ordered levels and matching true curves")
    west, north, east, south = 0., 0., float(field.width), float(field.height)
    lines = [shapely.LineString(((west,north),(east,north),(east,south),
                                (west,south),(west,north)))]
    lines.extend(shapely.LineString(chain) for chains in paths for chain in chains)
    noded = shapely.node(shapely.MultiLineString(lines))
    faces, cuts, dangles, invalid = shapely.polygonize_full(shapely.get_parts(noded))
    if not cuts.is_empty or not dangles.is_empty or not invalid.is_empty:
        raise ValueError("implicit contours must form a closed shared coverage")
    faces = shapely.get_parts(faces)
    probes = shapely.get_coordinates(shapely.point_on_surface(faces))
    owner = np.searchsorted(levels, field.sample_points(probes[:,0],probes[:,1]), side="right")
    return [shapely.union_all(faces[owner == band]) for band in range(len(levels)+1)]
