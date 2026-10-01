"""Native-constrained summaries of an unchanged physical land surface."""

import math

import numpy as np
import shapely


def _sweep_covers_native(chain, *, frame_shape):
    """Whether replacing this original chain can change a native centre."""
    height, width = frame_shape
    low = np.maximum(np.ceil(chain.min(axis=0) - .5).astype(np.int64), 0)
    high = np.minimum(np.floor(chain.max(axis=0) - .5).astype(np.int64), (width - 1, height - 1))
    if np.any(low > high):
        return False
    sweep = shapely.make_valid(shapely.Polygon(chain))
    xx, yy = np.meshgrid(np.arange(low[0], high[0] + 1) + .5,
                        np.arange(low[1], high[1] + 1) + .5)
    shapely.prepare(sweep)
    return bool(np.any(shapely.intersects_xy(sweep, xx, yy)))


def _native_safe_chain(chain, *, frame_shape):
    """Split a blocked summary chord until every replacement is native-safe.

    Only original vertices can split the chain. Canonical orientation makes
    both owners of a shared edge choose exactly the same subdivision.
    """
    reverse = tuple(chain[0]) > tuple(chain[-1])
    if reverse:
        chain = chain[::-1]
    height, width = frame_shape
    kept = [0]
    pending = [(0, len(chain)-1)]
    while pending:
        first, last = pending.pop()
        if last-first == 1:
            kept.append(last)
            continue
        points = chain[first:last+1]
        omitted = points[1:-1]
        touches_frame = bool(np.any((omitted[:, 0] == 0) | (omitted[:, 0] == width)
                                    | (omitted[:, 1] == 0) | (omitted[:, 1] == height)))
        if not touches_frame and not _sweep_covers_native(points, frame_shape=frame_shape):
            kept.append(last)
            continue
        chord = points[-1]-points[0]
        squared = float(chord @ chord)
        if squared:
            along = np.clip(((omitted-points[0]) @ chord)/squared, 0., 1.)
            displacement = omitted-(points[0]+along[:, None]*chord)
            distances = np.einsum('ij,ij->i', displacement, displacement)
            # A collinear blocked chain is split evenly, avoiding quadratic
            # work for exact frame segments and native centres on a chord.
            split = (last+first)//2 if not np.any(distances) else first+1+int(np.argmax(distances))
        else:
            split = (last+first)//2
        pending.append((split, last))
        pending.append((first, split))
    result = chain[kept]
    return result[::-1] if reverse else result


def _constrained_ring(original, simplified, *, frame_shape):
    points = np.asarray(original.coords, dtype=np.float64)[:-1]
    summary = np.asarray(simplified.coords, dtype=np.float64)[:-1]
    lookup = {tuple(point): index for index, point in enumerate(points)}
    try:
        indices = [lookup[tuple(point)] for point in summary]
    except KeyError as exc:
        raise ValueError("display simplification must retain original ring vertices") from exc
    count = len(points)
    if sum((last - first) % count for first, last in zip(indices, indices[1:] + indices[:1])) != count:
        raise ValueError("display simplification must retain original ring order")
    kept = []
    for first, last in zip(indices, indices[1:] + indices[:1]):
        distance = (last - first) % count
        if distance == 1:
            kept.append(points[first:first + 1])
            continue
        chain = points[(first + np.arange(distance + 1)) % count]
        # Each sweep is the exact region between the original subchain and
        # its replacement chord. Retaining it protects both wet and dry
        # centres, including centres lying directly on the proposed chord.
        kept.append(_native_safe_chain(chain, frame_shape=frame_shape)[:-1])
    result = np.concatenate(kept)
    return np.vstack((result, result[0]))


def generalize_display_surface(surface, *, frame_shape, tolerance):
    """Produce a DP summary using only classification-safe original chords.

    Fine rendering continues to use the original physical zero trace. This
    display copy adds no vertices, curvature, noise, or terrain deformation.
    Frame intersections and native-centre ownership cannot move. The result
    must retain valid, separate components and every original lake hole.
    """
    if (not math.isfinite(tolerance) or tolerance <= 0
            or len(frame_shape) != 2 or min(frame_shape) < 1
            or surface.geom_type not in ("Polygon", "MultiPolygon")
            or not bool(shapely.is_valid(surface))):
        raise ValueError("surface summary requires a valid polygon, native frame and positive tolerance")
    if surface.is_empty:
        return surface
    simplified = shapely.simplify(surface, tolerance, preserve_topology=True)
    original_parts = shapely.get_parts(surface)
    simplified_parts = shapely.get_parts(simplified)
    if len(original_parts) != len(simplified_parts):
        raise ValueError("surface summary must retain every physical component")
    parts = []
    for original, summary in zip(original_parts, simplified_parts, strict=True):
        if len(original.interiors) != len(summary.interiors):
            raise ValueError("surface summary must retain every physical lake hole")
        parts.append(shapely.Polygon(
            _constrained_ring(original.exterior, summary.exterior, frame_shape=frame_shape),
            [_constrained_ring(first, last, frame_shape=frame_shape)
             for first, last in zip(original.interiors, summary.interiors, strict=True)],
        ))
    result = parts[0] if surface.geom_type == "Polygon" else shapely.MultiPolygon(parts)
    if not bool(shapely.is_valid(result)):
        raise ValueError("native-constrained summary must preserve physical topology")
    return result
