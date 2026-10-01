"""Reconstruct river centre lines from their drainage corridor and ground field."""

import numpy as np
import shapely


def _forward_corridor_offsets(samples, normal, offsets):
    """Keep native station order while solving neighbouring valley sections.

    Independent transverse minima can fold the two banks of an elbow onto
    the same point. Constrain both consecutive stations and the two-station
    chord used by the channel normal: their forward projection retains at
    least half the accepted drainage chord. Only offsets that consume that
    corridor are reduced. Reducing any offset cannot increase its harmful
    contribution, so previously checked constraints remain satisfied. Each
    span's constraints are applied together, independently of flow direction.
    """
    for span in (1, 2):
        chord = samples[span:] - samples[:-span]
        budget = .5 * np.einsum("ij,ij->i", chord, chord)
        if np.any(budget <= 1e-24):
            raise ValueError("river drainage anchors cannot reverse at the same station")
        before = np.maximum(0.0, offsets[:-span] * np.einsum(
            "ij,ij->i", normal[:-span], chord))
        after = np.maximum(0.0, -offsets[span:] * np.einsum(
            "ij,ij->i", normal[span:], chord))
        harmful = before + after
        ratio = np.minimum(1.0, np.divide(budget, harmful,
            out=np.ones_like(budget), where=harmful > 0))
        factor = np.ones_like(offsets)
        factor[:-span] = np.minimum(factor[:-span], np.where(before > 0, ratio, 1.0))
        factor[span:] = np.minimum(factor[span:], np.where(after > 0, ratio, 1.0))
        offsets *= factor
    return offsets


def _source_stations(grid, points):
    source = np.asarray(points, dtype=np.float64)
    if (source.ndim != 2 or source.shape[1] != 2
            or not np.all(np.isfinite(source))):
        raise ValueError("river anchors must be finite (N, 2) coordinates")
    if len(source) < 2:
        return source.copy(), np.empty(0, dtype=np.int32), np.empty((0, 2, 2))
    edges = np.diff(source, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(np.abs(edges[:, 0]) > grid.shape[1] / 2):
        raise ValueError("river reaches must be split at the longitude seam")
    pieces = [source[:1]]
    identifiers = []
    segments = []
    for first, vector, length in zip(source[:-1], edges, lengths, strict=True):
        if length <= 1e-12:
            continue
        fraction = np.linspace(0, 1, max(1, int(np.ceil(length * 4))) + 1)[1:]
        pieces.append(first + fraction[:, None] * vector)
        identifiers.extend([len(segments)] * len(fraction))
        segments.append((first, first + vector))
    samples = np.vstack(pieces)
    return samples, np.asarray(identifiers, dtype=np.int32), np.asarray(segments).reshape(-1, 2, 2)


def _network_corridor_limits(stations):
    """Bound each source chord's displacement by the other source edges.

    Two foreign chords cannot meet when the sum of their displacement radii
    is smaller than their original separation. Source intersections have
    zero clearance, so their incident chords retain the actual embedding.
    Adjacent edges of one reach share their local moving station and use the
    forward-order constraint instead. Nonadjacent edges of that same reach
    participate in the clearance graph, including intrinsic intersections.
    """
    edge_groups = [segments for _samples, _ids, segments in stations if len(segments)]
    if not edge_groups:
        return [np.zeros(len(samples)) for samples, _ids, _segments in stations]
    edges = shapely.linestrings(np.concatenate(edge_groups))
    owners = np.concatenate([np.full(len(segments), index, dtype=np.int32)
                             for index, (_samples, _ids, segments) in enumerate(stations)
                             if len(segments)])
    ordinals = np.concatenate([np.arange(len(segments), dtype=np.int32)
                               for _samples, _ids, segments in stations if len(segments)])
    tree = shapely.STRtree(edges)
    result = []
    for owner, (samples, identifiers, _segments) in enumerate(stations):
        if len(samples) < 2:
            result.append(np.zeros(len(samples)))
            continue
        chords = shapely.linestrings(np.stack((samples[:-1], samples[1:]), axis=1))
        clearance = np.full(len(chords), np.inf)
        for start in range(0, len(chords), 16384):
            first, second = tree.query(chords[start:start+16384],
                                       predicate='dwithin', distance=.5)
            first = first + start
            foreign = ((owners[second] != owner)
                       | (np.abs(identifiers[first] - ordinals[second]) > 1))
            first, second = first[foreign], second[foreign]
            if len(first):
                distances = shapely.distance(chords[first], edges[second])
                np.minimum.at(clearance, first, distances)
        # Keep a relative margin between the source corridors: double and
        # delivered SVG precision must not turn a limiting tangency into a
        # new junction. This is a source-distance bound, not a paint width.
        radii = np.minimum(.25, .49 * clearance)
        limits = np.zeros(len(samples))
        limits[1:-1] = np.minimum(radii[:-1], radii[1:])
        result.append(limits)
    return result


def _terrain_channel(grid, samples, limits, *, terrain_field):
    if len(samples) < 3:
        return samples
    tangent = samples[2:] - samples[:-2]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    center = samples[1:-1]
    radius = limits[1:-1]
    sections = np.stack((center - radius[:, None] * normal, center,
                         center + radius[:, None] * normal))
    height, width = grid.shape
    inside = ((sections[:, :, 0] >= 0) & (sections[:, :, 0] < width)
              & (sections[:, :, 1] >= 0) & (sections[:, :, 1] < height))
    rows = np.clip(np.floor(sections[:, :, 1]).astype(int), 0, height - 1)
    columns = np.floor(sections[:, :, 0]).astype(int) % width
    supported = (radius > 0) & np.all(inside & (grid.water[rows, columns] == 0), axis=0)
    selected = np.flatnonzero(supported)
    if selected.size:
        cross = sections[:, selected]
        ground = terrain_field.sample_points(cross[:, :, 0], cross[:, :, 1])
        curvature = ground[0] + ground[2] - 2 * ground[1]
        offset = np.divide(radius[selected] * (ground[0] - ground[2]), 2 * curvature,
                           out=np.zeros_like(curvature), where=curvature > 1e-9)
        offset = np.clip(offset, -radius[selected], radius[selected])
        normals = np.zeros_like(samples)
        normals[1:-1] = normal
        offsets = np.zeros(len(samples), dtype=np.float64)
        offsets[selected + 1] = offset
        offsets = _forward_corridor_offsets(samples, normals, offsets)
        shifted = center[selected] + offsets[selected + 1, None] * normal[selected]
        shifted_ground = terrain_field.sample_points(shifted[:, 0], shifted[:, 1])
        # The accepted shoreline is the ground field's zero contour. A
        # native land cell can contain subcell water, so ownership alone
        # cannot license moving the channel below that datum.
        lower = (shifted_ground > 0) & (shifted_ground <= ground[1] + 1e-9)
        samples[selected[lower] + 1] = shifted[lower]
    return samples


def terrain_channel_paths(grid, paths, *, terrain_field):
    """Solve terrain sections inside one topology-preserving river network.

    Accepted source edges define each section's allowed displacement before
    the ground field is sampled. Shared confluence fans and existing edge
    intersections retain their native connectivity; distant valley bottoms
    can move within their source clearance. Flat and concave ground retain
    the original centre. Endpoints, native land and true positive ground
    remain mandatory, with no independent-reach reconstruction path.
    """
    if terrain_field.native_m.shape != grid.shape:
        raise ValueError("river reconstruction requires aligned physical ground")
    stations = [_source_stations(grid, points) for points in paths]
    limits = _network_corridor_limits(stations)
    return [_terrain_channel(grid, samples, allowed, terrain_field=terrain_field)
            for (samples, _ids, _segments), allowed in zip(stations, limits, strict=True)]
