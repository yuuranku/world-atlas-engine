"""Round one shared administrative arc graph before drawing its faces.

Mapshaper supplies the curve filter. Native observations, exterior arcs and
administrative junctions constrain that filter locally, on the shared graph.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import shapely

from ...runtime import require_renderer


_DISTANCE = .7
_MAX_DISPLACEMENT = .35
_SAMPLE_DISTANCE = .2

_SMOOTH_GRAPH = r"""
const fs = require('fs');
const path = require('path');
const m = require(process.argv[1]);
const directory = process.argv[2];
const content = JSON.parse(fs.readFileSync(path.join(directory, 'faces.json'), 'utf8'));
const dataset = m.internal.importContent({json: {filename: 'faces.geojson', content}}, {planar: true});
m.internal.buildTopology(dataset);
const original = dataset.arcs.getCopy();
const shapes = dataset.layers[0].shapes;
m.cmd.smooth(dataset, {distance: .7, planar: true, no_corners: true,
                      no_prefilter: true, gain: 0}, dataset.layers);
function save(arcs, prefix) {
  const data = arcs.getVertexData();
  for (const key of ['nn', 'xx', 'yy']) {
    const array = data[key];
    fs.writeFileSync(path.join(directory, prefix + '-' + key + '.bin'),
      Buffer.from(array.buffer, array.byteOffset, array.byteLength));
  }
}
save(original, 'original');
save(dataset.arcs, 'smooth');
fs.writeFileSync(path.join(directory, 'shapes.json'), JSON.stringify(shapes));
"""


def _shared_arc_candidates(faces):
    node, entry = require_renderer()
    with TemporaryDirectory(prefix='administrative-display-') as temporary:
        directory = Path(temporary)
        with (directory/'faces.json').open('w', encoding='utf8') as stream:
            stream.write('{"type":"FeatureCollection","features":[')
            for index, face in enumerate(faces):
                if index:
                    stream.write(',')
                stream.write('{"type":"Feature","properties":{},"geometry":')
                stream.write(shapely.to_geojson(face))
                stream.write('}')
            stream.write(']}')
        subprocess.run([node, '-e', _SMOOTH_GRAPH, str(entry.parent.parent), str(directory)],
                       check=True, capture_output=True, text=True)
        shapes = json.loads((directory/'shapes.json').read_text(encoding='utf8'))
        def read(prefix):
            counts = np.fromfile(directory/f'{prefix}-nn.bin', dtype=np.uint32)
            points = np.column_stack((np.fromfile(directory/f'{prefix}-xx.bin', dtype=np.float64),
                                      np.fromfile(directory/f'{prefix}-yy.bin', dtype=np.float64)))
            return np.split(points, np.cumsum(counts)[:-1])
        return read('original'), read('smooth'), shapes


def _arc_owners(shapes, count):
    owners = [set() for _ in range(count)]
    for owner, shape in enumerate(shapes):
        for ring in shape or ():
            for arc in ring:
                owners[arc if arc >= 0 else ~arc].add(owner)
    return owners


def _sampled_displacement(original, candidate):
    """Match the filtered curve to its nearest monotone source stations."""
    line = shapely.LineString(original)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(original, axis=0), axis=1))]
    closed = bool(np.array_equal(original[0], original[-1]))
    if closed:
        # A closed filter has no fixed first vertex. Rotate its samples to the
        # original origin so nearest source stations have the same seam.
        index = int(np.argmin(np.linalg.norm(candidate[:-1]-original[0], axis=1)))
        candidate = np.r_[candidate[index:-1], candidate[:index+1]]
    stations = shapely.line_locate_point(line, shapely.points(candidate))
    if closed:
        stations[0] = 0.
        stations[-1] = line.length
    else:
        stations[0], stations[-1] = 0., line.length
    # Keep all source turns and add only enough samples to test local sweeps.
    samples = np.unique(np.r_[lengths, np.linspace(0., line.length,
                                  max(2, int(np.ceil(line.length/_SAMPLE_DISTANCE))+1))])
    positions = shapely.get_coordinates(shapely.line_interpolate_point(line, samples))
    stations = np.maximum.accumulate(stations)
    keep = np.r_[True, np.diff(stations) > 1e-10]
    smooth = np.column_stack([np.interp(samples, stations[keep], candidate[keep, axis])
                              for axis in range(2)])
    delta = smooth-positions
    distance = np.linalg.norm(delta, axis=1)
    delta *= np.minimum(1., _MAX_DISPLACEMENT/np.maximum(distance, 1e-30))[:, None]
    if closed:
        delta[-1] = delta[0]
    else:
        # Junction positions and their incident ordering are fixed. Fade only
        # their neighbourhood, rather than weakening the entire border arc.
        fade = np.clip(np.minimum(samples, line.length-samples)/_DISTANCE, 0., 1.)
        delta *= (fade*fade*(3.-2.*fade))[:, None]
        delta[0] = delta[-1] = 0.
    return samples, positions, delta


def _unsafe_sweeps(original, moved, frame_shape):
    """Indices of local ribbons that cover a measured native centre."""
    height, width = frame_shape
    quads = np.stack((original[:-1], original[1:], moved[1:], moved[:-1]), axis=1)
    low = np.ceil(quads.min(axis=1)-.5).astype(np.int64)
    high = np.floor(quads.max(axis=1)-.5).astype(np.int64)
    possible = ((low[:, 0] <= high[:, 0]) & (low[:, 1] <= high[:, 1])
                & (low[:, 0] >= 0) & (low[:, 1] >= 0)
                & (low[:, 0] < width) & (low[:, 1] < height)
                & (np.any(original[:-1] != moved[:-1], axis=1)
                   | np.any(original[1:] != moved[1:], axis=1)))
    indices = np.flatnonzero(possible)
    if not len(indices):
        return indices
    sweeps = shapely.make_valid(shapely.polygons(quads[indices]))
    centres = low[indices]+.5
    unsafe = shapely.intersects(sweeps, shapely.points(centres))
    return indices[unsafe]


def _suppress_neighbourhood(samples, weight, centres):
    centres = np.unique(centres)
    indices = np.searchsorted(centres, samples)
    nearest = np.minimum(np.abs(samples-centres[np.maximum(indices-1, 0)]),
                         np.abs(samples-centres[np.minimum(indices, len(centres)-1)]))
    fade = np.clip(nearest/_DISTANCE, 0., 1.)
    return np.minimum(weight, fade*fade*(3.-2.*fade))


def _constrain_native(samples, original, delta, weight, *, frame_shape):
    while True:
        moved = original+delta*weight[:, None]
        unsafe = _unsafe_sweeps(original, moved, frame_shape)
        if not len(unsafe):
            return weight
        centres = np.r_[samples[unsafe], samples[unsafe+1]]
        # A local zero with a smooth ramp retains visible rounding on all
        # unconstrained parts, including the rest of a long shared border.
        reduced = _suppress_neighbourhood(samples, weight, centres)
        if np.array_equal(weight, reduced):
            raise ValueError('native centre constraint must reduce the local administrative sweep')
        weight = reduced


def _crossing_locations(arcs):
    """Locate new intersections without treating fixed junctions as failures."""
    lines = np.asarray([shapely.LineString(arc) for arc in arcs], dtype=object)
    pairs = shapely.STRtree(lines).query(lines, predicate='intersects')
    pairs = pairs[:, pairs[0] < pairs[1]]
    blocked = {}
    for first, last in pairs.T:
        intersection = shapely.intersection(lines[first], lines[last])
        common = set(map(tuple, arcs[first][[0, -1]])) & set(map(tuple, arcs[last][[0, -1]]))
        points = shapely.get_coordinates(intersection)
        if intersection.length == 0:
            points = np.asarray([point for point in points if tuple(point) not in common])
        if len(points):
            blocked.setdefault(int(first), []).extend(points)
            blocked.setdefault(int(last), []).extend(points)
    for index in np.flatnonzero(~shapely.is_simple(lines)):
        points = arcs[index]
        segments = shapely.linestrings(np.stack((points[:-1], points[1:]), axis=1))
        hits = shapely.STRtree(segments).query(segments, predicate='intersects')
        hits = hits[:, hits[0]+1 < hits[1]]
        closed = np.array_equal(points[0], points[-1])
        for first, last in hits.T:
            if closed and first == 0 and last == len(segments)-1:
                continue
            blocked.setdefault(int(index), []).extend(shapely.get_coordinates(
                shapely.intersection(segments[first], segments[last])))
    return blocked


def _face_geometries(arcs, shapes):
    result = []
    for shape in shapes:
        if not shape:
            result.append(shapely.Polygon())
            continue
        rings = []
        for identifiers in shape:
            parts = [arcs[arc] if arc >= 0 else arcs[~arc][::-1] for arc in identifiers]
            ring = np.concatenate([points[:-1] for points in parts]+[parts[-1][-1:]])
            rings.append(shapely.LineString(ring))
        geometry = shapely.build_area(shapely.MultiLineString(rings))
        if geometry.is_empty:
            geometry = shapely.Polygon()
        elif geometry.geom_type not in {'Polygon', 'MultiPolygon'}:
            raise ValueError('administrative arc rings must assemble polygonal faces')
        result.append(geometry)
    return np.asarray(result, dtype=object)


def administrative_display_coverage(faces, *, frame_shape):
    """Return rounded shared faces, in the original ownership order.

    The full frame and one-sided exterior boundaries remain exactly fixed.
    Native centres never enter the swept area of a changed administrative arc.
    """
    source = np.asarray(faces, dtype=object)
    if not len(source):
        return source
    if not bool(np.all(shapely.is_valid(source))) or not bool(shapely.coverage_is_valid(source)):
        raise ValueError('administrative display requires a valid shared source coverage')
    active = ~shapely.is_empty(source)
    if not np.any(active):
        return source.copy()
    painted = source[active]
    if any(face.geom_type not in {'Polygon', 'MultiPolygon'} for face in painted):
        raise ValueError('nonempty administrative source faces must be polygonal')
    original, smooth, shapes = _shared_arc_candidates(painted)
    owners = _arc_owners(shapes, len(original))
    fields = []
    for before, after, neighbours in zip(original, smooth, owners, strict=True):
        closed = np.array_equal(before[0], before[-1])
        unsupported_ring = closed and (shapely.LineString(before).length <= 2*_DISTANCE
                                       or len(after) < 4 or shapely.Polygon(after).area == 0)
        if len(neighbours) != 2 or len(before) <= 2 or unsupported_ring:
            fields.append(None)
            continue
        samples, positions, delta = _sampled_displacement(before, after)
        weight = _constrain_native(samples, positions, delta, np.ones(len(samples)),
                                   frame_shape=frame_shape)
        fields.append((samples, positions, delta, weight))
    while True:
        arcs = [before if field is None else field[1]+field[2]*field[3][:, None]
                for before, field in zip(original, fields, strict=True)]
        blocked = _crossing_locations(arcs)
        if not blocked:
            assembled = _face_geometries(arcs, shapes)
            collapsed = np.flatnonzero(shapely.is_empty(assembled))
            if not len(collapsed):
                break
            restored = False
            for owner in collapsed:
                for ring in shapes[owner] or ():
                    for identifier in ring:
                        index = identifier if identifier >= 0 else ~identifier
                        if fields[index] is not None and np.array_equal(original[index][0], original[index][-1]):
                            fields[index] = None
                            restored = True
            if restored:
                continue
            raise ValueError(f'nonempty administrative source faces must survive arc import: {collapsed.tolist()}')
        changed = False
        for index, points in blocked.items():
            field = fields[index]
            if field is None:
                continue
            samples, positions, delta, weight = field
            stations = shapely.line_locate_point(shapely.LineString(positions), shapely.points(points))
            indices = np.searchsorted(samples, stations)
            centres = np.r_[samples[np.maximum(indices-1, 0)], samples[np.minimum(indices, len(samples)-1)]]
            reduced = _suppress_neighbourhood(samples, weight, centres)
            reduced = _constrain_native(samples, positions, delta, reduced, frame_shape=frame_shape)
            changed |= not np.array_equal(weight, reduced)
            fields[index] = (samples, positions, delta, reduced)
        if not changed:
            raise ValueError('administrative crossing constraint must reduce a changed local arc')
    result = source.copy()
    result[active] = assembled
    if not bool(np.all(shapely.is_valid(result))) or not bool(shapely.coverage_is_valid(result)):
        raise ValueError('rounded administrative arcs must retain shared face topology')
    if not shapely.symmetric_difference(shapely.coverage_union_all(painted),
                                       shapely.coverage_union_all(assembled)).is_empty:
        raise ValueError('rounded administrative arcs must retain the complete frame')
    for before, after in zip(source, result, strict=True):
        first, last = shapely.get_parts(before), shapely.get_parts(after)
        if len(first) != len(last) or sorted(len(part.interiors) for part in first) != sorted(
                len(part.interiors) for part in last):
            raise ValueError('rounded administrative arcs must retain components and holes')
    return result
