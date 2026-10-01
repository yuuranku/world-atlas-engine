"""Classify recorded road/flow intersections by their actual native banks.

A shared overlap does not determine where a road changes banks. Junctions
can introduce another bank sector inside that overlap. Facility stations
therefore connect the two road approaches in a bank-face graph, rather than
selecting the entry endpoint of every overlapping centre line.
"""

import numpy as np
import shapely
from shapely.ops import substring
from scipy.sparse import csr_array
from scipy.sparse.csgraph import shortest_path


def _line_parts(geometry):
    if geometry.is_empty:
        return ()
    if geometry.geom_type == 'LineString':
        return (geometry,)
    if hasattr(geometry, 'geoms'):
        return tuple(line for part in geometry.geoms for line in _line_parts(part))
    return ()


def _point_parts(geometry):
    if geometry.geom_type == 'Point':
        return (geometry,)
    if hasattr(geometry, 'geoms'):
        return tuple(point for part in geometry.geoms for point in _point_parts(part))
    return ()


def _bank_faces(river, neighborhood):
    # Rays extend beyond the classification ring before a single shared
    # noding operation. Separately clipped rings/rays can leave an ulp gap
    # at the intersection and merge two distinct native banks.
    rays = shapely.intersection(river, neighborhood.buffer(.01))
    graph = shapely.node(shapely.union_all((neighborhood.boundary, rays)))
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(graph)))
    return [face for face in faces if neighborhood.covers(face.representative_point())]


def river_bank_owner(river, point, samples, *, neighborhood=None):
    """Return the local bank-face identifier of each approach sample.

    The enclosing region is only a local classification domain. The native
    river remains a zero-width separator, including odd-degree confluences.
    """
    radius = max(.1, max(point.distance(sample) for sample in samples) * 2)
    region = point.buffer(radius, quad_segs=8) if neighborhood is None else neighborhood
    faces = _bank_faces(river, region)
    return tuple(next((index for index, face in enumerate(faces)
                       if face.contains(sample)), None) for sample in samples)


def _bank_directions(river, point, faces, owners):
    """Choose each face's actual native fan sector at a crossing station."""
    coordinates = np.asarray(point.coords[0], dtype=np.float64)
    segments = np.concatenate([np.stack((np.asarray(reach.coords)[:-1],
                                          np.asarray(reach.coords)[1:]), axis=1)
                               for reach in _line_parts(river)])
    edges = shapely.linestrings(segments)
    distances = shapely.distance(edges, point)
    incident = distances < 1e-8
    rays = segments[incident].reshape((-1, 2)) - coordinates
    lengths = np.linalg.norm(rays, axis=1)
    rays, lengths = rays[lengths > 1e-8], lengths[lengths > 1e-8]
    if not len(rays):
        raise ValueError('a source crossing station must lie on a native flow edge')
    angles = np.unique(np.mod(np.arctan2(rays[:, 1], rays[:, 0]), 2*np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2*np.pi])
    midpoints = angles + gaps/2
    directions = np.column_stack((np.cos(midpoints), np.sin(midpoints)))
    # Stay inside the incident straight-edge star. Close nonincident edges
    # and short native branches constrain only this classification sample;
    # they never move a facility or create a physical crossing window.
    radius = min(.025, float(np.min(lengths))*.25)
    if np.any(~incident):
        radius = min(radius, float(np.min(distances[~incident]))*.25)
    samples = shapely.points(coordinates + directions*radius)
    result = []
    for owner in owners:
        contained = np.flatnonzero(shapely.contains(faces[owner], samples))
        if not len(contained):
            raise ValueError('a selected native bank must have an incident fan sector')
        # Canonical angular order makes bank metadata independent of route
        # reversal and of polygonization's face enumeration.
        direction = directions[contained[0]]
        result.append((float(direction[0]), float(direction[1])))
    return tuple(result)


def _overlap_crossing_transitions(road, river, overlap):
    ends = sorted((float(road.project(shapely.Point(overlap.coords[0]))),
                   float(road.project(shapely.Point(overlap.coords[-1])))))
    start, end = ends
    if start <= 1e-8 or road.length - end <= 1e-8:
        return ()
    samples = (road.interpolate(start - .025), road.interpolate(end + .025))
    neighborhood = substring(road, max(0., start - .05),
                             min(road.length, end + .05)).buffer(.1, quad_segs=8)
    faces = _bank_faces(river, neighborhood)
    owners = [next((index for index, face in enumerate(faces)
                    if face.contains(sample)), None) for sample in samples]
    if None in owners:
        raise ValueError('a native channel-overlap approach has no bank face')
    if owners[0] == owners[1]:
        return ()

    # Every recorded vertex of the overlap is a valid station. Native reach
    # endpoints also retain junctions that lie inside a merged overlap.
    candidates = {tuple(coordinates) for coordinates in overlap.coords}
    for reach in _line_parts(river):
        for coordinates in (reach.coords[0], reach.coords[-1]):
            point = shapely.Point(coordinates)
            if overlap.distance(point) < 1e-8:
                candidates.add(tuple(coordinates))
    candidates = sorted(candidates)
    bank_index = shapely.STRtree(faces)
    rows, columns, costs = [], [], []
    for index, coordinates in enumerate(candidates):
        point = shapely.Point(coordinates)
        # Native centres and half-edge intersections have exact shared
        # geometry; the small predicate tolerance covers noding roundoff.
        banks = bank_index.query(point, predicate='dwithin', distance=1e-8)
        if len(banks) < 2:
            continue
        # Traversing face -> candidate -> face costs one facility. A bounded
        # tie term prefers canonical stations without ever outweighing
        # a whole facility: its sum over all candidates is strictly < .5.
        cost = .5 + (index + 1) / (2 * (len(candidates) + 1) ** 2)
        for bank in banks:
            rows.append(int(bank)); columns.append(len(faces) + index); costs.append(cost)
    graph = csr_array((costs, (rows, columns)),
                      shape=(len(faces) + len(candidates),) * 2)
    distances, previous = shortest_path(graph, directed=False, indices=owners[0],
                                        method='D', return_predecessors=True)
    station = owners[1]
    if not np.isfinite(distances[station]):
        raise ValueError('native bank approaches have no source crossing station')
    chosen = [station]
    while station != owners[0]:
        station = int(previous[station])
        chosen.append(station)
    chosen.reverse()
    result = []
    for incoming, candidate, outgoing in zip(chosen[:-2:2], chosen[1::2], chosen[2::2], strict=True):
        point = shapely.Point(candidates[candidate - len(faces)])
        directions = _bank_directions(river, point, faces, (incoming, outgoing))
        result.append((point, *directions))
    return tuple(result)


def source_crossing_transitions(road, river):
    """Return source stations with their incoming and outgoing bank directions.

    Isolated transverse intersections are classified locally. Continuous
    channel overlaps choose their facilities through the native bank graph;
    tangent and same-bank visits require no facility.
    """
    result = []
    intersection = shapely.intersection(road, river)
    for point in _point_parts(intersection):
        distance = float(road.project(point))
        if distance <= 1e-8 or road.length - distance <= 1e-8:
            continue
        samples = (road.interpolate(max(0., distance - .025)),
                   road.interpolate(min(road.length, distance + .025)))
        neighborhood = point.buffer(.1, quad_segs=8)
        faces = _bank_faces(river, neighborhood)
        owners = tuple(next((index for index, face in enumerate(faces)
                            if face.contains(sample)), None) for sample in samples)
        if None not in owners and owners[0] != owners[1]:
            result.append((point, *_bank_directions(river, point, faces, owners)))
    overlaps = shapely.line_merge(shapely.union_all(_line_parts(intersection)))
    for overlap in _line_parts(overlaps):
        result.extend(_overlap_crossing_transitions(road, river, overlap))
    by_point = {transition[0]: transition for transition in result}
    return tuple(by_point[point] for point in sorted(by_point, key=road.project))


def source_crossing_points(road, river):
    """Return the source stations of the route's actual native bank changes."""
    return tuple(point for point, incoming, outgoing in source_crossing_transitions(road, river))


def endpoint_bank_direction(road, river, point):
    """Return an endpoint-star member's proven native bank sector.

    A channel overlap can defer the first off-channel approach. Its bank
    must remain incident to the source endpoint; an intervening branch must
    not be bypassed by inventing a local portal. A wholly overlapping road
    has no off-channel bank and returns None.
    """
    distance = float(road.project(point))
    if distance <= 1e-8:
        sample = road.interpolate(min(.025, road.length))
    elif road.length-distance <= 1e-8:
        sample = road.interpolate(max(0., road.length-.025))
    else:
        raise ValueError('an endpoint bank direction requires a native road endpoint')
    neighborhood = point.buffer(.1, quad_segs=8)
    if sample.distance(river) < 1e-8:
        overlaps = shapely.line_merge(shapely.union_all(_line_parts(
            shapely.intersection(road, river))))
        component = next((overlap for overlap in _line_parts(overlaps)
                          if overlap.distance(point) < 1e-8), None)
        if component is None:
            raise ValueError('an on-channel endpoint approach must have a source overlap')
        ends = sorted(float(road.project(shapely.Point(coordinates)))
                      for coordinates in (component.coords[0], component.coords[-1]))
        at_start = distance <= 1e-8
        exit_distance = ends[1] if at_start else ends[0]
        if (at_start and road.length-exit_distance <= 1e-8) or (
                not at_start and exit_distance <= 1e-8):
            return None
        sample_distance = min(road.length, exit_distance+.025) if at_start else max(0., exit_distance-.025)
        sample = road.interpolate(sample_distance)
        first, last = sorted((distance, sample_distance))
        neighborhood = substring(road, max(0., first-.05),
                                 min(road.length, last+.05)).buffer(.1, quad_segs=8)
    faces = _bank_faces(river, neighborhood)
    owner = next((index for index, face in enumerate(faces) if face.contains(sample)), None)
    if owner is None:
        raise ValueError('the first off-channel endpoint approach must have a native bank')
    return _bank_directions(river, point, faces, (owner,))[0]
