"""Actual interior/interior nearest-river switching candidates.

The accepted terrain model uses distance to native straight river edges.
The minimum of two distances can have a real derivative corner on their
bisector. These events belong to that source model, rather than to an SVG
approximation. Parallel edges have one bisector; nonparallel edges have two
angle bisectors. Endpoint projection branches are outside this helper's
explicit scope; candidates must be checked at the joint true-field root.
"""

from itertools import combinations

import numpy as np
import shapely


def _linear_interval(low, high, value, slope, minimum, maximum):
    """Intersect a line parameter interval with one coordinate constraint."""
    if slope == 0:
        return (low, high) if minimum <= value <= maximum else None
    first, last = sorted(((minimum-value)/slope, (maximum-value)/slope))
    low, high = max(low, first), min(high, last)
    return (low, high) if low < high else None


def _parallel_pair(first, second):
    direction = first[1]-first[0]
    other = second[1]-second[0]
    if not np.any(direction) or not np.any(other):
        return None
    if direction[0]*other[1] != direction[1]*other[0]:
        return None
    normal = np.array((-direction[1], direction[0]))
    if normal[np.flatnonzero(normal)[0]] < 0:
        normal = -normal
    squared_normal = float(normal@normal)
    first_offset, second_offset = float(normal@first[0]), float(normal@second[0])
    separation = second_offset-first_offset
    # The actual smoothstep river-protection profile is constant on either
    # side of this open interval. Its nearest identity cannot introduce a
    # height derivative corner there. Coincident supporting lines have no
    # isolated interior/interior switching line either.
    distance_squared = separation*separation/(4*squared_normal)
    if not .18*.18 < distance_squared < .5*.5:
        return None
    offset = (first_offset+second_offset)/2
    origin = first[0] + normal*((offset-first_offset)/squared_normal)
    return normal, offset, origin, direction, distance_squared


def _interior_pair(first, second):
    direction, other = first[1]-first[0], second[1]-second[0]
    if not np.any(direction) or not np.any(other):
        return []
    if direction[0]*other[1] == direction[1]*other[0]:
        model = _parallel_pair(first, second)
        # Preserve the actual D8 integer equation and all existing parallel
        # segment endpoints. These source events do not need a new chart.
        return [] if model is None else [(*model[:4], None)]
    normals = [np.array((-delta[1], delta[0])) / np.linalg.norm(delta)
               for delta in (direction, other)]
    offsets = [float(normal@points[0]) for normal, points in zip(normals, (first, second), strict=True)]
    result = []
    for sign in (-1, 1):
        normal = normals[0] + sign*normals[1]
        offset = offsets[0] + sign*offsets[1]
        if normal[np.flatnonzero(normal)[0]] < 0:
            normal, offset = -normal, -offset
        origin = first[0] + normal*((offset-float(normal@first[0]))/float(normal@normal))
        tangent = np.array((-normal[1], normal[0]))
        signed_distance = (float(normals[0]@origin)-offsets[0], float(normals[0]@tangent))
        result.append((normal, offset, origin, tangent, signed_distance))
    return result


def interior_river_medials(field, lower, upper):
    """Return source-edge bisectors intersecting the supplied physical boxes.

    Each record contains ``owner``, ``normal``, ``offset`` and
    ``sourceRiverIndices`` with ``normal @ point == offset``. ``segment`` is
    the clipped line parameter range; its endpoint closure does not license
    endpoint projections. The consumer must retain only true-field roots
    whose projections onto *both* source segments are strictly interior and
    whose nearest source identity is this pair. No buffer changes the model.

    Indices refer directly to ``field._rivers.geometries``, including the
    accepted model's periodic copies. Longitude is not separately wrapped.
    """
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.ndim != 2 or lower.shape[1:] != (2,) or upper.shape != lower.shape
            or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)):
        raise ValueError("river-medial events require finite positive-area physical boxes")
    if field._rivers is None or not len(lower):
        return []
    tree = field._rivers
    width = float(field.width)
    if not np.isfinite(width) or width <= 0 or len(tree.geometries) % 3:
        raise ValueError("river-medial events require the accepted three-period native source tree")
    native_count = len(tree.geometries)//3
    boxes = shapely.box(lower[:, 0]-.5, lower[:, 1]-.5,
                        upper[:, 0]+.5, upper[:, 1]+.5)
    owners, indices = tree.query(boxes)
    ordered = np.argsort(owners, kind="stable")
    owners, indices = owners[ordered], indices[ordered]
    beginnings = np.r_[0, np.flatnonzero(np.diff(owners))+1, len(owners)]
    coordinates, pairs, events = {}, {}, []
    for begin, end in zip(beginnings[:-1], beginnings[1:], strict=True):
        if begin == end:
            continue
        owner = int(owners[begin])
        for first_id, second_id in combinations(sorted(indices[begin:end]), 2):
            first_id, second_id = int(first_id), int(second_id)
            for identifier in (first_id, second_id):
                if identifier not in coordinates:
                    points = np.asarray(tree.geometries[identifier].coords, dtype=float)
                    if points.shape != (2, 2):
                        raise ValueError("river-medial events require actual straight native edges")
                    coordinates[identifier] = points
            pair = first_id, second_id
            if pair not in pairs:
                sources = sorted((identifier % native_count, identifier//native_count-1, identifier)
                                 for identifier in pair)
                (first_native, period, _), (second_native, second_period, _) = sources
                relative_period = second_period-period
                primary = []
                for native, source_period, identifier in sources:
                    points = np.asarray(tree.geometries[native_count+native].coords, dtype=float)
                    if not np.array_equal(coordinates[identifier], points+(source_period*width, 0)):
                        raise ValueError("river-medial copies must retain their actual native source coordinates")
                    primary.append(points+((source_period-period)*width, 0))
                parallel = (primary[0][1,0]-primary[0][0,0])*(primary[1][1,1]-primary[1][0,1]) == (
                    primary[0][1,1]-primary[0][0,1])*(primary[1][1,0]-primary[1][0,0])
                models = []
                if parallel:
                    for normal, offset, origin, direction, signed_distance in _interior_pair(
                            coordinates[first_id], coordinates[second_id]):
                        canonical_offset = (float(normal@primary[0][0])+float(normal@primary[1][0]))/2
                        models.append((normal, offset, origin, direction, signed_distance,
                                       (first_native, second_native, relative_period, 0), period, canonical_offset))
                else:
                    for bisector, (normal, canonical_offset, origin, direction, signed_distance) in enumerate(
                            _interior_pair(*primary)):
                        offset = canonical_offset+normal[0]*(period*width)
                        models.append((normal, offset, origin+(period*width, 0), direction, signed_distance,
                                       (first_native, second_native, relative_period, bisector), period, canonical_offset))
                pairs[pair] = models
            for normal, offset, origin, direction, signed_distance, key, period, canonical_offset in pairs[pair]:
                interval = -np.inf, np.inf
                for axis in range(2):
                    interval = _linear_interval(*interval, origin[axis], direction[axis],
                                                lower[owner, axis], upper[owner, axis])
                    if interval is None:
                        break
                if interval is None:
                    continue
                for identifier in pair:
                    points = coordinates[identifier]
                    delta = points[1]-points[0]
                    length_squared = float(delta@delta)
                    projection = float((origin-points[0])@delta)/length_squared
                    slope = float(direction@delta)/length_squared
                    interval = _linear_interval(*interval, projection, slope, 0., 1.)
                    if interval is None:
                        break
                if interval is None:
                    continue
                intervals = [interval] if signed_distance is None else [
                    _linear_interval(*interval, *signed_distance, a, b)
                    for a, b in ((-.5, -.18), (.18, .5))]
                for active in intervals:
                    if active is None:
                        continue
                    first, last = active
                    segment = np.stack((origin+first*direction, origin+last*direction))
                    if signed_distance is not None:
                        # Match the public true-field line solver's exact
                        # represented equation, rather than an epsilon test.
                        solved = int(np.argmax(abs(normal)))
                        free = 1-solved
                        segment[:, solved] = (offset-normal[free]*segment[:, free])/normal[solved]
                    events.append({"owner": owner, "normal": normal.copy(), "offset": offset,
                                   "sourceRiverIndices": pair, "segment": segment,
                                   "canonicalLineKey": key, "longitudePeriod": period,
                                   "canonicalOffset": canonical_offset})
    return events


def interior_medial_is_nearest(field, event, point):
    """Check an isolated candidate station against the unchanged source tree.

    The caller constructs the point from the recorded line equation while
    solving the actual terrain height. Analytic perpendicular distances share
    that equation; GEOS may select either nearest identity at a floating root.
    A third nearer source, a source endpoint, or an off-line point rejects it.
    """
    point = np.asarray(point, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("river-medial stations require a finite physical point")
    normal = event["normal"]
    solved = int(np.argmax(abs(normal)))
    free = 1-solved
    if point[solved] != (event["offset"]-normal[free]*point[free])/normal[solved]:
        return False
    first, last = sorted(event["segment"][:, free])
    if not first <= point[free] <= last:
        return False
    pair = event["sourceRiverIndices"]
    for identifier in pair:
        points = np.asarray(field._rivers.geometries[identifier].coords)
        delta = points[1]-points[0]
        projection = float((point-points[0])@delta)/float(delta@delta)
        if not 0 < projection < 1:
            return False
    nearest = field._rivers.query_nearest(shapely.Point(point), all_matches=True)
    if not any(int(identifier) in pair for identifier in nearest):
        return False
    distance = float(shapely.distance(field._rivers.geometries[int(nearest[0])], shapely.Point(point)))
    return .18 < distance < .5
