"""Resolve continuous river courses and their shared, descending bed.

The D8 graph owns reach connectivity. Its centres are drainage observations,
not mandatory drawing vertices or a second elevation datum. The continuous
base terrain owns transverse valley minima; one bed cap repairs depressions
without changing the saved graph or introducing decorative meanders.
"""
from dataclasses import dataclass
import heapq
import math

import numpy as np
import shapely
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components
from scipy.optimize import elementwise

from .river_network import hydrologic_outlet_targets
from .cartographic_curves import _forward_corridor_offsets, _source_stations


@dataclass(frozen=True)
class RiverCourses:
    source_paths: tuple
    paths: tuple
    beds: tuple
    diagnostics: dict


def native_river_paths(grid, base):
    """Extract logical reaches in the renderer's canonical source ordering."""
    active = np.asarray(grid.river_order).ravel() > 0
    flow = np.asarray(grid.flow_to).ravel()
    outlets = hydrologic_outlet_targets(grid, base.native_m)
    incoming = np.zeros(active.size, dtype=np.int32)
    ids = np.flatnonzero(active & (flow >= 0))
    ids = ids[active[flow[ids]]]
    np.add.at(incoming, flow[ids], 1)
    visited = set()
    paths = []
    width = grid.shape[1]

    def point(node):
        return np.array((node % width+.5, node // width+.5))

    def follow(start):
        current = int(start)
        points = []
        while 0 <= current < active.size and active[current]:
            first = point(current)
            points.append(first)
            target = outlets.get(current, int(flow[current]))
            if target < 0 or target >= active.size or not active[target]:
                if (0 <= target < active.size and grid.water.ravel()[target] != 0):
                    last = point(target)
                    last[0] = first[0]+((last[0]-first[0]+width/2) % width-width/2)
                    low, high = 0., 1.
                    for _ in range(40):
                        middle = (low+high)/2
                        p = first+(last-first)*middle
                        if base.sample_points(p[0], p[1]) > 0:
                            low = middle
                        else:
                            high = middle
                    points.append(first+(last-first)*((low+high)/2))
                break
            edge = (current, target)
            if edge in visited:
                break
            visited.add(edge)
            if incoming[target] != 1:
                points.append(point(target))
                break
            current = target
        if len(points) >= 2:
            path = np.asarray(points)
            # The source graph normally avoids the longitude cut. A reach
            # crossing it is two pieces of the same periodic physical course.
            jumps = np.flatnonzero(np.abs(np.diff(path[:, 0])) > width/2)
            start = 0
            for jump in jumps:
                first, last = path[jump].copy(), path[jump+1].copy()
                last[0] = first[0]+((last[0]-first[0]+width/2) % width-width/2)
                edge_x = width if last[0] > width else 0.
                t = (edge_x-first[0])/(last[0]-first[0])
                seam = first+(last-first)*t
                paths.append(np.vstack((path[start:jump+1], seam)))
                seam = seam.copy(); seam[0] = width-edge_x
                path[jump] = seam
                start = jump
            paths.append(path[start:].copy())

    for start in np.flatnonzero(active & (incoming != 1)):
        follow(start)
    for source in np.flatnonzero(active):
        target = int(flow[source])
        if 0 <= target < active.size and active[target] and (int(source), target) not in visited:
            follow(source)
    return tuple(paths)


def _station_limits(stations, radius=1.25):
    """Keep foreign reaches apart without treating dense self-chords as banks."""
    groups = [segments for _, _, segments in stations if len(segments)]
    if not groups:
        return [np.zeros(len(points)) for points, _, _ in stations]
    edges = shapely.linestrings(np.concatenate(groups))
    owners = np.concatenate([np.full(len(segments), owner, dtype=np.int32)
        for owner, (_, _, segments) in enumerate(stations) if len(segments)])
    ordinals = np.concatenate([np.arange(len(segments), dtype=np.int32)
        for _, _, segments in stations if len(segments)])
    tree = shapely.STRtree(edges)
    result = []
    for owner, (points, identifiers, _) in enumerate(stations):
        chords = shapely.linestrings(np.stack((points[:-1], points[1:]), axis=1))
        clearance = np.full(len(chords), np.inf)
        first, second = tree.query(chords, predicate="dwithin", distance=2*radius)
        foreign = ((owners[second] != owner)
                   | (np.abs(identifiers[first]-ordinals[second]) > 2))
        first, second = first[foreign], second[foreign]
        if len(first):
            np.minimum.at(clearance, first, shapely.distance(chords[first], edges[second]))
        limits = np.zeros(len(points))
        radii = np.minimum(radius, .49*clearance)
        limits[1:-1] = np.minimum(radii[:-1], radii[1:])
        result.append(limits)
    return result


def _locked_stations(points, locked_points):
    """Insert existing bridge portals exactly, retaining their native station."""
    line = shapely.LineString(points)
    locks = np.asarray(locked_points, dtype=float).reshape(-1, 2)
    if not len(locks):
        return points, np.zeros(len(points), dtype=bool)
    distance = shapely.distance(shapely.points(locks), line)
    locks = locks[distance <= 1e-7]
    if not len(locks):
        return points, np.zeros(len(points), dtype=bool)
    stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    along = shapely.line_locate_point(line, shapely.points(locks))
    positions = np.r_[stations, along]
    values = np.vstack((points, locks))
    order = np.argsort(positions, kind="stable")
    positions, values = positions[order], values[order]
    keep = np.r_[True, np.diff(positions) > 1e-9]
    positions, values = positions[keep], values[keep]
    fixed = np.any(np.linalg.norm(values[:, None]-locks[None], axis=2) <= 1e-7, axis=1)
    return values, fixed


def _smooth_course(target, fixed):
    if len(target) < 3:
        return target.copy()
    second = sparse.diags((np.ones(len(target)-2), -2*np.ones(len(target)-2),
                           np.ones(len(target)-2)), (0, 1, 2),
                          shape=(len(target)-2, len(target)), format="csc")
    matrix = sparse.eye(len(target), format="csc")+4*(second.T @ second)
    free = ~fixed
    result = target.copy()
    if np.any(free):
        rhs = target[free]-matrix[free][:, fixed] @ target[fixed]
        result[free] = spsolve(matrix[free][:, free], rhs)
    return result


def _valley_course(grid, base, points, limits, fixed):
    if len(points) < 3:
        return points.copy()
    tangent = points[2:]-points[:-2]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.zeros_like(points)
    normal[1:-1] = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    fractions = np.linspace(-1., 1., 17)
    sections = points[None]+fractions[:, None, None]*limits[None, :, None]*normal[None]
    height, width = grid.shape
    valid = ((sections[:, :, 0] >= 0) & (sections[:, :, 0] <= width)
             & (sections[:, :, 1] >= 0) & (sections[:, :, 1] <= height))
    ground = np.full(valid.shape, np.inf)
    ground[valid] = base.sample_points(sections[:, :, 0][valid], sections[:, :, 1][valid])
    ground[ground <= 0] = np.inf
    minimum = np.argmin(ground, axis=0)
    offsets = np.zeros(len(points))
    eligible = np.flatnonzero((minimum > 0) & (minimum < 16) & (limits > 0) & ~fixed)
    if len(eligible):
        index = minimum[eligible]
        left, middle, right = ground[index-1, eligible], ground[index, eligible], ground[index+1, eligible]
        curvature = left+right-2*middle
        supported = np.isfinite(left+middle+right) & (curvature > 1e-9)
        selected, index = eligible[supported], index[supported]
        delta = np.clip((left[supported]-right[supported])/(2*curvature[supported]), -.5, .5)
        offsets[selected] = limits[selected]*(fractions[index]+delta/8)
    # A convex valley may continue beyond this reach's legal corridor. Its
    # lower edge is supported, while a planar cross-slope provides no valley
    # curvature and retains the logical position.
    boundary = np.flatnonzero(((minimum == 0) | (minimum == 16))
                             & (limits > 0) & ~fixed)
    if len(boundary):
        center_curvature = ground[0, boundary]+ground[16, boundary]-2*ground[8, boundary]
        supported = np.isfinite(center_curvature) & (center_curvature > 1e-9)
        selected = boundary[supported]
        offsets[selected] = limits[selected]*fractions[minimum[selected]]
    wet = np.flatnonzero(~np.isfinite(ground[8]) & (limits > 0) & ~fixed)
    if len(wet):
        # Native land ownership does not make the entire cell dry. Route a
        # channel around a real subcell cove rather than carving a river into
        # negative ocean terrain or truncating it before its logical mouth.
        choices = np.where(np.isfinite(ground[:, wet]),
                           np.abs(fractions[:, None]), np.inf)
        nearest = np.argmin(choices, axis=0)
        supported = np.isfinite(choices[nearest, np.arange(len(wet))])
        selected = wet[supported]
        offsets[selected] = limits[selected]*fractions[nearest[supported]]
    offsets = _forward_corridor_offsets(points, normal, offsets)
    target = points+normal*offsets[:, None]
    proposed = _smooth_course(target, fixed)
    proposed[fixed] = points[fixed]
    # A curve must still lie in the accepted river corridor and on dry base
    # ground. Reject movement locally by reducing its shared displacement.
    displacement = proposed-points
    fraction = 1.
    for _ in range(12):
        candidate = points+fraction*displacement
        inside = ((candidate[:, 0] >= 0) & (candidate[:, 0] <= width)
                  & (candidate[:, 1] >= 0) & (candidate[:, 1] <= height))
        if inside.all():
            values = base.sample_points(candidate[:, 0], candidate[:, 1])
            dry = (values > 0) | fixed
            clearance = np.linalg.norm(candidate-points, axis=1) <= limits+1e-10
            if dry.all() and clearance.all() and shapely.LineString(candidate).is_simple:
                return candidate
        fraction *= .5
    # The unsmoothed valley solve is useful when a shoreline clearance is
    # tighter than the curvature solve. Its actual ground must still be dry.
    if (np.all((base.sample_points(target[:, 0], target[:, 1]) > 0) | fixed)
            and shapely.LineString(target).is_simple):
        return target
    return points.copy()


def _node_key(point, width):
    return (round(float(point[0] % width), 10), round(float(point[1]), 10))


def _dry_course_spans(base, points, reference, limits):
    """Route short shore approaches around actual subcell water pockets."""
    for _ in range(5):
        fractions = np.array((.25, .5, .75))
        delta = np.diff(points, axis=0)
        probes = points[:-1, None]+delta[:, None]*fractions[None, :, None]
        heights = base.sample_points(probes[:, :, 0], probes[:, :, 1])
        bad = np.flatnonzero(np.any(heights <= 0, axis=1))
        if not len(bad):
            return points
        inserts = {}
        for index in bad:
            fraction = fractions[int(np.argmin(heights[index]))]
            origin = points[index]+delta[index]*fraction
            source = reference[index]+(reference[index+1]-reference[index])*fraction
            radius = limits[index]+(limits[index+1]-limits[index])*fraction
            tangent = delta[index]/max(np.linalg.norm(delta[index]), 1e-15)
            normal = np.array((-tangent[1], tangent[0]))
            offset = origin-source
            along = float(offset @ normal)
            transverse = float(offset @ offset)-along*along
            extent = math.sqrt(max(radius*radius-transverse, 0.))
            shifts = np.linspace(-along-extent, -along+extent, 49)
            candidates = origin+shifts[:, None]*normal
            inside = ((candidates[:, 0] >= 0) & (candidates[:, 0] <= base.width)
                      & (candidates[:, 1] >= 0) & (candidates[:, 1] <= base.height))
            height = np.full(len(candidates), -np.inf)
            height[inside] = base.sample_points(candidates[inside, 0], candidates[inside, 1])
            selected = np.flatnonzero(height > 0)
            if not len(selected):
                raise ValueError("the logical river corridor has no dry subcell shore approach")
            selected = selected[np.argsort(np.abs(shifts[selected]), kind="stable")]
            leg_t = np.linspace(0., 1., 17)[1:-1]
            last = candidates[selected, None]+(points[index+1]-candidates[selected])[:, None]*leg_t[:, None]
            first = points[index]+(candidates[selected]-points[index])[:, None, :]*leg_t[None, :, None]
            supports = np.minimum(base.sample_points(first[:, :, 0], first[:, :, 1]).min(axis=1),
                                  base.sample_points(last[:, :, 0], last[:, :, 1]).min(axis=1))
            accepted = np.flatnonzero(supports > 0)
            chosen = selected[int(accepted[0])] if len(accepted) else selected[int(np.argmax(supports))]
            inserts[int(index)] = (candidates[chosen], source, radius)
        output, source_output, radius_output = [], [], []
        for index, point in enumerate(points):
            output.append(point); source_output.append(reference[index]); radius_output.append(limits[index])
            if index in inserts:
                candidate, source, radius = inserts[index]
                output.append(candidate); source_output.append(source); radius_output.append(radius)
        points, reference, limits = np.asarray(output), np.asarray(source_output), np.asarray(radius_output)
    raise ValueError("a continuous river shore approach must remain on positive ground")


def _shared_course_stations(paths, width):
    """Tie old geometric X intersections without rewriting the logical flow."""
    if not paths:
        return paths, 0
    lines = shapely.linestrings(paths) if len({len(path) for path in paths}) == 1 else np.array(
        [shapely.LineString(path) for path in paths], dtype=object)
    first, second = shapely.STRtree(lines).query(lines, predicate="intersects")
    unique = first < second
    first, second = first[unique], second[unique]
    locks = [[] for _ in paths]
    internal = set()
    for a, b, crossing in zip(first, second, shapely.intersection(lines[first], lines[second]), strict=True):
        for point in shapely.get_coordinates(crossing):
            locks[a].append(point); locks[b].append(point)
            key = _node_key(point, width)
            if (key not in {_node_key(paths[a][0], width), _node_key(paths[a][-1], width)}
                    or key not in {_node_key(paths[b][0], width), _node_key(paths[b][-1], width)}):
                internal.add(key)
    return [_locked_stations(path, points)[0] for path, points in zip(paths, locks, strict=True)], len(internal)


def _descending_beds(paths, base):
    """Greatest nonincreasing cap, including existing coincident river beds.

    Equal XY stations share a height constraint. An old geometric crossing
    can create a cycle in these constraints even when the D8 flow is acyclic;
    a strongly connected height component then has one level, rather than
    manufacturing an uphill leg. These are bed constraints, not new logical
    drainage edges or a claim to have reconstructed confluence topology.
    """
    raw = [np.maximum(base.sample_points(path[:, 0], path[:, 1]), 0.) for path in paths]
    nodes, identifiers = {}, []
    for path in paths:
        identifiers.append(np.array([nodes.setdefault(_node_key(point, base.width), len(nodes))
                                     for point in path], dtype=np.int32))
    if not nodes:
        return (), raw, 0
    source = np.concatenate([ids[:-1] for ids in identifiers])
    target = np.concatenate([ids[1:] for ids in identifiers])
    graph = sparse.csr_matrix((np.ones(len(source), dtype=np.uint8), (source, target)),
                              shape=(len(nodes), len(nodes)))
    count, labels = connected_components(graph, directed=True, connection="strong")
    levels = np.full(count, np.inf)
    for path, ids, heights in zip(paths, identifiers, raw, strict=True):
        # A sampled endpoint cap alone misses a PCHIP trough inside a chord.
        # Minimize the actual one-dimensional source ground on each short
        # course span, and place both incident bed caps below its floor.
        t = np.linspace(0., 1., 5)
        first, delta = path[:-1], np.diff(path, axis=0)
        probes = first[:, None]+delta[:, None]*t[None, :, None]
        ground = base.sample_points(probes[:, :, 0], probes[:, :, 1])
        minimum = np.argmin(ground, axis=1)
        floor = ground[np.arange(len(ground)), minimum].copy()
        selected = np.flatnonzero((minimum > 0) & (minimum < 4))
        if len(selected):
            middle = minimum[selected]

            def along(value, xx, yy, dx, dy):
                return base.sample_points(xx+value*dx, yy+value*dy)

            result = elementwise.find_minimum(along,
                (t[middle-1], t[middle], t[middle+1]),
                args=(first[selected, 0], first[selected, 1],
                      delta[selected, 0], delta[selected, 1]),
                tolerances={"xatol":1e-11, "fatol":1e-8}, maxiter=80)
            floor[selected] = np.minimum(floor[selected], result.f_x)
        # A true zero-datum mouth has a positive approach and a zero endpoint;
        # its terminal span must never be flattened onto the ocean datum.
        terminal = heights[1:] <= 1e-7
        floor[terminal] = heights[:-1][terminal]
        caps = heights.copy()
        positive = floor > 0
        caps[:-1][positive] = np.minimum(caps[:-1][positive], floor[positive])
        caps[1:][positive] = np.minimum(caps[1:][positive], floor[positive])
        np.minimum.at(levels, labels[ids], caps)
    pairs = np.column_stack((labels[source], labels[target]))
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    outgoing = sparse.csr_matrix((np.ones(len(pairs), dtype=np.uint8),
                                 (pairs[:, 0], pairs[:, 1])), shape=(count, count))
    indegree = np.bincount(pairs[:, 1], minlength=count)
    pending = np.flatnonzero(indegree == 0).tolist(); heapq.heapify(pending)
    while pending:
        node = heapq.heappop(pending)
        for following in outgoing.indices[outgoing.indptr[node]:outgoing.indptr[node+1]]:
            levels[following] = min(levels[following], levels[node])
            indegree[following] -= 1
            if indegree[following] == 0:
                heapq.heappush(pending, int(following))
    cycles = int(np.count_nonzero(np.bincount(labels, minlength=count) > 1))
    return tuple(levels[labels[ids]] for ids in identifiers), raw, cycles


def solve_river_courses(grid, base, *, source_paths=None, locked_points=()):
    """Solve XY and a graph-wide bed once, before terrain and transport."""
    sources = native_river_paths(grid, base) if source_paths is None else tuple(source_paths)
    prepared = [_locked_stations(path, locked_points)[0] for path in sources]
    stations = [_source_stations(grid, path) for path in prepared]
    limits = _station_limits(stations)
    paths = []
    locks = np.asarray(locked_points, dtype=float).reshape(-1, 2)
    for (points, _, _), allowed in zip(stations, limits, strict=True):
        fixed = np.zeros(len(points), dtype=bool); fixed[[0, -1]] = True
        fixed |= allowed <= 1e-12
        if len(locks):
            fixed |= np.any(np.linalg.norm(points[:, None]-locks[None], axis=2) <= 1e-7, axis=1)
        allowed[fixed] = 0.
        course = _valley_course(grid, base, points, allowed, fixed)
        paths.append(_dry_course_spans(base, course, points, allowed))
    paths, geometric_crossings = _shared_course_stations(paths, base.width)
    beds, raw, height_cycles = _descending_beds(paths, base)
    incision = [heights-bed for heights, bed in zip(raw, beds, strict=True)]
    return RiverCourses(tuple(np.asarray(path).copy() for path in sources), tuple(paths), beds,
        {"schema": "continuous-river-course-bed-v1", "reaches": len(paths),
         "courseModel": "continuous-base-valley-minima-with-shared-corridor",
         "bedModel": "greatest-nonincreasing-positive-ground-cap",
         "maximumIncisionMeters": max((float(value.max()) for value in incision), default=0.),
         "uphillSourceSteps": sum(int(np.count_nonzero(np.diff(value) > 1e-6)) for value in raw),
         "uphillBedSteps": sum(int(np.count_nonzero(np.diff(value) > 1e-6)) for value in beds),
         "sourceGeometricCrossings": geometric_crossings,
         "coincidentBedConstraintCycles": height_cycles,
         "lockedPortals": len(locks), "nativeArraysChanged": False})


class RiverBedRelief:
    """Compact valley sections sharing the exact resolved course and bed.

    Per-reach nearest stations avoid ownership changes between adjacent
    chords. Inverse-distance blending agrees exactly with every river axis
    and joins shared bed heights continuously at confluences. Outside the
    compact support, the previous terrain is unchanged; its shore sign is
    retained everywhere except the already fixed zero-datum river mouth.
    """
    def __init__(self, courses, grid, *, radius_km):
        self.width = grid.shape[1]
        pieces, owners, elevations, radii = [], [], [], []
        latitude_step = math.pi/grid.shape[0]
        y_km = latitude_step*radius_km
        x_km = 2*math.pi*radius_km/grid.shape[1]
        for owner, (path, bed) in enumerate(zip(courses.paths, courses.beds, strict=True)):
            row = np.clip(np.floor(path[:, 1]).astype(int), 0, grid.shape[0]-1)
            column = np.floor(path[:, 0]).astype(int) % self.width
            support = np.asarray(grid.discharge)[row, column]
            breadth_km = np.minimum(8., 2.5+1.15*np.log2(np.maximum(support, 1.)))
            east_km = x_km*np.maximum(np.cos(math.pi/2-path[:, 1]*latitude_step), .08)
            radius = np.minimum(.35, breadth_km/np.minimum(east_km, y_km))
            pieces.append(np.stack((path[:-1], path[1:]), axis=1))
            owners.append(np.full(len(path)-1, owner, dtype=np.int32))
            elevations.append(np.column_stack((bed[:-1], bed[1:])))
            radii.append(np.column_stack((radius[:-1], radius[1:])))
        segments = np.concatenate(pieces) if pieces else np.empty((0, 2, 2))
        self.segments = np.concatenate((segments-[self.width, 0], segments, segments+[self.width, 0]))
        self.owners = np.tile(np.concatenate(owners) if owners else np.empty(0, dtype=np.int32), 3)
        self.beds = np.tile(np.concatenate(elevations) if elevations else np.empty((0, 2)), (3, 1))
        self.radii = np.tile(np.concatenate(radii) if radii else np.empty((0, 2)), (3, 1))
        self._tree = shapely.STRtree(shapely.linestrings(self.segments))
        self.maximum_radius = float(self.radii.max(initial=0.))

    def sample(self, x, y, ground):
        x, y, ground = np.broadcast_arrays(x, y, ground)
        result = np.asarray(ground).ravel().copy()
        if not len(self.segments):
            return result.reshape(x.shape)
        xx, yy = np.asarray(x).ravel(), np.asarray(y).ravel()
        pairs = self._tree.query(shapely.box(xx-self.maximum_radius, yy-self.maximum_radius,
                                           xx+self.maximum_radius, yy+self.maximum_radius))
        if not pairs.shape[1]:
            return result.reshape(x.shape)
        point, edge = pairs
        start = self.segments[edge, 0]
        delta = self.segments[edge, 1]-start
        relative = np.column_stack((xx[point], yy[point]))-start
        fraction = np.clip(np.einsum("ij,ij->i", relative, delta)
                           /np.maximum(np.einsum("ij,ij->i", delta, delta), 1e-24), 0., 1.)
        offset = relative-delta*fraction[:, None]
        distance2 = np.einsum("ij,ij->i", offset, offset)
        order = np.lexsort((distance2, self.owners[edge], point))
        point, edge, fraction, distance2 = point[order], edge[order], fraction[order], distance2[order]
        keep = np.r_[True, (np.diff(point) != 0) | (np.diff(self.owners[edge]) != 0)]
        point, edge, fraction, distance2 = point[keep], edge[keep], fraction[keep], distance2[keep]
        radius = self.radii[edge, 0]+fraction*np.diff(self.radii[edge], axis=1).ravel()
        bed = self.beds[edge, 0]+fraction*np.diff(self.beds[edge], axis=1).ravel()
        t = np.sqrt(distance2)/radius
        inside = (t < 1.) & (result[point] > 0.)
        point, distance2, bed, t = point[inside], distance2[inside], bed[inside], t[inside]
        kernel = (1-t)**4*(1+4*t)
        axis = distance2 <= 1e-24
        weights = np.divide(kernel, distance2, out=np.zeros_like(kernel), where=~axis)
        depth = np.maximum(result[point]-bed, 0.)
        total, weighted = np.zeros(x.size), np.zeros(x.size)
        outside = np.ones(x.size)
        np.add.at(total, point, weights)
        np.add.at(weighted, point, weights*depth)
        np.multiply.at(outside, point, 1-kernel)
        change = np.divide(weighted, total, out=np.zeros_like(weighted), where=total > 0)
        result -= (1-outside)*change
        # A shared junction has one bed value; duplicate incident reaches
        # therefore agree at its exact coordinate rather than adding depths.
        axis_bed = np.full(x.size, np.inf)
        np.minimum.at(axis_bed, point[axis], bed[axis])
        on_axis = np.isfinite(axis_bed)
        result[on_axis] = np.minimum(np.asarray(ground).ravel()[on_axis], axis_bed[on_axis])
        return result.reshape(x.shape)
