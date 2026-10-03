"""Navigate an accepted polygon without changing its boundary or anchors.

GEOS constrained triangulation preserves holes and bank boundaries. Sparse
triangle adjacency selects a corridor; the portal funnel finds its straight
path. The graph chooses a corridor, not a globally shortest homotopy.
"""

from fractions import Fraction
from functools import cached_property

import numpy as np
import shapely
from scipy.sparse import csr_array
from scipy.sparse.csgraph import shortest_path


def _turn(first, second, third):
    """Orientation of the input doubles, including near-collinear portals."""
    x1, y1 = second[0] - first[0], second[1] - first[1]
    x2, y2 = third[0] - first[0], third[1] - first[1]
    positive, negative = x1 * y2, y1 * x2
    determinant = positive - negative
    if abs(determinant) > 4 * np.finfo(float).eps * (abs(positive) + abs(negative)):
        return determinant
    if positive == negative == 0:
        return 0.
    # Exact arithmetic is only needed at numerically ambiguous turns. In
    # particular, a tolerance must not erase a genuinely narrow bank sector.
    x1 = Fraction(second[0]) - Fraction(first[0])
    y1 = Fraction(second[1]) - Fraction(first[1])
    x2 = Fraction(third[0]) - Fraction(first[0])
    y2 = Fraction(third[1]) - Fraction(first[1])
    determinant = x1 * y2 - y1 * x2
    return (determinant > 0) - (determinant < 0)


def _funnel(portals):
    """Pull a string through ordered (left, right) triangle portals."""
    apex = left = right = portals[0][0]
    apex_index = left_index = right_index = 0
    result = [apex]
    index = 1
    while index < len(portals):
        next_left, next_right = portals[index]
        if _turn(apex, right, next_right) >= 0:
            if apex == right or _turn(apex, left, next_right) < 0:
                right, right_index = next_right, index
            else:
                if result[-1] != left:
                    result.append(left)
                apex, apex_index = left, left_index
                left = right = apex
                left_index = right_index = apex_index
                index = apex_index + 1
                continue
        if _turn(apex, left, next_left) <= 0:
            if apex == left or _turn(apex, right, next_left) > 0:
                left, left_index = next_left, index
            else:
                if result[-1] != right:
                    result.append(right)
                apex, apex_index = right, right_index
                left = right = apex
                left_index = right_index = apex_index
                index = apex_index + 1
                continue
        index += 1
    if result[-1] != portals[-1][0]:
        result.append(portals[-1][0])
    return np.asarray(result, dtype=float)


class PolygonNavigator:
    """Share one accepted polygon's triangulation between its bank passages."""

    def __init__(self, polygon):
        if polygon.geom_type != "Polygon" or polygon.is_empty or not polygon.is_valid:
            raise ValueError("polygon navigation requires one nonempty valid Polygon")
        self.polygon = polygon
        shapely.prepare(polygon)

    @cached_property
    def mesh(self):
        triangles = shapely.get_parts(shapely.constrained_delaunay_triangles(self.polygon))
        shapely.prepare(triangles)
        coordinates = [tuple(map(tuple, np.asarray(triangle.exterior.coords)[:3]))
                       for triangle in triangles]
        centres = np.asarray([np.mean(points, axis=0) for points in coordinates])
        starts, ends, costs = [], [], []
        edges, portals = {}, {}
        for index, points in enumerate(coordinates):
            for first_vertex, last_vertex in zip(points, (*points[1:], points[0]), strict=True):
                edge = tuple(sorted((first_vertex, last_vertex)))
                if edge not in edges:
                    edges[edge] = index
                    continue
                neighbour = edges.pop(edge)
                starts.append(index)
                ends.append(neighbour)
                costs.append(float(np.linalg.norm(centres[index] - centres[neighbour])))
                portals[min(index, neighbour), max(index, neighbour)] = edge
        return triangles, centres, starts, ends, costs, portals

    def path(self, first, last):
        """Navigate exact anchors without rebuilding this polygon's mesh."""
        anchors = np.asarray((first, last), dtype=float)
        if anchors.shape != (2, 2) or not np.isfinite(anchors).all():
            raise ValueError("polygon path anchors must be finite two-dimensional points")
        polygon = self.polygon
        if not polygon.covers(shapely.MultiPoint(anchors)):
            raise ValueError("polygon path anchors must belong to the accepted polygon")
        if polygon.covers(shapely.LineString(anchors)):
            return anchors.copy()

        # Reverse traversal must use the same equal-cost corridor.
        reverse = tuple(anchors[1]) < tuple(anchors[0])
        if reverse:
            anchors = anchors[::-1].copy()

        triangles, centres, base_starts, base_ends, base_costs, portals = self.mesh
        starts, ends, costs = base_starts.copy(), base_ends.copy(), base_costs.copy()
        count = len(triangles)
        for virtual, anchor in enumerate(anchors, start=count):
            matches = np.flatnonzero(shapely.covers(triangles, shapely.Point(anchor)))
            if not len(matches):
                raise ValueError("an accepted anchor has no constrained navigation triangle")
            for index in matches:
                starts.append(virtual)
                ends.append(int(index))
                costs.append(float(np.linalg.norm(anchor - centres[index])))
        graph = csr_array((costs, (starts, ends)), shape=(count + 2, count + 2))
        distance, previous = shortest_path(graph, directed=False, indices=count,
                                           return_predecessors=True, method="D")
        if not np.isfinite(distance[count + 1]):
            raise ValueError("an accepted polygon has no connected triangle corridor")
        corridor = [count + 1]
        while corridor[-1] != count:
            corridor.append(int(previous[corridor[-1]]))
        corridor = corridor[-2:0:-1]
        first, last = map(tuple, anchors)
        ordered = [(first, first)]
        for before, after in zip(corridor[:-1], corridor[1:]):
            left, right = portals[min(before, after), max(before, after)]
            if _turn(tuple(centres[before]), tuple(centres[after]), left) < 0:
                left, right = right, left
            ordered.append((left, right))
        ordered.append((last, last))
        path = _funnel(ordered)
        if not polygon.covers(shapely.LineString(path)):
            raise ValueError("a triangle funnel path left its accepted polygon")
        return path[::-1].copy() if reverse else path


def polygon_path(polygon, first, last):
    """Return a bank path between exact anchors in one valid Polygon.

    Boundary anchors and hole boundaries are navigable. Outside anchors,
    invalid polygons and disconnected triangle corridors raise ValueError.
    No buffers, projection, resampling or water-boundary relaxation occur.
    """
    return PolygonNavigator(polygon).path(first, last)
