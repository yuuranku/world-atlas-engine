"""Grade-constrained construction on continuous ground, before SVG encoding.

The native regional router selects valleys and passes. This local solver can
traverse a hillside at shallow headings and reverse direction in a confined
corridor. Turns result from rise/run and available land, never injected noise.
"""
import heapq
import math

import numpy as np
import shapely
from scipy.ndimage import map_coordinates

from .road_engineering import segment_lengths_km


_HEADINGS = tuple((dy, dx) for dy in range(-4, 5) for dx in range(-4, 5)
                  if (dy or dx) and math.gcd(abs(dy), abs(dx)) == 1)


def _route(first, last, domain, engineering, terrain_field, *, margin, resolution):
    shape = engineering.elevation_m.shape
    west, north = np.minimum(first, last)-margin
    east, south = np.maximum(first, last)+margin
    west, east = max(0., west), min(float(shape[1]), east)
    north, south = max(0., north), min(float(shape[0]), south)
    xs = np.unique(np.r_[np.arange(west, east, 1/resolution), east, first[0], last[0]])
    ys = np.unique(np.r_[np.arange(north, south, 1/resolution), south, first[1], last[1]])
    xx, yy = np.meshgrid(xs, ys)
    z = terrain_field.sample_points(xx.ravel(), yy.ravel()).reshape(xx.shape)
    allowed = shapely.intersects_xy(domain, xx, yy) & (z > 0)
    start = (int(np.searchsorted(ys, first[1])), int(np.searchsorted(xs, first[0])))
    goal = (int(np.searchsorted(ys, last[1])), int(np.searchsorted(xs, last[0])))
    allowed[start] = allowed[goal] = True
    costs = np.full((*z.shape, len(_HEADINGS)), np.inf, dtype=np.float32)
    row, col = np.indices(z.shape)
    # Intermediate heights prevent a long heading from hopping over a ridge.
    for k, (dy, dx) in enumerate(_HEADINGS):
        rr, cc = row+dy, col+dx
        valid = (rr >= 0) & (rr < len(ys)) & (cc >= 0) & (cc < len(xs))
        tr, tc = np.clip(rr, 0, len(ys)-1), np.clip(cc, 0, len(xs)-1)
        valid &= allowed & allowed[tr, tc]
        a = np.column_stack((xx.ravel(), yy.ravel()))
        b = np.column_stack((xs[tc.ravel()], ys[tr.ravel()]))
        # Spherical distances vary with latitude, even in the local solver.
        lat_a = math.pi/2-a[:, 1]*math.pi/shape[0]
        lat_b = math.pi/2-b[:, 1]*math.pi/shape[0]
        hav = np.sin((lat_b-lat_a)/2)**2 + np.cos(lat_a)*np.cos(lat_b)*np.sin((b[:, 0]-a[:, 0])*math.pi/shape[1])**2
        distance = (2*engineering.radius_km*np.arcsin(np.sqrt(np.clip(hav, 0, 1)))).reshape(z.shape)
        previous = z
        peak = np.zeros(z.shape)
        integral = np.zeros(z.shape)
        for t in (.25, .5, .75, 1.):
            height = map_coordinates(z, [row+dy*t, col+dx*t], order=1, mode='nearest')
            grade = np.abs(height-previous)/np.maximum(distance*250,1e-12)
            peak = np.maximum(peak, grade)
            integral += (grade/engineering.preferred_grade)**2/4
            valid &= map_coordinates(allowed.astype(float), [row+dy*t, col+dx*t], order=0, mode='nearest') > .5
            previous = height
        # A small margin leaves room for continuous-field verification.
        valid &= peak <= engineering.maximum_grade*.97
        costs[..., k] = np.where(valid, distance*(1+4*integral), np.inf)
    distances = {start: 0.}
    previous = {}
    queue = [(0., 0., start)]
    closed = set()
    def heuristic(node):
        return float(segment_lengths_km([(xs[node[1]], ys[node[0]]), last], shape, engineering.radius_km)[0])
    while queue:
        _, cost, node = heapq.heappop(queue)
        if node in closed:
            continue
        if node == goal:
            chain = [node]
            while chain[-1] != start:
                chain.append(previous[chain[-1]])
            return np.asarray([(xs[c], ys[r]) for r, c in reversed(chain)])
        closed.add(node)
        for k, (dy, dx) in enumerate(_HEADINGS):
            edge = float(costs[node[0], node[1], k])
            if not math.isfinite(edge):
                continue
            target = (node[0]+dy, node[1]+dx)
            proposed = cost+edge
            if proposed >= distances.get(target, math.inf):
                continue
            line = shapely.LineString(((xs[node[1]], ys[node[0]]), (xs[target[1]], ys[target[0]])))
            if not domain.covers(line):
                continue
            distances[target] = proposed
            previous[target] = node
            heapq.heappush(queue, (proposed+heuristic(target), proposed, target))
    return None


def engineer_mountain_path(points, engineering, *, terrain_field, road_surface):
    """Replace excessive grades with real detours, retaining junction anchors.

    Adaptive refinement is checked against the continuous physical field. A
    cliff with no constructible route is an error, not a straight road painted
    through it. Water and existing licensed crossing spans bound every search.
    """
    samples, _, _, grades = engineering.profile(points, terrain_field=terrain_field)
    bad = np.flatnonzero(np.abs(grades) > engineering.maximum_grade)
    if not len(bad):
        return np.asarray(points)
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))]
    intervals = []
    for index in bad:
        left = max(0, int(np.searchsorted(cumulative, cumulative[index]-.35))-1)
        right = min(len(samples)-1, int(np.searchsorted(cumulative, cumulative[index+1]+.35)))
        if intervals and left <= intervals[-1][1]:
            intervals[-1][1] = max(right, intervals[-1][1])
        else:
            intervals.append([left, right])
    result = []
    cursor = 0
    for left, right in intervals:
        result.extend(samples[cursor:left])
        replacement = None
        for margin, resolution in ((.5, 32), (1., 32), (2., 32), (2., 64)):
            first, last = samples[left], samples[right]
            box = shapely.box(*np.r_[np.minimum(first,last)-margin, np.maximum(first,last)+margin])
            domain = road_surface.intersection(box)
            shapely.prepare(domain)
            proposal = _route(first, last, domain, engineering, terrain_field,
                              margin=margin, resolution=resolution)
            if proposal is not None:
                _, _, _, verified = engineering.profile(proposal, terrain_field=terrain_field)
                if np.max(np.abs(verified), initial=0) <= engineering.maximum_grade:
                    replacement = proposal
                    break
        if replacement is None:
            raise ValueError(f'no grade-constrained mountain road between {samples[left]} and {samples[right]}')
        result.extend(replacement[:-1])
        cursor = right
    result.extend(samples[cursor:])
    return np.asarray(result)
