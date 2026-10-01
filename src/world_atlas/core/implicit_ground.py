"""Trace physical zero through its unchanged ground coordinate model.

The base tensor PCHIP owns branches and shared native-edge roots. A positive
height gain leaves those branches unchanged; the refined homeomorphism
transports them. Curvature is measured after transport, before one periodic
arrangement is cut to the actual map frame.
"""

import numpy as np
import shapely
from scipy.optimize import elementwise

from .continuous_scalar import _threshold_segments, pchip_curve_points, pchip_curve_sections
from .continuous_terrain import PhysicalTerrainField
from .implicit_curves import adaptive_curve_paths
from .implicit_pchip import split_pchip_branches
from .terrain_refinement import RefinedTerrainField


def ground_zero_paths(field):
    """Return true zero branches cut at solved physical meridian ports."""
    if isinstance(field, RefinedTerrainField):
        base = field.base
        forward = field.forward_ground_coordinates
        inverse = field.inverse_ground_coordinates
    elif isinstance(field, PhysicalTerrainField):
        base = field
        forward = inverse = lambda x, y: (x, y)
    else:
        raise TypeError("physical zero tracing requires the defined ground model")
    x = np.r_[0., np.arange(base.width)+.5, float(base.width)]
    y = np.r_[0., np.arange(base.height)+.5, float(base.height)]
    sampled = np.empty((len(y), len(x)))
    for begin in range(0, len(y), 64):
        sampled[begin:begin+64] = base.sample_rect(x, y[begin:begin+64])
    segments, lower, upper = _threshold_segments(base, x, y, sampled, 0.)
    if not len(segments):
        return []
    segments, lower, upper = split_pchip_branches(base, base.native_m, segments, lower, upper, 0.)
    # Source ports are one common graph. Longitude-equivalent ports share
    # the same inverse solve and retain their integer period separately;
    # adding then subtracting the world width cannot create a seam ULP gap.
    period = np.floor(segments[:, :, 0]/base.width).astype(int)
    normalized = segments.copy()
    normalized[:, :, 0] %= base.width
    unique, indices = np.unique(normalized.reshape(-1, 2), axis=0, return_inverse=True)
    xx, yy = inverse(unique[:, 0], unique[:, 1])
    canonical = np.column_stack((xx, yy))
    indices = indices.reshape(-1, 2)
    ports = canonical[indices].copy()
    ports[:, :, 0] += period*base.width

    def evaluate(starts, ends, owner):
        first_x, first_y = forward(starts[:, 0], starts[:, 1])
        last_x, last_y = forward(ends[:, 0], ends[:, 1])
        # A native port stays in its original interval rather than crossing
        # its root bracket by an ULP of the inverse coordinate solve.
        first = np.clip(np.column_stack((first_x, first_y)), lower[owner], upper[owner])
        last = np.clip(np.column_stack((last_x, last_y)), lower[owner], upper[owner])
        true = pchip_curve_sections(base, base.native_m, first, last, lower[owner], upper[owner], 0.)
        xx, yy = inverse(true[:, :, 0], true[:, :, 1])
        return np.stack((xx, yy), axis=-1)

    chains = adaptive_curve_paths(ports[:, 0], ports[:, 1], evaluate)
    owners = np.flatnonzero(np.any(ports[:, 0] != ports[:, 1], axis=1))
    result = []
    for chain, owner in zip(chains, owners, strict=True):
        first, last = chain[:-1], chain[1:]
        # Solve a crossing on the same true PCHIP branch, before encoding a
        # physical frame cut. GEOS chord intersections are not field roots.
        crossing = []
        meridians = range(int(np.floor(chain[:, 0].min()/base.width)),
                          int(np.ceil(chain[:, 0].max()/base.width))+1)
        for multiple in meridians:
            meridian = multiple*base.width
            active = np.flatnonzero((first[:, 0]-meridian)*(last[:, 0]-meridian) < 0)
            if not len(active):
                continue
            a, b = first[active], last[active]
            qax, qay = forward(a[:, 0], a[:, 1])
            qbx, qby = forward(b[:, 0], b[:, 1])
            qa = np.clip(np.column_stack((qax, qay)), lower[owner], upper[owner])
            qb = np.clip(np.column_stack((qbx, qby)), lower[owner], upper[owner])
            def positions(t, ax, ay, bx, by):
                qa, qb = np.column_stack((ax, ay)), np.column_stack((bx, by))
                q = pchip_curve_points(base, base.native_m,
                    qa+(qb-qa)*t[:, None], np.tile(lower[owner], (len(t), 1)),
                    np.tile(upper[owner], (len(t), 1)), 0.)
                px, py = inverse(q[:, 0], q[:, 1])
                return np.column_stack((px, py))

            def residual(t, ax, ay, bx, by):
                return positions(t, ax, ay, bx, by)[:, 0]-meridian

            root = elementwise.find_root(residual, (np.zeros(len(active)), np.ones(len(active))),
                args=(qa[:, 0], qa[:, 1], qb[:, 0], qb[:, 1]),
                tolerances={"xatol": 1e-14, "xrtol": 4*np.finfo(float).eps,
                            "fatol": 0., "frtol": 0.}, maxiter=100)
            if not np.all(root.success):
                raise ValueError("physical meridian zero ports must converge")
            point = positions(root.x, qa[:, 0], qa[:, 1], qb[:, 0], qb[:, 1])
            point[:, 0] = meridian
            crossing.extend((int(index), float(t), node) for index, t, node in
                            zip(active, root.x, point, strict=True))
        crossing.sort(key=lambda item: (item[0], item[1]))
        cuts = {}
        for index, _position, point in crossing:
            cuts.setdefault(index, []).append(point)
        expanded = []
        for index, point in enumerate(chain[:-1]):
            expanded.append(point)
            expanded.extend(cuts.get(index, ()))
        expanded.append(chain[-1])
        expanded = np.asarray(expanded)
        active_period = None
        part = []
        for index in range(len(expanded)-1):
            a, b = expanded[index:index+2]
            this_period = int(np.floor((a[0]+b[0])/(2*base.width)))
            aa, bb = a.copy(), b.copy()
            aa[0] -= this_period*base.width
            bb[0] -= this_period*base.width
            if index == 0:
                aa = canonical[indices[owner, 0]].copy()
                aa[0] += (period[owner, 0]-this_period)*base.width
            if index == len(expanded)-2:
                bb = canonical[indices[owner, 1]].copy()
                bb[0] += (period[owner, 1]-this_period)*base.width
            if active_period != this_period:
                if part:
                    result.append(np.asarray(part))
                active_period, part = this_period, [aa]
            part.append(bb)
        if part:
            result.append(np.asarray(part))
    return result


def ground_zero_surface(field):
    """Classify the single periodically cut arrangement with physical F > 0."""
    width, height = field.width, field.height
    frame = shapely.box(0., 0., float(width), float(height))
    lines = [frame.boundary]
    lines.extend(shapely.LineString(chain) for chain in ground_zero_paths(field))
    noded = shapely.node(shapely.MultiLineString(lines))
    faces, cuts, dangles, invalid = shapely.polygonize_full(shapely.get_parts(noded))
    if not cuts.is_empty or not dangles.is_empty or not invalid.is_empty:
        raise ValueError("physical zero branches must form a closed shared arrangement")
    faces = shapely.get_parts(faces)
    probes = shapely.get_coordinates(shapely.point_on_surface(faces))
    dry = field.sample_points(probes[:, 0], probes[:, 1]) > 0
    geometry = shapely.union_all(faces[dry])
    if not bool(shapely.is_valid(geometry)):
        raise ValueError("physical shoreline is invalid")
    return geometry
