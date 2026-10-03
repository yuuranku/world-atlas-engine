"""Trace continuous thematic quantities before assigning display classes."""

import math
from functools import cached_property

import numpy as np
import shapely
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from scipy.optimize import elementwise

from .implicit_curves import adaptive_curve_paths, level_bands, polygon_path
from .implicit_pchip import split_pchip_branches
from .continuous_pchip import periodic_horizontal_coefficients


class ContinuousScalarField:
    """Reconstruct saved scalar samples with a periodic longitude.

    Water has no observation of a land-only quantity. Its nearest land value
    extends the field for interpolation; the common physical land clip owns
    the displayed coastline. This avoids inventing a low-value coastal strip.
    """

    def __init__(self, values, land_mask):
        values = np.asarray(values, dtype=np.float64)
        land = np.asarray(land_mask, dtype=bool)
        if (values.ndim != 2 or min(values.shape) < 1 or values.shape != land.shape
                or not np.all(np.isfinite(values)) or not bool(land.any())):
            raise ValueError("a land scalar needs matching finite observations and land")
        self.height, self.width = values.shape
        self.native = values.copy()
        if not bool(land.all()):
            extended_land = np.tile(land, (1, 3))
            nearest = ndimage.distance_transform_edt(
                ~extended_land, return_distances=False, return_indices=True)
            rows, columns = nearest[:, :, self.width:2*self.width]
            self.native[~land] = values[rows[~land], columns[~land] % self.width]
        self.native.flags.writeable = False

    @cached_property
    def horizontal_coefficients(self):
        return periodic_horizontal_coefficients(self.native)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('horizontal_coefficients', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.native.flags.writeable = False

    def sample_rect(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if (x.ndim != 1 or y.ndim != 1 or not x.size or not y.size
                or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
                or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)
                or np.any(y < 0) or np.any(y > self.height)):
            raise ValueError("scalar sampling requires finite increasing map axes")
        columns = np.arange(math.floor(float(x[0])-.5)-2,
                            math.floor(float(x[-1])-.5)+4)
        rows = np.arange(math.floor(float(y[0])-.5)-2,
                         math.floor(float(y[-1])-.5)+4)
        local = self.native[np.clip(rows, 0, self.height-1)[:, None],
                            columns[None, :] % self.width]
        horizontal = PchipInterpolator(columns+.5, local, axis=1)(x)
        return PchipInterpolator(rows+.5, horizontal, axis=0)(y)

    def sample_points(self, x, y):
        """Evaluate paired points on exactly the same tensor PCHIP surface.

        Cached longitude coefficients and bounded latitude batches avoid
        reconstructing fixed polynomials or a Cartesian product of points.
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
                or np.any(y < 0) or np.any(y > self.height)):
            raise ValueError("scalar point sampling requires finite map coordinates")
        shape = x.shape
        x, y = x.ravel(), y.ravel()
        result = np.empty(x.size, dtype=float)
        offsets = np.arange(-1, 3)[:, None]
        for begin in range(0, x.size, 16384):
            stop = min(x.size, begin+16384)
            xx, yy = x[begin:stop], y[begin:stop]
            column = np.floor(xx-.5).astype(np.int64)
            row = np.floor(yy-.5).astype(np.int64)
            rows = np.clip(row[None, :]+offsets, 0, self.height-1)
            coefficients = self.horizontal_coefficients[:, rows, column % self.width]
            tx = xx-.5-column
            horizontal = ((coefficients[0]*tx+coefficients[1])*tx+coefficients[2])*tx+coefficients[3]
            coefficients = PchipInterpolator(np.arange(-1, 3), horizontal, axis=0).c[:, 1]
            ty = yy-.5-row
            result[begin:stop] = ((coefficients[0]*ty+coefficients[1])*ty+coefficients[2])*ty+coefficients[3]
        return result.reshape(shape)


def _threshold_segments(field, x, y, sampled, level):
    """Build the actual native-interval contour graph, including saddle cases.

    Each native horizontal edge is a monotone first-stage PCHIP interval.
    Each vertical edge is a monotone second-stage interval. Their exact roots
    are shared by both neighbouring cells. A checkerboard has two branches:
    the smaller horizontal root joins the left edge, the larger the right.
    This follows the vertical-monotone field rather than a bilinear decider.
    """
    above = sampled > level
    horizontal = above[:, :-1] != above[:, 1:]
    vertical = above[:-1] != above[1:]
    hr, hc = np.nonzero(horizontal)
    vr, vc = np.nonzero(vertical)
    if not len(hr) and not len(vr):
        return np.empty((0, 2, 2)), np.empty((0, 2)), np.empty((0, 2))

    def horizontal_residual(position, latitude):
        return field.sample_points(position, latitude)-level

    def vertical_residual(position, longitude):
        return field.sample_points(longitude, position)-level

    tolerances = {"xatol": 1e-14, "xrtol": 4*np.finfo(float).eps, "fatol": 0., "frtol": 0.}
    hroot = elementwise.find_root(horizontal_residual, (x[hc], x[hc+1]),
                                 args=(y[hr],), tolerances=tolerances, maxiter=100)
    vroot = elementwise.find_root(vertical_residual, (y[vr], y[vr+1]),
                                 args=(x[vc],), tolerances=tolerances, maxiter=100)
    if not np.all(hroot.success) or not np.all(vroot.success):
        raise ValueError("continuous scalar native-edge roots must converge")
    points = np.concatenate((np.column_stack((hroot.x, y[hr])),
                             np.column_stack((x[vc], vroot.x))))
    hi = np.full(horizontal.shape, -1, np.int32)
    vi = np.full(vertical.shape, -1, np.int32)
    hi[hr, hc] = np.arange(len(hr))
    vi[vr, vc] = np.arange(len(vr))+len(hr)
    count = horizontal[:-1].astype(np.int8)+horizontal[1:]+vertical[:, :-1]+vertical[:, 1:]
    rows, columns = np.nonzero(count)
    segments, lower, upper = [], [], []
    for row, column in zip(rows, columns, strict=True):
        edges = (hi[row, column], vi[row, column+1], hi[row+1, column], vi[row, column])
        active = [int(index) for index in edges if index >= 0]
        if len(active) == 2:
            pairs = (active,)
        elif len(active) == 4:
            north, east, south, west = map(int, edges)
            pairs = ((west, north), (south, east)) if points[north, 0] <= points[south, 0] else ((west, south), (north, east))
        else:
            raise ValueError("a continuous native interval must have two or four contour ports")
        for first, last in pairs:
            segments.append(points[[first, last]])
            lower.append((x[column], y[row]))
            upper.append((x[column+1], y[row+1]))
    return np.asarray(segments), np.asarray(lower), np.asarray(upper)


def _vertical_coefficients(field, native, longitude, lower):
    """Construct each fixed-longitude native polynomial once for root finding."""
    coefficients = np.empty((4, len(longitude)))
    native_scale = np.empty(len(longitude))
    row = np.floor(lower-.5).astype(np.int64)
    first, last = np.empty(len(row)), np.empty(len(row))
    offsets = np.arange(-1, 3)[:, None]
    for begin in range(0, len(row), 16384):
        stop = min(len(row), begin+16384)
        xx, rr = longitude[begin:stop], row[begin:stop]
        column = np.floor(xx-.5).astype(np.int64)
        columns = (column[None, :]+offsets) % field.width
        rows = np.clip(rr[None, :]+offsets, 0, field.height-1)
        local = native[rows[:, None, :], columns[None, :, :]]
        native_scale[begin:stop] = np.max(abs(local), axis=(0,1))
        horizontal_coefficients = field.horizontal_coefficients[:, rows, column % field.width]
        tx = xx-.5-column
        horizontal = ((horizontal_coefficients[0]*tx+horizontal_coefficients[1])*tx
                      + horizontal_coefficients[2])*tx+horizontal_coefficients[3]
        coefficients[:, begin:stop] = PchipInterpolator(np.arange(-1, 3), horizontal, axis=0).c[:, 1]
        first[begin:stop], last[begin:stop] = horizontal[1], horizontal[2]
    return (*coefficients, row+.5, first, last), native_scale


def pchip_curve_sections(field, native, starts, ends, lower, upper, level):
    """Solve three true ordinates, reusing their unchanged PCHIP coefficients."""
    fractions = np.asarray((.25, .5, .75))
    points = (starts[:, None, :]+(ends-starts)[:, None, :]*fractions[None, :, None]).reshape(-1, 2)
    lower, upper = np.repeat(lower, 3, axis=0), np.repeat(upper, 3, axis=0)
    return pchip_curve_points(field, native, points, lower, upper, level).reshape(-1,3,2)


def pchip_curve_points(field, native, points, lower, upper, level):
    """Solve true ordinates at the supplied parameters of native branches."""
    points = np.asarray(points, dtype=float).copy()
    arguments, native_scale = _vertical_coefficients(field, native, points[:, 0], lower[:, 1])

    def residual(position, c0, c1, c2, c3, origin, first, last):
        t = position-origin
        value = ((c0*t+c1)*t+c2)*t+c3
        # PCHIP owns native nodes exactly. At an endpoint choose its saved
        # value, matching sample_points' containing-interval convention.
        return np.where(t == 0, first, np.where(t == 1, last, value))-level

    numerical_zero = (8*np.finfo(float).eps*np.maximum(max(1., abs(level)), native_scale)
                      * np.maximum(1., np.max(abs(points), axis=1)))
    indices = np.flatnonzero(abs(residual(points[:, 1], *arguments)) > numerical_zero)
    if len(indices):
        root = elementwise.find_root(
            residual, (lower[indices, 1], upper[indices, 1]),
            args=tuple(argument[indices] for argument in arguments),
            tolerances={"xatol": 1e-14, "xrtol": 4*np.finfo(float).eps, "fatol": 0., "frtol": 0.}, maxiter=100,
        )
        if not np.all(root.success):
            raise ValueError("continuous scalar curve sections must converge")
        points[indices, 1] = root.x
    return points


def _adaptive_isolines(field, segments, lower, upper, level):
    """Trace this model's monotone roots with the shared true-curve encoder."""
    segments, lower, upper = split_pchip_branches(field, field.native, segments, lower, upper, level)
    def evaluate(starts, ends, owner):
        return pchip_curve_sections(field, field.native, starts, ends, lower[owner], upper[owner], level)

    return adaptive_curve_paths(segments[:, 0], segments[:, 1], evaluate)


def scalar_band_paths(values, land_mask, thresholds):
    """Extract one shared coverage from the actual nonlinear scalar field.

    Native intervals own topology and exact SciPy roots own their ports.
    Adaptive curve sections measure error relative to each feature's chord,
    so barely super-threshold peaks retain their true curved footprint.
    Every threshold is extracted once; Shapely nodes all contours with the
    frame and polygonizes the complete arrangement for adjacent display bands.
    """
    values = np.asarray(values, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if (values.ndim != 2 or values.shape != land.shape or not np.all(np.isfinite(values))
            or thresholds.ndim != 1 or not np.all(np.isfinite(thresholds))
            or np.any(np.diff(thresholds) <= 0)):
        raise ValueError("scalar display requires matching fields and increasing thresholds")
    if not bool(land.any()):
        return [[] for _ in range(len(thresholds)+1)]
    field = ContinuousScalarField(values, land)
    x = np.r_[0., np.arange(field.width)+.5, float(field.width)]
    y = np.r_[0., np.arange(field.height)+.5, float(field.height)]
    sampled = np.empty((len(y), len(x)), dtype=float)
    for begin in range(0, len(y), 64):
        sampled[begin:begin+64] = field.sample_rect(x, y[begin:begin+64])
    paths = []
    # Source classes are [lower, upper). A representable preceding cut assigns
    # exact threshold plateaus to their upper class without changing the field.
    for level in np.nextafter(thresholds, -np.inf):
        segments, lower, upper = _threshold_segments(field, x, y, sampled, float(level))
        if len(segments):
            paths.append(_adaptive_isolines(field, segments, lower, upper, float(level)))
        else:
            paths.append([])
    geometries = level_bands(field, thresholds, paths)
    return [[polygon_path(polygon) for polygon in shapely.get_parts(geometry)]
            for geometry in geometries]
