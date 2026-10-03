"""Expose the real smooth sections of the existing tensor PCHIP model.

The horizontal interpolation is cubic inside every native longitude interval.
The second, vertical PCHIP changes its harmonic-slope formula when adjacent
horizontal row values become equal. These are genuine model corners, rather
than curvature that an indefinitely smaller chord can resolve. SciPy's real
polynomial solver owns their positions; the unchanged field owns ordinates.
"""

import numpy as np
from scipy.interpolate import PPoly


def pchip_native_switch_abscissae(field, lower, upper):
    """Return isolated slope switches, including native-interval endpoints.

    These belong to the source tensor PCHIP model, independent of any level
    or transformed physical contour. Its adapter must solve its own ordinate.
    Periodic copies use one set of native coefficients.
    A row difference that is identically zero has no isolated switch point.
    """
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lower.ndim != 2 or lower.shape[1:] != (2,) or upper.shape != lower.shape:
        raise ValueError("PCHIP switch events require matching native brackets")
    if not len(lower):
        return []
    column = np.floor(lower[:, 0]-.5).astype(np.int64)
    row = np.floor(lower[:, 1]-.5).astype(np.int64)
    cells, inverse = np.unique(np.column_stack((column % field.width, row)),
                               axis=0, return_inverse=True)
    events = []
    offsets = np.arange(-1, 3)
    for begin in range(0, len(cells), 16384):
        local_cells = cells[begin:begin+16384]
        rows = np.clip(local_cells[:, 1][None, :]+offsets[:, None], 0, field.height-1)
        coefficients = field.horizontal_coefficients[:, rows, local_cells[:, 0]]
        differences = np.diff(coefficients, axis=1)
        roots = PPoly(differences[:, None, :, :], [0., 1.], extrapolate=False).solve(
            discontinuity=False, extrapolate=False)
        for index in range(len(local_cells)):
            # PPoly.solve returns a startpoint followed by NaN for an
            # identically-zero piece. That is a whole relation, not an
            # isolated source event; inclusive endpoints cannot retain it.
            isolated = [roots[difference, index] for difference in range(3)
                        if np.any(differences[:, difference, index] != 0)]
            roots_at_cell = np.concatenate(isolated) if isolated else np.empty(0)
            events.append(np.unique(roots_at_cell[np.isfinite(roots_at_cell)]))
    result = []
    for index, cell in enumerate(inverse):
        x = events[int(cell)]+column[index]+.5
        result.append(x)
    return result


def pchip_switch_abscissae(field, lower, upper):
    """Return interior switches; the base adapter already owns native ports."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    return [x[(x > first[0]) & (x < last[0])]
            for x, first, last in zip(
                pchip_native_switch_abscissae(field, lower, upper),
                lower, upper, strict=True)]


def split_pchip_branches(field, native, segments, lower, upper, level):
    """Split native branches at every actual vertical slope-formula event.

    The returned brackets remain the original native intervals. Original
    ports are retained exactly and each inserted event point is shared by
    its two new sections.
    """
    segments = np.asarray(segments, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if not len(segments):
        return segments, lower, upper
    if (segments.shape[1:] != (2, 2) or lower.shape != (len(segments), 2)
            or upper.shape != lower.shape):
        raise ValueError("PCHIP sections require paired native branches and brackets")
    events = pchip_switch_abscissae(field, lower, upper)

    # Import at the call boundary: the scalar adapter itself consumes this
    # model-section helper, while its public ordinate solver owns root finding.
    from .continuous_scalar import pchip_curve_points

    owners, positions, counts = [], [], []
    for index, (segment, x) in enumerate(zip(segments, events, strict=True)):
        minimum, maximum = sorted(segment[:, 0])
        x = x[(x > minimum) & (x < maximum)]
        if segment[1, 0] < segment[0, 0]:
            x = x[::-1]
        counts.append(len(x))
        if len(x):
            # A native bracket midpoint gives the ordinate solver identical
            # input in either branch direction, including its exact-root case.
            y = np.full(len(x), (lower[index, 1]+upper[index, 1])*.5)
            positions.extend(np.column_stack((x, y)))
            owners.extend([index]*len(x))
    if not positions:
        return segments, lower, upper
    owners = np.asarray(owners, dtype=np.int64)
    points = pchip_curve_points(field, native, np.asarray(positions), lower[owners], upper[owners], float(level))
    output, output_owners = [], []
    cursor = 0
    for index, (segment, count) in enumerate(zip(segments, counts, strict=True)):
        ports = np.vstack((segment[0], points[cursor:cursor+count], segment[1]))
        output.extend(np.stack((ports[:-1], ports[1:]), axis=1))
        output_owners.extend([index]*(count+1))
        cursor += count
    output_owners = np.asarray(output_owners, dtype=np.int64)
    return np.asarray(output), lower[output_owners], upper[output_owners]
