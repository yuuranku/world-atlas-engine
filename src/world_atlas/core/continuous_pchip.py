"""Shared immutable longitude polynomials for tensor PCHIP fields."""
import numpy as np
from scipy.interpolate import PchipInterpolator


def periodic_horizontal_coefficients(native):
    """Construct each native row interval once, with periodic neighbours.

    Both ends of every retained interval are interior SciPy nodes, including
    the longitude seam and one-column fields. Row batches bound temporary
    construction memory; consumers retain the four binary64 coefficients.
    """
    height, width = native.shape
    columns = np.arange(-1, width+2)
    coefficients = np.empty((4, height, width), dtype=np.float64)
    for begin in range(0, height, 64):
        local = native[begin:begin+64, columns % width]
        polynomial = PchipInterpolator(columns, local, axis=1)
        coefficients[:, begin:begin+64] = polynomial.c[:, 1:width+1].transpose(0, 2, 1)
    coefficients.flags.writeable = False
    return coefficients


def interior_uniform_coefficients(values):
    """The middle interval of SciPy's four-node, unit-spaced PCHIP.

    Point queries only use this interior interval. Its two slopes depend on
    the neighbouring three secants; constructing endpoint rules, validating
    fixed axes and allocating an interpolator for every root iteration adds
    no information. Keep SciPy's harmonic arithmetic and coefficient order,
    including its zero/sign-switch branch, exactly.
    """
    slopes = values[1:] - values[:-1]
    condition = ((np.sign(slopes[1:]) != np.sign(slopes[:-1]))
                 | (slopes[1:] == 0) | (slopes[:-1] == 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (3/slopes[:-1] + 3/slopes[1:])/6
    derivatives = np.zeros_like(slopes[:2])
    derivatives[~condition] = 1/harmonic[~condition]
    cubic = derivatives[0]+derivatives[1]-2*slopes[1]
    return np.stack((cubic, slopes[1]-derivatives[0]-cubic,
                     derivatives[0], values[1]))
