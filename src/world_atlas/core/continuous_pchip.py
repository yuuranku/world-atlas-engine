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
