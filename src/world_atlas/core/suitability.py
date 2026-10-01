"""Continuous physical responses shared by agricultural and human capacity."""

import math

import numpy as np
from scipy import ndimage


def smooth_transition(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Transition between physical limits without a jump at either endpoint."""
    unit=np.clip((np.asarray(values,dtype=np.float64)-lower)/(upper-lower),0.0,1.0)
    return unit*unit*(3.0-2.0*unit)


def relative_land_slope(elevation: np.ndarray, land: np.ndarray) -> np.ndarray:
    """Measure normalized elevation changes using only valid land neighbours.

    The canonical elevation is normalized, so this is relative change per
    native cell, not a slope angle or a metre-based DEM derivative. Ocean and
    lake display elevations are excluded from the land surface derivative.
    Longitude wraps; latitude stops at the first and last native rows.
    """
    values = np.asarray(elevation, dtype=np.float64)
    valid = np.asarray(land, dtype=bool)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("elevation and land must be aligned two-dimensional arrays")
    left = np.roll(valid, 1, axis=1)
    right = np.roll(valid, -1, axis=1)
    horizontal = np.where(right, np.roll(values, -1, axis=1) - values, 0.0)
    horizontal += np.where(left, values - np.roll(values, 1, axis=1), 0.0)
    np.divide(horizontal, left.astype(np.uint8) + right, out=horizontal,
              where=left | right)
    north = np.zeros_like(valid)
    south = np.zeros_like(valid)
    north[1:] = valid[:-1]
    south[:-1] = valid[1:]
    vertical = np.zeros_like(values)
    vertical[1:] += np.where(north[1:], values[1:] - values[:-1], 0.0)
    vertical[:-1] += np.where(south[:-1], values[1:] - values[:-1], 0.0)
    np.divide(vertical, north.astype(np.uint8) + south, out=vertical,
              where=north | south)
    slope = np.hypot(horizontal, vertical)
    slope[~valid] = 0.0
    return slope


def continuous_proximity(mask: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean source access with wrapped longitude and a smooth finite tail.

    This measures access before suitability is computed; it does not blur a
    completed population field or an already classified cartographic mask.
    """
    active=np.asarray(mask,dtype=bool)
    if active.ndim!=2 or not math.isfinite(radius) or radius<=0:
        raise ValueError("proximity needs a two-dimensional mask and positive finite radius")
    if not active.any():
        return np.zeros(active.shape,dtype=np.float64)
    halo=math.ceil(radius)
    padded=np.pad(~active,((0,0),(halo,halo)),mode="wrap")
    distance=ndimage.distance_transform_edt(padded)[:,halo:-halo]
    relative=np.clip(distance/radius,0.0,1.0)
    return np.square(1.0-relative*relative)
