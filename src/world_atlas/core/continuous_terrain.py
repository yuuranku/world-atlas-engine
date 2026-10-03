"""Interpolate one saved physical ground field, without adding terrain detail.

The field is in model metres relative to its sea-level datum.  Its zero
contour is the shoreline; positive samples also own land colours and contours.
The source has one shared zero for oceans and enclosed water, and does not
contain independent vertical lake-level observations.
"""

import math
from functools import cached_property

import numpy as np
from scipy.interpolate import PchipInterpolator

from .continuous_pchip import interior_uniform_coefficients, periodic_horizontal_coefficients


class PhysicalTerrainField:
    """A periodic, shape-preserving reconstruction of native ground samples.

    Rectangular queries use SciPy's tensor PCHIP construction in two batches.
    Unlike unrestricted bicubic splines, each sample stays within its four
    containing native corner values.  No shoreline distance, noise, smoothing,
    or inferred subgrid measurements are added.
    """

    def __init__(self, relative_elevation_m, *, land_mask, sea_level_m,
                 elevation_scale_m, elevation_exponent):
        values = np.asarray(relative_elevation_m, dtype=np.float64)
        land = np.asarray(land_mask, dtype=bool)
        if (values.ndim != 2 or min(values.shape) < 1 or land.shape != values.shape
                or not np.all(np.isfinite(values))):
            raise ValueError("physical terrain requires matching finite two-dimensional samples")
        if not np.array_equal(values > 0, land):
            raise ValueError("physical terrain sign differs from native land ownership")
        if (not math.isfinite(sea_level_m) or not math.isfinite(elevation_scale_m)
                or elevation_scale_m <= 0 or not math.isfinite(elevation_exponent)
                or elevation_exponent <= 0):
            raise ValueError("physical terrain requires a finite datum and positive palette mapping")
        self.native_m = values.copy()
        self.native_m.flags.writeable = False
        self.height, self.width = values.shape
        self.sea_level_m = float(sea_level_m)
        self.elevation_scale_m = float(elevation_scale_m)
        self.elevation_exponent = float(elevation_exponent)

    @cached_property
    def horizontal_coefficients(self):
        return periodic_horizontal_coefficients(self.native_m)

    def __getstate__(self):
        # Workers reconstruct this derived cache locally; do not transfer a
        # full four-coefficient raster with every spawned terrain field.
        state = self.__dict__.copy()
        state.pop('horizontal_coefficients', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.native_m.flags.writeable = False

    def sample_rect(self, x, y):
        """Sample native map coordinates, whose original centres are n + .5.

        Longitude is periodic, including queries outside the displayed frame.
        Latitude retains the outer native samples through the polar half-cell.
        The two ghost nodes on either side give interior PCHIP derivatives the
        same neighbours independently of tile bounds or sampling density.
        """
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if (x.ndim != 1 or y.ndim != 1 or not x.size or not y.size
                or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
                or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)
                or np.any(y < 0) or np.any(y > self.height)):
            raise ValueError("physical sampling requires finite increasing map axes")
        columns = np.arange(math.floor(float(x[0]) - .5) - 2,
                            math.floor(float(x[-1]) - .5) + 4)
        rows = np.arange(math.floor(float(y[0]) - .5) - 2,
                         math.floor(float(y[-1]) - .5) + 4)
        local = self.native_m[np.clip(rows, 0, self.height - 1)[:, None],
                              columns[None, :] % self.width]
        horizontal = PchipInterpolator(columns + .5, local, axis=1)(x)
        return PchipInterpolator(rows + .5, horizontal, axis=0)(y)

    def palette_elevation(self, relative_m):
        """Apply the generator's saved relative colour mapping after sampling."""
        values = np.asarray(relative_m, dtype=np.float64)
        return np.where(values > 0,
                        np.clip(.02 + .98 * (np.maximum(values, 0)
                                           / self.elevation_scale_m)**self.elevation_exponent,
                                .015, 1), 0)

    def sample_points(self, x, y):
        """Sample paired coordinates using the same tensor PCHIP definition.

        Longitude polynomials are shared by every query of the immutable
        native field. Only the latitude coefficients depend on longitude.
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   np.asarray(y, dtype=np.float64))
        if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
                or np.any(y < 0) or np.any(y > self.height)):
            raise ValueError("physical point sampling requires finite map coordinates")
        shape = x.shape
        x, y = x.ravel(), y.ravel()
        column, row = np.floor(x - .5).astype(np.int64), np.floor(y - .5).astype(np.int64)
        offsets = np.arange(-1, 3)[:, None]
        rows = np.clip(row[None, :] + offsets, 0, self.height - 1)
        x_coefficients = self.horizontal_coefficients[:, rows, column % self.width]
        tx = x - .5 - column
        horizontal = ((x_coefficients[0] * tx + x_coefficients[1]) * tx
                      + x_coefficients[2]) * tx + x_coefficients[3]
        y_coefficients = interior_uniform_coefficients(horizontal)
        ty = y - .5 - row
        result = ((y_coefficients[0] * ty + y_coefficients[1]) * ty
                  + y_coefficients[2]) * ty + y_coefficients[3]
        return result.reshape(shape)

    def contour_height_m(self, palette_height):
        """Express a land colour boundary in the same physical ground datum."""
        height = np.asarray(palette_height, dtype=np.float64)
        if not np.all(np.isfinite(height)) or np.any(height < 0) or np.any(height > 1):
            raise ValueError("land contour levels must be finite relative values in 0..1")
        return self.elevation_scale_m * (np.maximum(height - .02, 0)
                                         / .98)**(1 / self.elevation_exponent)
