"""Trace the zero branches of the one reconstructed physical ground field."""

import numpy as np
import shapely

from .implicit_ground import ground_zero_surface


def continuous_land_surface(grid, *, terrain_field):
    """Trace the saved ground sea-level crossing, shared by every map layer.

    Native PCHIP intervals isolate the actual base zero branches. The ground
    model transports them, and true curve sections control their adaptive
    encoding. No fixed sampling grid decides island size or replaces the
    continuous zero with linear edge interpolation.
    """
    water = np.asarray(grid.water)
    ground = np.asarray(terrain_field.native_m)
    if (water.ndim != 2 or min(water.shape) < 1 or ground.shape != water.shape
            or not np.array_equal(ground > 0, water == 0)):
        raise ValueError("physical shore requires matching ground and native land ownership")
    land = water == 0
    height, width = land.shape
    if not bool(land.any()):
        return shapely.MultiPolygon()
    if bool(land.all()):
        return shapely.box(0, 0, width, height)
    return ground_zero_surface(terrain_field)
