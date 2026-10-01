"""Extract land colours and contours from the actual continuous ground."""

import numpy as np
import shapely

from .implicit_curves import level_bands, polygon_path
from .implicit_terrain import terrain_level_curves


def physical_relief_paths(terrain_field, palette_levels, *, checkpoint_directory=None,
                          source_identity=None):
    """Derive nested fills and ink from one graph of true physical curves."""
    palette = np.asarray(palette_levels, dtype=float)
    metres = np.asarray(terrain_field.contour_height_m(palette), dtype=float)
    if palette.ndim != 1 or np.any(np.diff(palette) <= 0):
        raise ValueError("physical relief requires increasing palette levels")
    positive = np.flatnonzero(metres > 0)
    bands = [None for _ in palette]
    if not len(positive):
        return bands, [], []
    thresholds = metres[positive]
    if checkpoint_directory is None:
        curves = terrain_level_curves(terrain_field, thresholds)
    else:
        from .physical_contour_stage import staged_height_curves
        curves = staged_height_curves(terrain_field, thresholds, checkpoint_directory,
                                     source_identity=source_identity)
    regions = level_bands(terrain_field, thresholds, curves)
    higher = shapely.GeometryCollection()
    for index in range(len(positive)-1, -1, -1):
        higher = shapely.union_all((higher, regions[index+1]))
        bands[int(positive[index])] = [polygon_path(polygon)
                                      for polygon in shapely.get_parts(higher)]
    lines = [chain for chains in curves for chain in chains]
    line_levels = [float(palette[index])
                   for index, chains in zip(positive, curves, strict=True)
                   for _chain in chains]
    return bands, lines, line_levels
