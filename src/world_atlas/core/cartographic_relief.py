"""Extract land colours and contours from the actual continuous ground."""

import numpy as np
import shapely

from .implicit_curves import level_bands, polygon_path
from .implicit_terrain import _ROOT_TOLERANCES
from .cartographic_contours import cartographic_level_curves


def _closed_relief_curves(field, curves):
    """Close interior loops within the source solver's coordinate accuracy.

    The stored roots remain intact. A final segment bridges roundoff between
    the first and last root, and belongs to both the fill and its ink graph.
    Open curves on the map frame and gaps beyond root accuracy stay open so
    the shared-coverage validation still rejects real topology errors.
    """
    encoded=[]
    frame=np.array((field.width,field.height),dtype=float)
    for paths in curves:
        rings=[]
        for chain in paths:
            ends=chain[[0,-1]]
            on_frame=np.any((ends==0)|(ends==frame))
            tolerance=(_ROOT_TOLERANCES['xatol']+
                       _ROOT_TOLERANCES['xrtol']*np.max(abs(ends),axis=0))
            if (len(chain)>2 and not on_frame and np.any(ends[0]!=ends[1])
                    and np.all(abs(ends[0]-ends[1])<=tolerance)):
                chain=np.vstack((chain,chain[0]))
            rings.append(chain)
        encoded.append(rings)
    return encoded


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
        curves = cartographic_level_curves(terrain_field, thresholds)
    else:
        from .physical_contour_stage import staged_height_curves
        curves = staged_height_curves(terrain_field, thresholds, checkpoint_directory,
                                     source_identity=source_identity)
    curves = _closed_relief_curves(terrain_field, curves)
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
