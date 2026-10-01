"""Bind ecological supply to the same continuous physical waterways as the map."""

import numpy as np

from .cartographic_curves import terrain_channel_paths
from .cartographic_surface import continuous_land_surface
from .continuous_ecology import FreshwaterCorridors
from .terrain_refinement import terrain_from_source


def derive_ecological_sources(grid, physical_source):
    """Use accepted shore/outlet/reach factories without extracting relief cuts."""
    from .render import _geometry_filled_paths, _surface_outline_paths, _river_paths, _lake_surface

    terrain = terrain_from_source(grid, physical_source)
    land = continuous_land_surface(grid, terrain_field=terrain)
    coast = _surface_outline_paths(_geometry_filled_paths(land), grid.shape)
    sources = _river_paths(grid, coast, raw_elevation_m=physical_source.relative_elevation_m)
    orders = []
    for path in sources:
        columns = np.clip(np.floor(path[:, 0]).astype(int), 0, grid.shape[1]-1)
        rows = np.clip(np.floor(path[:, 1]).astype(int), 0, grid.shape[0]-1)
        orders.append(int(np.max(grid.river_order[rows, columns], initial=0)))
    reaches = terrain_channel_paths(grid, sources, terrain_field=terrain)
    return FreshwaterCorridors.from_surfaces(grid.shape, reaches, orders, _lake_surface(grid, land))
