"""Bind ecological supply to the same continuous physical waterways as the map."""

import numpy as np

from .cartographic_surface import continuous_land_surface
from .continuous_ecology import FreshwaterCorridors
from .terrain_refinement import terrain_from_source


def derive_ecological_sources(grid, physical_source):
    """Use accepted shore/outlet/reach factories without extracting relief cuts."""
    from .render import _lake_surface

    terrain = terrain_from_source(grid, physical_source)
    land = continuous_land_surface(grid, terrain_field=terrain)
    sources = terrain.river_source_paths
    orders = []
    for path in sources:
        columns = np.clip(np.floor(path[:, 0]).astype(int), 0, grid.shape[1]-1)
        rows = np.clip(np.floor(path[:, 1]).astype(int), 0, grid.shape[0]-1)
        orders.append(int(np.max(grid.river_order[rows, columns], initial=0)))
    return FreshwaterCorridors.from_surfaces(grid.shape, terrain.river_paths, orders, _lake_surface(grid, land))
