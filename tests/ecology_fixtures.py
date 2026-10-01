"""Declared continuous freshwater surfaces for small synthetic model fixtures."""

import numpy as np
import shapely

from world_atlas.core.continuous_ecology import FreshwaterCorridors


def declared_freshwater_sources(grid):
    height,width=grid.shape
    paths=[];orders=[]
    for order in np.unique(grid.river_order):
        if order==0:
            continue
        for column in np.flatnonzero(np.any(grid.river_order==order,axis=0)):
            rows=np.flatnonzero(grid.river_order[:,column]==order)
            if len(rows)>1:
                paths.append(np.array(((column+.5,rows[0]+.5),(column+.5,rows[-1]+.5))))
                orders.append(int(order))
    rows,columns=np.where(grid.water==2)
    lake=shapely.union_all(shapely.box(columns,rows,columns+1,rows+1))
    return FreshwaterCorridors.from_surfaces(grid.shape,paths,orders,lake)
