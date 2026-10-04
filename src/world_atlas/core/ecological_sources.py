"""Bind ecological supply to the same continuous physical waterways as the map."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from shapely.geometry.base import BaseGeometry

from .cartographic_surface import continuous_land_surface
from .continuous_ecology import FreshwaterCorridors
from .model import WorldGrid, _thaw_json
from .procedural_planet import ProceduralSurface
from .terrain_refinement import RefinedTerrainField, terrain_from_source
from .thematic import ThematicLayers, derive_thematic_layers
from ..timing import measure_stage


def _physical_source_digest(source):
    identity = hashlib.sha256()
    for field in fields(ProceduralSurface):
        if field.name == 'diagnostics':
            continue
        values = np.ascontiguousarray(getattr(source, field.name))
        identity.update(field.name.encode())
        identity.update(values.dtype.str.encode())
        identity.update(str(values.shape).encode())
        identity.update(memoryview(values).cast('B'))
    identity.update(json.dumps(_thaw_json(source.diagnostics), sort_keys=True,
                               separators=(',', ':'), allow_nan=False).encode())
    return identity.hexdigest()


@dataclass(frozen=True)
class PreparedWorldLayers:
    """One run's physical and ecological layers, shared with its renderer."""

    grid_digest: str
    physical_source_digest: str
    timing_output: Path
    terrain: RefinedTerrainField
    land_surface: BaseGeometry
    ecological_sources: FreshwaterCorridors
    thematic: ThematicLayers

    def validate(self, grid: WorldGrid, source: ProceduralSurface):
        if (grid.content_digest() != self.grid_digest
                or _physical_source_digest(source) != self.physical_source_digest):
            raise ValueError('prepared world layers differ from the current physical inputs')


def _sources_from_surfaces(grid, terrain, land):
    from .render import _lake_surface

    sources = terrain.river_source_paths
    orders = []
    for path in sources:
        columns = np.clip(np.floor(path[:, 0]).astype(int), 0, grid.shape[1]-1)
        rows = np.clip(np.floor(path[:, 1]).astype(int), 0, grid.shape[0]-1)
        orders.append(int(np.max(grid.river_order[rows, columns], initial=0)))
    return FreshwaterCorridors.from_surfaces(grid.shape, terrain.river_paths, orders, _lake_surface(grid, land))


def derive_ecological_sources(grid, physical_source):
    """Use accepted shore/outlet/reach factories without extracting relief cuts."""
    terrain = terrain_from_source(grid, physical_source)
    land = continuous_land_surface(grid, terrain_field=terrain)
    return _sources_from_surfaces(grid, terrain, land)


def prepare_world_layers(grid, physical_source, *, timing_output):
    """Compute physical inputs once for society and its subsequent rendering."""
    logger = logging.getLogger(__name__)
    timing_output = Path(timing_output).resolve()
    logger.info('Preparing the shared physical ground')
    with measure_stage(timing_output, 'physical-ground'):
        terrain = terrain_from_source(grid, physical_source)
    logger.info('Preparing the shared physical shoreline')
    with measure_stage(timing_output, 'physical-shoreline'):
        land = continuous_land_surface(grid, terrain_field=terrain)
    logger.info('Preparing the shared ecological sources')
    with measure_stage(timing_output, 'ecological-sources'):
        sources = _sources_from_surfaces(grid, terrain, land)
    logger.info('Preparing the shared thematic layers')
    with measure_stage(timing_output, 'thematic-layers'):
        thematic = derive_thematic_layers(grid, ecological_sources=sources)
    return PreparedWorldLayers(grid.content_digest(), _physical_source_digest(physical_source),
                               timing_output, terrain, land, sources, thematic)
