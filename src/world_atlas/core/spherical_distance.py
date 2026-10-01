"""Reusable, latitude-aware graph distances on a periodic world raster."""

from functools import lru_cache
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@lru_cache(maxsize=2)
def _grid_graph(height: int, width: int, latitudes: tuple[float, ...], radius_km: float):
    """Keep graph construction out of each tectonic distance-field query."""
    rows, columns = np.indices((height, width), dtype=np.int32)
    cells = rows * width + columns
    row_step = math.pi * radius_km / height
    longitude_steps = math.tau * radius_km / width * np.cos(np.radians(latitudes))
    starts, ends, weights = [], [], []
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        valid = (rows + dr >= 0) & (rows + dr < height)
        starts.append(cells[valid])
        ends.append(((rows + dr) * width + (columns + dc) % width)[valid])
        step = np.maximum(np.hypot(row_step * abs(dr), longitude_steps * abs(dc)), row_step * 0.035)
        weights.append(np.broadcast_to(step[:, None], (height, width))[valid])
    count = height * width
    return csr_matrix((np.concatenate(weights), (np.concatenate(starts), np.concatenate(ends))), shape=(count, count))


def distance_from_sources(mask: np.ndarray, latitude_degrees: np.ndarray, radius_km: float) -> np.ndarray:
    """Multi-source Dijkstra with a periodic longitude and closed polar edges.

    East-west lengths use the source row, retaining the engine's directed D8
    metric exactly; treating this graph as undirected changes polar distances.
    Only one result per cell is allocated, regardless of source count.
    """
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2 or not source.size:
        raise ValueError("source mask must be a nonempty two-dimensional raster")
    latitude = np.asarray(latitude_degrees, dtype=np.float64)
    if latitude.shape != (source.shape[0],) or not np.isfinite(latitude).all():
        raise ValueError("distance grid needs one finite latitude per row")
    indices = np.flatnonzero(source)
    if not indices.size:
        raise ValueError("distance source mask is empty")
    graph = _grid_graph(*source.shape, tuple(latitude), float(radius_km))
    return dijkstra(graph, directed=True, indices=indices, min_only=True).reshape(source.shape)
