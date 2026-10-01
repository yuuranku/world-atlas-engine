"""Spatial access to canonical transport corridors for lazy city recipes."""

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
from shapely import LineString, STRtree, box

from .society.model import TransportLayers


def refined_land_routes(routes, locations):
    """Refine only land-route endpoints within their accepted settlement cells."""
    result = []
    for route in routes:
        if route.mode not in {"road", "rail"}:
            result.append(route)
            continue
        points = list(route.path)
        if route.source_settlement_id in locations:
            row, column = locations[route.source_settlement_id]
            if points[0] != (column, row):
                points.insert(0, (column, row))
        if route.target_settlement_id in locations:
            row, column = locations[route.target_settlement_id]
            if points[-1] != (column, row):
                points.append((column, row))
        result.append(replace(route, path=tuple(points)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CityCorridor:
    identifier: str
    kind: str
    points: tuple[tuple[float, float], ...]

    def document(self) -> dict:
        return {"id": self.identifier, "kind": self.kind,
                "points": [{"column": x, "row": y} for x, y in self.points]}


class CityTransportIndex:
    """Build once per world; query nearby routes instead of all routes per city.

    Copies at the longitude seam preserve short wrapped paths. Intersections
    split disconnected runs, so no invented line joins a route's re-entries.
    """

    def __init__(self, transport: TransportLayers, width: int, *, corridors):
        self.connections = defaultdict(lambda: {"road": 0, "rail": 0, "river": 0, "sea": 0, "bridges": 0})
        bridge_counts = defaultdict(int)
        for bridge in transport.bridges:
            bridge_counts[bridge.route_identifier] += 1
        geometries = []
        self.routes = []
        for route in transport.routes:
            for identifier in {route.source_settlement_id, route.target_settlement_id} - {None}:
                counts = self.connections[identifier]
                counts[route.mode] += 1
                counts["bridges"] += bridge_counts[route.identifier]
        for corridor in corridors:
            if corridor.kind not in {"road", "rail"} or len(corridor.points) < 2:
                continue
            points = np.asarray(corridor.points, dtype=np.float64)
            columns = np.unwrap(points[:, 0], period=width)
            for offset in (-width, 0, width):
                geometries.append(LineString(np.column_stack((columns + offset, points[:, 1]))))
                self.routes.append(corridor)
        self.tree = STRtree(geometries)

    def corridors(self, bounds: tuple[float, float, float, float]) -> tuple[CityCorridor, ...]:
        north, west, south, east = bounds
        pad_x, pad_y = (east - west) * 0.2, (south - north) * 0.2
        window = box(west - pad_x, north - pad_y, east + pad_x, south + pad_y)
        result = []
        for index in sorted(self.tree.query(window, predicate="intersects")):
            route = self.routes[index]
            geometry = self.tree.geometries[index].intersection(window)
            parts = geometry.geoms if hasattr(geometry, "geoms") else (geometry,)
            for part in parts:
                if part.geom_type != "LineString" or part.length <= 1e-9:
                    continue
                result.append(CityCorridor(route.identifier, route.kind, tuple(part.coords)))
        return tuple(result)
