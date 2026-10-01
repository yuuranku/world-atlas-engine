"""Add true 100-metre minor contours around verified inhabited places."""

from dataclasses import dataclass
import math

import numpy as np
import shapely

from .cartographic_features import line_features
from .cartographic_tiles import TileFeature
from .implicit_terrain import terrain_height_range, terrain_level_curves


@dataclass(frozen=True)
class CityRelief:
    features: tuple[TileFeature, ...]
    supports: tuple[dict, ...]
    published_tiles: tuple[tuple[int, int], ...]
    diagnostics: dict


def city_relief_supports(grid, settlements, settlement_locations):
    """Actual cities and smaller town contexts; isolated sites add no load."""
    radii = {"metropolis":50., "city":35., "town":15.}
    height, width = grid.shape
    extents = grid.metadata["extents"]
    radius = grid.metadata["planet"]["radiusKm"]
    north_km = math.radians(extents["north"]-extents["south"])*radius/height
    east_km = math.radians(extents["east"]-extents["west"])*radius/width
    result = []
    for city in settlements:
        if city.tier not in radii:
            continue
        y, x = settlement_locations[city.identifier]
        if not np.isfinite(x) or not np.isfinite(y) or not 0 <= x <= width or not 0 <= y <= height:
            raise ValueError("city relief requires the verified displayed settlement anchor")
        latitude = extents["north"]-y/height*(extents["north"]-extents["south"])
        ry = radii[city.tier]/north_km
        rx = radii[city.tier]/(east_km*max(math.cos(math.radians(latitude)),.12))
        for offset in (-width,0,width):
            xx = x+offset
            if xx+rx > 0 and xx-rx < width:
                result.append({"id":city.identifier,"x":xx,"y":y,"rx":rx,"ry":ry,"tier":city.tier})
    return tuple(result)


def _patches(supports, width, height):
    """Small disjoint query rectangles, rather than whole 600-km tiles."""
    result = set()
    for support in supports:
        x,y,rx,ry = (support[name] for name in ("x","y","rx","ry"))
        for row in range(max(0,math.floor((y-ry)/2)),min(math.ceil(height/2),math.ceil((y+ry)/2))):
            for column in range(max(0,math.floor((x-rx)/2)),min(math.ceil(width/2),math.ceil((x+rx)/2))):
                west,north = column*2,row*2
                east,south = min(west+2,width),min(north+2,height)
                nearest_x,nearest_y = np.clip(x,west,east),np.clip(y,north,south)
                if ((nearest_x-x)/rx)**2+((nearest_y-y)/ry)**2 < 1:
                    result.add((row,column))
    return tuple(sorted(result))


def _intersects_support(geometry, supports, tree):
    """Check the actual elliptical support without approximating its boundary."""
    for index in tree.query(geometry):
        support = supports[int(index)]
        centre = np.array((support["x"], support["y"]))
        scale = np.array((support["rx"], support["ry"]))
        for line in shapely.get_parts(geometry):
            if line.geom_type != "LineString":
                continue
            points = (np.asarray(line.coords)-centre)/scale
            first, delta = points[:-1], np.diff(points, axis=0)
            squared = np.sum(delta*delta, axis=1)
            distinct = squared > 0
            fraction = np.zeros(len(delta))
            fraction[distinct] = np.clip(-np.sum(first[distinct]*delta[distinct], axis=1)
                                         / squared[distinct], 0, 1)
            nearest = first+delta*fraction[:, None]
            if np.any(np.sum(nearest*nearest, axis=1) < 1):
                return True
    return False


def derive_city_relief(grid, terrain_field, settlements, palette_levels, *, settlement_locations, tile_size=32):
    """Trace minor-only city ink on the same actual ground as global relief.

    The solver selects complete native boxes intersecting the city queries;
    its original model, branch discovery and shared ports remain authoritative.
    Global colour faces and major contours continue unchanged underneath this
    additive ink. Only the existing smooth city support and land clip mask it.
    """
    supports = city_relief_supports(grid,settlements,settlement_locations)
    patches = _patches(supports,grid.shape[1],grid.shape[0])
    major_m = np.asarray(terrain_field.contour_height_m(palette_levels))
    query_bounds = np.asarray([(column*2, row*2, min(column*2+2, grid.shape[1]),
                                min(row*2+2, grid.shape[0])) for row, column in patches], dtype=float).reshape(-1, 4)
    levels = np.empty(0, dtype=float)
    if len(query_bounds):
        minimum, maximum = terrain_height_range(terrain_field, query_bounds)
        levels = np.arange(max(100, math.ceil(minimum/100)*100),
                           math.floor(maximum/100)*100+1, 100, dtype=float)
        levels = levels[~np.any(abs(levels[:, None]-major_m[None, :]) < 20, axis=1)]
    curves = terrain_level_curves(terrain_field, levels, query_bounds=query_bounds) if len(levels) else []
    support_tree = shapely.STRtree([shapely.box(item["x"]-item["rx"], item["y"]-item["ry"],
        item["x"]+item["rx"], item["y"]+item["ry"]) for item in supports])
    features = []
    contour_points = 0
    for level, branches in zip(levels, curves, strict=True):
        points = [points for points in branches
                  if _intersects_support(shapely.LineString(points), supports, support_tree)]
        contour_points += sum(len(line) for line in points)
        features.extend(line_features(points,"elevation-contours","#7d7065",.40,opacity=.30,clip="land",
            attributes={"data-city-relief":"true","data-height-m":str(float(level)),
                        "data-source-field":"accepted-continuous-ground",
                        "data-base-stroke":"0.40","data-contour-kind":"minor"}))
    feature_tree = shapely.STRtree([feature.geometry for feature in features])
    tiles = []
    for row, column in sorted({(row*2//tile_size,column*2//tile_size) for row,column in patches}):
        rectangle = shapely.box(column*tile_size, row*tile_size,
            min((column+1)*tile_size, grid.shape[1]), min((row+1)*tile_size, grid.shape[0]))
        if any(_intersects_support(features[int(index)].geometry.intersection(rectangle),
                                   supports, support_tree) for index in feature_tree.query(rectangle)):
            tiles.append((row, column))
    diagnostics = {"schema":"city-relief-implicit-v2","interpretation":"True 100-metre minor contours on the accepted continuous ground; global colour and major contours remain shared.",
                   "placeCounts":{tier:sum(city.tier==tier for city in settlements) for tier in ("metropolis","city","town")},
                   "supportRadiusKm":{"metropolis":50,"city":35,"town":15},"queryPatches":len(patches),
                   "queryLevels":len(levels),"publishedTiles":len(tiles),"minorContourIntervalM":100,"minorContourPoints":contour_points,
                   "sourceField":"accepted-continuous-ground","supportAnchors":"verified-displayed-settlements",
                   "nativeHeightsChanged":0,"shorelineChanged":False}
    return CityRelief(tuple(features),supports,tuple(tiles),diagnostics)
