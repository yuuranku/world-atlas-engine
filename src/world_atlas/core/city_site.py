"""Bounded saved-world context for cities generated when a reader opens them."""

import math

import numpy as np
from .cartographic_rivers import river_width_field


def city_site_context(grid, recipe, terrain_field, *, river_widths=None):
    """Sample factual parent terrain, leaving all local geometry to the client."""
    height, width = grid.shape
    centre = recipe["location"]
    urban = recipe["urban"]
    row_km = urban["gridCellKilometres"]["row"]
    column_km = urban["gridCellKilometres"]["column"]
    radius_km = max(5., urban["radiusKm"] * 5, (recipe.get('harbor') or {}).get('distanceKm', 0)*1.35)
    bounds = {
        "north": max(0., centre["row"] - radius_km / row_km),
        "west": centre["column"] - radius_km / column_km,
        "south": min(float(height), centre["row"] + radius_km / row_km),
        "east": centre["column"] + radius_km / column_km,
    }
    rows = columns = 17
    yy, xx = np.meshgrid(np.linspace(bounds["north"], bounds["south"], rows),
                         np.linspace(bounds["west"], bounds["east"], columns), indexing="ij")
    # Sample the same continuous, refined ground used by the world shoreline.
    # Native water-cell edges must not become a second staircase coastline.
    elevation = terrain_field.sample_points(xx, yy)
    if not np.all(np.isfinite(elevation)):
        raise ValueError("city site context requires finite, saved metre elevations on the world grid")
    native_y = np.clip(np.floor(yy).astype(int), 0, height - 1)
    native_x = np.floor(xx).astype(int) % width
    if river_widths is None:
        river_widths=river_width_field(grid)
    source = _freshwater_source(grid, recipe, radius_km, row_km, column_km, river_widths)
    return {
        "schema": "city-site-context-v1",
        "bounds": {key: round(value, 7) for key, value in bounds.items()},
        "rows": rows,
        "columns": columns,
        "elevationMetres": np.round(elevation, 2).ravel().tolist(),
        "water": np.where(elevation > 0, 0,
                          np.maximum(1, np.asarray(grid.water)[native_y, native_x])).astype(int).ravel().tolist(),
        "riverOrder": np.asarray(grid.river_order)[native_y, native_x].astype(int).ravel().tolist(),
        "radiusKm": round(radius_km, 3),
        "latitudeDegrees":90-centre['row']/height*180,
        "gridCellKilometres": {"row": row_km, "column": column_km},
        "source": source,
    }


def _freshwater_source(grid, recipe, radius_km, row_km, column_km, river_widths):
    height, width = grid.shape
    row, column = recipe["anchorCell"]["row"], recipe["anchorCell"]["column"]
    offsets_y, offsets_x = np.meshgrid(np.arange(-18, 19), np.arange(-18, 19), indexing="ij")
    yy, xx = row + offsets_y, (column + offsets_x) % width
    inside = (yy >= 0) & (yy < height)
    yy = np.clip(yy, 0, height - 1)
    river = np.asarray(grid.river_order)[yy, xx]
    lake = np.asarray(grid.water)[yy, xx] == 2
    distance = np.hypot(offsets_y * row_km, offsets_x * column_km)
    distance[~(inside & ((river > 0) | lake))] = np.inf
    nearest = np.unravel_index(np.argmin(distance), distance.shape)
    distance_km = float(distance[nearest])
    strategic = recipe["siteType"] in {"fortress", "pass", "oasis"}
    if not math.isfinite(distance_km) or distance_km > radius_km:
        return {"kind": "groundwater" if strategic else "spring-fed-stream",
                "location": recipe["location"], "distanceKm": None,
                "referenceWidthMetres": max(2., .5*math.sqrt(radius_km**2*.6))}
    target_row = int(yy[nearest])
    target_column = column + int(offsets_x[nearest])
    return {
        "kind": "river" if river[nearest] > 0 else "lake",
        "location": {"row": target_row + .5, "column": target_column + .5},
        "distanceKm": round(distance_km, 3),
        "referenceWidthMetres": float(river_widths[target_row,target_column % width]) if river[nearest] > 0 else 4.,
    }
