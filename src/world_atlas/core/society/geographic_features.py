"""Physical support for named landforms, independent of name generation.

Summits use the verified, unclipped metre DEM. Local prominence is the
smallest drop available along eight rays within six native cells; it is
local ridge relief, not a claim of global topographic prominence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter

import numpy as np
from scipy import ndimage

from ..model import WorldGrid
from .model import SocietyLayers
from .spatial import connected_components


MOUNTAIN_ELEVATION_THRESHOLD = 0.56
LOCAL_PROMINENCE_RADIUS = 6
MINIMUM_LOCAL_PROMINENCE_M = 80.0
PEAK_RELOCATION_RADIUS = 17.0
_DIRECTIONS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


@dataclass(frozen=True, slots=True)
class RawSummit:
    row: int
    column: int
    elevation_m: float
    local_prominence_m: float
    component: int


def mountain_components(grid: WorldGrid) -> np.ndarray:
    """The exact connected support used to extract mountain-range anchors."""
    labels, _sizes = connected_components(
        (grid.water == 0) & (grid.elevation >= MOUNTAIN_ELEVATION_THRESHOLD)
    )
    return labels


def prominent_raw_summits(
    grid: WorldGrid, raw_elevation_m: np.ndarray, *, components: np.ndarray,
) -> tuple[RawSummit, ...]:
    values = np.asarray(raw_elevation_m, dtype=np.float64)
    if (values.shape != grid.shape or np.any(~np.isfinite(values))
            or components.shape != grid.shape):
        raise ValueError("raw summit DEM and mountain support must align with the native grid")
    land = grid.water == 0
    if np.any(values[land] <= 0.0):
        raise ValueError("raw summit DEM must contain actual positive metres above sea level on land")
    footprint = np.ones((3, 3), dtype=bool)
    footprint[1, 1] = False
    padded = np.pad(np.where(land, values, -np.inf), ((0, 0), (1, 1)), mode="wrap")
    padded = np.pad(padded, ((1, 1), (0, 0)), constant_values=-np.inf)
    neighbors = ndimage.maximum_filter(
        padded, footprint=footprint, mode="constant", cval=-np.inf,
    )[1:-1, 1:-1]
    # Strict maxima reject flattened mesas and the old clipped-elevation rim.
    rows, columns = np.nonzero(land & (components > 0) & (values > neighbors))
    if not len(rows):
        return ()
    height, width = grid.shape
    directional_bases = []
    for dy, dx in _DIRECTIONS:
        ray = []
        for distance in range(1, LOCAL_PROMINENCE_RADIUS + 1):
            next_rows = rows + dy * distance
            next_columns = (columns + dx * distance) % width
            valid = (next_rows >= 0) & (next_rows < height)
            next_rows = np.clip(next_rows, 0, height - 1)
            valid &= land[next_rows, next_columns]
            ray.append(np.where(valid, values[next_rows, next_columns], np.inf))
        directional_bases.append(np.min(ray, axis=0))
    prominence = values[rows, columns] - np.max(directional_bases, axis=0)
    supported = np.flatnonzero(prominence >= MINIMUM_LOCAL_PROMINENCE_M)
    return tuple(
        RawSummit(int(rows[index]), int(columns[index]),
                  float(values[rows[index], columns[index]]), float(prominence[index]),
                  int(components[rows[index], columns[index]]))
        for index in supported
    )


def refine_saved_geographic_features(
    grid: WorldGrid, society: SocietyLayers, *, raw_elevation_m: np.ndarray,
) -> tuple[SocietyLayers, dict]:
    """Reposition existing summit identities onto supported nearby summits.

All other human layers, names and physical fields retain their identities.
No files are saved and no missing landform is synthesized.
"""
    components = mountain_components(grid)
    summits = prominent_raw_summits(grid, raw_elevation_m, components=components)
    used: set[tuple[int, int]] = set()
    retained = []
    moved = []
    removed = []
    evidence = []
    range_support = []
    for feature in society.geographic_features:
        row, column = feature.row, feature.column
        if feature.feature_type == "mountain":
            component = int(components[row, column])
            if component <= 0:
                removed.append({"identifier": feature.identifier, "name": feature.name,
                                "reason": "no connected highland support at range anchor"})
                continue
            ys, xs = np.nonzero(components == component)
            range_support.append({"identifier": feature.identifier, "component": component,
                                  "cells": len(ys), "bounds": [int(xs.min()), int(ys.min()),
                                                               int(xs.max()) + 1, int(ys.max()) + 1]})
        if feature.feature_type != "peak":
            retained.append(feature)
            continue
        component = int(components[row, column])
        candidates = []
        for summit in summits:
            if summit.component != component or (summit.row, summit.column) in used:
                continue
            dx = abs(summit.column - column)
            dx = min(dx, grid.shape[1] - dx)
            squared_distance = (summit.row - row) ** 2 + dx ** 2
            if squared_distance <= PEAK_RELOCATION_RADIUS ** 2:
                candidates.append((squared_distance, -summit.local_prominence_m,
                                   -summit.elevation_m, summit.row, summit.column, summit))
        if not candidates:
            removed.append({"identifier": feature.identifier, "name": feature.name,
                            "reason": "no distinct raw summit with local ridge relief in its nearby highland component"})
            continue
        summit = min(candidates, key=lambda item: item[:-1])[-1]
        used.add((summit.row, summit.column))
        retained.append(replace(feature, row=summit.row, column=summit.column))
        evidence.append({"identifier": feature.identifier, "row": summit.row,
                         "column": summit.column, "elevationMeters": summit.elevation_m,
                         "localProminenceMeters": summit.local_prominence_m,
                         "component": summit.component})
        if (row, column) != (summit.row, summit.column):
            moved.append({"identifier": feature.identifier, "name": feature.name,
                          "before": [column, row], "after": [summit.column, summit.row]})
    refined = replace(society, geographic_features=tuple(retained))
    report = {
        "schema": "physical-geographic-features-v1", "featuresBefore": len(society.geographic_features),
        "featuresAfter": len(retained), "typeCounts": dict(Counter(item.feature_type for item in retained)),
        "supportedRawSummits": len(summits), "movedFeatures": moved, "removedFeatures": removed,
        "peakEvidence": evidence, "mountainSupport": range_support,
        "localProminence": {"units": "metres", "minimum": MINIMUM_LOCAL_PROMINENCE_M,
                            "radiusNativeCells": LOCAL_PROMINENCE_RADIUS,
                            "definition": "minimum of eight directional drops to the lowest valid land sample on each ray"},
        "unchangedRegionalFeatures": "Existing plain, basin, plateau, wetland and desert anchors retain their checked native support; a point record does not assert an unrecorded regional extent.",
    }
    return refined, report


def enrich_saved_geographic_features(grid: WorldGrid, society: SocietyLayers, *, thematic,
                                    raw_elevation_m: np.ndarray, lexicon) -> tuple[SocietyLayers, dict]:
    """Append physically supported regional landforms, retaining all saved records."""
    from ..landforms import derive_landform_inventory
    from .names import extend_geographic_features

    inventory = derive_landform_inventory(grid, thematic, raw_elevation_m=raw_elevation_m)
    features, additions = extend_geographic_features(
        grid, society.cultures, society.geographic_features, lexicon, inventory=inventory)
    enriched = replace(society, geographic_features=features)
    if additions:
        from .world_identity import assign_world_identity
        from ..presentation import society_content_digest
        request = grid.metadata["societyGeneration"]
        canonical = assign_world_identity(enriched, seed=int(request["namingSeed"]), forbidden=request["forbiddenNames"])
        if canonical.geographic_features[:len(society.geographic_features)] != society.geographic_features:
            raise AssertionError("landform naming replay would rename an existing geographic identity")
        if society_content_digest(replace(canonical, geographic_features=society.geographic_features)) != society_content_digest(society):
            raise AssertionError("landform naming replay would alter an existing human identity")
        enriched = replace(society, geographic_features=canonical.geographic_features)
        features = enriched.geographic_features
        final_names = {item.identifier: item.name for item in features}
        additions = [{**item, "name": final_names[item["identifier"]]} for item in additions]
    if features[:len(society.geographic_features)] != society.geographic_features:
        raise AssertionError("landform enrichment altered an existing geographic identity")
    return enriched, {
        "schema": "saved-landform-enrichment-v1", "featuresBefore": len(society.geographic_features),
        "featuresAfter": len(features), "originalRecordsPreserved": True,
        "addedFeatures": additions, "typeCounts": dict(Counter(item.feature_type for item in features)),
        "support": inventory.diagnostics,
    }
