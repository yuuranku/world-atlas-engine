"""Read-only acceptance evidence for final, high-resolution world geography.

This module deliberately consumes final rasters after all terrain processes.
It does not choose continents, alter sea level, or repair geometry: its job is
to make the difference between requested continent owners and actual connected
land masses explicit to an acceptance review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np
from world_atlas.core.raster_topology import periodic_component_labels as _periodic_labels


_MAJOR_LAND_SHARE = 0.05


def _validate_inputs(
    land_mask: Any,
    continent_id: Any,
    latitude_degrees: Any,
    recipe_continent_count: int,
    major_land_share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    land = np.asarray(land_mask, dtype=bool)
    owners = np.asarray(continent_id)
    latitude = np.asarray(latitude_degrees, dtype=np.float64)
    if land.ndim != 2 or not land.size:
        raise ValueError("land_mask must be a non-empty two-dimensional array")
    if owners.shape != land.shape:
        raise ValueError("continent_id must have the same shape as land_mask")
    if latitude.shape != (land.shape[0],) or not np.all(np.isfinite(latitude)):
        raise ValueError("latitude_degrees must contain one finite value per raster row")
    if np.any(np.abs(latitude) > 90.0):
        raise ValueError("latitude_degrees must be within [-90, 90]")
    if isinstance(recipe_continent_count, bool) or int(recipe_continent_count) < 1:
        raise ValueError("recipe_continent_count must be a positive integer")
    threshold = float(major_land_share)
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("major_land_share must be in (0, 1)")
    return land, owners, np.cos(np.radians(latitude)), int(recipe_continent_count), threshold




def _components(
    mask: np.ndarray,
    latitude_weights: np.ndarray,
    connectivity: int,
    *,
    total_area: float | None = None,
) -> dict[str, object]:
    labels, count = _periodic_labels(mask, connectivity)
    if not count:
        return {
            "connectivity": connectivity,
            "componentCount": 0,
            "weightedArea": 0.0,
            "components": [],
        }
    row_weights = np.broadcast_to(latitude_weights[:, None], labels.shape)
    component_cells = np.bincount(labels.reshape(-1), minlength=count + 1)
    component_area = np.bincount(
        labels.reshape(-1),
        weights=row_weights.reshape(-1),
        minlength=count + 1,
    )
    area = float(component_area[1:].sum())
    denominator = area if total_area is None else max(float(total_area), np.finfo(float).eps)
    order = np.argsort(component_area[1:])[::-1] + 1
    components = [
        {
            "rank": int(rank),
            "cellCount": int(component_cells[label]),
            "weightedArea": float(component_area[label]),
            "shareOfDomain": float(component_area[label] / denominator),
        }
        for rank, label in enumerate(order, 1)
    ]
    return {
        "connectivity": connectivity,
        "componentCount": count,
        "weightedArea": area,
        "components": components,
    }


def _owner_fragments(
    land: np.ndarray,
    owners: np.ndarray,
    latitude_weights: np.ndarray,
    total_land_area: float,
) -> list[dict[str, object]]:
    values = sorted(int(value) for value in np.unique(owners[land]) if int(value) > 0)
    result: list[dict[str, object]] = []
    for owner_id in values:
        owner_mask = land & (owners == owner_id)
        four = _components(owner_mask, latitude_weights, 4, total_area=total_land_area)
        eight = _components(owner_mask, latitude_weights, 8, total_area=total_land_area)
        owner_area = float(four["weightedArea"])
        four_components = list(four["components"])
        eight_components = list(eight["components"])
        result.append(
            {
                "ownerId": owner_id,
                "weightedArea": owner_area,
                "shareOfTotalLand": float(owner_area / max(total_land_area, np.finfo(float).eps)),
                "fourConnectedFragmentCount": int(four["componentCount"]),
                "eightConnectedFragmentCount": int(eight["componentCount"]),
                "largestFourConnectedShareWithinOwner": float(
                    four_components[0]["weightedArea"] / max(owner_area, np.finfo(float).eps)
                ) if four_components else 0.0,
                "largestEightConnectedShareWithinOwner": float(
                    eight_components[0]["weightedArea"] / max(owner_area, np.finfo(float).eps)
                ) if eight_components else 0.0,
                # Do not hide small fragments: raw ranked component areas are
                # intentionally retained for downstream acceptance review.
                "fourConnectedComponents": four_components,
                "eightConnectedComponents": eight_components,
            }
        )
    return result


def audit_final_geography(
    land_mask: Any,
    continent_id: Any,
    latitude_degrees: Any,
    recipe_continent_count: int,
    *,
    major_land_share: float = _MAJOR_LAND_SHARE,
) -> dict[str, object]:
    """Return reproducible final-raster geography acceptance evidence.

    A *major continent* is a 4- or 8-connected longitude-periodic land mass
    whose cosine-latitude weighted area is at least ``major_land_share`` of
    all land. The default 5% deliberately excludes islands while retaining a
    genuinely third continent. The report includes every raw fragment so a
    map cannot pass by hiding smaller disconnected land.
    """

    land, owners, weights, requested, threshold = _validate_inputs(
        land_mask,
        continent_id,
        latitude_degrees,
        recipe_continent_count,
        major_land_share,
    )
    land_four = _components(land, weights, 4)
    land_eight = _components(land, weights, 8)
    water_four = _components(~land, weights, 4)
    water_eight = _components(~land, weights, 8)
    total_land = float(land_four["weightedArea"])

    def major(components: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            component for component in components
            if float(component["shareOfDomain"]) >= threshold
        ]

    major_four = major(list(land_four["components"]))
    major_eight = major(list(land_eight["components"]))
    return {
        "schema": "world-atlas-final-geography-review-v1",
        "readOnly": True,
        "grid": {"height": int(land.shape[0]), "width": int(land.shape[1])},
        "areaWeighting": "cosine-latitude",
        "longitudeTopology": "periodic",
        "majorLandShareThreshold": threshold,
        "recipeContinentCount": requested,
        "land": {"fourConnected": land_four, "eightConnected": land_eight},
        "water": {"fourConnected": water_four, "eightConnected": water_eight},
        "continentAcceptance": {
            "fourConnectedMajorLandmassCount": len(major_four),
            "eightConnectedMajorLandmassCount": len(major_eight),
            "fourConnectedMatchesRecipe": len(major_four) == requested,
            "eightConnectedMatchesRecipe": len(major_eight) == requested,
            "fourConnectedMajorLandmasses": major_four,
            "eightConnectedMajorLandmasses": major_eight,
            "status": (
                "pass"
                if len(major_four) == requested and len(major_eight) == requested
                else "not_met"
            ),
        },
        "ownerFragments": _owner_fragments(land, owners, weights, total_land),
    }


def write_geography_review(path: str | Path, review: Mapping[str, object]) -> Path:
    """Write already-computed acceptance evidence without touching source fields."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(review), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = ["audit_final_geography", "write_geography_review"]
