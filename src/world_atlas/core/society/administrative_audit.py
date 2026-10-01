"""Read-only topology evidence for national and provincial partitions.

This module deliberately audits the final authored rasters rather than trying
to repair them.  A water-separated possession can be historically plausible;
a detached piece on the same physical landmass is instead evidence that needs
human review.  The distinction keeps the report useful without outlawing all
islands or real-world exclaves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..model import WorldGrid
from .model import SocietyLayers
from .administrative_topology import maritime_component_support
from .spatial import categorical_components, connected_components


_AUDIT_SCHEMA = "administrative-topology-audit-v2"


def _node_indices_for_cells(
    nodes: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    width: int,
) -> np.ndarray:
    """Resolve a small set of cell coordinates without a full-grid lookup."""

    flat = rows.astype(np.int64) * int(width) + columns.astype(np.int64)
    indices = np.searchsorted(nodes, flat)
    if np.any(indices >= nodes.size) or np.any(nodes[indices] != flat):
        raise ValueError("administrative core must lie inside its own territory")
    return indices.astype(np.int64, copy=False)


def _partition_audit(
    values: np.ndarray,
    records: Sequence[Any],
    settlements: dict[str, Any],
    land_component_id: np.ndarray,
    *,
    layer_name: str,
    routes=(),
) -> dict[str, Any]:
    """Summarise core-connected, same-land and island components per owner."""

    owners = np.asarray(values)
    nodes, component_by_node, owner_by_node = categorical_components(owners)
    record_by_identifier = {
        int(record.identifier): record for record in records
    }
    record_identifiers = tuple(sorted(record_by_identifier))
    observed_identifiers = tuple(
        int(value) for value in np.unique(owner_by_node) if value > 0
    )
    if observed_identifiers != record_identifiers:
        raise ValueError(f"{layer_name} records must match positive ownership labels")
    if not record_identifiers:
        return {
            "ownerCount": 0,
            "summary": {
                "componentCount": 0,
                "extraComponentCount": 0,
                "fragmentedOwnerCount": 0,
                "requiresReviewOwnerCount": 0,
                "sameLandNonCoreComponentCount": 0,
                "sameLandNonCoreAreaCells": 0,
                "seaSupportedSameLandComponentCount": 0,
                "unsupportedSameLandComponentCount": 0,
                "unsupportedSameLandAreaCells": 0,
                "waterSeparatedNonCoreComponentCount": 0,
                "waterSeparatedNonCoreAreaCells": 0,
                "seatOutsideLargestComponentCount": 0,
            },
            "owners": [],
        }

    component_count = int(component_by_node.max(initial=-1)) + 1
    component_area = np.bincount(
        component_by_node,
        minlength=component_count,
    ).astype(np.int64)
    owner_by_component = np.zeros(component_count, dtype=np.int32)
    np.maximum.at(owner_by_component, component_by_node, owner_by_node)
    land_by_node = np.asarray(land_component_id).ravel()[nodes].astype(
        np.int32,
        copy=False,
    )
    land_by_component = np.zeros(component_count, dtype=np.int32)
    np.maximum.at(land_by_component, component_by_node, land_by_node)

    maximum_identifier = max(record_identifiers)
    core_component_by_owner = np.full(maximum_identifier + 1, -1, dtype=np.int32)
    for identifier in record_identifiers:
        record = record_by_identifier[identifier]
        core = settlements.get(record.core_settlement_id)
        if core is None:
            raise ValueError(f"{layer_name} core settlement is missing")
        node_index = _node_indices_for_cells(
            nodes,
            np.asarray([int(core.row)], dtype=np.int64),
            np.asarray([int(core.column)], dtype=np.int64),
            owners.shape[1],
        )[0]
        if int(owner_by_node[node_index]) != identifier:
            raise ValueError("administrative core must lie inside its own territory")
        core_component_by_owner[identifier] = int(component_by_node[node_index])

    core_component = core_component_by_owner[owner_by_component]
    core_land_by_owner = land_by_component[core_component_by_owner]
    component_indexes = np.arange(component_count, dtype=np.int32)
    non_core = component_indexes != core_component
    same_land_non_core = non_core & (
        land_by_component == core_land_by_owner[owner_by_component]
    )
    water_separated_non_core = non_core & ~same_land_non_core
    component_grid = np.full(owners.shape, -1, dtype=np.int32)
    component_grid.ravel()[nodes] = component_by_node
    sea_supported, sea_evidence = maritime_component_support(
        component_grid, owner_by_component, core_component_by_owner,
        settlements.values(), routes,
    )
    unsupported_inland = same_land_non_core & ~sea_supported

    owner_component_count = np.bincount(
        owner_by_component,
        minlength=maximum_identifier + 1,
    ).astype(np.int64)
    owner_area = np.bincount(
        owner_by_component,
        weights=component_area,
        minlength=maximum_identifier + 1,
    ).astype(np.int64)
    owner_largest_component = np.zeros(maximum_identifier + 1, dtype=np.int64)
    np.maximum.at(
        owner_largest_component,
        owner_by_component,
        component_area,
    )
    same_land_component_count = np.bincount(
        owner_by_component[same_land_non_core],
        minlength=maximum_identifier + 1,
    ).astype(np.int64)
    same_land_area = np.bincount(
        owner_by_component[same_land_non_core],
        weights=component_area[same_land_non_core],
        minlength=maximum_identifier + 1,
    ).astype(np.int64)
    water_component_count = np.bincount(
        owner_by_component[water_separated_non_core],
        minlength=maximum_identifier + 1,
    ).astype(np.int64)
    water_area = np.bincount(
        owner_by_component[water_separated_non_core],
        weights=component_area[water_separated_non_core],
        minlength=maximum_identifier + 1,
    ).astype(np.int64)

    owner_records: list[dict[str, Any]] = []
    for identifier in record_identifiers:
        core_component_identifier = int(core_component_by_owner[identifier])
        core_area = int(component_area[core_component_identifier])
        same_land_count = int(same_land_component_count[identifier])
        supported_inland_count = int(np.count_nonzero(
            same_land_non_core & sea_supported & (owner_by_component == identifier)))
        owner_records.append(
            {
                "identifier": identifier,
                "coreSettlementId": str(
                    record_by_identifier[identifier].core_settlement_id
                ),
                "totalAreaCells": int(owner_area[identifier]),
                "componentCount": int(owner_component_count[identifier]),
                "coreComponentAreaCells": core_area,
                "largestComponentAreaCells": int(
                    owner_largest_component[identifier]
                ),
                "seatIsLargestComponent": bool(
                    core_area == int(owner_largest_component[identifier])
                ),
                "sameLandNonCoreComponentCount": same_land_count,
                "sameLandNonCoreAreaCells": int(same_land_area[identifier]),
                "waterSeparatedNonCoreComponentCount": int(
                    water_component_count[identifier]
                ),
                "waterSeparatedNonCoreAreaCells": int(water_area[identifier]),
                "seaSupportedSameLandComponentCount": supported_inland_count,
                "seaRouteEvidence": [item for item in sea_evidence if item["ownerIdentifier"] == identifier],
                "status": "requires_review" if same_land_count > supported_inland_count else "ok",
            }
        )

    return {
        "ownerCount": len(owner_records),
        "summary": {
            "componentCount": component_count,
            "extraComponentCount": int(component_count - len(owner_records)),
            "fragmentedOwnerCount": int(
                sum(item["componentCount"] > 1 for item in owner_records)
            ),
            "requiresReviewOwnerCount": int(
                sum(item["status"] == "requires_review" for item in owner_records)
            ),
            "sameLandNonCoreComponentCount": int(same_land_non_core.sum()),
            "sameLandNonCoreAreaCells": int(component_area[same_land_non_core].sum()),
            "seaSupportedSameLandComponentCount": int(np.count_nonzero(same_land_non_core & sea_supported)),
            "unsupportedSameLandComponentCount": int(np.count_nonzero(unsupported_inland)),
            "unsupportedSameLandAreaCells": int(component_area[unsupported_inland].sum()),
            "waterSeparatedNonCoreComponentCount": int(
                water_separated_non_core.sum()
            ),
            "waterSeparatedNonCoreAreaCells": int(
                component_area[water_separated_non_core].sum()
            ),
            "seatOutsideLargestComponentCount": int(
                sum(not item["seatIsLargestComponent"] for item in owner_records)
            ),
        },
        "owners": owner_records,
    }


def audit_administrative_topology(
    grid: WorldGrid,
    society: SocietyLayers,
) -> dict[str, Any]:
    """Return JSON-ready, deterministic evidence about final administration.

    A status of ``requires_review`` marks a same-land non-core component with
    no chain of same-owner sea routes back to its seat. Sea-supported inland
    exclaves retain their route IDs as explicit evidence. Water-separated
    components remain separately reported rather than geometrically banned.
    """

    state_id = np.asarray(society.politics.state_id)
    province_id = np.asarray(society.provinces.province_id)
    water = np.asarray(grid.water)
    if state_id.ndim != 2 or province_id.shape != state_id.shape:
        raise ValueError("administrative layers must share a two-dimensional shape")
    if water.shape != state_id.shape:
        raise ValueError("WorldGrid water field must match administrative layers")
    if tuple(getattr(grid, "shape", state_id.shape)) != state_id.shape:
        raise ValueError("WorldGrid shape must match administrative layers")

    land_component_id, land_component_sizes = connected_components(water == 0)
    settlements = {item.identifier: item for item in society.settlements}
    states = _partition_audit(
        state_id,
        society.politics.states,
        settlements,
        land_component_id,
        layer_name="state",
        routes=society.transport.routes,
    )
    provinces = _partition_audit(
        province_id,
        society.provinces.provinces,
        settlements,
        land_component_id,
        layer_name="province",
        routes=society.transport.routes,
    )
    needs_review = (
        states["summary"]["requiresReviewOwnerCount"] > 0
        or provinces["summary"]["requiresReviewOwnerCount"] > 0
    )
    return {
        "schema": _AUDIT_SCHEMA,
        "shape": [int(state_id.shape[0]), int(state_id.shape[1])],
        "physicalLandComponentCount": len(land_component_sizes),
        "physicalLandAreaCells": int(sum(land_component_sizes)),
        "status": "requires_review" if needs_review else "ok",
        "states": states,
        "provinces": provinces,
    }


__all__ = ["audit_administrative_topology"]
