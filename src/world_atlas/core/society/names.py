"""Name-only legacy boundary and deterministic language-aware toponymy."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..model import WorldGrid
from ..thematic import ThematicLayers, smooth_field
from .model import CultureLayers, GeographicFeature, NameLexicon, Settlement
from .onomastics import (
    lineage_roots,
    lineage_style_index,
    settlement_name_candidates,
)
from .spatial import (
    connected_components,
    reduce_field,
    select_spaced_seeds,
    society_domain_mask,
)


_GENERIC_ENDINGS = "国邦领盟城市镇村港堡关庭社侯伯公国群诸大塞尔汗炉海湾山河谷岛屿"
_FEATURE_SUFFIX = {
    "river": "河",
    "mountain": "山脉",
    "peak": "山",
    "lake": "湖",
    "sea": "海",
    "inland-sea": "内海",
    "bay": "湾",
    "strait": "海峡",
    "island": "岛",
    "island-group": "群岛",
    "plain": "平原",
    "plateau": "高原",
    "basin": "盆地",
}
_QUALIFIERS = ("上", "下", "东", "西", "南", "北", "中", "内", "外", "前", "后", "大", "小")
_POLITICAL_NAME_ENDINGS = (
    "关津国群", "盐港城邦", "海洋共和国", "城邦共和国",
    "城邦联盟", "伯国联", "男国群", "台吉领", "阿米尔国",
    "埃米尔国", "苏丹国", "海塞尔", "大塞尔", "诸领", "诸邦",
    "诸国", "诸城", "城盟", "炉盟", "汗国", "天畿", "公国", "侯国",
    "王国", "伯国", "部盟", "国群", "国联",
)


def _unique_names(values: list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        name = str(raw).strip()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return tuple(result)


def _root(name: str) -> str:
    cleaned = re.sub(r"[·\-—\s]", "", name)
    cleaned = cleaned.rstrip(_GENERIC_ENDINGS)
    if len(cleaned) >= 2:
        return cleaned[:3]
    return (cleaned or name)[:2]


def _expand_roots(values: list[str], *, limit: int = 128) -> tuple[str, ...]:
    """Build a neutral fallback pool without retaining obsolete culture tags."""

    expanded = list(dict.fromkeys(values))
    seen = set(expanded)
    base = tuple(expanded)
    for first_index, first in enumerate(base):
        for offset in range(1, len(base)):
            second = base[(first_index + offset) % len(base)]
            for candidate in (first[:1] + second[-1:], first[:2] + second[-1:]):
                if len(candidate) < 2 or candidate in seen:
                    continue
                seen.add(candidate)
                expanded.append(candidate)
                if len(expanded) >= limit:
                    return tuple(expanded)
    return tuple(expanded)


def _country_identity(raw_name: str, raw_type: str) -> str:
    """Remove the historical regime title from a reusable country name."""

    name = str(raw_name).strip()
    endings = tuple(dict.fromkeys((str(raw_type).strip(), *_POLITICAL_NAME_ENDINGS)))
    for ending in sorted((item for item in endings if item), key=len, reverse=True):
        if name.endswith(ending) and len(name) > len(ending):
            return name[: -len(ending)]
    return name


def _country_placement_role(raw_name: str, raw_type: str) -> str:
    """Keep geographic placement semantics, never the historical government."""

    source = f"{raw_name}{raw_type}"
    if "天畿" in source:
        return "ritual-core"
    if "汗" in source:
        return "steppe"
    if any(token in source for token in ("海", "湾", "港", "闽")):
        return "coastal"
    if any(token in source for token in ("炉", "岭", "关")):
        return "highland"
    if any(token in source for token in ("河", "江", "双河")):
        return "river"
    return "regional"


def load_name_lexicon(path: str | Path) -> NameLexicon:
    """Read setting names and naming traditions, discarding legacy geography."""

    source = json.loads(Path(path).read_text(encoding="utf-8"))
    major_entries = tuple(
        (
            _country_identity(item["name"], item["type"]),
            _country_placement_role(item["name"], item["type"]),
        )
        for item in source["polityAnchors"]
    )
    major = _unique_names([item[0] for item in major_entries])
    role_by_name = {name: role for name, role in major_entries}
    minor = _unique_names([item["name"] for item in source["minorEntries"]])
    roots = _expand_roots([_root(name) for name in (*major, *minor) if _root(name)])
    return NameLexicon(
        major_country_names=major,
        major_country_roles=tuple(role_by_name[name] for name in major),
        minor_country_names=minor,
        place_roots=roots,
    )


def generate_feature_name(
    lexicon: NameLexicon,
    *,
    feature_type: str,
    language_identifier: int,
    ordinal: int,
    used: set[str],
    name_family: str | None = None,
) -> str:
    suffix = _FEATURE_SUFFIX.get(feature_type)
    if suffix is None:
        raise ValueError(f"unsupported geographic feature type: {feature_type}")
    digest = hashlib.sha256(
        f"{feature_type}:{language_identifier}:{ordinal}".encode("utf-8")
    ).digest()
    roots = lineage_roots(name_family) if name_family is not None else lexicon.place_roots
    if not roots:
        raise ValueError(f"name family has no place roots: {name_family}")
    start = int.from_bytes(digest[:4], "big") % len(roots)
    root = roots[start]
    candidate = f"{root}{suffix}"
    if candidate not in used:
        used.add(candidate)
        return candidate
    for offset in range(len(_QUALIFIERS) ** 2):
        first = _QUALIFIERS[(start + offset) % len(_QUALIFIERS)]
        second_index = offset // len(_QUALIFIERS)
        qualifier = first if second_index == 0 else first + _QUALIFIERS[second_index - 1]
        candidate = f"{qualifier}{root}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError("unable to resolve geographic-name collision")


def name_settlements(
    settlements: tuple[Settlement, ...],
    cultures: CultureLayers,
    lexicon: NameLexicon,
    grid: WorldGrid,
    thematic: ThematicLayers,
) -> tuple[Settlement, ...]:
    """Assign unique local-language names grounded in each site's geography."""

    lineage_by_language = {
        item.identifier: item.name_family for item in cultures.languages
    }
    used: set[str] = set()
    result: list[Settlement] = []
    environment_masks = {
        "river": grid.river_order >= 2,
        "lake": grid.water == 2,
        "coast": (grid.water == 1) | (grid.water == 3),
        "mountain": (grid.water == 0) & (grid.elevation >= 0.50),
    }
    for ordinal, settlement in enumerate(settlements):
        language_identifier = _nearest_language(
            cultures.language_id,
            settlement.row,
            settlement.column,
        )
        lineage = lineage_by_language[language_identifier]
        digest = hashlib.sha256(
            f"settlement:{settlement.identifier}:{language_identifier}".encode("utf-8")
        ).digest()
        environment, relation = _settlement_environment(
            settlement,
            grid,
            thematic,
            environment_masks,
        )
        candidates = settlement_name_candidates(
            lineage,
            environment,
            relation,
            seed=int.from_bytes(digest[:8], "big"),
        )
        name = next((candidate for candidate in candidates if candidate not in used), None)
        if name is None:
            raise ValueError(f"lineage cannot provide a unique settlement name: {lineage}")
        used.add(name)
        result.append(replace(settlement, name=name))
    return tuple(result)


def _nearest_mask_offset(
    mask: np.ndarray,
    row: int,
    column: int,
    *,
    radius: int,
) -> tuple[int, int] | None:
    height, width = mask.shape
    rows = np.arange(max(0, row - radius), min(height, row + radius + 1))
    columns = (np.arange(column - radius, column + radius + 1) % width).astype(int)
    active_rows, active_columns = np.nonzero(mask[np.ix_(rows, columns)])
    if active_rows.size == 0:
        return None
    row_delta = rows[active_rows] - row
    column_delta = active_columns - radius
    distances = row_delta.astype(np.float64) ** 2 + column_delta.astype(np.float64) ** 2
    best = int(np.argmin(distances))
    return int(row_delta[best]), int(column_delta[best])


def _relation_from_offset(offset: tuple[int, int] | None) -> str | None:
    if offset is None:
        return None
    row_delta, column_delta = offset
    if row_delta == 0 and column_delta == 0:
        return None
    # The offset points from the settlement to the feature, so the settlement
    # lies on the opposite side of that direction.
    if abs(column_delta) > abs(row_delta):
        return "西" if column_delta > 0 else "东"
    return "北" if row_delta > 0 else "南"


def _settlement_environment(
    settlement: Settlement,
    grid: WorldGrid,
    thematic: ThematicLayers,
    masks: dict[str, np.ndarray],
) -> tuple[str, str | None]:
    radius = max(8, min(30, min(grid.shape) // 48))
    forced = {
        "port": "coast",
        "island-port": "coast",
        "lake-port": "lake",
        "river-city": "river",
        "pass": "pass",
        "fortress": "fortress",
        "oasis": "spring",
    }.get(settlement.site_type)
    if forced is not None:
        if forced == "spring":
            return forced, None
        base = "mountain" if forced in {"pass", "fortress"} else forced
        return forced, _relation_from_offset(
            _nearest_mask_offset(masks[base], settlement.row, settlement.column, radius=radius)
        )
    nearest: list[tuple[int, str, tuple[int, int]]] = []
    for kind, mask in masks.items():
        offset = _nearest_mask_offset(mask, settlement.row, settlement.column, radius=radius)
        if offset is not None:
            nearest.append((offset[0] ** 2 + offset[1] ** 2, kind, offset))
    if nearest:
        distance, kind, offset = min(nearest)
        if distance <= radius * radius * 0.42:
            return kind, _relation_from_offset(offset)
    annual = float(thematic.climate.annual_precipitation[settlement.row, settlement.column])
    potential = float(thematic.land_potential[settlement.row, settlement.column])
    if annual > 0.34:
        return "wetland", None
    if potential >= 0.48 and float(grid.elevation[settlement.row, settlement.column]) < 0.43:
        return "plain", None
    return "region", None


def _nearest_language(
    language_id: np.ndarray,
    row: int,
    column: int,
) -> int:
    height, width = language_id.shape
    direct = int(language_id[row, column])
    if direct > 0:
        return direct
    for radius in range(1, max(height, width)):
        top = max(0, row - radius)
        bottom = min(height, row + radius + 1)
        for candidate_row in range(top, bottom):
            for candidate_column in ((column - radius) % width, (column + radius) % width):
                value = int(language_id[candidate_row, candidate_column])
                if value > 0:
                    return value
        for candidate_column in range(column - radius + 1, column + radius):
            wrapped = candidate_column % width
            for candidate_row in (top, bottom - 1):
                value = int(language_id[candidate_row, wrapped])
                if value > 0:
                    return value
    return 1


def _component_anchors(
    mask: np.ndarray,
    importance: np.ndarray,
    *,
    limit: int,
    minimum_size: int,
) -> tuple[tuple[int, int, int], ...]:
    labels, sizes = connected_components(mask)
    ranked = sorted(
        (
            (size, label)
            for label, size in enumerate(sizes, start=1)
            if size >= minimum_size
        ),
        reverse=True,
    )[:limit]
    result: list[tuple[int, int, int]] = []
    for size, label in ranked:
        component = labels == label
        rows, columns = np.nonzero(component)
        values = np.asarray(importance[rows, columns], dtype=np.float64)
        low = float(np.min(values, initial=0.0))
        span = max(float(np.max(values, initial=1.0)) - low, 1.0e-12)
        normalized = np.clip((values - low) / span, 0.0, 1.0)
        # Labels belong in the readable interior of a landform, not at the
        # single highest-valued edge pixel.  Importance only breaks ties
        # between similarly interior locations.
        depth = _interior_depth(component)[rows, columns]
        best = int(np.argmax(depth + 0.32 * normalized))
        result.append((int(rows[best]), int(columns[best]), int(size)))
    return tuple(result)


def _snap_anchor(
    support: np.ndarray,
    row: int,
    column: int,
    *,
    radius: int,
    priority: np.ndarray | None = None,
) -> tuple[int, int]:
    """Snap a coarse label centre onto the physical feature it describes."""

    allowed = np.asarray(support, dtype=bool)
    if allowed.ndim != 2:
        raise ValueError("label support must be a two-dimensional mask")
    height, width = allowed.shape
    row = min(height - 1, max(0, int(row)))
    column = int(column) % width
    rows = np.arange(max(0, row - radius), min(height, row + radius + 1))
    columns = np.arange(column - radius, column + radius + 1) % width
    local = allowed[np.ix_(rows, columns)]
    if not np.any(local):
        return row, column
    row_delta = rows[:, None] - row
    column_delta = np.abs(columns[None, :] - column)
    column_delta = np.minimum(column_delta, width - column_delta)
    score = -np.hypot(row_delta, column_delta)
    if priority is not None:
        values = np.asarray(priority, dtype=np.float64)
        if values.shape != allowed.shape:
            raise ValueError("label priority must match its support mask")
        local_priority = values[np.ix_(rows, columns)]
        finite = local_priority[np.isfinite(local_priority)]
        if finite.size:
            low = float(np.min(finite))
            span = max(float(np.max(finite)) - low, 1.0e-12)
            score += 0.85 * radius * np.clip((local_priority - low) / span, 0.0, 1.0)
    score[~local] = -np.inf
    best = int(np.argmax(score))
    local_row, local_column = np.unravel_index(best, score.shape)
    return int(rows[local_row]), int(columns[local_column])


def _interior_depth(mask: np.ndarray) -> np.ndarray:
    """Return wrapped, four-neighbour distance from a component shoreline."""

    current = np.asarray(mask, dtype=bool).copy()
    depth = np.zeros(current.shape, dtype=np.float64)
    level = 1.0
    while np.any(current):
        depth[current] = level
        padded = np.pad(current, ((1, 1), (0, 0)), mode="constant")
        current = (
            current
            & np.roll(current, 1, axis=1)
            & np.roll(current, -1, axis=1)
            & padded[:-2]
            & padded[2:]
        )
        level += 1.0
    return depth


def _river_basin_anchors(
    grid: WorldGrid,
    drainage_basin: np.ndarray,
    *,
    limit: int,
    minimum_cells: int = 6,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return one principal-river anchor per terminal drainage basin."""

    basins = np.asarray(drainage_basin)
    if basins.shape != grid.shape:
        raise ValueError("drainage basin must match WorldGrid shape")
    principal = (grid.river_order >= 3) & (basins >= 0)
    candidates: list[tuple[float, int, int, int, int]] = []
    flat_basins = basins.reshape(-1)
    principal_cells = np.flatnonzero(principal.reshape(-1))
    order = np.argsort(flat_basins[principal_cells], kind="stable")
    sorted_cells = principal_cells[order]
    sorted_basins = flat_basins[sorted_cells]
    boundaries = np.flatnonzero(np.diff(sorted_basins)) + 1
    for cells in np.split(sorted_cells, boundaries):
        if cells.size < minimum_cells:
            continue
        basin_identifier = int(flat_basins[cells[0]])
        values = grid.discharge.reshape(-1)[cells].astype(np.float64)
        maximum = float(np.max(values, initial=0.0))
        target = maximum * 0.58
        anchor = int(cells[int(np.argmin(np.abs(values - target)))])
        score = float(cells.size) * max(1.0e-12, np.log1p(maximum))
        row, column = divmod(anchor, grid.shape[1])
        candidates.append(
            (score, int(row), int(column), int(cells.size), int(basin_identifier))
        )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return tuple(
        (row, column, size, basin_identifier)
        for _score, row, column, size, basin_identifier in candidates[:limit]
    )


def _minor_river_reach_anchors(
    grid: WorldGrid,
    *,
    limit: int,
    minimum_length: int = 7,
    drainage_basin: np.ndarray | None = None,
    named_basins: set[int] | None = None,
) -> tuple[tuple[int, int, float], ...]:
    """Name tributary reaches between headwaters and confluences."""

    active = grid.river_order.reshape(-1) > 0
    downstream = grid.flow_to.reshape(-1)
    discharge = grid.discharge.reshape(-1).astype(np.float64)
    basin_by_cell = None
    if drainage_basin is not None:
        basins = np.asarray(drainage_basin)
        if basins.shape != grid.shape:
            raise ValueError("drainage basin must match WorldGrid shape")
        basin_by_cell = basins.reshape(-1)
    indegree = np.zeros(active.size, dtype=np.int32)
    for source in np.flatnonzero(active):
        target = int(downstream[source])
        if 0 <= target < active.size and active[target]:
            indegree[target] += 1
    starts = [int(cell) for cell in np.flatnonzero(active) if indegree[cell] != 1]
    visited: set[tuple[int, int]] = set()
    reaches: list[tuple[float, int, int]] = []

    def follow(start: int) -> None:
        cells = [start]
        current = start
        while 0 <= current < active.size and active[current]:
            target = int(downstream[current])
            if target < 0 or target >= active.size or not active[target]:
                break
            edge = (current, target)
            if edge in visited:
                break
            visited.add(edge)
            cells.append(target)
            if indegree[target] != 1:
                break
            current = target
        if len(cells) < minimum_length:
            return
        orders = grid.river_order.reshape(-1)[cells]
        maximum_order = int(np.max(orders, initial=0))
        middle = cells[min(len(cells) - 1, max(1, round((len(cells) - 1) * 0.58)))]
        # A named terminal basin owns one continuous principal-river name.
        # Only low-order tributaries may receive additional detail names;
        # otherwise segmented reaches of the same trunk acquire aliases.
        if (
            basin_by_cell is not None
            and named_basins
            and int(basin_by_cell[middle]) in named_basins
            and maximum_order >= 3
        ):
            return
        score = (
            3.0 * maximum_order
            + 1.8 * np.log1p(float(np.max(discharge[cells], initial=0.0)))
            + min(18.0, len(cells) * 0.35)
        )
        reaches.append((float(score), int(middle), len(cells)))

    for start in starts:
        follow(start)
    for source in np.flatnonzero(active):
        target = int(downstream[source])
        if 0 <= target < active.size and active[target] and (int(source), target) not in visited:
            follow(int(source))
    selected = sorted(reaches, key=lambda item: (-item[0], item[1]))[: limit * 5]
    result: list[tuple[int, int, float]] = []
    for score, cell, _length in selected:
        row, column = divmod(cell, grid.shape[1])
        if any(
            np.hypot(
                row - other_row,
                min(abs(column - other_column), grid.shape[1] - abs(column - other_column)),
            ) < 26.0
            for other_row, other_column, _other_score in result
        ):
            continue
        result.append((row, column, score))
        if len(result) >= limit:
            break
    return tuple(result)


def _bounded_component_anchors(
    mask: np.ndarray,
    importance: np.ndarray,
    *,
    limit: int,
    minimum_size: int,
    maximum_size: int,
) -> tuple[tuple[int, int, int, int], ...]:
    labels, sizes = connected_components(mask)
    ranked = sorted(
        (
            (size, label)
            for label, size in enumerate(sizes, start=1)
            if minimum_size <= size <= maximum_size
        ),
        reverse=True,
    )[:limit]
    result: list[tuple[int, int, int, int]] = []
    for size, label in ranked:
        component = labels == label
        rows, columns = np.nonzero(component)
        values = np.asarray(importance[rows, columns], dtype=np.float64)
        low = float(np.min(values, initial=0.0))
        span = max(float(np.max(values, initial=1.0)) - low, 1.0e-12)
        normalized = np.clip((values - low) / span, 0.0, 1.0)
        depth = _interior_depth(component)[rows, columns]
        best = int(np.argmax(depth + 0.28 * normalized))
        result.append((int(rows[best]), int(columns[best]), int(size), int(label)))
    return tuple(result)


def _coverage_seeds(
    valid: np.ndarray,
    priority: np.ndarray,
    *,
    existing: tuple[tuple[int, int], ...] = (),
    count: int,
    minimum_distance: float,
) -> tuple[tuple[int, int], ...]:
    """Fill the largest remaining naming gaps on a wrapped world raster."""

    allowed = np.asarray(valid, dtype=bool)
    score = np.asarray(priority, dtype=np.float64)
    if allowed.shape != score.shape or allowed.ndim != 2:
        raise ValueError("coverage mask and priority must share a 2D shape")
    if count <= 0 or not np.any(allowed):
        return ()
    rows, columns = np.nonzero(allowed)
    width = allowed.shape[1]
    finite_score = np.where(np.isfinite(score[rows, columns]), score[rows, columns], 0.0)
    low = float(finite_score.min(initial=0.0))
    span = max(float(finite_score.max(initial=1.0)) - low, 1.0e-12)
    unit_score = np.clip((finite_score - low) / span, 0.0, 1.0)
    distance_squared = np.full(rows.shape, np.inf, dtype=np.float64)

    def admit(row: int, column: int) -> None:
        column_delta = np.abs(columns - int(column))
        column_delta = np.minimum(column_delta, width - column_delta)
        candidate_distance = (rows - int(row)) ** 2 + column_delta**2
        np.minimum(distance_squared, candidate_distance, out=distance_squared)

    for row, column in existing:
        admit(row, column)
    if not existing:
        first = int(np.argmax(unit_score))
        admit(int(rows[first]), int(columns[first]))
        selected = [(int(rows[first]), int(columns[first]))]
    else:
        selected = []
    occupied = set(existing)
    occupied.update(selected)
    while len(selected) < count:
        ranking = distance_squared * (0.88 + 0.12 * unit_score)
        order = np.argsort(-ranking, kind="stable")
        chosen_index = next(
            (
                int(index)
                for index in order
                if (int(rows[index]), int(columns[index])) not in occupied
            ),
            None,
        )
        if chosen_index is None:
            break
        if float(np.sqrt(distance_squared[chosen_index])) < minimum_distance:
            break
        cell = (int(rows[chosen_index]), int(columns[chosen_index]))
        selected.append(cell)
        occupied.add(cell)
        admit(*cell)
    return tuple(selected[:count])


def extract_geographic_features(
    grid: WorldGrid,
    thematic: ThematicLayers,
    cultures: CultureLayers,
    lexicon: NameLexicon,
) -> tuple[GeographicFeature, ...]:
    """Extract world, regional and close-zoom physical landmarks."""

    if thematic.land_potential.shape != grid.shape:
        raise ValueError("thematic layers must match WorldGrid shape")
    if cultures.language_id.shape != grid.shape:
        raise ValueError("culture layers must match WorldGrid shape")
    step = 1
    elevation = reduce_field(grid.elevation, step=step, mode="mean")
    potential = reduce_field(thematic.land_potential, step=step, mode="mean")
    land_fraction = reduce_field(grid.water == 0, step=step, mode="mean")
    water_fractions = np.stack(
        tuple(
            reduce_field(grid.water == code, step=step, mode="mean")
            for code in range(4)
        ),
        axis=0,
    )
    water_code = np.argmax(water_fractions, axis=0).astype(np.uint8)
    coarse_land = land_fraction >= 0.52
    coarse_habitable_land = (
        reduce_field(society_domain_mask(grid), step=step, mode="mean") >= 0.52
    )
    habitable_components, habitable_sizes = connected_components(coarse_habitable_land)
    island_maximum_size = max(4, int(coarse_habitable_land.size * 0.0025))
    mainland_identifiers = np.asarray(
        [
            identifier
            for identifier, size in enumerate(habitable_sizes, start=1)
            if size > island_maximum_size
        ],
        dtype=np.int32,
    )
    mainland_land = np.isin(habitable_components, mainland_identifiers)
    mainland_support = np.repeat(
        np.repeat(mainland_land, step, axis=0),
        step,
        axis=1,
    )[: grid.shape[0], : grid.shape[1]] & society_domain_mask(grid)
    island_support = society_domain_mask(grid) & ~mainland_support
    padded_land = np.pad(coarse_land, ((1, 1), (0, 0)), mode="constant")
    land_neighbors = (
        np.roll(coarse_land, 1, axis=1).astype(np.uint8)
        + np.roll(coarse_land, -1, axis=1).astype(np.uint8)
        + padded_land[:-2].astype(np.uint8)
        + padded_land[2:].astype(np.uint8)
        + np.roll(padded_land[:-2], 1, axis=1).astype(np.uint8)
        + np.roll(padded_land[:-2], -1, axis=1).astype(np.uint8)
        + np.roll(padded_land[2:], 1, axis=1).astype(np.uint8)
        + np.roll(padded_land[2:], -1, axis=1).astype(np.uint8)
    )
    open_ocean = reduce_field(grid.water == 1, step=step, mode="max").astype(bool)
    inland_sea = reduce_field(grid.water == 3, step=step, mode="max").astype(bool)
    maritime_water = open_ocean | inland_sea
    # The storage grid wraps at the antimeridian, but the review atlas has a
    # visible left/right cut.  Do not place a sea name across that cut: the
    # water remains continuous, only its label anchor is kept inside the page.
    sea_name_domain = np.ones_like(open_ocean, dtype=bool)
    seam_guard = max(3, open_ocean.shape[1] // 20)
    sea_name_domain[:, :seam_guard] = False
    sea_name_domain[:, -seam_guard:] = False
    left_right_land = np.roll(coarse_land, 1, axis=1) & np.roll(coarse_land, -1, axis=1)
    above_below_land = padded_land[:-2] & padded_land[2:]
    bay_mask = maritime_water & (land_neighbors >= 4)
    strait_mask = maritime_water & (left_right_land | above_below_land)
    smoothed_elevation = smooth_field(elevation, radius=2, passes=1)
    regional_elevation = smooth_field(elevation, radius=7, passes=2)
    gradient_row, gradient_column = np.gradient(smoothed_elevation)
    coarse_slope = np.hypot(gradient_row, gradient_column)
    roughness = smooth_field(
        np.abs(elevation - smoothed_elevation),
        radius=2,
        passes=1,
    )
    depression = regional_elevation - smoothed_elevation
    basin_mask = (
        mainland_land
        & (smoothed_elevation < 0.44)
        & (depression >= 0.026)
        & (coarse_slope < 0.055)
    )
    plateau_mask = (
        mainland_land
        & (smoothed_elevation >= 0.42)
        & (smoothed_elevation < 0.68)
        & (coarse_slope < 0.045)
        & (roughness < 0.052)
        & ~basin_mask
    )
    plain_mask = (
        mainland_land
        & (smoothed_elevation < 0.39)
        & (potential >= 0.46)
        & (coarse_slope < 0.038)
        & (roughness < 0.040)
        & ~basin_mask
    )
    language = cultures.language_id
    family_by_language = {
        item.identifier: item.name_family for item in cultures.languages
    }

    masks: tuple[tuple[str, np.ndarray, np.ndarray, int, int], ...] = (
        ("mountain", mainland_land & (elevation >= 0.56), elevation, 16, 3),
        ("lake", water_code == 2, np.ones_like(elevation), 18, 1),
        ("sea", open_ocean & sea_name_domain, _interior_depth(open_ocean), 3, 12),
        ("inland-sea", inland_sea, _interior_depth(inland_sea), 8, 2),
        ("bay", bay_mask, land_neighbors.astype(np.float64), 14, 1),
        ("strait", strait_mask, land_neighbors.astype(np.float64), 12, 1),
        ("plain", plain_mask, potential, 32, 5),
        ("plateau", plateau_mask, smoothed_elevation, 28, 4),
        ("basin", basin_mask, depression, 28, 4),
    )
    used: set[str] = set()
    result: list[GeographicFeature] = []
    ordinal_by_type: dict[str, int] = {}
    for feature_type, mask, importance, limit, minimum_size in masks:
        for coarse_row, coarse_column, size in _component_anchors(
            mask,
            importance,
            limit=limit,
            minimum_size=minimum_size,
        ):
            row = min(grid.shape[0] - 1, coarse_row * step + step // 2)
            column = min(grid.shape[1] - 1, coarse_column * step + step // 2)
            if feature_type in {"mountain", "plain", "plateau", "basin"}:
                support = mainland_support
                priority = grid.elevation if feature_type == "mountain" else None
            elif feature_type == "lake":
                support = grid.water == 2
                priority = None
            elif feature_type == "inland-sea":
                support = grid.water == 3
                priority = None
            else:
                support = np.isin(grid.water, (1, 3))
                priority = None
            row, column = _snap_anchor(
                support,
                row,
                column,
                radius=max(3, step * 2),
                priority=priority,
            )
            language_identifier = _nearest_language(language, row, column)
            name_family = family_by_language[language_identifier]
            ordinal = ordinal_by_type.get(feature_type, 0)
            ordinal_by_type[feature_type] = ordinal + 1
            name = generate_feature_name(
                lexicon,
                feature_type=feature_type,
                language_identifier=language_identifier,
                ordinal=ordinal,
                used=used,
                name_family=name_family,
            )
            if feature_type in {"bay", "strait"}:
                tier = "detail"
            elif feature_type in {"plain", "plateau", "basin"}:
                tier = (
                    "major"
                    if size >= max(minimum_size * 6, int(coarse_land.size * 0.0025))
                    else "secondary" if size >= minimum_size * 2 else "detail"
                )
            elif feature_type == "mountain":
                tier = "major" if size >= max(12, minimum_size * 5) else "secondary"
            elif feature_type == "lake":
                tier = "major" if size >= 12 else "secondary"
            else:
                tier = "major" if size >= max(minimum_size * 3, 8) else "secondary"
            result.append(
                GeographicFeature(
                    identifier=f"{feature_type}-{ordinal + 1:02d}",
                    feature_type=feature_type,
                    name=name,
                    row=row,
                    column=column,
                    language_identifier=language_identifier,
                    tier=tier,
                )
            )
    # One connected world ocean is not one useful geographic name.  Divide
    # its broad navigable expanses into spaced regional seas while allowing a
    # large basin behind a real strait to receive its own label.
    sea_depth = _interior_depth(maritime_water)
    sea_existing = tuple(
        (item.row // step, item.column // step)
        for item in result
        if item.feature_type in {"sea", "inland-sea"}
    )
    sea_domain = (
        maritime_water
        & sea_name_domain
        & (sea_depth >= 2.0)
        & (land_neighbors <= 4)
    )
    for coarse_row, coarse_column in _coverage_seeds(
        sea_domain,
        sea_depth,
        existing=sea_existing,
        count=10,
        minimum_distance=max(18.0, min(coarse_land.shape) / 8.5),
    ):
        row = min(grid.shape[0] - 1, coarse_row * step + step // 2)
        column = (coarse_column * step + step // 2) % grid.shape[1]
        row, column = _snap_anchor(
            np.isin(grid.water, (1, 3)),
            row,
            column,
            radius=max(3, step * 2),
        )
        language_identifier = _nearest_language(language, row, column)
        ordinal = ordinal_by_type.get("sea", 0)
        ordinal_by_type["sea"] = ordinal + 1
        result.append(
            GeographicFeature(
                identifier=f"sea-{ordinal + 1:02d}",
                feature_type="sea",
                name=generate_feature_name(
                    lexicon,
                    feature_type="sea",
                    language_identifier=language_identifier,
                    ordinal=ordinal,
                    used=used,
                    name_family=family_by_language[language_identifier],
                ),
                row=row,
                column=column,
                language_identifier=language_identifier,
                tier="secondary",
            )
        )
    named_river_basins: set[int] = set()
    for row, column, size, basin_identifier in _river_basin_anchors(
        grid,
        thematic.drainage_basin,
        limit=24,
        minimum_cells=3,
    ):
        if not bool(mainland_support[row, column]):
            continue
        language_identifier = _nearest_language(language, row, column)
        ordinal = ordinal_by_type.get("river", 0)
        ordinal_by_type["river"] = ordinal + 1
        named_river_basins.add(basin_identifier)
        result.append(
            GeographicFeature(
                identifier=f"river-{ordinal + 1:02d}",
                feature_type="river",
                name=generate_feature_name(
                    lexicon,
                    feature_type="river",
                    language_identifier=language_identifier,
                    ordinal=ordinal,
                    used=used,
                    name_family=family_by_language[language_identifier],
                ),
                row=row,
                column=column,
                language_identifier=language_identifier,
                tier="major" if size >= 48 else "secondary",
            )
        )
    island_anchors = _bounded_component_anchors(
        coarse_habitable_land,
        potential + 0.15 * elevation,
        limit=96,
        minimum_size=1,
        maximum_size=island_maximum_size,
    )
    # Archipelagos are overview-scale systems and may contain named large
    # islands.  Large members keep an individual close-zoom label, while the
    # group receives one shared label at world scale.
    large_island_cutoff = min(10, max(5, island_maximum_size // 20))
    cluster_radius = max(8.0, min(coarse_land.shape) / 28.0)
    parent = list(range(len(island_anchors)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, (first_row, first_column, _first_size, _first_label) in enumerate(island_anchors):
        for second in range(first + 1, len(island_anchors)):
            second_row, second_column, _second_size, _second_label = island_anchors[second]
            column_delta = abs(first_column - second_column)
            column_delta = min(column_delta, coarse_land.shape[1] - column_delta)
            if float(np.hypot(first_row - second_row, column_delta)) <= cluster_radius:
                union(first, second)
    clustered: dict[int, list[tuple[int, int, int, int]]] = {}
    for index, anchor in enumerate(island_anchors):
        clustered.setdefault(find(index), []).append(anchor)

    island_features = [
        ("island-group", members)
        for members in clustered.values()
        if len(members) >= 2
    ]
    island_features.extend(
        ("island", [item])
        for members in clustered.values()
        for item in members
        if len(members) == 1 or item[2] >= large_island_cutoff
    )
    island_features.sort(
        key=lambda item: (
            -sum(member[2] for member in item[1]),
            min(member[0] for member in item[1]),
            min(member[1] for member in item[1]),
        )
    )
    for feature_type, members in island_features[:48]:
        coarse_row, coarse_column, size, _label = max(
            members,
            key=lambda item: (item[2], -item[0], -item[1]),
        )
        total_size = sum(item[2] for item in members)
        row = min(grid.shape[0] - 1, coarse_row * step + step // 2)
        column = min(grid.shape[1] - 1, coarse_column * step + step // 2)
        row, column = _snap_anchor(
            island_support,
            row,
            column,
            radius=max(3, step * 3),
            priority=grid.elevation,
        )
        language_identifier = _nearest_language(language, row, column)
        name_family = family_by_language[language_identifier]
        ordinal = ordinal_by_type.get(feature_type, 0)
        ordinal_by_type[feature_type] = ordinal + 1
        result.append(
            GeographicFeature(
                identifier=f"{feature_type}-{ordinal + 1:02d}",
                feature_type=feature_type,
                name=generate_feature_name(
                    lexicon,
                    feature_type=feature_type,
                    language_identifier=language_identifier,
                    ordinal=ordinal,
                    used=used,
                    name_family=name_family,
                ),
                row=row,
                column=column,
                language_identifier=language_identifier,
                tier=(
                    "major"
                    if total_size >= 18
                    else "secondary" if total_size >= 5 else "detail"
                ),
            )
        )

    # The atlas overview above deliberately names broad systems.  A second,
    # finer pass supplies the tributaries, small lakes and named summits that
    # become useful only after zooming in.  They retain their own detail tier
    # instead of crowding the world view.
    detail_step = 1
    detail_elevation = reduce_field(grid.elevation, step=detail_step, mode="max")
    detail_land = reduce_field(mainland_support, step=detail_step, mode="max").astype(bool)
    detail_water = reduce_field(grid.water, step=detail_step, mode="max").astype(np.uint8)

    def add_named(
        feature_type: str,
        row: int,
        column: int,
        *,
        tier: str,
    ) -> None:
        row = min(grid.shape[0] - 1, max(0, int(row)))
        column = int(column) % grid.shape[1]
        language_identifier = _nearest_language(language, row, column)
        ordinal = ordinal_by_type.get(feature_type, 0)
        ordinal_by_type[feature_type] = ordinal + 1
        result.append(
            GeographicFeature(
                identifier=f"{feature_type}-{ordinal + 1:02d}",
                feature_type=feature_type,
                name=generate_feature_name(
                    lexicon,
                    feature_type=feature_type,
                    language_identifier=language_identifier,
                    ordinal=ordinal,
                    used=used,
                    name_family=family_by_language[language_identifier],
                ),
                row=row,
                column=column,
                language_identifier=language_identifier,
                tier=tier,
            )
        )

    peak_score = np.where(
        detail_land & (detail_elevation >= 0.54),
        detail_elevation,
        -np.inf,
    )
    peak_seeds = select_spaced_seeds(
        peak_score,
        np.isfinite(peak_score),
        count=20,
        minimum_distance=max(8.0, 34.0 / detail_step),
    )
    for coarse_row, coarse_column in peak_seeds:
        row_start = coarse_row * detail_step
        column_start = coarse_column * detail_step
        block = grid.elevation[
            row_start : min(grid.shape[0], row_start + detail_step),
            column_start : min(grid.shape[1], column_start + detail_step),
        ]
        local = int(np.argmax(block))
        local_row, local_column = np.unravel_index(local, block.shape)
        add_named(
            "peak",
            row_start + int(local_row),
            column_start + int(local_column),
            tier="detail",
        )

    existing_rivers = [
        (item.row, item.column) for item in result if item.feature_type == "river"
    ]
    local_rivers: list[tuple[int, int]] = []
    for row, column, _score in _minor_river_reach_anchors(
        grid,
        limit=72,
        drainage_basin=thematic.drainage_basin,
        named_basins=named_river_basins,
    ):
        if not bool(mainland_support[row, column]):
            continue
        if any(
            np.hypot(row - other_row, min(abs(column - other_column), grid.shape[1] - abs(column - other_column))) < 34.0
            for other_row, other_column in (*existing_rivers, *local_rivers)
        ):
            continue
        local_rivers.append((row, column))
        add_named("river", row, column, tier="detail")
        if len(local_rivers) >= 36:
            break

    existing_lakes = [
        (item.row, item.column) for item in result if item.feature_type == "lake"
    ]
    local_lakes: list[tuple[int, int]] = []
    for coarse_row, coarse_column, _size in _component_anchors(
        detail_water == 2,
        np.ones_like(detail_elevation),
        limit=72,
        minimum_size=1,
    ):
        row = coarse_row * detail_step + detail_step // 2
        column = coarse_column * detail_step + detail_step // 2
        row, column = _snap_anchor(
            grid.water == 2,
            row,
            column,
            radius=max(3, detail_step * 2),
        )
        if any(
            np.hypot(row - other_row, min(abs(column - other_column), grid.shape[1] - abs(column - other_column))) < 24.0
            for other_row, other_column in (*existing_lakes, *local_lakes)
        ):
            continue
        local_lakes.append((row, column))
        add_named("lake", row, column, tier="detail")
        if len(local_lakes) >= 16:
            break
    return tuple(result)
