"""Organized religions grown from holy cities over geography and travel."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from ..model import WorldGrid
from ..suitability import relative_land_slope
from ..thematic import ThematicLayers
from .model import (
    CultureLayers,
    PopulationLayers,
    Religion,
    ReligionLayers,
    Settlement,
    TransportLayers,
)
from .spatial import physical_transition_penalties, society_domain_mask
from .territorial_simulation import TerritorySeed, TerritorySimulation, simulate_territories


_TRADITIONS = (
    (
        "ancestral-rite",
        "祖仪",
        "现世秩序延续先民与家族的记忆",
        "守信、敬亲与履行共同体义务",
        "宗祠、礼官与地方祭社",
    ),
    (
        "river-mysteries",
        "河源圣礼",
        "生命随圣水循环并在河源获得更新",
        "节制取水、救济旅人与维护渡口",
        "河寺、渡祭司与巡河会",
    ),
    (
        "sky-law",
        "天律",
        "苍穹以恒常法则衡量誓言与统治",
        "诚实、勇毅与庇护弱者",
        "观星院、誓约师与季节大会",
    ),
    (
        "mountain-vow",
        "山庭圣约",
        "高峰连接尘世、祖灵与不可见的秩序",
        "忍耐、克己与款待朝圣者",
        "山院、修行团与关隘施舍所",
    ),
    (
        "tide-covenant",
        "潮汐神契",
        "潮汐往复见证生命、债务与归航",
        "守约、互助与敬畏风浪",
        "港祠、航海祭团与灯塔会",
    ),
    (
        "pilgrim-way",
        "行旅圣道",
        "世界由道路相连，善行使灵魂接近真理",
        "施舍、求知与保护远行者",
        "讲堂、驿院与巡礼兄弟会",
    ),
)


def _wrapped_distance(first: Settlement, second: Settlement, width: int) -> float:
    dx = abs(first.column - second.column)
    return math.hypot(first.row - second.row, min(dx, width - dx))


def _holy_city_candidates(
    settlements: tuple[Settlement, ...],
    cultures: CultureLayers,
    transport: TransportLayers,
    *,
    count: int,
    width: int,
) -> tuple[Settlement, ...]:
    eligible = [item for item in settlements if item.tier in {"metropolis", "city"}]
    if not eligible or count < 1:
        return ()
    maximum_population = max(item.population_max for item in eligible)

    def base_score(item: Settlement) -> float:
        return (
            0.40 * item.score
            + 0.34 * math.log1p(item.population_max) / math.log1p(maximum_population)
            + 0.26 * float(transport.accessibility[item.row, item.column])
        )

    # Start from distinct cultural hearths, then let distance and prosperity
    # choose the remaining pilgrimage centres. Religions may spread across
    # those cultural origins; the hearth list is not a copied culture map.
    best_by_civilization: dict[int, Settlement] = {}
    for item in eligible:
        civilization = int(cultures.civilization_id[item.row, item.column])
        previous = best_by_civilization.get(civilization)
        if civilization > 0 and (
            previous is None
            or (base_score(item), item.population_max, item.identifier)
            > (base_score(previous), previous.population_max, previous.identifier)
        ):
            best_by_civilization[civilization] = item
    pool = list(best_by_civilization.values())
    pool.extend(item for item in eligible if item not in pool)
    chosen: list[Settlement] = []
    while pool and len(chosen) < min(count, len(eligible)):
        selected = max(
            pool,
            key=lambda item: (
                base_score(item)
                + (
                    0.0
                    if not chosen
                    else 0.018
                    * min(_wrapped_distance(item, other, width) for other in chosen)
                ),
                item.population_max,
                item.identifier,
            ),
        )
        chosen.append(selected)
        pool = [item for item in pool if item.identifier != selected.identifier]
    return tuple(chosen)


def _tradition_for(
    settlement: Settlement,
    grid: WorldGrid,
    thematic: ThematicLayers,
) -> tuple[str, str, str, str, str]:
    row, column = settlement.row, settlement.column
    ocean = np.isin(grid.water, (1, 3))
    coastal = any(
        0 <= row + dy < grid.shape[0]
        and bool(ocean[row + dy, (column + dx) % grid.shape[1]])
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )
    precipitation = float(thematic.climate.annual_precipitation[row, column])
    if float(grid.elevation[row, column]) >= 0.58:
        return _TRADITIONS[3]
    if coastal or settlement.site_type in {"port", "island-port"}:
        return _TRADITIONS[4]
    if int(grid.river_order[row, column]) >= 2 or settlement.site_type == "river-city":
        return _TRADITIONS[1]
    if precipitation <= 0.27 or settlement.site_type == "oasis":
        return _TRADITIONS[2]
    if float(site_potential := thematic.land_potential[row, column]) < 0.38:
        return _TRADITIONS[5]
    return _TRADITIONS[0 if site_potential >= 0.58 else 5]


def derive_religions(
    grid: WorldGrid,
    thematic: ThematicLayers,
    population: PopulationLayers,
    settlements: tuple[Settlement, ...],
    cultures: CultureLayers,
    transport: TransportLayers,
    *,
    religion_count: int | None = None,
) -> tuple[ReligionLayers, tuple[Settlement, ...]]:
    """Diffuse organized faiths across barriers and along travel corridors."""

    shape = grid.shape
    if any(
        field.shape != shape
        for field in (
            thematic.land_potential,
            population.population_weight,
            cultures.civilization_id,
            transport.accessibility,
        )
    ):
        raise ValueError("religion inputs must match the WorldGrid shape")
    domain = society_domain_mask(grid)
    eligible_count = sum(item.tier in {"metropolis", "city"} for item in settlements)
    if religion_count is None:
        religion_count = max(3, int(round(len(cultures.civilizations) * 0.72)))
    religion_count = min(18, max(0, int(religion_count)), eligible_count)
    holy_cities = _holy_city_candidates(
        settlements,
        cultures,
        transport,
        count=religion_count,
        width=shape[1],
    )
    if not holy_cities:
        empty = np.where(domain, 0, -1).astype(np.int16)
        return ReligionLayers(religion_id=empty, religions=()), settlements

    transitions = physical_transition_penalties(
        grid.elevation,
        grid.river_order,
        land_mask=grid.water == 0,
    ) * np.float32(0.46)
    slope = relative_land_slope(grid.elevation, grid.water == 0)
    friction = np.clip(
        1.0
        + 4.2 * slope
        + 1.8 * np.clip(grid.elevation - 0.42, 0.0, 0.58)
        + 1.2 * (1.0 - np.clip(thematic.land_potential, 0.0, 1.0))
        - 0.72 * np.clip(transport.accessibility, 0.0, 1.0),
        0.44,
        None,
    ).astype(np.float32)
    simulation = simulate_territories(
        TerritorySimulation(
            valid=domain,
            friction=friction,
            transition_penalty=transitions.astype(np.float32),
            road_access=np.clip(transport.accessibility, 0.0, 1.0).astype(np.float32),
            bridge_edges=np.zeros((8, *shape), dtype=np.float32),
        ),
        tuple(
            TerritorySeed(
                row=item.row,
                column=item.column,
                owner=index,
                strength=float(np.clip(0.88 + 0.20 * item.score, 0.82, 1.24)),
            )
            for index, item in enumerate(holy_cities, start=1)
        ),
    )
    religion_id = simulation.owner.astype(np.int16)
    religion_id[~domain] = -1
    religions: list[Religion] = []
    for identifier, settlement in enumerate(holy_cities, start=1):
        key, suffix, worldview, moral_ideal, institution = _tradition_for(
            settlement, grid, thematic
        )
        religions.append(
            Religion(
                identifier=identifier,
                name=f"{settlement.name}{suffix}",
                tradition=key,
                holy_settlement_id=settlement.identifier,
                origin_civilization_identifier=int(
                    cultures.civilization_id[settlement.row, settlement.column]
                ),
                worldview=worldview,
                moral_ideal=moral_ideal,
                institution=institution,
            )
        )
    holy_by_settlement = {
        item.identifier: identifier
        for identifier, item in enumerate(holy_cities, start=1)
    }
    promoted = tuple(
        replace(
            item,
            holy_religion_identifier=holy_by_settlement.get(item.identifier),
        )
        for item in settlements
    )
    return ReligionLayers(religion_id=religion_id, religions=tuple(religions)), promoted


__all__ = ["derive_religions"]
