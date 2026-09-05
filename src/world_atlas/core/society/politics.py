"""Deterministic late-medieval state cores and control allocation."""

from __future__ import annotations

import math
from collections import Counter, deque

import numpy as np

from ..model import WorldGrid
from ..thematic import (
    BIOME_SAVANNA_DRY_GRASSLAND,
    BIOME_TEMPERATE_GRASSLAND,
    ThematicLayers,
)
from .model import (
    CultureLayers,
    NameLexicon,
    PoliticalEntity,
    PoliticalLayers,
    PopulationLayers,
    Settlement,
    State,
    TransportLayers,
)
from .institutions import (
    StateFormationProfile,
    derive_government_template,
    derive_state_formation_profiles,
    government_form_catalog,
)
from .frontiers import derive_frontier_groups, derive_stateless_frontier
from .onomastics import lineage_style_index
from .spatial import (
    connected_components,
    natural_compartment_ids,
    physical_transition_penalties,
    refine_partition_boundaries,
    reduce_field,
    snap_partition_to_natural_regions,
    society_domain_mask,
)
from .state_formation import (
    build_control_region_graph,
    repair_same_land_state_fragments,
    simulate_state_formation,
)
from .territorial_simulation import (
    TerritorySeed,
    TerritorySimulation,
    bridge_transition_discounts,
    fill_unassigned_territory,
    simulate_bounded_reach,
    simulate_territories,
)


_COAST_TOKENS = ("海", "港", "湾", "岛", "礁", "潮", "帆", "舟", "岬", "屿")
_STEPPE_TOKENS = ("汗", "帐", "部", "营", "牧", "马", "狼", "鹰", "毡")
_MOUNTAIN_TOKENS = ("山", "岭", "关", "矿", "岩", "炉", "谷", "崖", "寨")
_RIVER_TOKENS = ("河", "江", "川", "汊", "洲", "堤", "泽", "桥", "闸", "渠")


def _dominant(values: np.ndarray) -> int:
    positive = values[values > 0]
    if positive.size == 0:
        return 0
    return int(np.bincount(positive.astype(np.int64)).argmax())


def _name_affinity(
    name: str,
    *,
    coastal_share: float,
    steppe_share: float,
    mountain_share: float,
    river_share: float,
) -> float:
    """Score setting names against the actual geography of a generated state."""

    score = 0.0
    if any(token in name for token in _COAST_TOKENS):
        score += 3.0 * coastal_share - (8.0 if coastal_share < 0.005 else 0.0)
    if any(token in name for token in _STEPPE_TOKENS):
        score += 4.0 * steppe_share - (9.0 if steppe_share < 0.12 else 0.0)
    if any(token in name for token in _MOUNTAIN_TOKENS):
        score += 5.0 * mountain_share - (4.0 if mountain_share < 0.06 else 0.0)
    if any(token in name for token in _RIVER_TOKENS):
        score += 4.0 * river_share - (3.0 if river_share < 0.002 else 0.0)
    return score


def _derived_state_name(
    core: Settlement,
    identifier: int,
    profile: dict[str, float],
    style_index: int,
    used: set[str],
) -> str:
    """Return a proper country name without welding on a government title."""

    root = core.name
    if root not in used:
        used.add(root)
        return root
    qualifiers = (
        ("海", "岛", "潮", "湾")
        if profile["coastal_share"] >= 0.015
        else ("上", "岭", "山", "谷")
        if profile["mountain_share"] >= 0.10
        else ("东", "西", "南", "北", "内", "外")
    )
    start = (identifier + style_index) % len(qualifiers)
    for offset in range(len(qualifiers)):
        candidate = f"{qualifiers[(start + offset) % len(qualifiers)]}{root}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError(f"cannot derive a unique state name from {root}")


def _formal_state_name(
    proper_name: str,
    government_key: str,
    style_index: int,
    size_class: str,
    *,
    imperial: bool = False,
    paramount: bool = False,
) -> str:
    """Translate a regime into the country's own political naming register.

    ``State.name`` remains the stable proper-name root used by settlements and
    provinces.  This function supplies the public country title, so a regime
    never leaks a modern analytical label such as ``官僚君主制`` onto the map.
    """

    western_styles = {1, 2, 3, 9, 10}
    if paramount:
        return f"{proper_name}王朝"
    if imperial:
        suffix = (
            "大汗国"
            if style_index == 4 and government_key == "nomadic-confederacy"
            else "帝国"
        )
        return f"{proper_name}{suffix}"
    if government_key == "bureaucratic-monarchy":
        if style_index == 0:
            suffix = "王国"
        elif style_index == 4:
            suffix = "汗国"
        elif style_index == 5:
            suffix = "沙阿国"
        elif style_index == 6:
            suffix = "苏丹国"
        elif style_index == 7:
            suffix = "王国"
        elif style_index == 8:
            suffix = "苏丹国"
        else:
            suffix = "王国"
    elif government_key in {
        "court-monarchy",
        "hereditary-feudalism",
        "estate-monarchy",
    }:
        if style_index == 0:
            suffix = "王朝" if government_key == "court-monarchy" else "王国"
        elif style_index == 4:
            suffix = "汗国"
        elif style_index == 5:
            suffix = "沙阿国"
        elif style_index in {6, 8}:
            suffix = "苏丹国"
        elif style_index == 7:
            suffix = "王国"
        elif style_index in western_styles or style_index == 11:
            suffix = "王国"
        else:
            suffix = "王国"
    elif government_key == "maritime-republic":
        # Maritime orientation is an economic/geographic attribute, not part
        # of the constitutional rank.  Keep the public title legal and plain.
        suffix = "共和国"
    elif government_key == "city-republic":
        suffix = "自由市" if style_index in western_styles and size_class == "small" else "城邦共和国"
    elif government_key == "nomadic-confederacy":
        suffix = "汗国" if style_index == 4 else "部族邦联"
    elif government_key == "clan-league":
        suffix = {
            0: "宗盟",
            4: "部盟",
            7: "诸邦联盟",
            9: "氏族邦联",
            11: "部盟",
        }.get(style_index, "邦联")
    elif government_key == "tributary-chiefdom":
        suffix = {
            0: "藩国",
            7: "土邦",
            8: "岛邦",
        }.get(style_index, "酋邦")
    else:
        raise ValueError(f"unsupported government form: {government_key}")
    return f"{proper_name}{suffix}"


def _imperial_state_identifiers(
    identifiers: list[int],
    *,
    civilization_by_identifier: dict[int, int],
    civilization_style_by_identifier: dict[int, int],
    name_by_identifier: dict[int, str],
    government_key_by_identifier: dict[int, str],
    population_by_identifier: dict[int, float],
    profiles: dict[int, dict[str, float]],
    eastern_anchor_civilization: int | None,
) -> tuple[set[int], int | None]:
    """Select scarce imperial titles from people and productive capacity.

    Land area is deliberately absent.  Outside the eastern ritual system, a
    realm must lead its civilization and clear high global thresholds for
    population and productive output.  At most one realm per civilization can
    qualify, and only a globally scarce handful of the strongest qualifiers
    receive an imperial-grade title.
    """

    monarchical = {
        "bureaucratic-monarchy",
        "court-monarchy",
        "hereditary-feudalism",
        "estate-monarchy",
        "nomadic-confederacy",
    }
    paramount = next(
        (
            identifier
            for identifier in identifiers
            if name_by_identifier[identifier] == "中洛"
            and civilization_style_by_identifier[
                civilization_by_identifier[identifier]
            ]
            == 0
        ),
        None,
    )
    candidates = [
        identifier
        for identifier in identifiers
        if government_key_by_identifier[identifier] in monarchical
        and identifier != paramount
        and civilization_by_identifier[identifier] != eastern_anchor_civilization
    ]
    if not candidates:
        return set(), paramount
    populations = np.asarray(
        [population_by_identifier[identifier] for identifier in candidates],
        dtype=np.float64,
    )
    output_by_identifier = {
        identifier: population_by_identifier[identifier]
        * (0.42 + profiles[identifier].get("potential_mean", 0.0))
        * (1.0 + 0.08 * min(6.0, profiles[identifier].get("urban_count", 0.0)))
        for identifier in candidates
    }
    outputs = np.asarray(
        [output_by_identifier[identifier] for identifier in candidates],
        dtype=np.float64,
    )
    population_floor = float(np.quantile(populations, 0.82))
    output_floor = float(np.quantile(outputs, 0.86))
    by_civilization: dict[int, list[int]] = {}
    for identifier in candidates:
        by_civilization.setdefault(
            civilization_by_identifier[identifier], []
        ).append(identifier)
    qualifiers: list[int] = []
    for civilization_identifier, local in sorted(by_civilization.items()):
        best = max(
            local,
            key=lambda identifier: (
                output_by_identifier[identifier],
                population_by_identifier[identifier],
                profiles[identifier].get("potential_mean", 0.0),
                -identifier,
            ),
        )
        if (
            population_by_identifier[best] >= population_floor
            and output_by_identifier[best] >= output_floor
            and profiles[best].get("urban_count", 0.0) >= 2.0
        ):
            qualifiers.append(best)

    # Imperial dignity represents a singular trans-regional claim, not a size
    # adjective available independently to every civilization.  Keep roughly
    # one non-paramount emperor per forty-five states, with a hard world cap of
    # three; the eastern paramount title is counted separately.
    imperial_limit = max(1, min(3, round(len(identifiers) / 45)))
    qualifiers.sort(
        key=lambda identifier: (
            output_by_identifier[identifier],
            population_by_identifier[identifier],
            profiles[identifier].get("urban_count", 0.0),
            -identifier,
        ),
        reverse=True,
    )
    return set(qualifiers[:imperial_limit]), paramount


def _major_role_affinity(
    name: str,
    placement_role: str,
    profile: dict[str, float],
    *,
    population: float,
    centrality: float,
) -> float:
    """Match proper names by geographic role, never by government form."""

    score = _name_affinity(name, **{key: profile[key] for key in (
        "coastal_share", "steppe_share", "mountain_share", "river_share"
    )})
    potential = profile.get("potential_mean", 0.0)
    area = profile.get("area_share", 0.0)
    edge = profile.get("edge_share", 0.0)
    urban_support = min(1.0, profile.get("urban_count", 0.0) / 4.0)
    if placement_role == "ritual-core":
        return (
            score
            + 42.0 * profile.get("cultural_core", 0.0)
            + 28.0 * centrality
            + 6.0 * population
            + 3.0 * potential
            + 2.0 * profile["river_share"]
            + 12.0 * urban_support
        )
    if placement_role == "steppe":
        steppe_presence = min(1.0, profile["steppe_share"] / 0.35)
        return (
            score
            + 6.0 * steppe_presence
            + 8.0 * population
            + 2.0 * area
            + 1.5 * edge
            + 4.0 * urban_support
        )
    if placement_role == "coastal":
        coast_presence = min(1.0, profile["coastal_share"] / 0.02)
        return (
            score
            + 6.0 * coast_presence
            + 8.0 * population
            + 2.0 * centrality
            + 1.5 * area
            + 4.0 * urban_support
        )
    if placement_role == "highland":
        return (
            score
            + 8.0 * profile["mountain_share"]
            + 5.0 * population
            + 3.0 * centrality
            + 4.0 * urban_support
        )
    if placement_role == "river":
        return score + 7.0 * profile["river_share"] + 4.0 * centrality + 2.0 * population + 3.0 * urban_support
    return score + 3.5 * population + 2.5 * centrality + 1.5 * potential + 3.0 * urban_support


def _assign_major_family_names(
    identifiers: list[int],
    names: tuple[str, ...],
    placement_roles: tuple[str, ...],
    profiles: dict[int, dict[str, float]],
    population_by_state: dict[int, float],
    centrality_by_state: dict[int, float],
    used: set[str],
) -> dict[int, str]:
    """Solve the complete role-to-state match inside one coherent name register."""

    if len(names) != len(placement_roles):
        raise ValueError("major country names and placement roles must align")
    if len(identifiers) < len(names):
        raise ValueError("major realm candidates must cover every name")
    if not names:
        return {}
    populations = np.asarray([population_by_state[item] for item in identifiers], dtype=np.float64)
    low = float(populations.min())
    span = max(float(populations.max()) - low, 1.0e-12)
    normalized_population = {
        identifier: (population_by_state[identifier] - low) / span
        for identifier in identifiers
    }
    def affinity(name_index: int, identifier: int) -> float:
        return _major_role_affinity(
            names[name_index],
            placement_roles[name_index],
            profiles[identifier],
            population=normalized_population[identifier],
            centrality=centrality_by_state[identifier],
        )

    def specificity(name_index: int) -> tuple[int, int]:
        placement_role = placement_roles[name_index]
        priority = (
            6 if placement_role == "ritual-core" else
            5 if placement_role == "steppe" else
            4 if placement_role == "coastal" else
            3 if placement_role == "highland" else
            2 if placement_role == "river" else
            1
        )
        return (-priority, name_index)

    assignment: dict[int, int] = {}
    available = set(identifiers)
    for name_index in sorted(range(len(names)), key=specificity):
        selected = max(
            available,
            key=lambda identifier: (affinity(name_index, identifier), -identifier),
        )
        assignment[name_index] = selected
        available.remove(selected)

    # Greedy specificity keeps rare roles from being consumed by generic
    # names. Pairwise improvement then removes ordering artefacts in bounded
    # O(names^2) time, unlike the former exponential Cartesian search.
    for _pass in range(4):
        improved = False
        for first in range(len(names)):
            for second in range(first + 1, len(names)):
                first_state = assignment[first]
                second_state = assignment[second]
                current = affinity(first, first_state) + affinity(second, second_state)
                swapped = affinity(first, second_state) + affinity(second, first_state)
                if swapped > current + 1.0e-12:
                    assignment[first], assignment[second] = second_state, first_state
                    improved = True
        if not improved:
            break
    result = {
        assignment[name_index]: name
        for name_index, name in enumerate(names)
    }
    used.update(names)
    return result


def _major_name_register(name: str, placement_role: str) -> str:
    """Infer a linguistic register from proper names, not a culture template."""

    if name.startswith(("中洛", "东洛", "河社", "蜀宁", "江梁", "燕朔", "闽海", "越岭")):
        return "eastern-court"
    if placement_role == "steppe":
        return "steppe-court"
    if name.startswith(("诺玛", "罗赛尔")):
        return "maritime-league"
    if name.startswith(("维兰", "卡尔德")):
        return "saelic-court"
    if placement_role == "highland" or name.startswith("阿肯"):
        return "forge-court"
    return f"proper-{name}"


def _assign_coherent_major_names(
    identifiers: list[int],
    records: tuple[tuple[str, str], ...],
    profiles: dict[int, dict[str, float]],
    population_by_state: dict[int, float],
    centrality_by_state: dict[int, float],
    civilization_by_identifier: dict[int, int],
    civilization_style_by_identifier: dict[int, int],
    used: set[str],
    *,
    eastern_anchor_civilization: int | None,
) -> dict[int, str]:
    """Keep each retained historic name register in one generated civilization."""

    grouped: dict[str, list[tuple[str, str]]] = {}
    for name, placement_role in records:
        grouped.setdefault(_major_name_register(name, placement_role), []).append(
            (name, placement_role)
        )
    result: dict[int, str] = {}
    occupied_civilizations: set[int] = set()
    order = sorted(
        grouped,
        key=lambda register: (
            0 if register == "eastern-court" else 1,
            -len(grouped[register]),
            register,
        ),
    )
    for register in order:
        group = grouped[register]
        available = [identifier for identifier in identifiers if identifier not in result]
        by_civilization: dict[int, list[int]] = {}
        for identifier in available:
            by_civilization.setdefault(
                civilization_by_identifier[identifier], []
            ).append(identifier)
        viable_by_civilization = by_civilization
        register_style = {
            "eastern-court": 0,
            "steppe-court": 4,
            "maritime-league": 2,
            "saelic-court": 9,
            "forge-court": 5,
        }.get(register)
        styled = {
            civilization_identifier: state_ids
            for civilization_identifier, state_ids in viable_by_civilization.items()
            if register_style is not None
            and civilization_style_by_identifier.get(civilization_identifier)
            == register_style
        }
        register_pool = styled or viable_by_civilization
        candidates = {
            civilization_identifier: state_ids
            for civilization_identifier, state_ids in register_pool.items()
            if len(state_ids) >= len(group)
        }
        if not candidates:
            # Small fixtures or regional extracts may contain fewer states in
            # every civilization than the complete historic register. Retain
            # the coherent subset that fits instead of scattering the overflow.
            target_civilization, state_ids = max(
                register_pool.items(),
                key=lambda item: (len(item[1]), -item[0]),
            )
            group = group[: len(state_ids)]
            candidates = {target_civilization: state_ids}
        if (
            register == "eastern-court"
            and eastern_anchor_civilization in candidates
        ):
            target_civilization = int(eastern_anchor_civilization)
        else:
            fresh = {
                civilization_identifier: state_ids
                for civilization_identifier, state_ids in candidates.items()
                if civilization_identifier not in occupied_civilizations
            }
            pool = fresh or candidates

            def register_score(item: tuple[int, list[int]]) -> tuple[float, float, int]:
                civilization_identifier, state_ids = item
                local_profiles = [profiles[identifier] for identifier in state_ids]
                population = sum(population_by_state[identifier] for identifier in state_ids)
                if register == "steppe-court":
                    fit = max(profile["steppe_share"] for profile in local_profiles)
                elif register == "maritime-league":
                    fit = max(profile["coastal_share"] for profile in local_profiles)
                elif register == "forge-court":
                    fit = max(profile["mountain_share"] for profile in local_profiles)
                else:
                    fit = max(
                        0.52 * profile["potential_mean"]
                        + 0.28 * centrality_by_state[identifier]
                        + 0.20 * profile["river_share"]
                        for identifier, profile in zip(
                            state_ids, local_profiles, strict=True
                        )
                    )
                return fit, population, -civilization_identifier

            target_civilization = max(pool.items(), key=register_score)[0]
        occupied_civilizations.add(target_civilization)
        state_ids = candidates[target_civilization]
        result.update(
            _assign_major_family_names(
                state_ids,
                tuple(name for name, _realm_type in group),
                tuple(placement_role for _name, placement_role in group),
                profiles,
                population_by_state,
                centrality_by_state,
                used,
            )
        )
    return result


def _select_state_cores(
    candidates: list[Settlement],
    *,
    quota: int,
    region_cell_count: int,
    step: int,
    width: int,
    accessibility: np.ndarray,
    natural_compartment: np.ndarray,
) -> list[Settlement]:
    """Select cores across defensible compartments before duplicating a plain."""

    minimum_distance = max(1.0, math.sqrt(max(region_cell_count, 1) / max(quota, 1)) * 0.42)
    compartment_sizes = np.bincount(
        np.asarray(natural_compartment, dtype=np.int64).reshape(-1)
    )
    minimum_compartment_cells = max(
        32,
        min(
            96,
            int(round(max(region_cell_count, 1) / max(quota, 1) * 0.012)),
        ),
    )

    def compartment_size(settlement: Settlement) -> int:
        identifier = int(
            natural_compartment[settlement.row // step, settlement.column // step]
        )
        return (
            int(compartment_sizes[identifier])
            if 0 < identifier < len(compartment_sizes)
            else 0
        )

    selected: list[Settlement] = []
    deferred: list[Settlement] = []
    undersized: list[Settlement] = []
    coarse_width = int(math.ceil(width / step))
    ranked = sorted(
        candidates,
        key=lambda settlement: (
            -(
                settlement.score
                + 0.30 * float(accessibility[settlement.row, settlement.column])
                + (0.16 if settlement.tier == "metropolis" else 0.08 if settlement.tier == "city" else 0.0)
            ),
            settlement.identifier,
        ),
    )
    for settlement in ranked:
        if compartment_size(settlement) < minimum_compartment_cells:
            undersized.append(settlement)
            continue
        row = settlement.row // step
        column = settlement.column // step
        compartment = int(natural_compartment[row, column])
        if all(
            compartment != int(
                natural_compartment[other.row // step, other.column // step]
            )
            or math.hypot(
                    row - other.row // step,
                    min(
                        abs(column - other.column // step),
                        coarse_width - abs(column - other.column // step),
                    ),
                )
                >= minimum_distance
            for other in selected
        ):
            selected.append(settlement)
        else:
            deferred.append(settlement)
        if len(selected) == quota:
            return selected
    selected.extend(deferred[: quota - len(selected)])
    selected.extend(undersized[: quota - len(selected)])
    return selected


def _coarse_route_affinity(
    transport: TransportLayers,
    shape: tuple[int, int],
    *,
    step: int,
) -> np.ndarray:
    """Rasterize the durable overland network onto the political lattice."""

    height = int(math.ceil(shape[0] / step))
    width = int(math.ceil(shape[1] / step))
    affinity = np.zeros((height, width), dtype=np.float64)
    importance_weight = {"trunk": 1.0, "regional": 0.62, "local": 0.30}
    for route in transport.routes:
        # River routes describe travel *along* a channel, not a bridge across
        # it.  Treating them as generic crossings made long rivers unite both
        # banks instead of separating states.
        if route.mode != "road":
            continue
        strength = importance_weight[route.importance]
        for row, column in _route_cells(route.path, shape):
            coarse_row = min(height - 1, row // step)
            coarse_column = (column // step) % width
            affinity[coarse_row, coarse_column] = max(
                affinity[coarse_row, coarse_column],
                strength,
            )
    # A route is a corridor rather than a one-cell string.  One orthogonal
    # shoulder also keeps coarse sampling from breaking a pass or river road.
    padded = np.pad(affinity, ((1, 1), (0, 0)), mode="constant")
    shoulder = np.maximum.reduce(
        (
            padded[:-2],
            padded[2:],
            np.roll(affinity, 1, axis=1),
            np.roll(affinity, -1, axis=1),
        )
    )
    return np.maximum(affinity, shoulder * 0.54)


def _state_transition_penalties(
    elevation: np.ndarray,
    basin: np.ndarray,
    river_order: np.ndarray,
    language: np.ndarray,
    route_affinity: np.ndarray,
) -> np.ndarray:
    """Combine physical frontiers, cultural seams, and real crossing routes."""

    fields = (basin, river_order, language, route_affinity)
    if any(np.asarray(field).shape != np.asarray(elevation).shape for field in fields):
        raise ValueError("state transition fields must align")
    physical = physical_transition_penalties(
        elevation,
        basin,
        river_order,
    )
    # Keep the already-accepted river strength unchanged while giving ridges,
    # steep slope breaks and drainage divides more authority.  Scaling the
    # entire physical field would overfit every country to a whole river
    # system, so the river-only contribution is removed below before routes
    # and explicit bridges are applied.
    terrain_scale = 3.65
    river_scale = 2.75
    penalties = terrain_scale * physical
    directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    for direction, (dy, dx) in enumerate(directions):
        target_language = np.roll(language, shift=(-dy, -dx), axis=(0, 1))
        cultural_seam = (
            (language > 0)
            & (target_language > 0)
            & (language != target_language)
        )
        target_route = np.roll(route_affinity, shift=(-dy, -dx), axis=(0, 1))
        crossing_route = np.minimum(route_affinity, target_route)
        target_river = np.roll(river_order, shift=(-dy, -dx), axis=(0, 1))
        # Every river retained at this atlas scale is already a major feature.
        # Order-two reaches must therefore separate banks just as reliably as
        # the thickest lower course; only actual roads slightly soften them.
        major_bank = (river_order >= 2) ^ (target_river >= 2)
        atlas_major_bank = (river_order >= 3) ^ (target_river >= 3)
        river_strength = np.maximum(river_order, target_river).astype(np.float64)
        physical_river = (
            atlas_major_bank
            * (5.0 + 2.8 * np.clip(river_strength - 2.0, 0.0, 3.0))
            + major_bank * 1.8
        )
        penalties[direction] -= (terrain_scale - river_scale) * physical_river
        # Roads soften ridges and ordinary terrain.  A road near a river is
        # not automatically a ford: even demonstrated crossings only reduce
        # a small part of a major bank's defensive value.
        route_discount = np.clip(crossing_route, 0.0, 1.0)
        penalties[direction] *= np.where(
            major_bank,
            1.0 - 0.10 * route_discount,
            1.0 - 0.15 * route_discount,
        )
        penalties[direction] += major_bank * (
            11.0 + 2.2 * np.clip(river_strength - 2.0, 0.0, 3.0)
        )
        penalties[direction] += 8.0 * cultural_seam
        if dy < 0:
            penalties[direction, 0, :] = 0.0
        elif dy > 0:
            penalties[direction, -1, :] = 0.0
    return penalties


def _state_quotas(
    groups: dict[int, list[Settlement]],
    target: int,
    weights: dict[int, float],
    minimums: dict[int, int] | None = None,
    maximums: dict[int, int] | None = None,
) -> dict[int, int]:
    """Distribute states by demographic mass without starving a culture."""

    identifiers = sorted(identifier for identifier, values in groups.items() if values)
    if target < len(identifiers):
        identifiers = sorted(
            identifiers,
            key=lambda identifier: (-weights[identifier], identifier),
        )[:target]
    total_weight = max(1.0e-15, sum(weights[identifier] for identifier in identifiers))
    ideal = {
        identifier: target * weights[identifier] / total_weight
        for identifier in identifiers
    }
    minimums = minimums or {}
    maximums = maximums or {}
    quotas = {identifier: 1 for identifier in identifiers}
    while sum(quotas.values()) < target:
        available = [
            identifier
            for identifier in identifiers
            if quotas[identifier]
            < min(len(groups[identifier]), maximums.get(identifier, len(groups[identifier])))
        ]
        if not available:
            break
        selected = max(
            available,
            key=lambda identifier: (
                quotas[identifier] < min(
                    len(groups[identifier]),
                    int(minimums.get(identifier, 1)),
                ),
                (
                    min(len(groups[identifier]), int(minimums.get(identifier, 1)))
                    - quotas[identifier]
                )
                / max(1, int(minimums.get(identifier, 1))),
                ideal[identifier] - quotas[identifier],
                weights[identifier],
                len(groups[identifier]) - quotas[identifier],
                -identifier,
            ),
        )
        quotas[selected] += 1
    return quotas


def _route_cells(
    path: tuple[tuple[float, float], ...],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Rasterize a routed polyline with wrapped longitude."""

    height, width = shape
    cells: list[tuple[int, int]] = []
    for (start_column, start_row), (end_column, end_row) in zip(path, path[1:]):
        column_delta = end_column - start_column
        if column_delta > width * 0.5:
            column_delta -= width
        elif column_delta < -width * 0.5:
            column_delta += width
        row_delta = end_row - start_row
        steps = max(1, int(math.ceil(max(abs(column_delta), abs(row_delta)))))
        for offset in range(steps + 1):
            fraction = offset / steps
            row = int(round(start_row + row_delta * fraction - 0.5))
            column = int(round(start_column + column_delta * fraction - 0.5)) % width
            row = min(height - 1, max(0, row))
            cell = (row, column)
            if not cells or cells[-1] != cell:
                cells.append(cell)
    return tuple(cells)


def _settlement_protection_mask(
    settlements: tuple[Settlement, ...],
    shape: tuple[int, int],
    *,
    radius: int,
) -> np.ndarray:
    """Reserve a compact governed nucleus around every permanent settlement."""

    cells = np.zeros(shape, dtype=bool)
    for settlement in settlements:
        cells[settlement.row, settlement.column] = True
    protected = cells.copy()
    height, _width = shape
    for dy in range(-max(0, int(radius)), max(0, int(radius)) + 1):
        source_start = max(0, -dy)
        source_end = min(height, height - dy)
        target_start = source_start + dy
        target_end = source_end + dy
        for dx in range(-max(0, int(radius)), max(0, int(radius)) + 1):
            protected[target_start:target_end] |= np.roll(
                cells[source_start:source_end],
                dx,
                axis=1,
            )
    return protected


def _intra_state_route_protection(
    labels: np.ndarray,
    settlements: tuple[Settlement, ...],
    transport: TransportLayers,
    *,
    radius: int,
) -> np.ndarray:
    """Keep same-state road corridors from being erased as frontier."""

    state_labels = np.asarray(labels, dtype=np.int16)
    settlement_by_identifier = {item.identifier: item for item in settlements}
    state_by_settlement = {
        identifier: int(state_labels[item.row, item.column])
        for identifier, item in settlement_by_identifier.items()
    }
    route_cells = np.zeros(state_labels.shape, dtype=bool)
    radius = max(0, int(radius))
    for route in transport.routes:
        if route.mode != "road" or route.target_settlement_id is None:
            continue
        source_state = state_by_settlement.get(route.source_settlement_id, 0)
        target_state = state_by_settlement.get(route.target_settlement_id, 0)
        if source_state <= 0 or source_state != target_state:
            continue
        for row, column in _route_cells(route.path, state_labels.shape):
            if state_labels[row, column] == source_state:
                route_cells[row, column] = True
    protected = route_cells
    for _distance in range(radius):
        padded = np.pad(protected, ((1, 1), (0, 0)), mode="constant")
        protected = (
            protected
            | np.roll(protected, 1, axis=1)
            | np.roll(protected, -1, axis=1)
            | padded[:-2]
            | padded[2:]
        )
    return protected & (state_labels > 0)


def _smooth_state_boundaries(
    labels: np.ndarray,
    civilization: np.ndarray,
    state_civilization: np.ndarray,
    protected: np.ndarray,
    *,
    passes: int = 2,
) -> np.ndarray:
    """Remove one-cell teeth without crossing civilization boundaries."""

    result = np.asarray(labels, dtype=np.int16).copy()
    civilization_ids = np.asarray(civilization, dtype=np.int16)
    protected_cells = np.asarray(protected, dtype=bool)
    if result.shape != civilization_ids.shape or protected_cells.shape != result.shape:
        raise ValueError("state smoothing fields must align")
    for _pass in range(max(0, int(passes))):
        north = np.zeros_like(result)
        south = np.zeros_like(result)
        north[1:] = result[:-1]
        south[:-1] = result[1:]
        west = np.roll(result, 1, axis=1)
        east = np.roll(result, -1, axis=1)
        vertical = (north == south) & ((north == east) | (north == west))
        horizontal = (east == west) & ((east == north) | (east == south))
        candidate = np.where(vertical, north, np.where(horizontal, east, 0)).astype(np.int16)
        candidate_civilization = state_civilization[
            np.clip(candidate, 0, len(state_civilization) - 1)
        ]
        change = (
            ~protected_cells
            & (candidate > 0)
            & (candidate != result)
            & (candidate_civilization == civilization_ids)
        )
        if not np.any(change):
            break
        result[change] = candidate[change]
    return result


def _naturalize_straight_state_boundaries(
    labels: np.ndarray,
    elevation: np.ndarray,
    basin: np.ndarray,
    river_order: np.ndarray,
    civilization: np.ndarray,
    state_civilization: np.ndarray,
    protected: np.ndarray,
    *,
    minimum_run: int = 12,
    search_radius: int = 5,
) -> np.ndarray:
    """Replace long surveyed-looking borders with nearby physical seams.

    A dynamic program searches a narrow corridor around each long horizontal
    or vertical state edge.  Drainage divides, river banks, local relief, and
    ridge height lower the path cost; curvature and displacement raise it.
    Endpoints stay fixed, urban cells are immutable, and a state can only move
    into cells of its own civilization.
    """

    result = np.asarray(labels, dtype=np.int16).copy()
    heights = np.asarray(elevation, dtype=np.float64)
    basins = np.asarray(basin)
    rivers = np.asarray(river_order)
    civilizations = np.asarray(civilization, dtype=np.int16)
    protected_cells = np.asarray(protected, dtype=bool)
    fields = (heights, basins, rivers, civilizations, protected_cells)
    if any(field.shape != result.shape for field in fields):
        raise ValueError("straight-boundary geography must align with states")
    minimum_run = max(4, int(minimum_run))
    search_radius = max(1, int(search_radius))

    def naturalize_horizontal(
        active: np.ndarray,
        local_heights: np.ndarray,
        local_basins: np.ndarray,
        local_rivers: np.ndarray,
        local_civilizations: np.ndarray,
        local_protected: np.ndarray,
    ) -> np.ndarray:
        output = active.copy()
        height, width = output.shape
        source = active.copy()
        for boundary_row in range(height - 1):
            upper = source[boundary_row]
            lower = source[boundary_row + 1]
            boundary = (upper > 0) & (lower > 0) & (upper != lower)
            column = 0
            while column < width:
                if not boundary[column]:
                    column += 1
                    continue
                upper_identifier = int(upper[column])
                lower_identifier = int(lower[column])
                end = column + 1
                while (
                    end < width
                    and boundary[end]
                    and int(upper[end]) == upper_identifier
                    and int(lower[end]) == lower_identifier
                ):
                    end += 1
                run_length = end - column
                if run_length < minimum_run:
                    column = end
                    continue
                owner_upper = int(state_civilization[upper_identifier])
                owner_lower = int(state_civilization[lower_identifier])
                candidates = np.arange(
                    max(0, boundary_row - search_radius),
                    min(height - 2, boundary_row + search_radius) + 1,
                    dtype=np.int32,
                )
                count = len(candidates)
                costs = np.full((run_length, count), np.inf, dtype=np.float64)
                parent = np.full((run_length, count), -1, dtype=np.int16)
                straight_index = int(np.flatnonzero(candidates == boundary_row)[0])

                for offset in range(run_length):
                    active_column = column + offset
                    for candidate_index, candidate_row_value in enumerate(candidates):
                        candidate_row = int(candidate_row_value)
                        low = min(boundary_row, candidate_row) + 1
                        high = max(boundary_row, candidate_row) + 1
                        crossed = slice(low, high)
                        if candidate_row > boundary_row:
                            admissible = np.all(
                                (source[crossed, active_column] == lower_identifier)
                                & (local_civilizations[crossed, active_column] == owner_upper)
                                & ~local_protected[crossed, active_column]
                            )
                        elif candidate_row < boundary_row:
                            admissible = np.all(
                                (source[crossed, active_column] == upper_identifier)
                                & (local_civilizations[crossed, active_column] == owner_lower)
                                & ~local_protected[crossed, active_column]
                            )
                        else:
                            admissible = True
                        if not admissible:
                            continue
                        first_height = local_heights[candidate_row, active_column]
                        second_height = local_heights[candidate_row + 1, active_column]
                        relief = abs(second_height - first_height)
                        divide = (
                            local_basins[candidate_row, active_column] >= 0
                            and local_basins[candidate_row + 1, active_column] >= 0
                            and local_basins[candidate_row, active_column]
                            != local_basins[candidate_row + 1, active_column]
                        )
                        river_bank = (
                            local_rivers[candidate_row, active_column] >= 2
                        ) != (
                            local_rivers[candidate_row + 1, active_column] >= 2
                        )
                        ridge_height = max(first_height, second_height)
                        natural_strength = (
                            9.0 * float(divide)
                            + 8.0 * float(river_bank)
                            + 52.0 * relief
                            + 10.0 * max(0.0, ridge_height - 0.34)
                        )
                        displacement = candidate_row - boundary_row
                        # Natural seams dominate.  On a genuinely open plain,
                        # a slow two-frequency target prevents a surveyed
                        # axis-aligned cut without adding pixel noise or
                        # changing the broad political result.
                        plain_target = boundary_row + float(
                            np.clip(
                                1.55
                                * math.sin(active_column * 0.31 + boundary_row * 0.17)
                                + 0.65
                                * math.sin(active_column * 0.13 - boundary_row * 0.29),
                                -search_radius + 1,
                                search_radius - 1,
                            )
                        )
                        plain_deviation = candidate_row - plain_target
                        local_cost = (
                            0.018 * displacement * displacement
                            + 0.060 * plain_deviation * plain_deviation
                            - natural_strength
                        )
                        if offset == 0 or offset == run_length - 1:
                            if candidate_row != boundary_row:
                                continue
                        if offset == 0:
                            costs[offset, candidate_index] = local_cost
                            continue
                        previous_start = max(0, candidate_index - 1)
                        previous_end = min(count, candidate_index + 2)
                        previous_costs = costs[offset - 1, previous_start:previous_end]
                        if not np.any(np.isfinite(previous_costs)):
                            continue
                        transition_costs = previous_costs + 0.22 * np.abs(
                            candidates[previous_start:previous_end] - candidate_row
                        )
                        local_parent = int(np.argmin(transition_costs))
                        costs[offset, candidate_index] = (
                            float(transition_costs[local_parent]) + local_cost
                        )
                        parent[offset, candidate_index] = previous_start + local_parent

                if not np.isfinite(costs[-1, straight_index]):
                    column = end
                    continue
                path = np.empty(run_length, dtype=np.int32)
                active_index = straight_index
                complete_path = True
                for offset in range(run_length - 1, -1, -1):
                    path[offset] = candidates[active_index]
                    if offset:
                        active_index = int(parent[offset, active_index])
                        if active_index < 0:
                            complete_path = False
                            break
                if not complete_path:
                    column = end
                    continue
                if np.all(path == boundary_row):
                    column = end
                    continue
                for offset, candidate_row_value in enumerate(path):
                    active_column = column + offset
                    candidate_row = int(candidate_row_value)
                    if candidate_row > boundary_row:
                        output[
                            boundary_row + 1 : candidate_row + 1,
                            active_column,
                        ] = upper_identifier
                    elif candidate_row < boundary_row:
                        output[
                            candidate_row + 1 : boundary_row + 1,
                            active_column,
                        ] = lower_identifier
                column = end
        return output

    result = naturalize_horizontal(
        result,
        heights,
        basins,
        rivers,
        civilizations,
        protected_cells,
    )
    result = naturalize_horizontal(
        result.T,
        heights.T,
        basins.T,
        rivers.T,
        civilizations.T,
        protected_cells.T,
    ).T
    return result


def _absorb_tiny_state_fragments(
    labels: np.ndarray,
    protected: np.ndarray,
    *,
    minimum_size: int,
    land_components: np.ndarray | None = None,
    anchors: np.ndarray | None = None,
) -> np.ndarray:
    """Reassign detached land while preserving anchored overseas possessions."""

    result = np.asarray(labels, dtype=np.int16).copy()
    protected_cells = np.asarray(protected, dtype=bool)
    if result.shape != protected_cells.shape:
        raise ValueError("state fragment fields must align")
    anchor_cells = (
        protected_cells
        if anchors is None
        else np.asarray(anchors, dtype=bool)
    )
    if anchor_cells.shape != result.shape:
        raise ValueError("state fragment anchors must align")
    land_component_labels = (
        None
        if land_components is None
        else np.asarray(land_components, dtype=np.int32)
    )
    if land_component_labels is not None and land_component_labels.shape != result.shape:
        raise ValueError("land components must align with state labels")
    visited = np.zeros(result.shape, dtype=bool)
    height, width = result.shape
    components_by_state: dict[
        int,
        list[tuple[list[tuple[int, int]], Counter[int], bool, bool]],
    ] = {}
    for start_row, start_column in zip(*np.nonzero(result > 0), strict=True):
        if visited[start_row, start_column]:
            continue
        identifier = int(result[start_row, start_column])
        queue: deque[tuple[int, int]] = deque([(int(start_row), int(start_column))])
        visited[start_row, start_column] = True
        cells: list[tuple[int, int]] = []
        boundary: Counter[int] = Counter()
        contains_core = False
        contains_anchor = False
        while queue:
            row, column = queue.popleft()
            cells.append((row, column))
            contains_core |= bool(protected_cells[row, column])
            contains_anchor |= bool(anchor_cells[row, column])
            for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                next_row = row + dy
                if next_row < 0 or next_row >= height:
                    continue
                next_column = (column + dx) % width
                neighbor = int(result[next_row, next_column])
                if neighbor == identifier and not visited[next_row, next_column]:
                    visited[next_row, next_column] = True
                    queue.append((next_row, next_column))
                elif neighbor > 0 and neighbor != identifier:
                    boundary[neighbor] += 1
        components_by_state.setdefault(identifier, []).append(
            (cells, boundary, contains_core, contains_anchor)
        )
    for components in components_by_state.values():
        main = max(
            components,
            key=lambda component: (component[2], component[3], len(component[0])),
        )
        main_land_component = (
            int(land_component_labels[main[0][0]])
            if land_component_labels is not None
            else 0
        )
        relative_minimum = max(minimum_size, int(math.ceil(len(main[0]) * 0.01)))
        for component in components:
            cells, boundary, contains_core, contains_anchor = component
            same_landmass_detachment = (
                land_component_labels is not None
                and int(land_component_labels[cells[0]]) == main_land_component
            )
            if component is main or contains_core:
                continue
            if land_component_labels is not None and not same_landmass_detachment:
                # An overseas component needs a real settlement anchor.  Size
                # or geometric proximity alone does not create governability.
                if contains_anchor:
                    continue
                replacement = 0
            else:
                if len(cells) >= relative_minimum and not same_landmass_detachment:
                    continue
                replacement = (
                    min(
                        boundary,
                        key=lambda candidate: (-boundary[candidate], candidate),
                    )
                    if boundary
                    else 0
                )
            rows, columns = zip(*cells, strict=True)
            result[rows, columns] = replacement
    return result


def _territory_compactness(region: np.ndarray) -> float:
    """Measure whether a territory is a body rather than a long corridor.

    The score combines raster isoperimetric compactness with area relative to
    the territory's longest extent.  It is descriptive rather than a rectangle
    preference: river and ridge borders may remain irregular, while a
    many-cell-long ribbon scores poorly regardless of orientation.
    """

    mask = np.asarray(region, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("territory compactness expects a two-dimensional mask")
    rows, columns = np.nonzero(mask)
    area = int(rows.size)
    if area == 0:
        return 0.0
    height, width = mask.shape
    north = np.zeros_like(mask)
    south = np.zeros_like(mask)
    north[1:] = mask[:-1]
    south[:-1] = mask[1:]
    perimeter = int(
        np.count_nonzero(mask & ~north)
        + np.count_nonzero(mask & ~south)
        + np.count_nonzero(mask & ~np.roll(mask, 1, axis=1))
        + np.count_nonzero(mask & ~np.roll(mask, -1, axis=1))
    )
    row_span = int(rows.max() - rows.min() + 1)
    occupied_columns = np.unique(columns)
    if occupied_columns.size == 1:
        column_span = 1
    else:
        gaps = np.diff(
            np.concatenate((occupied_columns, occupied_columns[:1] + width))
        )
        column_span = int(width - int(gaps.max()) + 1)
    extent_fill = area / max(1, max(row_span, column_span) ** 2)
    isoperimetric = 4.0 * math.pi * area / max(1, perimeter * perimeter)
    return float(np.clip(math.sqrt(extent_fill * isoperimetric), 0.0, 1.0))


def _consolidate_corridor_states(
    labels: np.ndarray,
    state_civilization: np.ndarray,
    settlements: tuple[Settlement, ...],
    land_mask: np.ndarray,
    *,
    compactness_limit: float = 0.18,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Merge tiny ribbon states into a contiguous cultural neighbour.

    A one-city polity can be small when it is spatially compact.  What is not
    credible is a small country whose territory is mostly a long, narrow
    corridor.  Such allocations are merged across their longest same-culture
    border; isolated island polities and larger irregular countries survive.
    """

    source = np.asarray(labels, dtype=np.int16)
    civilizations = np.asarray(state_civilization, dtype=np.int16)
    land = np.asarray(land_mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("corridor-state consolidation expects two-dimensional labels")
    if land.shape != source.shape:
        raise ValueError("corridor-state consolidation requires an aligned land mask")
    identifiers = [int(value) for value in np.unique(source) if value > 0]
    if not identifiers:
        return source.copy(), ()
    if max(identifiers) >= len(civilizations):
        raise ValueError("state civilization lookup does not cover labels")
    areas = {
        identifier: int(np.count_nonzero(source == identifier))
        for identifier in identifiers
    }
    median_area = float(np.median(tuple(areas.values())))
    area_limit = max(48, int(round(median_area * 0.58)))
    sparse_corridor_limit = max(area_limit, int(round(median_area * 1.05)))
    urban_counts = Counter(
        int(source[item.row, item.column])
        for item in settlements
        if item.tier != "site" and int(source[item.row, item.column]) > 0
    )
    result = source.copy()
    for identifier in sorted(identifiers, key=lambda item: (areas[item], item)):
        region = result == identifier
        area = int(np.count_nonzero(region))
        compactness = _territory_compactness(region)
        sparse_corridor = (
            urban_counts[identifier] <= 1
            and compactness < 0.12
            and area <= sparse_corridor_limit
        )
        severe_corridor = (
            urban_counts[identifier] <= 4
            and compactness < 0.09
            and area <= sparse_corridor_limit
        )
        if area == 0 or compactness >= compactness_limit:
            continue
        if area > area_limit and not sparse_corridor and not severe_corridor:
            continue
        if urban_counts[identifier] > 2 and not severe_corridor:
            continue
        contacts: Counter[int] = Counter()
        for neighbor in (
            np.roll(result, 1, axis=1),
            np.roll(result, -1, axis=1),
            np.vstack((np.zeros((1, result.shape[1]), dtype=result.dtype), result[:-1])),
            np.vstack((result[1:], np.zeros((1, result.shape[1]), dtype=result.dtype))),
        ):
            boundary = region & (neighbor > 0) & (neighbor != identifier)
            for target in np.unique(neighbor[boundary]):
                target_identifier = int(target)
                if civilizations[target_identifier] == civilizations[identifier]:
                    contacts[target_identifier] += int(
                        np.count_nonzero(boundary & (neighbor == target_identifier))
                    )
        if not contacts:
            # Removing a polity here creates an unexplained inland vacuum.
            # Without a culturally compatible neighbour there is no honest
            # consolidation target, so retain it and let the civilization
            # partition determine its outer boundary.
            continue
        target = min(
            contacts,
            key=lambda item: (-contacts[item], -areas.get(item, 0), item),
        )
        result[region] = target
        areas[target] = areas.get(target, 0) + area
        urban_counts[target] += urban_counts[identifier]

    kept = tuple(int(value) for value in np.unique(result) if value > 0)
    remap = np.zeros(max(kept, default=0) + 1, dtype=np.int16)
    for new_identifier, old_identifier in enumerate(kept, start=1):
        remap[old_identifier] = new_identifier
    positive = result > 0
    result[positive] = remap[result[positive]]
    return result, kept


def _attach_adjacent_frontier_settlements(
    labels: np.ndarray,
    civilization_id: np.ndarray,
    state_civilization: np.ndarray,
    settlements: tuple[Settlement, ...],
) -> np.ndarray:
    """Attach a frontier city only when existing territory reaches its cell."""

    source = np.asarray(labels, dtype=np.int16)
    cultures = np.asarray(civilization_id, dtype=np.int16)
    state_cultures = np.asarray(state_civilization, dtype=np.int16)
    if source.shape != cultures.shape:
        raise ValueError("frontier settlement fields must align")
    result = source.copy()
    height, width = source.shape
    for settlement in settlements:
        row, column = settlement.row, settlement.column
        if int(source[row, column]) > 0:
            continue
        local_civilization = int(cultures[row, column])
        contacts: Counter[int] = Counter()
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            next_row = row + dy
            if next_row < 0 or next_row >= height:
                continue
            next_column = (column + dx) % width
            candidate = int(source[next_row, next_column])
            if (
                candidate > 0
                and candidate < len(state_cultures)
                and int(state_cultures[candidate]) == local_civilization
            ):
                contacts[candidate] += 1
        if contacts:
            result[row, column] = min(
                contacts,
                key=lambda identifier: (-contacts[identifier], identifier),
            )
    return result


def _automatic_state_count(urban_count: int) -> int:
    """Reserve enough cores for real island, valley, port, and pass polities."""

    return int(np.clip(round(max(0, int(urban_count)) * 0.30), 88, 144))


def derive_politics(
    grid: WorldGrid,
    thematic: ThematicLayers,
    population: PopulationLayers,
    settlements: tuple[Settlement, ...],
    cultures: CultureLayers,
    transport: TransportLayers,
    lexicon: NameLexicon,
    *,
    state_count: int | None = None,
    frontier_target_share: float = 0.35,
    state_formation_profiles: dict[int, StateFormationProfile] | None = None,
) -> PoliticalLayers:
    """Grow geography-bounded states from civilization urban networks."""

    if not settlements:
        raise ValueError("political generation requires settlements")
    if state_formation_profiles is None:
        state_formation_profiles = derive_state_formation_profiles(
            grid,
            thematic,
            population,
            settlements,
            cultures,
            transport,
        )
    if state_count is None:
        urban_count = sum(item.tier != "site" for item in settlements)
        # The core budget is large enough for defensible island, valley,
        # river-mouth and pass polities, while spatial separation and later
        # compactness checks still prevent random plain specks and ribbons.
        state_count = _automatic_state_count(urban_count)
    state_count = min(int(state_count), len(settlements))
    if state_count < 1:
        raise ValueError("state_count must be positive")
    step = 1
    physical_land = grid.water == 0
    land = society_domain_mask(grid)
    coarse_land = reduce_field(land, step=step, mode="max").astype(bool)
    sample_rows = np.minimum(
        np.arange(coarse_land.shape[0], dtype=np.int64) * step + step // 2,
        grid.shape[0] - 1,
    )
    sample_columns = np.minimum(
        np.arange(coarse_land.shape[1], dtype=np.int64) * step + step // 2,
        grid.shape[1] - 1,
    )
    coarse_civilization = cultures.civilization_id[
        np.ix_(sample_rows, sample_columns)
    ].astype(np.int16)
    civilization_core_by_identifier = {
        item.identifier: item.core_settlement_id for item in cultures.civilizations
    }
    major_records = tuple(
        zip(
            lexicon.major_country_names,
            lexicon.major_country_roles,
            strict=True,
        )
    )
    settlements_by_civilization: dict[int, list[Settlement]] = {}
    occupied: set[tuple[int, int]] = set()
    for settlement in sorted(settlements, key=lambda item: (-item.score, item.identifier)):
        if settlement.tier == "site":
            continue
        key = (settlement.row // step, settlement.column // step)
        civilization_identifier = int(
            cultures.civilization_id[settlement.row, settlement.column]
        )
        if civilization_identifier <= 0 or key in occupied:
            continue
        occupied.add(key)
        settlements_by_civilization.setdefault(civilization_identifier, []).append(
            settlement
        )
    eligible_count = sum(len(values) for values in settlements_by_civilization.values())
    state_count = min(state_count, eligible_count)
    civilization_population = {
        identifier: float(
            np.sum(
                population.population_weight[
                    cultures.civilization_id == identifier
                ],
                dtype=np.float64,
            )
        )
        for identifier in settlements_by_civilization
    }
    civilization_urban = {
        identifier: float(sum(item.population_max for item in values))
        for identifier, values in settlements_by_civilization.items()
    }
    civilization_access = {
        identifier: float(
            np.sum(
                population.population_weight[cultures.civilization_id == identifier]
                * transport.accessibility[cultures.civilization_id == identifier],
                dtype=np.float64,
            )
        )
        for identifier in settlements_by_civilization
    }

    def shares(values: dict[int, float]) -> dict[int, float]:
        total = max(1.0e-15, sum(values.values()))
        return {identifier: value / total for identifier, value in values.items()}

    population_share = shares(civilization_population)
    urban_share = shares(civilization_urban)
    access_share = shares(civilization_access)
    quota_weights = {
        identifier: (
            0.46 * population_share[identifier]
            + 0.36 * urban_share[identifier]
            + 0.18 * access_share[identifier]
        )
        * state_formation_profiles[identifier].quota_multiplier
        for identifier in settlements_by_civilization
    }
    quotas = _state_quotas(
        settlements_by_civilization,
        state_count,
        quota_weights,
        {
            identifier: state_formation_profiles[identifier].minimum_states
            for identifier in settlements_by_civilization
        },
        {
            identifier: state_formation_profiles[identifier].maximum_states
            for identifier in settlements_by_civilization
        },
    )
    native_route_affinity = _coarse_route_affinity(
        transport,
        grid.shape,
        step=1,
    ).astype(np.float32)
    native_transitions = _state_transition_penalties(
        grid.elevation,
        np.where(
            thematic.drainage_basin >= 0,
            thematic.drainage_basin.astype(np.int64) + 1,
            0,
        ),
        grid.river_order,
        cultures.language_id,
        native_route_affinity,
    )
    elevation = reduce_field(grid.elevation, step=step, mode="max")
    potential = reduce_field(thematic.land_potential, step=step, mode="mean")
    dry = reduce_field(
        thematic.climate.annual_precipitation < 0.05,
        step=step,
        mode="max",
    ).astype(bool)
    river = reduce_field(grid.river_order > 0, step=step, mode="max").astype(bool)
    population_support = reduce_field(population.population_weight, step=step, mode="max")
    accessibility = reduce_field(transport.accessibility, step=step, mode="mean")
    population_scale = population_support / max(
        float(population_support.max(initial=1.0)),
        1.0e-15,
    )
    friction = (
        1.0
        + 7.2 * np.square(np.clip(elevation, 0.0, 1.0))
        + 18.0 * np.hypot(
            np.gradient(elevation, axis=0),
            (np.roll(elevation, -1, axis=1) - np.roll(elevation, 1, axis=1)) * 0.5,
        )
        + 1.9 * (1.0 - np.clip(potential, 0.0, 1.0))
        + 2.6 * dry
        - 0.08 * river
        - 0.22 * population_scale
        - 0.68 * np.clip(accessibility, 0.0, 1.0)
    )
    bridge_discounts = bridge_transition_discounts(
        grid.river_order,
        transport.routes,
        transport.bridges,
    )
    settlement_anchor_protection = _settlement_protection_mask(
        settlements,
        grid.shape,
        radius=1,
    )
    governance_seeds = tuple(
        TerritorySeed(
            row=settlement.row,
            column=settlement.column,
            owner=index,
            domain=int(cultures.civilization_id[settlement.row, settlement.column]),
        )
        for index, settlement in enumerate(
            (
                item
                for item in settlements
                if item.tier != "site"
                and int(cultures.civilization_id[item.row, item.column]) > 0
            ),
            start=1,
        )
    )
    governed_nuclei = simulate_bounded_reach(
        TerritorySimulation(
            valid=land,
            friction=friction.astype(np.float32),
            transition_penalty=native_transitions,
            road_access=native_route_affinity,
            bridge_edges=bridge_discounts,
            owner_constraint=cultures.civilization_id.astype(np.int32),
        ),
        governance_seeds,
        maximum_cost=11.5,
    )
    frontier_reservation = derive_stateless_frontier(
        land,
        cultures.civilization_id,
        thematic.biome_zone,
        grid.snow,
        grid.elevation,
        thematic.climate.annual_precipitation,
        thematic.land_potential,
        population.population_weight,
        transport.accessibility,
        grid.river_order,
        native_transitions,
        governed_nuclei,
        target_share=frontier_target_share,
    )
    governed_land = coarse_land & ~frontier_reservation
    natural_compartments = natural_compartment_ids(
        governed_land & (coarse_civilization > 0),
        native_transitions,
        domain=coarse_civilization,
        barrier_threshold=14.0,
    )
    chosen_by_civilization = {
        identifier: _select_state_cores(
            settlements_by_civilization[identifier],
            quota=quota,
            region_cell_count=int(
                np.count_nonzero(
                    governed_land & (coarse_civilization == identifier)
                )
            ),
            step=step,
            width=grid.shape[1],
            accessibility=transport.accessibility,
            natural_compartment=natural_compartments,
        )
        for identifier, quota in quotas.items()
    }
    chosen = [
        settlement
        for identifier in sorted(chosen_by_civilization)
        for settlement in chosen_by_civilization[identifier]
    ]
    if len(chosen) != state_count:
        raise ValueError("not enough civilization-bound settlement cores for requested states")
    for civilization_identifier, civilization_cores in chosen_by_civilization.items():
        for core in civilization_cores:
            coarse_civilization[core.row // step, core.column // step] = (
                civilization_identifier
            )
    core_by_identifier: dict[int, Settlement] = {}
    civilization_by_identifier: dict[int, int] = {}
    strength_by_identifier: dict[int, float] = {}
    next_identifier = 1
    for civilization_identifier in sorted(chosen_by_civilization):
        civilization_cores = chosen_by_civilization[civilization_identifier]
        values = np.asarray([core.score for core in civilization_cores], dtype=np.float64)
        low = float(values.min())
        span = max(float(values.max()) - low, 1.0e-12)
        major_slots = max(1, int(round(len(civilization_cores) * 0.18)))
        strengths = tuple(
            float(
                np.clip(
                    0.72
                    + 0.60 * ((core.score - low) / span)
                    + 0.24 * float(transport.accessibility[core.row, core.column])
                    + (0.35 if index < major_slots else 0.0)
                    + (
                        0.40
                        if core.identifier
                        == civilization_core_by_identifier[civilization_identifier]
                        else 0.0
                    ),
                    0.68,
                    1.75,
                )
            )
            for index, core in enumerate(civilization_cores)
        )
        for core, strength in zip(civilization_cores, strengths, strict=True):
            core_by_identifier[next_identifier] = core
            civilization_by_identifier[next_identifier] = civilization_identifier
            strength_by_identifier[next_identifier] = strength
            next_identifier += 1
    territory_simulation = TerritorySimulation(
        valid=governed_land,
        friction=friction.astype(np.float32),
        transition_penalty=native_transitions,
        road_access=native_route_affinity,
        bridge_edges=bridge_discounts,
        owner_constraint=coarse_civilization.astype(np.int32),
    )

    # Countries are not a direct Voronoi allocation from a handful of
    # capitals.  First give every real urban settlement a connected daily
    # hinterland.  States then form by absorbing these local control regions
    # over several political rounds on their physical adjacency graph.
    control_settlements = tuple(
        settlement
        for civilization_identifier in sorted(settlements_by_civilization)
        for settlement in sorted(
            settlements_by_civilization[civilization_identifier],
            key=lambda item: item.identifier,
        )
    )
    control_seeds: list[TerritorySeed] = []
    for control_identifier, settlement in enumerate(control_settlements, start=1):
        civilization_identifier = int(
            cultures.civilization_id[settlement.row, settlement.column]
        )
        control_seeds.append(
            TerritorySeed(
                row=settlement.row,
                column=settlement.column,
                owner=control_identifier,
                strength=float(
                    np.clip(
                        0.92
                        + 0.18 * settlement.score
                        + 0.12
                        * float(transport.accessibility[settlement.row, settlement.column]),
                        0.82,
                        1.35,
                    )
                ),
                domain=civilization_identifier,
            )
        )
    control_result = simulate_territories(
        territory_simulation,
        tuple(control_seeds),
    )
    control_labels = control_result.owner.astype(np.int32)
    control_anchors = {
        (seed.row, seed.column): identifier
        for identifier, seed in enumerate(control_seeds, start=1)
    }
    control_domain = np.zeros(len(control_seeds) + 1, dtype=np.int32)
    for identifier, seed in enumerate(control_seeds, start=1):
        control_domain[identifier] = seed.domain

    # A city hinterland is first snapped to the physical compartments cut by
    # major river banks, drainage divides and ridges.  Only the narrow seam
    # between neighbouring hinterlands is then regrown.  This makes the graph
    # consumed by state formation geographic by construction: later politics
    # can transfer whole local regions, but cannot redraw a capital-distance
    # bisector across an available natural frontier.
    control_labels = snap_partition_to_natural_regions(
        control_labels,
        governed_land,
        native_transitions,
        control_anchors,
        owner_field=coarse_civilization.astype(np.int32),
        label_owner=control_domain,
        barrier_threshold=14.0,
        anchored_support=0.42,
        unanchored_majority=0.62,
    )
    control_labels = refine_partition_boundaries(
        control_labels,
        governed_land,
        native_transitions,
        control_anchors,
        owner_field=coarse_civilization.astype(np.int32),
        label_owner=control_domain,
        friction=friction.astype(np.float32),
        band_radius=12,
        jitter_strength=0.16,
    ).astype(np.int32)
    effective_civilization = coarse_civilization.astype(np.int32, copy=True)
    neutral_control = (effective_civilization <= 0) & (control_labels > 0)
    effective_civilization[neutral_control] = control_domain[
        control_labels[neutral_control]
    ]
    # A settlement hinterland may legitimately extend across a bridge or an
    # unopposed river.  It must not, however, hide that natural seam from the
    # later political graph.  Intersect each hinterland with the compartments
    # cut by strong river banks, drainage divides and ridges.  The resulting
    # micro-regions retain the same economic origin but may change political
    # owner independently during state formation.
    control_labels = natural_compartment_ids(
        governed_land & (control_labels > 0),
        native_transitions,
        domain=control_labels,
        barrier_threshold=14.0,
    ).astype(np.int32)
    population_normalized = population.population_weight.astype(np.float64)
    population_normalized /= max(
        float(population_normalized.max(initial=1.0)),
        1.0e-12,
    )
    local_resources = (
        0.52 * population_normalized
        + 0.28 * np.clip(thematic.land_potential, 0.0, 1.0)
        + 0.20 * np.clip(transport.accessibility, 0.0, 1.0)
    ).astype(np.float32)
    control_graph = build_control_region_graph(
        control_labels,
        native_transitions,
        native_route_affinity,
        bridge_discounts,
        effective_civilization,
        local_resources,
    )
    core_region_by_state = np.zeros(state_count + 1, dtype=np.int32)
    state_strength = np.zeros(state_count + 1, dtype=np.float32)
    for identifier in range(1, state_count + 1):
        core = core_by_identifier[identifier]
        core_region_by_state[identifier] = int(control_labels[core.row, core.column])
        state_strength[identifier] = strength_by_identifier[identifier]
    formation = simulate_state_formation(
        control_graph,
        core_region_by_state=core_region_by_state,
        state_strength=state_strength,
    )
    expanded = np.zeros(grid.shape, dtype=np.int16)
    controlled = control_labels > 0
    expanded[controlled] = formation.region_owner[control_labels[controlled]].astype(
        np.int16
    )
    expanded[~physical_land] = -1
    expanded[physical_land & ~land] = 0
    state_civilization = np.zeros(state_count + 1, dtype=np.int16)
    for identifier, civilization_identifier in civilization_by_identifier.items():
        state_civilization[identifier] = civilization_identifier
    assigned = expanded > 0
    cultural_mismatch = assigned & (
        (cultures.civilization_id > 0)
        & (
            state_civilization[np.clip(expanded, 0, state_count)]
            != cultures.civilization_id
        )
    )
    if np.any(cultural_mismatch):
        raise ValueError("territorial simulation crossed a civilization domain")
    # The pastoral and tribal domain was reserved before the political graph
    # was built.  States therefore stop at its physical edge instead of being
    # generated across the whole continent and punctured afterwards.
    frontier = frontier_reservation.copy()
    expanded = expanded.astype(np.int16)
    for settlement in settlements:
        if expanded[settlement.row, settlement.column] > 0:
            continue
        civilization_identifier = int(
            cultures.civilization_id[settlement.row, settlement.column]
        )
        candidates = [
            identifier
            for identifier, state_civilization_identifier in civilization_by_identifier.items()
            if state_civilization_identifier == civilization_identifier
        ]
        if not candidates:
            continue
        state_identifier = min(
            candidates,
            key=lambda identifier: (
                math.hypot(
                    core_by_identifier[identifier].row - settlement.row,
                    min(
                        abs(core_by_identifier[identifier].column - settlement.column),
                        grid.shape[1]
                        - abs(core_by_identifier[identifier].column - settlement.column),
                    ),
                ),
                identifier,
            ),
        )
        expanded[settlement.row, settlement.column] = state_identifier
        frontier[settlement.row, settlement.column] = False
    # A political core is always the minimum protected control space.
    for identifier, core in core_by_identifier.items():
        expanded[core.row, core.column] = identifier
        frontier[core.row, core.column] = False

    # The graph simulation already guarantees that every acquisition touches
    # existing state territory.  Pixel smoothing here would cut across the
    # physical control-region seams we just paid to compute, so it is
    # deliberately absent from the new path.
    assigned = expanded > 0
    expanded[
        assigned
        & (
            state_civilization[np.clip(expanded, 0, state_count)]
            != cultures.civilization_id
        )
    ] = 0
    expanded = _attach_adjacent_frontier_settlements(
        expanded,
        cultures.civilization_id,
        state_civilization,
        settlements,
    )
    political_core_cells = np.zeros(grid.shape, dtype=bool)
    for core in core_by_identifier.values():
        political_core_cells[core.row, core.column] = True
    expanded = repair_same_land_state_fragments(
        expanded,
        political_core_cells,
        connected_components(land)[0],
    )
    accidental_frontier = land & (expanded <= 0) & ~frontier
    expanded = fill_unassigned_territory(
        TerritorySimulation(
            valid=land,
            friction=friction.astype(np.float32),
            transition_penalty=native_transitions,
            road_access=native_route_affinity,
            bridge_edges=bridge_discounts,
            owner_constraint=cultures.civilization_id.astype(np.int32),
        ),
        expanded,
        accidental_frontier,
        state_civilization.astype(np.int32),
    ).astype(np.int16)
    if np.any(accidental_frontier & (expanded <= 0)):
        raise ValueError("ordinary land cannot remain an unexplained political vacuum")
    # Solve the final open-plain seam on the ownership grid itself.  This is
    # not display smoothing: candidate moves stay inside the two competing
    # states' cultural parent, protect every settlement, and are scored first
    # by watershed, river-bank and ridge evidence.  Only where that evidence
    # is absent does the low-frequency plain target break a surveyed straight.
    expanded = _naturalize_straight_state_boundaries(
        expanded,
        grid.elevation,
        thematic.drainage_basin,
        grid.river_order,
        cultures.civilization_id,
        state_civilization,
        settlement_anchor_protection,
        minimum_run=7,
        search_radius=8,
    )
    expanded = repair_same_land_state_fragments(
        expanded,
        political_core_cells,
        connected_components(land)[0],
    )
    frontier = land & (expanded <= 0)

    population_by_state: dict[int, float] = {}
    for identifier in range(1, state_count + 1):
        population_by_state[identifier] = float(
            population.population_weight[expanded == identifier].sum(dtype=np.float64)
        )
    urban_by_state = {identifier: 0 for identifier in range(1, state_count + 1)}
    urban_population_by_state = {
        identifier: 0.0 for identifier in range(1, state_count + 1)
    }
    port_urban_by_state = {
        identifier: 0 for identifier in range(1, state_count + 1)
    }
    for settlement in settlements:
        if settlement.tier == "site":
            continue
        identifier = int(expanded[settlement.row, settlement.column])
        if identifier > 0:
            urban_by_state[identifier] += 1
            urban_population_by_state[identifier] += settlement.population_max
            if settlement.site_type in {"port", "island-port"}:
                port_urban_by_state[identifier] += 1
    ranking = sorted(
        range(1, state_count + 1),
        key=lambda identifier: (-population_by_state[identifier], identifier),
    )
    ocean = np.isin(grid.water, (1, 3))
    ocean_adjacent = np.roll(ocean, 1, axis=1) | np.roll(ocean, -1, axis=1)
    if grid.shape[0] > 1:
        ocean_adjacent[1:] |= ocean[:-1]
        ocean_adjacent[:-1] |= ocean[1:]
    civilization_edge = np.zeros(grid.shape, dtype=bool)
    civilization_ids = cultures.civilization_id
    civilization_edge |= (
        (civilization_ids > 0)
        & (np.roll(civilization_ids, 1, axis=1) > 0)
        & (civilization_ids != np.roll(civilization_ids, 1, axis=1))
    )
    civilization_edge |= (
        (civilization_ids > 0)
        & (np.roll(civilization_ids, -1, axis=1) > 0)
        & (civilization_ids != np.roll(civilization_ids, -1, axis=1))
    )
    if grid.shape[0] > 1:
        civilization_edge[1:] |= (
            (civilization_ids[1:] > 0)
            & (civilization_ids[:-1] > 0)
            & (civilization_ids[1:] != civilization_ids[:-1])
        )
        civilization_edge[:-1] |= (
            (civilization_ids[:-1] > 0)
            & (civilization_ids[1:] > 0)
            & (civilization_ids[:-1] != civilization_ids[1:])
        )

    civilization_centers: dict[int, tuple[float, float, float]] = {}
    map_width = grid.shape[1]
    for civilization in cultures.civilizations:
        region = civilization_ids == civilization.identifier
        region_rows, region_columns = np.nonzero(region)
        weights = population.population_weight[region].astype(np.float64)
        if weights.sum() <= 0.0:
            weights = np.ones(region_rows.size, dtype=np.float64)
        center_row = float(np.average(region_rows, weights=weights))
        angles = region_columns.astype(np.float64) * (2.0 * np.pi / map_width)
        sine = float(np.sum(weights * np.sin(angles)))
        cosine = float(np.sum(weights * np.cos(angles)))
        center_column = float((math.atan2(sine, cosine) % (2.0 * np.pi)) * map_width / (2.0 * np.pi))
        scale = max(1.0, math.sqrt(float(np.count_nonzero(region)) / math.pi))
        civilization_centers[civilization.identifier] = (center_row, center_column, scale)

    profiles: dict[int, dict[str, float]] = {}
    centrality_by_state: dict[int, float] = {}
    mean_governed_area_per_urban = (
        np.count_nonzero(expanded > 0)
        / max(1, sum(urban_by_state.values()))
    )
    for identifier in range(1, state_count + 1):
        region = expanded == identifier
        civilization_identifier = civilization_by_identifier[identifier]
        center_row, center_column, center_scale = civilization_centers[civilization_identifier]
        core = core_by_identifier[identifier]
        column_distance = abs(core.column - center_column)
        column_distance = min(column_distance, map_width - column_distance)
        distance = math.hypot(core.row - center_row, column_distance)
        distance_score = math.exp(-distance / center_scale)
        influence = float(np.mean(cultures.civilization_influence[region])) / 3.0
        centrality_by_state[identifier] = float(
            np.clip(0.68 * distance_score + 0.22 * influence + 0.10 * core.score, 0.0, 1.0)
        )
        profiles[identifier] = {
            "coastal_share": float(np.mean(ocean_adjacent[region])),
            "steppe_share": float(
                np.mean(
                    np.isin(
                        thematic.biome_zone[region],
                        (BIOME_TEMPERATE_GRASSLAND, BIOME_SAVANNA_DRY_GRASSLAND),
                    )
                )
            ),
            "mountain_share": float(np.mean((grid.elevation > 0.55)[region])),
            "river_share": float(np.mean((grid.river_order > 0)[region])),
            "potential_mean": float(np.mean(thematic.land_potential[region])),
            "access_mean": float(np.mean(transport.accessibility[region])),
            "area_share": float(
                np.count_nonzero(region)
                / max(1, np.count_nonzero(civilization_ids == civilization_identifier))
            ),
            "edge_share": float(np.mean(civilization_edge[region])),
            "cultural_core": float(
                core.identifier
                == civilization_core_by_identifier[civilization_identifier]
            ),
            "urban_count": float(urban_by_state[identifier]),
            "territorial_load": float(
                np.count_nonzero(region)
                / max(
                    1.0,
                    mean_governed_area_per_urban
                    * max(1, urban_by_state[identifier]),
                )
            ),
            "shape_compactness": _territory_compactness(region),
            "core_urban_share": float(
                core.population_max
                / max(1.0, urban_population_by_state[identifier])
            ),
            "port_urban_share": float(
                port_urban_by_state[identifier]
                / max(1, urban_by_state[identifier])
            ),
            "centrality": centrality_by_state[identifier],
        }
    identifiers = list(range(1, state_count + 1))
    major_count = min(state_count, max(1, round(state_count * 0.18)))
    major_ids = set(
        sorted(
            identifiers,
            key=lambda identifier: (
                -(
                    population_by_state[identifier]
                    * (1.0 + 0.18 * min(urban_by_state[identifier], 5))
                    * (1.0 + 0.12 * profiles[identifier]["cultural_core"])
                ),
                identifier,
            ),
        )[:major_count]
    )
    medium_ids: set[int] = set()
    for civilization_identifier in sorted(set(civilization_by_identifier.values())):
        local = [
            identifier
            for identifier in ranking
            if civilization_by_identifier[identifier] == civilization_identifier
            and identifier not in major_ids
        ]
        fragmentation = state_formation_profiles[
            civilization_identifier
        ].fragmentation
        medium_count = min(
            len(local),
            max(
                1 if len(local) >= 4 else 0,
                round(len(local) * (0.10 + 0.25 * (1.0 - fragmentation))),
            ),
        )
        medium_ids.update(local[:medium_count])
    size_by_id = {
        identifier: (
            "major"
            if identifier in major_ids
            else "medium"
            if identifier in medium_ids
            else "small"
        )
        for identifier in identifiers
    }

    # Country identity is named without consulting the independently selected
    # government form. Historic names are proper-name roots only; all regime
    # titles were removed while loading the lexicon.
    name_by_id: dict[int, str] = {}
    used_names: set[str] = set()
    selected_country_records = tuple(major_records[: len(identifiers)])
    eastern_anchor_civilization = next(
        (
            item.identifier
            for item in cultures.civilizations
            if lineage_style_index(item.name_family) == 0
        ),
        None,
    )
    civilization_style_by_identifier = {
        item.identifier: lineage_style_index(item.name_family)
        for item in cultures.civilizations
    }
    name_by_id.update(
        _assign_coherent_major_names(
            identifiers,
            selected_country_records,
            profiles,
            population_by_state,
            centrality_by_state,
            civilization_by_identifier,
            civilization_style_by_identifier,
            used_names,
            eastern_anchor_civilization=eastern_anchor_civilization,
        )
    )
    remaining = [identifier for identifier in identifiers if identifier not in name_by_id]
    for identifier in remaining:
        name_by_id[identifier] = _derived_state_name(
            core_by_identifier[identifier],
            identifier,
            profiles[identifier],
            civilization_style_by_identifier[
                civilization_by_identifier[identifier]
            ],
            used_names,
        )

    # Governments are selected from universal templates only after every
    # country already has territory, scale, population and a proper name.
    government_template_by_state = {
        identifier: derive_government_template(
            core_by_identifier[identifier],
            size_by_id[identifier],
            profiles[identifier],
            state_formation_profiles[civilization_by_identifier[identifier]],
            ordinal=identifier,
        )
        for identifier in identifiers
    }
    government_forms = government_form_catalog(
        {template.key for template in government_template_by_state.values()}
    )
    government_form_by_key = {item.key: item for item in government_forms}
    imperial_state_ids, paramount_state_id = _imperial_state_identifiers(
        identifiers,
        civilization_by_identifier=civilization_by_identifier,
        civilization_style_by_identifier=civilization_style_by_identifier,
        name_by_identifier=name_by_id,
        government_key_by_identifier={
            identifier: government_template_by_state[identifier].key
            for identifier in identifiers
        },
        population_by_identifier=population_by_state,
        profiles=profiles,
        eastern_anchor_civilization=eastern_anchor_civilization,
    )
    states: list[State] = []
    for identifier in range(1, state_count + 1):
        core = core_by_identifier[identifier]
        region = expanded == identifier
        share = population_by_state[identifier]
        lower = max(1_000, int(round(share * population.population_min / 10_000.0)) * 10_000)
        upper = max(lower + 1_000, int(round(share * population.population_max / 10_000.0)) * 10_000)
        states.append(
            State(
                identifier=identifier,
                name=name_by_id[identifier],
                core_settlement_id=core.identifier,
                size_class=size_by_id[identifier],
                population_min=lower,
                population_max=upper,
                civilization_identifier=civilization_by_identifier[identifier],
                language_identifier=_dominant(cultures.language_id[region]),
            )
        )
    political_entities = tuple(
        PoliticalEntity(
            identifier=identifier,
            country_identifier=identifier,
            government_form_identifier=government_form_by_key[
                government_template_by_state[identifier].key
            ].identifier,
            formal_name=_formal_state_name(
                name_by_id[identifier],
                government_template_by_state[identifier].key,
                civilization_style_by_identifier[
                    civilization_by_identifier[identifier]
                ],
                size_by_id[identifier],
                imperial=identifier in imperial_state_ids,
                paramount=identifier == paramount_state_id,
            ),
        )
        for identifier in identifiers
    )
    frontier = frontier.astype(bool)
    frontier_groups = derive_frontier_groups(
        grid,
        thematic,
        population,
        cultures,
        frontier,
    )
    return PoliticalLayers(
        state_id=expanded,
        frontier=frontier,
        states=tuple(states),
        government_forms=government_forms,
        political_entities=political_entities,
        frontier_groups=frontier_groups,
    )
