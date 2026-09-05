"""Independent state-formation and government-form derivation.

Civilizations affect the spatial opportunity for several countries to form;
they never prescribe a government. Country identity and government form are
generated separately and joined only by the final political-entity relation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from ..model import WorldGrid
from ..thematic import ThematicLayers
from .model import CultureLayers, GovernmentForm, PopulationLayers, Settlement, TransportLayers


@dataclass(frozen=True, slots=True)
class StateFormationProfile:
    """Continuous spatial conditions controlling how many countries emerge."""

    civilization_identifier: int
    quota_multiplier: float
    minimum_states: int
    maximum_states: int
    fragmentation: float
    centralization: float


@dataclass(frozen=True, slots=True)
class GovernmentTemplate:
    """Reusable government form with no country, culture, or name ownership."""

    key: str
    name: str
    description: str


GOVERNMENT_TEMPLATES: tuple[GovernmentTemplate, ...] = (
    GovernmentTemplate("court-monarchy", "宫廷君主制", "以王室、宫廷与核心都会维持统治"),
    GovernmentTemplate("bureaucratic-monarchy", "官僚君主制", "以税收、文书与层级官僚统治高人口地区"),
    GovernmentTemplate("hereditary-feudalism", "分封王权", "共主与世袭领主分享土地、军役与征税权"),
    GovernmentTemplate("maritime-republic", "海商共和制", "港市商人、船东与行会掌握政权"),
    GovernmentTemplate("city-republic", "城邦共和制", "紧凑城市与周边乡地的自治议政体系"),
    GovernmentTemplate("nomadic-confederacy", "游牧部盟", "季节性汗庭与部落首领的议盟统治"),
    GovernmentTemplate("clan-league", "宗族议盟", "山地宗族、谷地首领与盟誓维持的联合"),
    GovernmentTemplate("tributary-chiefdom", "朝贡酋邦", "多个地方首领围绕一个仪礼中心组成的松散政权"),
    GovernmentTemplate("estate-monarchy", "等级君主制", "君主与贵族、城市和地方团体协商权利"),
)


def derive_state_formation_profiles(
    grid: WorldGrid,
    thematic: ThematicLayers,
    population: PopulationLayers,
    settlements: tuple[Settlement, ...],
    cultures: CultureLayers,
    transport: TransportLayers,
) -> dict[int, StateFormationProfile]:
    """Measure territorial fragmentation without assigning governments."""

    for field in (
        thematic.land_potential,
        population.population_weight,
        cultures.civilization_id,
        transport.accessibility,
    ):
        if field.shape != grid.shape:
            raise ValueError("state-formation inputs must match WorldGrid shape")

    settlement_by_identifier = {item.identifier: item for item in settlements}
    profiles: dict[int, StateFormationProfile] = {}
    for civilization in cultures.civilizations:
        identifier = civilization.identifier
        region = cultures.civilization_id == identifier
        urban = [
            item
            for item in settlements
            if item.tier != "site" and region[item.row, item.column]
        ]
        if not np.any(region) or not urban:
            continue
        core = settlement_by_identifier[civilization.core_settlement_id]
        urban_population = max(1, sum(item.population_max for item in urban))
        core_share = core.population_max / urban_population
        urban_count = len(urban)
        access = float(np.mean(transport.accessibility[region]))
        highland_share = float(np.mean((grid.elevation > 0.55)[region]))
        urban_polycentricity = 1.0 - float(np.clip(core_share / 0.42, 0.0, 1.0))
        network_friction = 1.0 - float(np.clip(access / 0.28, 0.0, 1.0))
        center_count = float(np.clip(math.log2(urban_count + 1) / 5.2, 0.0, 1.0))
        fragmentation = float(
            np.clip(
                0.40 * network_friction
                + 0.31 * urban_polycentricity
                + 0.22 * center_count
                + 0.07 * min(1.0, highland_share / 0.30),
                0.0,
                1.0,
            )
        )
        centralization = float(
            np.clip(
                0.46 * (1.0 - urban_polycentricity)
                + 0.34 * (1.0 - network_friction)
                + 0.20 * min(1.0, float(np.mean(thematic.land_potential[region])) / 0.68),
                0.0,
                1.0,
            )
        )
        root = math.sqrt(urban_count)
        minimum = min(urban_count, min(16, max(1, round(root * (0.90 + 3.0 * fragmentation)))))
        maximum = min(
            urban_count,
            min(28, max(minimum, round(root * (1.85 + 3.6 * fragmentation)))),
        )
        profiles[identifier] = StateFormationProfile(
            civilization_identifier=identifier,
            quota_multiplier=0.65 + 1.10 * fragmentation,
            minimum_states=minimum,
            maximum_states=maximum,
            fragmentation=fragmentation,
            centralization=centralization,
        )
    return profiles


def derive_government_template(
    core: Settlement,
    size_class: str,
    profile: Mapping[str, float],
    formation: StateFormationProfile,
    *,
    ordinal: int,
) -> GovernmentTemplate:
    """Choose a government independently for one already-formed country.

    No civilization identifier, country name, or naming-family value enters
    this function. Templates can be reused across cultures, while local
    conditions can produce several forms inside one civilization.
    """

    coastal = float(profile.get("coastal_share", 0.0))
    steppe = float(profile.get("steppe_share", 0.0))
    mountain = float(profile.get("mountain_share", 0.0))
    river = float(profile.get("river_share", 0.0))
    potential = float(profile.get("potential_mean", 0.0))
    urban_count = float(profile.get("urban_count", 0.0))
    urban = min(1.0, urban_count / 5.0)
    core_urban_share = float(np.clip(profile.get("core_urban_share", 0.35), 0.0, 1.0))
    port_urban_share = float(np.clip(profile.get("port_urban_share", 0.0), 0.0, 1.0))
    polycentricity = float(
        np.clip((urban_count - 1.0) / 4.0, 0.0, 1.0)
        * (1.0 - core_urban_share)
    )
    territorial_load = float(max(0.0, profile.get("territorial_load", 1.0)))
    shape_compactness = float(
        np.clip(profile.get("shape_compactness", 0.58), 0.0, 1.0)
    )
    compact_territory = float(
        np.clip(1.45 - territorial_load, 0.0, 1.0)
        * np.clip(shape_compactness / 0.55, 0.0, 1.0)
    )
    excessive_territory = float(np.clip(territorial_load - 1.15, 0.0, 2.5))
    corridor_shape = float(np.clip((0.38 - shape_compactness) / 0.28, 0.0, 1.0))
    access = min(1.0, float(profile.get("access_mean", 0.0)) / 0.28)
    centrality = float(profile.get("centrality", 0.0))
    small = 1.0 if size_class == "small" else 0.0
    major = 1.0 if size_class == "major" else 0.0
    jitter = ((ordinal * 2654435761) & 0xFFFF) / 0xFFFF - 0.5

    scores = {
        "court-monarchy": 0.68 * major + 0.72 * centrality + 0.35 * formation.centralization,
        "bureaucratic-monarchy": 0.70 * potential + 0.72 * urban + 0.48 * access + 0.26 * major,
        "hereditary-feudalism": 0.92 * formation.fragmentation + 0.58 * small + 0.26 * urban + 0.25 * centrality,
        "maritime-republic": (
            1.45 * coastal
            + 0.44 * urban
            + 0.45 * access
            + 1.10 * port_urban_share * polycentricity
            + (0.28 if core.site_type in {"port", "island-port"} else 0.0)
        ),
        "city-republic": (
            0.55 * small
            + 1.05 * core_urban_share
            + 0.34 * access
            + 0.38 * compact_territory
            - 0.18 * max(0.0, urban_count - 1.0)
            - 0.28 * polycentricity
            - 0.24 * major
            - 0.72 * excessive_territory
            - 1.35 * corridor_shape
        ),
        "nomadic-confederacy": 1.05 * steppe + 0.42 * (1.0 - potential) + 0.20 * formation.fragmentation,
        "clan-league": 1.12 * mountain + 0.42 * formation.fragmentation + (0.26 if core.site_type == "pass" else 0.0),
        "tributary-chiefdom": 0.62 * (1.0 - urban) + 0.46 * (1.0 - access) + 0.22 * small,
        "estate-monarchy": 0.38 + 0.36 * urban + 0.32 * formation.fragmentation + 0.22 * river,
    }
    if urban_count > 2.0 or territorial_load > 1.65 or shape_compactness < 0.28:
        scores["city-republic"] -= 3.0
    template_order = {template.key: index for index, template in enumerate(GOVERNMENT_TEMPLATES)}
    selected_key = max(
        scores,
        key=lambda key: (
            scores[key] + 0.035 * jitter * ((template_order[key] % 3) - 1),
            -template_order[key],
        ),
    )
    return next(template for template in GOVERNMENT_TEMPLATES if template.key == selected_key)


def government_form_catalog(keys: set[str]) -> tuple[GovernmentForm, ...]:
    """Materialize only the reusable forms used in the current world."""

    return tuple(
        GovernmentForm(
            identifier=index,
            key=template.key,
            name=template.name,
            description=template.description,
        )
        for index, template in enumerate(
            (template for template in GOVERNMENT_TEMPLATES if template.key in keys),
            start=1,
        )
    )
