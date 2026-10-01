"""Canonical seeded identities, independent of all previously published worlds.

Names are finalized on model records before any labels or files are rendered.
Stable IDs retain every physical, cultural, administrative and transport link.
Only common translated geographic/legal terms are shared with older atlases.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from typing import Collection, Mapping

from .model import SocietyLayers
from .naming_profiles import naming_profile, profile_lineage
from .names import _FEATURE_SUFFIX
from .onomastics import (
    geographic_name_candidates,
    lineage_branch,
    lineage_entity_candidates,
    lineage_key,
    lineage_style_index,
    settlement_name_candidates,
)
from .provinces import _province_suffixes
from .religion import _TRADITIONS


_SITE_ENVIRONMENT = {
    "port": "coast", "island-port": "coast", "lake-port": "lake", "river-city": "river",
    "market": "plain", "pass": "pass", "oasis": "spring", "fortress": "fortress",
}
_GROUP_SUFFIX = {
    "khan-court": "汗庭", "confederacy": "部盟", "tribes": "部落",
    "sea-clans": "海族", "boat-people": "舟族", "island-clans": "岛族",
    "hunters": "氏族", "forest-clans": "氏族", "village-league": "乡社联盟",
    "hill-tribes": "部落", "clan-league": "氏族联盟", "river-clans": "氏族", "fishers": "舟族",
}
_RECORD_KEYS = (
    "settlements", "civilizations", "languages", "religions", "geographicFeatures",
    "states", "politicalEntities", "frontierGroups", "provinces",
    "polityAnchors", "minorEntries", "cityLabels", "portLabels", "villageLabels", "minorMapLabels",
)


def collect_proper_names(document: Mapping) -> set[str]:
    """Extract identities, never common government/climate/category labels."""
    names = set()
    for key in _RECORD_KEYS:
        for item in document.get(key, ()):
            if isinstance(item, Mapping):
                for field in ("name", "formal_name", "label"):
                    if isinstance(item.get(field), str) and item[field].strip():
                        names.add(item[field].strip())
    if isinstance(document.get("worldName"), str):
        names.add(document["worldName"])
    return names


class NameRegistry:
    """Stable per-world name allocator with explicit old-name exclusion."""
    def __init__(self, seed: int, forbidden: Collection[str] = ()):
        self.seed = int(seed)
        self.forbidden = frozenset(str(name).strip() for name in forbidden if str(name).strip())
        self.used: set[str] = set()
        self.names: dict[tuple, str] = {}

    def available(self, name: str) -> bool:
        return (name not in self.used and name not in self.forbidden
                and not any(old in name for old in self.forbidden if len(old) >= 2))

    def name_seed(self, key: str) -> int:
        return int.from_bytes(hashlib.sha256(f"world-names-v3:{self.seed}:{key}".encode()).digest()[:8], "big")

    def name(self, key: str, style: int, *, suffix: str = "", preferred: str | None = None,
             following: str = "", candidates: tuple[str, ...] | None = None) -> str:
        cache_key = (key, style, suffix, preferred, following, candidates)
        if cache_key in self.names:
            return self.names[cache_key]
        if candidates is None:
            lineage = lineage_branch(lineage_key(1, style_index=style), self.seed)
            candidates = lineage_entity_candidates(lineage, key, seed=self.name_seed(key))
        roots = (preferred, *candidates) if preferred else candidates
        for root in roots:
            if any(a == b for a, b in zip(root, root[1:])):
                continue
            candidate = root + suffix
            if not self.available(candidate) or (following and not self.available(candidate + following)):
                continue
            self.used.add(candidate)
            if following:
                self.used.add(candidate + following)
            self.names[cache_key] = candidate
            return candidate
        raise ValueError(f"name inventory exhausted for {key}")


def assign_world_identity(society: SocietyLayers, *, seed: int, forbidden: Collection[str] = (),
                          profile_set: str = "mixed") -> SocietyLayers:
    """Name the fully simulated world, then update all referenced identities."""
    registry = NameRegistry(seed, forbidden)
    families = {item.identifier: lineage_style_index(item.name_family) for item in society.cultures.languages}
    cultures = {item.identifier: lineage_style_index(item.name_family) for item in society.cultures.civilizations}
    culture_profiles = {item.identifier: profile_lineage(item.name_family, seed=seed, profile_set=profile_set)
                        for item in society.cultures.civilizations}
    language_profiles = {item.identifier: culture_profiles[item.family_identifier]
                         for item in society.cultures.languages}
    language_lineages = {identifier: lineage_branch(lineage, seed)
                         for identifier, lineage in language_profiles.items()}
    culture_lineages = {identifier: lineage_branch(lineage, seed)
                        for identifier, lineage in culture_profiles.items()}

    def lineage_at(row, column):
        identifier = int(society.cultures.language_id[row, column])
        if identifier in language_lineages:
            return language_lineages[identifier]
        identifier = int(society.cultures.civilization_id[row, column])
        return culture_lineages[identifier]

    def entity_name(key, lineage, **kwargs):
        return registry.name(key, lineage_style_index(lineage), candidates=lineage_entity_candidates(
            lineage_branch(lineage, key), key.split(":")[0], seed=registry.name_seed(key)), **kwargs)

    cities = []
    for item in society.settlements:
        lineage = lineage_at(item.row, item.column)
        key = f"city:{item.identifier}"
        candidates = settlement_name_candidates(lineage, _SITE_ENVIRONMENT[item.site_type], None,
                                                 seed=registry.name_seed(key))
        cities.append(replace(item, name=registry.name(key, lineage_style_index(lineage), candidates=candidates)))
    cities = tuple(cities)
    city_names = {item.identifier: item.name for item in cities}
    civilizations = tuple(replace(item, name_family=culture_profiles[item.identifier],
        name=entity_name(f"culture:{item.identifier}", culture_lineages[item.identifier], suffix="文明圈"))
        for item in society.cultures.civilizations)
    languages = tuple(replace(item, name_family=language_profiles[item.identifier],
        name=entity_name(f"language:{item.identifier}", language_lineages[item.identifier], suffix="语"))
        for item in society.cultures.languages)
    religious_terms = {item[0]: item[1] for item in _TRADITIONS}
    religions = tuple(replace(item,
        name=entity_name(f"faith:{item.identifier}", culture_lineages[item.origin_civilization_identifier],
            suffix=religious_terms[item.tradition])) for item in society.religions.religions)
    old_states = {item.identifier: item for item in society.politics.states}
    state_suffixes = {}
    for item in society.politics.political_entities:
        old = old_states[item.country_identifier]
        if not item.formal_name.startswith(old.name):
            raise ValueError(f"formal country name does not reference its identity: {item.identifier}")
        state_suffixes[item.country_identifier] = item.formal_name[len(old.name):].replace(
            "海洋共和国", "共和国").replace("沙阿", "君王").replace("埃米尔", "亲王")
    states = tuple(replace(item, name=entity_name(f"state:{item.identifier}", culture_lineages[item.civilization_identifier],
        following=state_suffixes[item.identifier])) for item in society.politics.states)
    state_names = {item.identifier: item.name for item in states}
    entities = tuple(replace(item, formal_name=state_names[item.country_identifier] + state_suffixes[item.country_identifier])
                     for item in society.politics.political_entities)
    provinces = []
    for item in society.provinces.provinces:
        style = cultures[old_states[item.state_identifier].civilization_identifier]
        terms = _province_suffixes(item.administrative_system, item.administrative_function, item.administrative_rank, style)
        # Translate ranks semantically here. Borrowed title syllables (e.g.
        # 米尔 in 埃米尔) can themselves be old atlas city names; changing the
        # new root cannot remove that collision. The legal rank is unchanged.
        suffix = terms[(seed + item.identifier) % len(terms)].replace("埃米尔", "亲王").replace("沙阿", "君王")
        provinces.append(replace(item, name=entity_name(f"province:{item.identifier}",
            culture_lineages[old_states[item.state_identifier].civilization_identifier], suffix=suffix,
            preferred=city_names[item.core_settlement_id])))
    groups = tuple(replace(item, name=entity_name(f"people:{item.identifier}", language_lineages[item.language_identifier],
        suffix=_GROUP_SUFFIX[item.organization])) for item in society.politics.frontier_groups)
    features = tuple(replace(item, name=registry.name(f"feature:{item.identifier}", families[item.language_identifier],
        suffix=_FEATURE_SUFFIX[item.feature_type], candidates=geographic_name_candidates(
            language_lineages[item.language_identifier], item.feature_type,
            seed=registry.name_seed(f"feature:{item.identifier}")))) for item in society.geographic_features)
    result = replace(society, settlements=cities,
        cultures=replace(society.cultures, civilizations=civilizations, languages=languages),
        religions=replace(society.religions, religions=religions),
        politics=replace(society.politics, states=states, political_entities=tuple(entities), frontier_groups=groups),
        provinces=replace(society.provinces, provinces=tuple(provinces)), geographic_features=features)
    audit = naming_audit(result, forbidden)
    if audit["oldNameMatches"] or audit["duplicateNames"]:
        raise ValueError(f"world naming audit failed: {audit}")
    return result


def selected_naming_profiles(society: SocietyLayers) -> tuple[dict, ...]:
    """Read the selected naming traditions without changing any identities."""

    result = []
    for item in society.cultures.civilizations:
        style = lineage_style_index(item.name_family)
        profile = naming_profile(item.name_family, style=style)
        result.append({"civilizationId": item.identifier, "civilization": item.name,
                       "key": profile.key, "label": profile.label,
                       "set": profile.key.split("-", 1)[0], "style": style})
    return tuple(result)


def naming_audit(society: SocietyLayers, forbidden: Collection[str]) -> dict:
    categories = {
        "cities": [item.name for item in society.settlements],
        "civilizations": [item.name for item in society.cultures.civilizations],
        "languages": [item.name for item in society.cultures.languages],
        "religions": [item.name for item in society.religions.religions],
        "states": [item.name for item in society.politics.states],
        "formalStates": [item.formal_name for item in society.politics.political_entities],
        "provinces": [item.name for item in society.provinces.provinces],
        "peoples": [item.name for item in society.politics.frontier_groups],
        "geography": [item.name for item in society.geographic_features],
    }
    all_names = [name for values in categories.values() for name in values]
    matches = sorted({(name, old) for name in all_names for old in forbidden
                      if old == name or (len(old) >= 2 and old in name)})
    duplicates = sorted({name for values in categories.values() for name, count in Counter(values).items() if count > 1})
    return {"schema": "world-names-v3", "counts": {k: len(v) for k, v in categories.items()},
            "oldNameMatches": matches, "duplicateNames": duplicates,
            "profiles": selected_naming_profiles(society),
            "nameDigest": hashlib.sha256(json.dumps(categories, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "examples": {key: values[:6] for key, values in categories.items()}}
