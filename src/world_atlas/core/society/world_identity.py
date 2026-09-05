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
from .names import _FEATURE_SUFFIX
from .onomastics import lineage_style_index
from .provinces import _province_suffixes
from .religion import _TRADITIONS


# Phonotactic registers, not borrowed place-name lists. Each world/family draws
# a stable subset, giving local names related sounds without reusing old roots.
_REGISTERS = (
    ("澄岑晏澹雍弥绥榆宛芜栖棠", "岚芷蘅洵祁汶桓筠嵩荻莳宥", "陵阜堇洄岐庐垣汀砚棣岑芮"),
    ("埃瑟芙铎赫布奎缇温戈", "伦文瑞索莱芬蕾温瑟达", "恩特恩姆德什恩克勒尔"),
    ("艾薇瑟佩菲塞维陶芙奥", "伦娅雷泽维蒂菲罗萨涅", "雅娅欧恩娅亚昂蕾瑟雅"),
    ("兹弗博沃涅瑟杜耶卓科", "雷泽列维珀里涅瑞兹瓦", "什兹岑克恩夫茨什恩尔"),
    ("鄂图阔旭乌罕巴哲库额", "赛耶穆格岱沁都额哲兰", "图克沁勒罕兀岱什根齐"),
    ("斐赫泽努阿珀贾谢扎艾", "泽希杜兰芮梅鲁沙耶瑞", "恩兹达尔沙姆恩赫安兹"),
    ("伊扎努哈贾谢斐艾萨厄", "耶希麦杜扎斐赫贾鲁芮", "姆兹德恩法赫姆贾恩尔"),
    ("塔迦毗珈阿缇苏耶斐杜", "耶毗祢伽芮陀提缇沙穆", "拉耶提姆伽那耶陀珈尼"),
    ("珊蒂伊帕陶阿芙努乌玛", "珞娅缇努玛耶娅珊里塔", "娅乌珞伊娜瑙娅陶里乌"),
    ("艾奎温苔瑟铎芙布芮伊", "瑞洛耶温埃菲瑟林朔莱", "恩林温尔恩洛德瑞恩什"),
    ("于耶伊珀缇艾瑟乌维佩", "莱耶涅米珀瑞伊菲瑟陶", "涅宁耶尔米恩莱伊涅姆"),
    ("姆恩珀杜泽恩贝夸乌哲", "贝杜哲恩穆珀加迪耶戈", "尼贝加穆恩杜珀耶贡姆"),
)
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

    def name(self, key: str, style: int, *, suffix: str = "", preferred: str | None = None, following: str = "") -> str:
        cache_key = (key, style, suffix, preferred, following)
        if cache_key in self.names:
            return self.names[cache_key]
        profile = _REGISTERS[style % len(_REGISTERS)]
        for attempt in range(10000):
            digest = hashlib.sha256(f"world-names-v1:{self.seed}:{style}:{key}:{attempt}".encode()).digest()
            chars = [part[digest[index] % len(part)] for index, part in enumerate(profile)]
            if style != 0 and digest[4] % 5 == 0:
                chars.insert(2, profile[1][digest[5] % len(profile[1])])
            if any(a == b for a, b in zip(chars, chars[1:])):
                continue
            root = preferred if attempt == 0 and preferred else "".join(chars)
            candidate = root + suffix
            if not self.available(candidate) or (following and not self.available(candidate + following)):
                continue
            self.used.add(candidate)
            if following:
                self.used.add(candidate + following)
            self.names[cache_key] = candidate
            return candidate
        raise ValueError(f"name inventory exhausted for {key}")


def assign_world_identity(society: SocietyLayers, *, seed: int, forbidden: Collection[str] = ()) -> SocietyLayers:
    """Name the fully simulated world, then update all referenced identities."""
    registry = NameRegistry(seed, forbidden)
    families = {item.identifier: lineage_style_index(item.name_family) for item in society.cultures.languages}
    cultures = {item.identifier: lineage_style_index(item.name_family) for item in society.cultures.civilizations}

    def style_at(row, column):
        identifier = int(society.cultures.language_id[row, column])
        if identifier in families:
            return families[identifier]
        identifier = int(society.cultures.civilization_id[row, column])
        return cultures.get(identifier, 1)

    cities = tuple(replace(item, name=registry.name(f"city:{item.identifier}", style_at(item.row, item.column)))
                   for item in society.settlements)
    city_names = {item.identifier: item.name for item in cities}
    civilizations = tuple(replace(item,
        name=registry.name(f"culture:{item.identifier}", cultures[item.identifier], suffix="文明圈",
                           preferred=city_names[item.core_settlement_id])) for item in society.cultures.civilizations)
    languages = tuple(replace(item,
        name=registry.name(f"language:{item.identifier}", families[item.identifier], suffix="语",
                           preferred=city_names[item.core_settlement_id])) for item in society.cultures.languages)
    religious_terms = {item[0]: item[1] for item in _TRADITIONS}
    religions = tuple(replace(item,
        name=registry.name(f"faith:{item.identifier}", cultures[item.origin_civilization_identifier],
            suffix=religious_terms[item.tradition],
            preferred=city_names[item.holy_settlement_id])) for item in society.religions.religions)
    old_states = {item.identifier: item for item in society.politics.states}
    state_suffixes = {}
    for item in society.politics.political_entities:
        old = old_states[item.country_identifier]
        if not item.formal_name.startswith(old.name):
            raise ValueError(f"formal country name does not reference its identity: {item.identifier}")
        state_suffixes[item.country_identifier] = item.formal_name[len(old.name):].replace(
            "海洋共和国", "共和国").replace("沙阿", "君王").replace("埃米尔", "亲王")
    states = tuple(replace(item, name=registry.name(f"state:{item.identifier}", cultures[item.civilization_identifier],
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
        provinces.append(replace(item, name=registry.name(f"province:{item.identifier}", style, suffix=suffix,
            preferred=city_names[item.core_settlement_id] if item.administrative_function == "capital" else None)))
    groups = tuple(replace(item, name=registry.name(f"people:{item.identifier}", families[item.language_identifier],
        suffix=_GROUP_SUFFIX[item.organization])) for item in society.politics.frontier_groups)
    features = tuple(replace(item, name=registry.name(f"feature:{item.identifier}", families[item.language_identifier],
        suffix=_FEATURE_SUFFIX[item.feature_type])) for item in society.geographic_features)
    result = replace(society, settlements=cities,
        cultures=replace(society.cultures, civilizations=civilizations, languages=languages),
        religions=replace(society.religions, religions=religions),
        politics=replace(society.politics, states=states, political_entities=tuple(entities), frontier_groups=groups),
        provinces=replace(society.provinces, provinces=tuple(provinces)), geographic_features=features)
    audit = naming_audit(result, forbidden)
    if audit["oldNameMatches"] or audit["duplicateNames"]:
        raise ValueError(f"world naming audit failed: {audit}")
    return result


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
    return {"schema": "world-names-v1", "counts": {k: len(v) for k, v in categories.items()},
            "oldNameMatches": matches, "duplicateNames": duplicates,
            "nameDigest": hashlib.sha256(json.dumps(categories, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "examples": {key: values[:6] for key, values in categories.items()}}
