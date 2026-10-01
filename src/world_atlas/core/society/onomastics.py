"""Deterministic local-language morphology for generated toponyms.

Civilizations are generated from geography and connectivity.  Naming lineages
are therefore unique to those generated civilizations instead of being aliases
for the obsolete seven setting categories.  The profiles below are phonological
and morphological grammars, not civilization or territory templates.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

from .naming_profiles import naming_profile


_STYLE_COUNT = 12

_FORMANTS: tuple[dict[str, tuple[str, ...]], ...] = (
    {
        "river": ("津", "济", "浦", "浔"), "coast": ("海", "浦", "潮", "汀"),
        "lake": ("泽", "湄", "湖"), "mountain": ("陵", "岑", "岘"), "plain": ("原", "安", "平", "陵", "城"),
        "wetland": ("泽", "汀", "洲"), "spring": ("泉", "井"), "pass": ("关", "隘"),
        "fortress": ("寨", "堡", "台"), "region": ("宁", "陵", "原", "城"),
    },
    {
        "river": ("福德", "布鲁克", "阿姆"), "coast": ("维克", "哈芬", "松德", "霍尔姆"),
        "lake": ("梅尔", "塞"), "mountain": ("贝格", "达尔", "格伦"), "plain": ("顿", "汉姆", "肯特"),
        "wetland": ("芬", "马什"), "spring": ("凯尔德",), "pass": ("盖特",),
        "fortress": ("堡", "加德"), "region": ("顿", "维尔", "汉姆"),
    },
    {
        "river": ("里沃", "蓬特"), "coast": ("马雷", "波尔托"), "lake": ("拉戈",),
        "mountain": ("蒙特", "瓦尔"), "plain": ("维尔", "坎波"), "wetland": ("马雷斯",),
        "spring": ("泉特",), "pass": ("卡尔",), "fortress": ("卡塞尔",), "region": ("维尔", "纳", "利亚"),
    },
    {
        "river": ("沃达", "布罗德"), "coast": ("莫尔", "格勒"), "lake": ("奥泽尔",),
        "mountain": ("戈拉",), "plain": ("波列",), "wetland": ("波涛",), "spring": ("克利尼",),
        "pass": ("弗拉塔",), "fortress": ("格勒", "斯克"), "region": ("斯克", "维奇", "诺夫"),
    },
    {
        "river": ("苏", "奥孜"), "coast": ("德尼兹",), "lake": ("诺尔", "库勒"),
        "mountain": ("塔格",), "plain": ("亚孜", "达拉"), "wetland": ("萨兹",), "spring": ("布拉克",),
        "pass": ("达坂",), "fortress": ("巴勒克", "库尔干"), "region": ("萨雷", "卡尔", "兹"),
    },
    {
        "river": ("鲁德", "阿卜"), "coast": ("班达尔",), "lake": ("达里亚",),
        "mountain": ("库赫",), "plain": ("达什特",), "wetland": ("马兹",), "spring": ("恰尔",),
        "pass": ("德尔本",), "fortress": ("卡拉",), "region": ("阿巴德", "坎德"),
    },
    {
        "river": ("纳哈尔", "瓦迪"), "coast": ("米纳", "亚姆"), "lake": ("布哈伊拉",),
        "mountain": ("杰贝勒",), "plain": ("萨赫勒",), "wetland": ("哈沃尔",), "spring": ("艾因",),
        "pass": ("巴布",), "fortress": ("卡拉特",), "region": ("达尔", "迪亚"),
    },
    {
        "river": ("纳迪", "迦特"), "coast": ("帕坦", "萨穆德"), "lake": ("萨罗瓦",),
        "mountain": ("吉里",), "plain": ("普尔", "格拉姆"), "wetland": ("迪普",), "spring": ("库德",),
        "pass": ("迦特",), "fortress": ("杜尔格",), "region": ("普尔", "纳加尔"),
    },
    {
        "river": ("孙盖",), "coast": ("丹绒", "拉布汉"), "lake": ("达瑙",),
        "mountain": ("古农",), "plain": ("巴鲁", "萨瓦"), "wetland": ("拉娃",), "spring": ("马塔",),
        "pass": ("萨尔岔",), "fortress": ("本堡",), "region": ("努萨", "班达"),
    },
    {
        "river": ("阿冯", "林"), "coast": ("凯尔", "特雷"), "lake": ("林",),
        "mountain": ("本", "格伦"), "plain": ("莫尔",), "wetland": ("纳安",), "spring": ("蒂伦",),
        "pass": ("卡恩",), "fortress": ("敦",), "region": ("特雷", "布拉"),
    },
    {
        "river": ("约基",), "coast": ("涅米", "萨塔马"), "lake": ("耶尔维",),
        "mountain": ("瓦拉",), "plain": ("佩尔托",), "wetland": ("芬苏",), "spring": ("泉纳",),
        "pass": ("维亚",), "fortress": ("林纳",), "region": ("拉", "耶尔"),
    },
    {
        "river": ("穆托", "鲁阿"), "coast": ("姆巴尼",), "lake": ("尼亚萨",),
        "mountain": ("基里",), "plain": ("萨瓦纳",), "wetland": ("波宁",), "spring": ("奇娃",),
        "pass": ("卡雅",), "fortress": ("博马",), "region": ("巴", "纳", "利"),
    },
)

# A physical feature keeps both its cartographic classifier (added by the
# caller) and a local formative.  This avoids turning every mountain, marsh,
# or river into an opaque random root once the canonical identity pass runs.
_FEATURE_ENVIRONMENT = {
    "river": "river",
    "mountain": "mountain",
    "peak": "mountain",
    "lake": "lake",
    "sea": "coast",
    "inland-sea": "coast",
    "bay": "coast",
    "strait": "coast",
    "island": "coast",
    "island-group": "coast",
    "plain": "plain",
    "plateau": "plain",
    "basin": "plain",
    "desert": "region",
    "wetland": "wetland",
    "ridge": "mountain", "hills": "mountain", "valley": "river", "gorge": "river",
    "foothills": "mountain", "steep-slope": "mountain", "lowland-valley": "river",
    "snow-mountain": "mountain", "cape": "coast", "peninsula": "coast",
    "isthmus": "coast", "arid-upland": "region",
}


def lineage_key(identifier: int, *, style_index: int | None = None) -> str:
    """Return the unique naming lineage of one generated civilization."""

    if identifier < 1:
        raise ValueError("lineage identifier must be positive")
    if style_index is not None:
        if style_index < 0 or style_index >= _STYLE_COUNT:
            raise ValueError("lineage style index is out of range")
        return f"lineage-s{style_index:02d}-{identifier:02d}"
    return f"lineage-{identifier:02d}"


def lineage_branch(lineage: str, branch: str | int) -> str:
    """Return a deterministic local branch without changing its sound family.

    A civilization, its daughter languages, its states, and its settlements
    should share a phonological inventory without repeatedly reusing one city
    as every other proper name.  The branch remains an input to the root
    derivation, while :func:`lineage_style_index` still reads the original
    explicit style marker at the front of the lineage.
    """

    base = str(lineage).strip()
    label = str(branch).strip()
    if not base or not label:
        raise ValueError("lineage branches require non-empty lineage and branch")
    return f"{base}:branch-{label}"


def _lineage_number(lineage: str) -> int:
    match = re.search(r"(\d+)$", lineage)
    if match is not None:
        return max(1, int(match.group(1)))
    return int.from_bytes(hashlib.sha256(lineage.encode("utf-8")).digest()[:2], "big") + 1


def lineage_style_index(lineage: str) -> int:
    explicit = re.search(r"lineage-s(\d+)-", lineage)
    if explicit is not None:
        return int(explicit.group(1)) % _STYLE_COUNT
    number = _lineage_number(lineage)
    return (number * 7 + number // 3 + 1) % _STYLE_COUNT


@lru_cache(maxsize=2048)
def lineage_roots(lineage: str, *, count: int = 128) -> tuple[str, ...]:
    """Build roots from complete morphemes within one selected tradition."""

    style = lineage_style_index(lineage)
    profile = naming_profile(lineage, style=style)
    # The inherited compound layer joins intact local morphemes, allowing a
    # large civilization hundreds of distinct names without serial numbers or
    # borrowing another culture's inventory.
    inherited = (first + second for first in profile.stems for second in profile.stems
                 if first != second and not second.startswith(first) and not first.endswith(second))
    formed = (stem + ending for stem in profile.stems for ending in profile.endings
              if not ending.startswith(stem) and not stem.endswith(ending))
    inventory = tuple(dict.fromkeys(
        root for root in (*formed, *inherited)
        if not any(a == b for a, b in zip(root, root[1:]))
    ))
    roots = sorted(inventory, key=lambda root: hashlib.sha256(f"{lineage}:{root}".encode("utf-8")).digest())
    if len(roots) < count:
        raise ValueError(f"cannot build enough roots for {lineage}")
    return tuple(roots[:count])


def _ordered_roots(lineage: str, *, seed: int) -> tuple[str, ...]:
    roots = lineage_roots(lineage, count=384)
    start = int(seed) % len(roots)
    return roots[start:] + roots[:start]


def lineage_entity_candidates(
    lineage: str,
    category: str,
    *,
    seed: int,
) -> tuple[str, ...]:
    """Return short, related stems for non-settlement proper names.

    ``category`` deliberately changes the ordering rather than inventing a
    second unrelated character pool.  Callers use a distinct
    :func:`lineage_branch` for languages, civilizations, institutions, and
    states; this yields recognisable family resemblance without reducing those
    names to a city name plus a bureaucratic suffix.
    """

    category_digest = hashlib.sha256(category.encode("utf-8")).digest()
    ordered = _ordered_roots(
        lineage,
        seed=int(seed) + int.from_bytes(category_digest[:4], "big"),
    )
    profile = naming_profile(lineage, style=lineage_style_index(lineage))
    if category == "state" and profile.polities:
        offset = int(seed) % len(profile.polities)
        inherited = profile.polities[offset:] + profile.polities[:offset]
        qualified = tuple(direction + root for direction in ("东", "西", "南", "北") for root in inherited)
        return tuple(dict.fromkeys((*inherited, *qualified, *ordered)))
    if category in {"state", "culture", "language", "faith"}:
        endings = profile.endings
        if profile.style == 0 and category == "state":
            endings = tuple(ending for ending in endings if ending not in {"城", "津", "溪", "庭", "垣"})
        formed = tuple(stem + ending for stem in profile.stems for ending in endings
                       if not ending.startswith(stem) and not stem.endswith(ending)
                       and not any(a == b for a, b in zip(stem + ending, (stem + ending)[1:])))
        formed = tuple(sorted(formed, key=lambda root: hashlib.sha256(f"{seed}:{root}".encode("utf-8")).digest()))
        return tuple(dict.fromkeys((*formed, *ordered)))
    return tuple(sorted(ordered, key=len)) if profile.style == 0 else ordered


def geographic_name_candidates(
    lineage: str,
    feature_type: str,
    *,
    seed: int,
) -> tuple[str, ...]:
    """Return feature-name stems with a visible local environmental formative."""

    environment = _FEATURE_ENVIRONMENT.get(feature_type)
    if environment is None:
        raise ValueError(f"unsupported geographic feature type: {feature_type}")
    style = lineage_style_index(lineage)
    profile = naming_profile(lineage, style=style)
    ordered_roots = _ordered_roots(lineage, seed=seed)
    if style == 0:
        offset = int(seed) % len(profile.landmarks)
        landmarks = profile.landmarks[offset:] + profile.landmarks[:offset]
        return tuple(dict.fromkeys((*landmarks, *ordered_roots)))
    formants = _FORMANTS[style][environment]
    formant_start = (int(seed) // 11) % len(formants)
    ordered_formants = formants[formant_start:] + formants[:formant_start]
    result: list[str] = []
    stems = profile.stems[int(seed) % len(profile.stems):] + profile.stems[:int(seed) % len(profile.stems)]
    for root in stems:
        for formant in ordered_formants:
            if root[-1:] != formant[:1] and not formant.startswith(root) and not root.endswith(formant):
                result.append(root + formant)
    # Keep a deeper reserve in the same local lexicon; the normal path always
    # uses the formative above, so geography remains visibly site-grounded.
    result.extend(ordered_roots)
    return tuple(dict.fromkeys(result))


def settlement_name_candidates(
    lineage: str,
    environment: str,
    relation: str | None,
    *,
    seed: int,
) -> tuple[str, ...]:
    """Return proper-name candidates whose etymology follows the local site.

    Inherited roots mix with readable site formants. Transliterated traditions
    retain more of their local river, harbour and settlement morphology.
    """

    style = lineage_style_index(lineage)
    profile = naming_profile(lineage, style=style)
    ordered_roots = _ordered_roots(lineage, seed=seed)
    formants = _FORMANTS[style][environment]
    formant_start = (seed // 7) % len(formants)
    ordered_formants = formants[formant_start:] + formants[:formant_start]
    use_visible_etymology = environment in {"pass", "fortress"} or seed % 100 < 72
    result: list[str] = []
    shift = int(seed) % len(profile.stems)
    stems = profile.stems[shift:] + profile.stems[:shift]
    visible = tuple(root + formant for root in stems for formant in ordered_formants
                    if root[-1:] != formant[:1] and not formant.startswith(root)
                    and not root.endswith(formant))
    if style == 0 and relation is not None and environment in {"river", "lake", "mountain"}:
        # Cardinal modifiers are used only when the caller measured a site's
        # position relative to that physical feature.
        result.extend(relation + root + ordered_formants[0] for root in stems[:8])
    result.extend((*visible, *ordered_roots) if use_visible_etymology else (*ordered_roots, *visible))
    return tuple(dict.fromkeys(result))


def lineage_language_candidates(lineage: str, identifier: int) -> tuple[str, ...]:
    roots = lineage_roots(lineage)
    start = (identifier * 29 + 7) % len(roots)
    ordered = roots[start:] + roots[:start]
    return tuple(f"{root}语" for root in ordered)


def lineage_civilization_candidates(lineage: str, identifier: int) -> tuple[str, ...]:
    roots = lineage_roots(lineage)
    start = (identifier * 17 + 5) % len(roots)
    ordered = roots[start:] + roots[:start]
    return tuple(f"{root}文明圈" for root in ordered)
