"""Deterministic local-language morphology for generated toponyms.

Civilizations are generated from geography and connectivity.  Naming lineages
are therefore unique to those generated civilizations instead of being aliases
for the obsolete seven setting categories.  The profiles below are phonological
and morphological grammars, not civilization or territory templates.
"""

from __future__ import annotations

import hashlib
import re


_STYLE_SYLLABLES: tuple[tuple[str, ...], ...] = (
    ("洛", "衡", "宁", "安", "嘉", "宜", "临", "望", "清", "定", "崇", "绍"),
    ("阿", "维", "诺", "格", "哈", "斯", "伦", "德", "贝", "克", "霍", "兰"),
    ("罗", "塞", "蒙", "贝", "利", "亚", "托", "纳", "维", "奥", "雅", "卡"),
    ("诺", "维", "拉", "罗", "米", "格", "德", "涅", "斯", "科", "普", "奥"),
    ("阿", "尔", "卡", "乌", "苏", "巴", "特", "兰", "兹", "库", "蒂", "鲁"),
    ("阿", "达", "沙", "赫", "拉", "伊", "斯", "坎", "德", "巴", "法", "米"),
    ("阿", "麦", "萨", "法", "拉", "哈", "巴", "德", "尔", "纳", "卡", "里"),
    ("阿", "拉", "迪", "纳", "迦", "塔", "巴", "罗", "萨", "米", "普", "达"),
    ("拉", "玛", "纳", "巴", "鲁", "萨", "里", "达", "嘉", "卡", "蒂", "维"),
    ("凯", "安", "布", "兰", "格", "伊", "罗", "莫", "特", "里", "温", "达"),
    ("卡", "里", "维", "塔", "涅", "米", "萨", "尔", "瓦", "诺", "迪", "亚"),
    ("穆", "巴", "恩", "卡", "鲁", "萨", "达", "里", "维", "纳", "基", "莫"),
)

_FORMANTS: tuple[dict[str, tuple[str, ...]], ...] = (
    {
        "river": ("津", "济", "浦", "浔"), "coast": ("海", "浦", "潮", "汀"),
        "lake": ("泽", "泞", "湄"), "mountain": ("岳", "岭", "岘"), "plain": ("原", "昌", "平"),
        "wetland": ("泽", "汀", "洲"), "spring": ("泉", "井"), "pass": ("关", "险"),
        "fortress": ("寨", "戍", "台"), "region": ("宁", "安", "嘉"),
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

_TRANSPARENT_TERMS = {
    "river": ("河东", "河西", "江阴", "江阳", "临河", "济水"),
    "coast": ("临海", "潮阳", "海宁", "望海"),
    "lake": ("湖东", "湖西", "湖阴", "湖阳"),
    "mountain": ("山南", "山北", "岭东", "岭西"),
    "plain": ("广平", "安原", "宜丰", "昌平"),
    "wetland": ("临泽", "江洲", "清浦"),
    "spring": ("甘泉", "灵井"), "pass": ("山关", "石门"),
    "fortress": ("定边", "安戍"), "region": ("宜宁", "嘉定", "安昌"),
}


def lineage_key(identifier: int, *, style_index: int | None = None) -> str:
    """Return the unique naming lineage of one generated civilization."""

    if identifier < 1:
        raise ValueError("lineage identifier must be positive")
    if style_index is not None:
        if style_index < 0 or style_index >= len(_STYLE_SYLLABLES):
            raise ValueError("lineage style index is out of range")
        return f"lineage-s{style_index:02d}-{identifier:02d}"
    return f"lineage-{identifier:02d}"


def _lineage_number(lineage: str) -> int:
    match = re.search(r"(\d+)$", lineage)
    if match is not None:
        return max(1, int(match.group(1)))
    return int.from_bytes(hashlib.sha256(lineage.encode("utf-8")).digest()[:2], "big") + 1


def lineage_style_index(lineage: str) -> int:
    explicit = re.search(r"lineage-s(\d+)-", lineage)
    if explicit is not None:
        return int(explicit.group(1)) % len(_STYLE_SYLLABLES)
    number = _lineage_number(lineage)
    return (number * 7 + number // 3 + 1) % len(_STYLE_SYLLABLES)


def lineage_roots(lineage: str, *, count: int = 128) -> tuple[str, ...]:
    """Build an opaque local lexicon from one lineage's sound inventory."""

    style = lineage_style_index(lineage)
    syllables = _STYLE_SYLLABLES[style]
    digest = hashlib.sha256(lineage.encode("utf-8")).digest()
    start = int.from_bytes(digest[:2], "big") % len(syllables)
    roots: list[str] = []
    seen: set[str] = set()
    for ordinal in range(count * 4):
        first_index = ordinal % len(syllables)
        second_index = (ordinal // len(syllables)) % len(syllables)
        first = syllables[(start + first_index * 5) % len(syllables)]
        second = syllables[(start + second_index * 7 + 3) % len(syllables)]
        third = syllables[(start + first_index * 7 + second_index * 5 + 1) % len(syllables)]
        if second == first:
            second = syllables[(start + second_index * 7 + 4) % len(syllables)]
        if third in {first, second}:
            third = syllables[(start + first_index * 7 + second_index * 5 + 2) % len(syllables)]
        candidate = first + second + (third if (ordinal + digest[3]) % 5 == 0 else "")
        if candidate in seen or len(set(candidate)) == 1:
            continue
        seen.add(candidate)
        roots.append(candidate)
        if len(roots) >= count:
            break
    if len(roots) < count:
        raise ValueError(f"cannot build enough roots for {lineage}")
    return tuple(roots)


def settlement_name_candidates(
    lineage: str,
    environment: str,
    relation: str | None,
    *,
    seed: int,
) -> tuple[str, ...]:
    """Return proper-name candidates whose etymology follows the local site.

    Most surface forms are deliberately opaque, as real names become worn and
    conventional.  A minority preserves a recognizable geographic element.
    """

    style = lineage_style_index(lineage)
    roots = lineage_roots(lineage)
    root_start = seed % len(roots)
    ordered_roots = roots[root_start:] + roots[:root_start]
    formants = _FORMANTS[style][environment]
    formant_start = (seed // 7) % len(formants)
    ordered_formants = formants[formant_start:] + formants[:formant_start]
    # Most real settlement names preserve their geographic origin only after
    # phonetic wear or semantic drift.  Keep transparent "river/coast/plain"
    # compounds exceptional so the atlas does not read like a feature legend.
    use_visible_etymology = seed % 100 < (14 if style == 0 else 9)
    result: list[str] = []
    if style == 0 and use_visible_etymology:
        transparent = list(_TRANSPARENT_TERMS[environment])
        if relation is not None and environment in {"river", "lake", "mountain"}:
            stem = {"river": "河", "lake": "湖", "mountain": "山"}[environment]
            transparent.insert(0, f"{stem}{relation}")
            transparent.insert(1, f"{ordered_roots[0][:1]}{'\u9633' if relation in {'\u5317', '\u4e1c'} else '\u9634'}")
        shift = (seed // 13) % len(transparent)
        result.extend(transparent[shift:] + transparent[:shift])
    if use_visible_etymology:
        for root in ordered_roots[:24]:
            for formant in ordered_formants:
                if root[-1:] == formant[:1]:
                    continue
                result.append(root + formant)
    result.extend(ordered_roots)
    if not use_visible_etymology:
        for root in ordered_roots[:24]:
            for formant in ordered_formants:
                if root[-1:] == formant[:1]:
                    continue
                result.append(root + formant)
    # Rare historical layers: a transferred clan/personal root plus the local
    # element.  This gives the same lineage more depth without borrowing from
    # a global catch-all list.
    for first, second in zip(ordered_roots[:20], ordered_roots[17:37], strict=True):
        result.append(first[:2] + second[-1:])
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
