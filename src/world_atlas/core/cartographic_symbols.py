"""Shared SVG marks for the map and its topographic legend.

Symbols follow conventional point-height and marsh map vocabulary;
they are not a claim of compliance with a national surveying specification.
Fortified passes, fortresses and religious centres use this atlas's own
shared symbols, rather than implying a universal surveying convention.
"""

from html import escape


LANDFORM_STYLES = {
    "ridge": {"label": "山脊", "color": "#796451", "representation": "named-line", "min_scale": 8., "max_scale": 160.},
    "hills": {"label": "丘陵", "color": "#85765c", "representation": "named-area", "min_scale": 8., "max_scale": 160.},
    "valley": {"label": "河谷", "color": "#57796b", "representation": "named-line", "min_scale": 12., "max_scale": 256.},
    "gorge": {"label": "深切河谷", "color": "#776454", "representation": "named-line", "min_scale": 16., "max_scale": 256.},
    "foothills": {"label": "山麓", "color": "#88755f", "representation": "named-area", "min_scale": 8., "max_scale": 160.},
    "steep-slope": {"label": "陡坡地带", "color": "#8b6b52", "representation": "downslope-hachures", "min_scale": 32., "max_scale": 256.},
    "lowland-valley": {"label": "低平河谷", "color": "#648877", "representation": "named-area", "min_scale": 12., "max_scale": 256.},
    "snow-mountain": {"label": "雪山", "color": "#789aac", "representation": "snow-area", "min_scale": 4., "max_scale": 256.},
    "cape": {"label": "岬角", "color": "#687a78", "representation": "named-point", "min_scale": 12., "max_scale": 256.},
    "peninsula": {"label": "半岛", "color": "#697365", "representation": "named-area", "min_scale": 8., "max_scale": 160.},
    "isthmus": {"label": "地峡", "color": "#747a67", "representation": "named-point", "min_scale": 16., "max_scale": 256.},
    "arid-upland": {"label": "干旱起伏地", "color": "#988269", "representation": "arid-area", "min_scale": 8., "max_scale": 160.},
}

# Area textures express supported regional coverage. They do not assert a
# point landform, sand grain size, glacial flow or rock substrate.
AREA_PATTERNS = {
    "wetland": {"width": 8, "height": 8, "markup": '<path d="M.8 2.5h2.4 M4.2 5.5h2.6 M1.5 3.5h1.1 M5 6.5h1" stroke="#447d8c" stroke-width=".5" fill="none" opacity=".55"/>'},
    "snow-mountain": {"width": 8, "height": 8, "markup": '<rect width="8" height="8" fill="#f5fafc" opacity=".30"/><path d="M1 3h2 M5 6h1.6" stroke="#adc6d2" stroke-width=".35" opacity=".4"/>'},
    "arid-upland": {"width": 8, "height": 8, "markup": '<path d="M1 2l1.3 .8 M5 5l1.2 -.7" stroke="#9b8974" stroke-width=".5" fill="none" opacity=".35"/>'},
}


def pattern_markup(kind: str, identifier: str) -> str:
    pattern = AREA_PATTERNS[kind]
    return (f'<pattern id="{escape(identifier, quote=True)}" patternUnits="userSpaceOnUse" '
            f'patternContentUnits="userSpaceOnUse" width="{pattern["width"]}" height="{pattern["height"]}" '
            f'data-landform-pattern="{kind}">{pattern["markup"]}</pattern>')


MARKS = {
    "peak": '<circle cx="0" cy="0" r=".75" fill="#665448" stroke="none"/>',
}

SITE_MARKS = {
    "port": '<circle cy="-.72" r=".22" fill="none" stroke="#356a85" stroke-width=".22"/><path d="M0-.5V1M-.5-.1H.5M-1 .25Q-1 1 0 1Q1 1 1 .25M-1 .25l.3.1M1 .25l-.3.1" fill="none" stroke="#356a85" stroke-width=".24" stroke-linecap="round"/>',
    "pass": '<path d="M-1 1V-1H-.55V-.55H.55V-1H1V1H.45V.15A.45 .45 0 0 0-.45 .15V1Z" fill="#d7b978" stroke="#593f31" stroke-width="0.32" stroke-linejoin="round"/>',
    "fortress": '<path d="M-1 1V-1H-.6V-.55H-.2V-1H.2V-.55H.6V-1H1V1Z" fill="#d7b978" stroke="#593f31" stroke-width="0.32" stroke-linejoin="round"/><path d="M-.25 1V.25H.25V1" fill="none" stroke="#593f31" stroke-width="0.25"/>',
    "holy": '<polygon points="0,-1.85 .304,-.735 1.308,-1.308 .735,-.304 1.85,0 .735,.304 1.308,1.308 .304,.735 0,1.85 -.304,.735 -1.308,1.308 -.735,.304 -1.85,0 -.735,-.304 -1.308,-1.308 -.304,-.735" fill="#f6d889" stroke="#7f493b" stroke-width="0.34"/>',
}


def _legend_row(mark: str, label: str, *, view_box: str) -> str:
    return f'<div class="legend-row"><svg class="symbol-sample" aria-hidden="true" viewBox="{view_box}">{mark}</svg><span>{label}</span></div>'


def symbol_definitions() -> str:
    return '<defs>' + ''.join(f'<g id="mark-{kind}">{mark}</g>' for kind, mark in MARKS.items()) + '</defs>'


def geographic_legend() -> str:
    rows = []
    for kind, label in (("peak", "高程点 · m（模型海面基准）"),):
        mark = MARKS[kind]
        if kind == "peak":
            mark += '<text x="1.3" y="1" font-size="2.3" fill="#665448">1842</text>'
        rows.append(_legend_row(mark, label, view_box="-5 -3.5 12 7"))
    for kind, label in (("wetland", "湿地范围"), ("snow-mountain", "雪覆盖高山"), ("arid-upland", "干旱起伏地")):
        identifier = f'legend-landform-{kind}'
        mark = f'<defs>{pattern_markup(kind, identifier)}</defs><rect x="-5" y="-3.5" width="10" height="7" fill="url(#{identifier})"/>'
        rows.append(_legend_row(mark, label, view_box="-5 -3.5 10 7"))
    rows.append(_legend_row('<path d="M-3 -2l1 2 M0 -1l1 2 M3 -2l1 2" stroke="#8b6b52" stroke-width=".55"/>', "陡坡（短线顺实际下坡方向）", view_box="-5 -3.5 10 7"))
    rows.append('<p class="help">山脊、丘陵、河谷、山麓及海岸形态以地形和分级地名表达；干旱颜色不代表沙质地表。</p>')
    rows.append(_legend_row('<path d="M-4 0Q-3-3 0-2Q4-3 4 0Q3 3 0 2Q-4 3-4 0Z" fill="#b8dce8" stroke="#5885a1" stroke-width=".45"/>', "湖泊与内陆水体", view_box="-5 -3.5 10 7"))
    rows.append('<p class="help">图式参考 <a href="https://www.usgs.gov/ngp-standards-and-specifications/us-topo-map-symbol-guide" target="_blank" rel="noopener noreferrer">USGS</a> / <a href="https://www.swisstopo.admin.ch/dam/en/sd-web/2Z6YZ2roT2rm/symbols_en.pdf" target="_blank" rel="noopener noreferrer">Swisstopo</a>；关隘与要塞采用本图约定。</p>')
    return ''.join(rows)


def settlement_legend() -> str:
    rows = []
    for radius, stroke, fill, width, label in (
        (3.70, "#263447", "#f0cf83", 0.82, "都会"),
        (2.15, "#33475d", "#f7eed7", 0.62, "城市"),
        (1.05, "#435467", "#f7eed7", 0.42, "城镇"),
        (1.85, "#593f31", "#d7b978", 0.62, "边地据点"),
    ):
        mark = f'<circle cx="0" cy="0" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
        if label == "都会":
            mark += '<circle cx="0" cy="0" r="1.05" fill="#263447" stroke="none"/>'
        rows.append(_legend_row(mark, label, view_box="-6 -4.5 12 9"))
    capital = '<circle cx="0" cy="0" r="2.15" fill="#f7eed7" stroke="#33475d" stroke-width="0.62"/><circle cx="0" cy="0" r="3.053" fill="none" stroke="#263447" stroke-width="0.72"/>'
    rows.append(_legend_row(capital, "首都（外圈）", view_box="-6 -4.5 12 9"))
    for kind, label in (("pass", "关隘"), ("fortress", "要塞"), ("port", "海港 / 湖港（锚）"), ("holy", "宗教圣城（八芒星）")):
        mark = f'<g transform="scale(2.15)">{SITE_MARKS[kind]}</g>'
        rows.append(_legend_row(mark, label, view_box="-6 -4.5 12 9"))
    return ''.join(rows)
