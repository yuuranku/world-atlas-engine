"""Offline Three.js view of the published equirectangular vector atlas."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


_THEME_LABELS = {
    "terrain": "地形", "climate": "柯本气候", "biome": "生态群系",
    "watershed": "水文流域", "potential": "农业潜力", "habitability": "宜居度",
    "vegetation": "植被覆盖", "population": "人口分布", "civilizations": "文明区",
    "languages": "语言分布", "religions": "宗教分布", "political": "国家政区",
    "provinces": "省份政区",
}


def write_globe(output: Path, *, surface: str, ink: str, textures: dict[str, str], grid_digest: str, world_name: str) -> None:
    if "terrain" not in textures:
        raise ValueError("the globe requires its authoritative terrain surface")
    if set(textures) - _THEME_LABELS.keys():
        raise ValueError("the globe requires known thematic identifiers")
    expected_viewbox = None
    for source in (surface, ink, *textures.values()):
        try:
            root = ET.fromstring(source)
            viewbox = tuple(float(value) for value in root.attrib["viewBox"].replace(",", " ").split())
        except (ET.ParseError, KeyError, TypeError, ValueError) as error:
            raise ValueError("globe textures must be full-world 2:1 equirectangular SVG surfaces") from error
        if (root.tag != "{http://www.w3.org/2000/svg}svg" or len(viewbox) != 4
                or not all(math.isfinite(value) for value in viewbox)
                or viewbox[:2] != (0.0, 0.0) or viewbox[3] <= 0 or viewbox[2] != viewbox[3] * 2):
            raise ValueError("globe textures must be full-world 2:1 equirectangular SVG surfaces")
        if expected_viewbox is not None and viewbox != expected_viewbox:
            raise ValueError("all globe textures must use the same world grid")
        expected_viewbox = viewbox
    assets = Path(__file__).parent / "web"
    shutil.copy2(assets / "globe.js", output / "globe.js")
    shutil.copy2(assets / "three-license.txt", output / "three-license.txt")
    theme_urls = {}
    for key, source in textures.items():
        filename = f"globe-theme-{key}.svg"
        (output / filename).write_text(source, encoding="utf-8")
        theme_urls[key] = filename
    payload = {"gridDigest": grid_digest, "projection": "equirectangular", "surface": surface, "ink": ink, "textures": theme_urls}
    (output / "globe-data.js").write_text("window.WorldAtlasGlobe=" + json.dumps(payload, ensure_ascii=True) + ";\n", encoding="utf-8")
    title = html.escape(world_name or "世界地图")
    options = "".join(f'<option value="{key}">{label}</option>' for key, label in _THEME_LABELS.items() if key in textures)
    (output / "globe.html").write_text(f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><link rel="icon" href="data:,"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · 球体</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#192733;color:#e6e9e6;font:14px system-ui,sans-serif}}header{{position:absolute;z-index:2;top:22px;left:24px;right:24px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}h1{{font-size:18px;margin:0;font-weight:500}}a,button,select{{background:#283c49;color:#e6e9e6;border:1px solid #566977;padding:9px 14px;border-radius:4px;text-decoration:none;font:inherit}}#globe{{width:100vw;height:100vh}}canvas{{display:block;width:100%;height:100%;touch-action:none}}footer{{position:absolute;bottom:24px;left:24px;pointer-events:none;color:#bac8cd;max-width:75%;line-height:1.6}}small{{opacity:.7}}label{{display:flex;align-items:center;gap:6px}}:focus-visible{{outline:2px solid #e2bd73;outline-offset:3px}}@media(max-width:600px){{header{{top:12px;left:12px;right:12px;gap:8px}}h1{{font-size:15px}}a,button,select{{padding:7px 9px}}footer{{left:12px;bottom:12px;max-width:90%}}}}
</style></head><body><header><a href="index.html">← 平面图集</a><h1>{title}</h1><select id="globe-theme" aria-label="球体专题">{options}</select><label><input type="checkbox" id="globe-graticule">经纬网</label><button id="globe-reset">复位</button></header><main id="globe" aria-label="可拖动旋转的世界球体"></main><footer><div id="globe-status" role="status">正在载入球体…</div><div id="globe-coordinates">点击球面查看经纬度</div><small>Three.js · 等经纬纹理映射 · 无地形夸张 · 仅交互时重绘</small></footer><script src="globe-data.js"></script><script src="globe.js"></script></body></html>''', encoding="utf-8")
