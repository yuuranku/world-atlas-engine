"""Saved-world place records and the independent map interface shell."""

from collections.abc import Mapping
import html
import json
import math
from pathlib import Path
import shutil

from scipy import ndimage

from .model import WorldGrid
from .society.model import SocietyLayers
from .cartographic_symbols import LANDFORM_STYLES, geographic_legend, settlement_legend


THEMES = (
    ("none", "地形与水系"), ("climate", "柯本气候"), ("biome", "生物群系"),
    ("watershed", "水文流域"), ("potential", "农业潜力"), ("habitability", "宜居度"),
    ("vegetation", "植被覆盖"), ("population", "人口分布"), ("civilizations", "文明区"),
    ("languages", "语言分布"), ("religions", "宗教分布"), ("political", "国家政区"),
    ("provinces", "省份政区"),
)
SEASONS = (("vernal", "春分"), ("june", "夏至"), ("autumnal", "秋分"), ("december", "冬至"))
SITE_LABELS = {
    "river-city": "河畔聚落", "port": "海港", "lake-port": "湖港", "market": "商贸聚落",
    "pass": "山口聚落", "oasis": "绿洲聚落", "fortress": "堡垒", "island-port": "岛港",
}
TIER_LABELS = {"metropolis": "大都市", "city": "城市", "town": "城镇", "site": "聚落"}
FEATURE_LABELS = {
    "mountain": "山脉", "peak": "山峰", "river": "河流", "lake": "湖泊", "sea": "海域",
    "inland-sea": "内海", "bay": "海湾", "strait": "海峡", "island": "岛屿",
    "island-group": "群岛", "plain": "平原", "plateau": "高原", "basin": "盆地",
    "desert": "荒漠", "wetland": "湿地",
    **{kind: style["label"] for kind, style in LANDFORM_STYLES.items()},
}


def build_place_index(grid: WorldGrid, society: SocietyLayers, *, settlement_locations: Mapping) -> list[dict]:
    """Export factual locations without rerunning terrain or human generation.

    Region bounds come from canonical ownership cells. Geographic ownership
    describes its label anchor, not a territorial claim on an entire river or
    mountain range. Province population is omitted: it has no saved interval.
    """
    if society.population.population_weight.shape != grid.shape:
        raise ValueError("place data must belong to the same saved world grid")
    height, width = grid.shape
    extents = grid.metadata["extents"]
    states = {item.identifier: item for item in society.politics.states}
    provinces = {item.identifier: item for item in society.provinces.provinces}
    settlements = {item.identifier: item for item in society.settlements}
    holy_names = {item.identifier: item.name for item in society.religions.religions}
    capital_ids = {item.core_settlement_id for item in states.values()}
    state_bounds = ndimage.find_objects(society.politics.state_id)
    province_bounds = ndimage.find_objects(society.provinces.province_id)

    def bounds(slices):
        if slices is None:
            raise ValueError("a named administrative region must own saved cells")
        rows, columns = slices
        return [columns.start, rows.start, columns.stop, rows.stop]

    def location(row, column):
        if not (math.isfinite(row) and math.isfinite(column) and 0 <= row <= height and 0 <= column <= width):
            raise ValueError("place anchor lies outside its saved world")
        native = [column, row]
        return {
            "native": native,
            "longitude": float(extents["west"] + native[0]/width*(extents["east"]-extents["west"])),
            "latitude": float(extents["north"] - native[1]/height*(extents["north"]-extents["south"])),
        }

    def ownership(row, column):
        state_identifier = int(society.politics.state_id[row, column])
        state = states.get(state_identifier)
        province = provinces.get(int(society.provinces.province_id[row, column]))
        return {"state": {"id": state.identifier, "name": state.name} if state else None,
                "province": {"id": province.identifier, "name": province.name} if province else None,
                # The saved frontier mask describes the original frontier
                # geography; the final state grid decides current governance.
                "frontier": state_identifier == 0}

    result = []
    for city in society.settlements:
        result.append({
            "id": f"city:{city.identifier}", "sourceId": city.identifier, "kind": "city", "name": city.name,
            "typeLabel": TIER_LABELS[city.tier], "siteLabel": SITE_LABELS[city.site_type],
            "population": [city.population_min, city.population_max],
            "isCapital": city.identifier in capital_ids,
            "holyReligion": holy_names[city.holy_religion_identifier] if city.holy_religion_identifier is not None else None,
            **location(*settlement_locations[city.identifier]), **ownership(city.row, city.column),
        })
    for state in society.politics.states:
        capital = settlements[state.core_settlement_id]
        result.append({
            "id": f"state:{state.identifier}", "sourceId": state.identifier, "kind": "state", "name": state.name,
            "typeLabel": "国家", "capitalName": capital.name,
            "population": [state.population_min, state.population_max],
            "bounds": bounds(state_bounds[state.identifier-1]), **location(*settlement_locations[capital.identifier]),
        })
    for province in society.provinces.provinces:
        core = settlements[province.core_settlement_id]
        state = states[province.state_identifier]
        result.append({
            "id": f"province:{province.identifier}", "sourceId": province.identifier, "kind": "province", "name": province.name,
            "typeLabel": "省份", "capitalName": core.name, "state": {"id": state.identifier, "name": state.name},
            "bounds": bounds(province_bounds[province.identifier-1]), **location(*settlement_locations[core.identifier]),
        })
    for feature in society.geographic_features:
        result.append({
            "id": f"geography:{feature.identifier}", "sourceId": feature.identifier, "kind": "geography", "name": feature.name,
            "featureType": feature.feature_type,
            "typeLabel": FEATURE_LABELS[feature.feature_type], "ownerLabel": "标注位置", "tier": feature.tier,
            **location(feature.row+.5, feature.column+.5), **ownership(feature.row, feature.column),
        })
    if len({item["id"] for item in result}) != len(result):
        raise ValueError("place identifiers must be unique")
    return result


def write_map_app_assets(output: str | Path, *, grid: WorldGrid, society: SocietyLayers,
                         settlement_locations: Mapping) -> None:
    """Deliver the map interface and factual place index to a review directory.

    Display anchors are mandatory: substituting native cell centers would
    detach the selected place from its actual coastal or river-side symbol.
    The renderer separately writes the scene and real thematic previews.
    """
    places = build_place_index(grid, society, settlement_locations=settlement_locations)
    from .society.governance import derive_governance, write_governance
    governance = derive_governance(grid, society)
    cities = {record['id']: record for record in governance['settlements']}
    countries = {record['id']: record for record in governance['countries']}
    state_names = {state.identifier: state.name for state in society.politics.states}
    states = {state.identifier: state for state in society.politics.states}
    provinces = {province.identifier: province for province in society.provinces.provinces}
    labels = {'central': '中央常设治理', 'delegated': '地方代管',
              'autonomous': '地方治理（中央通达未证实）', 'frontier': '无常设政权区'}
    for place in places:
        city = (cities[place['sourceId']] if place['kind'] == 'city' else
                cities[states[place['sourceId']].core_settlement_id] if place['kind'] == 'state' else
                cities[provinces[place['sourceId']].core_settlement_id] if place['kind'] == 'province' else None)
        if city is None:
            continue
        country = countries.get(city['countryId'])
        place['eraLabel'] = governance['eraLabel']
        place['administrationLabel'] = labels[city['administration']]
        place['centralTravelDays'] = city['centralTravelDays']
        place['nominalRealm'] = ({'id': city['nominalRealmId'], 'name': state_names[city['nominalRealmId']]}
                                 if country and city['nominalRealmId'] != city['countryId'] else None)
        if place['kind'] == 'state':
            place['responseBudgetDays'] = country['responseBudgetDays']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_governance(output, grid, society, governance)
    (output / "place-index.json").write_text(
        json.dumps(places, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    web = Path(__file__).with_name("web")
    for filename in ("atlas-ui.css", "atlas-ui.js", "atlas-tiles.js", "atlas-overview.js", "atlas-interaction.js", "atlas-ruler.js",
                     "atlas-navigation.js", "atlas-navigation-worker.js",
                     "city-character.js", "city-detail.js", "city-site.js", "city-map.js"):
        shutil.copyfile(web / filename, output / filename)


def ui_markup(*, world_name: str, legends: Mapping[str, str], summary: str = "", info_html: str = "") -> str:
    """Build map-software controls; supplied legends are actual rendered keys.

    This markup lives beside the viewport, so typing, sheet scrolling and
    control clicks never enter its map drag handlers. The parent owns CSS.
    """
    escape = html.escape
    icons = {
        "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/>',
        "layers": '<path d="m12 3 9 5-9 5-9-5zM3 12l9 5 9-5M3 16l9 5 9-5"/>',
        "globe": '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/>',
        "ruler": '<path d="m3 16 13-13 5 5L8 21zM7 12l2 2M10 9l2 2M13 6l2 2"/>',
        "navigation": '<circle cx="5" cy="5" r="2"/><circle cx="19" cy="19" r="2"/><path d="M5 7v7a3 3 0 0 0 3 3h8M17 3l4 4-4 4M21 7H11"/>',
        "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v.5"/>',
    }

    def icon(key):
        return f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{icons[key]}</svg>'

    def preview(key):
        return f'<img class="theme-preview" src="map-previews/{key}.png" alt="" width="256" height="128">'

    theme_buttons = "".join(
        f'<button type="button" class="theme-card" data-theme-button="{key}" aria-pressed="false">{preview(key)}<span>{label}</span></button>'
        for key, label in THEMES)
    legend_bodies = "".join(
        f'<div data-legend-key="{escape(key, quote=True)}" hidden>{body}</div>'
        for key, body in legends.items())

    def switches(items):
        return "".join(
            f'<label class="layer-switch"><input id="{identifier}" type="checkbox" data-layer="{layer}"'
            f'{"" if layer in {"graticule", "nominal-realms"} else " checked"}><span>{label}</span></label>'
            for identifier, layer, label in items)

    physical_switches = switches((
        ("toggle-elevation-bands", "elevation-bands", "高程"), ("toggle-bathymetry-bands", "bathymetry-bands", "海深"),
        ("toggle-graticule", "graticule", "经纬线"), ("toggle-coast", "coast", "海岸"),
        ("toggle-lakes", "lakes", "湖泊"), ("toggle-elevation-contours", "elevation-contours", "等高线"),
        ("toggle-snow", "snow", "积雪"), ("toggle-sea-ice", "sea-ice", "海冰"),
        ("toggle-polar-references", "polar-references", "极圈与极点"), ("toggle-rivers", "rivers", "河流"),
        ("toggle-landmarks", "geographic-symbols", "地貌符号"), ("toggle-geographic-labels", "geographic-labels", "地理名称"),
        ("toggle-cities", "cities", "城市聚落"), ("toggle-transport", "transport", "交通网络"),
        ("toggle-state-outline", "state-outline", "国界与国名"),
        ("toggle-nominal-realms", "nominal-realms", "名义势力范围"),
    ))
    monsoon_switches = switches((
        ("toggle-monsoon-precipitation", "monsoon-precipitation", "降水"),
        ("toggle-monsoon-wind", "monsoon-wind", "风场"),
        ("toggle-seasonal-rivers", "seasonal-rivers", "季节河流"),
    ))
    seasons = "".join(
        f'<button id="season-{key}" type="button" data-season-button="{key}" aria-pressed="false">{label}</button>'
        for key, label in SEASONS)
    return f'''
<section id="place-panel" class="place-panel" aria-label="地点搜索与详情">
  <div class="atlas-search">{icon('search')}<label class="visually-hidden" for="city-search">搜索地点</label>
    <input id="city-search" type="search" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="city-results" autocomplete="off" placeholder="搜索城市、国家、省份或地貌">
    <button id="search-clear" class="icon-button" type="button" aria-label="清除搜索" hidden>×</button>
  </div>
  <div id="search-suggestions" class="search-suggestions" hidden><p id="search-summary" class="panel-eyebrow"></p><ul id="city-results" class="place-results" role="listbox" aria-label="地点搜索结果"></ul></div>
  <section id="place-card" class="place-card" aria-labelledby="place-name" hidden>
    <div id="place-summary" class="place-summary">
    <button id="place-close" class="icon-button" type="button" aria-label="关闭地点详情">×</button>
    <p id="place-kind" class="panel-eyebrow"></p><h2 id="place-name"></h2>
    <dl id="place-facts" class="place-facts"></dl>
    <details id="place-extra" hidden><summary>行政与位置</summary><dl id="place-extra-facts"></dl></details>
    </div>
    <section id="city-map-section" aria-label="城市与周边地图" hidden>
      <div id="city-map-viewport" tabindex="0" aria-label="城市地图，可拖动和缩放">
        <svg id="city-map-canvas" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="城市、地形、水系与农田">
          <g id="city-map-camera"><g id="city-fabric-layer"></g></g>
        </svg>
        <div class="city-map-extents" aria-label="城市地图范围"><button id="city-map-city" type="button" aria-pressed="true">城市</button><button id="city-map-region" type="button" aria-pressed="false">周边</button></div>
        <div class="city-map-compass" aria-hidden="true">N<span>↑</span></div>
        <output id="city-map-status" role="status" aria-live="polite"></output>
        <nav class="city-map-tools" aria-label="城市地图缩放"><button id="city-map-zoom-in" type="button" aria-label="放大城市图">＋</button><button id="city-map-zoom-out" type="button" aria-label="缩小城市图">−</button><button id="city-map-reset" type="button" aria-label="复位城市图">⌂</button></nav>
        <div class="city-map-scale"><span id="city-scale-label"></span><i id="city-scale-rule"></i></div>
      </div>
      <div class="city-map-caption"><span><i class="city-key-building"></i>建筑</span><span><i class="city-key-water"></i>水系</span><span><i class="city-key-field"></i>农田</span><span>拖动平移 · 滚轮缩放</span></div>
    </section>
    <div class="panel-actions"><button id="place-recenter" class="map-button" type="button">在地图中定位</button><button id="place-regenerate" class="map-button" type="button" hidden>重新生成</button></div>
  </section>
</section>
<div id="world-heading" class="world-heading"><strong>{escape(world_name)}</strong><span>{escape(summary)}</span></div>
<nav class="map-toolbar" aria-label="地图工具">
  <button id="toggle-controls" class="map-button" type="button" title="地图图层" aria-label="地图图层" aria-controls="atlas-controls" aria-expanded="false">{icon('layers')}<span class="visually-hidden">图层</span></button>
  <a id="globe-link" class="map-button" href="globe.html" title="球体视图" aria-label="球体视图">{icon('globe')}<span class="visually-hidden">球体</span></a>
  <button id="ruler-toggle" class="map-button" type="button" title="测量距离" aria-label="测量距离" aria-pressed="false">{icon('ruler')}<span class="visually-hidden">测距</span></button>
  <button id="navigation-toggle" class="map-button" type="button" title="路线导航" aria-label="路线导航" aria-controls="navigation-panel" aria-expanded="false">{icon('navigation')}<span class="visually-hidden">导航</span></button>
  <button id="map-info-toggle" class="map-button" type="button" title="地图信息" aria-label="地图信息" aria-controls="map-info" aria-expanded="false">{icon('info')}<span class="visually-hidden">地图信息</span></button>
</nav>
<aside id="atlas-controls" class="map-panel layer-panel" aria-labelledby="layers-heading" hidden inert>
  <div class="panel-heading"><h2 id="layers-heading">地图图层</h2><button id="layers-close" class="icon-button" type="button" aria-label="关闭图层">×</button></div>
  <div class="mode-switch" aria-label="地图视图"><button id="view-physical" type="button" data-view-button="physical">地图</button><button id="view-monsoon" type="button" data-view-button="monsoon">季风</button><button id="view-tectonic" type="button" data-view-button="tectonic">板块</button></div>
  <div id="physical-controls"><h3>专题着色</h3><div id="physical-theme" class="theme-cards">{theme_buttons}</div>
    <details class="layer-group"><summary>辅助图层</summary><div class="layer-switches">{physical_switches}</div></details>
  </div>
  <div id="monsoon-controls" hidden><h3>季节</h3><div class="season-switch">{seasons}</div><h3>季风图层</h3><div class="layer-switches">{monsoon_switches}</div></div>
  <div id="tectonic-controls" hidden><p class="panel-note">查看当前世界的板块、运动方向与构造边界。</p></div>
</aside>
<nav class="map-tools" aria-label="地图缩放"><button id="zoom-in" type="button" aria-label="放大地图">＋</button><button id="zoom-out" type="button" aria-label="缩小地图">−</button><button id="zoom-reset" type="button" aria-label="返回全图">全图</button><output id="zoom-level" aria-live="polite">全图</output></nav>
<section id="map-legend" class="map-legend" aria-label="当前地图图例"><button id="legend-toggle" type="button" aria-controls="map-legend-body" aria-expanded="true"><span id="legend-title">地图图例</span></button><div id="map-legend-body">{legend_bodies}<details id="landform-legend" class="symbol-key"><summary>地貌符号</summary><div class="legend">{geographic_legend()}</div></details><details id="settlement-legend" class="symbol-key"><summary>聚落符号</summary><div class="legend">{settlement_legend()}</div></details><p id="nominal-legend" class="panel-note" hidden><span style="display:inline-block;width:26px;border-top:2px dashed #73583f;vertical-align:middle;margin-right:6px"></span>名义势力范围 · 实际国界保留</p></div></section>
<section id="ruler-card" class="ruler-card" aria-labelledby="ruler-heading" hidden><h2 id="ruler-heading">球面测距</h2><output id="ruler-summary" aria-live="polite">0 m</output><p id="ruler-hint" class="panel-note"></p><ol id="ruler-segments"></ol><div class="panel-actions"><button id="ruler-undo" class="map-button" type="button" disabled>撤销上一点</button><button id="ruler-clear" class="map-button" type="button" disabled>清空</button><button id="ruler-finish" class="map-button" type="button" disabled>完成</button></div></section>
<section id="navigation-panel" class="map-panel navigation-panel" aria-labelledby="navigation-heading" hidden>
  <div class="panel-heading"><h2 id="navigation-heading">路线导航</h2><button id="navigation-close" class="icon-button" type="button" aria-label="关闭导航">×</button></div>
  <form id="navigation-form">
    <label for="navigation-start">起点</label><div class="navigation-location"><input id="navigation-start" list="navigation-cities" placeholder="输入聚落名称" autocomplete="off" required><button id="navigation-pick-start" type="button" title="在地图上选择起点">选点</button></div>
    <button id="navigation-swap" class="navigation-swap" type="button" aria-label="交换起终点">↕ 交换</button>
    <label for="navigation-end">终点</label><div class="navigation-location"><input id="navigation-end" list="navigation-cities" placeholder="输入聚落名称" autocomplete="off" required><button id="navigation-pick-end" type="button" title="在地图上选择终点">选点</button></div>
    <datalist id="navigation-cities"></datalist>
    <label for="navigation-mode">交通方式</label><select id="navigation-mode" aria-label="交通方式"></select>
    <button id="navigation-go" class="navigation-go" type="submit">规划路线</button>
  </form>
  <p id="navigation-status" class="panel-note" role="status" aria-live="polite"></p>
  <div id="navigation-result" hidden><output id="navigation-distance"></output><p id="navigation-time"></p><p id="navigation-detail" class="panel-note"></p><button id="navigation-clear" type="button" class="map-button">清除路线</button></div>
  <details class="navigation-rules"><summary>行程规则</summary><p id="navigation-rules" class="panel-note"></p></details>
</section>
<section id="map-info" class="map-panel map-info" aria-label="地图信息" hidden inert><div class="panel-heading"><h2>地图信息</h2><button id="map-info-close" class="icon-button" type="button" aria-label="关闭地图信息">×</button></div>{info_html}</section>
<output id="map-status" class="map-status" role="status" aria-live="polite"></output>
'''
