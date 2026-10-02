"""Independently check the detailed tile payloads consumed by the live atlas."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import shapely


_spec = importlib.util.spec_from_file_location(
    "tile_polygon_decoder", Path(__file__).with_name("check_coastal_coverage.py"),
)
_svg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_svg)
_CLIP_REFERENCE = re.compile(r"url\(#([^)]*)\)")
MAX_TILE_BYTES = 2 * 1024 * 1024
BASE_LEVELS = [{"id":"regional","minScale":8}, {"id":"local","minScale":32},
               {"id":"detail","minScale":64}]


def base_levels(manifest):
    """The three complete maps; city-detail only adds physical relief."""
    return manifest["levels"][:3]


def read_manifest(review: Path):
    manifest = json.loads((review / "atlas-manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or any(
        type(manifest.get(key)) is not int or manifest[key] <= 0
        for key in ("width", "height", "tileSize", "columns", "rows")
    ):
        raise ValueError("Atlas manifest must declare positive native integer dimensions")
    if (manifest["columns"] != math.ceil(manifest["width"] / manifest["tileSize"])
            or manifest["rows"] != math.ceil(manifest["height"] / manifest["tileSize"])):
        raise ValueError("Atlas manifest tile counts do not cover its declared world")
    themes = manifest.get("themes")
    if (not isinstance(themes, list) or "none" not in themes or not themes
            or any(not isinstance(theme, str) or not theme for theme in themes)
            or len(set(themes)) != len(themes)):
        raise ValueError("Atlas manifest must declare unique themes including none")
    levels = manifest.get("levels")
    if "detailMinScale" in manifest or not isinstance(levels, list) or not levels:
        raise ValueError("Atlas manifest must declare its ordered detail levels")
    previous, identifiers = 0, set()
    for level in levels:
        if (not isinstance(level, dict) or not isinstance(level.get("id"), str)
                or not re.fullmatch(r"[a-z][a-z0-9-]*",level["id"]) or level["id"] in identifiers
                or type(level.get("minScale")) not in (int,float)
                or not math.isfinite(level["minScale"]) or level["minScale"] <= previous):
            raise ValueError("Atlas levels must have unique safe IDs and increasing finite minScale")
        previous = level["minScale"]
        identifiers.add(level["id"])
    if len(levels) != 4 or levels[:3] != BASE_LEVELS:
        raise ValueError("Atlas manifest levels differ from the actual frontend contract")
    overlay = levels[3]
    if (set(overlay) != {"id","minScale","kind","publishedTiles"}
            or overlay["id"] != "city-detail" or overlay["minScale"] != 128
            or overlay["kind"] != "relief-overlay"
            or not isinstance(overlay["publishedTiles"],list)):
        raise ValueError("Atlas city-detail must declare its sparse relief-overlay tiles")
    published = overlay["publishedTiles"]
    for key in published:
        if not isinstance(key,str) or not re.fullmatch(r"(?:0|[1-9]\d*)-(?:0|[1-9]\d*)",key):
            raise ValueError("Atlas city-detail declares an unsafe tile key")
        row,column = map(int,key.split("-"))
        if row >= manifest["rows"] or column >= manifest["columns"]:
            raise ValueError("Atlas city-detail declares a tile outside the world")
    if len(set(published)) != len(published):
        raise ValueError("Atlas city-detail declares duplicate published tiles")
    expected_count = manifest["rows"]*manifest["columns"]*3 + len(published)
    stats = manifest.get("stats")
    if (not isinstance(stats,dict) or set(stats) != {"totalTiles","maxTileBytes","totalTileBytes"}
            or any(type(value) is not int or value <= 0 for value in stats.values())
            or stats["totalTiles"] != expected_count
            or stats["maxTileBytes"] > MAX_TILE_BYTES or stats["totalTileBytes"] < stats["maxTileBytes"]):
        raise ValueError("Atlas manifest must report measured tiles within the 2 MiB per-tile budget")
    return manifest


def tile_document(markup: str):
    return ET.fromstring(
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
        + markup + '</svg>'
    )


def tag(element):
    return element.tag.rsplit("}", 1)[-1]


def read_tile(review: Path, manifest, level: str, row: int, column: int):
    if level not in {item["id"] for item in base_levels(manifest)}:
        raise ValueError("Only complete base levels can be read as thematic tiles")
    if not (0 <= row < manifest["rows"] and 0 <= column < manifest["columns"]):
        raise ValueError("Tile coordinates are outside the declared world")
    path = review / "tiles" / level / f"{row}-{column}.json"
    raw = path.read_bytes()
    if len(raw) > MAX_TILE_BYTES:
        raise ValueError(f"{level}/{path.name}: measured payload exceeds the 2 MiB per-tile budget")
    payload = json.loads(raw.decode("utf-8"))
    x, y = column * manifest["tileSize"], row * manifest["tileSize"]
    bounds = [x, y, min(manifest["tileSize"], manifest["width"] - x),
              min(manifest["tileSize"], manifest["height"] - y)]
    if (not isinstance(payload, dict) or payload.get("bounds") != bounds
            or any(not isinstance(payload.get(section), str) for section in ("surface", "ink"))
            or not isinstance(payload.get("themes"), dict)
            or set(payload["themes"]) != set(manifest["themes"])
            or any(not isinstance(markup, str) for markup in payload["themes"].values())):
        raise ValueError(f"{path.name}: tile bounds or paint sections contradict the manifest")
    if payload["themes"]["none"]:
        raise ValueError(f"{path.name}: theme none must contain no thematic paint")
    clip_ids = {"land":f"tile-land-{level}-{row}-{column}", "water":f"tile-water-{level}-{row}-{column}"}
    surface = tile_document(payload["surface"])
    definitions = [node for node in surface.iter() if tag(node) == "clipPath"]
    if len(definitions) != 2 or {node.get("id") for node in definitions} != set(clip_ids.values()):
        raise ValueError(f"{path.name}: tile must own exactly its unique native land and water clips")
    clips = {}
    for node in definitions:
        if node.get("clipPathUnits") != "userSpaceOnUse":
            raise ValueError(f"{path.name}: tile clip must use native coordinates")
        paths = [path.attrib for path in node.iter() if tag(path) == "path"]
        if not paths:
            raise ValueError(f"{path.name}: tile clip lacks an explicit path")
        clips[node.get("id")] = _svg.union_paths(paths)
    # Parse static surface/ink once. Every theme combines with those exact
    # sections in the browser, so check their IDs and references together.
    base_ids = []
    for section, composition in (("surface", surface), ("ink", tile_document(payload["ink"]))):
        base_ids.extend(node.get("id") for node in composition.iter() if node.get("id"))
        for node in composition.iter():
            reference = node.get("clip-path")
            if reference is not None and reference != "none":
                match = _CLIP_REFERENCE.fullmatch(reference)
                if not match or match[1] not in clips:
                    raise ValueError(f"{path.name}/{section}: clip reference is not owned by this tile")
    if len(base_ids) != len(set(base_ids)):
        raise ValueError(f"{path.name}: duplicate IDs in mounted tile surface/ink composition")
    # Paint servers are definitions, never classified ground geometry. Their
    # references must nevertheless resolve in this exact mounted native tile.
    for composition in (surface,tile_document(payload["ink"])):
        for node in composition.iter():
            for value in node.attrib.values():
                for identifier in re.findall(r"url\(#([^)]*)\)",value):
                    if identifier not in base_ids:
                        raise ValueError(f"{path.name}: paint-server reference is not owned by this tile")
    tile_id_suffix = re.compile(rf"-tile-{re.escape(level)}-{row}-{column}(?:-part-\d+)?$")
    if any(identifier not in clips and not tile_id_suffix.search(identifier) for identifier in base_ids):
        raise ValueError(f"{path.name}: feature ID is not unique to this native tile")
    for theme, markup in payload["themes"].items():
        composition = tile_document(markup)
        ids = base_ids + [node.get("id") for node in composition.iter() if node.get("id")]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path.name}/{theme}: duplicate IDs in mounted tile composition")
        if any(identifier not in clips and not tile_id_suffix.search(identifier) for identifier in ids):
            raise ValueError(f"{path.name}/{theme}: feature ID is not unique to this native tile")
        for node in composition.iter():
            for attribute,value in node.attrib.items():
                if attribute == "clip-path":
                    continue
                if any(identifier not in ids for identifier in re.findall(r"url\(#([^)]*)\)",value)):
                    raise ValueError(f"{path.name}/{theme}: paint-server reference is not owned by this composition")
        for node in composition.iter():
            reference = node.get("clip-path")
            if reference is not None and reference != "none":
                match = _CLIP_REFERENCE.fullmatch(reference)
                if not match or match[1] not in clips:
                    raise ValueError(f"{path.name}/{theme}: clip reference is not owned by this tile")
    rectangle = shapely.box(x, y, x + bounds[2], y + bounds[3])
    return {"payload":payload, "land":clips[clip_ids["land"]],
            "water":clips[clip_ids["water"]], "rectangle":rectangle,
            "clips":clips, "landClipId":clip_ids["land"], "key":f"{level}/{row}-{column}",
            "byteLength":len(raw)}


def check_published_city_files(review: Path, manifest):
    directory = review / "tiles" / "city-detail"
    expected = {f"{key}.json" for key in manifest["levels"][3]["publishedTiles"]}
    if not directory.is_dir():
        raise ValueError("Atlas must publish its city-detail directory, including an empty sparse level")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != expected or any(path.is_dir() for path in directory.iterdir()):
        raise ValueError(f"City-detail files differ from publishedTiles: missing={sorted(expected-actual)}, undeclared={sorted(actual-expected)}")


def _attributes(node):
    attributes = dict(node.attrib)
    for rule in attributes.get("style", "").split(";"):
        if ":" in rule:
            name,value = rule.split(":",1)
            attributes[name.strip()] = value.strip()
    return attributes


def read_city_overlay(review: Path, manifest, row: int, column: int, *, base_tile=None):
    """Read the additive surface/ink with its actual detail base definitions."""
    key = f"{row}-{column}"
    if key not in manifest["levels"][3]["publishedTiles"]:
        raise ValueError("City-detail tile is not in publishedTiles")
    base = base_tile if base_tile is not None else read_tile(review,manifest,"detail",row,column)
    if base["key"] != f"detail/{key}":
        raise ValueError("City-detail must compose with the matching detail base tile")
    path = review / "tiles" / "city-detail" / f"{key}.json"
    raw = path.read_bytes()
    if len(raw) > MAX_TILE_BYTES:
        raise ValueError(f"city-detail/{key}: measured payload exceeds the 2 MiB per-tile budget")
    payload = json.loads(raw.decode("utf-8"))
    if (not isinstance(payload,dict) or set(payload) != {"bounds","surface","ink"}
            or payload["bounds"] != base["payload"]["bounds"]
            or any(not isinstance(payload[section],str) for section in ("surface","ink"))):
        raise ValueError(f"city-detail/{key}: additive payload must have matching bounds and only surface/ink; no themes")
    surface,ink = tile_document(payload["surface"]),tile_document(payload["ink"])
    nodes = list(surface.iter()) + list(ink.iter())
    ids = [node.get("id") for node in nodes if node.get("id")]
    base_ids = {node.get("id") for markup in (base["payload"]["surface"],base["payload"]["ink"])
                for node in tile_document(markup).iter() if node.get("id")}
    for markup in base["payload"]["themes"].values():
        base_ids.update(node.get("id") for node in tile_document(markup).iter() if node.get("id"))
    suffix = re.compile(rf"-city-detail-{row}-{column}$")
    if (len(ids) != len(set(ids)) or set(ids) & base_ids
            or any(not suffix.search(identifier) for identifier in ids)):
        raise ValueError(f"city-detail/{key}: definitions must have unique IDs for this additive tile")
    if any(tag(node) == "clipPath" for node in nodes):
        raise ValueError(f"city-detail/{key}: additive relief must reuse detail clips without repeating shore geometry")
    mask_id = f"city-relief-mask-city-detail-{key}"
    masks = {node.get("id"):node for node in nodes if tag(node) == "mask"}
    if set(masks) != {mask_id}:
        raise ValueError(f"city-detail/{key}: one native alpha support mask is mandatory")
    for identifier,kind in ((mask_id,"alpha"),):
        mask = masks[identifier]
        if (mask.get("maskUnits") != "userSpaceOnUse" or mask.get("maskContentUnits") != "userSpaceOnUse"
                or _attributes(mask).get("mask-type") != kind
                or any(float(mask.get(name,"nan")) != value for name,value in
                       zip(("x","y","width","height"),payload["bounds"]))):
            raise ValueError(f"city-detail/{key}: native alpha support must match the tile bounds")
    gradients = {node.get("id"):node for node in nodes if tag(node) == "radialGradient"}
    ellipses = list(masks[mask_id])
    if not gradients or not ellipses or any(tag(node) != "ellipse" for node in ellipses):
        raise ValueError(f"city-detail/{key}: city support needs radial gradients and explicit ellipses")
    used_gradients = set()
    for ellipse in ellipses:
        reference = _CLIP_REFERENCE.fullmatch(ellipse.get("fill",""))
        if not reference or reference[1] not in gradients:
            raise ValueError(f"city-detail/{key}: support ellipse references no local gradient")
        gradient = gradients[reference[1]]
        matrix = re.fullmatch(r"matrix\(([^)]*)\)",gradient.get("gradientTransform",""))
        transform = [float(value) for value in re.split(r"[\s,]+",matrix[1].strip())] if matrix else []
        center_radius = [float(ellipse.get(name,"nan")) for name in ("cx","cy","rx","ry")]
        cx,cy,rx,ry = center_radius
        if (not all(math.isfinite(value) for value in center_radius) or rx <= 0 or ry <= 0
                or transform != [rx,0,0,ry,cx,cy]
                or gradient.get("gradientUnits") != "userSpaceOnUse"
                or any(float(gradient.get(name,"nan")) != value for name,value in (("cx",0),("cy",0),("r",1)))):
            raise ValueError(f"city-detail/{key}: radial gradient does not match its city support ellipse")
        stops = list(gradient)
        if (len(stops) != 3 or any(tag(stop) != "stop" for stop in stops)
                or [float(stop.get("offset","nan")) for stop in stops] != [0,.65,1]
                or [float(stop.get("stop-opacity","1")) for stop in stops] != [1,1,0]
                or any(stop.get("stop-color") != "white" for stop in stops)):
            raise ValueError(f"city-detail/{key}: city support must fade continuously to zero at its boundary")
        used_gradients.add(reference[1])
    if used_gradients != set(gradients):
        raise ValueError(f"city-detail/{key}: unreferenced city support gradients")
    allowed_references = set(ids) | set(base["clips"])
    for node in nodes:
        for value in node.attrib.values():
            for identifier in _CLIP_REFERENCE.findall(value):
                if identifier not in allowed_references:
                    raise ValueError(f"city-detail/{key}: foreign clip/mask/gradient reference")
    path_counts = {"surface":0,"ink":0}
    for section,document,layer in (("surface",surface,"elevation-bands"),("ink",ink,"elevation-contours")):
        def visit(node, *, in_defs=False, land=False, masked=False, refinement=False, active_layer=None):
            in_defs = in_defs or tag(node) == "defs"
            attributes = _attributes(node)
            if in_defs:
                return
            if "data-tile-layer" in attributes:
                active_layer = attributes["data-tile-layer"]
                if active_layer != layer:
                    raise ValueError(f"city-detail/{key}: additive relief may not duplicate shore, themes or other base layers")
            clip = attributes.get("clip-path")
            if clip is not None and clip != f'url(#{base["landClipId"]})':
                raise ValueError(f"city-detail/{key}: paint must use its exact shared detail land clip")
            land = land or clip == f'url(#{base["landClipId"]})'
            mask = attributes.get("mask")
            if mask is not None and mask != f"url(#{mask_id})":
                raise ValueError(f"city-detail/{key}: paint must use its own city support mask")
            masked = masked or mask == f"url(#{mask_id})"
            refinement = refinement or attributes.get("data-city-refinement") == "true"
            if tag(node) == "path":
                if (section != "ink" or not land or not masked or not refinement or active_layer != layer
                        or attributes.get("data-city-relief") != "true"
                        or attributes.get("data-contour-kind") != "minor"
                        or attributes.get("data-source-field") != "accepted-continuous-ground"
                        or not math.isfinite(float(attributes.get("data-height-m", "nan")))):
                    raise ValueError(f"city-detail/{key}: fine paint lacks mandatory shared-land/city support clipping")
                path_counts[section] += 1
            elif tag(node) not in {"svg","g"}:
                raise ValueError(f"city-detail/{key}: unexpected visible additive geometry")
            for child in node:
                visit(child,in_defs=in_defs,land=land,masked=masked,refinement=refinement,active_layer=active_layer)
        visit(document)
    if path_counts["surface"] or not path_counts["ink"]:
        raise ValueError(f"city-detail/{key}: published city relief must contain only true minor contours")
    return {**base,"payload":payload,"key":f"city-detail/{key}","composedBase":"detail",
            "byteLength":len(raw),"supportEllipseCount":len(ellipses),"finePathCounts":path_counts,
            "citySupportMaskId":mask_id}


def painted_regions(markup: str, tile, *, owner_attribute: str | None = None):
    """Decode actual fill and ancestor clipping without importing the producer."""
    regions = []

    def visit(node, inherited_fill="black", inherited_fill_opacity=1.0,
              clip=None, has_land_clip=False, owner=None):
        if tag(node) in {"defs", "clipPath", "pattern"}:
            return
        attributes = dict(node.attrib)
        for rule in attributes.get("style", "").split(";"):
            if ":" in rule:
                name, value = rule.split(":", 1)
                attributes[name.strip()] = value.strip()
        if ("hidden" in attributes or attributes.get("display") == "none"
                or attributes.get("visibility") == "hidden"
                or float(attributes.get("opacity", "1")) == 0):
            return
        fill = attributes.get("fill", inherited_fill)
        fill_opacity = float(attributes.get("fill-opacity", inherited_fill_opacity))
        if owner_attribute and owner_attribute in attributes:
            owner = int(attributes[owner_attribute])
        reference = attributes.get("clip-path")
        if reference and reference != "none":
            match = _CLIP_REFERENCE.fullmatch(reference)
            if not match or match[1] not in tile["clips"]:
                raise ValueError(f'{tile["key"]}: thematic path uses a foreign physical clip')
            physical_clip = tile["clips"][match[1]]
            clip = physical_clip if clip is None else clip.intersection(physical_clip)
            has_land_clip = has_land_clip or match[1] == tile["landClipId"]
        if tag(node) == "path" and fill != "none" and fill_opacity > 0:
            if not has_land_clip:
                raise ValueError(f'{tile["key"]}: thematic paint lacks this tile\'s authoritative land clip')
            geometry = _svg.path_geometry(attributes.get("d", ""))
            if clip is not None:
                geometry = geometry.intersection(clip)
            geometry = geometry.intersection(tile["rectangle"])
            if not geometry.is_empty:
                if owner_attribute and owner is None:
                    raise ValueError(f'{tile["key"]}: administrative paint lacks {owner_attribute}')
                regions.append((owner, geometry))
        for child in node:
            visit(child, fill, fill_opacity, clip, has_land_clip, owner)

    visit(tile_document(markup))
    return regions


def tile_owner_regions(tile, theme, attribute):
    grouped = {}
    for owner, geometry in painted_regions(tile["payload"]["themes"][theme], tile,
                                          owner_attribute=attribute):
        grouped.setdefault(owner, []).append(geometry)
    return {owner:shapely.union_all(parts) for owner,parts in grouped.items()}


def read_overview(review: Path, manifest, theme: str):
    markup = (review / f"overview-{theme}.svg").read_text(encoding="utf-8")
    root = tile_document(markup)
    clips = {}
    for node in root.iter():
        if tag(node) == "clipPath":
            identifier = node.get("id")
            if identifier in clips or node.get("clipPathUnits") != "userSpaceOnUse":
                raise ValueError("Actual overview has duplicate or non-native physical clips")
            clips[identifier] = _svg.union_paths([path.attrib for path in node.iter() if tag(path) == "path"])
    if set(clips) != {"land-silhouette-clip","water-silhouette-clip"}:
        raise ValueError("Actual overview lacks its shared physical land/water clips")
    return {"payload":{"themes":{theme:markup}},"key":f"overview/{theme}",
            "clips":clips,"landClipId":"land-silhouette-clip",
            "land":clips["land-silhouette-clip"],"water":clips["water-silhouette-clip"],
            "rectangle":shapely.box(0,0,manifest["width"],manifest["height"])}


def _check_overview(review: Path, manifest, water):
    land, ocean = None, None
    themes = {}
    for theme in manifest["themes"]:
        if theme == "none":
            continue
        overview = read_overview(review,manifest,theme)
        if land is None:
            land, ocean = overview["land"], overview["water"]
        elif not land.equals(overview["land"]) or not ocean.equals(overview["water"]):
            raise ValueError("Actual overview themes do not share one physical silhouette")
        fills = painted_regions(overview["payload"]["themes"][theme],overview)
        missing = overview["land"].difference(shapely.union_all([geometry for _,geometry in fills]))
        themes[theme] = {"missingArea":float(missing.area),
                         "example":None if missing.is_empty else list(missing.representative_point().coords[0])}
    if land is None:
        raise ValueError("Actual overview must publish a thematic land silhouette")
    native = _svg.native_shoreline_metrics(land,water)
    rectangle = shapely.box(0,0,manifest["width"],manifest["height"])
    physical = {"landWaterOverlapArea":float(land.intersection(ocean).area),
                "unpaintedWorldArea":float(rectangle.difference(land.union(ocean)).area),
                "outsideWorldArea":float(land.union(ocean).difference(rectangle).area)}
    passed = (native["mismatchCount"] == 0 and all(value < .01 for value in physical.values())
              and all(metrics["missingArea"] < .01 for metrics in themes.values()))
    return {"status":"ok" if passed else "failed","sharedLandClipIdentical":True,
            "nativeShoreline":native,"physicalCoverage":physical,"themes":themes}


def _empty_level_metrics(manifest):
    return {"checkedTiles":0,"payloadBytes":{"maxTileBytes":0,"totalTileBytes":0},
            "themes":{theme:{"missingArea":0.0,"missingTileCount":0,"examples":[]}
                      for theme in manifest["themes"] if theme != "none"},
            "nativeShoreline":{"checkedNativeCenters":0,"landCentersDisplayedAsWater":0,
                               "waterCentersDisplayedAsLand":0,"mismatchCount":0,"examples":[]},
            "physicalCoverage":{"landWaterOverlapArea":0.0,"unpaintedWorldArea":0.0,"outsideTileArea":0.0}}


def _finish_level_metrics(metrics):
    native = metrics["nativeShoreline"]
    native["mismatchCount"] = native["landCentersDisplayedAsWater"] + native["waterCentersDisplayedAsLand"]
    passed = (native["mismatchCount"] == 0 and all(value < .01 for value in metrics["physicalCoverage"].values())
              and all(result["missingArea"] < .01 for result in metrics["themes"].values()))
    metrics["status"] = "ok" if passed else "failed"
    return metrics


def _check_level(review: Path, manifest, level: str, water):
    result = _empty_level_metrics(manifest)
    theme_metrics,native,physical = result["themes"],result["nativeShoreline"],result["physicalCoverage"]
    overlay_result = _empty_level_metrics(manifest) if level == "detail" else None
    published = set(manifest["levels"][3]["publishedTiles"])
    checked = 0
    sizes = result["payloadBytes"]
    for row in range(manifest["rows"]):
        for column in range(manifest["columns"]):
            tile = read_tile(review, manifest, level, row, column)
            sizes["maxTileBytes"] = max(sizes["maxTileBytes"],tile["byteLength"])
            sizes["totalTileBytes"] += tile["byteLength"]
            land, ocean, rectangle = tile["land"], tile["water"], tile["rectangle"]
            local_physical = {"landWaterOverlapArea":float(land.intersection(ocean).area),
                              "unpaintedWorldArea":float(rectangle.difference(land.union(ocean)).area),
                              "outsideTileArea":float(land.union(ocean).difference(rectangle).area)}
            for name,value in local_physical.items():
                physical[name] += value
            x,y,width,height = tile["payload"]["bounds"]
            points_x, points_y = np.meshgrid(np.arange(x,x+width) + .5, np.arange(y,y+height) + .5)
            shapely.prepare(land)
            displayed = shapely.covers(land, shapely.points(points_x, points_y))
            expected = water[y:y+height, x:x+width] == 0
            local_native = {"checkedNativeCenters":int(expected.size),
                            "landCentersDisplayedAsWater":int(np.count_nonzero(expected & ~displayed)),
                            "waterCentersDisplayedAsLand":int(np.count_nonzero(~expected & displayed))}
            for name,value in local_native.items():
                native[name] += value
            for point_row,point_column in np.argwhere(expected != displayed)[:max(0,10-len(native["examples"]))]:
                native["examples"].append({"row":int(y+point_row),"column":int(x+point_column),
                                           "expectedLand":bool(expected[point_row,point_column]),
                                           "displayedLand":bool(displayed[point_row,point_column])})
            local_themes = {}
            for theme,metrics in theme_metrics.items():
                fills = painted_regions(tile["payload"]["themes"][theme], tile)
                painted = shapely.union_all([geometry for _,geometry in fills])
                missing = land.difference(painted)
                area = float(missing.area)
                local_themes[theme] = (area,None if missing.is_empty else list(missing.representative_point().coords[0]))
                metrics["missingArea"] += area
                if area > 1e-9:
                    metrics["missingTileCount"] += 1
                    if len(metrics["examples"]) < 8:
                        metrics["examples"].append({"tile":tile["key"],"missingArea":area,
                                                     "point":list(missing.representative_point().coords[0])})
            if overlay_result is not None and f"{row}-{column}" in published:
                overlay = read_city_overlay(review,manifest,row,column,base_tile=tile)
                overlay_result["checkedTiles"] += 1
                overlay_result["payloadBytes"]["maxTileBytes"] = max(
                    overlay_result["payloadBytes"]["maxTileBytes"],overlay["byteLength"])
                overlay_result["payloadBytes"]["totalTileBytes"] += overlay["byteLength"]
                for name,value in local_physical.items():
                    overlay_result["physicalCoverage"][name] += value
                for name,value in local_native.items():
                    overlay_result["nativeShoreline"][name] += value
                for theme,(area,point) in local_themes.items():
                    metrics = overlay_result["themes"][theme]
                    metrics["missingArea"] += area
                    if area > 1e-9:
                        metrics["missingTileCount"] += 1
                        if len(metrics["examples"]) < 8:
                            metrics["examples"].append({"tile":overlay["key"],"missingArea":area,"point":point})
            checked += 1
    result["checkedTiles"] = checked
    if overlay_result is not None:
        result["cityOverlay"] = {**_finish_level_metrics(overlay_result),"kind":"relief-overlay",
                                 "composedBase":"detail","sharedBaseLandClip":True,
                                 "supportMaskVerified":True,"baseMajorContoursPreserved":True,
                                 "compositionOrder":["detail.surface","city-detail.surface","detail.theme",
                                                     "city-detail.ink","detail.ink"]}
    return _finish_level_metrics(result)


def check_review(review: Path, *, grid: Path):
    manifest = read_manifest(review)
    check_published_city_files(review,manifest)
    with np.load(grid / "world-grid.npz", allow_pickle=False) as source:
        water = source["water"]
    if water.shape != (manifest["height"], manifest["width"]):
        raise ValueError("Canonical water array does not match the published tile manifest")
    levels = {level["id"]:_check_level(review,manifest,level["id"],water) for level in base_levels(manifest)}
    levels["city-detail"] = levels["detail"].pop("cityOverlay")
    measured = {"totalTiles":sum(result["checkedTiles"] for result in levels.values()),
                "maxTileBytes":max(result["payloadBytes"]["maxTileBytes"] for result in levels.values()),
                "totalTileBytes":sum(result["payloadBytes"]["totalTileBytes"] for result in levels.values())}
    if measured != manifest["stats"]:
        raise ValueError(f"Actual tile bytes differ from manifest statistics: measured={measured}, declared={manifest['stats']}")
    overview = _check_overview(review,manifest,water)
    passed = overview["status"] == "ok" and all(metrics["status"] == "ok" for metrics in levels.values())
    native = {name:sum(metrics["nativeShoreline"][name] for metrics in levels.values())
              for name in ("checkedNativeCenters","landCentersDisplayedAsWater",
                           "waterCentersDisplayedAsLand","mismatchCount")}
    return {"schema":"published-atlas-tiles-v3","status":"ok" if passed else "failed",
            "consumer":"atlas-manifest.json and every tiles/{level}/{row}-{column}.json consumed by atlas-tiles.js",
            "criterion":"In actual overview, three complete base levels and declared city-detail additive compositions: zero canonical native-centre land/water changes; total missing area below .01 native cell squared for each base theme; shared physical clips and mandatory city support masks; no undeclared city tiles or repeated themes/shore.",
            "checkedTiles":sum(metrics["checkedTiles"] for metrics in levels.values()),
            "manifest":manifest,"nativeShoreline":native,"levels":levels,"overview":overview,
            "payloadBudget":{"perTileLimitBytes":MAX_TILE_BYTES,"verified":True,**measured}}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("review", type=Path)
    cli.add_argument("--grid", type=Path, required=True, help="Canonical grid directory containing world-grid.npz")
    cli.add_argument("--report", type=Path)
    args = cli.parse_args()
    report = check_review(args.review, grid=args.grid)
    destination = args.report or args.review / "atlas-tiles-check.json"
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2) + "\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"report":str(destination),
                      "checkedTiles":report["checkedTiles"],
                      "nativeMismatches":report["nativeShoreline"]["mismatchCount"],
                      "themeMissingAreas":{level:{theme:metrics["missingArea"] for theme,metrics in result["themes"].items()}
                                           for level,result in report["levels"].items()},
                      "overviewNativeMismatches":report["overview"]["nativeShoreline"]["mismatchCount"],
                      "overviewThemeMissingAreas":{theme:metrics["missingArea"] for theme,metrics in report["overview"]["themes"].items()}}))
    raise SystemExit(0 if report["status"] == "ok" else 1)
