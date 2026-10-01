"""Slice authoritative cartographic geometry into native-coordinate SVG blocks."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import html
import json
import logging
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import shapely

from shapely import STRtree, is_valid_reason, union_all
from shapely.geometry import GeometryCollection, box
from shapely.geometry.base import BaseGeometry

from .svg_paths import COORDINATE_SCALE, integer_subpath_data


@dataclass(frozen=True)
class TileFeature:
    geometry: BaseGeometry
    attributes: dict[str, str]
    layer: str
    theme: str | None = None
    section: str = "ink"
    path_data: str | None = None
    definitions: Mapping[str, str] | None = None


@dataclass(frozen=True)
class TileLevel:
    id: str
    min_scale: float
    land_surface: BaseGeometry
    features: Iterable[TileFeature]


_ATTRIBUTE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_PRECISION = COORDINATE_SCALE
_MAX_TILE_BYTES = 2 * 1024 * 1024


def _segment(coordinates, *, closed: bool) -> str:
    # Subtract rounded absolute integer coordinates, so long relative paths
    # retain their shared endpoints without cumulative rounding drift.
    rounded = []
    for x, y, *_ in coordinates:
        point = (round(x * _PRECISION), round(y * _PRECISION))
        if not rounded or rounded[-1] != point:
            rounded.append(point)
    return integer_subpath_data(rounded, closed=closed)


def _geometry_paths(geometry: BaseGeometry):
    """Yield polygon and line paths separately; point intersections draw no ink."""
    kind = geometry.geom_type
    if kind == "Polygon":
        pieces = [_segment(geometry.exterior.coords, closed=True)]
        pieces.extend(_segment(ring.coords, closed=True) for ring in geometry.interiors)
        data = " ".join(piece for piece in pieces if piece)
        if data:
            yield data, True
    elif kind in {"LineString", "LinearRing"}:
        data = _segment(geometry.coords, closed=False)
        if data:
            yield data, False
    elif kind == "GeometryCollection":
        parts = []
        pending = list(reversed(geometry.geoms))
        while pending:
            part = pending.pop()
            if part.geom_type in {"GeometryCollection", "MultiPolygon", "MultiLineString"}:
                pending.extend(reversed(part.geoms))
            else:
                parts.append(part)
        polygons = [part for part in parts if part.geom_type == "Polygon"]
        if polygons:
            # Collection polygons can overlap while the collection is valid.
            # One even-odd path must describe their geometric union, rather
            # than cancelling the overlap by concatenating their rings.
            yield from _geometry_paths(union_all(polygons))
        for part in parts:
            if part.geom_type != "Polygon":
                yield from _geometry_paths(part)
    elif kind in {"MultiPolygon", "MultiLineString"}:
        for part in geometry.geoms:
            yield from _geometry_paths(part)


def geometry_path_data(geometry: BaseGeometry) -> str:
    """Serialize native geometry with shared eight-decimal absolute endpoints."""
    return " ".join(data for data, _ in _geometry_paths(geometry))


def feature_definitions(features, painted_markup, id_map=None) -> str:
    """Publish each referenced cartographic paint server once in this block."""
    definitions = {}
    for feature in features:
        for identifier, markup in (feature.definitions or {}).items():
            if identifier in definitions and definitions[identifier] != markup:
                raise ValueError("cartographic definitions must have one unambiguous identity")
            definitions[identifier] = markup
    parsed = {}
    for identifier, markup in definitions.items():
        root = ET.fromstring(markup)
        if root.tag != "pattern" or root.attrib.get("id") != identifier:
            raise ValueError("cartographic paint definitions must declare their own pattern identity")
        if id_map:
            for node in root.iter():
                for name,value in list(node.attrib.items()):
                    if name == "id":
                        node.set(name,id_map.get(value,value))
                    else:
                        node.set(name,re.sub(r"url\(#([^)]+)\)",
                            lambda match:f'url(#{id_map.get(match[1],match[1])})',value))
        parsed[root.attrib["id"]]=root
    used=set()
    pending=re.findall(r"url\(#([^)]+)\)",painted_markup)
    while pending:
        identifier=pending.pop()
        if identifier in used or identifier not in parsed:
            continue
        used.add(identifier)
        pending.extend(re.findall(r"url\(#([^)]+)\)",ET.tostring(parsed[identifier],encoding="unicode")))
    return "".join(ET.tostring(root,encoding="unicode")for identifier,root in parsed.items()if identifier in used)


def feature_markup(
    feature: TileFeature,
    geometry: BaseGeometry | None = None,
    *,
    clip_ids: Mapping[str, str] | None = None,
    id_map: Mapping[str, str] | None = None,
) -> str:
    """Render a feature; an explicit curve path is retained byte for byte.

    ``geometry`` overrides the source only for non-curve features. The optional
    id map makes repeated detail feature IDs and their references tile-local.
    """
    if clip_ids is None:
        clip_ids = {"land":"land-silhouette-clip", "water":"water-silhouette-clip"}
    if geometry is None:
        geometry = feature.geometry
    grouped_paths = {}
    for data, polygon in ([(feature.path_data, False)] if feature.path_data else _geometry_paths(geometry)):
        if not polygon and feature.path_data is None and feature.geometry.geom_type in {"Polygon", "MultiPolygon"}:
            # Exact clipping can leave isolated lines where a filled face
            # merely touches a tile. Those intersections draw no filled area.
            continue
        grouped_paths.setdefault(polygon, []).append(data)
    pieces = []
    for part, (polygon, data_parts) in enumerate(grouped_paths.items()):
        data = " ".join(data_parts)
        attributes = dict(feature.attributes)
        clip = attributes.pop("clip", None)
        if clip:
            attributes["clip-path"] = f"url(#{clip_ids[clip]})"
        if polygon:
            # An outline on a clipped polygon would also outline its tile cut.
            # Every cartographic outline is supplied as its own line feature.
            attributes["fill-rule"] = "evenodd"
            attributes["stroke"] = "none"
        if id_map:
            if "id" in attributes:
                attributes["id"] = id_map[attributes["id"]]
            for name, value in attributes.items():
                value = re.sub(r"url\(#([^)]+)\)",
                               lambda match: f'url(#{id_map.get(match[1], match[1])})', str(value))
                if name in {"href", "xlink:href"} and value.startswith("#"):
                    value = "#" + id_map.get(value[1:], value[1:])
                attributes[name] = value
        if part and "id" in attributes:
            attributes["id"] += f"-part-{part}"
        encoded = "".join(
            f' {name}="{html.escape(str(value), quote=True)}"'
            for name, value in attributes.items()
        )
        pieces.append(f'<path d="{html.escape(data, quote=True)}"{encoded} />')
    return "".join(pieces)


def _layer_groups(features: list[tuple[TileFeature, BaseGeometry]], clip_ids: Mapping[str, str],
                  id_map: Mapping[str, str]) -> str:
    # Keep producer paint order, including repeated layers, rather than sorting
    # independently generated colour bands or moving ink behind filled faces.
    result, body, current_layer = [], [], None
    pending=[]; previous=None

    def flush():
        if not pending:return
        first=pending[0][0]
        geometries=[geometry for _,geometry in pending]
        lines=all(geometry.geom_type in {'LineString','MultiLineString'} for geometry in geometries)
        if len(pending)>1 and (lines or shapely.coverage_is_valid(geometries)):
            # One multipart fill retains every ring and exact vertex while
            # eliminating repeated attributes and separate DOM paint nodes.
            body.append(feature_markup(first,GeometryCollection(geometries),clip_ids=clip_ids,id_map=id_map))
        else:
            body.extend(feature_markup(feature,geometry,clip_ids=clip_ids,id_map=id_map)for feature,geometry in pending)
        pending.clear()

    for feature, geometry in features:
        mergeable=(feature.path_data is None and not feature.definitions and 'id' not in feature.attributes
                   and ((feature.section=='theme' and geometry.geom_type in {'Polygon','MultiPolygon'})
                        or (feature.section=='ink' and feature.layer=='transport-network'
                            and feature.attributes.get('data-route-mode')=='road'
                            and geometry.geom_type in {'LineString','MultiLineString'})))
        key=(feature.layer,feature.theme,tuple(feature.attributes.items())) if mergeable else None
        if pending and (key is None or key!=previous):flush()
        if current_layer is not None and current_layer != feature.layer:
            result.append(f'<g data-tile-layer="{html.escape(current_layer, quote=True)}">{"".join(body)}</g>')
            body = []
        current_layer = feature.layer
        if mergeable:pending.append((feature,geometry))
        else:body.append(feature_markup(feature,geometry,clip_ids=clip_ids,id_map=id_map))
        previous=key
    flush()
    if current_layer is not None:
        result.append(f'<g data-tile-layer="{html.escape(current_layer, quote=True)}">{"".join(body)}</g>')
    return "".join(result)


def _clip_geometry(geometry: BaseGeometry, rectangle: BaseGeometry, line_rectangle: BaseGeometry) -> BaseGeometry:
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry.intersection(rectangle)
    if geometry.geom_type == "GeometryCollection":
        return GeometryCollection([_clip_geometry(part, rectangle, line_rectangle) for part in geometry.geoms])
    return geometry.intersection(line_rectangle)


def _feature_error(level: str, index: int, feature: TileFeature, reason: str) -> ValueError:
    geometry = feature.geometry
    bounds = geometry.bounds if isinstance(geometry, BaseGeometry) else None
    validity = is_valid_reason(geometry) if isinstance(geometry, BaseGeometry) else "not Shapely geometry"
    return ValueError(
        f"{reason}; level={level!r} index={index} layer={feature.layer!r} "
        f"theme={feature.theme!r} section={feature.section!r} "
        f"is_valid_reason={validity!r} bounds={bounds!r}"
    )


def _write_tile_level(
    output_dir: str | Path,
    width: int,
    height: int,
    level: TileLevel,
    *,
    tile_size: int = 32,
):
    """Publish exact shared-geometry blocks without modifying source features.

    ``clip='land'`` or ``clip='water'`` in attributes requests a tile-local
    physical clip. Themes are always clipped to the same authoritative land.
    Filled polygons have no stroke; outlines must be supplied as line features.
    """
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("tile dimensions must be positive integers")
    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError("tile_size must be a positive integer")
    land_surface = level.land_surface
    if not isinstance(land_surface, BaseGeometry) or not land_surface.is_valid:
        raise ValueError("land_surface must be valid Shapely geometry")
    source = tuple(level.features)
    for index, feature in enumerate(source):
        if not isinstance(feature, TileFeature):
            raise TypeError(f"features must contain TileFeature values; level={level.id!r} index={index}")
        if feature.section not in {"surface", "theme", "ink"}:
            raise _feature_error(level.id,index,feature,"section must be surface, theme, or ink")
        if (feature.section == "theme") != bool(feature.theme) or feature.theme == "none":
            raise _feature_error(level.id,index,feature,"only theme features may declare a nonempty theme other than none")
        if (not feature.layer or not isinstance(feature.geometry, BaseGeometry)
                or (feature.path_data is None and not feature.geometry.is_valid)
                or (feature.path_data is not None and not all(math.isfinite(value) for value in feature.geometry.bounds))):
            raise _feature_error(level.id,index,feature,"features must declare a layer and valid Shapely geometry")
        if any(not _ATTRIBUTE_NAME.fullmatch(name) or name == "d" for name in feature.attributes):
            raise _feature_error(level.id,index,feature,"feature attributes must be valid SVG names and cannot replace geometry")
        if feature.attributes.get("clip") not in {None, "land", "water"}:
            raise _feature_error(level.id,index,feature,"feature clip must be land or water")
        if feature.path_data is not None and (not isinstance(feature.path_data, str) or not feature.path_data):
            raise _feature_error(level.id,index,feature,"path_data must be a nonempty SVG curve path when supplied")

    columns, rows = math.ceil(width / tile_size), math.ceil(height / tile_size)
    themes = ["none", *dict.fromkeys(feature.theme for feature in source if feature.theme)]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tiles = target / "tiles" / level.id
    tiles.mkdir(parents=True, exist_ok=True)
    tree = STRtree([feature.geometry for feature in source])
    source_ids = tuple(dict.fromkeys([feature.attributes["id"] for feature in source if "id" in feature.attributes]
        + [identifier for feature in source for identifier in (feature.definitions or {})]))
    stats = {"totalTiles":0,"maxTileBytes":0,"totalTileBytes":0}
    for row in range(rows):
        for column in range(columns):
            west, north = column * tile_size, row * tile_size
            east, south = min(width, west + tile_size), min(height, north + tile_size)
            rectangle = box(west, north, east, south)
            line_rectangle = box(west - .5, north - .5, east + .5, south + .5)
            physical_land = land_surface.intersection(rectangle)
            land_data = " ".join(data for data, polygon in _geometry_paths(physical_land) if polygon)
            rectangle_data = _segment(rectangle.exterior.coords, closed=True)
            clip_ids = {"land":f"tile-land-{level.id}-{row}-{column}",
                        "water":f"tile-water-{level.id}-{row}-{column}"}
            id_map = {identifier:f"{identifier}-tile-{level.id}-{row}-{column}" for identifier in source_ids}
            definitions = (
                '<defs>'
                + ''.join(
                    f'<clipPath id="{clip_ids[kind]}" clipPathUnits="userSpaceOnUse">'
                    f'<path d="{data}" clip-rule="evenodd" /></clipPath>'
                    for kind, data in (("land", land_data), ("water", rectangle_data + " " + land_data))
                )
                + '</defs>'
            )
            sections = {"surface":[], "ink":[]}
            thematic = {theme:[] for theme in themes if theme != "none"}
            for index in sorted(tree.query(rectangle).tolist()):
                feature = source[index]
                clipped = feature.geometry if feature.path_data else _clip_geometry(feature.geometry, rectangle, line_rectangle)
                if clipped.is_empty:
                    continue
                destination = thematic[feature.theme] if feature.section == "theme" else sections[feature.section]
                destination.append((feature, clipped))
            surface_markup=_layer_groups(sections["surface"],clip_ids,id_map)
            ink_markup=_layer_groups(sections["ink"],clip_ids,id_map)
            theme_markup={theme:f'<g clip-path="url(#{clip_ids["land"]})">{_layer_groups(items,clip_ids,id_map)}</g>'
                          if items else ""for theme,items in thematic.items()}
            paint_definitions = feature_definitions([feature for items in (*sections.values(),*thematic.values())
                                                    for feature,_ in items],surface_markup+ink_markup+"".join(theme_markup.values()),id_map)
            definitions = definitions.removesuffix('</defs>')+paint_definitions+'</defs>'
            payload = {
                "bounds":[west,north,east-west,south-north],
                "surface":definitions + surface_markup,
                "themes":{"none":"",**theme_markup},
                "ink":ink_markup,
            }
            encoded = json.dumps(payload,ensure_ascii=False,separators=(",", ":")).encode("utf-8")
            if len(encoded) > _MAX_TILE_BYTES:
                raise ValueError(
                    f"tile JSON byte budget exceeded; level={level.id!r} row={row} column={column} "
                    f"bytes={len(encoded)} limit={_MAX_TILE_BYTES}"
                )
            (tiles / f"{row}-{column}.json").write_bytes(encoded)
            stats["totalTiles"] += 1
            stats["maxTileBytes"] = max(stats["maxTileBytes"],len(encoded))
            stats["totalTileBytes"] += len(encoded)
    return stats


def write_atlas_tiles(
    output_dir: str | Path,
    width: int,
    height: int,
    levels: Sequence[TileLevel],
    *,
    tile_size: int = 32,
) -> dict:
    """Publish ordered detail levels sharing native units and the same themes."""
    if not levels or any(not isinstance(level, TileLevel) for level in levels):
        raise ValueError("levels must contain TileLevel values")
    prepared = [TileLevel(level.id, level.min_scale, level.land_surface, tuple(level.features))
                for level in levels]
    identifiers, previous_scale, themes = set(), 0, None
    for level in prepared:
        if not isinstance(level.id, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", level.id) or level.id in identifiers:
            raise ValueError("level ids must be unique lowercase path names")
        if not math.isfinite(level.min_scale) or level.min_scale <= previous_scale:
            raise ValueError("level min_scale values must be positive, finite and strictly increasing")
        identifiers.add(level.id)
        previous_scale = level.min_scale
        if any(not isinstance(feature, TileFeature) for feature in level.features):
            raise TypeError("features must contain TileFeature values")
        level_themes = ["none", *dict.fromkeys(feature.theme for feature in level.features if feature.theme)]
        if themes is not None and themes != level_themes:
            raise ValueError("every level must publish the same themes in the same order")
        themes = level_themes
    stats = {"totalTiles":0,"maxTileBytes":0,"totalTileBytes":0}
    for level in prepared:
        measured = _write_tile_level(output_dir,width,height,level,tile_size=tile_size)
        stats["totalTiles"] += measured["totalTiles"]
        stats["maxTileBytes"] = max(stats["maxTileBytes"],measured["maxTileBytes"])
        stats["totalTileBytes"] += measured["totalTileBytes"]
    manifest = {"width":width, "height":height, "tileSize":tile_size,
                "columns":math.ceil(width/tile_size), "rows":math.ceil(height/tile_size),
                "themes":themes,
                "levels":[{"id":level.id,"minScale":level.min_scale} for level in prepared],
                "stats":stats}
    (Path(output_dir) / "atlas-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    return manifest


def write_city_relief_tiles(output_dir: str | Path, relief) -> dict:
    """Publish true minor contours through the detail base's unique land clip."""
    from .city_relief import CityRelief
    if not isinstance(relief,CityRelief):
        raise TypeError("city relief must be the traced same-ground spatial records")
    target = Path(output_dir)
    manifest_path = target/"atlas-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if [level["id"] for level in manifest["levels"]] != ["regional","local","detail"]:
        raise ValueError("city refinement requires the three published physical base levels")
    source = relief.features
    if any(feature.section != "ink" or feature.theme
           or feature.layer != "elevation-contours"
           or feature.attributes.get("data-contour-kind") != "minor"
           or feature.attributes.get("data-source-field") != "accepted-continuous-ground"
           or not feature.geometry.is_valid or feature.path_data
           or feature.attributes.get("clip")!="land" for feature in source):
        raise ValueError("city refinement may contain only valid land-clipped physical relief")
    width,height,size = manifest["width"],manifest["height"],manifest["tileSize"]
    tree = STRtree([feature.geometry for feature in source])
    tiles = target/"tiles/city-detail"
    tiles.mkdir(parents=True,exist_ok=True)
    stats = {"totalTiles":0,"maxTileBytes":0,"totalTileBytes":0}
    for row,column in relief.published_tiles:
        if not 0<=row<manifest["rows"] or not 0<=column<manifest["columns"]:
            raise ValueError("published city refinement must remain in the world frame")
        west,north = column*size,row*size
        east,south = min(width,west+size),min(height,north+size)
        rectangle = box(west,north,east,south)
        line_rectangle = box(west-.5,north-.5,east+.5,south+.5)
        sections = {"surface":[],"ink":[]}
        for index in sorted(tree.query(rectangle).tolist()):
            feature = source[index]
            clipped = _clip_geometry(feature.geometry,rectangle,line_rectangle)
            if not clipped.is_empty:
                sections[feature.section].append((feature,clipped))
        suffix = f"city-detail-{row}-{column}"
        mask_id = f"city-relief-mask-{suffix}"
        gradients,ellipses = [],[]
        for index,support in enumerate(relief.supports):
            x,y,rx,ry = (support[name] for name in ("x","y","rx","ry"))
            if x+rx<=west or x-rx>=east or y+ry<=north or y-ry>=south:
                continue
            gradient_id = f'city-gradient-{support["id"]}-{index}-{suffix}'
            gradients.append(f'<radialGradient id="{gradient_id}" gradientUnits="userSpaceOnUse" '
                f'gradientTransform="matrix({rx:.8f} 0 0 {ry:.8f} {x:.8f} {y:.8f})" cx="0" cy="0" r="1">'
                '<stop offset="0" stop-color="white"/><stop offset="0.65" stop-color="white"/>'
                '<stop offset="1" stop-color="white" stop-opacity="0"/></radialGradient>')
            ellipses.append(f'<ellipse cx="{x:.8f}" cy="{y:.8f}" rx="{rx:.8f}" ry="{ry:.8f}" '
                            f'fill="url(#{gradient_id})"/>')
        definitions = ('<defs>'+''.join(gradients)+f'<mask id="{mask_id}" maskUnits="userSpaceOnUse" '
            f'maskContentUnits="userSpaceOnUse" x="{west}" y="{north}" width="{east-west}" height="{south-north}" '
            'style="mask-type:alpha">'+''.join(ellipses)+'</mask></defs>')
        clips = {"land":f"tile-land-detail-{row}-{column}","water":f"tile-water-detail-{row}-{column}"}
        wrapper = f'<g data-city-refinement="true" mask="url(#{mask_id})" clip-path="url(#{clips["land"]})">'
        payload = {"bounds":[west,north,east-west,south-north],
                   "surface":definitions,
                   "ink":wrapper+_layer_groups(sections["ink"],clips,{})+'</g>'}
        encoded = json.dumps(payload,ensure_ascii=False,separators=(",",":")).encode("utf-8")
        if len(encoded)>_MAX_TILE_BYTES:
            raise ValueError(f"city relief JSON byte budget exceeded: {suffix}: {len(encoded)} > {_MAX_TILE_BYTES}")
        (tiles/f"{row}-{column}.json").write_bytes(encoded)
        stats["totalTiles"]+=1
        stats["totalTileBytes"]+=len(encoded)
        stats["maxTileBytes"]=max(stats["maxTileBytes"],len(encoded))
    manifest["levels"].append({"id":"city-detail","minScale":128,"kind":"relief-overlay",
        "publishedTiles":[f"{row}-{column}" for row,column in relief.published_tiles]})
    manifest["stats"]["totalTiles"]+=stats["totalTiles"]
    manifest["stats"]["totalTileBytes"]+=stats["totalTileBytes"]
    manifest["stats"]["maxTileBytes"]=max(manifest["stats"]["maxTileBytes"],stats["maxTileBytes"])
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    (target/"city-relief-generation.json").write_text(json.dumps({**relief.diagnostics,"overlayBytes":stats},
        ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return manifest


def refresh_atlas_tile_themes(source_dir: Path, target_dir: Path, features: Sequence[TileFeature],
                             *, ink_layers: Mapping[str, Sequence[TileFeature]] | None = None) -> dict:
    """Rebuild selected derived paints with the current tile writer's rules.

    Surfaces, clips and unselected paint remain byte-identical payload strings.
    The caller owns staging the complete review; this operation never rewrites
    its source directory or changes a canonical world array.
    """
    if source_dir.resolve() == target_dir.resolve():
        raise ValueError("derived tile refresh requires a separate staging directory")
    manifest = json.loads((source_dir / "atlas-manifest.json").read_text(encoding="utf-8"))
    if (not features and not ink_layers) or any(feature.section != "theme" or not feature.theme or feature.path_data
                           or not feature.geometry.is_valid
                           or feature.attributes.get("clip") != "land" for feature in features):
        raise ValueError("tile theme refresh requires valid filled land-clipped features")
    themes = tuple(dict.fromkeys(feature.theme for feature in features))
    if not set(themes) <= set(manifest["themes"]):
        raise ValueError("refreshed themes must belong to the current manifest")
    ink_layers = dict(ink_layers or {})
    if any(feature.layer != layer or feature.section != "ink" or feature.theme
           or not feature.geometry.is_valid
           for layer, records in ink_layers.items() for feature in records):
        raise ValueError("refreshed ink must declare its complete owned vector layers")
    from .svg_groups import replace_group
    ink_features = tuple(feature for records in ink_layers.values() for feature in records)
    ink_tree = STRtree([feature.geometry for feature in ink_features])
    tree = STRtree([feature.geometry for feature in features])
    stats = {"totalTiles":0,"maxTileBytes":0,"totalTileBytes":0}
    for level in manifest["levels"]:
        target = target_dir / "tiles" / level["id"]
        target.mkdir(parents=True, exist_ok=True)
        if level.get("kind") == "relief-overlay":
            for position in level["publishedTiles"]:
                encoded = (source_dir / "tiles" / level["id"] / f"{position}.json").read_bytes()
                (target / f"{position}.json").write_bytes(encoded)
                stats["totalTiles"] += 1
                stats["maxTileBytes"] = max(stats["maxTileBytes"],len(encoded))
                stats["totalTileBytes"] += len(encoded)
            continue
        for row in range(manifest["rows"]):
            for column in range(manifest["columns"]):
                path = source_dir / "tiles" / level["id"] / f"{row}-{column}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                west, north, width, height = payload["bounds"]
                rectangle = box(west,north,west+width,north+height)
                line_rectangle = box(west-.5,north-.5,west+width+.5,north+height+.5)
                clip_ids = {"land":f"tile-land-{level['id']}-{row}-{column}",
                            "water":f"tile-water-{level['id']}-{row}-{column}"}
                grouped = {theme:[] for theme in themes}
                for index in sorted(tree.query(rectangle).tolist()):
                    feature = features[index]
                    geometry = _clip_geometry(feature.geometry,rectangle,rectangle)
                    if not geometry.is_empty:
                        grouped[feature.theme].append((feature,geometry))
                for theme, items in grouped.items():
                    payload["themes"][theme] = (
                        f'<g clip-path="url(#{clip_ids["land"]})">{_layer_groups(items,clip_ids,{})}</g>'
                        if items else ""
                    )
                ink_grouped = {layer:[] for layer in ink_layers}
                for index in sorted(ink_tree.query(rectangle).tolist()):
                    feature = ink_features[index]
                    geometry = feature.geometry if feature.path_data else _clip_geometry(feature.geometry,rectangle,line_rectangle)
                    if not geometry.is_empty:
                        ink_grouped[feature.layer].append((feature,geometry))
                for layer, items in ink_grouped.items():
                    payload["ink"] = replace_group(payload["ink"],"data-tile-layer",layer,
                        _layer_groups(items,clip_ids,{}),
                        before="transport-network" if layer in {"state-boundaries","province-boundaries"} else None)
                encoded = json.dumps(payload,ensure_ascii=False,separators=(",", ":")).encode("utf-8")
                if len(encoded) > _MAX_TILE_BYTES:
                    raise ValueError(f"tile JSON byte budget exceeded; level={level['id']!r} row={row} "
                                     f"column={column} bytes={len(encoded)} limit={_MAX_TILE_BYTES}")
                (target / path.name).write_bytes(encoded)
                stats["totalTiles"] += 1
                stats["maxTileBytes"] = max(stats["maxTileBytes"],len(encoded))
                stats["totalTileBytes"] += len(encoded)
        logging.info("Refreshed %s numeric tile paints; %d tiles complete",level["id"],stats["totalTiles"])
    manifest["stats"] = stats
    (target_dir / "atlas-manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,separators=(",", ":")),encoding="utf-8")
    return manifest
