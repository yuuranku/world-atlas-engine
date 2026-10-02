"""Check delivered detail colours and contours against the saved raw ground."""
from __future__ import annotations

import argparse
from decimal import Decimal
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
import shapely

from world_atlas.core.terrain_refinement import terrain_from_source
from world_atlas.core.hypsometry import elevation_display_indices, elevation_display_thresholds
from world_atlas.core.model import WorldGrid
from world_atlas.core.procedural_planet import load_surface_bundle
from world_atlas.core.render import _elevation_palette_for
from world_atlas.core.implicit_terrain_bounds import field_bounds
from world_atlas.core.svg_artifacts import read_svgz


_spec = importlib.util.spec_from_file_location("physical_tile_consumer", Path(__file__).with_name("check_atlas_tiles.py"))
_tiles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tiles)


def _svgz_document(path):
    """Read the canonical compressed export without changing its SVG source."""
    if path.suffix!=".svgz":
        raise ValueError("Complete physical exports require their canonical SVGZ artifact")
    if path.stat().st_size>64*1024*1024:
        raise ValueError("Canonical physical export exceeds its 64 MiB encoded file budget")
    markup=read_svgz(path)
    decoded=markup.encode("utf-8")
    artifact={"encoding":"gzip","bytes":path.stat().st_size,"digest":_file_digest(path),
              "decodedBytes":len(decoded),"decodedDigest":sha256(decoded).hexdigest()}
    return _tiles.ET.fromstring(markup),artifact


def _line_points(data):
    """Decode delivered linear SVG subpaths using exact decimal accumulation."""
    token = _tiles._svg.TOKEN
    if _tiles._svg.re.sub(r"[\s,]+", "", token.sub("", data)):
        raise ValueError("physical contours must use linear SVG path commands")
    tokens = token.findall(data)
    lines, line, index, command = [], [], 0, None
    current, start = (Decimal(0), Decimal(0)), None
    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
            if command.upper() == "Z":
                if start is not None:
                    line.append(start)
                if line:
                    lines.append(np.asarray(line, dtype=float))
                line, start, command = [], None, None
                continue
        if command in ("M", "m", "L", "l"):
            if command.upper() == "M" and line:
                lines.append(np.asarray(line, dtype=float))
                line = []
            value = (Decimal(tokens[index]), Decimal(tokens[index + 1]))
            if command.islower():
                value = (current[0] + value[0], current[1] + value[1])
            current = value
            if command.upper() == "M":
                start = current
            line.append(current)
            index += 2
            command = "l" if command.islower() else "L"
        elif command in ("H", "h", "V", "v") and line:
            value = Decimal(tokens[index])
            index += 1
            axis = 0 if command.upper() == "H" else 1
            if command.islower():
                value += current[axis]
            current = (value, current[1]) if axis == 0 else (current[0], value)
            line.append(current)
        else:
            raise ValueError("unsupported physical contour SVG command")
    if line:
        lines.append(np.asarray(line, dtype=float))
    return lines


def _major_tile_paths(markup,tile):
    def visit(node,active=False,land=False):
        if _tiles.tag(node)in {"defs","mask","clipPath","pattern"}:return
        attributes=_tiles._attributes(node)
        reference=attributes.get("clip-path")
        if reference is not None and reference!=f'url(#{tile["landClipId"]})':
            if active or attributes.get("data-tile-layer")=="elevation-contours":
                raise ValueError("Major contours must use this exact shared detail land clip")
        land=land or reference==f'url(#{tile["landClipId"]})'
        active=active or attributes.get("data-tile-layer")=="elevation-contours"
        if active and _tiles.tag(node)=="path":
            if not land or attributes.get("fill")!="none" or "data-sampling-cells"in attributes:
                raise ValueError("Major contour ink must inherit actual land clipping, without a sampled-mesh contract")
            yield node
        for child in node:yield from visit(child,active,land)
    return list(visit(_tiles.tile_document(markup)))


def _colour_regions(markup, tile):
    """Retain the delivered ancestor clips while selecting elevation paint."""
    document = _tiles.tile_document(markup)

    def select(node, *, active=False, definitions=False):
        active = active or node.get("data-tile-layer") == "elevation-bands"
        definitions = definitions or _tiles.tag(node) == "defs"
        for child in list(node):
            if _tiles.tag(child) == "path" and not definitions:
                if not active:
                    node.remove(child)
                    continue
                fill = child.get("fill", "black")
                if len(fill) != 7 or not fill.startswith("#"):
                    raise ValueError("physical elevation paint requires an explicit RGB colour")
                child.set("data-check-colour",str(int(fill[1:],16)))
            select(child,active=active,definitions=definitions)
    select(document)
    return _tiles.painted_regions(_tiles.ET.tostring(document,encoding="unicode"),tile,
                                  owner_attribute="data-check-colour")


def _contour_check(points, heights_m, terrain, *, coordinate_error):
    """Check source roots on the actual model, allowing only text quantization."""
    if not len(points):
        return {"checkedPoints":0, "mismatchCount":0, "maxResidualMeters":0.0,
                "maxSerializationEnclosureMeters":0.0}
    points = np.asarray(points,dtype=float)
    physical_level = np.asarray(heights_m,dtype=float)
    if (points.shape != (len(physical_level),2) or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(physical_level)) or coordinate_error not in (0.,5e-9)
            or np.any(points<0) or np.any(points>np.array((terrain.width,terrain.height)))):
        raise ValueError("physical source curves require finite in-frame roots and their text precision")
    actual = terrain.sample_points(points[:,0],points[:,1])
    low,high = actual.copy(),actual.copy()
    if coordinate_error:
        low,high = field_bounds(terrain,np.maximum(0,points-coordinate_error),
                               np.minimum((terrain.width,terrain.height),points+coordinate_error))
    # This is the physical root gate used for source curves, in metres. It
    # grants no chord, city sampling or geometric displacement allowance.
    mismatch = (physical_level < low-2e-8)|(physical_level > high+2e-8)
    return {"checkedPoints":len(points), "mismatchCount":int(np.count_nonzero(mismatch)),
            "maxResidualMeters":float(np.max(np.abs(actual - physical_level))),
            "maxSerializationEnclosureMeters":float(np.max(high-low))}


def _source_curves(records,terrain,*,coordinate_error):
    """Validate authoritative vertices once, then index their unchanged chords."""
    geometries,heights,proof = [],[],_contour_check(np.empty((0,2)),np.empty(0),terrain,
                                                   coordinate_error=coordinate_error)
    for height,geometry in records:
        height = float(height)
        if (not np.isfinite(height) or height<=0 or geometry.is_empty or not geometry.is_valid
                or geometry.geom_type not in {"LineString","MultiLineString"}):
            raise ValueError("physical contour source must contain valid positive metre curves")
        points = shapely.get_coordinates(geometry)
        for begin in range(0,len(points),32768):
            part=points[begin:begin+32768]
            checked=_contour_check(part,np.full(len(part),height),terrain,coordinate_error=coordinate_error)
            for name in ("checkedPoints","mismatchCount"):
                proof[name]+=checked[name]
            for name in ("maxResidualMeters","maxSerializationEnclosureMeters"):
                proof[name]=max(proof[name],checked[name])
        geometries.append(geometry);heights.append(height)
    return {"geometries":geometries,"heights":np.asarray(heights),
            "index":shapely.STRtree(geometries)},proof


def _replay_contours(paths,tile,source):
    """Tile cuts inherit source-root proof; their new endpoints are chord cuts."""
    x,y,width,height=tile["payload"]["bounds"]
    rectangle=shapely.box(x,y,x+width,y+height)
    expanded=shapely.box(x-.5,y-.5,x+width+.5,y+height+.5)
    expected={}
    for index in source["index"].query(rectangle):
        geometry=source["geometries"][index].intersection(expanded)
        if not geometry.is_empty:
            expected.setdefault(float(source["heights"][index]),[]).append(geometry)
    result={"checkedPaths":sum(len(v)for v in paths.values()),"mismatchCount":0,
            "maxDeliveryDistanceCells":0.,"unexpectedHeightCount":0}
    for height in set(expected)|set(paths):
        original=shapely.union_all(expected.get(height,[]))
        delivered=shapely.union_all(paths.get(height,[]))
        if original.is_empty and not delivered.is_empty:
            result["unexpectedHeightCount"]+=1
        if original.is_empty or delivered.is_empty:
            mismatch=original.is_empty!=delivered.is_empty
            distance=0. if not mismatch else float("inf")
        else:
            distance=float(shapely.hausdorff_distance(original,delivered,densify=.5))
            mismatch=distance>1e-7
        result["mismatchCount"]+=int(mismatch)
        result["maxDeliveryDistanceCells"]=max(result["maxDeliveryDistanceCells"],distance)
    return result


def check_tile(tile, *, terrain, palette, water, source):
    x, y, width, height = tile["payload"]["bounds"]
    rows, columns = np.meshgrid(np.arange(y, y + height), np.arange(x, x + width), indexing="ij")
    xx, yy = columns + .5, rows + .5
    dry = water[rows, columns] == 0
    colors = np.asarray(palette, dtype=np.int32)
    packed_palette = (colors[:, 0] << 16) + (colors[:, 1] << 8) + colors[:, 2]
    expected = packed_palette[elevation_display_indices(terrain.palette_elevation(terrain.native_m[rows, columns]))]
    painted = np.full(xx.shape, -1, dtype=np.int32)
    fill_regions = []
    for colour,geometry in _colour_regions(tile["payload"]["surface"],tile):
        shapely.prepare(geometry)
        selected = shapely.intersects_xy(geometry, xx, yy)
        painted[selected] = colour
        fill_regions.append(geometry)
    missing_area = float(tile["land"].difference(shapely.union_all(fill_regions)).area)
    wrong = dry & (painted != expected)
    wet_paint = ~dry & (painted != -1)
    examples = [{"column":int(columns.ravel()[i]), "row":int(rows.ravel()[i]),
                 "expected":f"#{int(expected.ravel()[i]):06x}",
                 "actual":None if painted.ravel()[i] < 0 else f"#{int(painted.ravel()[i]):06x}"}
                for i in np.flatnonzero(wrong.ravel())[:5]]
    paths, zero_lines = {}, 0
    allowed_levels = set(elevation_display_thresholds())
    for node in _major_tile_paths(tile["payload"]["ink"],tile):
        level = float(node.get("data-level", "nan"))
        if level not in allowed_levels:
            raise ValueError("physical contour level is not a declared land boundary")
        if float(terrain.contour_height_m(level)) <= 0:
            zero_lines += 1
            continue
        for line in _line_points(node.get("d", "")):
            if len(line)>=2:
                paths.setdefault(float(terrain.contour_height_m(level)),[]).append(shapely.LineString(line))
    contour = _replay_contours(paths,tile,source)
    return {"tile":tile["key"], "checkedDryCenters":int(dry.sum()),
            "checkedWetCenters":int((~dry).sum()), "colorMismatchCount":int(wrong.sum()),
            "waterPaintCount":int(wet_paint.sum()), "missingLandArea":missing_area,
            "zeroContourPathCount":zero_lines, "contours":contour, "examples":examples}


def check_city_tile(tile, *, terrain, palette, water, source):
    """Verify additive minor ink, its inherited support, and exact source cuts."""
    x,y,width,height = tile["payload"]["bounds"]
    rows,columns = np.meshgrid(np.arange(y,y+height),np.arange(x,x+width),indexing="ij")
    xx,yy = columns+.5,rows+.5
    support = np.zeros(xx.shape,dtype=bool)
    masks = [node for node in _tiles.tile_document(tile["payload"]["surface"]).iter()
             if _tiles.tag(node) == "mask" and _tiles._attributes(node).get("mask-type") == "alpha"]
    if len(masks) != 1:
        raise ValueError("City physical relief needs its single delivered alpha support mask")
    for ellipse in masks[0]:
        if _tiles.tag(ellipse) != "ellipse":
            raise ValueError("City physical relief support must use explicit ellipses")
        cx,cy,rx,ry = (float(ellipse.get(name,"nan")) for name in ("cx","cy","rx","ry"))
        if not np.all(np.isfinite((cx,cy,rx,ry))) or rx<=0 or ry<=0:
            raise ValueError("City alpha support must contain finite positive ellipses")
        support |= ((xx-cx)/rx)**2+((yy-cy)/ry)**2 < 1
    dry = water[rows,columns] == 0
    paths,count={},0
    mask_id=masks[0].get("id")
    if mask_id!=tile["citySupportMaskId"]:
        raise ValueError("City relief references a foreign alpha support")
    for section in ("surface","ink"):
        def visit(node,land=False,masked=False,refinement=False,layer=None):
            nonlocal count
            tag=_tiles.tag(node)
            if tag in {"defs","mask","clipPath","pattern"}:return
            attributes=_tiles._attributes(node)
            if "clip-path"in attributes and attributes["clip-path"]!=f'url(#{tile["landClipId"]})':
                raise ValueError("City ink must inherit the exact shared detail land clip")
            if "mask"in attributes and attributes["mask"]!=f'url(#{mask_id})':
                raise ValueError("City ink must inherit its single alpha support")
            land=land or attributes.get("clip-path")==f'url(#{tile["landClipId"]})'
            masked=masked or attributes.get("mask")==f'url(#{mask_id})'
            refinement=refinement or attributes.get("data-city-refinement")=="true"
            layer=attributes.get("data-tile-layer",layer)
            if tag=="path":
                height_m=float(attributes.get("data-height-m","nan"))
                if (section!="ink" or not land or not masked or not refinement
                        or layer!="elevation-contours" or attributes.get("fill")!="none"
                        or attributes.get("data-city-relief")!="true"
                        or attributes.get("data-contour-kind")!="minor"
                        or attributes.get("data-source-field")!="accepted-continuous-ground"
                        or "data-sampling-cells"in attributes or "data-level"in attributes
                        or not np.isfinite(height_m) or height_m<=0 or height_m%100!=0):
                    raise ValueError("City relief may add only supported true 100-metre minor ink; no colour or major override")
                count+=1
                for line in _line_points(attributes.get("d","")):
                    if len(line)>=2:paths.setdefault(height_m,[]).append(shapely.LineString(line))
            elif tag not in {"svg","g"}:
                raise ValueError("City relief contains unsupported visible geometry")
            for child in node:visit(child,land,masked,refinement,layer)
        visit(_tiles.tile_document(tile["payload"][section]))
    if not count:
        raise ValueError("Flat city relief must not publish an empty overlay")
    contour=_replay_contours(paths,tile,source)
    return {"tile":tile["key"],"checkedDryCenters":int((dry&support).sum()),
            "checkedWetCenters":int((~dry&support).sum()),"fineMinorContourPathCount":count,
            "contours":contour}


def _read_city_sources(path,terrain,grid_digest):
    document=json.loads(path.read_text(encoding="utf-8"))
    if (set(document)!={"schema","coordinateSpace","gridDigest","curves"}
            or document["schema"]!="accepted-physical-minor-curves-v1"
            or document["coordinateSpace"]!="native-cell" or document["gridDigest"]!=grid_digest
            or not isinstance(document["curves"],list)):
        raise ValueError("City contour proof must identify this canonical grid and true native minor curves")
    records=[]
    major=np.asarray(terrain.contour_height_m(elevation_display_thresholds()))
    for item in document["curves"]:
        if set(item)!={"heightM","geometry"}:
            raise ValueError("City contour proof contains unsupported fields")
        height=float(item["heightM"])
        if height%100!=0 or np.any(abs(major-height)<20):
            raise ValueError("City source may contain only 100-metre minor curves separated from major contours")
        records.append((height,shapely.geometry.shape(item["geometry"])))
    return _source_curves(records,terrain,coordinate_error=0.)


def _read_major_sources(path,terrain,*,physical_land):
    allowed=set(elevation_display_thresholds())
    records=[]
    document,artifact=_svgz_document(path)
    if (document.tag!="{http://www.w3.org/2000/svg}svg"
            or [float(v)for v in document.get("viewBox","").split()]
            !=[0.,0.,float(terrain.width),float(terrain.height)]):
        raise ValueError("Major source export must use the actual native world frame")
    parents={child:node for node in document.iter()for child in node}
    clips=[node for node in document.iter()if _tiles.tag(node)=="clipPath"]
    if (len(clips)!=1 or clips[0].get("id")!="land-silhouette-clip"
            or clips[0].get("clipPathUnits")!="userSpaceOnUse"
            or parents[clips[0]].tag!="{http://www.w3.org/2000/svg}defs"
            or parents.get(parents[clips[0]])is not document
            or clips[0].get("transform")or clips[0].get("style")
            or parents[clips[0]].get("transform")or parents[clips[0]].get("style")
            or sum(node.get("id")=="land-silhouette-clip"for node in document.iter())!=1):
        raise ValueError("Major source export must own its sole authoritative native land clip")
    clip_paths=list(clips[0])
    if not clip_paths or any(node.tag!="{http://www.w3.org/2000/svg}path"
            or node.get("clip-rule")!="evenodd"or not node.get("d")
            or node.get("transform")or node.get("clip-path")or node.get("style")for node in clip_paths):
        raise ValueError("Major land authority must retain its actual evenodd native paths")
    land=_tiles._svg.union_paths([node.attrib for node in clip_paths])
    if not land.equals(physical_land):
        raise ValueError("Major source export differs from the published physical land authority")
    def visit(node,clipped=False):
        if _tiles.tag(node)in {"defs","mask","clipPath","pattern"}:return
        attributes=_tiles._attributes(node)
        if attributes.get("transform")or attributes.get("mask"):
            raise ValueError("Major source export must retain actual native coordinates and land clipping")
        reference=attributes.get("clip-path")
        if reference is not None and reference!="url(#land-silhouette-clip)":
            raise ValueError("Major source export references a foreign or secondary land clip")
        clipped=clipped or reference=="url(#land-silhouette-clip)"
        if _tiles.tag(node)=="path":
            level=float(node.get("data-level","nan"))
            if not clipped or node.get("fill")!="none" or level not in allowed:
                raise ValueError("Major source export must declare the exact physical palette threshold")
            for points in _line_points(node.get("d","")):
                if len(points)>=2:
                    records.append((float(terrain.contour_height_m(level)),shapely.LineString(points)))
        for child in node:visit(child,clipped)
    visit(document)
    source,proof=_source_curves(records,terrain,coordinate_error=5e-9)
    proof["artifact"]=artifact
    return source,proof


def check_staged_major_sources(directory, source, *, terrain, source_identity):
    """Replay every committed raw source path against actual SVG integer nodes.

    The stage's binding recipe is shared metadata, not a geometry reader. This
    consumer independently checks the manifest, headers, file digests and NPZ
    schema. It never invokes extraction or substitutes a reconstructed curve.
    """
    from world_atlas.core.physical_contour_stage import _binding

    directory=Path(directory)
    heights=np.asarray(terrain.contour_height_m(elevation_display_thresholds()),dtype=float)
    heights=heights[heights>0]
    binding=_binding(terrain,heights,source_identity)
    fingerprint=sha256(json.dumps(binding,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    manifest_path=directory/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if (set(manifest)!={"schema","binding","fingerprint","status","completedHeights","totalHeights"}
            or manifest["schema"]!="physical-height-graphs-v1"or manifest["binding"]!=binding
            or manifest["fingerprint"]!=fingerprint or manifest["status"]!="complete"
            or manifest["completedHeights"]!=len(heights)or manifest["totalHeights"]!=len(heights)):
        raise ValueError("Source-stage proof requires complete graphs for this exact current field and height recipe")
    wanted_files={"manifest.json"}
    for index in range(len(heights)):
        wanted_files.update((f"level-{index:03d}.json",f"level-{index:03d}.npz"))
    if {path.name for path in directory.iterdir()if path.suffix in {".json",".npz"}}!=wanted_files:
        raise ValueError("Source-stage proof contains missing or foreign height graphs")
    raw_proof=_contour_check(np.empty((0,2)),np.empty(0),terrain,coordinate_error=0.)
    report={"schema":"accepted-physical-stage-svg-integer-replay-v1","status":"ok",
            "fingerprint":fingerprint,"heightLevelsHex":[float(value).hex()for value in heights],
            "checkedHeights":len(heights),"checkedSourcePaths":0,"checkedSourceVertices":0,
            "checkedDeliveredPaths":len(source["geometries"]),"encodedAdjacentDuplicates":0,
            "mismatchCount":0,"pathCountMismatchCount":0,"integerCoordinateMismatchCount":0,
            "sourceVertices":raw_proof,"files":{},"examples":[]}
    for index,height in enumerate(heights):
        header_path=directory/f"level-{index:03d}.json"
        path=directory/f"level-{index:03d}.npz"
        header=json.loads(header_path.read_text(encoding="utf-8"))
        digest=_file_digest(path)
        if (set(header)!={"schema","fingerprint","index","heightMHex","sha256","paths","vertices"}
                or header["schema"]!="physical-height-graphs-v1"or header["fingerprint"]!=fingerprint
                or header["index"]!=index or header["heightMHex"]!=float(height).hex()
                or header["sha256"]!=digest):
            raise ValueError("Source height header or bytes differ from the committed physical graph")
        with np.load(path,allow_pickle=False)as bundle:
            if set(bundle.files)!={"level","points","offsets"}:
                raise ValueError("Source-stage proof requires the current numeric array schema")
            saved,points,offsets=bundle["level"],bundle["points"],bundle["offsets"]
        if (saved.dtype!=np.dtype("float64")or saved.shape!=()or float(saved).hex()!=float(height).hex()
                or points.dtype!=np.dtype("float64")or points.ndim!=2 or points.shape[1]!=2
                or not np.isfinite(points).all()or np.any(points<0)
                or np.any(points>np.array((terrain.width,terrain.height)))
                or offsets.dtype!=np.dtype("int64")or offsets.ndim!=1 or not len(offsets)
                or offsets[0]!=0 or offsets[-1]!=len(points)or np.any(np.diff(offsets)<2)
                or len(offsets)-1!=header["paths"]or len(points)!=header["vertices"]):
            raise ValueError("Source-stage proof has invalid native coordinates or path boundaries")
        for begin in range(0,len(points),32768):
            part=points[begin:begin+32768]
            proof=_contour_check(part,np.full(len(part),height),terrain,coordinate_error=0.)
            for name in ("checkedPoints","mismatchCount"):raw_proof[name]+=proof[name]
            for name in ("maxResidualMeters","maxSerializationEnclosureMeters"):
                raw_proof[name]=max(raw_proof[name],proof[name])
        report["checkedSourcePaths"]+=len(offsets)-1
        report["checkedSourceVertices"]+=len(points)
        delivered=[source["geometries"][i]for i in np.flatnonzero(source["heights"]==height)]
        if len(delivered)!=len(offsets)-1:
            report["pathCountMismatchCount"]+=1
            report["examples"].append({"heightM":float(height),"sourcePaths":len(offsets)-1,
                                        "deliveredPaths":len(delivered)})
        for part_index,(begin,end)in enumerate(zip(offsets[:-1],offsets[1:],strict=True)):
            # These exact integer coordinates are the project's delivery
            # contract. Remove only adjacent nodes that encode identically;
            # every other source node must survive, including a straight run.
            encoded=np.rint(points[begin:end]*100_000_000).astype(np.int64)
            distinct=np.r_[True,np.any(encoded[1:]!=encoded[:-1],axis=1)]
            report["encodedAdjacentDuplicates"]+=int((~distinct).sum())
            encoded=encoded[distinct]
            if part_index>=len(delivered):continue
            geometry=delivered[part_index]
            if geometry.geom_type!="LineString":
                raise ValueError("Major SVG must retain each saved source path as its own linear subpath")
            actual=np.rint(shapely.get_coordinates(geometry)*100_000_000).astype(np.int64)
            if not np.array_equal(encoded,actual):
                report["integerCoordinateMismatchCount"]+=1
                if len(report["examples"])<10:
                    report["examples"].append({"heightM":float(height),"pathIndex":part_index,
                        "sourceEncodedNodes":len(encoded),"deliveredNodes":len(actual)})
        report["files"][path.name]={"bytes":path.stat().st_size,"digest":digest,
                                    "headerDigest":_file_digest(header_path)}
    report["mismatchCount"]=report["pathCountMismatchCount"]+report["integerCoordinateMismatchCount"]
    report["files"][manifest_path.name]={"bytes":manifest_path.stat().st_size,
                                         "digest":_file_digest(manifest_path)}
    if report["mismatchCount"]or raw_proof["mismatchCount"]:report["status"]="failed"
    return report


def check_physical_export(path,*,terrain,palette,water):
    """Decode owned land clipping and opaque elevation paint in SVG order."""
    document,artifact=_svgz_document(path)
    height,width=water.shape
    if (document.tag!="{http://www.w3.org/2000/svg}svg"
            or [float(v)for v in document.get("viewBox","").split()]!=[0.,0.,float(width),float(height)]
            or terrain.native_m.shape!=water.shape):
        raise ValueError("Physical export must declare this native world frame")
    parents={child:parent for parent in document.iter()for child in parent}
    clips=[node for node in document.iter()if _tiles.tag(node)=="clipPath"
           and node.get("id")=="land-silhouette-clip"]
    if (len(clips)!=1 or clips[0].get("clipPathUnits")!="userSpaceOnUse"
            or parents[clips[0]].tag!="{http://www.w3.org/2000/svg}defs"
            or parents.get(parents[clips[0]])is not document
            or clips[0].get("transform") or clips[0].get("style")
            or parents[clips[0]].get("transform") or parents[clips[0]].get("style")
            or sum(node.get("id")=="land-silhouette-clip"for node in document.iter())!=1):
        raise ValueError("Published physical surface must own one native land silhouette")
    clip_paths=list(clips[0])
    if not clip_paths or any(node.tag!="{http://www.w3.org/2000/svg}path"
            or node.get("clip-rule")!="evenodd" or not node.get("d")
            or node.get("transform") or node.get("clip-path") or node.get("style")for node in clip_paths):
        raise ValueError("Physical land authority must use its actual evenodd native paths")
    land=_tiles._svg.union_paths([node.attrib for node in clip_paths])
    if land.is_empty or not land.is_valid:
        raise ValueError("Physical land authority must be valid and nonempty")
    partitions=[node for node in document.iter()if _tiles.tag(node)=="g"and node.get("id")=="elevation-bands"]
    if len(partitions)!=1:
        raise ValueError("Physical export must publish exactly one elevation-bands paint layer")
    ancestors=[];ancestor=parents.get(partitions[0])
    while ancestor is not None:
        if _tiles.tag(ancestor)in {"defs","clipPath","mask","pattern"}:
            raise ValueError("Physical elevation paint cannot be a definition")
        ancestors.append(ancestor);ancestor=parents.get(ancestor)
    def inherit(node,fill,clipped):
        values=_tiles._attributes(node)
        reference=values.get("clip-path")
        if reference is not None:
            if reference!="url(#land-silhouette-clip)"or clipped:
                raise ValueError("Physical elevation paint must inherit its single owned land clip")
            clipped=True
        if (values.get("transform") or values.get("mask") or "hidden"in values
                or values.get("display")=="none"or values.get("visibility")=="hidden"
                or float(values.get("opacity","1"))!=1 or float(values.get("fill-opacity","1"))!=1):
            raise ValueError("Physical elevation paint must be visible opaque native colour")
        return values.get("fill",fill),clipped,values
    fill,clipped="black",False
    for ancestor in reversed(ancestors):fill,clipped,_values=inherit(ancestor,fill,clipped)
    regions=[]
    def visit(node,fill,clipped,band=None):
        if _tiles.tag(node)in {"defs","clipPath","mask","pattern"}:return
        if not node.tag.startswith("{http://www.w3.org/2000/svg}"):
            raise ValueError("Physical elevation paint must use the SVG namespace")
        fill,clipped,values=inherit(node,fill,clipped)
        if "data-band"in values:
            if not values["data-band"].isdigit():
                raise ValueError("Physical elevation paint requires its integer palette band")
            band=int(values["data-band"])
        if _tiles.tag(node)=="path":
            if (not clipped or band is None or not 0<=band<len(palette)
                    or values.get("fill-rule")!="evenodd"or values.get("stroke","none")!="none"
                    or fill!="#"+"".join(f"{int(v):02x}"for v in palette[band])):
                raise ValueError("Physical elevation paint must declare its actual band colour and owned clipping")
            geometry=_tiles._svg.path_geometry(values.get("d",""))
            if not geometry.is_valid:
                raise ValueError("Physical elevation paint must remain valid without repair")
            regions.append((band,geometry))
        elif _tiles.tag(node)not in {"g","title","desc"}:
            raise ValueError("Physical elevation layer contains unsupported painted geometry")
        for child in node:visit(child,fill,clipped,band)
    visit(partitions[0],fill,clipped)
    shapely.prepare(land)
    displayed=np.zeros(water.shape,dtype=bool)
    for first in range(0,height,32):
        last=min(height,first+32)
        x,y=np.meshgrid(np.arange(width)+.5,np.arange(first,last)+.5)
        displayed[first:last]=shapely.intersects_xy(land,x,y)
    rows,columns=np.nonzero(displayed)
    xx,yy=columns+.5,rows+.5
    painted=np.full(len(rows),-1,dtype=np.int16)
    for band,geometry in regions:
        shapely.prepare(geometry)
        # Membership in the actual owned clip is already represented by these
        # query points. This is exactly SVG clipping, not an external mask or
        # reconstructed physical land supplied in place of the artifact.
        painted[shapely.intersects_xy(geometry,xx,yy)]=band
    all_paint=np.full(water.shape,-1,dtype=np.int16);all_paint[rows,columns]=painted
    dry=water==0
    expected=elevation_display_indices(terrain.palette_elevation(terrain.native_m))
    missing=dry&(all_paint<0);wrong=dry&(all_paint>=0)&(all_paint!=expected)
    clip_wrong=displayed!=dry
    mismatch=missing|wrong
    report={"schema":"accepted-physical-surface-native-display-v1","status":"ok",
            "layer":"elevation-bands","checkedNativeCenters":int(water.size),
            "checkedDryCenters":int(dry.sum()),"checkedWetCenters":int((~dry).sum()),
            "clipMismatchCount":int(clip_wrong.sum()),
            "landCentersDisplayedAsWater":int((dry&~displayed).sum()),
            "waterCentersDisplayedAsLand":int((~dry&displayed).sum()),
            "missingLandColorCount":int(missing.sum()),"wrongLandColorCount":int(wrong.sum()),
            "colorMismatchCount":int(mismatch.sum()),
            "waterPaintCount":int((~dry&(all_paint>=0)).sum()),
            "bandPathCount":len(regions),"paintOrder":[band for band,_geometry in regions],
            "exportBytes":artifact["bytes"],"exportDigest":artifact["digest"],
            "exportEncoding":artifact["encoding"],"decodedExportBytes":artifact["decodedBytes"],
            "decodedExportDigest":artifact["decodedDigest"],
            "nativeViewBox":[0,0,width,height],"ownedClipId":"land-silhouette-clip",
            "examples":[{"row":int(row),"column":int(column),"expectedBand":int(expected[row,column]),
                         "paintedBand":int(all_paint[row,column]),"displayedLand":bool(displayed[row,column])}
                        for row,column in np.argwhere(mismatch|clip_wrong)[:10]]}
    if report["clipMismatchCount"]or report["colorMismatchCount"]or report["waterPaintCount"]:
        report["status"]="failed"
    return land,report


def _file_digest(path):
    digest=sha256()
    with path.open("rb")as stream:
        while block:=stream.read(1<<20):digest.update(block)
    return digest.hexdigest()


def check_review(review, *, grid_path, physical_source_path):
    grid = WorldGrid.load(grid_path)
    surface = load_surface_bundle(physical_source_path)
    original_grid_digest=grid.content_digest()
    original_source_digest=_file_digest(physical_source_path)
    original_heights_digest=sha256(surface.relative_elevation_m.tobytes()).hexdigest()
    terrain = terrain_from_source(grid, surface)
    palette, _ = _elevation_palette_for(grid)
    manifest = _tiles.read_manifest(review)
    _tiles.check_published_city_files(review,manifest)
    if (manifest["height"], manifest["width"]) != grid.shape:
        raise ValueError("delivered detail tiles differ from the physical grid dimensions")
    physical_land,physical_export=check_physical_export(review/"physical-surface.svgz",
        terrain=terrain,palette=palette,water=grid.water)
    major_source,major_proof=_read_major_sources(review/"elevation-contours.svgz",terrain,
                                               physical_land=physical_land)
    stage_proof=check_staged_major_sources(review.parent/"physical-contours",major_source,terrain=terrain,
        source_identity={"gridDigest":original_grid_digest,
                         "rawElevationSha256":original_heights_digest,
                         "physicalDiagnostics":dict(surface.diagnostics)})
    city_source,city_proof=_read_city_sources(review/"city-contours.json",terrain,original_grid_digest)
    replay_empty={"checkedPaths":0,"mismatchCount":0,"maxDeliveryDistanceCells":0.,"unexpectedHeightCount":0}
    report = {"status":"ok", "level":"detail", "checkedTiles":0,
              "checkedDryCenters":0, "checkedWetCenters":0, "colorMismatchCount":0,
              "waterPaintCount":0, "missingLandArea":0., "zeroContourPathCount":0,
              "contours":dict(replay_empty),"sourceVertices":major_proof,"examples":[]}
    city_report = {"status":"ok","kind":"relief-overlay","composedBase":"detail",
                   "sourceField":"accepted-continuous-ground","checkedTiles":0,
                   "checkedDryCenters":0,"checkedWetCenters":0,"fineMinorContourPathCount":0,
                   "contours":dict(replay_empty),"sourceVertices":city_proof}
    published = set(manifest["levels"][3]["publishedTiles"])
    for row in range(manifest["rows"]):
        for column in range(manifest["columns"]):
            tile = _tiles.read_tile(review, manifest, "detail", row, column)
            result = check_tile(tile, terrain=terrain, palette=palette, water=grid.water,source=major_source)
            report["checkedTiles"] += 1
            for name in ("checkedDryCenters", "checkedWetCenters", "colorMismatchCount", "waterPaintCount",
                         "missingLandArea", "zeroContourPathCount"):
                report[name] += result[name]
            for name in ("checkedPaths", "mismatchCount","unexpectedHeightCount"):
                report["contours"][name] += result["contours"][name]
            report["contours"]["maxDeliveryDistanceCells"]=max(report["contours"]["maxDeliveryDistanceCells"],
                                                                result["contours"]["maxDeliveryDistanceCells"])
            report["examples"].extend({"tile":result["tile"], **item} for item in result["examples"][:max(0, 10-len(report["examples"]))])
            if f"{row}-{column}" in published:
                overlay = _tiles.read_city_overlay(review,manifest,row,column,base_tile=tile)
                fine = check_city_tile(overlay,terrain=terrain,palette=palette,water=grid.water,source=city_source)
                city_report["checkedTiles"] += 1
                for name in ("checkedDryCenters","checkedWetCenters","fineMinorContourPathCount"):
                    city_report[name] += fine[name]
                for name in ("checkedPaths","mismatchCount","unexpectedHeightCount"):
                    city_report["contours"][name] += fine["contours"][name]
                city_report["contours"]["maxDeliveryDistanceCells"]=max(city_report["contours"]["maxDeliveryDistanceCells"],
                                                                         fine["contours"]["maxDeliveryDistanceCells"])
    if (city_proof["mismatchCount"] or city_report["contours"]["mismatchCount"]):
        city_report["status"] = "failed"
    if (report["colorMismatchCount"] or report["waterPaintCount"] or report["missingLandArea"] > 1e-7
            or report["zeroContourPathCount"] or report["contours"]["mismatchCount"] or major_proof["mismatchCount"]
            or city_report["status"] != "ok"or physical_export["status"]!="ok"or stage_proof["status"]!="ok"):
        report["status"] = "failed"
    report["contours"]["verification"] = "Actual continuous-ground source roots, then the delivered cuts of those same shared linear curves; eight-decimal coordinate encoding only."
    city_report["contours"]["verification"] = "Original continuous-ground source roots at true 100-metre minor levels, inherited single alpha support and exact detail land clip; no colour or major replacement."
    immutable=(original_grid_digest==grid.content_digest()
               and original_source_digest==_file_digest(physical_source_path)
               and original_heights_digest==sha256(surface.relative_elevation_m.tobytes()).hexdigest()
               and np.array_equal(terrain.native_m,surface.relative_elevation_m))
    report["sourceImmutable"]={"passed":bool(immutable),"gridDigest":original_grid_digest,
                              "physicalSourceDigest":original_source_digest,
                              "physicalSourceBytes":physical_source_path.stat().st_size,
                              "nativeHeightDigest":original_heights_digest}
    if not immutable:report["status"]="failed"
    report["cityDetail"] = city_report
    report["physicalExport"] = physical_export
    report["sourceStage"] = stage_proof
    report["sourceArtifacts"]={
        "physical-surface.svgz":{"encoding":"gzip","bytes":physical_export["exportBytes"],
            "digest":physical_export["exportDigest"],"decodedBytes":physical_export["decodedExportBytes"],
            "decodedDigest":physical_export["decodedExportDigest"]},
        "elevation-contours.svgz":major_proof["artifact"],
        "city-contours.json":{"bytes":(review/"city-contours.json").stat().st_size,
                              "digest":_file_digest(review/"city-contours.json")}}
    report["sourceArtifacts"]["city-contours.json"]["schema"]="accepted-physical-minor-curves-v1"
    report["terrainRefinement"] = terrain.diagnostics
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--physical-source", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = check_review(args.review, grid_path=args.grid, physical_source_path=args.physical_source)
    text = json.dumps(result, indent=2) + "\n"
    report_path = args.report or args.review / "physical-relief-display-check.json"
    report_path.write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
