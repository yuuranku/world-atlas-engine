"""Consume delivered landform paint; source masks are numeric evidence only."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import shapely

from world_atlas.core.model import WorldGrid
from world_atlas.core.procedural_planet import load_surface_bundle
from world_atlas.core.thematic import derive_thematic_layers
from world_atlas.core.landforms import derive_landform_inventory


def _consumer(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+".py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tiles=_consumer("check_atlas_tiles")
_physical=_consumer("check_physical_relief_display")
AREA_TYPES=("wetland","snow-mountain","arid-upland")
_URL=re.compile(r"url\(#([^)]*)\)")


def source_evidence(grid_path,physical_source_path):
    grid=WorldGrid.load(grid_path)
    source=load_surface_bundle(physical_source_path)
    from world_atlas.core.ecological_sources import derive_ecological_sources
    inventory=derive_landform_inventory(grid,derive_thematic_layers(grid,
                                       ecological_sources=derive_ecological_sources(grid, source)),
                                       raw_elevation_m=source.relative_elevation_m)
    return grid,inventory


def export_tile(markup,grid,*,key,overview=False):
    document=_tiles.tile_document(markup)
    required={"land-silhouette-clip","water-silhouette-clip"} if overview else {"land-silhouette-clip"}
    clips={}
    for node in document.iter():
        if _tiles.tag(node)!="clipPath":
            continue
        identifier=node.get("id")
        if identifier not in required or identifier in clips or node.get("clipPathUnits")!="userSpaceOnUse":
            raise ValueError(f"{key}: export has a foreign, duplicate or non-native shared clip")
        clips[identifier]=_tiles._svg.union_paths([child.attrib for child in node.iter() if _tiles.tag(child)=="path"])
    if set(clips)!=required:
        raise ValueError(f"{key}: export lacks its unique native shared physical clips")
    land=clips["land-silhouette-clip"]
    return {"key":key,"clips":clips,"landClipId":"land-silhouette-clip",
            "land":land,"rectangle":shapely.box(0,0,grid.shape[1],grid.shape[0]),
            "payload":{"bounds":[0,0,grid.shape[1],grid.shape[0]]}}


def _filtered_regions(document,tile,*,textured):
    selected=copy.deepcopy(document)
    def visit(node,in_defs=False):
        in_defs=in_defs or _tiles.tag(node) in {"defs","clipPath","pattern"}
        if in_defs:
            return
        if _tiles.tag(node)=="path":
            kind=node.get("data-landform-type")
            chosen=kind in AREA_TYPES and (node.get("data-landform-texture")=="true")==textured
            if chosen:
                node.set("data-check-landform",str(AREA_TYPES.index(kind)))
            else:
                node.set("fill","none")
                node.set("style",node.get("style","")+";fill:none")
        for child in node:
            visit(child,in_defs)
    visit(selected)
    regions=_tiles.painted_regions(_tiles.ET.tostring(selected,encoding="unicode"),tile,
                                  owner_attribute="data-check-landform")
    return {kind:shapely.union_all([geometry for owner,geometry in regions if owner==index])
            for index,kind in enumerate(AREA_TYPES)}


def _visible_features(document,tile,*,overview=False):
    features=[]
    def visit(node,in_defs=False,clipped=False,layer=None):
        tag=_tiles.tag(node)
        in_defs=in_defs or tag in {"defs","clipPath","pattern"}
        if in_defs:
            return
        attributes=_tiles._attributes(node)
        if attributes.get("display")=="none" or attributes.get("visibility")=="hidden" or "hidden" in attributes:
            if any(child.get("data-landform-type") for child in node.iter()):
                raise ValueError(f'{tile["key"]}: delivered landform source contains hidden paint')
            return
        reference=attributes.get("clip-path")
        if reference and reference!="none":
            match=_URL.fullmatch(reference)
            if not match or match[1] not in tile["clips"]:
                raise ValueError(f'{tile["key"]}: visible paint references a foreign shared clip')
            clipped=clipped or match[1]==tile["landClipId"]
        layer=attributes.get("data-tile-layer",layer)
        kind=attributes.get("data-landform-type")
        if kind:
            if tag!="path" or layer!="geographic-textures" or not clipped or attributes.get("transform"):
                raise ValueError(f'{tile["key"]}: landform geometry lacks the real geographic-textures/shared-land composition')
            if kind not in (*AREA_TYPES,"steep-slope"):
                raise ValueError(f'{tile["key"]}: unexpected landform paint category {kind}')
            if float(attributes.get("opacity","1"))<=0:
                raise ValueError(f'{tile["key"]}: landform paint is invisible')
            fine=attributes.get("data-landform-texture")=="true"
            if fine:
                minimum=float(attributes.get("data-min-scale","nan"))
                maximum=float(attributes.get("data-max-scale","nan"))
                if not math.isfinite(minimum) or not math.isfinite(maximum) or not 0<minimum<maximum:
                    raise ValueError(f'{tile["key"]}: detailed landform paint has an invalid scale window')
            if overview and (fine or "data-landform-hachure" in attributes):
                raise ValueError(f'{tile["key"]}: overview contains detailed texture or slope hachures')
            features.append((node,attributes))
        for child in node:
            visit(child,in_defs,clipped,layer)
    visit(document)
    return features


def _patterns(document,features,tile,*,local=False,overview=False):
    patterns=[node for node in document.iter() if _tiles.tag(node)=="pattern"]
    counts=Counter(node.get("id") for node in patterns)
    if any(count!=1 for count in counts.values()):
        raise ValueError(f'{tile["key"]}: pattern definitions are duplicated')
    if overview and patterns:
        raise ValueError(f'{tile["key"]}: overview contains detailed paint-server definitions')
    definitions={node.get("id"):node for node in patterns}
    used=set()
    for node,attributes in features:
        if attributes.get("data-landform-texture")!="true" or attributes["data-landform-type"]=="steep-slope":
            continue
        reference=_URL.fullmatch(attributes.get("fill",""))
        if not reference or reference[1] not in definitions:
            raise ValueError(f'{tile["key"]}: texture has no unique local surface pattern')
        identifier=reference[1]
        definition=definitions[identifier]
        used.add(identifier)
        if (definition.get("data-landform-pattern")!=attributes["data-landform-type"]
                or definition.get("patternUnits")!="userSpaceOnUse"
                or definition.get("patternContentUnits")!="userSpaceOnUse"
                or float(definition.get("width","nan"))!=8 or float(definition.get("height","nan"))!=8):
            raise ValueError(f'{tile["key"]}: pattern kind/coordinates contradict its actual region')
        if local:
            level,position=tile["key"].split("/")
            if not identifier.endswith(f"-tile-{level}-{position}"):
                raise ValueError(f'{tile["key"]}: pattern identifier can collide with another mounted block')
        glyphs=[child for child in definition.iter() if _tiles.tag(child) in {"path","rect","circle"}]
        if not glyphs or (attributes["data-landform-type"]=="arid-upland" and any(_tiles.tag(glyph)=="circle" for glyph in glyphs)):
            raise ValueError(f'{tile["key"]}: missing texture or unsupported sand-grain convention')
    if set(definitions)!=used:
        raise ValueError(f'{tile["key"]}: orphan cartographic pattern definitions')
    return len(patterns)


def _native_regions(regions,tile,grid,inventory):
    x,y,width,height=tile["payload"]["bounds"]
    rows,columns=np.mgrid[y:y+height,x:x+width]
    wet=grid.water[rows,columns]!=0
    report={}
    for kind in AREA_TYPES:
        actual=shapely.contains_xy(regions[kind],columns+.5,rows+.5)
        expected=inventory.masks[kind][rows,columns]
        mismatch=actual!=expected
        examples=np.column_stack(np.nonzero(mismatch))[:5]
        report[kind]={"checkedCenters":int(actual.size),"sourceSupportCenters":int(expected.sum()),
                      "paintedCenters":int(actual.sum()),"mismatchCount":int(mismatch.sum()),
                      "waterPaintCount":int((actual&wet).sum()),
                      "examples":[[int(x+column),int(y+row)] for row,column in examples]}
    return report


def _hachures(features,tile,grid,inventory):
    centers=[]
    directions=[]
    for node,attributes in features:
        if attributes["data-landform-type"]!="steep-slope":
            continue
        if (attributes.get("data-landform-hachure")!="downslope" or attributes.get("data-landform-texture")!="true"
                or attributes.get("fill")!="none" or attributes.get("vector-effect")!="non-scaling-stroke"):
            raise ValueError(f'{tile["key"]}: slope annotation lacks measured-direction/stable-stroke metadata')
        for line in _physical._line_points(attributes.get("d","")):
            if len(line)!=2:
                raise ValueError(f'{tile["key"]}: slope hachures must be independent two-point short lines')
            if not shapely.intersects(shapely.LineString(line),tile["rectangle"]):
                continue
            centers.append(line.mean(axis=0))
            directions.append(line[1]-line[0])
    x,y,width,height=tile["payload"]["bounds"]
    expected=inventory.masks["steep-slope"][y:y+height,x:x+width]
    expected_count=int(expected.sum())
    if not centers:
        return {"checkedLines":0,"sourceSupportCenters":expected_count,"missingCenters":expected_count,
                "unsupportedCenters":0,"directionMismatchCount":0,"duplicateCenters":0,"maxDirectionResidual":0.}
    centers=np.asarray(centers)
    directions=np.asarray(directions)
    columns=np.floor(centers[:,0]).astype(int)
    rows=np.floor(centers[:,1]).astype(int)
    inside=(columns>=x)&(columns<x+width)&(rows>=y)&(rows<y+height)
    if not inside.all():
        raise ValueError(f'{tile["key"]}: a visible hachure lacks a unique native support center')
    supported=inventory.masks["steep-slope"][rows,columns]&(grid.water[rows,columns]==0)
    anchored=np.max(np.abs(centers-np.column_stack((columns+.5,rows+.5))),axis=1)<=1e-8
    lengths=np.sum(np.abs(directions),axis=1)
    if np.any(np.abs(lengths-.16)>4e-8):
        raise ValueError(f'{tile["key"]}: slope glyph exceeds its declared native-cell support footprint')
    extents=grid.metadata["extents"]
    latitude=extents["north"]-(rows+.5)*(extents["north"]-extents["south"])/grid.shape[0]
    physical=np.column_stack((directions[:,0]*(extents["east"]-extents["west"])/grid.shape[1]*np.cos(np.radians(latitude)),
                              -directions[:,1]*(extents["north"]-extents["south"])/grid.shape[0]))
    physical/=np.linalg.norm(physical,axis=1)[:,None]
    truth=np.column_stack((inventory.direction_east[rows,columns],inventory.direction_north[rows,columns]))
    residual=np.linalg.norm(physical-truth,axis=1)
    keys=rows*grid.shape[1]+columns
    unique,count=np.unique(keys,return_counts=True)
    return {"checkedLines":len(centers),"sourceSupportCenters":expected_count,
            "missingCenters":max(0,expected_count-int(np.unique(keys[supported&anchored]).size)),
            "unsupportedCenters":int((~supported|~anchored).sum()),"directionMismatchCount":int((residual>2e-6).sum()),
            "duplicateCenters":int(np.maximum(count-1,0).sum()),"maxDirectionResidual":float(residual.max())}


def check_markup(markup,tile,grid,inventory,*,overview=False,local=False):
    document=_tiles.tile_document(markup)
    features=_visible_features(document,tile,overview=overview)
    pattern_count=_patterns(document,features,tile,local=local,overview=overview)
    tints=_native_regions(_filtered_regions(document,tile,textured=False),tile,grid,inventory)
    result={"key":tile["key"],"patterns":pattern_count,"tints":tints}
    if overview:
        # Overview is a restrained simplified hint, not a second native class.
        result["scope"]="simplified regional hints with common clipping; detailed texture/hachures excluded"
        result["status"]="ok" if all(item["waterPaintCount"]==0 for item in tints.values()) else "failed"
        return result
    result["textures"]=_native_regions(_filtered_regions(document,tile,textured=True),tile,grid,inventory)
    result["hachures"]=_hachures(features,tile,grid,inventory)
    failures=sum(item["mismatchCount"]+item["waterPaintCount"] for regions in (tints,result["textures"]) for item in regions.values())
    failures+=sum(result["hachures"][name] for name in ("missingCenters","unsupportedCenters","directionMismatchCount","duplicateCenters"))
    result["status"]="ok" if failures==0 else "failed"
    return result


def check_static(review,*,grid_path,physical_source_path):
    started=time.monotonic()
    source_hashes,contract=accepted_source_evidence(review,grid_path,physical_source_path,require_complete=False)
    grid,inventory=source_evidence(grid_path,physical_source_path)
    path=review/"landform-regions.svg"
    encoded=path.read_bytes()
    markup=encoded.decode("utf8")
    result=check_markup(markup,export_tile(markup,grid,key=path.name),grid,inventory)
    after,_contract=accepted_source_evidence(review,grid_path,physical_source_path,require_complete=False)
    if after!=source_hashes:
        raise ValueError("Static landform source changed during its native check")
    return {"schema":"delivered-landform-display-v1","scope":"final-static-landform-precheck",
            "status":result["status"],"static":result,"sourceEvidence":inventory.diagnostics,
            "inputs":{"review":str(review),"grid":str(grid_path),"physicalSource":str(physical_source_path),
                      "landformSVG_SHA256":hashlib.sha256(encoded).hexdigest(),
                      "sourceSHA256":source_hashes},"acceptedSource":contract,
            "elapsedSeconds":round(time.monotonic()-started,3)}


def accepted_source_evidence(review,grid_path,physical_source_path,*,require_complete):
    """Validate the current human rebuild's canonical preserved source files."""
    world=review.parent.resolve()
    if grid_path.resolve()!=world/"grid" or physical_source_path.resolve()!=world/"source/physical-fields.npz":
        raise ValueError("Landform evidence must use the current world's canonical source paths")
    record_path=world/"regeneration.json"
    record=json.loads(record_path.read_text(encoding="utf8"))
    if (record.get("schema")!="accepted-world-v2" or record.get("rebuildKind")!="human-only-v1"
            or record.get("humanStatus")!="ok"
            or record.get("status") not in ({"complete"}if require_complete else {"building","complete"})):
        raise ValueError("Landform evidence requires the current accepted human rebuild contract")
    original=Path(record["sourceWorld"]).resolve()
    if original==world:
        raise ValueError("Preserved landform source world must be a separate world")
    paths=(grid_path/"world-grid.npz",grid_path/"world-grid.json",physical_source_path)
    hashes={path.resolve().relative_to(world).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()for path in paths}
    if any(hashlib.sha256((original/name).read_bytes()).hexdigest()!=value for name,value in hashes.items()):
        raise ValueError("Landform source changed from the preserved accepted world")
    bundle=hashes["source/physical-fields.npz"]
    proof_path=world/"source/physical-fields-verification.json"
    proof=json.loads(proof_path.read_text(encoding="utf8"))
    provenance=json.loads((world/"source/provenance.json").read_text(encoding="utf8"))
    if (record.get("fieldSha256","").lower()!=bundle
            or proof.get("schema")!="verified-physical-surface-replay-v1"
            or any(proof.get(key)is not True for key in ("verifiedExact","originalSourceUnchanged","nativePaletteReconstructionExact"))
            or proof.get("verifiedBundleSha256","").lower()!=bundle
            or proof.get("recipe")!=provenance["recipe"]):
        raise ValueError("Landform source physical replay contract has incorrect SHA or evidence")
    return hashes,{"sourceWorld":str(original),"regeneration":str(record_path),
                   "physicalVerification":str(proof_path),"physicalVerificationSHA256":hashlib.sha256(proof_path.read_bytes()).hexdigest()}


def bind_static_proof(review,*,grid_path,physical_source_path,static_report_path):
    """Reuse an exact export only with its own recorded input fingerprints."""
    encoded=static_report_path.read_bytes()
    proof=json.loads(encoded)
    if (proof.get("schema")!="delivered-landform-display-v1"
            or proof.get("scope")!="final-static-landform-precheck"
            or proof.get("status")!="ok" or proof.get("static",{}).get("status")!="ok"):
        raise ValueError("The static landform proof must be a successful independent native check")
    for name,path in (("review",review),("grid",grid_path),("physicalSource",physical_source_path)):
        if Path(proof["inputs"][name]).resolve()!=path.resolve():
            raise ValueError("The static landform proof refers to different delivered/source files")
    svg_hash=hashlib.sha256((review/"landform-regions.svg").read_bytes()).hexdigest()
    if svg_hash!=proof["inputs"]["landformSVG_SHA256"]:
        raise ValueError("The delivered landform export changed after its static native check")
    measured,contract=accepted_source_evidence(review,grid_path,physical_source_path,require_complete=True)
    if proof["inputs"]["sourceSHA256"]!=measured:
        raise ValueError("Static landform source changed from its own recorded input fingerprints")
    if proof["acceptedSource"]!=contract:
        raise ValueError("Static landform proof is bound to a different accepted physical source contract")
    static=proof["static"]
    if static["patterns"]!=len(AREA_TYPES):
        raise ValueError("The reused static proof lacks the three actual cartographic patterns")
    for family in ("tints","textures"):
        if set(static[family])!=set(AREA_TYPES) or any(
            item["mismatchCount"] or item["waterPaintCount"]
            or item["paintedCenters"]!=item["sourceSupportCenters"] for item in static[family].values()
        ):
            raise ValueError("The reused static proof contains native classification errors")
    if (any(static["hachures"][name] for name in
           ("missingCenters","unsupportedCenters","directionMismatchCount","duplicateCenters"))
            or not 0<=static["hachures"]["maxDirectionResidual"]<=2e-6):
        raise ValueError("The reused static proof contains physical slope glyph errors")
    return proof,{"report":str(static_report_path),"reportSHA256":hashlib.sha256(encoded).hexdigest(),
                  "landformSVG_SHA256":svg_hash,"acceptedSource":contract,
                  "immutableSourceSHA256":measured,
                  "scope":"Exact static SVG proof reused; its own recorded source fingerprints and current accepted source contract remain unchanged."}


def check_review(review,*,grid_path,physical_source_path,static_report_path):
    started=time.monotonic()
    proof,binding=bind_static_proof(review,grid_path=grid_path,physical_source_path=physical_source_path,
                                  static_report_path=static_report_path)
    grid,inventory=source_evidence(grid_path,physical_source_path)
    if proof["sourceEvidence"]!=inventory.diagnostics:
        raise ValueError("Current numeric landform evidence differs from the reused static proof")
    for family in ("tints","textures"):
        for kind in AREA_TYPES:
            metric=proof["static"][family][kind]
            if metric["checkedCenters"]!=grid.water.size or metric["sourceSupportCenters"]!=int(inventory.masks[kind].sum()):
                raise ValueError("Reused static proof did not check every current native centre")
    steep_count=int(inventory.masks["steep-slope"].sum())
    if (proof["static"]["hachures"]["checkedLines"]!=steep_count
            or proof["static"]["hachures"]["sourceSupportCenters"]!=steep_count):
        raise ValueError("Reused static proof did not check every current slope glyph")
    manifest=_tiles.read_manifest(review)
    _tiles.check_published_city_files(review,manifest)
    published=set(manifest["levels"][3]["publishedTiles"])
    city_checked=0
    static=proof["static"]
    scene=_tiles.tile_document((review/"atlas-scene.svg").read_text(encoding="utf8"))
    overview=[node for node in scene.iter() if node.get("id")=="overview-source"]
    if len(overview)!=1:
        raise ValueError("Actual atlas scene must contain its unique runtime overview-source")
    body=_tiles.ET.tostring(overview[0],encoding="unicode")
    far=check_markup(body,export_tile(body,grid,key="atlas-scene.svg#overview-source",overview=True),grid,inventory,overview=True)
    levels=[]
    for level in _tiles.base_levels(manifest):
        aggregate={"id":level["id"],"checkedTiles":0,"patterns":0,"nativeMismatchCount":0,"waterPaintCount":0,
                   "hachures":{"checkedLines":0,"sourceSupportCenters":0,"missingCenters":0,"unsupportedCenters":0,
                               "directionMismatchCount":0,"duplicateCenters":0,"maxDirectionResidual":0.},"failedTiles":[]}
        for row in range(manifest["rows"]):
            for column in range(manifest["columns"]):
                tile=_tiles.read_tile(review,manifest,level["id"],row,column)
                if any("data-landform-" in theme for theme in tile["payload"]["themes"].values()):
                    raise ValueError(f'{tile["key"]}: thematic markup duplicates physical landforms')
                if any(_tiles.tag(node)=="pattern" for node in _tiles.tile_document(tile["payload"]["ink"]).iter()):
                    raise ValueError(f'{tile["key"]}: pattern definitions must be published only in surface')
                measured=check_markup(tile["payload"]["surface"]+tile["payload"]["ink"],tile,grid,inventory,local=True)
                if level["id"]=="detail" and f"{row}-{column}" in published:
                    overlay=_tiles.read_city_overlay(review,manifest,row,column,base_tile=tile)
                    if any("data-landform-" in overlay["payload"][section] for section in ("surface","ink")):
                        raise ValueError(f'{overlay["key"]}: additive relief duplicates physical landform paint')
                    city_checked+=1
                aggregate["checkedTiles"]+=1
                aggregate["patterns"]+=measured["patterns"]
                for family in ("tints","textures"):
                    aggregate["nativeMismatchCount"]+=sum(item["mismatchCount"] for item in measured[family].values())
                    aggregate["waterPaintCount"]+=sum(item["waterPaintCount"] for item in measured[family].values())
                for name,value in measured["hachures"].items():
                    if name=="maxDirectionResidual":
                        aggregate["hachures"][name]=max(aggregate["hachures"][name],value)
                    else:
                        aggregate["hachures"][name]+=value
                if measured["status"]!="ok":
                    aggregate["failedTiles"].append(measured)
            if (row+1)%8==0:
                print(json.dumps({"consumer":"actual-landform-tiles","level":level["id"],"checkedRows":row+1,
                                  "nativeMismatchCount":aggregate["nativeMismatchCount"]}),flush=True)
        aggregate["status"]="ok" if not aggregate["failedTiles"] else "failed"
        levels.append(aggregate)
    failures=static["status"]!="ok" or far["status"]!="ok" or any(level["status"]!="ok" for level in levels)
    return {"schema":"delivered-landform-display-v1","scope":"final-static-and-actual-runtime-overview/base-tiles",
            "status":"failed" if failures else "ok","static":static,"staticProofBinding":binding,"overview":far,"levels":levels,
            "cityDetail":{"composition":"inherits unchanged detail landforms; adds relief only",
                          "composedBase":"detail","checkedOverlays":city_checked},
            "sourceEvidence":inventory.diagnostics,"inputs":{"review":str(review),"grid":str(grid_path),
                      "physicalSource":str(physical_source_path),"landformSVG_SHA256":binding["landformSVG_SHA256"]},
            "elapsedSeconds":round(time.monotonic()-started,3)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review",type=Path)
    parser.add_argument("--grid",type=Path,required=True)
    parser.add_argument("--physical-source",type=Path,required=True)
    parser.add_argument("--static-report",type=Path,required=True,
                        help="Successful immutable static native proof for this exact delivered export and saved world")
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    report=check_review(args.review,grid_path=args.grid,physical_source_path=args.physical_source,
                        static_report_path=args.static_report)
    destination=args.report or args.review/"landform-display-check.json"
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":report["status"],"report":str(destination),"elapsedSeconds":report["elapsedSeconds"]}))
    raise SystemExit(0 if report["status"]=="ok" else 1)


if __name__=="__main__":
    main()
