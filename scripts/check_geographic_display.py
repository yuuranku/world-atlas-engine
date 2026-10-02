"""Verify saved summit evidence and delivered mountain lettering independently."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy import ndimage
import shapely

from world_atlas.core.model import WorldGrid
from world_atlas.core.society.storage import load_society


PEAK_TYPES = {"peak"}
_TOKEN = re.compile(r"[MmLlQq]|[-+]?(?:\d*\.?\d+)(?:[eE][-+]?\d+)?")


def _tag(node):
    return node.tag.rsplit("}", 1)[-1]


def _mountain_components(grid):
    """Four-way highland components, including the actual longitude seam."""
    labels,count = ndimage.label((grid.water == 0) & (grid.elevation >= .56))
    parent = np.arange(count+1)

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for a,b in zip(labels[:,0],labels[:,-1]):
        if a and b:
            parent[root(int(b))] = root(int(a))
    return np.asarray([root(index) for index in range(count+1)])[labels]


def _peak_evidence(grid,raw,row,column):
    height,width = grid.shape
    value = float(raw[row,column])
    neighbours = [float(raw[row+dy,(column+dx)%width])
                  for dy in (-1,0,1) for dx in (-1,0,1)
                  if (dy or dx) and 0 <= row+dy < height
                  and grid.water[row+dy,(column+dx)%width] == 0]
    maxima = bool(neighbours and value > max(neighbours))
    drops = []
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if not (dy or dx):
                continue
            samples = [float(raw[row+dy*k,(column+dx*k)%width]) for k in range(1,7)
                       if 0 <= row+dy*k < height and grid.water[row+dy*k,(column+dx*k)%width] == 0]
            drops.append(value-min(samples) if samples else -math.inf)
    prominence = min(drops)
    return {"elevationMeters":value,"strictRawLocalMaximum":maxima,
            "localRidgeReliefMeters":prominence if math.isfinite(prominence) else None,
            "supported":bool(grid.water[row,column] == 0 and grid.elevation[row,column] >= .56
                             and value > 0 and maxima and prominence >= 80)}


def _baseline_envelopes(data):
    """Each delivered quadratic lies inside its three-point convex envelope."""
    if re.sub(r"[\s,]+","",_TOKEN.sub("",data)):
        raise ValueError("Delivered mountain baseline must contain only M/L/Q commands")
    tokens,index,command,current = _TOKEN.findall(data),0,None,None
    envelopes = []
    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
        count = 4 if command in ("Q","q") else 2
        if command not in ("M","m","L","l","Q","q") or index+count > len(tokens):
            raise ValueError("Malformed delivered mountain baseline")
        points = np.asarray([float(value) for value in tokens[index:index+count]]).reshape(-1,2)
        index += count
        if command.islower():
            points += np.zeros(2) if current is None else current
        if command.upper() == "M":
            current = points[0]
            envelopes.append(shapely.Point(current))
            command = "l" if command.islower() else "L"
        else:
            if current is None:
                raise ValueError("Mountain baseline has no starting point")
            envelopes.append(shapely.MultiPoint(np.vstack((current,points))).convex_hull)
            current = points[-1]
    return envelopes


def _outside_component(envelope,components,component):
    if envelope.is_empty:
        return True
    west,north,east,south = envelope.bounds
    height,width = components.shape
    support = [shapely.box(column,row,column+1,row+1)
               for row in range(max(0,math.floor(north)-1),min(height,math.floor(south)+2))
               for column in range(math.floor(west)-1,math.floor(east)+2)
               if components[row,column % width] == component]
    return not shapely.covers(shapely.union_all(support),envelope)


def check_scene(grid,raw,society,markup):
    raw = np.asarray(raw,dtype=float)
    if raw.shape != grid.shape or not np.all(np.isfinite(raw)):
        raise ValueError("Verified raw metre bundle must exactly match the native grid")
    document = ET.fromstring(markup)
    groups = [node for node in document.iter() if node.get("id") == "geographic-labels"]
    if len(groups) != 1:
        raise ValueError("Delivered scene must contain its single geographical label layer")
    labels = [node for node in groups[0].iter() if _tag(node) == "text" and node.get("data-feature-type")]
    signature = lambda feature:(feature.feature_type,feature.tier,feature.name)
    displayed = lambda node:(node.get("data-feature-type"),node.get("data-feature-tier"),"".join(node.itertext()))
    expected = Counter(signature(feature) for feature in society.geographic_features)
    actual = Counter(displayed(node) for node in labels)
    missing,unexpected = expected-actual,actual-expected
    label_by_signature = {}
    for node in labels:
        label_by_signature.setdefault(displayed(node),[]).append(node)
    paths = {node.get("id"):node.get("d","") for node in groups[0].iter() if _tag(node) == "path"}
    components = _mountain_components(grid)
    peaks,mountains,errors = [],[],[]
    peak_positions = Counter()
    for feature in society.geographic_features:
        row,column = feature.row,feature.column
        if not (0 <= row < grid.shape[0] and 0 <= column < grid.shape[1]):
            errors.append({"id":feature.identifier,"reason":"saved feature lies outside the native grid"})
            continue
        matched = label_by_signature.get(signature(feature),[])
        if feature.feature_type in PEAK_TYPES:
            evidence = _peak_evidence(grid,raw,row,column)
            peaks.append({"id":feature.identifier,"row":row,"column":column,**evidence})
            peak_positions[(column+.5,row+.5,feature.feature_type)] += 1
            if not evidence["supported"]:
                errors.append({"id":feature.identifier,"reason":"summit lacks strict raw maximum/local ridge or snow support"})
        if feature.feature_type != "mountain":
            continue
        component = int(components[row,column])
        if not component or len(matched) != 1:
            errors.append({"id":feature.identifier,"reason":"mountain anchor or unique delivered text lacks support"})
            continue
        label = matched[0]
        label_natural = not any(attribute in label.attrib
                                for attribute in ("textLength","lengthAdjust","data-base-text-length","transform"))
        child = next((node for node in label if _tag(node) == "textPath"),None)
        if child is None:
            correct = (float(label.get("x","nan")),float(label.get("y","nan"))) == (column+.5,row+.5)
            mountains.append({"id":feature.identifier,"mode":"point","correctNativeAnchor":correct})
            if not correct or not label_natural:
                errors.append({"id":feature.identifier,"reason":"compact massif label moved or forced glyph fitting"})
            continue
        href = child.get("href",child.get("{http://www.w3.org/1999/xlink}href",""))
        if href != f"#geographic-path-{feature.identifier}" or href[1:] not in paths:
            errors.append({"id":feature.identifier,"reason":"mountain text references another feature baseline"})
            continue
        envelopes = _baseline_envelopes(paths[href[1:]])
        outside = sum(_outside_component(envelope,components,component) for envelope in envelopes)
        natural = label_natural and not any(attribute in child.attrib
                                            for attribute in ("textLength","lengthAdjust","data-base-text-length"))
        mountains.append({"id":feature.identifier,"mode":"curve","checkedPrimitiveEnvelopes":len(envelopes),
                          "outsideOwnComponent":outside,"naturalGlyphs":natural})
        if outside or not natural:
            errors.append({"id":feature.identifier,"reason":"mountain baseline leaves its component or forces glyph fitting"})
    markers = Counter((float(node.get("data-map-x","nan")),float(node.get("data-map-y","nan")),
                       node.get("data-geographic-symbol"))
                      for node in document.iter() if node.get("data-geographic-symbol") in PEAK_TYPES)
    marker_mismatches = sum((peak_positions-markers).values())+sum((markers-peak_positions).values())
    height_errors = []
    for node in document.iter():
        if node.get("data-geographic-symbol") != "peak":
            continue
        column,row = float(node.get("data-map-x","nan"))-.5,float(node.get("data-map-y","nan"))-.5
        heights = [child for child in node.iter() if child.get("data-height-m") is not None]
        if (not column.is_integer() or not row.is_integer() or not 0 <= column < grid.shape[1]
                or not 0 <= row < grid.shape[0] or len(heights) != 1):
            height_errors.append("summit marker lacks a unique delivered native metre height")
            continue
        height = heights[0]
        value = float(raw[int(row),int(column)])
        if (height.get("data-height-datum") != "model-sea-level"
                or abs(float(height.get("data-height-m"))-value) > 1e-8
                or height.text != str(round(value))):
            height_errors.append("delivered summit number differs from accepted raw metres/datum")
    if height_errors:
        errors.extend({"reason": reason} for reason in height_errors)
    passed = not missing and not unexpected and not errors and marker_mismatches == 0
    return {"schema":"delivered-geographical-display-v1","status":"ok" if passed else "failed",
            "scope":"Saved summit physical evidence, delivered summit markers and mountain text; regional point records do not assert unrecorded landform extents.",
            "savedFeatures":len(society.geographic_features),"deliveredLabels":len(labels),
            "missingLabels":[{"type":key[0],"tier":key[1],"name":key[2],"count":count} for key,count in missing.items()],
            "unexpectedLabels":[{"type":key[0],"tier":key[1],"name":key[2],"count":count} for key,count in unexpected.items()],
            "peaks":peaks,"mountains":mountains,"peakMarkerMismatches":marker_mismatches,
            "peakHeightNumberErrors":height_errors,"errors":errors,
            "ridgeReliefCriterion":{"minimumMeters":80,"radiusNativeCells":6,
                                    "meaning":"minimum eight-direction ridge drop; not global topographic prominence"}}


def check_review(review, *, grid_path, society_path, physical_source_path):
    grid = WorldGrid.load(grid_path)
    society = load_society(society_path,expected_grid_digest=grid.content_digest())
    with np.load(physical_source_path,allow_pickle=False) as bundle:
        raw = bundle["relative_elevation_m"]
    scene = review/"atlas-scene.svg"
    raw_markup = scene.read_bytes()
    result = check_scene(grid,raw,society,raw_markup.decode("utf-8"))
    result["inputs"] = {"review":str(review),"grid":str(grid_path),"society":str(society_path),
                        "physicalSource":str(physical_source_path),"sceneSHA256":hashlib.sha256(raw_markup).hexdigest()}
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("review",type=Path)
    cli.add_argument("--grid",type=Path,required=True)
    cli.add_argument("--society",type=Path,required=True)
    cli.add_argument("--physical-source",type=Path,required=True)
    cli.add_argument("--report",type=Path)
    args = cli.parse_args()
    report = check_review(args.review,grid_path=args.grid,society_path=args.society,
                          physical_source_path=args.physical_source)
    destination = args.report or args.review/"geographic-display-check.json"
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"report":str(destination),
                      "savedFeatures":report["savedFeatures"],"checkedPeaks":len(report["peaks"]),
                      "checkedMountains":len(report["mountains"]),"errors":report["errors"]},ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
