"""Reencode published globe SVG paths with the producer's exact frame anchors."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from world_atlas.core.svg_paths import COORDINATE_SCALE, integer_subpath_data


TOKEN = re.compile(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")


def linear_subpaths(data):
    tokens = TOKEN.findall(data)
    if any(token.isalpha() and token not in "MmLlHhVvZz" for token in tokens):
        return None  # Explicit Bezier/control paths already use absolute coordinates.
    paths, points = [], []
    current = (0,0)
    start = (0,0)
    command = None
    index = 0
    def coordinate(token):
        scaled = Decimal(token) * COORDINATE_SCALE
        if scaled != scaled.to_integral_value():
            raise ValueError("published geometry must use eight-decimal delivered coordinates")
        return int(scaled)
    while index < len(tokens):
        if tokens[index].isalpha():
            command=tokens[index];index+=1
            if command in "Zz":
                if points:paths.append((points,True));points=[]
                current=start
                continue
            if command in "Mm" and points:
                paths.append((points,False));points=[]
        if command is None:raise ValueError("SVG coordinates require a command")
        if command in "Hh":
            value=coordinate(tokens[index]);index+=1
            point=(current[0]+value if command=='h' else value,current[1])
        elif command in "Vv":
            value=coordinate(tokens[index]);index+=1
            point=(current[0],current[1]+value if command=='v' else value)
        else:
            a,b=coordinate(tokens[index]),coordinate(tokens[index+1]);index+=2
            point=(current[0]+a,current[1]+b) if command.islower() else (a,b)
        if command in "Mm":start=point;command='l' if command=='m' else 'L'
        current=point;points.append(point)
    if points:paths.append((points,False))
    return [(points[:-1] if closed and points[-1]==points[0] else points,closed) for points,closed in paths]


def reencode(markup):
    root=ET.fromstring(markup)
    changed=curves=0
    for node in root.iter():
        if node.tag.rsplit('}',1)[-1]!='path' or not node.get('d'):continue
        before=node.get('d');paths=linear_subpaths(before)
        if paths is None:curves+=1;continue
        after=' '.join(integer_subpath_data(points,closed=closed) for points,closed in paths)
        if linear_subpaths(after)!=paths:
            raise ValueError("absolute anchors must retain every delivered vertex and ring")
        node.set('d',after);changed+=1
    return ET.tostring(root,encoding='unicode'),changed,curves


def rewrite(review,backup):
    if backup.exists():raise FileExistsError(f"serialization audit backup must be fresh: {backup}")
    ET.register_namespace('','http://www.w3.org/2000/svg')
    source=review/'globe-data.js'
    original=source.read_text(encoding='utf-8')
    payload=json.loads(original.removeprefix('window.WorldAtlasGlobe=').strip().removesuffix(';'))
    backup.mkdir(parents=True)
    metrics={}
    for name in ('surface','ink'):
        before=payload[name]
        payload[name],count,curves=reencode(before)
        if len(payload[name].encode('utf-8'))>64*1024*1024:raise ValueError(f"offline SVG byte budget: {name}")
        metrics[name]={'paths':count,'preservedCurves':curves,'coordinatesChanged':0,'bytes':len(payload[name].encode('utf-8'))}
    prepared={}
    for theme,filename in payload['textures'].items():
        path=review/filename
        markup,count,curves=reencode(path.read_text(encoding='utf-8'))
        if len(markup.encode('utf-8'))>64*1024*1024:raise ValueError(f"offline SVG byte budget: {theme}")
        prepared[path]=markup
        metrics[theme]={'paths':count,'preservedCurves':curves,'coordinatesChanged':0,'bytes':len(markup.encode('utf-8'))}
        print(json.dumps({'stage':'exact-svg-reencoding','theme':theme,**metrics[theme]}),flush=True)
    prepared[source]='window.WorldAtlasGlobe='+json.dumps(payload,ensure_ascii=True)+';\n'
    hashes={}
    for path,markup in prepared.items():
        shutil.copy2(path,backup/path.name)
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_text(markup,encoding='utf-8')
        hashes[path.name]={'before':before,'after':hashlib.sha256(path.read_bytes()).hexdigest()}
    report={'schema':'exact-svg-frame-anchor-reencoding-v1','status':'ok','role':'lossless-delivery-serialization',
            'sourceGeometryChanged':False,'coordinateScale':COORDINATE_SCALE,'maxRelativeChainVertices':64,
            'backup':str(backup),'metrics':metrics,'assetHashes':hashes}
    (review/'globe-serialization-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('review',type=Path);cli.add_argument('--backup',type=Path,required=True)
    args=cli.parse_args();print(json.dumps(rewrite(args.review,args.backup)),flush=True)
