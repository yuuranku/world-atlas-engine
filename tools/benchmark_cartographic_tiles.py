"""Compare bounded exports of saved native geometry with a prior Git version."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import shapely
from shapely.affinity import translate

from world_atlas.core import cartographic_tiles as current


def digest(directory):
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob('*.json'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference-ref', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('coverage_decoder', root/'scripts/check_coastal_coverage.py')
    coverage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coverage)
    tree = ET.parse(args.review/'land-potential.svg')
    ns = {'s':'http://www.w3.org/2000/svg'}
    land = coverage.path_geometry(tree.find('.//s:clipPath/s:path', ns).attrib['d'])
    land = translate(land, xoff=-1280, yoff=-448)
    features = []
    for path in tree.findall('.//s:g/s:path', ns):
        geometry = translate(coverage.path_geometry(path.attrib['d']), xoff=-1280, yoff=-448)
        attributes = {name:value for name,value in path.attrib.items() if name not in ('d','clip-path')}
        for part in shapely.get_parts(geometry):
            features.append((part, attributes))
    reference = subprocess.check_output(
        ['git','show',args.reference_ref+':src/world_atlas/core/cartographic_tiles.py'], cwd=root)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/'reference.py'
        path.write_bytes(reference)
        spec = importlib.util.spec_from_file_location('world_atlas.core._reference_tiles', path)
        previous = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = previous
        spec.loader.exec_module(previous)
        reports = []
        for name, module in (('before',previous),('after',current)):
            shared = [module.TileFeature(geometry,attrs,'potential',theme='potential',section='theme')
                      for geometry,attrs in features]
            levels = [module.TileLevel(identifier, scale, land, shared)
                      for identifier,scale in (('regional',8),('local',32),('detail',64))]
            output = args.output/name
            started = time.perf_counter()
            manifest = module.write_atlas_tiles(output,128,128,levels,tile_size=16)
            elapsed = time.perf_counter()-started
            reports.append({'version':name,'seconds':elapsed,'tiles':manifest['stats']['totalTiles']})
        identical = digest(args.output/'before') == digest(args.output/'after')
    report = {'source':str(args.review/'land-potential.svg'), 'domain':[1280,448,128,128],
              'sourceVertices':int(shapely.get_num_coordinates(land)),
              'features':len(features),'timings':reports,'filesByteIdentical':identical,
              'speedup':reports[0]['seconds']/reports[1]['seconds'],
              'scope':'Saved native polygons; 64 tiles at three levels; excludes full-world generation'}
    (args.output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)
    if not identical:
        raise AssertionError('Optimized output differs from reference')


if __name__ == '__main__':
    main()
