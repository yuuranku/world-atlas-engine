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
import numpy as np
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
    parser.add_argument('--region', type=int, nargs=4, default=[1280,448,128,128],
                        metavar=('WEST','NORTH','WIDTH','HEIGHT'))
    parser.add_argument('--contours', type=Path,
                        help='Existing physical-contours directory; no terrain is recomputed')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--tile-size', type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('coverage_decoder', root/'scripts/check_coastal_coverage.py')
    coverage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coverage)
    tree = ET.parse(args.review/'land-potential.svg')
    ns = {'s':'http://www.w3.org/2000/svg'}
    land = coverage.path_geometry(tree.find('.//s:clipPath/s:path', ns).attrib['d'])
    west, north, width, height = args.region
    land = translate(land, xoff=-west, yoff=-north)
    frame = shapely.box(-.5,-.5,width+.5,height+.5)
    land = shapely.intersection(land, frame)
    features = []
    for path in tree.findall('.//s:g/s:path', ns):
        geometry = translate(coverage.path_geometry(path.attrib['d']), xoff=-west, yoff=-north)
        geometry = shapely.intersection(geometry, frame)
        attributes = {name:value for name,value in path.attrib.items() if name not in ('d','clip-path')}
        for part in shapely.get_parts(geometry):
            if not part.is_empty:
                features.append((part, attributes,'potential','potential','theme'))
    contour_vertices = 0
    if args.contours:
        for file in sorted(args.contours.glob('level-*.npz')):
            with np.load(file) as record:
                points, offsets = record['points'], record['offsets']
                for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
                    line = shapely.LineString(points[start:stop] - [west,north])
                    line = shapely.intersection(line,frame)
                    if not line.is_empty:
                        contour_vertices += int(shapely.get_num_coordinates(line))
                        features.append((line,{'stroke':'#73583f','stroke-width':'.1',
                            'fill':'none','clip':'land','data-height-m':str(float(record['level']))},
                            'elevation-contours',None,'ink'))
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
        for name, module, workers in (('before',previous,None),('after-serial',current,1),
                                     ('after-parallel',current,args.workers)):
            shared = [module.TileFeature(geometry,attrs,layer,theme=theme,section=section)
                      for geometry,attrs,layer,theme,section in features]
            levels = [module.TileLevel(identifier, scale, land, shared)
                      for identifier,scale in (('regional',8),('local',32),('detail',64))]
            output = args.output/name
            started = time.perf_counter()
            options = {} if workers is None else {'workers':workers}
            manifest = module.write_atlas_tiles(output,width,height,levels,tile_size=args.tile_size,**options)
            elapsed = time.perf_counter()-started
            reports.append({'version':name,'seconds':elapsed,**manifest['stats']})
        reference_digest = digest(args.output/'before')
        identical = all(reference_digest == digest(args.output/name)
                        for name in ('after-serial','after-parallel'))
    report = {'source':str(args.review/'land-potential.svg'), 'domain':args.region,
              'tileSize':args.tile_size,
              'contoursSource':str(args.contours) if args.contours else None,
              'contourVertices':contour_vertices,
              'sourceVertices':int(shapely.get_num_coordinates(land)),
              'features':len(features),'timings':reports,'filesByteIdentical':identical,
              'serialSpeedup':reports[0]['seconds']/reports[1]['seconds'],
              'parallelSpeedup':reports[0]['seconds']/reports[2]['seconds'],
              'scope':'Saved native geometry at three levels; excludes full-world generation and input decoding'}
    (args.output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)
    if not identical:
        raise AssertionError('Optimized output differs from reference')


if __name__ == '__main__':
    main()
