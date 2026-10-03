"""Compare road drawing on saved world geometry without generating a world."""

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import shapely

from world_atlas.core import transport_geometry


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _saved_land_surface(review):
    decoder = _load_module('transport_benchmark_svg_decoder',
        Path(__file__).resolve().parents[1]/'scripts/reencode_globe_paths.py')
    document = ET.fromstring(gzip.decompress((review/'physical-surface.svgz').read_bytes()))
    clip = next(node for node in document.iter() if node.get('id') == 'land-silhouette-clip')
    polygons = []
    for path in clip:
        rings = [np.asarray(points, dtype=float)/100_000_000
                 for points, closed in decoder.linear_subpaths(path.get('d'))]
        polygons.append(shapely.Polygon(rings[0], rings[1:]))
    return shapely.MultiPolygon(polygons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('world', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--count', type=int, default=6)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    review = args.world/'review'
    crossing_path = review/'transport-crossings.json'
    record = json.loads(crossing_path.read_text(encoding='utf-8'))
    channel = shapely.geometry.shape(record['channelGeometry'])
    river = shapely.geometry.shape(record['riverGeometry'])
    land = _saved_land_surface(review)
    surface = land.difference(channel)
    selected = []
    roads = sorted(record['preparedRoadGeometry'], key=lambda road:
        len(shapely.get_coordinates(shapely.geometry.shape(road['geometry']))), reverse=True)
    for road in roads[:args.count]:
        line = max(shapely.get_parts(shapely.geometry.shape(road['geometry'])), key=lambda part:part.length)
        selected.append((road['identifier'], np.asarray(line.coords)))
    baseline = _load_module('world_atlas.core._transport_benchmark_baseline', args.baseline_source)
    outputs, timings = {}, {}
    for label, module in (('baseline', baseline), ('optimized', transport_geometry)):
        # Keep each side's prepared state independent. The baseline renderer
        # prepared land, but never the channel-cut road surface used here.
        ground = shapely.from_wkb(shapely.to_wkb(surface))
        water = shapely.from_wkb(shapely.to_wkb(river))
        start = time.perf_counter()
        if label == 'optimized':
            shapely.prepare(ground)
            shapely.prepare(water)
        paths = []
        for identifier, points in selected:
            simplified = module._simplify(points, 'road', ground, water)
            paths.append(module._curved_transport_points(simplified, 'road', ground, water))
        timings[label] = time.perf_counter()-start
        outputs[label] = paths
        print(f'{label}: {timings[label]:.6f}s', flush=True)
    if any(not np.array_equal(old, new) for old, new in
           zip(outputs['baseline'], outputs['optimized'], strict=True)):
        raise ValueError('road optimization changed the delivered coordinates')
    result = {
        'scope': 'road simplification and corner curves; excludes generation, engineering and file loading',
        'source': str(args.world.resolve()),
        'crossingSourceSha256': hashlib.sha256(crossing_path.read_bytes()).hexdigest(),
        'landCoordinates': len(shapely.get_coordinates(land)),
        'channelCoordinates': len(shapely.get_coordinates(channel)),
        'roads': [{'identifier':identifier, 'vertices':len(points)} for identifier, points in selected],
        'elapsedSeconds': timings,
        'speedup': timings['baseline']/timings['optimized'],
        'identicalCoordinates': True,
    }
    (args.output/'timing.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
