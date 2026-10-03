"""Compare a fixed synthetic numeric-ecology workload with a Git revision.

This is a bounded 256 by 128 comparison, not a full-world generation or a
forecast of full-world runtime. Each version evaluates three different fields,
including river orders 1, 2, and 4, a lake, and five/nine/five class cuts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

import numpy as np
import shapely
from shapely.affinity import scale


def synthetic_inputs():
    width, height = 256, 128
    yy, xx = np.mgrid[:height, :width]
    x, y = xx + .5, yy + .5
    land_surface = shapely.union_all((
        scale(shapely.Point(114, 67).buffer(1, quad_segs=96), 107, 58),
        scale(shapely.Point(236, 44).buffer(1, quad_segs=64), 15, 24),
    ))
    lake = scale(shapely.Point(169, 86).buffer(1, quad_segs=64), 7, 5)
    land_surface = shapely.difference(land_surface, lake)
    land = shapely.contains_xy(land_surface, x, y)
    t = np.linspace(0, 1, 64)
    rivers = (
        np.column_stack((32 + 167*t, 75 + 12*np.sin(1.3*np.pi*t))),
        np.column_stack((77 + 49*t, 20 + 58*t + 5*np.sin(2*np.pi*t))),
        np.column_stack((112 + 52*t, 111 - 26*t + 3*np.sin(3*np.pi*t))),
    )
    fields = []
    for index in range(3):
        background = np.clip(.36 + .16*np.sin(x/21 + index*.4)
                             + .10*np.cos(y/13 - index*.5)
                             + .035*np.sin((x+y)/7), .04, .75)
        ceiling = np.minimum(1., background + .22 + .06*np.cos(x/31-y/19))
        cuts = (np.linspace(.22, .66, 5) if index != 1
                else np.linspace(.18, .82, 9))
        fields.append((background, ceiling, cuts))
    return land_surface, lake, land, rivers, fields


def decode_paths(paths):
    polygons = []
    for points, codes in paths:
        starts = np.flatnonzero(codes == 1)
        rings = [points[first:last] for first, last in
                 zip(starts, (*starts[1:], len(points)), strict=True)]
        polygons.append(shapely.Polygon(rings[0], rings[1:]))
    return shapely.coverage_union_all(polygons)


def run_worker(source: Path, output: Path, parallel: bool):
    sys.path.insert(0, str(source))
    from world_atlas.core.continuous_ecology import (
        ContinuousEcologyField, FreshwaterCorridors)

    surface, lake, land, rivers, inputs = synthetic_inputs()
    sources = FreshwaterCorridors.from_surfaces(land.shape, rivers, (1, 2, 4), lake)
    y, x = np.where(land)

    def theme(index):
        started = time.perf_counter()
        background, ceiling, cuts = inputs[index]
        field = ContinuousEcologyField(background, ceiling, land, sources,
            supply_capacity=(.72, .81, .78)[index],
            river_radii=(1.8, 2.5, 3.6), lake_radius=3.4)
        native = field.native
        native_seconds = time.perf_counter()-started
        paths = field.band_paths(cuts, working_surface=surface)
        generated_seconds = time.perf_counter()-started
        regions = [decode_paths(path) for path in paths]
        expected = np.searchsorted(cuts, native[y, x], side='right')
        observed = np.full(len(x), -1, dtype=int)
        for owner, region in enumerate(regions):
            observed[shapely.intersects_xy(region, x+.5, y+.5)] = owner
        region_array = np.asarray(regions, dtype=object)
        return {
            'index': index, 'cutCount': len(cuts),
            'nativeSeconds': native_seconds,
            'nativeAndPathsSeconds': generated_seconds,
            'nativeSHA256': hashlib.sha256(native.tobytes()).hexdigest(),
            'nativeWitnesses': len(x),
            'nativeWitnessMismatches': int(np.count_nonzero(expected != observed)),
            'coverageValid': bool(shapely.coverage_is_valid(region_array)),
            'geometriesValid': bool(np.all(shapely.is_valid(region_array))),
            'coverageDifferenceArea': float(shapely.symmetric_difference(
                shapely.union_all(region_array), shapely.set_precision(surface, 1e-8)).area),
            'vertices': int(shapely.get_num_coordinates(region_array).sum()),
            'regionsWKB': [shapely.to_wkb(region, hex=True) for region in regions],
        }, native

    started = time.perf_counter()
    if parallel:
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(theme, range(3)))
    else:
        results = [theme(index) for index in range(3)]
    elapsed = time.perf_counter()-started
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output/'native.npz', **{f'theme{index}': value[1]
                                   for index, value in enumerate(results)})
    record = {'seconds': elapsed, 'parallelThreads': 3 if parallel else 1,
              'themes': [result[0] for result in results]}
    (output/'result.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'seconds': elapsed}), flush=True)


def compare_results(reference_dir: Path, candidate_dir: Path):
    reference = json.loads((reference_dir/'result.json').read_text(encoding='utf-8'))
    candidate = json.loads((candidate_dir/'result.json').read_text(encoding='utf-8'))
    themes = []
    with np.load(reference_dir/'native.npz') as old, np.load(candidate_dir/'native.npz') as new:
        for index, (before, after) in enumerate(zip(reference['themes'], candidate['themes'], strict=True)):
            old_regions = shapely.from_wkb(before['regionsWKB'])
            new_regions = shapely.from_wkb(after['regionsWKB'])
            differences = shapely.symmetric_difference(old_regions, new_regions)
            themes.append({
                'index': index,
                'nativeByteIdentical': np.array_equal(old[f'theme{index}'], new[f'theme{index}']),
                'nativeMaxAbsDifference': float(np.max(abs(old[f'theme{index}']-new[f'theme{index}']))),
                'regionTopologicallyEqual': bool(np.all(shapely.equals(old_regions, new_regions))),
                'regionSymmetricDifferenceArea': float(shapely.area(differences).sum()),
                'referenceVertices': before['vertices'], 'candidateVertices': after['vertices'],
                'candidateNativeWitnessMismatches': after['nativeWitnessMismatches'],
                'candidateCoverageValid': after['coverageValid'],
                'candidateGeometriesValid': after['geometriesValid'],
                'candidateCoverageDifferenceArea': after['coverageDifferenceArea'],
            })
    return themes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference-ref', default='HEAD')
    parser.add_argument('--worker-source', type=Path)
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()
    if args.worker_source:
        run_worker(args.worker_source, args.output, args.parallel)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='atlas-numeric-baseline-') as temporary:
        baseline = Path(temporary)
        archive = baseline/'source.tar'
        with archive.open('wb') as stream:
            subprocess.run(['git', 'archive', args.reference_ref, 'src/world_atlas'],
                           cwd=root, stdout=stream, check=True)
        with tarfile.open(archive) as tar:
            tar.extractall(baseline, filter='data')
        for name, source, parallel in (
            ('before', baseline/'src', False),
            ('after-serial', root/'src', False),
            ('after-parallel', root/'src', True),
        ):
            command = [sys.executable, str(Path(__file__).resolve()),
                       str(args.output/name), '--worker-source', str(source)]
            if parallel:
                command.append('--parallel')
            environment = os.environ.copy()
            environment.pop('PYTHONPATH', None)
            subprocess.run(command, cwd=root, env=environment, check=True)
    records = {}
    for name in ('before', 'after-serial', 'after-parallel'):
        record = json.loads((args.output/name/'result.json').read_text(encoding='utf-8'))
        for theme in record['themes']:
            theme.pop('regionsWKB')
        records[name] = record
    comparisons = {name: compare_results(args.output/'before', args.output/name)
                   for name in ('after-serial', 'after-parallel')}
    report = {
        'scope': 'Fixed synthetic 256x128 fields, three themes, river orders 1/2/4, one lake, 5/9/5 cuts; excludes complete-world runtime, deriving real-grid inputs, and packaging.',
        'referenceRef': args.reference_ref, 'timings': records,
        'serialSpeedup': records['before']['seconds']/records['after-serial']['seconds'],
        'parallelSpeedup': records['before']['seconds']/records['after-parallel']['seconds'],
        'parallelVersusSerial': records['after-serial']['seconds']/records['after-parallel']['seconds'],
        'comparisons': comparisons,
    }
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if any(not theme['nativeByteIdentical'] or not theme['regionTopologicallyEqual']
           or theme['candidateNativeWitnessMismatches'] != 0
           or not theme['candidateCoverageValid'] or not theme['candidateGeometriesValid']
           for themes in comparisons.values() for theme in themes):
        raise AssertionError('Numeric ecology comparison differs from reference or loses source witnesses')


if __name__ == '__main__':
    main()
