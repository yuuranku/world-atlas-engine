"""Measure saved-world terrain queries without regenerating the world."""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import time

import numpy as np

from world_atlas import __version__
from world_atlas.core.implicit_terrain import terrain_level_curves
from world_atlas.core.implicit_terrain_bounds import field_range_gradient_bounds
from world_atlas.core.model import WorldGrid
from world_atlas.core.procedural_planet import load_surface_bundle
from world_atlas.core.terrain_refinement import terrain_from_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('world', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--query', type=float, nargs=4, required=True,
                        metavar=('WEST', 'NORTH', 'EAST', 'SOUTH'))
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    grid = WorldGrid.load(args.world/'grid')
    source = load_surface_bundle(args.world/'source/physical-fields.npz')
    field = terrain_from_source(grid, source)
    build_seconds = time.perf_counter()-started
    west, north, east, south = args.query
    rng = np.random.default_rng(20261003)
    points = rng.uniform((west, north), (east, south), (65536, 2))
    measurements = []
    arrays = {}

    def measure(name, operation):
        started = time.perf_counter()
        value = operation()
        measurements.append({'stage': name, 'seconds': time.perf_counter()-started})
        return value

    for index in range(3):
        arrays['sample'] = measure(f'paired-sampling-{index+1}',
            lambda: field.sample_points(points[:, 0], points[:, 1]))
    lower = points[:1024]
    upper = np.minimum(lower+.125, (east, south))
    bounds = measure('interval-bounds', lambda: field_range_gradient_bounds(field, lower, upper))
    for index, values in enumerate(bounds):
        arrays[f'bounds-{index}'] = values
    levels = np.unique(np.quantile(arrays['sample'], (.25, .5, .75)))
    profiler = cProfile.Profile() if args.profile else None
    if profiler is not None:
        profiler.enable()
    curves = measure('refined-contours', lambda: terrain_level_curves(
        field, levels, query_bounds=np.asarray((args.query,))))
    if profiler is not None:
        profiler.disable()
        profiler.dump_stats(args.output/'contours.prof')
        with (args.output/'profile.txt').open('w', encoding='utf-8') as stream:
            pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(24)
    for level, paths in enumerate(curves):
        for index, points in enumerate(paths):
            arrays[f'curve-{level}-{index}'] = points
    np.savez(args.output/'results.npz', **arrays)
    report = {
        'engineVersion': __version__,
        'world': str(args.world), 'query': args.query, 'levels': levels.tolist(),
        'nativeShape': list(grid.shape), 'buildSeconds': build_seconds,
        'measurements': measurements, 'profiledContours': args.profile,
        'curvePoints': sum(len(points) for paths in curves for points in paths),
        'horizontalCacheBytes': sum(
            item.__dict__['horizontal_coefficients'].nbytes
            for item in (field.base, field._height.pchip)
            if 'horizontal_coefficients' in item.__dict__),
        'digests': {name: hashlib.sha256(values.tobytes()).hexdigest()
                    for name, values in arrays.items()},
        'scope': 'Saved-world paired samples, interval certificates and local refined contours; excludes full-world generation',
    }
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'digests'}), flush=True)


if __name__ == '__main__':
    main()
