"""Measure source curves on small saved-world regions, never re-render a world."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import time
from contextlib import nullcontext

import numpy as np

from world_atlas.core.model import WorldGrid
from world_atlas.core.procedural_planet import load_surface_bundle
from world_atlas.core.terrain_refinement import terrain_from_source
from world_atlas.core.implicit_terrain import terrain_level_curves
from world_atlas.core.cartographic_contours import cartographic_level_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bounds", type=float, nargs=4, default=(1650., 440., 1666., 456.))
    parser.add_argument("--levels", type=float, nargs="*", default=(700., 950., 1200., 1500.))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--separate", action="store_true")
    parser.add_argument("--cartographic", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    grid = WorldGrid.load(Path(args.world) / "grid")
    raw = load_surface_bundle(Path(args.world) / "source/physical-fields.npz")
    field = terrain_from_source(grid, raw)
    construction = time.perf_counter() - started
    profile = cProfile.Profile()
    started = time.perf_counter()
    curves=cartographic_level_curves if args.cartographic else terrain_level_curves
    with profile if args.profile else nullcontext():
        paths = ([curves(field, [level], query_bounds=args.bounds)[0] for level in args.levels]
                 if args.separate else curves(field, args.levels, query_bounds=args.bounds))
    duration = time.perf_counter() - started
    stream = io.StringIO()
    if args.profile:
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(35)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for index, graphs in enumerate(paths):
        arrays[f"points_{index}"] = np.concatenate(graphs) if graphs else np.empty((0, 2))
        arrays[f"offsets_{index}"] = np.r_[0, np.cumsum([len(p) for p in graphs])]
    np.savez(output.with_suffix(".npz"), **arrays)
    result = {"world": args.world, "bounds": args.bounds, "levels": args.levels,
              "constructionSeconds": construction, "curveSeconds": duration,
              "profile":args.profile,"separateLevels":args.separate,
              "cartographic":args.cartographic,
              "paths": [len(p) for p in paths], "vertices": [sum(map(len, p)) for p in paths]}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".profile.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(json.dumps(result))
    print(stream.getvalue())


if __name__ == "__main__":
    main()
