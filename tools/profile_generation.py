"""Measure real generation stages on a saved world, including result identity."""
import argparse
import cProfile
import json
import logging
from pathlib import Path
import pstats
import time

from world_atlas.core.model import WorldGrid
from world_atlas.core.render import _society_generation_request
from world_atlas.core.society.pipeline import derive_society_layers
from world_atlas.core.society.storage import load_society, save_society
from world_atlas.core.society.transport import derive_transport
from world_atlas.core.society.politics import derive_politics
from world_atlas.core.thematic import derive_thematic_layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", choices=("society", "transport", "politics"), default="society")
    parser.add_argument("--no-profile", action="store_true", help="measure normal wall time without profiler overhead")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    grid = WorldGrid.load(args.world / "grid")
    thematic = derive_thematic_layers(grid)
    source, request = _society_generation_request(grid)
    previous = load_society(args.world / "society", expected_grid_digest=grid.content_digest())
    profile = cProfile.Profile()
    started = time.perf_counter()
    run = (lambda function, *args, **kwargs: function(*args, **kwargs)) if args.no_profile else profile.runcall
    try:
        if args.stage == "society":
            result = run(derive_society_layers, grid, thematic, source, **request)
        elif args.stage == "politics":
            from world_atlas.core.society.names import load_name_lexicon
            from world_atlas.core.society.model import NameLexicon
            result = run(derive_politics, grid, thematic, previous.population,
                         previous.settlements, previous.cultures, previous.transport,
                         source if isinstance(source, NameLexicon) else load_name_lexicon(source), state_count=request.get("state_count"),
                         frontier_target_share=request.get("frontier_target_share", .35))
        else:
            result = run(derive_transport, grid, thematic, previous.population, previous.settlements)
    finally:
        if not args.no_profile:
            profile.create_stats()
            if profile.stats:
                profile.dump_stats(str(args.output / "profile.pstats"))
                with (args.output / "profile.txt").open("w", encoding="utf-8") as stream:
                    pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(60)
    elapsed = time.perf_counter() - started
    if args.stage == "society":
        save_society(result, args.output / "society", grid_digest=grid.content_digest())
    report = {"stage": args.stage, "elapsedSeconds": elapsed, "gridDigest": grid.content_digest()}
    if args.stage == "society":
        from world_atlas.checks import array_digest
        expected = args.world / "society"
        actual = args.output / "society"
        report["arraysIdentical"] = array_digest(expected / "society.npz") == array_digest(actual / "society.npz")
        report["entitiesIdentical"] = json.loads((expected / "society.json").read_text(encoding="utf-8")) == json.loads((actual / "society.json").read_text(encoding="utf-8"))
    (args.output / "timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
