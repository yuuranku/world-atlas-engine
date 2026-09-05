"""Command-line entry points for the standalone computation package."""
import argparse
import json
import logging
from pathlib import Path
import sys

from .api import generate_terrain, generate_world, reproduce_world, verify_world
from .runtime import doctor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic world atlas engine; no AI service required")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("doctor", help="Check Python libraries and explicit Node/Mapshaper renderer")
    check.add_argument("--mapshaper", type=Path)
    seeds = sub.add_parser("seeds", help="Sample representative geological morphology seeds, without generating maps")
    seeds.add_argument("--root-seed", type=int, required=True)
    seeds.add_argument("--count", type=int, default=6)
    terrain = sub.add_parser("terrain", help="Generate tectonics and terrain from a recipe into a new directory")
    terrain.add_argument("--recipe", type=Path, required=True)
    terrain.add_argument("--output", type=Path, required=True)
    world = sub.add_parser("world", help="Generate all downstream systems on an accepted terrain bundle")
    world.add_argument("--terrain", type=Path, required=True)
    world.add_argument("--seed", type=int, required=True)
    world.add_argument("--output", type=Path, required=True)
    world.add_argument("--exclude", type=Path, action="append", default=[])
    world.add_argument("--mapshaper", type=Path)
    replay = sub.add_parser("reproduce", help="Replay frozen inputs and compare complete semantic fingerprints")
    replay.add_argument("--inputs", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--mapshaper", type=Path)
    verify = sub.add_parser("verify", help="Verify a completed world without rerunning its simulation")
    verify.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        if args.command == "doctor":
            result = doctor(args.mapshaper)
        elif args.command == "terrain":
            result = generate_terrain(args.recipe, args.output)
        elif args.command == "world":
            result = generate_world(args.terrain, args.output, seed=args.seed, exclusions=args.exclude, mapshaper=args.mapshaper)
        elif args.command == "reproduce":
            result = reproduce_world(args.inputs, args.output, mapshaper=args.mapshaper)
        elif args.command == "verify":
            result = verify_world(args.output)
        else:
            from .core.planet_morphology import representative_morphology_seeds
            if not 1 <= args.count <= 24 or not 0 <= args.root_seed < 2**32:
                raise ValueError("count must be 1..24 and root-seed unsigned 32-bit")
            result = {"rootSeed": args.root_seed, "seeds": representative_morphology_seeds(args.root_seed, args.count),
                      "visualAcceptance": "not evaluated"}
    except (ValueError, OSError, KeyError, AssertionError) as error:
        print(f"world-atlas: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok", True) else 1
