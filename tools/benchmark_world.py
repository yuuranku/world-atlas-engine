"""Measure one cold complete world with the delivered world's frozen inputs."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
import traceback

from world_atlas import __version__, generate_terrain, generate_world
from world_atlas.timing import measure_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapshaper", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_world.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(output / "generation.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    recipe = json.loads((source / "source/provenance.json").read_text(encoding="utf-8"))["recipe"]
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    for name in ("world-settings.json", "naming-exclusions.json"):
        (output / name).write_bytes((source / name).read_bytes())
    result = {"engineVersion": __version__, "scope": "cold-complete-world",
              "frozenInputsFrom": str(source), "output": str(output),
              "reusedStages": [], "status": "running", "recipe": recipe,
              "inputSha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                              for name in ("recipe.json", "world-settings.json", "naming-exclusions.json")}}

    def save():
        (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    save()
    started = time.perf_counter()
    try:
        with measure_stage(output, "terrain"):
            generate_terrain(output / "recipe.json", output / "terrain")
        with measure_stage(output, "world"):
            record = generate_world(output / "terrain", output / "world-settings.json",
                                    output / "world", exclusions=(output / "naming-exclusions.json",),
                                    mapshaper=args.mapshaper)
    except BaseException:
        result.update(status="failed", elapsedSeconds=time.perf_counter()-started,
                      failure=traceback.format_exc())
        save()
        raise
    result.update(status="complete", elapsedSeconds=time.perf_counter()-started,
                  worldName=record["worldName"])
    result["phases"] = json.loads((output / "world/timing.json").read_text(encoding="utf-8"))
    result["renderPhases"] = json.loads((output / "world/review/timing.json").read_text(encoding="utf-8"))
    save()
    print(json.dumps({key: result[key] for key in ("status", "scope", "elapsedSeconds", "output")}))


if __name__ == "__main__":
    main()
