"""Time one fresh render from saved physical and human simulation data.

This benchmark never changes the delivered world or rebinds a contour cache.
It deliberately measures rendering, not cold physical/human generation.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
import traceback

import world_atlas
from world_atlas import __version__
from world_atlas.core.model import WorldGrid
from world_atlas.core.procedural_planet import load_surface_bundle
from world_atlas.core.render import render_review
from world_atlas.core.society.storage import load_society
from world_atlas.settings import load_world_settings
from world_atlas.timing import measure_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.world.resolve(), args.output.resolve()
    output.mkdir(exist_ok=False, parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(output / "render.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    grid = WorldGrid.load(source / "grid")
    society = load_society(source / "society", expected_grid_digest=grid.content_digest())
    physical_source = load_surface_bundle(source / "source/physical-fields.npz")
    settings = load_world_settings(source / "world-settings.json")
    package = Path(world_atlas.__file__).resolve().parent
    modules = {str(path.relative_to(package.parent)):
               hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(package.rglob("*.py"))}
    result = {"engineVersion": __version__, "sourceWorld": str(source),
              "output": str(output), "scope": "fresh-render-from-saved-simulation",
              "reusedStages": ["terrain", "climate-hydrology", "society"],
              "reusedContourGraphs": False, "shape": list(grid.shape),
              "gridDigest": grid.content_digest(), "sourceModulesSha256": modules,
              "status": "running"}

    def save():
        (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    save()
    started = time.perf_counter()
    try:
        with measure_stage(output, "render"):
            render_review(grid, output / "review", society=society,
                          physical_source=physical_source,
                          travel_capabilities=settings.travel_capabilities)
    except BaseException:
        result.update(status="failed", elapsedSeconds=time.perf_counter()-started,
                      failure=traceback.format_exc())
        save()
        raise
    result.update(status="complete", elapsedSeconds=time.perf_counter()-started)
    result["phases"] = json.loads((output / "review/timing.json").read_text(encoding="utf-8"))
    save()
    print(json.dumps({key: result[key] for key in ("status", "scope", "elapsedSeconds", "output")}))


if __name__ == "__main__":
    main()
