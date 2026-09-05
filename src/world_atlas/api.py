"""Small end-to-end interfaces; physical computation never depends on the skill."""
from dataclasses import fields
import json
import math
import os
from pathlib import Path

from .core.procedural_planet import PlanetRecipe
from .inputs import prepare_world_inputs
from .runtime import require_renderer


def load_recipe(path: str | Path) -> PlanetRecipe:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    names = {field.name for field in fields(PlanetRecipe)}
    if not isinstance(record, dict) or set(record) != names:
        raise ValueError(f"recipe must contain exactly: {', '.join(sorted(names))}")
    integers = {"seed", "width", "height", "plate_count", "continent_count"}
    for name, value in record.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"recipe.{name} must be a finite number")
        if name in integers and not isinstance(value, int):
            raise ValueError(f"recipe.{name} must be an integer")
    if not 0 <= record["seed"] < 2**32:
        raise ValueError("recipe.seed must be an unsigned 32-bit integer")
    return PlanetRecipe(**record)


def _fresh_output(output: str | Path) -> Path:
    path = Path(output).resolve()
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    return path


def generate_terrain(recipe_path: str | Path, output: str | Path) -> dict:
    from .core.terrain_review import publish_terrain_review
    from .checks import array_digest
    root = _fresh_output(output)
    recipe = load_recipe(recipe_path)
    prepared = prepare_world_inputs(recipe, source_directory=root / "source", config_path=root / "worldgen.json", output_directory=root)
    result = publish_terrain_review(root / "source", root / "review", version=f"seed-{recipe.seed}")
    result["fieldArrayDigest"] = array_digest(prepared.bundle_path)
    result["mode"] = "new-terrain"
    (root / "terrain-run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def generate_world(terrain: str | Path, output: str | Path, *, seed: int, exclusions=(), mapshaper=None) -> dict:
    from .rebuild import build_accepted_world
    root = _fresh_output(output)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("world seed must be an unsigned 32-bit integer")
    _node, entry = require_renderer(mapshaper)
    source = Path(terrain).resolve()
    previous = os.environ.get("WORLD_ATLAS_MAPSHAPER")
    os.environ["WORLD_ATLAS_MAPSHAPER"] = str(entry)
    try:
        result = build_accepted_world(source / "worldgen.json", source / "source/provenance.json", root,
                                      [Path(path) for path in exclusions], seed=seed)
        from .checks import semantic_checks
        checks = semantic_checks(root)
        (root / "review/release-checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        if previous is None:
            os.environ.pop("WORLD_ATLAS_MAPSHAPER", None)
        else:
            os.environ["WORLD_ATLAS_MAPSHAPER"] = previous


def reproduce_world(inputs: str | Path, output: str | Path, *, mapshaper=None) -> dict:
    from .checks import semantic_checks
    source = Path(inputs).resolve()
    record = json.loads((source / "regeneration.json").read_text(encoding="utf-8"))
    expected = json.loads((source / "release-checks.json").read_text(encoding="utf-8"))
    result = generate_world(source, output, seed=record["namingSeed"], exclusions=[source / "naming-exclusions.json"], mapshaper=mapshaper)
    actual = semantic_checks(Path(output))
    comparison = {"ok": actual == expected, "expected": expected, "observed": actual}
    (Path(output) / "review/replay-check.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    if actual != expected:
        raise ValueError(f"semantic replay mismatch: {comparison}")
    return {**result, "replay": comparison}


def verify_world(output: str | Path) -> dict:
    from .acceptance import verify_release
    return verify_release(Path(output).resolve())
