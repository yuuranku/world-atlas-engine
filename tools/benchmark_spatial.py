"""Compare exact full-resolution spatial outputs with a published git revision."""
import argparse
import ast
from collections import deque
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from world_atlas.core.model import WorldGrid
from world_atlas.core.society import population, spatial


def reference_function(root, revision, module, name):
    source = subprocess.run(["git", "show", f"{revision}:src/world_atlas/core/society/{module}.py"],
        cwd=root, check=True, capture_output=True, encoding="utf-8").stdout
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace = dict(np=np, math=math, deque=deque)
    exec(compile(tree, f"{revision}/{module}.py", "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", default="v1.1.0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    grid = WorldGrid.load(args.world / "grid")
    masks = grid.water == 0
    site_mask = np.zeros(grid.shape, bool)
    settlements = json.loads((args.world / "society/society.json").read_text(encoding="utf-8"))["settlements"]
    for city in settlements:
        site_mask[city["row"], city["column"]] = True
    records = []
    for module, name, values, keywords in (
        ("spatial", "connected_components", masks, {}),
        ("population", "_proximity", site_mask, {"radius":58}),
    ):
        before = reference_function(root, args.revision, module, name)
        after = getattr(spatial if module == "spatial" else population, name)
        begin = time.perf_counter()
        original = before(values, **keywords)
        old_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        optimized = after(values, **keywords)
        new_seconds = time.perf_counter() - begin
        if name == "connected_components":
            np.testing.assert_array_equal(original[0], optimized[0])
            assert original[1] == optimized[1]
        else:
            np.testing.assert_array_equal(original, optimized)
        records.append(dict(operation=name, beforeSeconds=old_seconds, afterSeconds=new_seconds,
                            speedup=old_seconds/new_seconds, identical=True))
    report = dict(shape=grid.shape, referenceRevision=args.revision, records=records)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
