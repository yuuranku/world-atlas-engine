"""Generate each independently selected polar layout and inspect physical truth."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from world_atlas.api import generate_terrain, load_recipe
from world_atlas.core.baseline import load_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recipe", type=Path, default=Path(__file__).resolve().parents[1]/"examples/terrain.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = replace(load_recipe(args.recipe), width=144, height=72, plate_count=8, continent_count=5)
    records = []
    for north, south in ((False,False),(False,True),(True,False),(True,True)):
        name = f"north-{int(north)}-south-{int(south)}"
        recipe = replace(base, north_polar_continent=north, south_polar_continent=south)
        recipe_path = args.output / (name + ".json")
        recipe_path.write_text(json.dumps(asdict(recipe)), encoding="utf-8")
        root = args.output / name
        generate_terrain(recipe_path, root)
        with np.load(root / "source/physical-fields.npz") as arrays:
            land = arrays["land_mask"]
            assert np.all(land[0] == north), name
            assert np.all(land[-1] == south), name
            weights = np.cos(np.radians(90.-(np.arange(land.shape[0])+.5)*180./land.shape[0]))[:,None]
            share = float(np.sum(land * weights)/(weights.sum()*land.shape[1]))
            assert abs(share-recipe.land_fraction) < .005, (name, share)
        grid = load_baseline(root / "worldgen.json")
        assert np.array_equal(land, grid.water == 0), name
        assert (grid.metadata["polarRegions"]["arctic"]["surface"] == "continental-land") == north
        assert (grid.metadata["polarRegions"]["antarctic"]["surface"] == "continental-land") == south
        records.append(dict(name=name, areaShare=share, physicalGridAgrees=True, review=str(root/"review/index.html")))
        print(json.dumps(records[-1]), flush=True)
    (args.output/"checks.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
