"""Small map-style previews drawn from the saved world's actual fields."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image


def write_theme_previews(
    output: Path,
    *,
    water: np.ndarray,
    values: Mapping[str, np.ndarray],
    palettes: Mapping[str, Sequence[tuple[str, str]]],
) -> dict[str, str]:
    base = np.asarray(Image.open(output / "terrain.png").convert("RGB"))
    if base.shape[:2] != water.shape or set(values) != set(palettes):
        raise ValueError("map previews must share the saved world and theme inventory")
    directory = output / "map-previews"
    directory.mkdir(exist_ok=True)
    size = (256, 128)
    Image.fromarray(base).resize(size, Image.Resampling.LANCZOS).save(directory / "none.png")
    paths = {"none": "map-previews/none.png"}
    land = water == 0
    for theme, field in values.items():
        if field.shape != water.shape or field.dtype.kind not in "iu":
            raise ValueError("map preview categories must be native integer fields")
        colors = np.asarray([
            tuple(int(color[index:index+2], 16) for index in (1, 3, 5))
            for _, color in palettes[theme]
        ], dtype=np.uint8)
        if np.any(field[land] < 0) or np.any(field[land] >= len(colors)):
            raise ValueError("map preview categories lie outside their actual palette")
        pixels = base.copy()
        pixels[land] = np.rint(.88 * colors[field[land]] + .12 * base[land]).astype(np.uint8)
        filename = f"{theme}.png"
        Image.fromarray(pixels).resize(size, Image.Resampling.LANCZOS).save(directory / filename)
        paths[theme] = f"map-previews/{filename}"
    return paths
