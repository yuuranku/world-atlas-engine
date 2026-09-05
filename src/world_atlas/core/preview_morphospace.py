"""Render a deterministic coverage sample of the continuous planet morphospace."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from world_atlas.inputs import (
    BATHYMETRY_PALETTE,
    LAND_PALETTE,
    default_recipe,
)
from world_atlas.core.planet_morphology import representative_morphology_seeds
from world_atlas.core.procedural_planet import generate_planet_surface, save_palette_source


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def render_morphospace_preview(
    output: str | Path,
    *,
    root_seed: int = 300378369,
    count: int = 6,
    columns: int = 3,
    tile_width: int = 640,
) -> tuple[Path, Path]:
    if columns <= 0 or count <= 0 or tile_width < 128:
        raise ValueError("preview count, columns and tile width must be positive")
    tile_height = tile_width // 2
    header_height = 54
    rows = (count + columns - 1) // columns
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + header_height)),
        (31, 42, 58),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    detail_font = _font(13)
    records: list[dict[str, object]] = []
    seeds = representative_morphology_seeds(root_seed, count)
    for ordinal, seed in enumerate(seeds):
        recipe = replace(
            default_recipe(width=tile_width, height=tile_height),
            seed=seed,
        )
        surface = generate_planet_surface(recipe)
        temporary = output_path.with_name(f".{output_path.stem}-{ordinal + 1}.png")
        save_palette_source(
            surface,
            temporary,
            land_palette=LAND_PALETTE,
            bathymetry_palette=BATHYMETRY_PALETTE,
        )
        tile = Image.open(temporary).convert("RGB")
        column = ordinal % columns
        row = ordinal // columns
        left = column * tile_width
        top = row * (tile_height + header_height)
        canvas.paste(tile, (left, top + header_height))
        draw.text(
            (left + 12, top + 7),
            str(surface.diagnostics["worldFamily"]),
            font=title_font,
            fill=(242, 245, 248),
        )
        draw.text(
            (left + 12, top + 32),
            f"seed {seed} · 连续参数采样后按成图指标命名",
            font=detail_font,
            fill=(169, 183, 200),
        )
        records.append(
            {
                "seed": seed,
                "worldFamily": surface.diagnostics["worldFamily"],
                "morphologyAxes": dict(surface.diagnostics["morphologyAxes"]),
                "classificationMetrics": dict(surface.diagnostics["classificationMetrics"]),
            }
        )
        temporary.unlink()

    canvas.save(output_path, optimize=True)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "rootSeed": root_seed,
                "selection": "maximin-sample-over-nine-continuous-geological-axes",
                "worlds": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root-seed", type=int, default=300378369)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=640)
    arguments = parser.parse_args()
    image_path, metadata_path = render_morphospace_preview(
        arguments.output,
        root_seed=arguments.root_seed,
        count=arguments.count,
        columns=arguments.columns,
        tile_width=arguments.tile_width,
    )
    print(image_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
