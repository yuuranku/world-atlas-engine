"""Render a compact, review-oriented view directly from a WorldGrid."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import colorsys
import html
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import contourpy
import numpy as np
from PIL import Image, ImageDraw
import shapely

from .model import (
    WorldGrid,
    _atomic_replace_files,
    _file_identity,
    _temporary_file,
    _unlink_owned_file,
)
from .polar import polar_continent_mask
from .thematic import (
    BIOME_BOREAL_FOREST,
    BIOME_COUNT,
    BIOME_HUMID_SUBTROPICAL,
    BIOME_SAVANNA_DRY_GRASSLAND,
    BIOME_TEMPERATE_GRASSLAND,
    BIOME_TEMPERATE_MIXED_FOREST,
    BIOME_TROPICAL_RAINFOREST,
    BIOME_TROPICAL_SEASONAL_FOREST,
    BIOME_TUNDRA_ALPINE,
    ThematicLayers,
    derive_thematic_layers,
)
from .society import (
    SocietyLayers,
    boundary_alignment,
    derive_society_layers,
    extreme_frontier_environment,
    longest_unjustified_straight_run,
    ordinary_frontier_components,
    write_society_files,
)
from .society.politics import _coarse_route_affinity, _state_transition_penalties
from .society.provinces import _province_transition_penalties
from .society.spatial import society_domain_mask
from .society.transport import _path_cells, road_network_fields
from .society.fictional_names import procedural_name_lexicon
from .society.model import NameLexicon
from .tectonic_review import (
    TectonicReview,
    derive_tectonic_review,
    render_tectonic_svg,
)


_ELEVATION_PALETTE = np.asarray(
    (
        (226, 220, 188),
        (218, 211, 178),
        (209, 202, 167),
        (201, 194, 158),
        (192, 186, 148),
        (183, 179, 141),
        (174, 172, 135),
        (165, 165, 130),
        (156, 159, 125),
        (147, 153, 122),
        (138, 147, 118),
        (130, 141, 115),
        (122, 135, 114),
        (115, 129, 112),
        (108, 123, 110),
        (101, 117, 108),
    ),
    dtype=np.uint8,
)
_WATER_PALETTE = np.asarray(
    (
        (159, 184, 231),
        (153, 179, 228),
        (145, 173, 225),
        (153, 175, 209),
        (140, 165, 203),
        (124, 152, 195),
        (107, 139, 188),
        (103, 135, 184),
    ),
    dtype=np.uint8,
)
_SHALLOW_WATER_COLOR = "#9fb8e7"
_COAST_COLOR = "#314a70"
_CONTOUR_COLOR = "#76694f"
_RIVER_COLOR = "#3f83bd"
_SNOW_COLOR = "#ffffff"
_POLAR_LAND_COLOR = "#ecefed"
_SEA_ICE_COLOR = "#e9f3f4"
_SEA_ICE_EDGE_COLOR = "#8fb9c8"
_SNOW_TERRAIN_RGB = np.asarray((255, 255, 255), dtype=np.uint8)
_POLAR_LAND_RGB = np.asarray((236, 239, 237), dtype=np.uint8)
_SEASON_IDS = ("vernal", "june", "autumnal", "december")
_SEASON_LABELS = (
    "北春／南秋",
    "北夏／南冬",
    "北秋／南春",
    "北冬／南夏",
)
_PRECIPITATION_PALETTE = np.asarray(
    (
        (206, 164, 106),
        (222, 203, 128),
        (170, 205, 150),
        (101, 183, 170),
        (72, 142, 188),
        (47, 92, 158),
    ),
    dtype=np.uint8,
)
_CLIMATE_ZONES = (
    ("冰原／永久积雪", "#f7faf7"),
    ("苔原／高寒", "#d8dfd1"),
    ("亚寒带", "#819c78"),
    ("温带海洋性", "#8fc7b1"),
    ("温带大陆性", "#b5c97d"),
    ("地中海型", "#d4b66a"),
    ("湿润亚热带", "#70b58b"),
    ("热带雨林", "#35765a"),
    ("热带季风／草原", "#aabd63"),
    ("半干旱草原", "#c9a45f"),
    ("荒漠", "#e0c28b"),
)
_BIOME_ZONES = (
    ("冰原／永久积雪", "#f4f5ed"),
    ("苔原／高寒草甸", "#ccd6bf"),
    ("亚寒带针叶林", "#708f73"),
    ("温带混交林", "#91b883"),
    ("温带草原", "#b8c77d"),
    ("地中海灌丛", "#c9ad72"),
    ("湿润亚热带林", "#70aa7c"),
    ("热带雨林", "#39745b"),
    ("热带季雨林", "#68a171"),
    ("稀树草原／干草原", "#c6b66c"),
    ("荒漠／旱生灌丛", "#d7be8b"),
    ("极地冰原／永冻荒漠", "#eef1ef"),
    ("极地沿岸苔原", "#d9ddd7"),
)
_WATERSHED_ZONES = (
    ("次级小流域", "#d8d5c4"),
    ("第一大流域", "#77a5a1"),
    ("第二大流域", "#9bad74"),
    ("第三大流域", "#c1a66f"),
    ("第四大流域", "#839db7"),
    ("第五大流域", "#b78573"),
    ("第六大流域", "#7eaa87"),
    ("第七大流域", "#aaa36f"),
    ("第八大流域", "#8c8eb1"),
    ("第九大流域", "#c09269"),
    ("第十大流域", "#6f9eab"),
    ("第十一大流域", "#93aa78"),
    ("第十二大流域", "#b69482"),
    ("第十三大流域", "#8197a8"),
    ("第十四大流域", "#a8a067"),
    ("第十五大流域", "#739a86"),
    ("第十六大流域", "#aa8678"),
    ("第十七大流域", "#879f75"),
    ("第十八大流域", "#8990a7"),
    ("第十九大流域", "#bd9d67"),
    ("第二十大流域", "#729aa4"),
    ("第二十一大流域", "#9aab7d"),
    ("第二十二大流域", "#b48770"),
    ("第二十三大流域", "#7e9b91"),
    ("第二十四大流域", "#9b9570"),
)
_LAND_POTENTIAL_ZONES = (
    ("受限地区", "#d7d3c1"),
    ("较低潜力", "#c9bf91"),
    ("一般潜力", "#b7c486"),
    ("较高潜力", "#8eb58f"),
    ("高潜力", "#629b7e"),
    ("核心宜居带", "#3f7463"),
)
_POPULATION_ZONES = (
    ("无定居人口", "#ded9c7"),
    ("极低密度", "#d7d0a4"),
    ("低密度", "#c9c68c"),
    ("中低密度", "#aebb7e"),
    ("中等密度", "#83a978"),
    ("高密度", "#548f71"),
    ("核心人口带", "#2f655b"),
)
# The current 1,812x906 review is a few thousand paths and a compact sub-5 MiB
# HTML bundle. Keep generous fixed headroom while bounding pathological
# fragmentation from an unexpected source.
_MAX_SVG_PATHS = 10_000
# Hierarchical ocean islands and closed-sea bathymetry add legitimate physical
# contours while path and byte budgets still cap pathological fragmentation.
_MAX_SVG_POINTS = 1_200_000
# Native-resolution administrative and cultural geometry is deliberately kept
# in the review document.  Sixteen MiB still catches accidental path explosions
# without forcing the atlas back through a block-reduction stage.
_MAX_HTML_BYTES = 16 * 1024 * 1024
_PARTITION_SIMPLIFICATION_TOLERANCE = 1.10
_MAX_CLIMATE_SVG_BYTES = 4 * 1024 * 1024
_MAX_THEMATIC_SVG_BYTES = 4 * 1024 * 1024
_MAX_THEMATIC_SVG_TOTAL_BYTES = 24 * 1024 * 1024


class WorldGridRenderError(ValueError):
    """Raised when review rendering would exceed a fixed safety boundary."""


def _society_generation_request(
    grid: WorldGrid,
) -> tuple[str | Path | NameLexicon, dict[str, int]]:
    """Resolve the society profile carried by this world's immutable metadata."""

    raw = grid.metadata.get("societyGeneration")
    if not isinstance(raw, Mapping):
        raise WorldGridRenderError("world_atlas requires an explicit procedural societyGeneration profile")
    profile = str(raw.get("namingProfile", ""))
    if profile == "procedural":
        seed = raw.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise WorldGridRenderError(
                "procedural societyGeneration.seed must be an integer"
            )
        lexicon = procedural_name_lexicon(seed)
    else:
        raise WorldGridRenderError(
            f"unknown society naming profile: {profile}"
        )

    options: dict[str, int] = {}
    keys = {
        "settlementCount": "settlement_count",
        "civilizationCount": "civilization_count",
        "minimumCivilizationCount": "minimum_civilization_count",
        "languageCount": "language_count",
        "religionCount": "religion_count",
        "stateCount": "state_count",
        "populationMin": "population_min",
        "populationMax": "population_max",
    }
    for metadata_key, argument_key in keys.items():
        value = raw.get(metadata_key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise WorldGridRenderError(
                f"societyGeneration.{metadata_key} must be a positive integer"
            )
        options[argument_key] = value
    return lexicon, options


def _uses_legacy_society_names(grid: WorldGrid) -> bool:
    raw = grid.metadata.get("societyGeneration")
    return not isinstance(raw, Mapping) or str(raw.get("namingProfile", "legacy")) == "legacy"


def _format_population_range(population_min: int, population_max: int) -> str:
    def in_hundred_millions(value: int) -> str:
        return f"{value / 100_000_000:g}"

    return (
        f"{in_hundred_millions(population_min)}–"
        f"{in_hundred_millions(population_max)} 亿"
    )


def _is_reparse_point(path: Path) -> bool:
    """Return whether *path* is a symlink, junction, or Windows reparse point."""

    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = os.lstat(path).st_file_attributes
    except (AttributeError, FileNotFoundError):
        return False
    except OSError:
        return True
    return bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT


def _reject_reparse_paths(output_dir: Path) -> None:
    """Reject output paths that could redirect writes outside the workspace."""

    absolute = Path(os.path.abspath(output_dir))
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate) and _is_reparse_point(candidate):
            raise WorldGridRenderError(
                f"render output path contains a reparse point: {candidate}"
            )

    if not absolute.is_dir():
        return
    for root, directories, files in os.walk(
        absolute, topdown=True, followlinks=False
    ):
        for name in (*directories, *files):
            candidate = Path(root) / name
            if _is_reparse_point(candidate):
                raise WorldGridRenderError(
                    f"render output path contains a reparse point: {candidate}"
                )


def _validate_svg_budget(
    layers: Sequence[Sequence[np.ndarray]],
    *,
    path_layers: Sequence[Sequence[np.ndarray]] | None = None,
) -> None:
    """Check path and point counts before serializing any SVG strings."""

    counted_path_layers = layers if path_layers is None else path_layers
    path_count = sum(len(paths) for paths in counted_path_layers)
    if path_count > _MAX_SVG_PATHS:
        raise WorldGridRenderError(
            f"SVG path budget exceeded: {path_count} paths > {_MAX_SVG_PATHS}"
        )

    point_count = sum(
        int(path.shape[0])
        for paths in layers
        for path in paths
    )
    if point_count > _MAX_SVG_POINTS:
        raise WorldGridRenderError(
            f"SVG point budget exceeded: {point_count} points > {_MAX_SVG_POINTS}"
        )


def _rgb_hex(color: np.ndarray | Sequence[int]) -> str:
    return "#" + "".join(f"{int(channel):02x}" for channel in color)


def _metadata_classification(grid: WorldGrid) -> Mapping[str, Any]:
    metadata = grid.metadata
    if not isinstance(metadata, Mapping):
        return {}
    classification = metadata.get("classification")
    if classification is None:
        return {}
    if not isinstance(classification, Mapping):
        raise TypeError("WorldGrid metadata classification must be a mapping")
    return classification


def _parse_hex_color(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None:
        raise ValueError("palette colors must be #RRGGBB strings")
    return tuple(int(value[offset : offset + 2], 16) for offset in (1, 3, 5))


def _metadata_palette(
    grid: WorldGrid,
    key: str,
) -> np.ndarray | None:
    classification = _metadata_classification(grid)
    if key not in classification:
        return None
    record = classification[key]
    if isinstance(record, Mapping):
        if "colors" not in record:
            raise ValueError(f"classification.{key}.colors is required")
        record = record["colors"]
    if isinstance(record, (str, bytes)) or not isinstance(record, Sequence):
        raise TypeError(f"classification.{key} must be a color sequence")
    colors = tuple(_parse_hex_color(color) for color in record)
    if len(colors) < 2:
        raise ValueError(f"classification.{key} needs at least two colors")
    return np.asarray(colors, dtype=np.uint8)


def _metadata_level_count(
    grid: WorldGrid,
    key: str,
) -> int:
    metadata = grid.metadata
    record = metadata.get(key)
    if not isinstance(record, Mapping):
        raise TypeError(f"metadata.{key} must be a mapping")
    levels = record.get("levels")
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1:
        raise ValueError(f"metadata.{key}.levels must be a positive integer")
    return levels


def _resample_palette(anchors: np.ndarray, level_count: int) -> np.ndarray:
    level_count = max(1, int(level_count))
    if anchors.shape[0] == level_count:
        return anchors
    positions = np.linspace(0, anchors.shape[0] - 1, level_count)
    lower = np.floor(positions).astype(np.intp)
    upper = np.ceil(positions).astype(np.intp)
    fraction = positions - lower
    palette = (
        anchors[lower] * (1.0 - fraction[:, None])
        + anchors[upper] * fraction[:, None]
    )
    return np.rint(palette).astype(np.uint8)


def _elevation_palette_for(grid: WorldGrid) -> tuple[np.ndarray, int]:
    level_count = _metadata_level_count(grid, "elevation")
    anchors = _metadata_palette(grid, "landPalette")
    if anchors is None:
        anchors = _ELEVATION_PALETTE
    return _resample_palette(anchors, level_count), level_count


def _bathymetry_palette_for(grid: WorldGrid) -> tuple[np.ndarray, int]:
    level_count = _metadata_level_count(grid, "bathymetry")
    anchors = _metadata_palette(grid, "bathymetryPalette")
    if anchors is None:
        anchors = _WATER_PALETTE
    return _resample_palette(anchors, level_count), level_count


def _elevation_band_count(grid: WorldGrid) -> int:
    return _elevation_palette_for(grid)[1]


def _bathymetry_band_count(grid: WorldGrid) -> int:
    return _bathymetry_palette_for(grid)[1]


def _shallow_ocean_ordinal(grid: WorldGrid, palette_size: int) -> int:
    """Return the shallowest valid maritime-water ordinal in this grid."""

    valid_ocean = np.isin(grid.water, (1, 3)) & (grid.bathymetry_band >= 0)
    if not np.any(valid_ocean):
        return 0
    ordinal = int(grid.bathymetry_band[valid_ocean].min())
    if not 0 <= ordinal < palette_size:
        raise ValueError("ocean bathymetry_band is outside the renderer palette")
    return ordinal


def _review_dimensions(grid: WorldGrid) -> tuple[int, int, int, int]:
    """Return display width/height and integer authored-to-grid scale factors."""

    grid_width, grid_height = grid.shape[1], grid.shape[0]
    metadata_grid = grid.metadata.get("grid")
    if metadata_grid is None:
        return grid_width, grid_height, 1, 1
    if not isinstance(metadata_grid, Mapping):
        raise WorldGridRenderError("WorldGrid metadata grid must be a mapping")
    try:
        board_width = metadata_grid["boardWidthPx"]
        board_height = metadata_grid["boardHeightPx"]
    except KeyError as error:
        raise WorldGridRenderError(
            "WorldGrid metadata grid must declare board dimensions"
        ) from error
    if (
        isinstance(board_width, bool)
        or not isinstance(board_width, int)
        or isinstance(board_height, bool)
        or not isinstance(board_height, int)
        or board_width < grid_width
        or board_height < grid_height
        or board_width % grid_width
        or board_height % grid_height
    ):
        raise WorldGridRenderError(
            "WorldGrid board dimensions must be integer multiples of the grid"
        )
    scale_x = board_width // grid_width
    scale_y = board_height // grid_height
    return board_width, board_height, scale_x, scale_y


def _terrain_pixels(grid: WorldGrid) -> np.ndarray:
    """Map canonical bands to opaque RGB pixels at authored board resolution."""

    elevation_palette, _ = _elevation_palette_for(grid)
    water_palette, _ = _bathymetry_palette_for(grid)
    shallow_water_color = water_palette[_shallow_ocean_ordinal(grid, len(water_palette))]
    pixels = np.empty((*grid.shape, 3), dtype=np.uint8)
    land = grid.water == 0
    maritime_water = np.isin(grid.water, (1, 3))
    lake = grid.water == 2
    pixels[land] = elevation_palette[grid.elevation_band[land]]
    pixels[maritime_water] = water_palette[grid.bathymetry_band[maritime_water]]
    pixels[lake] = shallow_water_color
    pixels[polar_continent_mask(grid)] = _POLAR_LAND_RGB
    pixels[grid.snow] = _SNOW_TERRAIN_RGB
    display_width, display_height, scale_x, scale_y = _review_dimensions(grid)
    if (display_width, display_height) == (grid.shape[1], grid.shape[0]):
        return pixels
    return np.repeat(np.repeat(pixels, scale_y, axis=0), scale_x, axis=1)


def _polar_inset_data_url(
    grid: WorldGrid,
    *,
    north: bool,
    size: int = 224,
) -> str:
    """Render one Lambert azimuthal equal-area polar overview as an inline PNG."""

    if size < 64:
        raise ValueError("polar inset size must be at least 64 pixels")
    planet = grid.metadata.get("planet", {})
    axial_tilt = float(planet.get("axialTiltDegrees", 23.44))
    maximum_colatitude = np.deg2rad(float(np.clip(axial_tilt, 1.0, 45.0)))
    coordinate = (
        np.arange(size, dtype=np.float64) + 0.5 - size / 2.0
    ) / (size / 2.0 - 3.0)
    x, y = np.meshgrid(coordinate, coordinate)
    normalized_radius = np.hypot(x, y)
    valid = normalized_radius <= 1.0
    colatitude = 2.0 * np.arcsin(
        np.clip(
            normalized_radius * np.sin(maximum_colatitude / 2.0),
            0.0,
            1.0,
        )
    )
    latitude = (
        90.0 - np.rad2deg(colatitude)
        if north
        else -90.0 + np.rad2deg(colatitude)
    )
    longitude = np.rad2deg(np.arctan2(x, -y))
    row = np.clip(
        np.floor((90.0 - latitude) / 180.0 * grid.shape[0]).astype(np.intp),
        0,
        grid.shape[0] - 1,
    )
    column = np.mod(
        np.floor((longitude + 180.0) / 360.0 * grid.shape[1]).astype(np.intp),
        grid.shape[1],
    )
    water = grid.water[row, column]
    elevation = grid.elevation[row, column]
    snow = grid.snow[row, column]
    sea_ice = grid.sea_ice[row, column]
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[valid] = (105, 137, 186, 255)
    rgba[valid & sea_ice] = (213, 228, 238, 255)
    land = valid & (water == 0)
    land_shade = np.clip(241.0 - 24.0 * elevation, 214.0, 241.0).astype(np.uint8)
    rgba[land, 0] = land_shade[land]
    rgba[land, 1] = land_shade[land]
    rgba[land, 2] = np.minimum(255, land_shade[land].astype(np.int16) + 2).astype(
        np.uint8
    )
    rgba[land, 3] = 255
    rgba[land & snow] = (247, 248, 245, 255)
    coast = land & ~(
        np.roll(land, 1, axis=0)
        & np.roll(land, -1, axis=0)
        & np.roll(land, 1, axis=1)
        & np.roll(land, -1, axis=1)
    )
    rgba[coast] = (49, 83, 112, 255)
    image = Image.fromarray(rgba, mode="RGBA")
    ImageDraw.Draw(image).ellipse(
        (2, 2, size - 3, size - 3),
        outline=(71, 92, 117, 230),
        width=2,
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _seasonal_precipitation_pixels(grid: WorldGrid, season_index: int) -> np.ndarray:
    """Return one transparent RGBA thematic overlay on the shared uint8 scale."""

    if not 0 <= season_index < len(_SEASON_IDS):
        raise ValueError("season_index is outside the four-season climate contract")
    values = grid.seasonal_precipitation[season_index].astype(np.float64) / 255.0
    positions = values * (_PRECIPITATION_PALETTE.shape[0] - 1)
    lower = np.floor(positions).astype(np.intp)
    upper = np.ceil(positions).astype(np.intp)
    fraction = positions - lower
    rgb = np.rint(
        _PRECIPITATION_PALETTE[lower] * (1.0 - fraction[..., None])
        + _PRECIPITATION_PALETTE[upper] * fraction[..., None]
    ).astype(np.uint8)
    rgba = np.empty((*grid.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(grid.water == 0, 224, 0).astype(np.uint8)
    display_width, display_height, scale_x, scale_y = _review_dimensions(grid)
    if (display_width, display_height) == (grid.shape[1], grid.shape[0]):
        return rgba
    return np.repeat(np.repeat(rgba, scale_y, axis=0), scale_x, axis=1)


def _wind_arrow_groups(grid: WorldGrid) -> tuple[str, dict[str, int]]:
    """Pack latitude-thinned incoming-wind arrows into three paths per season."""

    height, width = grid.shape
    latitude_rows = np.linspace(90.0, -90.0, height, endpoint=False) - 90.0 / height
    base_step = max(6, int(round(min(height, width) / 28.0)))
    parts: list[str] = []
    counts: dict[str, int] = {}
    for season_index, season_id in enumerate(_SEASON_IDS):
        bins: list[list[str]] = [[], [], []]
        arrow_count = 0
        for row in range(base_step // 2, height, base_step):
            cosine = max(0.28, abs(math.cos(math.radians(float(latitude_rows[row])))))
            column_step = max(base_step, int(round(base_step / cosine)))
            for column in range(column_step // 2, width, column_step):
                east = float(grid.seasonal_wind_east[season_index, row, column]) / 127.0
                north = float(grid.seasonal_wind_north[season_index, row, column]) / 127.0
                speed = math.hypot(east, north)
                if speed < 0.12:
                    continue
                flow_east = east / speed
                flow_north = north / speed
                length = base_step * (0.40 + 0.58 * min(speed, 1.0))
                end_x = column + 0.5
                end_y = row + 0.5
                start_x = end_x - flow_east * length
                start_y = end_y + flow_north * length
                normal_x = -flow_north
                normal_y = -flow_east
                head_length = max(1.5, length * 0.24)
                head_width = max(0.9, length * 0.13)
                left_x = end_x - flow_east * head_length + normal_x * head_width
                left_y = end_y + flow_north * head_length + normal_y * head_width
                right_x = end_x - flow_east * head_length - normal_x * head_width
                right_y = end_y + flow_north * head_length - normal_y * head_width
                command = (
                    f"M{_compact_coordinate(start_x)},{_compact_coordinate(start_y)} "
                    f"L{_compact_coordinate(end_x)},{_compact_coordinate(end_y)} "
                    f"M{_compact_coordinate(left_x)},{_compact_coordinate(left_y)} "
                    f"L{_compact_coordinate(end_x)},{_compact_coordinate(end_y)} "
                    f"L{_compact_coordinate(right_x)},{_compact_coordinate(right_y)}"
                )
                bin_index = min(2, int(speed * 3.0))
                bins[bin_index].append(command)
                arrow_count += 1
        path_parts = []
        for bin_index, commands in enumerate(bins):
            if not commands:
                continue
            stroke_width = (0.34, 0.48, 0.64)[bin_index]
            opacity = (0.58, 0.72, 0.88)[bin_index]
            path_parts.append(
                f'<path d="{" ".join(commands)}" fill="none" stroke="#234e86" '
                f'stroke-width="{stroke_width:.2f}" opacity="{opacity:.2f}" '
                f'data-base-stroke="{stroke_width:.2f}" stroke-linecap="round" '
                'stroke-linejoin="round" />'
            )
        counts[season_id] = arrow_count
        parts.append(
            f'<g id="monsoon-wind-{season_id}" class="seasonal-wind" '
            f'data-layer="monsoon-wind" data-season="{season_id}" hidden '
            f'aria-label="{html.escape(_SEASON_LABELS[season_index])}近地风场">'
            f'{"".join(path_parts)}</g>'
        )
    return (
        '<g id="monsoon-wind" data-layer="monsoon-wind" aria-label="四季近地风场">'
        + "".join(parts)
        + "</g>"
    ), counts


def _river_seasonal_strengths(
    grid: WorldGrid,
    paths: Sequence[np.ndarray],
) -> list[tuple[int, int, int, int]]:
    """Sample each rendered river reach against the four canonical strengths."""

    result: list[tuple[int, int, int, int]] = []
    for path in paths:
        columns = np.clip(np.floor(path[:, 0]).astype(int), 0, grid.shape[1] - 1)
        rows = np.clip(np.floor(path[:, 1]).astype(int), 0, grid.shape[0] - 1)
        result.append(
            tuple(
                int(
                    np.max(
                        grid.seasonal_river_strength[season, rows, columns],
                        initial=0,
                    )
                )
                for season in range(4)
            )
        )
    return result


def _contour_segments(
    values: np.ndarray | np.ma.MaskedArray, levels: list[float]
) -> list[tuple[np.ndarray, float]]:
    if not levels:
        return []
    # contourpy requires a 2x2 sample lattice.  WorldGrid intentionally
    # allows cropped/degenerate previews as small as one cell, so an absent
    # contour is preferable to leaking contourpy's TypeError for those grids.
    if np.ndim(values) != 2 or min(np.shape(values)) < 2:
        return []
    generator = contourpy.contour_generator(
        z=values,
        line_type=contourpy.LineType.Separate,
    )
    paths: list[tuple[np.ndarray, float]] = []
    for level in levels:
        for line in generator.lines(float(level)):
            points = np.asarray(line, dtype=np.float64)
            if points.shape[0] >= 2:
                paths.append((points, float(level)))
    return paths


def _contour_paths(values: np.ndarray | np.ma.MaskedArray, levels: list[float]) -> list[np.ndarray]:
    return [points for points, _ in _contour_segments(values, levels)]


def _mask_paths(mask: np.ndarray) -> list[np.ndarray]:
    return _contour_paths(mask.astype(np.float64), [0.5])


def _filled_range_paths(
    values: np.ndarray,
    active_mask: np.ndarray,
    lower: float,
    upper: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract closed SVG polygons for one categorical surface interval."""

    if values.shape != active_mask.shape:
        raise WorldGridRenderError("filled surface values and mask must share a shape")
    if values.ndim != 2 or min(values.shape) < 2:
        return []
    masked_values = np.ma.masked_where(
        ~active_mask,
        np.asarray(values, dtype=np.float64),
    )
    generator = contourpy.contour_generator(
        z=masked_values,
        line_type=contourpy.LineType.Separate,
        fill_type=contourpy.FillType.OuterCode,
    )
    vertices, codes = generator.filled(float(lower), float(upper))
    return _filled_vertices_to_paths(vertices, codes)


def _filled_vertices_to_paths(
    vertices: Sequence[np.ndarray],
    codes: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Normalize one contourpy OuterCode result for SVG serialization."""

    paths: list[tuple[np.ndarray, np.ndarray]] = []
    for points, path_codes in zip(vertices, codes, strict=True):
        points_array = np.asarray(points, dtype=np.float64)
        codes_array = np.asarray(path_codes, dtype=np.uint8)
        if (
            points_array.ndim == 2
            and points_array.shape[0] >= 3
            and codes_array.shape[0] == points_array.shape[0]
        ):
            paths.append((points_array, codes_array))
    return paths


def _categorical_partition_paths(
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    category_count: int,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Extract compact filled zones for ordered thematic categories."""

    categories = np.asarray(values)
    active = np.asarray(active_mask, dtype=bool)
    if categories.shape != active.shape:
        raise WorldGridRenderError("categorical values and mask must share a shape")
    if isinstance(category_count, bool) or int(category_count) < 1:
        raise WorldGridRenderError("category_count must be a positive integer")
    count = int(category_count)
    if categories.ndim != 2 or min(categories.shape) < 2:
        return [[] for _ in range(count)]
    masked_values = np.ma.masked_where(~active, categories.astype(np.float64))
    generator = contourpy.contour_generator(
        z=masked_values,
        fill_type=contourpy.FillType.OuterCode,
    )
    levels = np.arange(count + 1, dtype=np.float64) - 0.5
    return [
        _filled_vertices_to_paths(vertices, codes)
        for vertices, codes in generator.multi_filled(levels)
    ]


def _geometry_polygons(geometry: Any) -> tuple[Any, ...]:
    """Return every polygonal part of one Shapely geometry."""

    if geometry is None or bool(shapely.is_empty(geometry)):
        return ()
    if geometry.geom_type == "Polygon":
        return (geometry,)
    if geometry.geom_type == "MultiPolygon":
        return tuple(geometry.geoms)
    return tuple(
        part
        for part in shapely.get_parts(geometry)
        if part.geom_type == "Polygon" and not part.is_empty
    )


def _geometry_filled_paths(geometry: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert polygonal coverage geometry to contour-style SVG path inputs."""

    paths: list[tuple[np.ndarray, np.ndarray]] = []
    for polygon in _geometry_polygons(geometry):
        point_groups: list[np.ndarray] = []
        code_groups: list[np.ndarray] = []
        for ring in (polygon.exterior, *polygon.interiors):
            points = np.asarray(ring.coords, dtype=np.float64)
            if len(points) < 4:
                continue
            codes = np.full(len(points), 2, dtype=np.uint8)
            codes[0] = 1
            codes[-1] = 79
            point_groups.append(points)
            code_groups.append(codes)
        if point_groups:
            paths.append((np.vstack(point_groups), np.concatenate(code_groups)))
    return paths


def _coverage_edge_inventory(
    faces: Sequence[Any],
    labels: Sequence[int] | np.ndarray,
) -> dict[
    tuple[tuple[float, float], tuple[float, float]],
    list[int],
]:
    """Index exact shared coverage edges together with their adjacent labels."""

    inventory: dict[
        tuple[tuple[float, float], tuple[float, float]],
        list[int],
    ] = {}
    for face, label in zip(faces, labels, strict=True):
        for polygon in _geometry_polygons(face):
            for ring in (polygon.exterior, *polygon.interiors):
                points = [
                    (round(float(x), 8), round(float(y), 8))
                    for x, y in ring.coords
                ]
                for first, second in zip(points[:-1], points[1:], strict=True):
                    if first == second:
                        continue
                    edge = (
                        (first, second)
                        if first < second
                        else (second, first)
                    )
                    inventory.setdefault(edge, []).append(int(label))
    return inventory


def _coverage_edge_chains(
    edges: set[tuple[tuple[float, float], tuple[float, float]]],
) -> list[np.ndarray]:
    """Trace an undirected edge set into arcs split at true junctions."""

    if not edges:
        return []
    neighbors: dict[tuple[float, float], set[tuple[float, float]]] = {}
    for first, second in edges:
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)
    unused = set(edges)

    def consume(
        start: tuple[float, float], following: tuple[float, float]
    ) -> np.ndarray:
        path = [start, following]
        unused.discard(
            (start, following) if start < following else (following, start)
        )
        previous, current = start, following
        while len(neighbors[current]) == 2:
            candidate = next(point for point in neighbors[current] if point != previous)
            edge = (
                (current, candidate) if current < candidate else (candidate, current)
            )
            if edge not in unused:
                break
            unused.remove(edge)
            path.append(candidate)
            previous, current = current, candidate
        return np.asarray(path, dtype=np.float64)

    paths: list[np.ndarray] = []
    junctions = sorted(point for point, adjacent in neighbors.items() if len(adjacent) != 2)
    for start in junctions:
        for following in sorted(neighbors[start]):
            edge = (
                (start, following) if start < following else (following, start)
            )
            if edge in unused:
                paths.append(consume(start, following))
    while unused:
        start, following = min(unused)
        paths.append(consume(start, following))
    return paths


def _chaikin_arc(
    points: np.ndarray,
    *,
    alpha: float,
    passes: int,
) -> np.ndarray:
    """Round a shared arc while keeping junction endpoints coincident."""

    result = np.asarray(points, dtype=np.float64)
    if len(result) < 3 or passes <= 0:
        return result
    for _ in range(passes):
        closed = np.allclose(result[0], result[-1])
        if closed:
            core = result[:-1]
            rounded: list[np.ndarray] = []
            for first, second in zip(core, np.roll(core, -1, axis=0), strict=True):
                rounded.extend(
                    (
                        (1.0 - alpha) * first + alpha * second,
                        alpha * first + (1.0 - alpha) * second,
                    )
                )
            result = np.asarray(rounded, dtype=np.float64)
            result = np.vstack((result, result[0]))
            continue
        rounded = [result[0]]
        for first, second in zip(result[:-1], result[1:], strict=True):
            rounded.extend(
                (
                    (1.0 - alpha) * first + alpha * second,
                    alpha * first + (1.0 - alpha) * second,
                )
            )
        rounded.append(result[-1])
        result = np.asarray(rounded, dtype=np.float64)
    return result


def _naturalize_surface_path(
    points: np.ndarray,
    *,
    simplification_tolerance: float = 0.72,
    smoothing_alpha: float = 0.22,
    smoothing_passes: int = 1,
) -> np.ndarray:
    """Remove one-cell stair steps from a coastline or mask perimeter."""

    values = _remove_collinear_vertices(np.asarray(points, dtype=np.float64))
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
        return values
    closed = bool(np.allclose(values[0], values[-1]))
    if closed and len(values) < 6:
        return values
    simplified = shapely.simplify(
        shapely.LineString(values),
        float(simplification_tolerance),
        preserve_topology=True,
    )
    if simplified.geom_type != "LineString" or simplified.is_empty:
        return values
    result = np.asarray(simplified.coords, dtype=np.float64)
    if closed and not np.allclose(result[0], result[-1]):
        result = np.vstack((result, result[0]))
    if len(result) < (4 if closed else 2):
        return values
    return _chaikin_arc(
        result,
        alpha=float(smoothing_alpha),
        passes=int(smoothing_passes),
    )


def _naturalize_surface_paths(paths: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Apply the same restrained cartographic curve to every mask outline."""

    return [_naturalize_surface_path(path) for path in paths]


def _naturalize_filled_surface_paths(
    paths: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Naturalize every exterior and hole while preserving SVG fill semantics."""

    result: list[tuple[np.ndarray, np.ndarray]] = []
    for points, codes in paths:
        values = np.asarray(points, dtype=np.float64)
        path_codes = np.asarray(codes, dtype=np.uint8)
        starts = np.flatnonzero(path_codes == 1)
        if not len(starts):
            result.append((values, path_codes))
            continue
        point_groups: list[np.ndarray] = []
        code_groups: list[np.ndarray] = []
        stops = (*starts[1:].tolist(), len(values))
        for start, stop in zip(starts.tolist(), stops, strict=True):
            ring = values[start:stop]
            closed = bool(path_codes[stop - 1] == 79)
            if closed and not np.allclose(ring[0], ring[-1]):
                ring = np.vstack((ring, ring[0]))
            curved = _naturalize_surface_path(ring)
            curved_codes = np.full(len(curved), 2, dtype=np.uint8)
            curved_codes[0] = 1
            if closed:
                curved_codes[-1] = 79
            point_groups.append(curved)
            code_groups.append(curved_codes)
        result.append((np.vstack(point_groups), np.concatenate(code_groups)))
    return result


def _shared_partition_topology(
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    category_count: int,
    simplification_tolerance: float = 3.0,
) -> tuple[tuple[Any, ...], np.ndarray]:
    """Build one valid cartographically smoothed polygon coverage.

    Raster runs first become a fully noded polygon coverage. Mapshaper then
    converts the polygons to shared arcs and applies its adaptive smoother and
    topology cleaner. Each internal edge is therefore moved exactly once for
    both owners instead of being independently rounded in SVG.
    The returned faces are the single source for fills and visible border meshes,
    so no independent curve pass can open junctions or erase enclosed regions.
    """

    categories = np.asarray(values)
    active = np.asarray(active_mask, dtype=bool)
    if categories.shape != active.shape:
        raise WorldGridRenderError("categorical values and mask must share a shape")
    if isinstance(category_count, bool) or int(category_count) < 1:
        raise WorldGridRenderError("category_count must be a positive integer")
    count = int(category_count)
    if categories.ndim != 2 or min(categories.shape) < 2 or not np.any(active):
        return (), np.empty(0, dtype=np.int32)
    positive = categories[active & (categories > 0)]
    if positive.size and int(positive.max()) >= count:
        raise WorldGridRenderError("categorical identifier exceeds category_count")

    sampled_active = active
    sampled_categories = categories
    sampled_categories = np.where(sampled_categories > 0, sampled_categories, 0)

    runs_by_category: dict[int, list[Any]] = {}
    for row_index, row in enumerate(sampled_categories):
        row_active = sampled_active[row_index]
        start = 0
        while start < len(row):
            if not row_active[start]:
                start += 1
                continue
            category = int(row[start])
            end = start + 1
            while (
                end < len(row)
                and row_active[end]
                and int(row[end]) == category
            ):
                end += 1
            minimum_x = max(0.0, start - 0.5)
            maximum_x = min(float(categories.shape[1]), end - 0.5)
            minimum_y = max(0.0, row_index - 0.5)
            maximum_y = min(float(categories.shape[0]), row_index + 0.5)
            runs_by_category.setdefault(category, []).append(
                shapely.box(minimum_x, minimum_y, maximum_x, maximum_y)
            )
            start = end

    coarse_geometries = [
        shapely.union_all(runs) for runs in runs_by_category.values() if runs
    ]
    if not coarse_geometries:
        return (), np.empty(0, dtype=np.int32)
    linework = shapely.unary_union(
        [geometry.boundary for geometry in coarse_geometries]
    )
    raw_faces = tuple(
        part
        for part in shapely.get_parts(
            shapely.polygonize(shapely.get_parts(linework))
        )
        if part.geom_type == "Polygon" and not part.is_empty
    )

    def classify_faces(faces: Sequence[Any]) -> tuple[tuple[Any, ...], np.ndarray]:
        kept: list[Any] = []
        labels: list[int] = []
        for face in faces:
            point = face.representative_point()
            column = int(np.clip(round(point.x), 0, categories.shape[1] - 1))
            row = int(np.clip(round(point.y), 0, categories.shape[0] - 1))
            if not active[row, column]:
                continue
            kept.append(face)
            labels.append(max(0, int(categories[row, column])))
        return tuple(kept), np.asarray(labels, dtype=np.int32)

    faces, face_labels = classify_faces(raw_faces)
    if not faces or not bool(shapely.coverage_is_valid(faces, gap_width=1.0e-6)):
        raise WorldGridRenderError("administrative polygon coverage is invalid")
    simplified, simplified_labels = _mapshaper_smooth_coverage(
        faces,
        face_labels,
        distance=max(0.18, float(simplification_tolerance) * 0.82),
    )
    if not bool(shapely.coverage_is_valid(simplified, gap_width=1.0e-6)):
        raise WorldGridRenderError("simplified administrative coverage is invalid")
    return simplified, simplified_labels


def _mapshaper_smooth_coverage(
    faces: Sequence[Any],
    labels: Sequence[int] | np.ndarray,
    *,
    distance: float,
) -> tuple[tuple[Any, ...], np.ndarray]:
    """Smooth one polygon coverage through Mapshaper's shared-arc topology."""

    face_labels = np.asarray(labels, dtype=np.int32)
    if len(faces) != len(face_labels):
        raise WorldGridRenderError("coverage labels must correspond to faces")
    if not faces:
        return (), np.empty(0, dtype=np.int32)
    from world_atlas.runtime import require_renderer
    _node, executable = require_renderer()
    features = [
        {
            "type": "Feature",
            "properties": {"partitionLabel": int(label)},
            "geometry": json.loads(shapely.to_geojson(face)),
        }
        for face, label in zip(faces, face_labels, strict=True)
    ]
    payload = {"type": "FeatureCollection", "features": features}
    with tempfile.TemporaryDirectory(prefix="eirenor-mapshaper-") as temporary:
        source = Path(temporary) / "coverage.geojson"
        target = Path(temporary) / "smoothed.geojson"
        source.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        completed = subprocess.run(
            (
                _node,
                str(executable),
                str(source),
                "-simplify",
                "weighted",
                f"interval={max(0.24, float(distance) * 0.62):.5f}",
                "planar",
                "keep-shapes",
                "-smooth",
                f"{float(distance):.5f}",
                "no-corners",
                "max-bend-angle=10",
                "-clean",
                "-o",
                str(target),
                "format=geojson",
                "precision=0.001",
            ),
            cwd=temporary,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
            check=False,
        )
        if completed.returncode != 0 or not target.is_file():
            details = (completed.stderr or completed.stdout).strip()
            raise WorldGridRenderError(f"Mapshaper topology smoothing failed: {details}")
        document = json.loads(target.read_text(encoding="utf-8"))
    smoothed_faces: list[Any] = []
    smoothed_labels: list[int] = []
    for feature in document.get("features", ()): 
        geometry_record = feature.get("geometry")
        properties = feature.get("properties") or {}
        if not geometry_record or "partitionLabel" not in properties:
            continue
        geometry = shapely.from_geojson(
            json.dumps(geometry_record, ensure_ascii=False, separators=(",", ":"))
        )
        for part in shapely.get_parts(geometry):
            if part.geom_type == "Polygon" and not part.is_empty:
                smoothed_faces.append(part)
                smoothed_labels.append(int(properties["partitionLabel"]))
    if not smoothed_faces:
        raise WorldGridRenderError("Mapshaper removed the administrative coverage")
    return tuple(smoothed_faces), np.asarray(smoothed_labels, dtype=np.int32)


def _validate_partition_inventory(
    values: np.ndarray,
    active_mask: np.ndarray,
    rendered_labels: np.ndarray,
    *,
    layer_name: str,
) -> None:
    """Require every authored positive region to survive SVG topology work."""

    categories = np.asarray(values)
    active = np.asarray(active_mask, dtype=bool)
    labels = np.asarray(rendered_labels)
    if categories.shape != active.shape:
        raise WorldGridRenderError("categorical values and mask must share a shape")
    expected = {
        int(value)
        for value in np.unique(categories[active & (categories > 0)])
    }
    rendered = {int(value) for value in np.unique(labels[labels > 0])}
    missing = sorted(expected - rendered)
    unexpected = sorted(rendered - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("lost identifiers: " + ", ".join(map(str, missing)))
        if unexpected:
            details.append("invented identifiers: " + ", ".join(map(str, unexpected)))
        raise WorldGridRenderError(f"{layer_name} topology " + "; ".join(details))


def _shared_topology_zone_paths(
    faces: Sequence[Any],
    labels: Sequence[int] | np.ndarray,
    *,
    category_count: int,
    include_zero: bool = False,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Dissolve shared topology faces into fill paths for one hierarchy level."""

    face_labels = np.asarray(labels, dtype=np.int32)
    zones: list[list[tuple[np.ndarray, np.ndarray]]] = [
        [] for _ in range(int(category_count))
    ]
    for category in range(0 if include_zero else 1, int(category_count)):
        selected = [
            face for face, label in zip(faces, face_labels, strict=True) if label == category
        ]
        if selected:
            zones[category] = _geometry_filled_paths(shapely.union_all(selected))
    return zones


def _shared_topology_boundary_paths(
    faces: Sequence[Any],
    labels: Sequence[int] | np.ndarray,
    *,
    include_unassigned: bool = False,
) -> list[np.ndarray]:
    """Extract each visible shared border exactly once from a polygon coverage."""

    inventory = _coverage_edge_inventory(faces, labels)
    edges = {
        edge
        for edge, owners in inventory.items()
        if len(owners) >= 2
        and len(set(owners)) > 1
        and (
            any(owner > 0 for owner in owners)
            if include_unassigned
            else all(owner > 0 for owner in owners)
        )
    }
    return _coverage_edge_chains(edges)


def _filled_band_paths(
    values: np.ndarray,
    active_mask: np.ndarray,
    band_count: int,
    *,
    higher_values_on_top: bool = True,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Extract nested, opaque SVG terraces for an ordinal surface.

    A physical hypsometric map is a stack of solid layers, not a mosaic of
    mutually exclusive polygons.  Exact-band polygonisation leaves ambiguous
    marching-squares holes where three or more categories meet; the lowland
    or ocean backfill then shows through as the cyan diamonds seen at high
    zoom.  Cumulative masks guarantee that every upper terrace rests on the
    immediately lower terrace.  Bathymetry uses the inverse ordering so the
    deep-ocean base is successively covered by shallower shelves.
    """

    ordinal_values = np.asarray(values, dtype=np.float64)
    if ordinal_values.shape != np.asarray(active_mask).shape:
        raise WorldGridRenderError("filled band values and mask must share a shape")

    bands: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for band in range(int(band_count)):
        threshold_mask = (
            ordinal_values >= band
            if higher_values_on_top
            else ordinal_values <= band
        )
        bands.append(_filled_mask_paths(active_mask & threshold_mask))
    return bands


def _elevation_thresholds(grid: WorldGrid) -> list[float]:
    """Return the authored boundaries shared by land fills and contours."""

    band_count = _elevation_band_count(grid)
    return [band / band_count for band in range(1, band_count)]


def _elevation_band_paths(
    grid: WorldGrid,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Extract nested solid terraces from the canonical continuous DEM.

    ``elevation_band`` is the quantized view of the relative ``0..1`` DEM.
    Tracing the categorical array a second time moves every crossing to the
    midpoint between two cells.  The contour line, however, crosses at the
    actual interpolated threshold.  Building both from the continuous DEM and
    the same authored thresholds keeps their SVG geometry coincident.
    """

    land = grid.water == 0
    bands: list[list[tuple[np.ndarray, np.ndarray]]] = [
        _filled_mask_paths(land)
    ]
    if not np.any(land):
        return bands + [[] for _ in _elevation_thresholds(grid)]

    elevation = grid.elevation.astype(np.float64)
    land_maximum = float(elevation[land].max())
    for threshold in _elevation_thresholds(grid):
        if land_maximum < threshold:
            bands.append([])
            continue
        upper = np.nextafter(max(1.0, land_maximum), np.inf)
        bands.append(
            _filled_range_paths(elevation, land, threshold, upper)
        )
    return bands


def _filled_mask_paths(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract vector polygons for a boolean surface mask."""

    values = np.asarray(mask, dtype=np.float64)
    return _filled_range_paths(values, np.ones(mask.shape, dtype=bool), 0.5, 1.5)


def _compact_coordinate(value: float) -> str:
    """Format SVG coordinates without serialising predictable zero padding."""

    numeric = float(value)
    if abs(numeric - round(numeric)) <= 1.0e-9:
        return str(int(round(numeric)))
    if abs(numeric * 2.0 - round(numeric * 2.0)) <= 1.0e-9:
        return f"{numeric:.1f}".rstrip("0").rstrip(".")
    # Hundredths are far below a screen pixel even at the closest supported
    # zoom.  More decimals noticeably bloat the self-contained review page
    # once shared coast and administrative curves are serialized.
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


def _remove_collinear_vertices(points: np.ndarray) -> np.ndarray:
    """Drop exact straight-run vertices while retaining every corner."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        return values
    incoming = values[1:-1] - values[:-2]
    outgoing = values[2:] - values[1:-1]
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = np.einsum("ij,ij->i", incoming, outgoing)
    keep = np.ones(values.shape[0], dtype=bool)
    keep[1:-1] = ~((np.abs(cross) <= 1.0e-9) & (dot >= -1.0e-9))
    return values[keep]


def _review_extents(grid: WorldGrid) -> tuple[float, float, float, float]:
    """Return the authoritative ``west, south, east, north`` review extents."""

    extents = grid.metadata.get("extents")
    if not isinstance(extents, Mapping):
        raise WorldGridRenderError("WorldGrid metadata extents are required for graticule")
    try:
        west = float(extents["west"])
        south = float(extents["south"])
        east = float(extents["east"])
        north = float(extents["north"])
    except (KeyError, TypeError, ValueError) as error:
        raise WorldGridRenderError("WorldGrid metadata extents are invalid") from error
    if not all(np.isfinite((west, south, east, north))) or not west < east or not south < north:
        raise WorldGridRenderError("WorldGrid metadata extents must be finite and ordered")
    return west, south, east, north


def _graticule_paths(
    grid: WorldGrid,
) -> tuple[list[np.ndarray], list[bool], list[tuple[str, float, float]]]:
    """Build subtle 10-degree graticule paths and labels from world extents."""

    west, south, east, north = _review_extents(grid)
    width, height = grid.shape[1], grid.shape[0]

    def x_for(longitude: float) -> float:
        return (longitude - west) / (east - west) * width

    def y_for(latitude: float) -> float:
        return (north - latitude) / (north - south) * height

    paths: list[np.ndarray] = []
    major_flags: list[bool] = []
    labels: list[tuple[str, float, float]] = []
    minor_longitude_start = int(np.ceil(west / 10.0)) * 10
    for longitude in range(minor_longitude_start, int(np.floor(east / 10.0)) * 10 + 1, 10):
        if longitude <= west or longitude >= east:
            continue
        x = x_for(float(longitude))
        paths.append(np.asarray(((x, 0.0), (x, float(height))), dtype=np.float64))
        major_flags.append(longitude % 30 == 0)
        if longitude % 30 == 0:
            labels.append((_format_longitude(longitude), x, float(height - 7)))

    minor_latitude_start = int(np.ceil(south / 10.0)) * 10
    for latitude in range(minor_latitude_start, int(np.floor(north / 10.0)) * 10 + 1, 10):
        if latitude <= south or latitude >= north:
            continue
        y = y_for(float(latitude))
        paths.append(np.asarray(((0.0, y), (float(width), y)), dtype=np.float64))
        major_flags.append(latitude % 30 == 0)
        if latitude % 30 == 0:
            labels.append((_format_latitude(latitude), 5.0, y - 3.0))
    return paths, major_flags, labels


def _format_longitude(longitude: int) -> str:
    if longitude == 0:
        return "0°"
    return f"{abs(longitude)}°{'E' if longitude > 0 else 'W'}"


def _format_latitude(latitude: int) -> str:
    if latitude == 0:
        return "0°"
    return f"{abs(latitude)}°{'N' if latitude > 0 else 'S'}"


def _graticule_svg(
    grid: WorldGrid,
    paths: list[np.ndarray],
    major_flags: list[bool],
    labels: list[tuple[str, float, float]],
) -> str:
    """Serialize graticule paths and labels beneath the physical overlays."""

    west, south, east, north = _review_extents(grid)
    if len(paths) != len(major_flags):
        raise WorldGridRenderError("graticule path styles must match path count")
    body = []
    for path, major in zip(paths, major_flags, strict=True):
        body.append(
            f'<path d="{_path_data(path)}" fill="none" stroke="#3b5876" '
            f'stroke-width="{0.50 if major else 0.28:.2f}" opacity="{0.34 if major else 0.16:.2f}" '
            f'data-base-stroke="{0.50 if major else 0.28:.2f}" '
            'stroke-linecap="round" />'
        )
    for text, x, y in labels:
        body.append(
            f'<text x="{x:.3f}" y="{y:.3f}" fill="#314a70" opacity="0.72" '
            'font-size="9" font-family="Microsoft YaHei, sans-serif" '
            'paint-order="stroke" stroke="#dbe3ec" stroke-width="2" stroke-opacity="0.75">'
            f'{html.escape(text)}</text>'
        )
    return (
        f'<g id="graticule" data-layer="graticule" data-west="{west:g}" '
        f'data-south="{south:g}" data-east="{east:g}" data-north="{north:g}" '
        f'aria-label="经纬线">{"".join(body)}</g>'
    )


def _polar_reference_svg(grid: WorldGrid) -> str:
    """Draw polar circles and explain how exact poles occupy projected edges."""

    west, _south, east, north = _review_extents(grid)
    planet = grid.metadata.get("planet")
    if not isinstance(planet, Mapping):
        raise WorldGridRenderError("WorldGrid metadata planet is required for polar circles")
    try:
        tilt = float(planet["axialTiltDegrees"])
    except (KeyError, TypeError, ValueError) as error:
        raise WorldGridRenderError(
            "WorldGrid metadata planet.axialTiltDegrees is required for polar circles"
        ) from error
    if not np.isfinite(tilt) or not 0.0 < tilt < 45.0:
        raise WorldGridRenderError("planet axial tilt must be finite and physically plausible")
    circle_latitude = 90.0 - tilt
    width, height = grid.shape[1], grid.shape[0]

    def y_for(latitude: float) -> float:
        return (north - latitude) / 180.0 * height

    rows = []
    for latitude, label, baseline in (
        (circle_latitude, "北极圈", 1.0),
        (-circle_latitude, "南极圈", -4.0),
    ):
        y = y_for(latitude)
        rows.append(
            f'<path d="M0,{y:.3f} L{width},{y:.3f}" fill="none" '
            'stroke="#365d79" stroke-width="0.70" stroke-dasharray="5 4" '
            'opacity="0.55" data-base-stroke="0.70" />'
            f'<text x="8" y="{y + baseline:.3f}" fill="#294c68" font-size="9" '
            'font-family="Microsoft YaHei, sans-serif" paint-order="stroke" '
            'stroke="#edf3f2" stroke-width="2">'
            f'{label} {circle_latitude:.2f}°</text>'
        )
    rows.append(
        f'<path d="M0,0 L{width},0" stroke="#365d79" stroke-width="1.1" '
        'opacity="0.72" data-base-stroke="1.10" />'
        f'<path d="M0,{height} L{width},{height}" stroke="#365d79" '
        'stroke-width="1.1" opacity="0.72" data-base-stroke="1.10" />'
        f'<text x="{width / 2:.3f}" y="14" text-anchor="middle" fill="#294c68" '
        'font-size="11" font-weight="700" font-family="Microsoft YaHei, sans-serif" '
        'paint-order="stroke" stroke="#edf3f2" stroke-width="2.5">'
        '90°N 北极（经度在上边界汇聚）</text>'
        f'<text x="{width / 2:.3f}" y="{height - 7:.3f}" text-anchor="middle" '
        'fill="#294c68" font-size="11" font-weight="700" '
        'font-family="Microsoft YaHei, sans-serif" paint-order="stroke" '
        'stroke="#edf3f2" stroke-width="2.5">'
        '90°S 南极（经度在下边界汇聚）</text>'
    )
    return (
        f'<g id="polar-references" data-layer="polar-references" '
        f'data-axial-tilt="{tilt:g}" data-west="{west:g}" data-east="{east:g}" '
        f'aria-label="极圈与极点投影边界">{"".join(rows)}</g>'
    )


def _polar_label_overlay(grid: WorldGrid) -> str:
    width, height = grid.shape[1], grid.shape[0]
    labels = (
        ("北极地区", "极区投影参考", width * 0.50, height * 0.055),
        ("南极地区", "极区投影参考", width * 0.50, height * 0.925),
    )
    return (
        '<g id="polar-labels" data-layer="polar-labels" aria-label="极区名称">'
        + "".join(
            f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="middle" fill="#24465f" '
            'font-size="16" font-weight="700" font-family="Microsoft YaHei, sans-serif" '
            'paint-order="stroke" stroke="#f3f7f4" stroke-width="3.2">'
            f'{name}<tspan x="{x:.3f}" dy="13" font-size="9" font-weight="500">'
            f'{subtitle}</tspan></text>'
            for name, subtitle, x, y in labels
        )
        + '</g>'
    )


def _elevation_paths(grid: WorldGrid) -> tuple[list[np.ndarray], list[float]]:
    land_elevation = np.ma.masked_where(
        grid.water != 0, grid.elevation.astype(np.float64)
    )
    if land_elevation.count() == 0:
        return [], []
    levels = _elevation_thresholds(grid)
    segments = _contour_segments(land_elevation, levels)
    return [points for points, _ in segments], [level for _, level in segments]


def _path_data(points: np.ndarray) -> str:
    compact = _remove_collinear_vertices(points)
    if compact.shape[0] == 0:
        return ""
    coordinates = " ".join(
        f"{_compact_coordinate(x)},{_compact_coordinate(y)}" for x, y in compact
    )
    # SVG treats every coordinate pair after the initial moveto as an implicit
    # lineto.  Keeping only one command preserves the full path while avoiding
    # hundreds of thousands of redundant ``L`` tokens in the standalone atlas.
    return f"M{coordinates}"


def _river_path_data(points: np.ndarray) -> str:
    """Round canonical flow-cell anchors without leaving their corridor."""

    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 2:
        return _path_data(raw)
    keep = np.ones(raw.shape[0], dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(raw, axis=0)) > 1.0e-12, axis=1)
    values = raw[keep]
    if values.ndim != 2 or values.shape[0] < 3:
        return _path_data(values)
    commands = [
        f"M{_compact_coordinate(values[0, 0])},{_compact_coordinate(values[0, 1])}"
    ]
    for index in range(1, values.shape[0] - 1):
        control = values[index]
        endpoint = (
            values[-1]
            if index == values.shape[0] - 2
            else (values[index] + values[index + 1]) * 0.5
        )
        commands.append(
            "Q"
            f"{_compact_coordinate(control[0])},{_compact_coordinate(control[1])} "
            f"{_compact_coordinate(endpoint[0])},{_compact_coordinate(endpoint[1])}"
        )
    return " ".join(commands)


def _filled_path_data(points: np.ndarray, codes: np.ndarray) -> str:
    """Serialize contourpy OuterCode polygons as one even-odd SVG path."""

    commands: list[str] = []
    values = np.asarray(points, dtype=np.float64)
    code_values = np.asarray(codes, dtype=np.uint8)
    if values.shape[0] != code_values.shape[0]:
        raise WorldGridRenderError("filled path coordinates and codes must share a length")
    segment_start = 0
    compact_segments: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for index, code in enumerate(code_values):
        if int(code) == 1 and index != segment_start:
            compact_segments.append(
                (values[segment_start:index], code_values[segment_start:index], False)
            )
            segment_start = index
        if int(code) == 79:
            compact_segments.append(
                (values[segment_start:index], code_values[segment_start:index], True)
            )
            segment_start = index + 1
    if segment_start < values.shape[0]:
        compact_segments.append((values[segment_start:], code_values[segment_start:], False))

    for segment_points, segment_codes, closed in compact_segments:
        if segment_points.shape[0] == 0:
            continue
        compact_points = _remove_collinear_vertices(segment_points)
        for index, ((x, y), code) in enumerate(
            zip(compact_points, segment_codes, strict=False)
        ):
            code_value = int(code)
            if code_value == 1:
                commands.append(f"M{_compact_coordinate(x)},{_compact_coordinate(y)}")
            elif code_value == 2:
                # Repeated coordinate pairs after M are implicit lineto.
                # Preserve every vertex and ring while reducing SVG bytes.
                commands.append(f"{_compact_coordinate(x)},{_compact_coordinate(y)}")
            elif code_value == 79:
                commands.append("Z")
            else:
                raise WorldGridRenderError(f"unsupported filled contour code: {code_value}")
        # ``_remove_collinear_vertices`` preserves the ring's close point, but
        # the contour code is emitted separately below so it remains explicit.
        if closed:
            commands.append("Z")
    return " ".join(commands)


def _flatten_filled_paths(
    bands: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
) -> list[np.ndarray]:
    return [points for band in bands for points, _ in band]


def _packed_band_budget_paths(
    bands: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
) -> list[np.ndarray]:
    """Represent one serialized SVG path element per non-empty band for budgets."""

    return [
        np.empty((1, 2), dtype=np.float64)
        for band in bands
        if band
    ]


def _svg_filled_band_group(
    layer_id: str,
    label: str,
    bands: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    palette: np.ndarray,
    *,
    reverse_bands: bool = False,
) -> str:
    body_parts: list[str] = []
    band_id_prefix = layer_id[:-1] if layer_id.endswith("s") else layer_id
    band_indices = range(len(bands) - 1, -1, -1) if reverse_bands else range(len(bands))
    for band in band_indices:
        paths = bands[band]
        if not paths:
            continue
        color = _rgb_hex(palette[band])
        path_data = " ".join(
            _filled_path_data(points, codes)
            for points, codes in paths
        )
        paths_markup = (
            f'<path d="{path_data}" fill="{color}" '
            'fill-rule="evenodd" stroke="none" />'
        )
        body_parts.append(
            f'<g id="{band_id_prefix}-{band:02d}" data-layer="{layer_id}" '
            f'data-band="{band}" aria-label="{html.escape(label)} {band + 1}">'
            f"{paths_markup}</g>"
        )
    return (
        f'<g id="{layer_id}" data-layer="{layer_id}" '
        f'aria-label="{html.escape(label)}">{"".join(body_parts)}</g>'
    )


def _svg_filled_surface_group(
    layer_id: str,
    label: str,
    fill_paths: Sequence[tuple[np.ndarray, np.ndarray]],
    fill_color: str,
    outline_paths: Sequence[np.ndarray] = (),
    outline_color: str = "none",
    outline_width: float = 0.0,
    opacity: float = 1.0,
) -> str:
    body_parts = [
        f'<path d="{_filled_path_data(points, codes)}" fill="{fill_color}" '
        'fill-rule="evenodd" stroke="none" />'
        for points, codes in fill_paths
    ]
    if outline_paths and outline_width > 0:
        body_parts.extend(
            f'<path d="{_path_data(path)}" fill="none" stroke="{outline_color}" '
            f'stroke-width="{outline_width:.2f}" stroke-linecap="round" '
            f'data-base-stroke="{outline_width:.2f}" stroke-linejoin="round" />'
            for path in outline_paths
        )
    return (
        f'<g id="{layer_id}" data-layer="{layer_id}" opacity="{opacity:.2f}" '
        f'aria-label="{html.escape(label)}">{"".join(body_parts)}</g>'
    )


def _society_color(identifier: int, *, saturation: float, lightness: float) -> str:
    hue = ((identifier * 0.618033988749895) + 0.09) % 1.0
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _society_zones(
    names: Sequence[str],
    *,
    prefix: str,
    saturation: float,
    lightness: float,
) -> tuple[tuple[str, str], ...]:
    return (
        (f"{prefix}边缘", "#d8d3c2"),
        *(
            (name, _society_color(index, saturation=saturation, lightness=lightness))
            for index, name in enumerate(names, start=1)
        ),
    )


def _hex_hls(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
    return colorsys.rgb_to_hls(red, green, blue)


def _hls_hex(hue: float, lightness: float, saturation: float) -> str:
    red, green, blue = colorsys.hls_to_rgb(hue % 1.0, lightness, saturation)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _culture_zones(society: SocietyLayers) -> tuple[tuple[str, str], ...]:
    zones: list[tuple[str, str]] = [("极端荒地／近无人带", "#d8d3c2")]
    for item in society.cultures.civilizations:
        zones.append(
            (
                item.name,
                _society_color(item.identifier, saturation=0.34, lightness=0.66),
            )
        )
    zones.extend(
        (
            ("山地部族文化带", "#9b92a5"),
            ("游牧与半农半牧文化带", "#b49c68"),
            ("渔猎与林河文化带", "#71999a"),
            ("疏居农牧文化带", "#9da77b"),
        )
    )
    return tuple(zones)


def _religion_zones(society: SocietyLayers) -> tuple[tuple[str, str], ...]:
    return _society_zones(
        [item.name for item in society.religions.religions],
        prefix="宗教",
        saturation=0.30,
        lightness=0.68,
    )


def _civilization_display_values(
    grid: WorldGrid,
    thematic: ThematicLayers,
    society: SocietyLayers,
) -> np.ndarray:
    """Split non-urban frontier into evidence-based local culture belts."""

    values = np.asarray(society.cultures.civilization_id, dtype=np.int16).copy()
    local = (grid.water == 0) & (values == 0)
    base = len(society.cultures.civilizations)
    mountain = local & (grid.elevation >= 0.58) & ~grid.snow
    pastoral = local & np.isin(
        thematic.biome_zone,
        (BIOME_TEMPERATE_GRASSLAND, BIOME_SAVANNA_DRY_GRASSLAND),
    )

    water_network = (grid.river_order > 0) | (grid.water == 2)
    for _pass in range(5):
        expanded = np.roll(water_network, 1, axis=1) | np.roll(
            water_network, -1, axis=1
        )
        expanded[1:] |= water_network[:-1]
        expanded[:-1] |= water_network[1:]
        water_network |= expanded
    forest = np.isin(
        thematic.biome_zone,
        (
            BIOME_TUNDRA_ALPINE,
            BIOME_BOREAL_FOREST,
            BIOME_TEMPERATE_MIXED_FOREST,
            BIOME_HUMID_SUBTROPICAL,
            BIOME_TROPICAL_RAINFOREST,
            BIOME_TROPICAL_SEASONAL_FOREST,
        ),
    )
    foraging = local & forest & water_network
    sparse_agropastoral = local & (
        (thematic.land_potential >= 0.20)
        & (thematic.climate.annual_precipitation >= 0.045)
        & ~grid.snow
    )

    values[mountain] = base + 1
    values[pastoral & ~mountain] = base + 2
    values[foraging & ~mountain & ~pastoral] = base + 3
    values[sparse_agropastoral & ~mountain & ~pastoral & ~foraging] = base + 4
    return values


def _language_zones(society: SocietyLayers) -> tuple[tuple[str, str], ...]:
    counters: dict[int, int] = {}
    zones: list[tuple[str, str]] = [("地方语言连续带", "#d8d3c2")]
    for item in society.cultures.languages:
        index = counters.get(item.family_identifier, 0)
        counters[item.family_identifier] = index + 1
        hue, lightness, saturation = _hex_hls(
            _society_color(item.family_identifier, saturation=0.32, lightness=0.67)
        )
        hue += (((index * 0.61803398875) % 1.0) - 0.5) * 0.075
        lightness = float(np.clip(lightness + (((index * 3) % 7) - 3) * 0.025, 0.56, 0.76))
        zones.append((item.name, _hls_hex(hue, lightness, max(0.22, saturation))))
    return tuple(zones)


def _political_zones(society: SocietyLayers) -> tuple[tuple[str, str], ...]:
    """Assign adjacent states visibly distinct cartographic fills."""

    state_ids = society.politics.state_id
    adjacency: dict[int, set[int]] = {
        state.identifier: set() for state in society.politics.states
    }
    for first, second in (
        (state_ids[:, :-1], state_ids[:, 1:]),
        (state_ids[:-1, :], state_ids[1:, :]),
    ):
        changed = (first > 0) & (second > 0) & (first != second)
        for left, right in zip(first[changed], second[changed], strict=True):
            left_identifier = int(left)
            right_identifier = int(right)
            adjacency[left_identifier].add(right_identifier)
            adjacency[right_identifier].add(left_identifier)
    state_by_id = {state.identifier: state for state in society.politics.states}
    ordered = sorted(
        state_by_id,
        key=lambda identifier: (-len(adjacency[identifier]), identifier),
    )
    # Political colors describe states, not civilizations.  A deterministic
    # graph coloring keeps every shared frontier legible without introducing
    # a second, heavy civilization-border hierarchy.
    palette = (
        "#d2ae73",
        "#79aa98",
        "#8da5c6",
        "#c18c89",
        "#a994bb",
        "#72aeb1",
        "#bea078",
        "#9eae73",
        "#ca906f",
    )
    color_index: dict[int, int] = {}
    for identifier in ordered:
        blocked = {
            color_index[neighbor]
            for neighbor in adjacency[identifier]
            if neighbor in color_index
        }
        preferred = (identifier * 5 + 3) % len(palette)
        color_index[identifier] = next(
            (
                (preferred + offset) % len(palette)
                for offset in range(len(palette))
                if (preferred + offset) % len(palette) not in blocked
            ),
            preferred,
        )
    zones: list[tuple[str, str]] = [("部族地／无常设政权", "#d8d3c2")]
    for state in society.politics.states:
        zones.append((state.name, palette[color_index[state.identifier]]))
    return tuple(zones)


def _province_color_slots(
    province_id: np.ndarray,
    state_by_province: Mapping[int, int],
) -> dict[int, int]:
    """Graph-color provinces so same-state neighbours never share a slot."""

    identifiers = tuple(sorted(int(identifier) for identifier in state_by_province))
    adjacency: dict[int, set[int]] = {identifier: set() for identifier in identifiers}
    values = np.asarray(province_id)
    pairs = [
        (values[:, :-1], values[:, 1:]),
        (values[:-1, :], values[1:, :]),
    ]
    if values.shape[1] > 1:
        # The Plate Carree map wraps at the antimeridian.  Provinces touching
        # through that seam are real neighbours and need contrasting fills.
        pairs.append((values[:, -1:], values[:, :1]))
    for first, second in pairs:
        changed = (first > 0) & (second > 0) & (first != second)
        for left, right in zip(first[changed], second[changed], strict=True):
            left_identifier = int(left)
            right_identifier = int(right)
            if (
                left_identifier not in adjacency
                or right_identifier not in adjacency
                or state_by_province[left_identifier]
                != state_by_province[right_identifier]
            ):
                continue
            adjacency[left_identifier].add(right_identifier)
            adjacency[right_identifier].add(left_identifier)

    # DSATUR uses very few variants on ordinary planar province graphs and is
    # deterministic.  Non-neighbouring provinces may reuse a slot, preserving
    # a calm national color family without sacrificing local legibility.
    slots: dict[int, int] = {}
    uncolored = set(identifiers)
    while uncolored:
        identifier = min(
            uncolored,
            key=lambda candidate: (
                -len(
                    {
                        slots[neighbor]
                        for neighbor in adjacency[candidate]
                        if neighbor in slots
                    }
                ),
                -len(adjacency[candidate]),
                candidate,
            ),
        )
        blocked = {
            slots[neighbor]
            for neighbor in adjacency[identifier]
            if neighbor in slots
        }
        slot = 0
        while slot in blocked:
            slot += 1
        slots[identifier] = slot
        uncolored.remove(identifier)
    return slots


def _province_variant_color(base_color: str, slot: int) -> str:
    """Derive one legible province fill from a parent-state color."""

    variants = (
        (-0.020, -0.110, 1.08),
        (+0.015, +0.080, 0.82),
        (+0.055, -0.020, 0.95),
        (-0.055, +0.015, 1.15),
        (+0.085, -0.160, 0.75),
        (-0.085, +0.125, 0.75),
        (+0.110, +0.025, 1.15),
        (-0.110, -0.065, 0.95),
    )
    hue, lightness, saturation = _hex_hls(base_color)
    hue_delta, lightness_delta, saturation_factor = variants[
        slot % len(variants)
    ]
    # The adjacency graph of contiguous planar provinces normally needs at
    # most four slots.  Extra cycles still yield distinct colors if malformed
    # or fragmented data creates an unusually dense adjacency graph.
    cycle = slot // len(variants)
    hue = (hue + hue_delta + cycle * 0.085) % 1.0
    lightness = float(np.clip(lightness + lightness_delta, 0.48, 0.79))
    saturation = float(np.clip(saturation * saturation_factor, 0.20, 0.58))
    return _hls_hex(hue, lightness, saturation)


def _province_zones(society: SocietyLayers) -> tuple[tuple[str, str], ...]:
    """Color adjacent provinces distinctly inside their parent-state palette."""

    political = _political_zones(society)
    state_by_province = {
        province.identifier: province.state_identifier
        for province in society.provinces.provinces
    }
    slots = _province_color_slots(
        society.provinces.province_id,
        state_by_province,
    )
    zones: list[tuple[str, str]] = [("部族地／无常设政权", "#d8d3c2")]
    # Each slot remains recognizably derived from the state fill, while the
    # combined hue, value and chroma changes remain visible on muted palettes.
    for province in society.provinces.provinces:
        base_color = political[province.state_identifier][1]
        slot = slots[province.identifier]
        zones.append((province.name, _province_variant_color(base_color, slot)))
    return tuple(zones)


def _partition_boundary_paths(
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    include_unassigned: bool = False,
) -> list[np.ndarray]:
    categories = np.asarray(values)
    active = np.asarray(active_mask, dtype=bool)
    # Store vertices at twice their map coordinate so every shared cell edge is
    # represented exactly with integers.  Only interfaces between two active
    # land cells are admitted: coasts and lake shores belong to the physical
    # map and must never be restroked as political borders.
    segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def add_segment(first: tuple[int, int], second: tuple[int, int]) -> None:
        segments.add((first, second) if first < second else (second, first))

    horizontal_pair = active[:, :-1] & active[:, 1:]
    horizontal_categories = categories[:, :-1] != categories[:, 1:]
    horizontal_ownership = (
        ((categories[:, :-1] > 0) | (categories[:, 1:] > 0))
        if include_unassigned
        else ((categories[:, :-1] > 0) & (categories[:, 1:] > 0))
    )
    horizontal_change = horizontal_pair & horizontal_categories & horizontal_ownership
    for row, column in np.argwhere(horizontal_change):
        x = 2 * int(column) + 1
        y = 2 * int(row)
        add_segment((x, y - 1), (x, y + 1))

    vertical_pair = active[:-1, :] & active[1:, :]
    vertical_categories = categories[:-1, :] != categories[1:, :]
    vertical_ownership = (
        ((categories[:-1, :] > 0) | (categories[1:, :] > 0))
        if include_unassigned
        else ((categories[:-1, :] > 0) & (categories[1:, :] > 0))
    )
    vertical_change = vertical_pair & vertical_categories & vertical_ownership
    for row, column in np.argwhere(vertical_change):
        x = 2 * int(column)
        y = 2 * int(row) + 1
        add_segment((x - 1, y), (x + 1, y))

    if not segments:
        return []

    neighbors: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for first, second in segments:
        neighbors.setdefault(first, set()).add(second)
        neighbors.setdefault(second, set()).add(first)
    unused = set(segments)

    def consume(start: tuple[int, int], following: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start, following]
        unused.discard((start, following) if start < following else (following, start))
        previous, current = start, following
        while len(neighbors[current]) == 2:
            candidate = next(point for point in neighbors[current] if point != previous)
            edge = (current, candidate) if current < candidate else (candidate, current)
            if edge not in unused:
                break
            unused.remove(edge)
            path.append(candidate)
            previous, current = current, candidate
        return path

    paths: list[np.ndarray] = []
    for start in sorted(point for point, adjacent in neighbors.items() if len(adjacent) != 2):
        for following in sorted(neighbors[start]):
            edge = (start, following) if start < following else (following, start)
            if edge in unused:
                paths.append(
                    np.asarray(consume(start, following), dtype=np.float64) * 0.5
                )
    while unused:
        start, following = min(unused)
        paths.append(np.asarray(consume(start, following), dtype=np.float64) * 0.5)
    return paths


def _partition_boundary_overlay(
    layer_id: str,
    paths: Sequence[np.ndarray],
    *,
    color: str,
    width: float,
    opacity: float = 1.0,
    dash: str | None = None,
    hidden: bool = False,
    pre_smoothed: bool = False,
    river_paths: Sequence[np.ndarray] = (),
) -> str:
    dash_attribute = (
        f' stroke-dasharray="{dash}" data-screen-dash="{dash}"' if dash else ""
    )
    path_data = _path_data if pre_smoothed else _smooth_partition_path_data
    body = "".join(
        f'<path d="{path_data(path)}" fill="none" stroke="{color}" '
        f'stroke-width="{width:.2f}" stroke-linejoin="round" '
        f'stroke-linecap="round" data-screen-stroke="{width:.2f}"{dash_attribute} />'
        for path in paths
    )
    body += "".join(
        f'<path d="{_river_path_data(path)}" fill="none" stroke="{color}" '
        f'stroke-width="{width:.2f}" stroke-linejoin="round" '
        f'stroke-linecap="round" data-screen-stroke="{width:.2f}"{dash_attribute} '
        'data-boundary-source="river-axis" />'
        for path in river_paths
    )
    hidden_attribute = " hidden" if hidden else ""
    return (
        f'<g id="{layer_id}" aria-label="分区边界" opacity="{opacity:.2f}"'
        f'{hidden_attribute}>'
        f'{body}</g>'
    )


def _smooth_partition_path_data(points: np.ndarray) -> str:
    """Round a shared administrative polyline without moving its endpoints.

    The boundary graph is still extracted once from the categorical surface,
    so junctions remain topologically exact.  Midpoint quadratic segments only
    round interior grid corners; open endpoints (including three-way joins)
    stay fixed.  A separate seam seal in the state renderer covers the small
    difference between this cartographic ink line and the categorical fill.
    """

    path = np.asarray(points, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        return _path_data(path)
    cleaned = [path[0]]
    for point in path[1:]:
        if not np.allclose(point, cleaned[-1]):
            cleaned.append(point)
    if len(cleaned) < 3:
        return _path_data(np.asarray(cleaned, dtype=np.float64))

    compact = [cleaned[0]]
    for index, point in enumerate(cleaned[1:-1], start=1):
        previous = compact[-1]
        following = cleaned[index + 1]
        first = point - previous
        second = following - point
        cross = first[0] * second[1] - first[1] * second[0]
        if abs(cross) <= 1.0e-9 and float(np.dot(first, second)) >= 0.0:
            continue
        compact.append(point)
    compact.append(cleaned[-1])
    if len(compact) < 3:
        return _path_data(np.asarray(compact, dtype=np.float64))

    commands = [
        f"M{_compact_coordinate(compact[0][0])},{_compact_coordinate(compact[0][1])}"
    ]
    first_midpoint = 0.5 * (compact[0] + compact[1])
    commands.append(
        f"L{_compact_coordinate(first_midpoint[0])},{_compact_coordinate(first_midpoint[1])}"
    )
    for index in range(1, len(compact) - 1):
        midpoint = 0.5 * (compact[index] + compact[index + 1])
        commands.append(
            "Q"
            f"{_compact_coordinate(compact[index][0])},{_compact_coordinate(compact[index][1])} "
            f"{_compact_coordinate(midpoint[0])},{_compact_coordinate(midpoint[1])}"
        )
    commands.append(
        f"L{_compact_coordinate(compact[-1][0])},{_compact_coordinate(compact[-1][1])}"
    )
    return " ".join(commands)


def _state_boundary_overlay(
    paths: Sequence[np.ndarray],
    *,
    river_paths: Sequence[np.ndarray] = (),
    pre_smoothed: bool = False,
) -> str:
    """Render administrative arcs, substituting canonical river axes where used."""

    casing_width = 1.42
    ink_width = 0.70
    dash = "5.6 3.0"
    path_data = _path_data if pre_smoothed else _smooth_partition_path_data
    geometry = "".join(
        f'<path d="{path_data(path)}" fill="none" />'
        for path in paths
    )
    geometry += "".join(
        f'<path d="{_river_path_data(path)}" fill="none" '
        'data-boundary-source="river-axis" />'
        for path in river_paths
    )
    return (
        '<g id="state-boundaries" aria-label="国家边界" hidden>'
        f'<defs><g id="state-boundary-geometry">{geometry}</g></defs>'
        '<use href="#state-boundary-geometry" class="state-boundary-casing" '
        'fill="none" stroke="#f4eee1" stroke-opacity="0.44" '
        f'stroke-width="{casing_width:.2f}" data-screen-stroke="{casing_width:.2f}" '
        f'stroke-dasharray="{dash}" data-screen-dash="{dash}" '
        'stroke-linejoin="round" stroke-linecap="round" />'
        '<use href="#state-boundary-geometry" class="state-boundary-ink" '
        'fill="none" stroke="#46413b" stroke-opacity="0.88" '
        f'stroke-width="{ink_width:.2f}" data-screen-stroke="{ink_width:.2f}" '
        f'stroke-dasharray="{dash}" data-screen-dash="{dash}" '
        'stroke-linejoin="round" stroke-linecap="round" />'
        '</g>'
    )


def _frontier_overlay(
    paths: Sequence[tuple[np.ndarray, np.ndarray]],
) -> str:
    """Render ungoverned land as a quiet cartographic hatch."""

    if not paths:
        return '<g id="political-frontier" aria-label="部族地与无常设政权区"></g>'
    path_data = " ".join(
        _filled_path_data(points, codes) for points, codes in paths
    )
    return (
        '<defs><pattern id="frontier-hatch" width="12" height="12" '
        'patternUnits="userSpaceOnUse" patternTransform="rotate(28)">'
        '<path d="M0,0 V12" fill="none" stroke="#716b61" '
        'stroke-width="0.56" stroke-opacity="0.34" /></pattern></defs>'
        '<g id="political-frontier" aria-label="部族地与无常设政权区">'
        f'<path d="{path_data}" fill="#d8d2c5" fill-opacity="0.27" '
        'fill-rule="evenodd" stroke="none" />'
        f'<path d="{path_data}" fill="url(#frontier-hatch)" fill-rule="evenodd" '
        'stroke="none" />'
        '</g>'
    )


def _partition_label_anchors(
    values: np.ndarray,
) -> dict[int, tuple[int, int, int]]:
    """Return a pole-like interior anchor for every positive partition."""

    categories = np.asarray(values)
    current = np.where(categories > 0, categories, 0)
    depth = np.zeros(categories.shape, dtype=np.int16)
    while np.any(current > 0):
        active = current > 0
        depth[active] += 1
        interior = np.zeros(current.shape, dtype=bool)
        if current.shape[0] > 2 and current.shape[1] > 2:
            center = current[1:-1, 1:-1]
            interior[1:-1, 1:-1] = (
                (center > 0)
                & (center == current[:-2, 1:-1])
                & (center == current[2:, 1:-1])
                & (center == current[1:-1, :-2])
                & (center == current[1:-1, 2:])
            )
        current = np.where(interior, current, 0)

    anchors: dict[int, tuple[int, int, int]] = {}
    for identifier in sorted(
        int(item) for item in np.unique(categories) if item > 0
    ):
        candidates = np.argwhere(categories == identifier)
        if candidates.size == 0:
            original = np.argwhere(categories == identifier)
            if original.size:
                row, column = original[len(original) // 2]
                anchors[identifier] = (int(row), int(column), 0)
            continue
        candidate_depths = depth[candidates[:, 0], candidates[:, 1]]
        maximum_depth = int(candidate_depths.max(initial=0))
        deepest = candidates[candidate_depths == maximum_depth]
        center = candidates.mean(axis=0)
        selected = deepest[
            int(np.argmin(np.sum(np.square(deepest - center), axis=1)))
        ]
        anchors[identifier] = (
            int(selected[0]),
            int(selected[1]),
            maximum_depth,
        )
    return anchors


def _split_transport_path(
    points: tuple[tuple[float, float], ...],
    width: int,
) -> tuple[np.ndarray, ...]:
    """Split wrapped routes at the longitude seam instead of drawing across the world."""

    segments: list[list[tuple[float, float]]] = [[points[0]]]
    for previous, current in zip(points, points[1:]):
        if abs(current[0] - previous[0]) > width * 0.5:
            segments.append([current])
        else:
            segments[-1].append(current)
    return tuple(
        np.asarray(segment, dtype=np.float64)
        for segment in segments
        if len(segment) >= 2
    )


def _transport_path_data(mode: str, segment: np.ndarray) -> str:
    """Preserve verified road/sea corridors; smooth only literal river routes."""

    return _river_path_data(segment) if mode == "river" else _path_data(segment)


def _transport_overlay(society: SocietyLayers, *, width: int) -> str:
    styles = {
        "road": ("#8b5e3c", None, "#f5eddc", 0.85),
        "river": ("#246da0", "5 3", "#dbe8ef", 0.52),
        "sea": ("#294f75", "8 6", "#7595b5", 0.48),
    }
    widths = {"trunk": 1.35, "regional": 0.88, "local": 0.52}
    parts = ['<g id="transport-network" aria-label="城市交通网络" hidden>']
    for route in society.transport.routes:
        color, dash, casing, casing_opacity = styles[route.mode]
        base_width = widths[route.importance]
        for segment in _split_transport_path(route.path, width):
            # Roads and sea lanes must stay inside their verified cell
            # corridors.  A tangent curve can cut across the inside of a bend
            # and visibly cross a ridge, cape, or island.  Only routes that
            # literally follow a river may use cartographic curve smoothing.
            path_data = _transport_path_data(route.mode, segment)
            parts.append(
                f'<path d="{path_data}" fill="none" stroke="{casing}" '
                f'stroke-width="{base_width + (0.85 if route.mode == "road" else 0.42):.2f}" '
                f'stroke-opacity="{casing_opacity:.2f}" '
                f'data-screen-stroke="{base_width + (0.85 if route.mode == "road" else 0.42):.2f}" '
                f'data-route-mode="{route.mode}" data-route-importance="{route.importance}" '
                'data-route-casing="true" stroke-linecap="round" stroke-linejoin="round" />'
            )
            parts.append(
                f'<path d="{path_data}" fill="none" stroke="{color}" '
                f'stroke-width="{base_width:.2f}" stroke-opacity="0.88" '
                f'{f"stroke-dasharray=\"{dash}\" " if dash else ""}'
                'stroke-linecap="round" stroke-linejoin="round" '
                f'data-screen-stroke="{base_width:.2f}" '
                f'{f"data-screen-dash=\"{dash}\" " if dash else ""}'
                f'data-route-mode="{route.mode}" data-route-importance="{route.importance}" />'
            )
    parts.append("</g>")
    return "".join(parts)


def _bridge_overlay(society: SocietyLayers, *, width: int) -> str:
    """Draw compact bridge rails aligned to the crossed road tangent."""

    route_by_identifier = {
        route.identifier: route for route in society.transport.routes
    }
    parts = [
        '<g id="bridge-layer" data-layer="bridges" '
        'aria-label="道路跨河桥梁" hidden>'
    ]
    for bridge in society.transport.bridges:
        route = route_by_identifier[bridge.route_identifier]
        centre = np.asarray((bridge.column + 0.5, bridge.row + 0.5), dtype=np.float64)
        points = np.asarray(route.path, dtype=np.float64).copy()
        points[:, 0] = centre[0] + (
            (points[:, 0] - centre[0] + width * 0.5) % width - width * 0.5
        )
        best_distance = math.inf
        tangent = np.asarray((1.0, 0.0), dtype=np.float64)
        for first, second in zip(points[:-1], points[1:], strict=True):
            delta = second - first
            length_squared = float(np.dot(delta, delta))
            if length_squared <= 1.0e-12:
                continue
            fraction = float(
                np.clip(np.dot(centre - first, delta) / length_squared, 0.0, 1.0)
            )
            distance = float(np.linalg.norm(centre - (first + fraction * delta)))
            if distance < best_distance:
                best_distance = distance
                tangent = delta
        angle = math.degrees(math.atan2(float(tangent[1]), float(tangent[0])))
        x = float(centre[0])
        y = float(centre[1])
        parts.append(
            f'<g class="bridge-symbol" transform="translate({x:.2f} {y:.2f}) '
            f'rotate({angle:.2f})" data-map-x="{x:.2f}" data-map-y="{y:.2f}" '
            f'data-base-angle="{angle:.2f}" data-bridge-id="{html.escape(bridge.identifier)}" '
            f'data-route-id="{html.escape(bridge.route_identifier)}" '
            f'data-bridge-importance="{bridge.importance}" '
            f'data-river-order="{bridge.river_order}">'
            '<path d="M-0.9,-1.75 L-0.9,1.75 M0.9,-1.75 L0.9,1.75" '
            'fill="none" stroke="#f7f0df" stroke-width="1.18" '
            'stroke-linecap="round" />'
            '<path d="M-0.9,-1.75 L-0.9,1.75 M0.9,-1.75 L0.9,1.75" '
            'fill="none" stroke="#5f4938" stroke-width="0.46" '
            'stroke-linecap="round" />'
            '</g>'
        )
    parts.append('</g>')
    return "".join(parts)


def _territorial_quality_metrics(
    grid: WorldGrid,
    thematic: ThematicLayers,
    society: SocietyLayers,
) -> dict[str, int | float]:
    """Measure whether administrative geometry is explained by the world."""

    route_affinity = _coarse_route_affinity(
        society.transport,
        grid.shape,
        step=1,
    )
    state_transitions = _state_transition_penalties(
        np.asarray(grid.elevation),
        np.asarray(thematic.drainage_basin),
        np.asarray(grid.river_order),
        np.zeros(grid.shape, dtype=np.int16),
        route_affinity,
    )
    state_alignment = boundary_alignment(
        society.politics.state_id,
        state_transitions,
    )
    state_straight = longest_unjustified_straight_run(
        society.politics.state_id,
        state_transitions,
    )
    del state_transitions, route_affinity

    road_corridor, _road_junction = road_network_fields(
        grid.shape,
        society.transport.routes,
    )
    province_transitions = _province_transition_penalties(
        grid,
        thematic,
        road_corridor,
    )
    province_alignment = boundary_alignment(
        society.provinces.province_id,
        province_transitions,
    )
    province_straight = longest_unjustified_straight_run(
        society.provinces.province_id,
        province_transitions,
    )
    del province_transitions, road_corridor

    human_domain = society_domain_mask(grid)
    extreme = extreme_frontier_environment(
        human_domain,
        grid.snow,
        grid.elevation,
        thematic.climate.annual_precipitation,
        thematic.land_potential,
        society.population.population_band,
        grid.river_order,
    )
    route_cells = {
        route.identifier: set(_path_cells(route.path, grid.shape))
        for route in society.transport.routes
        if route.mode == "road"
    }
    invalid_bridges = sum(
        int(grid.river_order[bridge.row, bridge.column]) <= 0
        or (bridge.row, bridge.column)
        not in route_cells.get(bridge.route_identifier, set())
        for bridge in society.transport.bridges
    )
    return {
        "stateBoundaryAlignment": float(state_alignment),
        "provinceBoundaryAlignment": float(province_alignment),
        "ordinaryFrontierComponents": ordinary_frontier_components(
            society.politics.state_id,
            human_domain,
            extreme,
            maximum_size=512,
        ),
        "stateStraightRunMaximum": int(state_straight),
        "provinceStraightRunMaximum": int(province_straight),
        "bridgesOffRoadOrRiver": int(invalid_bridges),
    }


def _city_overlay(society: SocietyLayers) -> str:
    symbol = {
        "metropolis": (3.70, "#263447", "#f0cf83", 0.82),
        "city": (2.15, "#33475d", "#f7eed7", 0.62),
        "town": (1.05, "#435467", "#f7eed7", 0.42),
        "site": (1.85, "#593f31", "#d7b978", 0.62),
    }
    label_size = {"metropolis": 13.2, "city": 10.5, "town": 9.0, "site": 9.2}
    capital_ids = {item.core_settlement_id for item in society.politics.states}
    parts = ['<g id="city-layer" aria-label="分级城市、首都与圣城" hidden>']
    parts.append('<g id="city-symbols">')
    for settlement in society.settlements:
        radius, stroke, fill, stroke_width = symbol[settlement.tier]
        x = settlement.column + 0.5
        y = settlement.row + 0.5
        parts.append(
            f'<g transform="translate({x:.2f} {y:.2f})" '
            f'data-city-symbol-tier="{settlement.tier}" '
            f'data-map-x="{x:.2f}" data-map-y="{y:.2f}" '
            f'data-national-capital="{str(settlement.identifier in capital_ids).lower()}" '
            f'data-holy-city="{str(settlement.holy_religion_identifier is not None).lower()}">'
        )
        if settlement.holy_religion_identifier is not None:
            holy_radius = radius * 1.85
            inner = holy_radius * 0.43
            star_points = []
            for index in range(16):
                angle = -math.pi / 2.0 + index * math.pi / 8.0
                active_radius = holy_radius if index % 2 == 0 else inner
                star_points.append(
                    f"{math.cos(angle) * active_radius:.2f},{math.sin(angle) * active_radius:.2f}"
                )
            parts.append(
                f'<polygon points="{" ".join(star_points)}" fill="#f6d889" '
                'stroke="#7f493b" stroke-width="0.72" '
                f'data-holy-city="true" data-religion-id="{settlement.holy_religion_identifier}" />'
            )
        if settlement.site_type == "pass":
            parts.append(
                f'<rect x="{-radius:.2f}" y="{-radius:.2f}" width="{radius * 2:.2f}" '
                f'height="{radius * 2:.2f}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{stroke_width:.2f}" transform="rotate(45)" '
                f'data-city-tier="site" data-site-type="pass" />'
            )
        elif settlement.site_type == "fortress":
            parts.append(
                f'<path d="M{-radius:.2f},{-radius:.2f} L{radius:.2f},{-radius:.2f} '
                f'L{radius:.2f},{radius * 0.55:.2f} L0,{radius:.2f} '
                f'L{-radius:.2f},{radius * 0.55:.2f} Z" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{stroke_width:.2f}" '
                f'data-city-tier="site" '
                'data-site-type="fortress" />'
            )
        else:
            parts.append(
                f'<circle cx="0" cy="0" r="{radius:.2f}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{stroke_width:.2f}" '
                f'data-city-tier="{settlement.tier}" '
                f'data-site-type="{settlement.site_type}" />'
            )
        if settlement.tier == "metropolis":
            parts.append(
                '<circle cx="0" cy="0" r="1.05" fill="#263447" '
                'stroke="none" aria-hidden="true" />'
            )
        if settlement.identifier in capital_ids:
            parts.append(
                f'<circle cx="0" cy="0" r="{radius * 1.42:.2f}" fill="none" '
                'stroke="#263447" stroke-width="0.72" data-national-capital="true" />'
            )
        parts.append('</g>')
    parts.append('</g><g id="city-labels" data-capital-font-weight="750">')
    for settlement in society.settlements:
        size = label_size[settlement.tier]
        offset = {"metropolis": 5.6, "city": 4.0, "town": 2.7, "site": 3.4}[settlement.tier]
        minimum_screen_size = {
            "metropolis": 10.2,
            "city": 8.9,
            "town": 8.0,
            "site": 8.2,
        }[settlement.tier]
        parts.append(
            f'<text x="{settlement.column + 0.5 + offset:.2f}" '
            f'y="{settlement.row + 0.5 - offset * 0.35:.2f}" '
            f'font-size="{size:.2f}" fill="#263447" font-weight="'
            f'{"750" if settlement.identifier in capital_ids else "700" if settlement.tier == "metropolis" else "500"}" '
            'paint-order="stroke" stroke="#f7f0df" stroke-width="1.25" '
            'stroke-linejoin="round" '
            f'data-city-label-tier="{settlement.tier}" '
            f'data-national-capital="{str(settlement.identifier in capital_ids).lower()}" '
            f'data-holy-city="{str(settlement.holy_religion_identifier is not None).lower()}" '
            f'data-map-x="{settlement.column + 0.5:.2f}" '
            f'data-map-y="{settlement.row + 0.5:.2f}" '
            f'data-base-offset-x="{offset:.2f}" '
            f'data-base-offset-y="{-offset * 0.35:.2f}" '
            f'data-base-font-size="{size:.2f}" '
            f'data-min-screen-font-size="{minimum_screen_size:.2f}" '
            'data-base-text-stroke="1.25">'
            f'{html.escape(settlement.name)}</text>'
        )
    parts.append("</g></g>")
    return "".join(parts)


def _orient_label_path(points: np.ndarray) -> np.ndarray:
    """Keep SVG path text upright while preserving the physical trace."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return values
    delta = values[-1] - values[0]
    if delta[0] < -1.0e-9 or (abs(delta[0]) <= 1.0e-9 and delta[1] < 0.0):
        return values[::-1].copy()
    return values


def _smooth_text_path(points: np.ndarray, *, sample_count: int = 15) -> np.ndarray:
    """Return a gently sampled baseline so adjacent glyphs cannot fold together."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 4:
        return values
    smoothed = values.copy()
    # River centerlines contain cell-scale elbows that are useful for the
    # hydrography itself but hostile to text-on-path: two adjacent glyphs can
    # rotate towards each other at a sharp bend.  Repeated low-pass smoothing
    # retains the reach's broad direction while removing those local elbows.
    for _pass in range(9):
        next_values = smoothed.copy()
        next_values[1:-1] = (
            smoothed[:-2] + 2.0 * smoothed[1:-1] + smoothed[2:]
        ) / 4.0
        smoothed = next_values
    lengths = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1.0e-9:
        return smoothed
    distances = np.linspace(0.0, float(cumulative[-1]), max(5, sample_count))
    sampled = np.column_stack(
        (
            np.interp(distances, cumulative, smoothed[:, 0]),
            np.interp(distances, cumulative, smoothed[:, 1]),
        )
    )
    return sampled


def _mountain_label_path(grid: WorldGrid, feature: Any) -> np.ndarray | None:
    """Fit a gently curved label baseline to the local highland long axis."""

    radius = 92 if feature.tier == "major" else 68
    row_min = max(0, feature.row - radius)
    row_max = min(grid.shape[0], feature.row + radius + 1)
    row_values = np.arange(row_min, row_max, dtype=np.int32)
    column_offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    column_values = (feature.column + column_offsets) % grid.shape[1]
    local = grid.elevation[np.ix_(row_values, column_values)].astype(np.float64)
    local_land = grid.water[np.ix_(row_values, column_values)] == 0
    dy, dx = np.meshgrid(
        row_values.astype(np.float64) - float(feature.row),
        column_offsets.astype(np.float64),
        indexing="ij",
    )
    distance = np.hypot(dx, dy)
    available = local[local_land & (distance <= radius)]
    if available.size < 12:
        return None
    threshold = max(0.48, float(np.quantile(available, 0.70)))
    ridge = local_land & (distance <= radius) & (local >= threshold)
    if np.count_nonzero(ridge) < 12:
        ridge = local_land & (distance <= radius) & (local >= 0.44)
    if np.count_nonzero(ridge) < 8:
        return None

    selected_x = dx[ridge]
    selected_y = dy[ridge]
    selected_height = local[ridge]
    weights = np.square(np.clip(selected_height - threshold + 0.06, 0.015, None))
    weights *= np.exp(-np.square(distance[ridge] / max(1.0, radius * 0.92)))
    weight_sum = float(weights.sum())
    if weight_sum <= 1.0e-12:
        return None
    center_x = float(np.sum(selected_x * weights) / weight_sum)
    center_y = float(np.sum(selected_y * weights) / weight_sum)
    centered = np.column_stack((selected_x - center_x, selected_y - center_y))
    covariance = (centered * weights[:, None]).T @ centered / weight_sum
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if axis[0] < 0.0 or (abs(axis[0]) < 1.0e-9 and axis[1] < 0.0):
        axis = -axis
    perpendicular = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    projection = centered @ axis
    minimum_half_length = max(34.0, len(feature.name) * (7.4 if feature.tier == "major" else 6.0))
    half_length = min(
        radius * 0.86,
        max(minimum_half_length, float(np.quantile(np.abs(projection), 0.84))),
    )
    strip_width = max(7.0, half_length * 0.18)
    points: list[tuple[float, float]] = []
    for position in np.linspace(-half_length, half_length, 7):
        along_delta = np.abs(projection - position)
        across_delta = np.abs(centered @ perpendicular)
        candidates = (along_delta <= max(5.0, half_length / 7.0)) & (
            across_delta <= strip_width
        )
        if np.any(candidates):
            candidate_indices = np.flatnonzero(candidates)
            score = (
                selected_height[candidate_indices]
                - 0.004 * across_delta[candidate_indices]
                - 0.0015 * along_delta[candidate_indices]
            )
            best = int(candidate_indices[int(np.argmax(score))])
            point_x = feature.column + selected_x[best]
            point_y = feature.row + selected_y[best]
        else:
            point_x = feature.column + center_x + axis[0] * position
            point_y = feature.row + center_y + axis[1] * position
        points.append((point_x + 0.5, point_y + 0.5))
    values = np.asarray(points, dtype=np.float64)
    if np.any(values[:, 0] < -radius) or np.any(values[:, 0] > grid.shape[1] + radius):
        return None
    return _orient_label_path(values)


def _river_label_paths(grid: WorldGrid, features: Sequence[Any]) -> dict[str, np.ndarray]:
    """Trace each named river along its dominant upstream/downstream stem."""

    river_features = [feature for feature in features if feature.feature_type == "river"]
    if not river_features:
        return {}
    active = grid.river_order.reshape(-1) > 0
    downstream = grid.flow_to.reshape(-1)
    discharge = grid.discharge.reshape(-1).astype(np.float64)
    primary_upstream = np.full(active.size, -1, dtype=np.int32)
    for source in np.flatnonzero(active):
        target = int(downstream[source])
        if 0 <= target < active.size and active[target]:
            previous = int(primary_upstream[target])
            if previous < 0 or discharge[source] > discharge[previous]:
                primary_upstream[target] = int(source)

    def nearest_river(row: int, column: int) -> int | None:
        candidates: list[tuple[int, float, float, int]] = []
        for dy in range(-10, 11):
            next_row = row + dy
            if next_row < 0 or next_row >= grid.shape[0]:
                continue
            for dx in range(-10, 11):
                next_column = (column + dx) % grid.shape[1]
                flat = next_row * grid.shape[1] + next_column
                if not active[flat]:
                    continue
                distance = math.hypot(dy, dx)
                candidates.append(
                    (
                        int(grid.river_order[next_row, next_column] >= 3),
                        discharge[flat] / (1.0 + distance),
                        -distance,
                        flat,
                    )
                )
        return max(candidates)[-1] if candidates else None

    def trace(start: int, *, upstream: bool, distance_limit: float) -> list[int]:
        cells = [start]
        seen = {start}
        travelled = 0.0
        current = start
        while travelled < distance_limit:
            next_cell = (
                int(primary_upstream[current])
                if upstream
                else int(downstream[current])
            )
            if next_cell < 0 or next_cell >= active.size or not active[next_cell] or next_cell in seen:
                break
            row, column = divmod(current, grid.shape[1])
            next_row, next_column = divmod(next_cell, grid.shape[1])
            dx = abs(next_column - column)
            travelled += math.hypot(next_row - row, min(dx, grid.shape[1] - dx))
            cells.append(next_cell)
            seen.add(next_cell)
            current = next_cell
        return cells

    result: dict[str, np.ndarray] = {}
    for feature in river_features:
        start = nearest_river(feature.row, feature.column)
        if start is None:
            continue
        half_length = (
            max(28.0, len(feature.name) * 4.8)
            if feature.tier == "detail"
            else max(
                44.0,
                len(feature.name) * (6.8 if feature.tier == "major" else 5.8),
            )
        )
        upstream = trace(start, upstream=True, distance_limit=half_length)
        downstream_cells = trace(start, upstream=False, distance_limit=half_length)
        cells = list(reversed(upstream)) + downstream_cells[1:]
        if len(cells) < 3:
            continue
        points: list[tuple[float, float]] = []
        previous_x: float | None = None
        for cell in cells:
            row, column = divmod(cell, grid.shape[1])
            x = column + 0.5
            if previous_x is not None:
                while x - previous_x > grid.shape[1] * 0.5:
                    x -= grid.shape[1]
                while previous_x - x > grid.shape[1] * 0.5:
                    x += grid.shape[1]
            points.append((x, row + 0.5))
            previous_x = x
        values = np.asarray(points, dtype=np.float64)
        mean_x = float(values[:, 0].mean())
        if mean_x < 0.0:
            values[:, 0] += grid.shape[1]
        elif mean_x > grid.shape[1]:
            values[:, 0] -= grid.shape[1]
        if np.any(values[:, 0] < -12.0) or np.any(values[:, 0] > grid.shape[1] + 12.0):
            continue
        smoothed = _smooth_text_path(
            values,
            sample_count=(
                max(9, min(13, len(feature.name) * 2 + 3))
                if feature.tier == "detail"
                else max(23, min(33, len(feature.name) * 5 + 7))
            ),
        )
        result[feature.identifier] = _orient_label_path(smoothed)
    return result


def _toponymy_overlay(grid: WorldGrid, society: SocietyLayers) -> str:
    styles = {
        "mountain": ("#564a43", 14.0, "letter-spacing:1.2px"),
        "peak": ("#5f5148", 9.8, "letter-spacing:0.35px"),
        "river": ("#2f6f9f", 11.5, "font-style:italic;letter-spacing:2.40px"),
        "lake": ("#2f608e", 12.0, ""),
        "sea": ("#315f88", 16.0, "letter-spacing:2px;font-style:italic"),
        "inland-sea": ("#2f608e", 15.0, "letter-spacing:1.4px"),
        "bay": ("#426d91", 11.0, ""),
        "strait": ("#426d91", 10.5, ""),
        "island": ("#584f46", 10.5, ""),
        "island-group": ("#584f46", 11.0, ""),
        "plain": ("#596b46", 13.0, "letter-spacing:1px"),
        "plateau": ("#745d42", 13.0, "letter-spacing:1px"),
        "basin": ("#6d6047", 12.0, ""),
    }

    def scale_window(feature: Any) -> tuple[float, float]:
        minimums = {
            "major": {
                "mountain": 0.0,
                "river": 0.0,
                "sea": 0.0,
                "inland-sea": 0.0,
                "plain": 0.0,
                "plateau": 0.0,
                "basin": 0.0,
                "lake": 0.70,
                "island": 1.05,
                "island-group": 0.0,
            },
            "secondary": {
                "sea": 1.05,
                "inland-sea": 0.70,
                "island": 1.20,
                "island-group": 0.75,
                "mountain": 1.35,
                "plain": 1.45,
                "plateau": 1.45,
                "basin": 1.45,
                "lake": 1.45,
                "river": 1.65,
            },
            "detail": {
                "plain": 2.20,
                "plateau": 2.20,
                "basin": 2.20,
                "island": 2.30,
                "island-group": 1.60,
                "peak": 2.40,
                "bay": 2.50,
                "strait": 2.50,
                "lake": 2.60,
                "river": 3.10,
            },
        }
        maximum = 99.0
        if feature.feature_type == "mountain" and feature.tier == "major":
            maximum = 3.00
        elif feature.feature_type == "mountain" and feature.tier == "secondary":
            maximum = 6.20
        elif feature.feature_type in {"sea", "inland-sea"} and feature.tier == "major":
            maximum = 4.20
        elif feature.feature_type == "river" and feature.tier == "major":
            maximum = 6.40
        elif feature.feature_type == "river" and feature.tier == "secondary":
            maximum = 10.50
        elif feature.feature_type in {"plain", "plateau", "basin"}:
            maximum = 6.80 if feature.tier == "major" else 9.20
        minimum = minimums[feature.tier].get(feature.feature_type, 1.60)
        return minimum, maximum
    curved_paths: dict[str, np.ndarray] = _river_label_paths(
        grid,
        society.geographic_features,
    )
    detail_river_angles: dict[str, float] = {}
    for feature in society.geographic_features:
        if feature.feature_type != "river" or feature.tier != "detail":
            continue
        path = curved_paths.pop(feature.identifier, None)
        if path is None or path.shape[0] < 2:
            continue
        delta = path[-1] - path[0]
        angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0
        detail_river_angles[feature.identifier] = angle
    for feature in society.geographic_features:
        if feature.feature_type == "mountain":
            path = _mountain_label_path(grid, feature)
            if path is not None:
                curved_paths[feature.identifier] = path
    definitions = [
        f'<path id="geographic-path-{html.escape(identifier, quote=True)}" '
        f'd="{_river_path_data(path)}" fill="none" />'
        for identifier, path in curved_paths.items()
    ]
    parts = [
        '<g id="geographic-labels" aria-label="世界尺度地理名称">',
        f'<defs>{"".join(definitions)}</defs>',
    ]
    for feature in society.geographic_features:
        color, size, style = styles[feature.feature_type]
        opacity = {"major": "0.92", "secondary": "0.78", "detail": "0.74"}[
            feature.tier
        ]
        minimum_screen_size = {
            "mountain": 10.0,
            "river": 9.2,
            "inland-sea": 10.5,
        }.get(feature.feature_type, 8.8)
        maximum_screen_size = {
            "mountain": 40.0,
            "sea": 42.0,
            "inland-sea": 40.0,
            "river": 32.0,
            "island": 30.0,
            "island-group": 32.0,
        }.get(feature.feature_type, 30.0)
        minimum_visible_scale, maximum_visible_scale = scale_window(feature)
        common = (
            f'fill="{color}" font-size="{size:.2f}" text-anchor="middle" '
            'font-family="Noto Serif CJK SC,Source Han Serif SC,STSong,SimSun,serif" '
            f'paint-order="stroke" stroke="#f4efdf" stroke-width="1.80" '
            f'stroke-linejoin="round" opacity="{opacity}" style="{style}" '
            f'data-feature-type="{feature.feature_type}" '
            f'data-feature-tier="{feature.tier}" '
            f'data-base-font-size="{size:.2f}" '
            f'data-min-screen-font-size="{minimum_screen_size:.2f}" '
            f'data-max-screen-font-size="{maximum_screen_size:.2f}" '
            f'data-min-visible-scale="{minimum_visible_scale:.2f}" '
            f'data-max-visible-scale="{maximum_visible_scale:.2f}" '
            'data-base-text-stroke="1.80"'
        )
        if feature.identifier in detail_river_angles:
            x = feature.column + 0.5
            y = feature.row + 0.5
            parts.append(
                f'<text x="{x:.2f}" y="{y:.2f}" {common} '
                f'transform="rotate({detail_river_angles[feature.identifier]:.2f} {x:.2f} {y:.2f})" '
                'data-local-river-label="true">'
                f'{html.escape(feature.name)}</text>'
            )
        elif feature.identifier in curved_paths:
            path_identifier = html.escape(feature.identifier, quote=True)
            text_length = (
                max(size * 2.0, len(feature.name) * size * 1.34)
                if feature.feature_type == "river"
                else None
            )
            spacing_attributes = (
                f' textLength="{text_length:.2f}" lengthAdjust="spacing"'
                f' data-base-text-length="{text_length:.2f}"'
                if text_length is not None
                else ""
            )
            parts.append(
                f'<text {common} dy="{-2.0 if feature.feature_type == "mountain" else -2.6:.1f}" '
                'data-curved-label="true">'
                f'<textPath href="#geographic-path-{path_identifier}" startOffset="50%"'
                f'{spacing_attributes}>'
                f'{html.escape(feature.name)}</textPath></text>'
            )
        else:
            parts.append(
                f'<text x="{feature.column + 0.5:.2f}" y="{feature.row + 0.5:.2f}" '
                f'{common}>{html.escape(feature.name)}</text>'
            )
    parts.append("</g>")
    return "".join(parts)


def _political_label_overlay(society: SocietyLayers) -> str:
    parts = ['<g id="state-labels" aria-label="国家名称" hidden>']
    labels = society.politics.state_id
    anchors = _partition_label_anchors(labels)
    entity_by_country = {
        item.country_identifier: item for item in society.politics.political_entities
    }
    government_by_identifier = {
        item.identifier: item for item in society.politics.government_forms
    }
    size_rank = {"major": 0, "medium": 1, "small": 2}
    for state in sorted(
        society.politics.states,
        key=lambda item: (
            size_rank[item.size_class],
            -item.population_max,
            item.identifier,
        ),
    ):
        rows, columns = np.nonzero(labels == state.identifier)
        if rows.size == 0:
            continue
        anchor_row, anchor_column, _depth = anchors[state.identifier]
        label_row = float(anchor_row) + 0.5
        label_column = float(anchor_column) + 0.5
        area_size = math.sqrt(float(rows.size)) * 0.28
        minimum, maximum = {
            "major": (12.0, 18.0),
            "medium": (8.0, 11.5),
            "small": (5.2, 7.4),
        }[state.size_class]
        size = float(np.clip(area_size, minimum, maximum))
        minimum_screen_size = {"major": 14.0, "medium": 12.0, "small": 10.5}[
            state.size_class
        ]
        maximum_screen_size = {"major": 23.0, "medium": 19.0, "small": 15.5}[
            state.size_class
        ]
        entity = entity_by_country[state.identifier]
        government = government_by_identifier[entity.government_form_identifier]
        parts.append(
            f'<text x="{label_column:.2f}" y="{label_row:.2f}" '
            f'fill="#2f3948" font-size="{size:.2f}" text-anchor="middle" '
            'dominant-baseline="middle" font-weight="600" letter-spacing="0.35" '
            'font-family="Noto Serif CJK SC,Source Han Serif SC,STSong,SimSun,serif" '
            'paint-order="stroke" '
            'stroke="#f7f1e5" stroke-width="0.82" stroke-opacity="0.90" '
            f'data-state-name="{html.escape(state.name, quote=True)}" '
            f'data-government-form="{html.escape(government.name, quote=True)}" '
            f'data-formal-name="{html.escape(entity.formal_name, quote=True)}" '
            f'data-state-size="{state.size_class}" data-base-font-size="{size:.2f}" '
            f'data-min-screen-font-size="{minimum_screen_size:.2f}" '
            f'data-max-screen-font-size="{maximum_screen_size:.2f}" '
            'data-base-text-stroke="0.82">'
            f'{html.escape(entity.formal_name)}</text>'
        )
    parts.append("</g>")
    return "".join(parts)


def _province_label_overlay(society: SocietyLayers) -> str:
    settlements = {item.identifier: item for item in society.settlements}
    labels = society.provinces.province_id
    anchors = _partition_label_anchors(labels)
    function_labels = {
        "capital": "首都辖域",
        "civil": "民政",
        "military": "军政",
        "frontier": "边疆",
        "maritime": "海政",
        "vassal": "封臣领",
        "crown": "王冠直领",
        "pastoral": "牧地",
        "civic": "城市自治",
    }
    rank_labels = {
        "capital-district": "首都直辖",
        "royal-seat": "王廷直领",
        "crown-domain": "王冠领",
        "metropolitan-district": "都城辖地",
        "court-domain": "盟庭直领",
        "estate-seat": "等级王廷辖域",
        "duchy": "公爵等第",
        "march": "边侯等第",
        "county": "伯爵等第",
        "viscounty": "子爵等第",
        "barony": "男爵等第",
        "civil-prefecture": "民政层级",
        "military-command": "戍防行政层级",
        "frontier-command": "边疆行政层级",
        "maritime-circuit": "海政层级",
        "defence-district": "巡防层级",
        "naval-trade-district": "海商层级",
        "civic-district": "城市自治层级",
        "frontier-wing": "边翼层级",
        "clan-territory": "宗族层级",
        "tribal-territory": "部族层级",
        "service-march": "军役边区",
        "chartered-port": "特许港辖",
        "estate-district": "身份辖地",
    }
    parts = ['<g id="province-labels" aria-label="省份名称" hidden>']
    for province in society.provinces.provinces:
        core = settlements.get(province.core_settlement_id)
        if (
            core is None
            or not np.any(labels == province.identifier)
            or province.identifier not in anchors
        ):
            continue
        anchor_row, anchor_column, _depth = anchors[province.identifier]
        size = (
            8.0
            if province.administrative_function == "capital"
            else 7.5
            if province.administrative_rank in {"duchy", "march", "frontier-command"}
            else 7.0
            if core.tier in {"metropolis", "city"}
            else 6.4
        )
        accessible_label = (
            f"{province.name}，{rank_labels[province.administrative_rank]}，"
            f"{function_labels[province.administrative_function]}，"
            f"{ {'dense': '人口稠密', 'settled': '人口常住', 'sparse': '人口稀疏'}[province.population_density_class] }"
        )
        parts.append(
            f'<text x="{anchor_column + 0.5:.2f}" y="{anchor_row + 0.5:.2f}" '
            'fill="#3f4650" text-anchor="middle" dominant-baseline="middle" '
            'font-weight="560" '
            f'aria-label="{html.escape(accessible_label, quote=True)}" '
            'letter-spacing="0.24" paint-order="stroke" '
            'stroke="#f8f2e6" stroke-width="0.78" stroke-opacity="0.92" '
            f'font-size="{size:.2f}" data-province-id="{province.identifier}" '
            f'data-parent-state="{province.state_identifier}" '
            f'data-province-type="{province.region_type}" '
            f'data-administrative-system="{province.administrative_system}" '
            f'data-administrative-function="{province.administrative_function}" '
            f'data-administrative-rank="{province.administrative_rank}" '
            f'data-population-density="{province.population_density_class}" '
            f'data-area-cells="{province.area_cells}" '
            f'data-base-font-size="{size:.2f}" data-min-screen-font-size="9.8" '
            'data-max-screen-font-size="14.5" data-base-text-stroke="0.78">'
            f'<title>{html.escape(accessible_label)}</title>'
            f'{html.escape(province.name)}</text>'
        )
    parts.append("</g>")
    return "".join(parts)


def _frontier_group_label_overlay(society: SocietyLayers) -> str:
    """Render mobile peoples as labels only, never as claimed polygons."""

    livelihood_labels = {
        "pastoral": "游牧",
        "foraging": "渔猎",
        "maritime": "海洋",
        "agropastoral": "半农半牧",
        "highland": "山地",
        "riverine": "河湖",
    }
    parts = ['<g id="frontier-group-labels" aria-label="无常设政权区当地群体" hidden>']
    tier_rank = {"major": 0, "secondary": 1}
    for group in sorted(
        society.politics.frontier_groups,
        key=lambda item: (tier_rank[item.tier], item.identifier),
    ):
        size = 10.2 if group.tier == "major" else 8.6
        minimum_screen_size = 11.5 if group.tier == "major" else 10.5
        maximum_screen_size = 17.0 if group.tier == "major" else 14.5
        parts.append(
            f'<text x="{group.column + 0.5:.2f}" y="{group.row + 0.5:.2f}" '
            'fill="#4f4538" text-anchor="middle" font-weight="560" '
            'font-style="italic" letter-spacing="0.28" paint-order="stroke" '
            'stroke="#f8f1df" stroke-width="1.05" stroke-opacity="0.94" '
            f'font-size="{size:.2f}" '
            f'data-frontier-group="{html.escape(group.identifier, quote=True)}" '
            f'data-frontier-tier="{group.tier}" '
            f'data-livelihood="{group.livelihood}" '
            f'data-livelihood-label="{livelihood_labels[group.livelihood]}" '
            f'data-organization="{group.organization}" '
            f'data-language="{group.language_identifier}" '
            f'data-civilization="{group.civilization_identifier}" '
            f'data-base-font-size="{size:.2f}" '
            f'data-min-screen-font-size="{minimum_screen_size:.2f}" '
            f'data-max-screen-font-size="{maximum_screen_size:.2f}" '
            'data-base-text-stroke="1.05">'
            f'{html.escape(group.name)}</text>'
        )
    parts.append("</g>")
    return "".join(parts)


def _culture_label_overlay(society: SocietyLayers, *, kind: str) -> str:
    settlements = {item.identifier: item for item in society.settlements}
    items = (
        society.cultures.civilizations
        if kind == "civilization"
        else society.cultures.languages
    )
    parts = [f'<g id="{kind}-labels" aria-label="{kind}名称" hidden>']
    for item in items:
        core = settlements[item.core_settlement_id]
        parts.append(
            f'<text x="{core.column + 0.5:.2f}" y="{core.row + 0.5:.2f}" '
            'fill="#263447" font-size="13.0" text-anchor="middle" '
            'paint-order="stroke" stroke="#f5f0e6" stroke-width="1.45" '
            f'data-{kind}-name="{html.escape(item.name, quote=True)}" '
            'data-theme-label="true" data-base-font-size="13.0" '
            'data-min-screen-font-size="10.0" '
            'data-base-text-stroke="1.45">'
            f'{html.escape(item.name)}</text>'
        )
    parts.append("</g>")
    return "".join(parts)


def _religion_label_overlay(society: SocietyLayers) -> str:
    settlements = {item.identifier: item for item in society.settlements}
    parts = ['<g id="religion-labels" aria-label="宗教名称" hidden>']
    for item in society.religions.religions:
        holy = settlements[item.holy_settlement_id]
        parts.append(
            f'<text x="{holy.column + 0.5:.2f}" y="{holy.row - 5.0:.2f}" '
            'fill="#613f3a" font-size="12.4" text-anchor="middle" font-weight="650" '
            'paint-order="stroke" stroke="#f7f0df" stroke-width="1.35" '
            f'data-religion-name="{html.escape(item.name, quote=True)}" '
            'data-theme-label="true" data-base-font-size="12.4" '
            'data-min-screen-font-size="9.8" data-base-text-stroke="1.35">'
            f'{html.escape(item.name)}</text>'
        )
    parts.append("</g>")
    return "".join(parts)


def _partition_overlay_svg_document(
    grid: WorldGrid,
    zone_paths: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    *,
    title: str,
    partition_id: str,
    data_attribute: str,
    zones: Sequence[tuple[str, str]],
    extra_overlay: str = "",
    zone_outline_color: str | None = None,
    zone_outline_width: float = 0.35,
    fill_opacity: float = 0.58,
) -> str:
    """Return a transparent partition overlay for the physical map."""

    zone_groups: list[str] = []
    for zone, ((label, color), paths) in enumerate(
        zip(zones, zone_paths, strict=True)
    ):
        if not paths:
            continue
        path_data = " ".join(
            _filled_path_data(points, codes) for points, codes in paths
        )
        stroke_attributes = (
            f'stroke="{zone_outline_color or color}" '
            f'stroke-width="{zone_outline_width:.2f}" '
            f'stroke-opacity="{fill_opacity:.2f}" stroke-linejoin="round" '
            'paint-order="stroke fill" '
            if zone_outline_width > 0.0
            else 'stroke="none" '
        )
        zone_groups.append(
            f'<g data-{data_attribute}="{zone}" aria-label="{html.escape(label)}">'
            f'<path d="{path_data}" fill="{color}" fill-opacity="{fill_opacity:.2f}" '
            'fill-rule="evenodd" '
            f'{stroke_attributes}/></g>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {grid.shape[1]} {grid.shape[0]}" '
        f'preserveAspectRatio="none" role="img" aria-label="{html.escape(title)}" '
        'shape-rendering="geometricPrecision">'
        f'<g id="{partition_id}" data-{data_attribute}="partition" '
        'data-partition="mutually-exclusive" '
        f'aria-label="{html.escape(title)}分区">{"".join(zone_groups)}</g>'
        f'{extra_overlay}</svg>'
    )


def _line_overlay_svg_document(
    grid: WorldGrid,
    body: str,
    *,
    title: str,
) -> str:
    """Package one heavy static line layer as a reusable vector artifact."""

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {grid.shape[1]} {grid.shape[0]}" '
        'preserveAspectRatio="none" role="img">'
        f'<title>{html.escape(title)}</title>{body}</svg>'
    )


def _svg_group(
    layer_id: str,
    paths: list[np.ndarray],
    stroke: str,
    *,
    stroke_width: float = 0.8,
    opacity: float = 1.0,
    path_levels: list[float] | None = None,
    curve_paths: bool = False,
    stroke_widths: Sequence[float] | None = None,
    stroke_space: str = "screen",
    path_attributes: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    if path_levels is not None and len(path_levels) != len(paths):
        raise ValueError("path_levels must correspond to paths")
    if stroke_widths is not None and len(stroke_widths) != len(paths):
        raise ValueError("stroke_widths must correspond to paths")
    if path_attributes is not None and len(path_attributes) != len(paths):
        raise ValueError("path_attributes must correspond to paths")
    if stroke_space not in {"screen", "map"}:
        raise ValueError("stroke_space must be either screen or map")
    body_parts = []
    for index, path in enumerate(paths):
        level_attribute = ""
        if path_levels is not None:
            level_attribute = f' data-level="{path_levels[index]:.6g}"'
        extra_attributes = ""
        if path_attributes is not None:
            extra_attributes = "".join(
                f' data-{html.escape(str(key), quote=True)}="{html.escape(str(value), quote=True)}"'
                for key, value in path_attributes[index].items()
            )
        serialized_path = _river_path_data(path) if curve_paths else _path_data(path)
        path_stroke_width = (
            float(stroke_widths[index])
            if stroke_widths is not None
            else float(stroke_width)
        )
        if not np.isfinite(path_stroke_width) or path_stroke_width <= 0.0:
            raise ValueError("stroke widths must be finite positive numbers")
        stroke_space_attribute = (
            f'data-base-stroke="{path_stroke_width:.2f}"'
            if stroke_space == "screen"
            else 'data-stroke-space="map"'
        )
        body_parts.append(
            f'<path d="{serialized_path}" fill="none" stroke="{stroke}" '
            f'stroke-width="{path_stroke_width:.2f}" stroke-linecap="round" '
            f'{stroke_space_attribute} stroke-linejoin="round"{level_attribute}{extra_attributes} />'
        )
    body = "".join(body_parts)
    return (
        f'<g id="{layer_id}" data-layer="{layer_id}" opacity="{opacity:.2f}">'
        f"{body}</g>"
    )


def _river_paths(grid: WorldGrid) -> list[np.ndarray]:
    """Turn flow_to links into edge/reach chains, not cell polygons.

    A reach stops at every indegree junction so each tributary owns the
    shared confluence point and the downstream reach starts at that point.
    """

    active = grid.river_order.reshape(-1) > 0
    if not np.any(active):
        return []
    downstream = grid.flow_to.reshape(-1)
    cell_count = active.size
    indegree = np.zeros(cell_count, dtype=np.int32)
    for source in np.flatnonzero(active):
        target = int(downstream[source])
        if 0 <= target < cell_count and active[target]:
            indegree[target] += 1

    starts = [int(cell) for cell in np.flatnonzero(active) if indegree[cell] != 1]
    paths: list[np.ndarray] = []
    visited_edges: set[tuple[int, int]] = set()

    def water_boundary_point(
        current: int,
        previous: tuple[float, float] | None,
        target: int,
    ) -> tuple[float, float] | None:
        row, column = divmod(current, grid.shape[1])
        candidates: list[tuple[int, int]] = []
        if 0 <= target < cell_count:
            target_row, target_column = divmod(target, grid.shape[1])
            if (
                grid.water[target_row, target_column] > 0
                and max(abs(target_row - row), abs(target_column - column)) == 1
            ):
                candidates.append((target_row, target_column))
        if not candidates:
            candidates = [
                (next_row, next_column)
                for next_row in range(max(0, row - 1), min(grid.shape[0], row + 2))
                for next_column in range(max(0, column - 1), min(grid.shape[1], column + 2))
                if (next_row, next_column) != (row, column)
                and grid.water[next_row, next_column] > 0
            ]
        if not candidates:
            return None
        if previous is not None:
            direction_x = column + 0.5 - previous[0]
            direction_y = row + 0.5 - previous[1]
            candidates.sort(
                key=lambda candidate: (
                    -(
                        (candidate[1] - column) * direction_x
                        + (candidate[0] - row) * direction_y
                    ),
                    abs(
                        (candidate[1] - column) * direction_y
                        - (candidate[0] - row) * direction_x
                    ),
                    candidate,
                )
            )
        water_row, water_column = candidates[0]
        return (
            column + 0.5 + (water_column - column) * 0.5,
            row + 0.5 + (water_row - row) * 0.5,
        )

    def follow(start: int) -> None:
        points: list[tuple[float, float]] = []
        current = start
        while 0 <= current < cell_count and active[current]:
            row, column = divmod(current, grid.shape[1])
            points.append((column + 0.5, row + 0.5))
            target = int(downstream[current])
            if target < 0 or target >= cell_count or not active[target]:
                boundary = water_boundary_point(
                    current,
                    points[-2] if len(points) >= 2 else None,
                    target,
                )
                if boundary is not None:
                    points.append(boundary)
                break
            edge = (current, target)
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            if indegree[target] != 1:
                row, column = divmod(target, grid.shape[1])
                points.append((column + 0.5, row + 0.5))
                break
            current = target
        if len(points) >= 2:
            paths.append(np.asarray(points, dtype=np.float64))

    for start in starts:
        follow(start)
    for remaining in np.flatnonzero(active):
        source = int(remaining)
        target = int(downstream[source])
        if 0 <= target < cell_count and active[target] and (source, target) not in visited_edges:
            follow(source)
    return paths


def _partition_river_dividers(
    river_order: np.ndarray,
    land: np.ndarray,
    river_paths: Sequence[np.ndarray],
    partition: np.ndarray,
    *,
    minimum_order: int = 2,
    include_unassigned: bool = False,
) -> list[np.ndarray]:
    """Return canonical river reaches that actually separate two owners.

    River geometry and terrain come from the same authored grid.  Sampling the
    two banks of each flow segment lets administrative ink use that physical
    centreline instead of the edge of whichever raster cell happened to own
    the channel.  No independent cartographic displacement is allowed here.
    """

    rivers = np.asarray(river_order)
    ground = np.asarray(land, dtype=bool)
    owners = np.asarray(partition)
    if rivers.ndim != 2 or ground.shape != rivers.shape or owners.shape != rivers.shape:
        raise WorldGridRenderError("river-divider fields must share a shape")
    if minimum_order < 1:
        raise WorldGridRenderError("river-divider minimum order must be positive")
    height, width = rivers.shape

    def sample_owner(point: np.ndarray, normal: np.ndarray, sign: float) -> int:
        for radius in (1.15, 1.75, 2.35):
            sample = point + sign * radius * normal
            row = int(math.floor(float(sample[1])))
            column = int(math.floor(float(sample[0]))) % width
            if row < 0 or row >= height:
                continue
            if ground[row, column] and int(rivers[row, column]) == 0:
                return int(owners[row, column])
        return 0

    result: list[np.ndarray] = []
    for path in river_paths:
        values = np.asarray(path, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            continue
        current: list[np.ndarray] = []
        for start, end in zip(values[:-1], values[1:], strict=True):
            midpoint = 0.5 * (start + end)
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length <= 1.0e-12:
                separates = False
            else:
                row = int(np.clip(math.floor(float(midpoint[1])), 0, height - 1))
                column = int(math.floor(float(midpoint[0]))) % width
                if int(rivers[row, column]) < minimum_order:
                    separates = False
                else:
                    normal = np.asarray((-delta[1], delta[0]), dtype=np.float64) / length
                    left = sample_owner(midpoint, normal, 1.0)
                    right = sample_owner(midpoint, normal, -1.0)
                    separates = left != right and (
                        (left > 0 and right > 0)
                        or (include_unassigned and (left > 0 or right > 0))
                    )
            if separates:
                if not current:
                    current.append(start)
                current.append(end)
            elif len(current) >= 2:
                result.append(np.asarray(current, dtype=np.float64))
                current = []
        if len(current) >= 2:
            result.append(np.asarray(current, dtype=np.float64))
    return result


def _line_geometry_paths(geometry: Any) -> list[np.ndarray]:
    """Flatten a Shapely line result into SVG-ready coordinate arrays."""

    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type in {"LineString", "LinearRing"}:
        coordinates = np.asarray(geometry.coords, dtype=np.float64)
        return [coordinates] if coordinates.shape[0] >= 2 else []
    result: list[np.ndarray] = []
    for part in geometry.geoms:
        result.extend(_line_geometry_paths(part))
    return result


def _remove_offset_boundaries_along_rivers(
    boundary_paths: Sequence[np.ndarray],
    river_dividers: Sequence[np.ndarray],
    *,
    clearance: float = 1.05,
) -> list[np.ndarray]:
    """Remove raster-bank duplicates before drawing the shared river axis."""

    if not river_dividers:
        return [np.asarray(path, dtype=np.float64) for path in boundary_paths]
    boundaries = [
        shapely.LineString(np.asarray(path, dtype=np.float64))
        for path in boundary_paths
        if np.asarray(path).shape[0] >= 2
    ]
    dividers = [
        shapely.LineString(np.asarray(path, dtype=np.float64))
        for path in river_dividers
        if np.asarray(path).shape[0] >= 2
    ]
    if not boundaries or not dividers:
        return [np.asarray(path, dtype=np.float64) for path in boundary_paths]
    river_corridor = shapely.union_all(dividers).buffer(
        float(clearance),
        cap_style="round",
        join_style="round",
    )
    clipped = shapely.union_all(boundaries).difference(river_corridor)
    return _line_geometry_paths(clipped)


def _river_width_segments(
    grid: WorldGrid,
    paths: Sequence[np.ndarray],
    *,
    minimum_width: float = 0.42,
    maximum_width: float = 1.80,
) -> tuple[list[np.ndarray], list[float]]:
    """Split reaches into locally scaled hydraulic-geometry stroke segments.

    Bankfull channel width primarily follows accumulated discharge, so every
    sampled river cell receives a square-root-discharge width before the reach
    is split at visible width changes.  This makes tributaries narrow and lets
    the receiving channel widen *after* a confluence instead of assigning one
    maximum width to the whole reach.  Low-gradient bends receive only a small
    local adjustment (at most eight percent); curvature never substitutes for
    catchment flow.
    """

    if not paths:
        return [], []
    active = grid.river_order > 0
    active_discharge = grid.discharge[active].astype(np.float64)
    positive = active_discharge[active_discharge > 0.0]
    if positive.size == 0:
        return list(paths), [float(minimum_width) for _ in paths]

    lower = math.sqrt(float(positive.min()))
    upper = math.sqrt(float(np.quantile(positive, 0.99)))
    scale = max(1.0e-12, upper - lower)

    def cell_index(point: np.ndarray) -> int:
        column = int(np.clip(round(float(point[0]) - 0.5), 0, grid.shape[1] - 1))
        row = int(np.clip(round(float(point[1]) - 0.5), 0, grid.shape[0] - 1))
        return row * grid.shape[1] + column

    segments: list[np.ndarray] = []
    widths: list[float] = []
    for path in paths:
        if path.shape[0] < 2:
            continue
        samples = np.asarray([cell_index(point) for point in path], dtype=np.int64)
        transformed = np.sqrt(
            np.maximum(0.0, grid.discharge.ravel()[samples].astype(np.float64))
        )
        normalized = np.clip((transformed - lower) / scale, 0.0, 1.0)
        local_widths = minimum_width + normalized * (
            maximum_width - minimum_width
        )

        bend_factor = np.ones(path.shape[0], dtype=np.float64)
        for index in range(1, path.shape[0] - 1):
            window = min(3, index, path.shape[0] - 1 - index)
            incoming_vector = path[index] - path[index - window]
            outgoing_vector = path[index + window] - path[index]
            incoming_length = float(np.linalg.norm(incoming_vector))
            outgoing_length = float(np.linalg.norm(outgoing_vector))
            if incoming_length <= 1.0e-12 or outgoing_length <= 1.0e-12:
                continue
            cosine = float(
                np.clip(
                    np.dot(incoming_vector, outgoing_vector)
                    / (incoming_length * outgoing_length),
                    -1.0,
                    1.0,
                )
            )
            turn_fraction = math.acos(cosine) / math.pi
            previous_elevation = float(grid.elevation.flat[samples[index - window]])
            next_elevation = float(grid.elevation.flat[samples[index + window]])
            longitudinal_change = abs(next_elevation - previous_elevation) / max(
                1.0,
                incoming_length + outgoing_length,
            )
            flatness = 1.0 - float(np.clip(longitudinal_change / 0.02, 0.0, 1.0))
            bend_factor[index] = 1.0 + 0.08 * turn_fraction * flatness
        if bend_factor.size >= 3:
            bend_factor = np.convolve(
                np.pad(bend_factor, (1, 1), mode="edge"),
                np.asarray((0.25, 0.5, 0.25)),
                mode="valid",
            )
        local_widths = np.minimum(maximum_width, local_widths * bend_factor)
        quantized = np.clip(
            np.round(local_widths / 0.01) * 0.01,
            minimum_width,
            maximum_width,
        )

        start = 0
        current_width = float(quantized[0])
        for index in range(1, path.shape[0]):
            next_width = float(quantized[index])
            if abs(next_width - current_width) < 0.009:
                continue
            if index - start >= 1:
                segments.append(np.asarray(path[start : index + 1], dtype=np.float64))
                widths.append(current_width)
            start = index
            current_width = next_width
        if path.shape[0] - 1 - start >= 1:
            segments.append(np.asarray(path[start:], dtype=np.float64))
            widths.append(current_width)
        elif segments:
            # A one-point tail belongs visually to the preceding round-capped
            # segment; extend it rather than emitting an invalid SVG path.
            segments[-1] = np.vstack((segments[-1], path[-1]))
    return segments, widths


def _html_document(
    grid: WorldGrid,
    digest: str,
    overlay: str,
    *,
    society: SocietyLayers,
    tectonics: TectonicReview,
    territorial_qa: Mapping[str, int | float],
    width: int,
    height: int,
    viewbox_width: int,
    viewbox_height: int,
) -> str:
    safe_digest = html.escape(digest, quote=True)
    world_name = str(grid.metadata.get("worldProfile", {}).get("name", ""))
    page_title = html.escape(f"{world_name} · 世界地图" if world_name else "世界网格审图")
    tectonic_source_help = (
        "直接读取这张地形所用的板块编号、运动速度和边界类型，与已封存的物理场逐格一致。"
        if tectonics.diagnostics["derivation"] == "direct-causal-procedural-plate-fields"
        else "山系、岛弧—海沟和洋底高地是边界反推的地形证据。"
    )
    polar_scene_band = 392
    polar_inset_size = 336
    north_polar_inset = _polar_inset_data_url(
        grid, north=True, size=polar_inset_size
    )
    south_polar_inset = _polar_inset_data_url(
        grid, north=False, size=polar_inset_size
    )
    scene_height = height + polar_scene_band * 2
    coordinate_system = grid.metadata.get("coordinateReferenceSystem")
    if not isinstance(coordinate_system, Mapping):
        raise WorldGridRenderError("coordinateReferenceSystem metadata is required")
    if (
        coordinate_system.get("contractId") != "EIR-GEOG-1"
        or coordinate_system.get("reviewProjection") != "equirectangular"
        or coordinate_system.get("storageGridMapping") != "plate-carree"
        or coordinate_system.get("readerProjection") != "equal-earth"
    ):
        raise WorldGridRenderError("unsupported review projection contract")
    season_buttons = "".join(
        f'<button id="season-{season_id}" type="button" data-season-button="{season_id}">'
        f'{html.escape(label)}</button>'
        for season_id, label in zip(_SEASON_IDS, _SEASON_LABELS, strict=True)
    )
    climate_legend = "".join(
        f'<div class="legend-row"><span class="swatch" style="background:{color}"></span>'
        f'<span>{html.escape(label)}</span></div>'
        for label, color in _CLIMATE_ZONES
    )
    biome_legend = "".join(
        f'<div class="legend-row"><span class="swatch" style="background:{color}"></span>'
        f'<span>{html.escape(label)}</span></div>'
        for label, color in _BIOME_ZONES
    )
    watershed_key = "".join(
        f'<span class="basin-swatch" style="background:{color}" aria-hidden="true"></span>'
        for _label, color in _WATERSHED_ZONES[1:9]
    )
    potential_legend = "".join(
        f'<div class="legend-row"><span class="swatch" style="background:{color}"></span>'
        f'<span>{html.escape(label)}</span></div>'
        for label, color in _LAND_POTENTIAL_ZONES
    )
    population_legend = "".join(
        f'<div class="legend-row"><span class="swatch" style="background:{color}"></span>'
        f'<span>{html.escape(label)}</span></div>'
        for label, color in _POPULATION_ZONES
    )
    population_range_label = _format_population_range(
        society.population.population_min,
        society.population.population_max,
    )
    civilization_count = len(society.cultures.civilizations)
    religion_count = len(society.religions.religions)
    religion_tradition_counts: dict[str, int] = {}
    for religion in society.religions.religions:
        religion_tradition_counts[religion.tradition] = (
            religion_tradition_counts.get(religion.tradition, 0) + 1
        )
    religion_summary = "、".join(
        f"{html.escape(tradition)}×{count}"
        for tradition, count in sorted(
            religion_tradition_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    civilization_profile_counts: dict[str, int] = {}
    for civilization in society.cultures.civilizations:
        profile = civilization.internal_diversity[0].removeprefix("主导物质形态：")
        civilization_profile_counts[profile] = civilization_profile_counts.get(profile, 0) + 1
    civilization_profile_summary = "、".join(
        f"{html.escape(profile)}×{count}"
        for profile, count in sorted(
            civilization_profile_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    government_by_identifier = {
        item.identifier: item for item in society.politics.government_forms
    }
    government_counts: dict[str, int] = {}
    for entity in society.politics.political_entities:
        government = government_by_identifier[entity.government_form_identifier]
        government_counts[government.name] = government_counts.get(government.name, 0) + 1
    government_summary = "、".join(
        f"{html.escape(profile)}×{count}"
        for profile, count in sorted(
            government_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    province_system_labels = {
        "central-bureaucracy": "中央官僚区划",
        "feudal-vassalage": "封建封臣领",
        "civic-administration": "城市／海商辖区",
        "confederal-territory": "部盟领地",
        "estate-administration": "等级身份辖地",
    }
    province_system_counts: dict[str, int] = {}
    province_function_counts: dict[str, int] = {}
    for province in society.provinces.provinces:
        system_label = province_system_labels[province.administrative_system]
        province_system_counts[system_label] = province_system_counts.get(system_label, 0) + 1
        function_label = {
            "capital": "直辖核心",
            "civil": "民政",
            "military": "军政",
            "frontier": "边疆",
            "maritime": "海政",
            "vassal": "封臣领",
            "crown": "王冠直领",
            "pastoral": "牧地",
            "civic": "城市自治",
        }[province.administrative_function]
        province_function_counts[function_label] = province_function_counts.get(function_label, 0) + 1
    province_system_summary = "、".join(
        f"{html.escape(label)}×{count}"
        for label, count in sorted(
            province_system_counts.items(), key=lambda item: (-item[1], item[0])
        )
    )
    province_function_summary = "、".join(
        f"{html.escape(label)}×{count}"
        for label, count in sorted(
            province_function_counts.items(), key=lambda item: (-item[1], item[0])
        )
    )
    language_count = len(society.cultures.languages)
    route_counts = {
        mode: sum(route.mode == mode for route in society.transport.routes)
        for mode in ("road", "river", "sea")
    }
    state_alignment_percent = 100.0 * float(
        territorial_qa["stateBoundaryAlignment"]
    )
    province_alignment_percent = 100.0 * float(
        territorial_qa["provinceBoundaryAlignment"]
    )
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="grid-digest" content="{safe_digest}">
  <link rel="icon" href="data:,">
  <title>{page_title}</title>
  <style>
    :root {{ color-scheme: light; font-family: "Microsoft YaHei", sans-serif; }}
    body {{ margin: 0; color: #263447; background: #e8e1d3; }}
    header {{ padding: 14px 20px; background: #263447; color: #f5f0e6; }}
    header h1 {{ margin: 0 0 4px; font-size: 20px; }}
    header p {{ margin: 0; font-size: 12px; opacity: .8; }}
    main {{ display: grid; grid-template-columns: 226px 1fr; min-height: calc(100vh - 72px); }}
    aside {{ padding: 16px; background: #f5f0e6; border-right: 1px solid #cfc5b3; box-shadow: 3px 0 12px rgb(67 55 39 / 7%); z-index: 2; max-height: calc(100vh - 72px); overflow-y: auto; }}
    aside h2 {{ margin: 0 0 12px; font-size: 15px; }}
    aside h3 {{ margin: 18px 0 8px; font-size: 13px; }}
    label {{ display: block; margin: 10px 0; font-size: 13px; cursor: pointer; }}
    button {{ border: 1px solid #8f8069; background: #ece4d5; color: #263447; padding: 7px 8px; cursor: pointer; font: inherit; font-size: 12px; }}
    button[aria-pressed="true"] {{ background: #263447; color: #f5f0e6; }}
    button:hover {{ background: #e2d7c4; }}
    button[aria-pressed="true"]:hover {{ background: #31445c; }}
    button:focus-visible, select:focus-visible, summary:focus-visible, input:focus-visible {{ outline: 2px solid #3d6f96; outline-offset: 2px; }}
    .mode-switch, .season-switch {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 6px; }}
    .season-switch {{ grid-template-columns: 1fr; }}
    .control-section {{ margin-top: 16px; }}
    .control-section > h3 {{ margin-top: 0; }}
    .select-label {{ margin: 0 0 6px; font-weight: 700; }}
    select {{ width: 100%; padding: 8px 28px 8px 9px; border: 1px solid #93856f; background: #fffdf8; color: #263447; font: inherit; font-size: 12px; }}
    .theme-detail {{ margin-top: 8px; padding: 8px 0 2px; border-top: 1px solid #d8cfbf; }}
    .theme-detail .legend {{ margin-top: 8px; }}
    .layer-group {{ margin-top: 14px; border-top: 1px solid #d8cfbf; padding-top: 10px; }}
    .layer-group summary {{ cursor: pointer; font-size: 13px; font-weight: 700; color: #263447; }}
    .check-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0 8px; margin-top: 6px; }}
    .check-grid label {{ margin: 6px 0; font-size: 12px; }}
    .inline-legend {{ margin-top: 9px; }}
    .inline-legend summary {{ font-size: 11px; font-weight: 600; color: #586577; }}
    [hidden] {{ display: none !important; }}
    #viewport {{ position: relative; overflow: hidden; touch-action: none; cursor: grab; background: #d8d0c0; }}
    #viewport:focus-visible {{ outline: 3px solid #3d6f96; outline-offset: -3px; }}
    #viewport.dragging {{ cursor: grabbing; }}
    .map-tools {{ position: absolute; z-index: 6; top: 14px; right: 14px; display: grid; grid-template-columns: 34px 58px 34px; gap: 1px; padding: 3px; border: 1px solid rgb(57 63 70 / 38%); background: rgb(248 244 235 / 92%); box-shadow: 0 2px 9px rgb(38 52 71 / 18%); cursor: default; }}
    .map-tools button {{ min-width: 34px; min-height: 34px; padding: 0; border: 0; background: transparent; font-size: 18px; line-height: 1; }}
    .map-tools button:hover {{ background: #e5dccb; }}
    .map-tools button:active {{ background: #d9cbb7; }}
    #zoom-reset {{ font-size: 12px; font-weight: 700; }}
    #zoom-level {{ position: absolute; top: calc(100% + 5px); right: 0; min-width: 82px; padding: 4px 7px; background: rgb(38 52 71 / 86%); color: #f8f4eb; font-size: 11px; line-height: 1.2; text-align: center; pointer-events: none; }}
    #scene {{ position: absolute; left: 0; top: 0; width: {width}px; height: {scene_height}px; transform-origin: 0 0; backface-visibility: hidden; contain: layout paint; display: grid; grid-template-rows: {polar_scene_band}px {height}px {polar_scene_band}px; justify-items: center; }}
    #map-frame {{ position: relative; grid-row: 2; width: {width}px; height: {height}px; }}
    #terrain {{ display: block; image-rendering: crisp-edges; }}
    #terrain[hidden] {{ display: none !important; }}
    .monsoon-precipitation {{ pointer-events: none; image-rendering: auto; }}
    .physical-theme-map {{ pointer-events: none; image-rendering: auto; }}
    #overlay {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }}
    #overlay {{ shape-rendering: geometricPrecision; }}
    .polar-insets {{ display: contents; pointer-events: none; }}
    .polar-inset {{ margin: 10px 0; padding: 10px 10px 8px; width: {polar_inset_size}px; align-self: center; border: 1px solid rgb(71 92 117 / 55%); border-radius: 3px; background: rgb(248 246 239 / 92%); box-shadow: 0 2px 8px rgb(38 52 71 / 18%); }}
    .polar-inset.north {{ grid-row: 1; }}
    .polar-inset.south {{ grid-row: 3; }}
    .polar-inset img {{ display: block; width: {polar_inset_size}px; height: {polar_inset_size}px; }}
    .polar-inset figcaption {{ margin-top: 6px; text-align: center; color: #33475e; font-size: 16px; font-weight: 700; line-height: 1.25; }}
    .help {{ color: #586577; font-size: 12px; line-height: 1.6; }}
    .frontier-key {{ display: flex; align-items: center; gap: 7px; margin-top: 8px; }}
    .frontier-swatch {{ width: 28px; height: 12px; border: 1px solid #777065; background: repeating-linear-gradient(66deg, #d4cec0 0 4px, #8b8478 4px 5px, #d4cec0 5px 9px); flex: 0 0 auto; }}
    .legend {{ margin-top: 14px; font-size: 11px; line-height: 1.5; color: #44536a; }}
    .legend-row {{ display: flex; align-items: center; gap: 7px; margin: 4px 0; }}
    .swatch {{ width: 36px; height: 8px; border: 1px solid rgb(83 75 61 / 55%); flex: 0 0 auto; }}
    .basin-key {{ display: grid; grid-template-columns: repeat(8, 1fr); gap: 2px; margin: 12px 0 8px; }}
    .basin-swatch {{ height: 10px; border: 1px solid rgb(83 75 61 / 35%); }}
    .precip-scale {{ background: linear-gradient(90deg,#cea46a,#decb80,#aacd96,#65b7aa,#488ebc,#2f5c9e); }}
    .river-line {{ width: 42px; height: 0; border-top: 2px solid #5799cf; }}
    .river-line.seasonal {{ border-top-style: dashed; opacity: .7; }}
    .city-dot {{ width: 10px; height: 10px; border-radius: 50%; border: 1px solid #263447; background: #f4dfad; flex: 0 0 auto; }}
    .city-dot.city {{ width: 8px; height: 8px; background: #f7eed7; }}
    .city-dot.town {{ width: 5px; height: 5px; background: #f7eed7; }}
    .site-mark {{ width: 12px; color: #7a5838; font-size: 12px; line-height: 1; text-align: center; flex: 0 0 auto; }}
    .route-line {{ width: 42px; height: 0; border-top: 2px solid; flex: 0 0 auto; }}
    .route-line.road {{ border-color: #8b5e3c; }}
    .route-line.river {{ border-color: #246da0; border-top-style: dashed; }}
    .route-line.sea {{ border-color: #365f83; border-top-style: dashed; }}
    .bridge-mark {{ width: 14px; height: 8px; border-left: 2px solid #5f4938; border-right: 2px solid #5f4938; flex: 0 0 auto; }}
    html[data-active-theme="political"] #province-labels,
    html[data-active-theme="political"] #province-boundaries {{ display: none !important; }}
    #state-labels, #geographic-labels, #religion-labels {{ font-family: "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif; }}
    #province-labels, #city-labels, #civilization-labels, #language-labels {{ font-family: "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif; }}
    html[data-active-theme="provinces"] #state-labels {{ opacity: .68; }}
    html[data-active-theme="provinces"] #province-labels {{ opacity: .94; }}
    #civilization-boundaries, #language-boundaries,
    #language-civilization-boundaries, #state-boundaries,
    #province-boundaries, #transport-network, #bridge-layer, #city-layer {{
      shape-rendering: geometricPrecision;
    }}
    @media (prefers-reduced-motion: no-preference) {{
      .theme-detail {{ transition: opacity 180ms ease-out; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{page_title}</h1>
    <p>确定性网格指纹：{safe_digest}</p>
  </header>
  <main>
    <aside aria-label="图层控制">
      <h2>图层控制</h2>
      <div class="mode-switch" aria-label="专题切换">
        <button id="view-physical" type="button" data-view-button="physical">物理图</button>
        <button id="view-monsoon" type="button" data-view-button="monsoon">季风专题</button>
        <button id="view-tectonic" type="button" data-view-button="tectonic">板块图</button>
      </div>
      <details class="layer-group" open>
        <summary>星球投影</summary>
        <p class="help"><strong>EIR-GEOG-1</strong> 球面经纬坐标；本页为等距圆柱（Plate Carrée）审图投影，中央经线 0°、标准纬线 0°。两极在上下边界被展开，高纬面积会被放大。</p>
        <p class="help">世界面积与文明范围对比以 Equal Earth 等积阅读投影为准；南北极近景使用方位等积极投影。</p>
      </details>
      <div id="physical-controls">
        <section class="control-section" aria-labelledby="theme-heading">
          <label id="theme-heading" class="select-label" for="physical-theme">专题着色</label>
          <select id="physical-theme">
            <option value="none">无，保留纯物理底图</option>
            <option value="climate">气候分区</option>
            <option value="biome">生物群系</option>
            <option value="watershed">水文流域</option>
            <option value="potential">农业与宜居潜力</option>
            <option value="population">人口分布</option>
            <option value="civilizations">文明区</option>
            <option value="languages">语言区</option>
            <option value="religions">宗教</option>
            <option value="political">国家政区</option>
            <option value="provinces">省份政区</option>
          </select>
          <div class="theme-detail" data-theme-detail="none">
            <p class="help">不加专题颜色，直接审查高程、水系与海岸。</p>
          </div>
          <div class="theme-detail" data-theme-detail="climate" hidden>
            <p class="help">气候区由纬度热量、海拔、四季降水与雨影共同推导。</p>
            <details class="inline-legend"><summary>查看气候图例</summary><div class="legend">{climate_legend}</div></details>
          </div>
          <div class="theme-detail" data-theme-detail="biome" hidden>
            <p class="help">自然植被与生物群系，与山脉、雪线和水分条件对照显示。</p>
            <details class="inline-legend"><summary>查看生物群系图例</summary><div class="legend">{biome_legend}</div></details>
          </div>
          <div class="theme-detail" data-theme-detail="watershed" hidden>
            <p class="help">显示主要流域分水范围，底下仍保留地形与完整河网。</p>
            <div class="basin-key" aria-label="主要流域配色">{watershed_key}</div>
          </div>
          <div class="theme-detail" data-theme-detail="potential" hidden>
            <p class="help">综合生长季、水源、坡度、沿海调节与灾害风险。</p>
            <details class="inline-legend"><summary>查看潜力图例</summary><div class="legend">{potential_legend}</div></details>
          </div>
          <div class="theme-detail" data-theme-detail="population" hidden>
            <p class="help">总人口按 {population_range_label}的区间投影，叠加后可直接比对河谷与城镇。</p>
            <details class="inline-legend"><summary>查看人口图例</summary><div class="legend">{population_legend}</div></details>
          </div>
          <div class="theme-detail" data-theme-detail="civilizations" hidden>
            <p class="help">{civilization_count} 个大小不一的文明圈沿人口、交通、河谷、草原与岛链传播；山脊、雪线、荒漠和断裂的聚落网会切断辐射。文明层只记录生业、语言、习俗与传播范围，不预设国家或政体。生业：{civilization_profile_summary}。</p>
          </div>
          <div class="theme-detail" data-theme-detail="languages" hidden>
            <p class="help">{language_count} 个语言区，保留文明边界与更细的地形隔离效应。</p>
          </div>
          <div class="theme-detail" data-theme-detail="religions" hidden>
            <p class="help">{religion_count} 个宗教传播圈，从各自圣城沿人口与交通网络传播，同时受山脊、水系与文化距离约束。圣城以星形标记。传统：{religion_summary}。</p>
          </div>
          <div class="theme-detail" data-theme-detail="political" hidden>
            <p class="help">{len(society.politics.states)} 个国家身份与独立政体模板组合成当前政治实体；国名和疆域不内嵌政体，同一政体可被不同文明的国家复用。当前组合：{government_summary}。连续的低密度草原、部落腹地、荒漠与高山可保留为游牧联盟或无常设政权区；普通平原上的小块空洞不会被伪装成“蛮荒”。</p>
            <p class="help" data-territorial-qa="states">自然边界贴合率 {state_alignment_percent:.1f}%；无地理依据的最长轴向国界 {int(territorial_qa['stateStraightRunMaximum'])} 格；普通环境小型无政权空洞 {int(territorial_qa['ordinaryFrontierComponents'])} 处。</p>
            <p class="help frontier-key"><span class="frontier-swatch" aria-hidden="true"></span><span>斜线：无常设政权区；文字标出当地游牧、渔猎、山地或海洋群体</span></p>
          </div>
          <div class="theme-detail" data-theme-detail="provinces" hidden>
            <p class="help">国家形成后，再由人口密度、山河交通、政体和地方职能共同划分 {len(society.provinces.provinces)} 个地方单位：稠密核心较小，稀疏边疆、军政区和牧地较大。分封国家会区分王廷所在辖域、分布在其他地方且由君主直接控制的王冠领，以及不同等级的封臣辖地；中央集权国家则区分首都直属辖域、民政区、军政区、海政区与边疆辖区。这些是内部制度概念，地图上仍显示由首府或地方专名与文化制度后缀构成的真实名称，不直接把“直隶”或“京畿直辖”当地名。</p>
            <p class="help">制度：{province_system_summary}。职能：{province_function_summary}。</p>
            <p class="help">细短虚线为地方界，较粗长虚线为国界；山脊、主要河流、分水岭和交通阻力共同约束边界。</p>
            <p class="help" data-territorial-qa="provinces">省界自然边界贴合率 {province_alignment_percent:.1f}%；无地理依据的最长轴向省界 {int(territorial_qa['provinceStraightRunMaximum'])} 格；显式桥梁 {len(society.transport.bridges)} 座。</p>
            <p class="help frontier-key"><span class="frontier-swatch" aria-hidden="true"></span><span>无常设国家治理的部族地不划省</span></p>
          </div>
        </section>
        <details class="layer-group" open>
          <summary>基础地理</summary>
          <div class="check-grid">
            <label><input id="toggle-elevation-bands" type="checkbox" data-layer="elevation-bands" checked> 高程</label>
            <label><input id="toggle-bathymetry-bands" type="checkbox" data-layer="bathymetry-bands" checked> 海深</label>
            <label><input id="toggle-graticule" type="checkbox" data-layer="graticule" checked> 经纬线</label>
            <label><input id="toggle-coast" type="checkbox" data-layer="coast" checked> 海岸</label>
            <label><input id="toggle-lakes" type="checkbox" data-layer="lakes" checked> 湖泊</label>
            <label><input id="toggle-elevation-contours" type="checkbox" data-layer="elevation-contours" checked> 等高线</label>
            <label><input id="toggle-snow" type="checkbox" data-layer="snow" checked> 积雪</label>
            <label><input id="toggle-sea-ice" type="checkbox" data-layer="sea-ice" checked> 海冰</label>
            <label><input id="toggle-polar-references" type="checkbox" data-layer="polar-references" checked> 极圈／极点</label>
            <label><input id="toggle-rivers" type="checkbox" data-layer="rivers" checked> 河流</label>
          </div>
          <details class="inline-legend"><summary>河流图例</summary>
            <div class="legend" aria-label="物理图河流图例">
              <div class="legend-row"><span class="river-line"></span><span>常年河</span></div>
              <div class="legend-row"><span class="river-line seasonal"></span><span>季节性断流河</span></div>
            </div>
          </details>
        </details>
        <details class="layer-group" open>
          <summary>人文与交通</summary>
          <div class="check-grid">
            <label><input id="toggle-geographic-labels" type="checkbox" data-layer="geographic-labels" checked> 地理名称</label>
            <label><input id="toggle-cities" type="checkbox" data-layer="cities"> 城市聚落</label>
            <label><input id="toggle-transport" type="checkbox" data-layer="transport"> 交通网络</label>
          </div>
          <p class="help">地理名称按世界、区域、近景三级显示；放大后会逐步出现支流、小湖与山峰名。</p>
          <details class="inline-legend"><summary>城市与交通图例</summary>
            <div class="legend" aria-label="城市交通图例">
              <div class="legend-row"><span class="city-dot metropolis"></span><span>都会</span></div>
              <div class="legend-row"><span class="city-dot city"></span><span>城市</span></div>
              <div class="legend-row"><span class="city-dot town"></span><span>城镇</span></div>
              <div class="legend-row"><span class="site-mark pass">◆</span><span>关隘</span></div>
              <div class="legend-row"><span class="site-mark fortress">⬟</span><span>要塞</span></div>
              <div class="legend-row"><span class="site-mark" style="color:#8b5a3b">✦</span><span>宗教圣城</span></div>
              <div class="legend-row"><span class="route-line road"></span><span>陆路</span></div>
              <div class="legend-row"><span class="route-line river"></span><span>河运</span></div>
              <div class="legend-row"><span class="route-line sea"></span><span>海运／湖运</span></div>
              <div class="legend-row"><span class="bridge-mark"></span><span>跨越主要河流的桥</span></div>
            </div>
          </details>
          <p class="help">聚落 {len(society.settlements)} 处；道路 {route_counts['road']} 条、河运 {route_counts['river']} 条、航线 {route_counts['sea']} 条。</p>
        </details>
        <p class="help">深链接：<code>?theme=climate</code> 或 <code>?theme=provinces</code>；滚轮或右上角按钮缩放，拖动平移，按 0 返回全图。</p>
      </div>
      <div id="monsoon-controls" hidden>
        <h3>季节</h3>
        <div class="season-switch">{season_buttons}</div>
        <h3>专题图层</h3>
        <label><input id="toggle-monsoon-precipitation" type="checkbox" checked> 降水</label>
        <label><input id="toggle-monsoon-wind" type="checkbox" checked> 风场</label>
        <label><input id="toggle-seasonal-rivers" type="checkbox" checked> 季节河流</label>
        <div class="legend" aria-label="季风专题图例">
          <div class="legend-row"><span class="swatch precip-scale"></span><span>相对降水：少 → 多（四季同一色标）</span></div>
          <div class="legend-row"><span class="river-line"></span><span>常年河（仅水量季节变化）</span></div>
          <div class="legend-row"><span class="river-line seasonal"></span><span>季节性断流河</span></div>
          <p>箭头朝向表示气流来向与流入方向，长度和线宽表示相对风速。结果为相对气候场，不代表毫米或米每秒。</p>
          <p>深链接示例：<code>?view=monsoon&amp;season=vernal</code></p>
        </div>
      </div>
      <div id="tectonic-controls" hidden>
        <h3>板块构造基础图</h3>
        <p class="help">独立显示 {tectonics.diagnostics['plateCount']} 个板块的范围、名称与运动方向；{tectonic_source_help}普通海岸不是板块边界。</p>
        <div class="legend" aria-label="板块构造图例">
          <div class="legend-row"><span class="route-line" style="border-color:#405d5d"></span><span>碰撞／俯冲边界（齿线）</span></div>
          <div class="legend-row"><span class="route-line" style="border-color:#b94e43"></span><span>分离／扩张边界</span></div>
          <div class="legend-row"><span class="route-line" style="border-color:#bd7a35;border-top-style:dashed"></span><span>转换断层</span></div>
          <p>边界两侧的成对箭头表示相对运动；板块内长箭头表示整体运动，长度按 cm/年缩放。</p>
          <p>深链接：<code>?view=tectonic</code></p>
        </div>
      </div>
    </aside>
    <section id="viewport" aria-label="世界网格审图视口" tabindex="0">
      <nav class="map-tools" aria-label="地图缩放">
        <button id="zoom-out" type="button" aria-label="缩小地图" title="缩小（-）">−</button>
        <button id="zoom-reset" type="button" aria-label="返回全图" title="返回全图（0）">全图</button>
        <button id="zoom-in" type="button" aria-label="放大地图" title="放大（+）">＋</button>
        <output id="zoom-level" aria-live="polite">全图</output>
      </nav>
      <div id="scene">
        <div id="map-frame">
          <img id="terrain" src="terrain.png" width="{width}" height="{height}" alt="栅格导出预览" hidden aria-hidden="true">
          <svg id="overlay" width="{width}" height="{height}" viewBox="0 0 {viewbox_width} {viewbox_height}" preserveAspectRatio="none" data-render-mode="vector-bands" role="img" aria-label="审图矢量图层">
            {overlay}
          </svg>
        </div>
        <div class="polar-insets" aria-label="与主地图共同缩放的方位等积极地近景">
          <figure class="polar-inset north">
            <img src="{north_polar_inset}" width="{polar_inset_size}" height="{polar_inset_size}" alt="北极方位等积极投影">
            <figcaption>北极 · 方位等积</figcaption>
          </figure>
          <figure class="polar-inset south">
            <img src="{south_polar_inset}" width="{polar_inset_size}" height="{polar_inset_size}" alt="南极方位等积极投影">
            <figcaption>南极 · 方位等积</figcaption>
          </figure>
        </div>
      </div>
    </section>
  </main>
  <script>
    const viewport = document.getElementById('viewport');
    const scene = document.getElementById('scene');
    let scale = 1;
    let translateX = 0;
    let translateY = 0;
    let dragging = false;
    let cameraInteracted = false;
    let lastX = 0;
    let lastY = 0;
    let adaptiveUpdateTimer = 0;
    let cameraFrame = 0;
    let minimumScale = 0.2;
    const maximumScale = 18;
    const seasons = ['vernal', 'june', 'autumnal', 'december'];
    const themes = ['none','climate','biome','watershed','potential','population','civilizations','languages','religions','political','provinces'];
    const query = new URLSearchParams(window.location.search);
    let activeView = ['monsoon','tectonic'].includes(query.get('view')) ? query.get('view') : 'physical';
    let activeTheme = themes.includes(query.get('theme')) ? query.get('theme') : 'none';
    let activeSeason = seasons.includes(query.get('season')) ? query.get('season') : 'vernal';
    document.getElementById('physical-theme').value = activeTheme;

    function updateAdaptiveStrokes() {{
      // Deep zoom needs more visual hierarchy than a constant one-pixel hairline.
      // Source strokes shrink slower than the camera grows, so their apparent
      // screen weight increases gently without becoming cartoonishly heavy.
      const widthFactor = scale <= 1 ? scale : Math.pow(scale, -0.72);
      document.querySelectorAll('[data-base-stroke]').forEach((path) => {{
        const base = Number(path.dataset.baseStroke);
        if (Number.isFinite(base)) {{
          const width = base * widthFactor;
          path.setAttribute('stroke-width', width.toFixed(3));
        }}
      }});
      const safeStrokeScale = Math.max(0.01, scale);
      const screenMapFactor = safeStrokeScale <= 1
        ? 1 / safeStrokeScale
        : Math.pow(safeStrokeScale, -0.72);
      const detailLevel = scale < 0.85 ? 'world' : scale < 2.4 ? 'regional' : scale < 6 ? 'local' : 'close';
      document.documentElement.dataset.mapDetail = detailLevel;
      document.querySelectorAll('[data-screen-stroke]').forEach((path) => {{
        const screenWidth = Number(path.dataset.screenStroke);
        if (Number.isFinite(screenWidth)) {{
          const mapWidth = screenWidth * screenMapFactor;
          path.setAttribute('stroke-width', mapWidth.toFixed(3));
        }}
        const screenDash = (path.dataset.screenDash || '')
          .trim().split(/\\s+/).map(Number);
        if (screenDash.length && screenDash.every(Number.isFinite)) {{
          path.setAttribute(
            'stroke-dasharray',
            screenDash.map((length) => (length * screenMapFactor).toFixed(3)).join(' '),
          );
        }}
      }});
      const placed = [];
      const placeLabel = (label, eligible, horizontalMargin = 3, verticalMargin = 2) => {{
        label.style.display = eligible ? '' : 'none';
        if (!eligible) return;
        const fontSize = Number(label.dataset.baseFontSize);
        const minimumScreenFontSize = Number(label.dataset.minScreenFontSize);
        const maximumScreenFontSize = Number(label.dataset.maxScreenFontSize);
        const textStroke = Number(label.dataset.baseTextStroke);
        const safeScale = Math.max(0.01, scale);
        let sourceFontSize = fontSize;
        if (Number.isFinite(fontSize)) {{
          // Screen text grows with the map instead of becoming relatively
          // smaller against magnified terrain.
          const naturalScreenSize = fontSize * Math.pow(Math.max(1, safeScale), 0.48);
          const effectiveMaximum = Number.isFinite(maximumScreenFontSize)
            ? maximumScreenFontSize
            : fontSize * 3.0;
          const clampedScreenSize = Math.min(
            effectiveMaximum,
            Math.max(
              Number.isFinite(minimumScreenFontSize) ? minimumScreenFontSize : 9,
              naturalScreenSize,
            ),
          );
          sourceFontSize = clampedScreenSize / safeScale;
          label.setAttribute('font-size', sourceFontSize.toFixed(3));
        }}
        if (Number.isFinite(textStroke)) {{
          const glyphScale = Number.isFinite(fontSize) && fontSize > 0
            ? sourceFontSize / fontSize
            : 1 / safeScale;
          label.setAttribute('stroke-width', (textStroke * glyphScale).toFixed(3));
        }}
        const curvedText = label.querySelector('textPath[data-base-text-length]');
        if (curvedText && Number.isFinite(fontSize) && fontSize > 0) {{
          const baseTextLength = Number(curvedText.dataset.baseTextLength);
          if (Number.isFinite(baseTextLength)) {{
            curvedText.setAttribute(
              'textLength',
              (baseTextLength * sourceFontSize / fontSize).toFixed(3),
            );
          }}
        }}
        const rect = label.getBoundingClientRect();
        const collision = placed.some((other) => !(
          rect.right + horizontalMargin < other.left || rect.left - horizontalMargin > other.right ||
          rect.bottom + verticalMargin < other.top || rect.top - verticalMargin > other.bottom
        ));
        if (collision) label.style.display = 'none';
        else placed.push(rect);
      }};

      const physical = activeView === 'physical';
      if (physical && ['political','provinces'].includes(activeTheme)) {{
        const thresholds = activeTheme === 'provinces'
          ? {{ major: 0, medium: 1.08, small: 2.05 }}
          : {{ major: 0, medium: 1.28, small: 2.45 }};
        const stateLabelMaximum = activeTheme === 'provinces' ? 5.4 : 10.5;
        document.querySelectorAll('#state-labels [data-state-size]').forEach((label) => {{
          placeLabel(
            label,
            scale >= thresholds[label.dataset.stateSize] && scale < stateLabelMaximum,
            5,
            3,
          );
        }});
        const frontierThresholds = {{ major: 0, secondary: 1.42 }};
        document.querySelectorAll('#frontier-group-labels [data-frontier-tier]').forEach((label) => {{
          placeLabel(
            label,
            scale >= frontierThresholds[label.dataset.frontierTier] && scale < 8.5,
            7,
            4,
          );
        }});
      }} else if (physical && ['civilizations','languages','religions'].includes(activeTheme)) {{
        const minimumScale = activeTheme === 'languages' ? 1.18 : 0;
        const maximumScale = activeTheme === 'languages' ? 7.2 : 5.6;
        const thematicLabelId = activeTheme === 'civilizations'
          ? 'civilization'
          : activeTheme === 'languages' ? 'language' : 'religion';
        document.querySelectorAll(`#${{thematicLabelId}}-labels [data-theme-label]`).forEach((label) => {{
          placeLabel(label, scale >= minimumScale && scale < maximumScale, 5, 3);
        }});
      }}

      // Province names are the primary reading layer in the province theme.
      // Reserve their deep-interior anchors before geographic and city labels
      // compete for the remaining screen space.
      const provinceLabels = Array.from(
        document.querySelectorAll('#province-labels [data-province-id]'),
      );
      if (physical && activeTheme === 'provinces') {{
        provinceLabels.forEach((label) => {{
          placeLabel(label, scale >= 0.98, 5, 3);
        }});
      }} else {{
        provinceLabels.forEach((label) => {{ label.style.display = 'none'; }});
      }}

      const geographicLabelsEnabled = physical && document.getElementById('toggle-geographic-labels').checked;
      const geographicPriority = {{
        mountain: 0, sea: 1, 'inland-sea': 1,
        'island-group': 2, island: 3, river: 4,
        plateau: 5, plain: 5, basin: 5, lake: 6,
        bay: 7, strait: 7, peak: 8,
      }};
      const geographicTierPriority = {{ major: 0, secondary: 1, detail: 2 }};
      const geographicLabels = Array.from(
        document.querySelectorAll('#geographic-labels text[data-feature-tier]'),
      ).sort((first, second) => {{
        const tierDelta = geographicTierPriority[first.dataset.featureTier]
          - geographicTierPriority[second.dataset.featureTier];
        if (tierDelta) return tierDelta;
        return (geographicPriority[first.dataset.featureType] ?? 4)
          - (geographicPriority[second.dataset.featureType] ?? 4);
      }});
      geographicLabels.forEach((label) => {{
        const minimumScale = Number(label.dataset.minVisibleScale || 0);
        const maximumScale = Number(label.dataset.maxVisibleScale || 99);
        const isRiver = label.dataset.featureType === 'river';
        placeLabel(
          label,
          geographicLabelsEnabled && scale >= minimumScale && scale < maximumScale,
          isRiver ? 18 : 4,
          isRiver ? 9 : 2,
        );
      }});

      const labelsEnabled = physical && (
        document.getElementById('toggle-cities').checked ||
        document.getElementById('toggle-transport').checked
      );
      const transportEnabled = physical && document.getElementById('toggle-transport').checked;
      const labelProfiles = {{
        provinces: {{ metropolis: 0, city: 1.05, town: 1.78, site: 1.28 }},
        political: {{ metropolis: 0, city: 1.28, town: 2.35, site: 1.52 }},
        civilizations: {{ metropolis: 0.72, city: 1.55, town: 2.85, site: 1.85 }},
        languages: {{ metropolis: 0.72, city: 1.55, town: 2.85, site: 1.85 }},
        religions: {{ metropolis: 0.62, city: 1.42, town: 2.72, site: 1.72 }},
        default: {{ metropolis: 0.82, city: 1.48, town: 2.65, site: 1.78 }},
      }};
      const thresholds = labelProfiles[activeTheme] || labelProfiles.default;
      document.querySelectorAll('#city-labels [data-city-label-tier]').forEach((label) => {{
        const tier = label.dataset.cityLabelTier;
        const x = Number(label.dataset.mapX);
        const y = Number(label.dataset.mapY);
        const offsetX = Number(label.dataset.baseOffsetX);
        const offsetY = Number(label.dataset.baseOffsetY);
        const markerScale = scale <= 1 ? 1 : Math.pow(scale, -0.62);
        if ([x, y, offsetX, offsetY].every(Number.isFinite)) {{
          label.setAttribute('x', (x + offsetX * markerScale).toFixed(3));
          label.setAttribute('y', (y + offsetY * markerScale).toFixed(3));
        }}
        placeLabel(label, labelsEnabled && scale >= thresholds[tier]);
      }});
      const symbolProfiles = {{
        provinces: {{ metropolis: 0, city: 0.78, town: 1.42, site: 1.12 }},
        political: {{ metropolis: 0, city: 0.96, town: 1.92, site: 1.28 }},
        civilizations: {{ metropolis: 0.55, city: 1.18, town: 2.25, site: 1.55 }},
        languages: {{ metropolis: 0.55, city: 1.18, town: 2.25, site: 1.55 }},
        religions: {{ metropolis: 0.48, city: 1.06, town: 2.12, site: 1.42 }},
        default: {{ metropolis: 0.62, city: 1.08, town: 2.08, site: 1.46 }},
      }};
      const symbolThresholds = symbolProfiles[activeTheme] || symbolProfiles.default;
      document.querySelectorAll('#city-symbols [data-city-symbol-tier]').forEach((symbol) => {{
        const tier = symbol.dataset.citySymbolTier;
        const eligible = labelsEnabled && scale >= symbolThresholds[tier];
        symbol.style.display = eligible ? '' : 'none';
        if (!eligible) return;
        const x = Number(symbol.dataset.mapX);
        const y = Number(symbol.dataset.mapY);
        const symbolScale = scale <= 1 ? 1 : Math.pow(scale, -0.62);
        symbol.setAttribute(
          'transform',
          `translate(${{x.toFixed(2)}} ${{y.toFixed(2)}}) scale(${{symbolScale.toFixed(4)}})`,
        );
      }});
      const routeThresholds = activeTheme === 'provinces'
        ? {{ trunk: 0, regional: 0.96, local: 1.82 }}
        : activeTheme === 'political'
          ? {{ trunk: 0, regional: 1.08, local: 2.18 }}
          : {{ trunk: 0, regional: 1.22, local: 2.48 }};
      document.querySelectorAll('#transport-network [data-route-importance]').forEach((path) => {{
        path.style.display = transportEnabled && scale >= routeThresholds[path.dataset.routeImportance]
          ? ''
          : 'none';
      }});
      document.querySelectorAll('#bridge-layer [data-bridge-importance]').forEach((symbol) => {{
        const eligible = transportEnabled && scale >= routeThresholds[symbol.dataset.bridgeImportance];
        symbol.style.display = eligible ? '' : 'none';
        if (!eligible) return;
        const x = Number(symbol.dataset.mapX);
        const y = Number(symbol.dataset.mapY);
        const angle = Number(symbol.dataset.baseAngle);
        const symbolScale = scale <= 1 ? 1 : Math.pow(scale, -0.62);
        symbol.setAttribute(
          'transform',
          `translate(${{x.toFixed(2)}} ${{y.toFixed(2)}}) rotate(${{angle.toFixed(2)}}) scale(${{symbolScale.toFixed(4)}})`,
        );
      }});
    }}

    function scheduleAdaptiveUpdate(immediate = false) {{
      window.clearTimeout(adaptiveUpdateTimer);
      if (immediate) {{
        updateAdaptiveStrokes();
        return;
      }}
      adaptiveUpdateTimer = window.setTimeout(updateAdaptiveStrokes, 120);
    }}

    function flushCameraTransform() {{
      cameraFrame = 0;
      scene.style.transform = `translate3d(${{translateX}}px, ${{translateY}}px, 0) scale(${{scale}})`;
    }}

    function requestCameraTransform() {{
      if (!cameraFrame) cameraFrame = window.requestAnimationFrame(flushCameraTransform);
    }}

    function applyTransform() {{
      if (cameraFrame) window.cancelAnimationFrame(cameraFrame);
      flushCameraTransform();
      const levelNames = {{ world: '全图', regional: '区域', local: '地方', close: '近景' }};
      const detailLevel = scale < 0.85 ? 'world' : scale < 2.4 ? 'regional' : scale < 6 ? 'local' : 'close';
      document.documentElement.dataset.mapDetail = detailLevel;
      const zoomLevel = document.getElementById('zoom-level');
      zoomLevel.value = levelNames[detailLevel];
      zoomLevel.textContent = levelNames[detailLevel];
      zoomLevel.title = `当前缩放 ${{Math.round(scale * 100)}}%`;
      scheduleAdaptiveUpdate(false);
    }}

    function zoomAt(factor, clientX, clientY) {{
      const bounds = viewport.getBoundingClientRect();
      const cursorX = (clientX - bounds.left - translateX) / scale;
      const cursorY = (clientY - bounds.top - translateY) / scale;
      const nextScale = Math.min(maximumScale, Math.max(minimumScale, scale * factor));
      translateX = clientX - bounds.left - cursorX * nextScale;
      translateY = clientY - bounds.top - cursorY * nextScale;
      scale = nextScale;
      cameraInteracted = true;
      applyTransform();
    }}

    function zoomAtCenter(factor) {{
      const bounds = viewport.getBoundingClientRect();
      zoomAt(factor, bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
    }}

    function resetView() {{
      cameraInteracted = false;
      fitScene();
      applyTransform();
    }}

    function fitScene() {{
      const bounds = viewport.getBoundingClientRect();
      const width = {width};
      const height = {scene_height};
      scale = Math.min(1, bounds.width / width, bounds.height / height);
      minimumScale = scale;
      translateX = Math.max(0, (bounds.width - width * scale) / 2);
      translateY = Math.max(0, (bounds.height - height * scale) / 2);
    }}

    function setElementVisible(element, visible) {{
      if (!element) return;
      element.toggleAttribute('hidden', !visible);
      element.style.display = visible ? '' : 'none';
    }}

    function physicalToggleEnabled(layerId) {{
      const control = document.querySelector(`[data-layer="${{layerId}}"]`);
      return !control || control.checked;
    }}

    function updateRiverAppearance() {{
      const riverGroup = document.getElementById('rivers');
      if (!riverGroup) return;
      if (activeView === 'tectonic') {{
        setElementVisible(riverGroup, false);
        return;
      }}
      const seasonIndex = seasons.indexOf(activeSeason);
      const seasonalEnabled = document.getElementById('toggle-seasonal-rivers').checked;
      if (activeView === 'physical') {{
        setElementVisible(riverGroup, physicalToggleEnabled('rivers'));
        riverGroup.querySelectorAll('path').forEach((path) => {{
          const strengths = (path.dataset.seasonalStrengths || '0,0,0,0').split(',').map(Number);
          const disappears = strengths.some((strength) => strength <= 0);
          const base = Number(path.dataset.physicalWidth || path.getAttribute('stroke-width'));
          path.style.display = '';
          path.setAttribute('stroke', path.dataset.physicalStroke || '{_RIVER_COLOR}');
          path.setAttribute('stroke-width', path.dataset.physicalWidth || path.getAttribute('stroke-width'));
          path.setAttribute(
            'stroke-dasharray',
            disappears
              ? `${{Math.max(1, base * 2.4).toFixed(3)}} ${{Math.max(0.8, base * 1.8).toFixed(3)}}`
              : 'none',
          );
          path.setAttribute('opacity', '1');
        }});
        return;
      }}
      setElementVisible(riverGroup, seasonalEnabled);
      riverGroup.querySelectorAll('path').forEach((path) => {{
        const strengths = (path.dataset.seasonalStrengths || '0,0,0,0').split(',').map(Number);
        const strength = Number.isFinite(strengths[seasonIndex]) ? strengths[seasonIndex] : 0;
        const disappears = strengths.some((value) => value <= 0);
        path.style.display = strength > 0 ? '' : 'none';
        path.setAttribute('stroke', '{_RIVER_COLOR}');
        const base = Number(path.dataset.physicalWidth || path.getAttribute('stroke-width'));
        path.setAttribute('stroke-width', (base * (0.42 + strength / 255 * 0.92)).toFixed(3));
        path.setAttribute(
          'stroke-dasharray',
          disappears
            ? `${{Math.max(1, base * 2.4).toFixed(3)}} ${{Math.max(0.8, base * 1.8).toFixed(3)}}`
            : 'none',
        );
        path.setAttribute('opacity', strength >= 48 ? '0.96' : '0.68');
      }});
    }}

    function updateThematicState() {{
      const monsoon = activeView === 'monsoon';
      const tectonic = activeView === 'tectonic';
      const physical = activeView === 'physical';
      document.getElementById('physical-controls').hidden = !physical;
      document.getElementById('monsoon-controls').hidden = !monsoon;
      document.getElementById('tectonic-controls').hidden = !tectonic;
      document.getElementById('physical-theme').value = activeTheme;
      document.querySelectorAll('[data-theme-detail]').forEach((detail) => {{
        detail.hidden = !physical || detail.dataset.themeDetail !== activeTheme;
      }});
      document.querySelectorAll('[data-view-button]').forEach((button) => {{
        button.setAttribute('aria-pressed', String(button.dataset.viewButton === activeView));
      }});
      document.querySelectorAll('[data-season-button]').forEach((button) => {{
        button.setAttribute('aria-pressed', String(button.dataset.seasonButton === activeSeason));
      }});

      const precipitationEnabled = document.getElementById('toggle-monsoon-precipitation').checked;
      document.querySelectorAll('.monsoon-precipitation').forEach((image) => {{
        setElementVisible(image, monsoon && precipitationEnabled && image.dataset.season === activeSeason);
      }});
      const windEnabled = document.getElementById('toggle-monsoon-wind').checked;
      document.querySelectorAll('.seasonal-wind').forEach((group) => {{
        setElementVisible(group, monsoon && windEnabled && group.dataset.season === activeSeason);
      }});
      document.querySelectorAll('[data-theme-map]').forEach((image) => {{
        const visible = physical && image.dataset.themeMap === activeTheme;
        if (visible && !image.getAttribute('href')) image.setAttribute('href', image.dataset.source);
        setElementVisible(image, visible);
      }});
      const tectonicMap = document.getElementById('tectonic-map');
      if (tectonic && !tectonicMap.getAttribute('href')) tectonicMap.setAttribute('href', tectonicMap.dataset.source);
      setElementVisible(tectonicMap, tectonic);

      // A thematic map must read as the primary map, not as a faint tint
      // fighting the hypsometric palette. Preserve physical evidence as a
      // restrained relief underlay while a theme is active.
      const thematicActive = physical && activeTheme !== 'none';
      const politicalActive = physical && activeTheme === 'political';
      const provinceActive = physical && activeTheme === 'provinces';
      const administrativeActive = politicalActive || provinceActive;
      document.documentElement.dataset.activeTheme = physical ? activeTheme : activeView;
      const elevationBands = document.getElementById('elevation-bands');
      const bathymetryBands = document.getElementById('bathymetry-bands');
      const elevationContours = document.getElementById('elevation-contours');
      if (elevationBands) elevationBands.setAttribute('opacity', administrativeActive ? '0.16' : thematicActive ? '0.34' : '1');
      if (bathymetryBands) bathymetryBands.setAttribute('opacity', administrativeActive ? '0.72' : thematicActive ? '0.62' : '1');
      if (elevationContours) elevationContours.setAttribute('opacity', administrativeActive ? '0.18' : thematicActive ? '0.42' : '1');

      const citiesEnabled = physical && document.getElementById('toggle-cities').checked;
      const transportEnabled = physical && document.getElementById('toggle-transport').checked;
      setElementVisible(document.getElementById('city-layer'), citiesEnabled || transportEnabled);
      setElementVisible(document.getElementById('transport-network'), transportEnabled);
      setElementVisible(document.getElementById('bridge-layer'), transportEnabled);
      setElementVisible(
        document.getElementById('society-thematic-dimmer'),
        activeTheme === 'none' && (citiesEnabled || transportEnabled),
      );
      setElementVisible(document.getElementById('civilization-labels'), physical && activeTheme === 'civilizations');
      setElementVisible(document.getElementById('language-labels'), physical && activeTheme === 'languages');
      setElementVisible(document.getElementById('religion-labels'), physical && activeTheme === 'religions');
      setElementVisible(document.getElementById('civilization-boundaries'), physical && ['civilizations','languages'].includes(activeTheme));
      setElementVisible(document.getElementById('language-boundaries'), physical && activeTheme === 'languages');
      setElementVisible(document.getElementById('language-civilization-boundaries'), physical && activeTheme === 'languages');
      setElementVisible(document.getElementById('religion-boundaries'), physical && activeTheme === 'religions');
      setElementVisible(document.getElementById('state-labels'), administrativeActive);
      setElementVisible(document.getElementById('province-labels'), provinceActive);
      setElementVisible(document.getElementById('frontier-group-labels'), administrativeActive);
      setElementVisible(document.getElementById('province-boundaries'), provinceActive);
      setElementVisible(document.getElementById('state-boundaries'), administrativeActive);
      setElementVisible(document.getElementById('surface-backfill'), !tectonic);

      for (const layerId of ['elevation-bands','bathymetry-bands','graticule','coast','lakes','inland-seas','sea-ice','polar-references']) {{
        setElementVisible(document.getElementById(layerId), !tectonic && physicalToggleEnabled(layerId));
      }}
      for (const layerId of ['elevation-contours','snow']) {{
        setElementVisible(document.getElementById(layerId), physical && physicalToggleEnabled(layerId));
      }}
      setElementVisible(
        document.getElementById('polar-labels'),
        physical && physicalToggleEnabled('polar-references'),
      );
      setElementVisible(
        document.getElementById('geographic-labels'),
        physical && physicalToggleEnabled('geographic-labels'),
      );
      updateRiverAppearance();
      updateAdaptiveStrokes();

      const nextQuery = new URLSearchParams(window.location.search);
      if (monsoon) {{
        nextQuery.set('view', 'monsoon');
        nextQuery.set('season', activeSeason);
        nextQuery.delete('theme');
      }} else if (tectonic) {{
        nextQuery.set('view', 'tectonic');
        nextQuery.delete('season');
        nextQuery.delete('theme');
      }} else {{
        nextQuery.delete('view');
        nextQuery.delete('season');
        if (activeTheme === 'none') nextQuery.delete('theme');
        else nextQuery.set('theme', activeTheme);
      }}
      const suffix = nextQuery.toString();
      window.history.replaceState(null, '', `${{window.location.pathname}}${{suffix ? '?' + suffix : ''}}`);
    }}

    document.querySelectorAll('[data-layer]').forEach((control) => {{
      control.addEventListener('change', () => {{
        updateThematicState();
      }});
    }});
    document.getElementById('physical-theme').addEventListener('change', (event) => {{
      activeTheme = event.target.value;
      updateThematicState();
    }});
    document.querySelectorAll('[data-view-button]').forEach((button) => {{
      button.addEventListener('click', () => {{
        activeView = button.dataset.viewButton;
        updateThematicState();
      }});
    }});
    document.querySelectorAll('[data-season-button]').forEach((button) => {{
      button.addEventListener('click', () => {{
        activeSeason = button.dataset.seasonButton;
        updateThematicState();
      }});
    }});
    for (const controlId of ['toggle-monsoon-precipitation','toggle-monsoon-wind','toggle-seasonal-rivers']) {{
      document.getElementById(controlId).addEventListener('change', updateThematicState);
    }}

    viewport.addEventListener('wheel', (event) => {{
      event.preventDefault();
      zoomAt(event.deltaY < 0 ? 1.14 : 1 / 1.14, event.clientX, event.clientY);
    }}, {{ passive: false }});

    document.getElementById('zoom-in').addEventListener('click', () => zoomAtCenter(1.5));
    document.getElementById('zoom-out').addEventListener('click', () => zoomAtCenter(1 / 1.5));
    document.getElementById('zoom-reset').addEventListener('click', resetView);
    document.querySelector('.map-tools').addEventListener('pointerdown', (event) => {{
      event.stopPropagation();
    }});
    viewport.addEventListener('dblclick', (event) => {{
      event.preventDefault();
      zoomAt(1.5, event.clientX, event.clientY);
    }});
    viewport.addEventListener('keydown', (event) => {{
      if (event.key === '+' || event.key === '=') {{
        event.preventDefault();
        zoomAtCenter(1.5);
      }} else if (event.key === '-' || event.key === '_') {{
        event.preventDefault();
        zoomAtCenter(1 / 1.5);
      }} else if (event.key === '0') {{
        event.preventDefault();
        resetView();
      }}
    }});

    viewport.addEventListener('pointerdown', (event) => {{
      dragging = true;
      cameraInteracted = true;
      viewport.focus({{ preventScroll: true }});
      lastX = event.clientX;
      lastY = event.clientY;
      viewport.classList.add('dragging');
      window.clearTimeout(adaptiveUpdateTimer);
      viewport.setPointerCapture(event.pointerId);
    }});
    viewport.addEventListener('pointermove', (event) => {{
      if (!dragging) return;
      translateX += event.clientX - lastX;
      translateY += event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      requestCameraTransform();
    }});
    function stopDragging() {{
      dragging = false;
      viewport.classList.remove('dragging');
      if (cameraFrame) window.cancelAnimationFrame(cameraFrame);
      flushCameraTransform();
      scheduleAdaptiveUpdate(true);
    }}
    viewport.addEventListener('pointerup', stopDragging);
    viewport.addEventListener('pointercancel', stopDragging);
    fitScene();
    updateThematicState();
    applyTransform();
    window.addEventListener('resize', () => {{
      if (!dragging && !cameraInteracted) {{
        fitScene();
        applyTransform();
      }}
    }});
  </script>
</body>
</html>
'''


def render_review(grid: WorldGrid, output_dir: str | Path, *, society: SocietyLayers | None = None) -> dict[str, Any]:
    """Write a raster export and an inline SVG review page from canonical arrays.

    The raster is retained as an export/debug artifact; the visible map base
    is the generated SVG band geometry assembled below.
    """

    if not isinstance(grid, WorldGrid):
        raise TypeError("render_review requires a WorldGrid")
    if society is not None and society.population.population_weight.shape != grid.shape:
        raise ValueError("saved society must match the physical grid shape")
    output_dir = Path(output_dir)
    _reject_reparse_paths(output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError("render output path must be a directory")
        if any(output_dir.iterdir()):
            raise ValueError(
                "render output directory must be nonexistent or empty"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    terrain_path = output_dir / "terrain.png"
    contours_path = output_dir / "elevation-contours.svg"
    index_path = output_dir / "index.html"
    qa_path = output_dir / "qa.json"
    climate_path = output_dir / "climate.svg"
    biome_path = output_dir / "biome.svg"
    watersheds_path = output_dir / "watersheds.svg"
    land_potential_path = output_dir / "land-potential.svg"
    population_path = output_dir / "population.svg"
    civilizations_path = output_dir / "civilizations.svg"
    languages_path = output_dir / "languages.svg"
    religions_path = output_dir / "religions.svg"
    political_path = output_dir / "political.svg"
    provinces_path = output_dir / "provinces.svg"
    tectonic_path = output_dir / "tectonic.svg"
    society_npz_path = output_dir / "society.npz"
    society_json_path = output_dir / "society.json"
    precipitation_paths = tuple(
        output_dir / f"monsoon-precipitation-{season_id}.png"
        for season_id in _SEASON_IDS
    )
    digest = grid.content_digest()
    display_width, display_height, _scale_x, _scale_y = _review_dimensions(grid)

    coast_paths = _naturalize_surface_paths(_mask_paths(grid.water == 1))
    lake_paths = _naturalize_surface_paths(_mask_paths(grid.water == 2))
    inland_sea_paths = _naturalize_surface_paths(_mask_paths(grid.water == 3))
    contour_paths, contour_path_levels = _elevation_paths(grid)
    snow_paths = _naturalize_surface_paths(_mask_paths(grid.snow))
    polar_land_mask = polar_continent_mask(grid)
    polar_land_paths = _naturalize_surface_paths(_mask_paths(polar_land_mask))
    sea_ice_paths = _naturalize_surface_paths(_mask_paths(grid.sea_ice))
    river_paths, river_stroke_widths = _river_width_segments(
        grid,
        _river_paths(grid),
    )
    if river_paths:
        # Paint smaller tributaries first so the wider receiving channel owns
        # the confluence join instead of being capped by a thin source reach.
        river_order = sorted(
            range(len(river_paths)),
            key=lambda index: (river_stroke_widths[index], index),
        )
        river_paths = [river_paths[index] for index in river_order]
        river_stroke_widths = [river_stroke_widths[index] for index in river_order]
    river_seasonal_strengths = _river_seasonal_strengths(grid, river_paths)
    river_orders = []
    for path in river_paths:
        columns = np.clip(np.floor(path[:, 0]).astype(int), 0, grid.shape[1] - 1)
        rows = np.clip(np.floor(path[:, 1]).astype(int), 0, grid.shape[0] - 1)
        river_orders.append(int(np.max(grid.river_order[rows, columns], initial=0)))
    river_path_attributes = [
        {
            "seasonal-strengths": ",".join(str(value) for value in strengths),
            "river-order": river_orders[index],
            "physical-width": f"{river_stroke_widths[index]:.3f}",
            "physical-stroke": _RIVER_COLOR,
        }
        for index, strengths in enumerate(river_seasonal_strengths)
    ]
    wind_overlay, wind_arrow_counts = _wind_arrow_groups(grid)
    graticule_paths, graticule_major_flags, graticule_labels = _graticule_paths(grid)
    elevation_palette, _ = _elevation_palette_for(grid)
    water_palette, _ = _bathymetry_palette_for(grid)
    # A single opaque backfill closes the half-cell gaps that marching squares
    # intentionally leaves at mixed categorical edges.  The land mask is then
    # painted back over the deep-ocean base before the ordinal bands are drawn;
    # therefore a missing edge fragment can only reveal the correct surface
    # family, never the beige viewport or the opposite surface.
    land_surface_paths = _naturalize_filled_surface_paths(
        _filled_mask_paths(grid.water == 0)
    )
    elevation_band_paths = _elevation_band_paths(grid)
    bathymetry_band_paths = [
        _naturalize_filled_surface_paths(paths)
        for paths in _filled_band_paths(
            grid.bathymetry_band.astype(np.float64),
            np.isin(grid.water, (1, 3)),
            len(water_palette),
            higher_values_on_top=False,
        )
    ]
    lake_fill_paths = _naturalize_filled_surface_paths(
        _filled_mask_paths(grid.water == 2)
    )
    inland_sea_fill_paths: list[tuple[np.ndarray, np.ndarray]] = []
    snow_fill_paths = _naturalize_filled_surface_paths(
        _filled_mask_paths(grid.snow & (grid.water == 0))
    )
    polar_land_fill_paths = _naturalize_filled_surface_paths(
        _filled_mask_paths(polar_land_mask)
    )
    sea_ice_fill_paths = _naturalize_filled_surface_paths(
        _filled_mask_paths(grid.sea_ice)
    )
    thematic = derive_thematic_layers(grid)
    tectonics = derive_tectonic_review(grid)
    tectonic_svg = render_tectonic_svg(grid, tectonics)
    name_source, society_request = _society_generation_request(grid)
    if society is None:
        society = derive_society_layers(grid, thematic, name_source, **society_request)
    territorial_qa = _territorial_quality_metrics(grid, thematic, society)
    climate_zones = thematic.climate.climate_zone
    land_mask = grid.water == 0
    climate_zone_paths = _categorical_partition_paths(
        climate_zones,
        land_mask,
        category_count=len(_CLIMATE_ZONES),
    )
    biome_zone_paths = _categorical_partition_paths(
        thematic.biome_zone,
        land_mask,
        category_count=BIOME_COUNT,
    )
    watershed_zone_paths = _categorical_partition_paths(
        thematic.major_basin_rank,
        land_mask,
        category_count=len(_WATERSHED_ZONES),
    )
    potential_zone_paths = _categorical_partition_paths(
        thematic.land_potential_band,
        land_mask,
        category_count=len(_LAND_POTENTIAL_ZONES),
    )
    population_zone_paths = _categorical_partition_paths(
        society.population.population_band,
        land_mask,
        category_count=len(_POPULATION_ZONES),
    )
    civilization_zones = _culture_zones(society)
    civilization_display = _civilization_display_values(grid, thematic, society)
    civilization_faces, civilization_face_ids = _shared_partition_topology(
        civilization_display,
        land_mask,
        category_count=len(civilization_zones),
        simplification_tolerance=_PARTITION_SIMPLIFICATION_TOLERANCE,
    )
    _validate_partition_inventory(
        civilization_display,
        land_mask,
        civilization_face_ids,
        layer_name="civilization",
    )
    civilization_zone_paths = _shared_topology_zone_paths(
        civilization_faces,
        civilization_face_ids,
        category_count=len(civilization_zones),
        include_zero=True,
    )
    language_zones = _language_zones(society)
    language_faces, language_face_ids = _shared_partition_topology(
        society.cultures.language_id,
        land_mask,
        category_count=len(language_zones),
        simplification_tolerance=_PARTITION_SIMPLIFICATION_TOLERANCE,
    )
    _validate_partition_inventory(
        society.cultures.language_id,
        land_mask,
        language_face_ids,
        layer_name="language",
    )
    language_zone_paths = _shared_topology_zone_paths(
        language_faces,
        language_face_ids,
        category_count=len(language_zones),
        include_zero=True,
    )
    civilization_boundary_paths = _shared_topology_boundary_paths(
        civilization_faces,
        civilization_face_ids,
    )
    language_boundary_paths = _shared_topology_boundary_paths(
        language_faces,
        language_face_ids,
    )
    religion_zones = _religion_zones(society)
    religion_faces, religion_face_ids = _shared_partition_topology(
        society.religions.religion_id,
        land_mask,
        category_count=len(religion_zones),
        simplification_tolerance=_PARTITION_SIMPLIFICATION_TOLERANCE,
    )
    _validate_partition_inventory(
        society.religions.religion_id,
        land_mask,
        religion_face_ids,
        layer_name="religion",
    )
    religion_zone_paths = _shared_topology_zone_paths(
        religion_faces,
        religion_face_ids,
        category_count=len(religion_zones),
        include_zero=True,
    )
    religion_boundary_paths = _shared_topology_boundary_paths(
        religion_faces,
        religion_face_ids,
    )
    political_zones = _political_zones(society)
    province_zones = _province_zones(society)
    administrative_faces, province_face_ids = _shared_partition_topology(
        society.provinces.province_id,
        land_mask,
        category_count=len(province_zones),
        simplification_tolerance=_PARTITION_SIMPLIFICATION_TOLERANCE,
    )
    _validate_partition_inventory(
        society.provinces.province_id,
        land_mask,
        province_face_ids,
        layer_name="province",
    )
    province_to_state = np.zeros(len(province_zones), dtype=np.int32)
    for province in society.provinces.provinces:
        province_to_state[province.identifier] = province.state_identifier
    state_face_ids = province_to_state[province_face_ids]
    political_zone_paths = _shared_topology_zone_paths(
        administrative_faces,
        state_face_ids,
        category_count=len(political_zones),
    )
    province_zone_paths = _shared_topology_zone_paths(
        administrative_faces,
        province_face_ids,
        category_count=len(province_zones),
    )
    state_boundary_paths = _shared_topology_boundary_paths(
        administrative_faces,
        state_face_ids,
        include_unassigned=True,
    )
    province_boundary_paths = _shared_topology_boundary_paths(
        administrative_faces,
        province_face_ids,
    )
    state_river_dividers = _partition_river_dividers(
        grid.river_order,
        land_mask,
        river_paths,
        society.politics.state_id,
        minimum_order=2,
        include_unassigned=True,
    )
    province_river_dividers = _partition_river_dividers(
        grid.river_order,
        land_mask,
        river_paths,
        society.provinces.province_id,
        minimum_order=2,
    )
    state_boundary_paths = _remove_offset_boundaries_along_rivers(
        state_boundary_paths,
        state_river_dividers,
    )
    province_boundary_paths = _remove_offset_boundaries_along_rivers(
        province_boundary_paths,
        province_river_dividers,
    )
    territorial_qa.update(
        {
            "riverRenderDeviationMaximum": 0.0,
            "stateRiverAxisSegments": int(
                sum(max(0, len(path) - 1) for path in state_river_dividers)
            ),
            "provinceRiverAxisSegments": int(
                sum(max(0, len(path) - 1) for path in province_river_dividers)
            ),
        }
    )
    frontier_fill_paths = _geometry_filled_paths(
        shapely.union_all(
            [
                face
                for face, state_identifier in zip(
                    administrative_faces,
                    state_face_ids,
                    strict=True,
                )
                if state_identifier == 0
            ]
        )
    )
    vector_band_paths = (
        _flatten_filled_paths(elevation_band_paths)
        + _flatten_filled_paths(bathymetry_band_paths)
        + [points for points, _ in land_surface_paths]
        + [points for points, _ in lake_fill_paths]
        + [points for points, _ in snow_fill_paths]
        + [points for points, _ in polar_land_fill_paths]
        + [points for points, _ in sea_ice_fill_paths]
    )
    packed_vector_paths = (
        _packed_band_budget_paths(elevation_band_paths)
        + _packed_band_budget_paths(bathymetry_band_paths)
        + [points for points, _ in land_surface_paths]
        + [points for points, _ in lake_fill_paths]
        + [points for points, _ in snow_fill_paths]
        + [points for points, _ in sea_ice_fill_paths]
    )
    # Keep the safety budget focused on physical overlays.  Graticule lines
    # are a fixed, bounded review aid (52 paths at the default global extent)
    # and must not make a fragmented physical layer fail with an opaque count.
    _validate_svg_budget(
        (
            coast_paths,
            lake_paths,
            inland_sea_paths,
            snow_paths,
            polar_land_paths,
            sea_ice_paths,
            river_paths,
            vector_band_paths,
        ),
        path_layers=(
            coast_paths,
            lake_paths,
            inland_sea_paths,
            snow_paths,
            polar_land_paths,
            sea_ice_paths,
            river_paths,
            packed_vector_paths,
        ),
    )
    # Contours live in elevation-contours.svg, not the inline physical SVG.
    # Each document gets its own unchanged DOM/point budget; aggregating them
    # incorrectly rejects a detailed world that neither document overloads.
    _validate_svg_budget((contour_paths,))
    _validate_svg_budget((graticule_paths,))
    for partition_paths in (
        climate_zone_paths,
        biome_zone_paths,
        watershed_zone_paths,
        potential_zone_paths,
        population_zone_paths,
        civilization_zone_paths,
        language_zone_paths,
        religion_zone_paths,
        political_zone_paths,
        province_zone_paths,
    ):
        _validate_svg_budget(
            (_flatten_filled_paths(partition_paths),),
            path_layers=(
                [
                    np.empty((1, 2), dtype=np.float64)
                    for paths in partition_paths
                    if paths
                ],
            ),
        )
    shallow_ocean_ordinal = _shallow_ocean_ordinal(grid, len(water_palette))
    shallow_water_color = _rgb_hex(water_palette[shallow_ocean_ordinal])
    deep_water_color = _rgb_hex(water_palette[-1])
    lowland_color = _rgb_hex(elevation_palette[0])
    climate_svg = _partition_overlay_svg_document(
        grid,
        climate_zone_paths,
        title="世界气候图",
        partition_id="climate-zones",
        data_attribute="climate-zone",
        zones=_CLIMATE_ZONES,
        fill_opacity=0.82,
    )
    biome_svg = _partition_overlay_svg_document(
        grid,
        biome_zone_paths,
        title="世界生物群系图",
        partition_id="biome-zones",
        data_attribute="biome-zone",
        zones=_BIOME_ZONES,
        fill_opacity=0.80,
    )
    watersheds_svg = _partition_overlay_svg_document(
        grid,
        watershed_zone_paths,
        title="世界水文流域图",
        partition_id="major-watersheds",
        data_attribute="basin-rank",
        zones=_WATERSHED_ZONES,
        fill_opacity=0.70,
    )
    land_potential_svg = _partition_overlay_svg_document(
        grid,
        potential_zone_paths,
        title="世界农业与宜居潜力图",
        partition_id="land-potential-bands",
        data_attribute="potential-band",
        zones=_LAND_POTENTIAL_ZONES,
        fill_opacity=0.76,
    )
    population_svg = _partition_overlay_svg_document(
        grid,
        population_zone_paths,
        title="世界人口分布图",
        partition_id="population-bands",
        data_attribute="population-band",
        zones=_POPULATION_ZONES,
        fill_opacity=0.74,
    )
    civilizations_svg = _partition_overlay_svg_document(
        grid,
        civilization_zone_paths,
        title="世界文明区图",
        partition_id="civilization-regions",
        data_attribute="civilization",
        zones=civilization_zones,
        zone_outline_width=0.44,
        fill_opacity=0.68,
    )
    languages_svg = _partition_overlay_svg_document(
        grid,
        language_zone_paths,
        title="世界语言分布图",
        partition_id="language-regions",
        data_attribute="language",
        zones=language_zones,
        zone_outline_width=0.40,
        fill_opacity=0.64,
    )
    religions_svg = _partition_overlay_svg_document(
        grid,
        religion_zone_paths,
        title="世界宗教分布图",
        partition_id="religion-regions",
        data_attribute="religion",
        zones=religion_zones,
        zone_outline_width=0.44,
        fill_opacity=0.66,
    )
    political_svg = _partition_overlay_svg_document(
        grid,
        political_zone_paths,
        title="世界国家政区图层",
        partition_id="political-regions",
        data_attribute="state",
        zones=political_zones,
        extra_overlay=_frontier_overlay(frontier_fill_paths),
        zone_outline_width=0.58,
        fill_opacity=0.74,
    )
    provinces_svg = _partition_overlay_svg_document(
        grid,
        province_zone_paths,
        title="世界省份政区图层",
        partition_id="province-regions",
        data_attribute="province",
        zones=province_zones,
        extra_overlay=_frontier_overlay(frontier_fill_paths),
        zone_outline_width=0.52,
        fill_opacity=0.72,
    )
    contours_svg = _line_overlay_svg_document(
        grid,
        _svg_group(
            "elevation-contour-lines",
            contour_paths,
            _CONTOUR_COLOR,
            stroke_width=0.38,
            path_levels=contour_path_levels,
        ),
        title="世界等高线",
    )
    thematic_documents = {
        contours_path.name: contours_svg,
        climate_path.name: climate_svg,
        biome_path.name: biome_svg,
        watersheds_path.name: watersheds_svg,
        land_potential_path.name: land_potential_svg,
        population_path.name: population_svg,
        civilizations_path.name: civilizations_svg,
        languages_path.name: languages_svg,
        religions_path.name: religions_svg,
        political_path.name: political_svg,
        provinces_path.name: provinces_svg,
        tectonic_path.name: tectonic_svg,
    }
    climate_svg_bytes = len(climate_svg.encode("utf-8"))
    if climate_svg_bytes > _MAX_CLIMATE_SVG_BYTES:
        raise WorldGridRenderError(
            "climate SVG byte budget exceeded: "
            f"{climate_svg_bytes} bytes > {_MAX_CLIMATE_SVG_BYTES}"
        )
    additional_thematic_bytes = {
        name: len(document.encode("utf-8"))
        for name, document in thematic_documents.items()
        if name != climate_path.name
    }
    oversized = {
        name: size
        for name, size in additional_thematic_bytes.items()
        if size > _MAX_THEMATIC_SVG_BYTES
    }
    if oversized:
        raise WorldGridRenderError(f"thematic SVG byte budget exceeded: {oversized}")
    if sum(additional_thematic_bytes.values()) > _MAX_THEMATIC_SVG_TOTAL_BYTES:
        raise WorldGridRenderError(
            "combined thematic SVG byte budget exceeded: "
            f"{sum(additional_thematic_bytes.values())} bytes > "
            f"{_MAX_THEMATIC_SVG_TOTAL_BYTES}"
        )
    land_silhouette_clip = (
        '<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
        + "".join(
            f'<path d="{_filled_path_data(points, codes)}" fill-rule="evenodd" />'
            for points, codes in land_surface_paths
        )
        + '</clipPath></defs>'
    )
    surface_backfill = (
        land_silhouette_clip
        + '<g id="surface-backfill" data-layer="surface-backfill" '
        'aria-label="海陆不透底色">'
        f'<rect x="0" y="0" width="{grid.shape[1]}" height="{grid.shape[0]}" '
        f'fill="{deep_water_color}" />'
        + "".join(
            f'<path d="{_filled_path_data(points, codes)}" fill="{lowland_color}" '
            'fill-rule="evenodd" stroke="none" />'
            for points, codes in land_surface_paths
        )
        + '<g id="lake-land-underlay" data-underlay-for="lakes">'
        + "".join(
            f'<path d="{_filled_path_data(points, codes)}" fill="{lowland_color}" '
            'fill-rule="evenodd" stroke="none" />'
            for points, codes in lake_fill_paths
        )
        + '</g>'
        + '</g>'
    )
    precipitation_overlay = (
        '<g id="monsoon-precipitation" data-layer="monsoon-precipitation" '
        'aria-label="四季相对降水" clip-path="url(#land-silhouette-clip)">'
        + "".join(
            f'<image id="monsoon-precipitation-{season_id}" '
            f'class="monsoon-precipitation" data-season="{season_id}" '
            f'href="monsoon-precipitation-{season_id}.png" x="0" y="0" '
            f'width="{grid.shape[1]}" height="{grid.shape[0]}" '
            f'preserveAspectRatio="none" aria-label="{html.escape(label)}相对降水" hidden />'
            for season_id, label in zip(_SEASON_IDS, _SEASON_LABELS, strict=True)
        )
        + '</g>'
    )
    thematic_overlay_images = (
        '<g id="physical-theme-overlays" aria-label="物理图专题着色" '
        'clip-path="url(#land-silhouette-clip)">'
        + "".join(
            f'<image id="{theme_id}-map" class="physical-theme-map" '
            f'data-theme-map="{theme_id}" data-source="{filename}" x="0" y="0" '
            f'width="{grid.shape[1]}" height="{grid.shape[0]}" '
            f'preserveAspectRatio="none" aria-label="{label}" hidden />'
            for theme_id, filename, label in (
                ("climate", climate_path.name, "气候分区叠加"),
                ("biome", biome_path.name, "生物群系叠加"),
                ("watershed", watersheds_path.name, "水文流域叠加"),
                ("potential", land_potential_path.name, "农业宜居叠加"),
                ("population", population_path.name, "人口分布叠加"),
                ("civilizations", civilizations_path.name, "文明区叠加"),
                ("languages", languages_path.name, "语言区叠加"),
                ("religions", religions_path.name, "宗教分布叠加"),
                ("political", political_path.name, "国家政区叠加"),
                ("provinces", provinces_path.name, "省份政区叠加"),
            )
        )
        + '</g>'
    )
    overlay = "".join(
        (
            surface_backfill,
            _svg_filled_band_group(
                "bathymetry-bands",
                "海深色带",
                bathymetry_band_paths,
                water_palette,
                reverse_bands=True,
            ),
            '<g clip-path="url(#land-silhouette-clip)">'
            + _svg_filled_band_group(
                "elevation-bands",
                "高程色带",
                elevation_band_paths,
                elevation_palette,
            )
            + '</g>',
            _svg_filled_surface_group(
                "polar-land-surface",
                "极地灰白陆面",
                polar_land_fill_paths,
                _POLAR_LAND_COLOR,
                polar_land_paths,
                _POLAR_LAND_COLOR,
                0.0,
                1.0,
            ),
            _svg_group(
                "rivers",
                river_paths,
                _RIVER_COLOR,
                curve_paths=True,
                stroke_widths=river_stroke_widths,
                stroke_space="map",
                path_attributes=river_path_attributes,
            ),
            thematic_overlay_images,
            _partition_boundary_overlay(
                "civilization-boundaries",
                civilization_boundary_paths,
                color="#4c5360",
                width=0.82,
                opacity=0.92,
                hidden=True,
                pre_smoothed=True,
            ),
            _partition_boundary_overlay(
                "language-boundaries",
                language_boundary_paths,
                color="#625f59",
                width=0.42,
                opacity=0.82,
                hidden=True,
                pre_smoothed=True,
            ),
            _partition_boundary_overlay(
                "language-civilization-boundaries",
                civilization_boundary_paths,
                color="#f8f3e8",
                width=0.78,
                opacity=0.92,
                hidden=True,
                pre_smoothed=True,
            ),
            _partition_boundary_overlay(
                "religion-boundaries",
                religion_boundary_paths,
                color="#6f5550",
                width=0.62,
                opacity=0.78,
                dash="3.2 2.3",
                hidden=True,
                pre_smoothed=True,
            ),
            _state_boundary_overlay(
                state_boundary_paths,
                river_paths=state_river_dividers,
                pre_smoothed=True,
            ),
            _partition_boundary_overlay(
                "province-boundaries",
                province_boundary_paths,
                color="#685f54",
                width=0.52,
                opacity=0.72,
                dash="2.4 1.9",
                hidden=True,
                pre_smoothed=True,
                river_paths=province_river_dividers,
            ),
            _svg_filled_surface_group(
                "lakes",
                "湖泊",
                lake_fill_paths,
                shallow_water_color,
                lake_paths,
                _COAST_COLOR,
                0.45,
            ),
            _svg_filled_surface_group(
                "inland-seas",
                "内海",
                (),
                shallow_water_color,
                inland_sea_paths,
                _COAST_COLOR,
                0.70,
            ),
            _svg_filled_surface_group(
                "snow",
                "积雪",
                snow_fill_paths,
                _SNOW_COLOR,
                snow_paths,
                _SNOW_COLOR,
                0.50,
                0.80,
            ),
            _svg_filled_surface_group(
                "sea-ice",
                "多年海冰",
                sea_ice_fill_paths,
                _SEA_ICE_COLOR,
                sea_ice_paths,
                _SEA_ICE_EDGE_COLOR,
                0.44,
                0.88,
            ),
            precipitation_overlay,
            _graticule_svg(
                grid,
                graticule_paths,
                graticule_major_flags,
                graticule_labels,
            ),
            _polar_reference_svg(grid),
            _svg_group("coast", coast_paths, _COAST_COLOR, stroke_width=0.70),
            '<image id="elevation-contours" href="elevation-contours.svg" '
            f'x="0" y="0" width="{grid.shape[1]}" height="{grid.shape[0]}" '
            'preserveAspectRatio="none" aria-label="等高线" />',
            '<g id="society-thematic-dimmer" aria-hidden="true" hidden>'
            f'<rect x="0" y="0" width="{grid.shape[1]}" height="{grid.shape[0]}" '
            'fill="#f5efdf" fill-opacity="0.22" /></g>',
            _transport_overlay(society, width=grid.shape[1]),
            _bridge_overlay(society, width=grid.shape[1]),
            _toponymy_overlay(grid, society),
            _polar_label_overlay(grid),
            _culture_label_overlay(society, kind="civilization"),
            _culture_label_overlay(society, kind="language"),
            _religion_label_overlay(society),
            _political_label_overlay(society),
            _province_label_overlay(society),
            _frontier_group_label_overlay(society),
            _city_overlay(society),
            '<g id="seasonal-rivers" data-layer="seasonal-rivers" aria-label="季节河流动态样式"></g>',
            wind_overlay,
            '<image id="tectonic-map" class="tectonic-foundation-map" '
            f'data-source="{tectonic_path.name}" x="0" y="0" '
            f'width="{grid.shape[1]}" height="{grid.shape[0]}" '
            'preserveAspectRatio="none" aria-label="板块构造基础图" hidden />',
        )
    )
    html_kwargs = {
        "society": society,
        "tectonics": tectonics,
        "territorial_qa": territorial_qa,
        "width": display_width,
        "height": display_height,
        "viewbox_width": grid.shape[1],
        "viewbox_height": grid.shape[0],
    }
    empty_document = _html_document(grid, digest, "", **html_kwargs)
    estimated_html_bytes = len(empty_document.encode("utf-8")) + len(
        overlay.encode("utf-8")
    )
    if estimated_html_bytes > _MAX_HTML_BYTES:
        raise WorldGridRenderError(
            "HTML byte budget exceeded: "
            f"{estimated_html_bytes} bytes > {_MAX_HTML_BYTES}"
        )
    index_html = _html_document(grid, digest, overlay, **html_kwargs)
    html_bytes = len(index_html.encode("utf-8"))
    if html_bytes > _MAX_HTML_BYTES:
        raise WorldGridRenderError(
            f"HTML byte budget exceeded: {html_bytes} bytes > {_MAX_HTML_BYTES}"
        )

    _reject_reparse_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _reject_reparse_paths(output_dir)

    temporary_files: list[tuple[Path, tuple[int, int] | None]] = []
    try:
        temporary_terrain = _temporary_file(output_dir, terrain_path.name, ".png")
        temporary_files.append((temporary_terrain, _file_identity(temporary_terrain)))
        temporary_index = _temporary_file(output_dir, index_path.name, ".html")
        temporary_files.append((temporary_index, _file_identity(temporary_index)))
        temporary_qa = _temporary_file(output_dir, qa_path.name, ".json")
        temporary_files.append((temporary_qa, _file_identity(temporary_qa)))
        temporary_climate = _temporary_file(output_dir, climate_path.name, ".svg")
        temporary_files.append((temporary_climate, _file_identity(temporary_climate)))
        temporary_thematic: dict[Path, Path] = {climate_path: temporary_climate}
        for final_path in (
            contours_path,
            biome_path,
            watersheds_path,
            land_potential_path,
            population_path,
            civilizations_path,
            languages_path,
            religions_path,
            political_path,
            provinces_path,
            tectonic_path,
        ):
            temporary_path = _temporary_file(output_dir, final_path.name, ".svg")
            temporary_files.append((temporary_path, _file_identity(temporary_path)))
            temporary_thematic[final_path] = temporary_path
        temporary_society_npz = _temporary_file(output_dir, society_npz_path.name, ".npz")
        temporary_files.append((temporary_society_npz, _file_identity(temporary_society_npz)))
        temporary_society_json = _temporary_file(output_dir, society_json_path.name, ".json")
        temporary_files.append((temporary_society_json, _file_identity(temporary_society_json)))
        temporary_precipitation: list[Path] = []
        for precipitation_path in precipitation_paths:
            temporary_path = _temporary_file(
                output_dir, precipitation_path.name, ".png"
            )
            temporary_files.append((temporary_path, _file_identity(temporary_path)))
            temporary_precipitation.append(temporary_path)

        Image.fromarray(_terrain_pixels(grid)).save(
            temporary_terrain, format="PNG", optimize=True
        )
        for season_index, temporary_path in enumerate(temporary_precipitation):
            Image.fromarray(
                _seasonal_precipitation_pixels(grid, season_index),
                mode="RGBA",
            ).save(temporary_path, format="PNG", optimize=True)
        temporary_index.write_text(index_html, encoding="utf-8", newline="\n")
        write_society_files(
            society,
            temporary_society_npz,
            temporary_society_json,
            grid_digest=digest,
        )
        for final_path, temporary_path in temporary_thematic.items():
            temporary_path.write_text(
                thematic_documents[final_path.name],
                encoding="utf-8",
                newline="\n",
            )

        elevation_band_count = _elevation_band_count(grid)
        bathymetry_band_count = _bathymetry_band_count(grid)
        observed_bathymetry_bands = sorted(
            int(value)
            for value in np.unique(
                grid.bathymetry_band[
                    np.isin(grid.water, (1, 3)) & (grid.bathymetry_band >= 0)
                ]
            )
        )
        settlement_by_identifier = {
            item.identifier: item for item in society.settlements
        }
        cross_civilization_routes = 0
        for route in society.transport.routes:
            if route.target_settlement_id is None:
                continue
            source = settlement_by_identifier[route.source_settlement_id]
            target = settlement_by_identifier[route.target_settlement_id]
            if (
                society.cultures.civilization_id[source.row, source.column]
                != society.cultures.civilization_id[target.row, target.column]
            ):
                cross_civilization_routes += 1
        _west, south, _east, north = _review_extents(grid)
        latitude_rows = north - (
            np.arange(grid.shape[0], dtype=np.float64) + 0.5
        ) * ((north - south) / grid.shape[0])
        planet_metadata = grid.metadata.get("planet", {})
        axial_tilt = float(planet_metadata.get("axialTiltDegrees", 23.44))
        polar_circle_latitude = 90.0 - float(np.clip(axial_tilt, 0.0, 45.0))
        polar_latitude_domain = (
            np.abs(latitude_rows)[:, None] >= polar_circle_latitude
        )
        human_domain = society_domain_mask(grid)
        polar_land_domain = (grid.water == 0) & ~human_domain
        polar_settlements = [
            settlement
            for settlement in society.settlements
            if polar_land_domain[settlement.row, settlement.column]
        ]
        polar_transport_routes = []
        for route in society.transport.routes:
            for column, row in route.path:
                sample_row = int(np.clip(round(row), 0, grid.shape[0] - 1))
                sample_column = int(round(column)) % grid.shape[1]
                if polar_land_domain[sample_row, sample_column]:
                    polar_transport_routes.append(route)
                    break
        polar_activity = {
            "lakeCells": int(
                np.count_nonzero((grid.water >= 2) & polar_latitude_domain)
            ),
            "riverCells": int(
                np.count_nonzero((grid.river_order > 0) & polar_land_domain)
            ),
            "populationCells": int(
                np.count_nonzero(
                    (society.population.population_weight > 0.0)
                    & polar_land_domain
                )
            ),
            "civilizationCells": int(
                np.count_nonzero(
                    (society.cultures.civilization_id > 0) & polar_land_domain
                )
            ),
            "stateCells": int(
                np.count_nonzero(
                    (society.politics.state_id > 0) & polar_land_domain
                )
            ),
            "provinceCells": int(
                np.count_nonzero(
                    (society.provinces.province_id > 0) & polar_land_domain
                )
            ),
            "settlementCount": len(polar_settlements),
            "transportRouteCount": len(polar_transport_routes),
        }
        if any(polar_activity.values()):
            raise WorldGridRenderError(
                "polar continents must stop after climate and biome: "
                + ", ".join(
                    f"{name}={count}"
                    for name, count in polar_activity.items()
                    if count
                )
            )
        north_polar_ice_domain = (
            polar_land_domain
            & grid.snow
            & (latitude_rows[:, None] >= 0.0)
        )
        south_polar_ice_domain = (
            polar_land_domain
            & grid.snow
            & (latitude_rows[:, None] < 0.0)
        )
        qa = {
            "format": "eirenor-world-grid-review",
            "schema": "world-grid-review-v3",
            "gridDigest": digest,
            "shape": {"width": grid.shape[1], "height": grid.shape[0]},
            "reviewDimensions": {"width": display_width, "height": display_height},
            "elevationBandCount": elevation_band_count,
            "elevationContourLevelCount": max(0, elevation_band_count - 1),
            "bathymetryBandCount": bathymetry_band_count,
            "observedBathymetryBands": observed_bathymetry_bands,
            "layers": {
                "graticule": len(graticule_paths),
                "coast": len(coast_paths),
                "lakes": len(lake_paths),
                "inland-seas": len(inland_sea_paths),
                "elevation-bands": len(_flatten_filled_paths(elevation_band_paths)),
                "bathymetry-bands": len(_flatten_filled_paths(bathymetry_band_paths)),
                "elevation-contours": len(contour_paths),
                "snow": len(snow_paths),
                "sea-ice": len(sea_ice_paths),
                "polar-references": 4,
                "polar-labels": 2,
                "rivers": len(river_paths),
                "monsoon-precipitation": 4,
                "monsoon-wind": int(sum(wind_arrow_counts.values())),
                "seasonal-rivers": len(river_paths),
                "climate-zones": int(sum(bool(paths) for paths in climate_zone_paths)),
                "biome-zones": int(sum(bool(paths) for paths in biome_zone_paths)),
                "major-watersheds": int(
                    sum(bool(paths) for paths in watershed_zone_paths)
                ),
                "land-potential-bands": int(
                    sum(bool(paths) for paths in potential_zone_paths)
                ),
                "population-bands": int(
                    sum(bool(paths) for paths in population_zone_paths)
                ),
                "civilization-regions": len(society.cultures.civilizations),
                "language-regions": len(society.cultures.languages),
                "religion-regions": len(society.religions.religions),
                "political-regions": len(society.politics.states),
                "province-regions": len(society.provinces.provinces),
                "frontier-group-labels": len(society.politics.frontier_groups),
                "geographic-labels": len(society.geographic_features),
                "city-layer": len(society.settlements),
                "holy-cities": sum(
                    item.holy_religion_identifier is not None
                    for item in society.settlements
                ),
                "transport-network": len(society.transport.routes),
                "bridges": len(society.transport.bridges),
                "tectonic-plates": int(tectonics.diagnostics["plateCount"]),
                "tectonic-boundaries": int(tectonics.diagnostics["boundaryPairCount"]),
                "tectonic-relative-motion": int(
                    sum(2 * len(boundary.paths) for boundary in tectonics.boundaries)
                ),
                "tectonic-plate-motion": len(tectonics.plates),
            },
            "tectonics": {
                **dict(tectonics.diagnostics),
                "classificationPairCounts": {
                    classification: sum(
                        boundary.classification == classification
                        for boundary in tectonics.boundaries
                    )
                    for classification in ("convergent", "divergent", "transform")
                },
            },
            "territorialQA": territorial_qa,
            "society": {
                "populationRange": [
                    society.population.population_min,
                    society.population.population_max,
                ],
                "settlementCount": len(society.settlements),
                "settlementTierCounts": {
                    tier: sum(item.tier == tier for item in society.settlements)
                    for tier in ("metropolis", "city", "town", "site")
                },
                "settlementSiteTypeCounts": {
                    site_type: sum(
                        item.site_type == site_type for item in society.settlements
                    )
                    for site_type in (
                        "port",
                        "lake-port",
                        "river-city",
                        "market",
                        "oasis",
                        "pass",
                        "fortress",
                        "island-port",
                    )
                },
                "transportRouteCount": len(society.transport.routes),
                "bridgeCount": len(society.transport.bridges),
                "crossCivilizationRouteCount": cross_civilization_routes,
                "transportModeCounts": {
                    mode: sum(route.mode == mode for route in society.transport.routes)
                    for mode in ("road", "river", "sea")
                },
                "civilizationCount": len(society.cultures.civilizations),
                "languageCount": len(society.cultures.languages),
                "religionCount": len(society.religions.religions),
                "religionTraditionCounts": {
                    tradition: sum(
                        item.tradition == tradition
                        for item in society.religions.religions
                    )
                    for tradition in sorted(
                        {item.tradition for item in society.religions.religions}
                    )
                },
                "geographicFeatureCount": len(society.geographic_features),
                "stateCount": len(society.politics.states),
                "provinceCount": len(society.provinces.provinces),
                "statesWithoutProvince": sum(
                    not any(
                        province.state_identifier == state.identifier
                        for province in society.provinces.provinces
                    )
                    for state in society.politics.states
                ),
                "smallStateCount": sum(
                    item.size_class == "small" for item in society.politics.states
                ),
                "frontierLandShare": float(
                    np.mean(society.politics.frontier[society_domain_mask(grid)])
                ),
                "frontierGroupCount": len(society.politics.frontier_groups),
                "frontierLivelihoodCounts": {
                    livelihood: sum(
                        item.livelihood == livelihood
                        for item in society.politics.frontier_groups
                    )
                    for livelihood in (
                        "pastoral",
                        "foraging",
                        "maritime",
                        "agropastoral",
                        "highland",
                        "riverine",
                    )
                },
                "legacyGeographyInherited": False,
                "namingProfile": (
                    "legacy" if _uses_legacy_society_names(grid) else "procedural"
                ),
                "legacyNamesOnly": _uses_legacy_society_names(grid),
            },
            "seasonalClimate": {
                "seasons": list(_SEASON_IDS),
                "labels": list(_SEASON_LABELS),
                "fixedPrecipitationScale": [0, 255],
                "windArrowCounts": wind_arrow_counts,
                "relativeUnitsOnly": True,
            },
            "projection": {
                "coordinateContract": "EIR-GEOG-1",
                "review": "equirectangular",
                "storageGridMapping": "plate-carree",
                "reader": "equal-earth",
                "polarView": "azimuthal-equal-area",
            },
            "polarRegions": {
                "northPolarIceLandCells": int(
                    np.count_nonzero(north_polar_ice_domain)
                ),
                "southPolarIceLandCells": int(
                    np.count_nonzero(south_polar_ice_domain)
                ),
                "seaIceCells": int(np.count_nonzero(grid.sea_ice)),
                "northPoleSurface": "continental-ice-sheet",
                "southPoleSurface": "continental-ice-sheet",
                "generationStage": "pre-hydrology",
                "forbiddenLayerActivity": polar_activity,
            },
            "artifacts": {
                "terrain.png": temporary_terrain.stat().st_size,
                "index.html": temporary_index.stat().st_size,
                **{
                    final_path.name: temporary_path.stat().st_size
                    for final_path, temporary_path in temporary_thematic.items()
                },
                **{
                    precipitation_paths[index].name: temporary_path.stat().st_size
                    for index, temporary_path in enumerate(temporary_precipitation)
                },
                society_npz_path.name: temporary_society_npz.stat().st_size,
                society_json_path.name: temporary_society_json.stat().st_size,
            },
        }
        temporary_qa.write_text(
            json.dumps(qa, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _atomic_replace_files(
            (
                (temporary_terrain, terrain_path),
                (temporary_index, index_path),
                (temporary_qa, qa_path),
                (temporary_society_npz, society_npz_path),
                (temporary_society_json, society_json_path),
                *((temporary_path, final_path) for final_path, temporary_path in temporary_thematic.items()),
            )
            + tuple(zip(temporary_precipitation, precipitation_paths, strict=True))
        )
    finally:
        for temporary, identity in temporary_files:
            _unlink_owned_file(temporary, identity)

    return {
        "terrain": terrain_path,
        "elevationContours": contours_path,
        "index": index_path,
        "qa": qa_path,
        "climate": climate_path,
        "biome": biome_path,
        "watersheds": watersheds_path,
        "landPotential": land_potential_path,
        "population": population_path,
        "civilizations": civilizations_path,
        "languages": languages_path,
        "religions": religions_path,
        "political": political_path,
        "provinces": provinces_path,
        "tectonic": tectonic_path,
        "society": (society_npz_path, society_json_path),
        "precipitation": precipitation_paths,
        "gridDigest": digest,
    }
