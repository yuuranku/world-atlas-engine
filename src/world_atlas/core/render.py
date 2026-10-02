"""Render a compact, review-oriented view directly from a WorldGrid."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
import colorsys
import html
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
from dataclasses import replace
from typing import Any

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
from .hypsometry import (
    ELEVATION_DISPLAY_LEVELS,
    elevation_display_indices,
    elevation_display_thresholds,
)
from .polar import polar_continent_mask
from .globe import write_globe
from .globe_assets import globe_theme_documents
from .review_interface import write_interface_snapshot
from .governance_render import write_governance_overlay
from .presentation import society_content_digest
from .cartographic_symbols import LANDFORM_STYLES, SITE_MARKS, symbol_definitions
from .cartographic_curves import terrain_channel_paths
from .cartographic_surface import continuous_land_surface
from .cartographic_tiles import (
    TileFeature, TileLevel, geometry_path_data, write_atlas_tiles, write_city_relief_tiles,
)
from .city_relief import derive_city_relief
from .landforms import derive_landform_inventory
from .landform_render import landform_features, landform_svg_body
from .svg_paths import COORDINATE_SCALE, integer_subpath_data
from .svg_artifacts import encode_svgz
from .raster_topology import categorical_coverage
from .cartographic_features import (
    filled_geometries, filled_features, line_features, overview_markup,
    overview_coverage, shared_display_coverage,
)
from .continuous_terrain import PhysicalTerrainField
from .terrain_refinement import terrain_from_source
from .continuous_scalar import scalar_band_paths
from .continuous_ecology import FreshwaterCorridors
from .vegetation import derive_vegetation_cover, derive_vegetation_field, VEGETATION_THRESHOLDS
from .cartographic_relief import physical_relief_paths
from .cartographic_generalization import generalize_display_surface
from .procedural_planet import ProceduralSurface
from .cartographic_rivers import RIVER_WIDTH_MODEL, river_width_field, river_width_profile, river_channel_surface
from .coastal_partition import CoastalPartition, extend_coastal_partition, clip_partition_to_surface, enforce_homogeneous_components
from .transport_geometry import prepare_transport_geometry, river_navigation_attributes, river_navigation_segments
from .river_network import hydrologic_outlet_targets
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
    derive_land_potential_field,
    derive_habitability_field,
    KOPPEN_LEGEND,
    derive_thematic_layers,
    LAND_POTENTIAL_THRESHOLDS,
    HABITABILITY_THRESHOLDS,
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
from .society.population import population_density, POPULATION_DENSITY_THRESHOLDS
from .society.geographic_features import mountain_components
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
_CLIMATE_ZONES = tuple((f"{item.symbol} · {item.label}", item.color) for item in KOPPEN_LEGEND)
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
    ("高农业潜力", "#3f7463"),
)
_HABITABILITY_ZONES = (
    ("极低适宜度", "#e1dccb"),
    ("低适宜度", "#d0cda1"),
    ("一般适宜度", "#b3c6a0"),
    ("较宜居", "#86b6a4"),
    ("宜居", "#579c92"),
    ("高适宜度", "#357975"),
)
_VEGETATION_ZONES = tuple(zip(
    (f"覆盖率 {index * 10}–{(index + 1) * 10}%" for index in range(10)),
    ("#e4bc76", "#eed092", "#e7dfa6", "#cfda9d", "#adca86",
     "#87b56f", "#63a15d", "#438b4e", "#2c7143", "#18583b"),
    strict=True,
))
_POPULATION_ZONES = (
    ("无定居人口", "#ded9c7"),
    ("低于 0.1 人/km²", "#d7d0a4"),
    ("0.1–0.5 人/km²", "#c9c68c"),
    ("0.5–1 人/km²", "#aebb7e"),
    ("1–2 人/km²", "#83a978"),
    ("2–5 人/km²", "#548f71"),
    ("至少 5 人/km²", "#2f655b"),
)
# The current 1,812x906 review is a few thousand paths and a compact sub-5 MiB
# HTML bundle. Keep generous fixed headroom while bounding pathological
# fragmentation from an unexpected source.
# A fragmented multi-continent world can legitimately exceed the original
# 10k path ceiling.  Keep a bounded budget while allowing a complete physical
# overlay to publish without a runtime-only override.
# High-detail coasts and their derived thematic layers legitimately exceed
# the original 12k guard on a full-resolution world.  Keep a finite ceiling
# for accidental path explosions while allowing the accepted 3-continent
# terrain to render every real vector feature.
_MAX_SVG_PATHS = 24_000
# Hierarchical ocean islands and closed-sea bathymetry add legitimate physical
# contours while path and byte budgets still cap pathological fragmentation.
_MAX_SVG_POINTS = 1_200_000
# Native-resolution administrative and cultural geometry is deliberately kept
# in the review document.  Sixteen MiB still catches accidental path explosions
# without forcing the atlas back through a block-reduction stage.
_MAX_HTML_BYTES = 16 * 1024 * 1024
# Full-resolution coast and contour geometry can legitimately produce a
# single thematic overlay above 4 MiB. These remain bounded safeguards, but
# fit the accepted high-detail 3-continent world without rasterizing it.
_MAX_THEMATIC_SVG_BYTES = 10 * 1024 * 1024
_MAX_THEMATIC_SVG_TOTAL_BYTES = 48 * 1024 * 1024
_MAX_EXPORT_SVG_BYTES = 64 * 1024 * 1024


class WorldGridRenderError(ValueError):
    """Raised when review rendering would exceed a fixed safety boundary."""


def _society_generation_request(
    grid: WorldGrid,
) -> tuple[str | Path | NameLexicon, dict[str, int | float]]:
    """Resolve the society profile carried by this world's immutable metadata."""

    raw = grid.metadata.get("societyGeneration")
    if not isinstance(raw, Mapping):
        raise WorldGridRenderError("world_atlas requires an explicit procedural societyGeneration profile")
    profile = str(raw.get("namingProfile", ""))
    if profile == "procedural":
        seed = raw.get("namingSeed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise WorldGridRenderError(
                "procedural societyGeneration.namingSeed must be an integer"
            )
        lexicon = procedural_name_lexicon(seed)
    else:
        raise WorldGridRenderError(
            f"unknown society naming profile: {profile}"
        )

    human_seed = raw.get("humanSeed")
    if isinstance(human_seed, bool) or not isinstance(human_seed, int):
        raise WorldGridRenderError(
            "procedural societyGeneration.humanSeed must be an integer"
        )
    options: dict[str, int | float] = {}
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
    frontier_share = raw.get("frontierTargetShare")
    if (
        isinstance(frontier_share, bool)
        or not isinstance(frontier_share, (int, float))
        or not 0.0 <= float(frontier_share) < 1.0
    ):
        raise WorldGridRenderError(
            "societyGeneration.frontierTargetShare must lie in [0, 1)"
        )
    options["frontier_target_share"] = float(frontier_share)
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
    level_count = ELEVATION_DISPLAY_LEVELS
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


def _terrain_pixels(grid: WorldGrid, *, terrain_field, bathymetry) -> np.ndarray:
    """Use the vector map's continuous-height intervals for terrain RGB pixels."""

    elevation_palette, _ = _elevation_palette_for(grid)
    water_palette, _ = _bathymetry_palette_for(grid)
    shallow_water_color = water_palette[_shallow_ocean_ordinal(grid, len(water_palette))]
    pixels = np.empty((*grid.shape, 3), dtype=np.uint8)
    land = grid.water == 0
    maritime_water = np.isin(grid.water, (1, 3))
    lake = grid.water == 2
    physical_elevation = terrain_field.palette_elevation(terrain_field.native_m)
    pixels[land] = elevation_palette[elevation_display_indices(physical_elevation[land])]
    if np.any(maritime_water):
        source_depth = np.asarray(bathymetry, dtype=float)
        if source_depth.shape != grid.shape or not np.all(np.isfinite(source_depth)):
            raise ValueError("terrain pixels require the accepted continuous depth field")
        display_depth = np.clip(np.floor(source_depth * len(water_palette)), 0, len(water_palette)-1).astype(np.intp)
        pixels[maritime_water] = water_palette[display_depth[maritime_water]]
    pixels[lake] = shallow_water_color
    pixels[polar_continent_mask(grid)] = _POLAR_LAND_RGB
    # Snow is a surface cover, not a replacement elevation class.  Preserve
    # the land band beneath it so high-latitude terrain does not turn into a
    # featureless white slab away from the actual poles.
    snow = grid.snow
    pixels[snow] = np.rint(
        0.48 * pixels[snow].astype(np.float64)
        + 0.52 * np.asarray(_SNOW_TERRAIN_RGB, dtype=np.float64)
    ).astype(np.uint8)
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


def _categorical_overlay_pixels(
    values: np.ndarray,
    active_mask: np.ndarray,
    zones: Sequence[tuple[str, str]],
    *,
    opacity: float,
) -> np.ndarray:
    """Render an exact full-resolution categorical overlay without SVG path noise."""

    categories = np.asarray(values)
    active = np.asarray(active_mask, dtype=bool)
    if categories.shape != active.shape:
        raise WorldGridRenderError("categorical values and mask must share a shape")
    if not 0.0 <= float(opacity) <= 1.0:
        raise WorldGridRenderError("categorical overlay opacity must lie in [0, 1]")
    if np.any(categories[active] < 0) or np.any(categories[active] >= len(zones)):
        raise WorldGridRenderError("categorical overlay value exceeds its zone palette")
    palette = np.asarray(
        [
            (*_parse_hex_color(color), round(float(opacity) * 255.0))
            for _name, color in zones
        ],
        dtype=np.uint8,
    )
    image = np.zeros((*categories.shape, 4), dtype=np.uint8)
    image[active] = palette[categories[active].astype(np.int64)]
    return image


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


def _mask_paths(mask: np.ndarray) -> list[np.ndarray]:
    return _surface_outline_paths(_filled_mask_paths(mask), np.shape(mask))


def _categorical_partition_paths(
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    category_count: int,
    land_surface: Any,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Draw actual category labels, without interpolating their identifiers."""

    partition = _coastal_partition_topology(
        values, active_mask, category_count=category_count,
        land_surface=land_surface,
    )
    return _shared_topology_zone_paths(
        partition.faces, partition.labels, category_count=category_count, include_zero=True,
    )


def _coastal_partition_topology(values, active_mask, *, category_count,
                                land_surface):
    """Build a complete working coverage for the shared physical shore clip.

    Theme assets contain the coastal working band. Their visible boundary is
    the one mandatory physical clip in the atlas and globe, rather than ten
    independent copies of the high-detail shore.
    """

    if not np.any(active_mask):
        empty_labels = np.empty(0, dtype=np.int32)
        return CoastalPartition((), empty_labels, (), empty_labels)
    logging.getLogger(__name__).info("Building shared coverage for %s categories", category_count)
    extended, domain = extend_coastal_partition(
        values, active_mask, margin=4.0,
        wrap_longitude=True,
    )
    faces, labels = _shared_partition_topology(
        extended, domain, category_count=category_count,
    )
    logging.getLogger(__name__).info("Clipping %s faces to the physical shore", len(faces))
    faces, labels = enforce_homogeneous_components(faces, labels, values, active_mask, land_surface)
    visible_faces, visible_labels = clip_partition_to_surface(faces, labels, land_surface)
    logging.getLogger(__name__).info("Physical coverage verified")
    return CoastalPartition(faces, labels, visible_faces, visible_labels)


def _lake_surface(grid, land_surface):
    """Select lakes from the complementary water surface, sharing every shore."""

    height, width = grid.shape
    wet_parts = _geometry_polygons(shapely.difference(
        shapely.box(0, 0, width, height), land_surface,
    ))
    rows, columns = np.nonzero(grid.water == 2)
    if not wet_parts or not len(rows):
        return shapely.MultiPolygon()
    index = shapely.STRtree(wet_parts)
    matches = index.query(shapely.points(columns + .5, rows + .5), predicate="within")
    return shapely.union_all([wet_parts[int(i)] for i in np.unique(matches[1])])


def _geometry_polygons(geometry: Any) -> tuple[Any, ...]:
    """Return every polygonal part of one Shapely geometry."""

    if geometry is None or bool(shapely.is_empty(geometry)):
        return ()
    if geometry.geom_type == "Polygon":
        return (geometry,)
    if not hasattr(geometry, "geoms"):
        return ()
    return tuple(
        polygon
        for part in geometry.geoms
        for polygon in _geometry_polygons(part)
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


def _scalar_working_surface(working_surface):
    """Dissolve the computation domain before its single delivery-grid rounding.

    Internal category-overlay roundoff must not become a numeric-field cut.
    Clear GEOS precision metadata so the following source intersection keeps
    its floating curves until the common coverage is noded once.
    """
    outer = shapely.union_all(shapely.get_parts(working_surface))
    return shapely.set_precision(shapely.set_precision(outer, 1e-8), 0)


def _scalar_zone_paths(values, land_mask, thresholds, *, working_surface):
    """Extract shared numeric coverage within the coastal working surface."""
    working_surface = _scalar_working_surface(working_surface)
    regions = [shapely.union_all(list(filled_geometries(band)))
               for band in scalar_band_paths(values, land_mask, thresholds)]
    return [_geometry_filled_paths(region)
            for region in shared_display_coverage(
                shapely.intersection(regions, working_surface))]


def _population_zone_paths(density, land_mask, *, working_surface):
    """Keep uninhabited support distinct from continuous positive density.

    Zero weight denotes an unsupported physical cell, including permanent
    snow. A zero-valued isolated minimum of an interpolated density surface
    has no area; tracing only its zero isoline would paint it as inhabited.
    A shared continuous support boundary preserves those source cells, while
    positive density retains its numeric, area-normalized logarithmic scale.
    """
    working_surface = _scalar_working_surface(working_surface)
    density = np.asarray(density, dtype=np.float64)
    paths = scalar_band_paths(np.log1p(density), land_mask,
                              np.log1p(POPULATION_DENSITY_THRESHOLDS))
    support = scalar_band_paths((density > 0.0).astype(np.float64), land_mask, (0.5,))
    uninhabited = shapely.union_all(list(filled_geometries(support[0])))
    regions = [uninhabited] + [
        shapely.difference(
            shapely.union_all(list(filled_geometries(band))), uninhabited)
        for band in paths[1:]
    ]
    # Node support and density boundaries together before any SVG rounds their
    # coordinates. All exports and display levels consume this same coverage.
    return [_geometry_filled_paths(region)
            for region in shared_display_coverage(
                shapely.intersection(regions, working_surface))]


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


def _surface_outline_paths(
    paths: Sequence[tuple[np.ndarray, np.ndarray]],
    shape: tuple[int, int],
) -> list[np.ndarray]:
    """Trace the visible land silhouette exactly, without drawing map edges."""

    height, width = shape
    result: list[np.ndarray] = []
    for points, codes in paths:
        starts = np.flatnonzero(codes == 1)
        for start, stop in zip(starts, (*starts[1:], len(points)), strict=True):
            ring = points[start:stop]
            visible_edges = np.ones(len(ring) - 1, dtype=bool)
            for axis, edge in ((0, 0.0), (0, float(width)), (1, 0.0), (1, float(height))):
                visible_edges &= ~(
                    np.isclose(ring[:-1, axis], edge) & np.isclose(ring[1:, axis], edge)
                )
            runs = np.flatnonzero(np.diff(np.pad(visible_edges.astype(np.int8), (1, 1))))
            for first, last in zip(runs[::2], runs[1::2], strict=True):
                result.append(ring[first:last + 1])
    return result


def _shared_partition_topology(
    values: np.ndarray,
    active_mask: np.ndarray,
    *,
    category_count: int,
) -> tuple[tuple[Any, ...], np.ndarray]:
    """Extract one continuous ownership coverage, shared by fills and borders.

    Identifiers select indicator fields, never numeric interpolation values.
    The max-envelope boundary is extracted once before SVG serialization.
    No cell-box union or independent smoothing stage changes its geometry.
    """
    faces, labels = categorical_coverage(values, active_mask, category_count=category_count)
    return tuple(faces), np.asarray(labels, dtype=np.int32)


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


def _administrative_boundary_paths(
    faces: Sequence[Any],
    province_face_ids: Sequence[int] | np.ndarray,
    province_to_state: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Derive state and province ink from the same shared coverage arcs.

    A province line belongs to its parent state only.  An inter-state edge is
    emitted once as state ink, rather than once again as a dashed province
    line.  Rivers do not alter this topology: the ownership coverage is the
    sole source of every administrative path.
    """

    province_ids = np.asarray(province_face_ids, dtype=np.int32)
    state_by_province = np.asarray(province_to_state, dtype=np.int32)
    if province_ids.ndim != 1 or len(province_ids) != len(faces):
        raise WorldGridRenderError("province labels must correspond to coverage faces")
    if state_by_province.ndim != 1 or not len(state_by_province):
        raise WorldGridRenderError("province-to-state mapping must be a non-empty vector")
    if np.any(province_ids < 0) or np.any(province_ids >= len(state_by_province)):
        raise WorldGridRenderError("province identifier is outside the state mapping")

    state_face_ids = state_by_province[province_ids]
    state_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    province_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for edge, owners in _coverage_edge_inventory(faces, province_ids).items():
        provinces = set(owners)
        if len(provinces) < 2:
            continue
        states = {int(state_by_province[identifier]) for identifier in provinces}
        positive_states = {identifier for identifier in states if identifier > 0}
        if len(states) > 1 and positive_states:
            state_edges.add(edge)
        elif len(states) == 1 and positive_states:
            province_edges.add(edge)

    return (
        state_face_ids,
        _coverage_edge_chains(state_edges),
        _coverage_edge_chains(province_edges),
    )


def _administrative_overview_features(features, faces, province_ids, province_to_state,
                                      *, frame_shape):
    """Derive both administrative maps and all their ink from one summary."""
    summary = overview_coverage(faces, frame_shape=frame_shape)
    province_ids = np.asarray(province_ids, dtype=np.int32)
    state_ids = np.asarray(province_to_state, dtype=np.int32)[province_ids]
    regions = {
        theme: {int(identifier): shapely.union_all(summary[labels == identifier])
                for identifier in np.unique(labels)}
        for theme, labels in (("political", state_ids), ("provinces", province_ids))
    }
    # Keep complete shared arcs. Fills and ink use the same SVG land clip;
    # separately intersecting ink would round new coast endpoints before
    # delivery and move them away from the clipped fill boundary.
    _, state_paths, province_paths = _administrative_boundary_paths(
        summary, province_ids, province_to_state)
    boundaries = {"state-boundaries": shapely.MultiLineString(state_paths),
                  "province-boundaries": shapely.MultiLineString(province_paths)}
    result, seen = [], set()
    for feature in features:
        if feature.theme in regions:
            attribute = "data-state" if feature.theme == "political" else "data-province"
            identifier = int(feature.attributes[attribute])
            key = (feature.theme, identifier)
            geometry = regions[feature.theme][identifier]
        elif feature.layer in boundaries:
            key = (feature.layer, tuple(sorted(feature.attributes.items())))
            geometry = boundaries[feature.layer]
        else:
            result.append(feature)
            continue
        if key not in seen:
            seen.add(key)
            result.append(replace(feature, geometry=geometry))
    return result


def _elevation_thresholds(grid: WorldGrid) -> list[float]:
    """Return display boundaries shared by land fills, contours and textures."""

    return elevation_display_thresholds()


def _elevation_band_paths(
    grid: WorldGrid,
    *, terrain_field,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Extract nested land colours from the same physical ground as the coast."""
    bands, _, _ = physical_relief_paths(terrain_field, _elevation_thresholds(grid))
    land = _geometry_filled_paths(continuous_land_surface(grid, terrain_field=terrain_field))
    return [land] + [land if band is None else band for band in bands]


def _filled_mask_paths(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract a class from the common continuous ownership coverage."""
    active = np.asarray(mask, dtype=bool)
    if active.ndim != 2 or min(active.shape) < 1 or not active.any():
        return []
    faces, labels = categorical_coverage(
        active.astype(np.int32), np.ones(active.shape, dtype=bool), category_count=2)
    return _geometry_filled_paths(shapely.union_all(
        [face for face, label in zip(faces, labels, strict=True) if label == 1]))


def _compact_coordinate(value: float) -> str:
    """Format SVG coordinates without serialising predictable zero padding."""

    numeric = float(value)
    if abs(numeric - round(numeric)) <= 1.0e-9:
        return str(int(round(numeric)))
    if abs(numeric * 2.0 - round(numeric * 2.0)) <= 1.0e-9:
        return f"{numeric:.1f}".rstrip("0").rstrip(".")
    # A native terrain height can be less than a millimetre below its colour
    # threshold. Eight decimals keep that centre on its correct contour side
    # and retain narrow river banks at maximum zoom.
    return f"{numeric:.8f}".rstrip("0").rstrip(".")


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


def _elevation_paths(grid: WorldGrid, *, terrain_field) -> tuple[list[np.ndarray], list[float]]:
    _, lines, levels = physical_relief_paths(terrain_field, _elevation_thresholds(grid))
    return lines, levels


def _path_data(points: np.ndarray) -> str:
    if len(points) < 2:
        return ""
    return geometry_path_data(shapely.LineString(points))


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


def _river_channel_markup(grid, source_paths, curve_paths, attributes, *, tile_features=None,
                          navigation_segments=None):
    """Ground-scaled channel faces, with minimum overview ink kept separate."""

    widths_m = river_width_field(grid)
    channels, lines = [], []
    for index, (source, curve, attrs) in enumerate(zip(source_paths, curve_paths, attributes, strict=True)):
        attrs = {**attrs, "river-color": _RIVER_COLOR, "navigation-color": "#195d84"}
        profile = river_width_profile(grid, source, curve, widths_m)
        channel = river_channel_surface(grid, curve, profile)
        shape_data = " ".join(_filled_path_data(points, codes) for points, codes in _geometry_filled_paths(channel))
        extra = "".join(f' data-{key}="{value}"' for key, value in attrs.items())
        reach_id = f"river-{index:04d}"
        if tile_features is not None:
            common = {f"data-{key}": str(value) for key, value in attrs.items()}
            common["data-reach-id"] = reach_id
            tile_features.append(TileFeature(channel, {
                **common, "class": "river-channel", "fill": _RIVER_COLOR,
                "fill-rule": "evenodd", "stroke": "none", "clip": "land",
                "data-width-min-m": f"{float(profile.min()):.3f}",
                "data-width-max-m": f"{float(profile.max()):.3f}",
            }, "rivers", section="ink"))
        channels.append(
            f'<path class="river-channel" data-reach-id="{reach_id}" '
            f'data-width-min-m="{float(profile.min()):.3f}" data-width-max-m="{float(profile.max()):.3f}" '
            f'd="{shape_data}" fill="{_RIVER_COLOR}" fill-rule="evenodd" stroke="none"{extra} />'
        )
        sections = navigation_segments[index] if navigation_segments is not None else [(curve,{})]
        for section_curve, navigation in sections:
            section_attrs = {**attrs, **navigation}
            if tile_features is not None:
                common = {f"data-{key}":str(value) for key,value in section_attrs.items()}
                tile_features.extend(line_features([section_curve], "rivers", _RIVER_COLOR, 1.10,
                    attributes={**common, "data-reach-id":reach_id, "class":"river-readable-line",
                                "data-base-stroke":"1.10"}))
            section_extra = "".join(f' data-{key}="{value}"' for key,value in section_attrs.items())
            lines.append(
                f'<path class="river-readable-line" data-reach-id="{reach_id}" '
                f'd="{_path_data(section_curve)}" fill="none" stroke="{_RIVER_COLOR}" '
                'stroke-width="1.10" data-base-stroke="1.10" '
                f'stroke-linecap="round" stroke-linejoin="round"{section_extra} />'
            )
    return (
        f'<g id="rivers" data-layer="rivers" data-width-model="{RIVER_WIDTH_MODEL}">'
        '<g class="river-channel-surfaces" clip-path="url(#land-silhouette-clip)">'
        + "".join(channels) + '</g><g class="river-overview-lines">'
        + "".join(lines) + '</g></g>'
    )


def _filled_path_data(points: np.ndarray, codes: np.ndarray) -> str:
    """Encode shared polygon rings as one even-odd SVG path."""

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
        if np.any(~np.isin(segment_codes, (1, 2, 79))):
            raise WorldGridRenderError("unsupported filled contour code")
        # Encoding must retain shared junctions even when they lie on a
        # straight segment. Only identical delivered neighbours are redundant.
        rounded = []
        for x, y in segment_points:
            point = (round(x * COORDINATE_SCALE), round(y * COORDINATE_SCALE))
            if not rounded or point != rounded[-1]:
                rounded.append(point)
        commands.append(integer_subpath_data(rounded, closed=closed))
    return " ".join(commands)


def _flatten_filled_paths(
    bands: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
) -> list[np.ndarray]:
    return [points for band in bands for points, _ in band]


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


def _religion_display_values(society: SocietyLayers) -> np.ndarray:
    """Paint land outside the religion domain with the neutral legend entry."""
    values = society.religions.religion_id
    return np.where(values == -1, 0, values)


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
    from .cartographic_colors import area_colors
    colors = area_colors(society.politics.state_id, [s.identifier for s in society.politics.states])
    return (("稀疏边疆／无常设政权", "#e5e1d4"), *( (s.name, colors[s.identifier]) for s in society.politics.states ))


def _province_zones(society: SocietyLayers) -> tuple[tuple[str, str], ...]:
    from .cartographic_colors import area_colors
    colors = area_colors(society.provinces.province_id, [p.identifier for p in society.provinces.provinces])
    return (("稀疏边疆／无常设政权", "#e5e1d4"), *( (p.name, colors[p.identifier]) for p in society.provinces.provinces ))


def _partition_boundary_overlay(
    layer_id: str,
    paths: Sequence[np.ndarray],
    *,
    color: str,
    width: float,
    opacity: float = 1.0,
    dash: str | None = None,
    hidden: bool = False,
) -> str:
    dash_attribute = (
        f' stroke-dasharray="{dash}" data-screen-dash="{dash}"' if dash else ""
    )
    body = "".join(
        f'<path d="{_path_data(path)}" fill="none" stroke="{color}" '
        f'stroke-width="{width:.2f}" stroke-linejoin="round" '
        f'stroke-linecap="round" data-screen-stroke="{width:.2f}"{dash_attribute} />'
        for path in paths
    )
    hidden_attribute = " hidden" if hidden else ""
    return (
        f'<g id="{layer_id}" aria-label="分区边界" opacity="{opacity:.2f}"'
        f' clip-path="url(#land-silhouette-clip)"{hidden_attribute}>'
        f'{body}</g>'
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


def _transport_overlay(prepared, *, technology_era: str, tile_features=None) -> str:
    styles = {
        "road": ("#8b5e3c", None, "#f5eddc", 0.85),
        # A dark interrupted centre with a pale casing is the conventional
        # small-scale railway language: it remains legible above relief while
        # the gaps read as sleepers rather than another kind of road.
        "rail": ("#252b32", "1.45 0.82", "#f7f2e8", 0.96),
        "sea": ("#294f75", "8 6", "#7595b5", 0.48),
    }
    widths = {"trunk": 2.20, "regional": 1.65, "local": 1.15}
    if technology_era in {"industrial", "contemporary"}:
        styles["road"] = ("#fff1bb", None, "#6d665c", 0.92)
    parts = ['<g id="transport-network" aria-label="城市交通网络" hidden="hidden">']
    tile_casings,tile_centres=[],[]
    drawing = [
        (mode, importance, _path_data(points),
         shapely.LineString([(round(float(x)*100_000_000)/100_000_000,
                              round(float(y)*100_000_000)/100_000_000) for x,y in points]))
        for mode, importance, points in prepared.paths
    ]
    for mode, importance, path_data, geometry in drawing:
            # Adjacent source points can become one delivered integer point.
            # Such a fragment has no line paint; retain every nonempty path.
            if not path_data:
                continue
            color, dash, casing, casing_opacity = styles[mode]
            base_width = widths[importance]
            if tile_features is not None:
                # Road geometry is already the final engineered polyline.
                # Clip its offscreen runs instead of copying a complete long
                # road into every tile touched by its bounding box. Authored
                # dashed rail/sea paths retain their continuous dash phase.
                tile_geometry=geometry
                tile_path=None if mode=='road' else path_data
                common = {"fill": "none", "stroke-linecap": "round", "stroke-linejoin": "round",
                          "vector-effect": "non-scaling-stroke", "data-route-mode": mode,
                          "data-route-importance": importance, "clip": "water" if mode == "sea" else "land"}
                casing_width = base_width + (0.85 if mode == "road" else 0.72 if mode == "rail" else 0.42)
                tile_casings.append(TileFeature(tile_geometry, {
                    **common, "stroke": casing, "stroke-width": str(casing_width),
                    "stroke-opacity": str(casing_opacity), "data-screen-stroke": str(casing_width),
                    "data-route-casing": "true",
                }, "transport-network", section="ink", path_data=tile_path))
                tile_centres.append(TileFeature(tile_geometry, {
                    **common, "stroke": color, "stroke-width": str(base_width), "stroke-opacity": "0.88",
                    "data-screen-stroke": str(base_width),
                    **({"stroke-dasharray": dash, "data-screen-dash": dash} if dash else {}),
                }, "transport-network", section="ink", path_data=tile_path))
            parts.append(
                f'<path d="{path_data}" fill="none" stroke="{casing}" '
                f'stroke-width="{base_width + (0.85 if mode == "road" else 0.72 if mode == "rail" else 0.42):.2f}" '
                f'stroke-opacity="{casing_opacity:.2f}" '
                f'data-screen-stroke="{base_width + (0.85 if mode == "road" else 0.72 if mode == "rail" else 0.42):.2f}" '
                f'data-route-mode="{mode}" data-route-importance="{importance}" '
                'data-route-casing="true" stroke-linecap="round" stroke-linejoin="round" />'
            )
            parts.append(
                f'<path d="{path_data}" fill="none" stroke="{color}" '
                f'stroke-width="{base_width:.2f}" stroke-opacity="0.88" '
                f'{f"stroke-dasharray=\"{dash}\" " if dash else ""}'
                'stroke-linecap="round" stroke-linejoin="round" '
                f'data-screen-stroke="{base_width:.2f}" '
                f'{f"data-screen-dash=\"{dash}\" " if dash else ""}'
                f'data-route-mode="{mode}" data-route-importance="{importance}" />'
            )
    if tile_features is not None:
        priority={'local':0,'regional':1,'trunk':2}
        order=lambda feature:(feature.attributes['data-route-mode'],priority[feature.attributes['data-route-importance']])
        # Paint all casings before centres so junctions remain open, and
        # neighbouring equal-style roads share one multipart SVG paint.
        tile_features.extend(sorted(tile_casings,key=order))
        tile_features.extend(sorted(tile_centres,key=order))
    parts.append("</g>")
    return "".join(parts)


def _bridge_overlay(prepared) -> str:
    """Draw an overview symbol and the same physical deck used by routing."""

    parts = [
        '<g id="bridge-layer" data-layer="bridges" '
        'aria-label="道路与铁路跨河桥梁" hidden="hidden">'
    ]
    for bridge in prepared.bridges:
        centre = np.asarray(prepared.positions[bridge.identifier], dtype=np.float64)
        tangent = prepared.tangents[bridge.identifier]
        angle = math.degrees(math.atan2(float(tangent[1]), float(tangent[0])))
        x = float(centre[0])
        y = float(centre[1])
        span = shapely.from_geojson(json.dumps(prepared.spans[bridge.identifier]["geometry"]))
        deck_data = geometry_path_data(span)
        parts.append(
            f'<g class="bridge-symbol" transform="translate({_compact_coordinate(x)} {_compact_coordinate(y)})" '
            f'data-map-x="{_compact_coordinate(x)}" data-map-y="{_compact_coordinate(y)}" '
            f'data-base-angle="{_compact_coordinate(angle)}" data-bridge-id="{html.escape(bridge.identifier)}" '
            f'data-route-id="{html.escape(bridge.route_identifier)}" '
            f'data-bridge-importance="{bridge.importance}" '
            f'data-river-order="{bridge.river_order}">'
            f'<g class="bridge-far" transform="rotate({_compact_coordinate(angle)})">'
            '<path d="M-0.9,-1.75 L-0.9,1.75 M0.9,-1.75 L0.9,1.75" '
            'fill="none" stroke="#f7f0df" stroke-width="1.18" '
            'stroke-linecap="round" />'
            '<path d="M-0.9,-1.75 L-0.9,1.75 M0.9,-1.75 L0.9,1.75" '
            'fill="none" stroke="#5f4938" stroke-width="0.46" '
            'stroke-linecap="round" />'
            '</g>'
            f'<g class="bridge-close" transform="translate({_compact_coordinate(-x)} {_compact_coordinate(-y)})" style="display:none">'
            f'<path d="{deck_data}" fill="none" stroke="#514b42" stroke-width="3.2" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>'
            f'<path class="bridge-physical-span" d="{deck_data}" fill="none" stroke="#f3e5c6" stroke-width="2.2" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>'
            f'<path d="{deck_data}" fill="none" stroke="#d7aa71" stroke-width="1.0" vector-effect="non-scaling-stroke" stroke-linejoin="round"/></g>'
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
        np.asarray(grid.river_order),
        np.zeros(grid.shape, dtype=np.int16),
        route_affinity,
        land_mask=grid.water == 0,
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
        population_density(grid, society.population),
        grid.river_order,
    )
    route_cells = {
        route.identifier: set(_path_cells(route.path, grid.shape))
        for route in society.transport.routes
        if route.mode in {"road", "rail"}
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


def _city_overlay(society: SocietyLayers, locations: Mapping[str, tuple[float, float]]) -> str:
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
        y, x = locations[settlement.identifier]
        parts.append(
            f'<g transform="translate({x:.5f} {y:.5f})" '
            f'data-city-symbol-tier="{settlement.tier}" '
            f'data-settlement-id="{html.escape(settlement.identifier, quote=True)}" style="cursor:pointer" '
            f'data-map-x="{x:.5f}" data-map-y="{y:.5f}" '
            f'data-national-capital="{str(settlement.identifier in capital_ids).lower()}" '
            f'data-holy-city="{str(settlement.holy_religion_identifier is not None).lower()}">'
        )
        if settlement.holy_religion_identifier is not None:
            parts.append(
                f'<g transform="scale({radius:.3f})" data-holy-city="true" '
                f'data-religion-id="{settlement.holy_religion_identifier}">{SITE_MARKS["holy"]}</g>'
            )
        if settlement.site_type in {"pass", "fortress"}:
            parts.append(
                f'<g transform="scale({radius:.3f})" data-city-tier="site" '
                f'data-site-type="{settlement.site_type}">{SITE_MARKS[settlement.site_type]}</g>'
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
        if settlement.site_type in {'port', 'island-port', 'lake-port'}:
            parts.append(f'<g transform="translate({radius+2.7:.3f} 0) scale(2.1)" data-harbor-mark="true">{SITE_MARKS["port"]}</g>')
        if settlement.identifier in capital_ids:
            parts.append(
                f'<circle cx="0" cy="0" r="{radius * 1.42:.2f}" fill="none" '
                'stroke="#263447" stroke-width="0.72" data-national-capital="true" />'
            )
        parts.append('</g>')
    parts.append('</g><g id="city-labels" data-capital-font-weight="750">')
    for settlement in society.settlements:
        size = label_size[settlement.tier]
        y, x = locations[settlement.identifier]
        offset = {"metropolis": 5.6, "city": 4.0, "town": 2.7, "site": 3.4}[settlement.tier]
        minimum_screen_size = {
            "metropolis": 10.2,
            "city": 8.9,
            "town": 8.0,
            "site": 8.2,
        }[settlement.tier]
        parts.append(
            f'<text x="{x + offset:.5f}" '
            f'y="{y - offset * 0.35:.5f}" '
            f'font-size="{size:.2f}" fill="#263447" font-weight="'
            f'{"750" if settlement.identifier in capital_ids else "700" if settlement.tier == "metropolis" else "500"}" '
            'paint-order="stroke" stroke="#f7f0df" stroke-width="1.25" '
            'stroke-linejoin="round" '
            f'data-city-label-tier="{settlement.tier}" '
            f'data-settlement-id="{html.escape(settlement.identifier, quote=True)}" style="cursor:pointer" '
            f'data-national-capital="{str(settlement.identifier in capital_ids).lower()}" '
            f'data-holy-city="{str(settlement.holy_religion_identifier is not None).lower()}" '
            f'data-map-x="{x:.5f}" '
            f'data-map-y="{y:.5f}" '
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


def _mountain_label_path(
    grid: WorldGrid, feature: Any, components: np.ndarray,
) -> np.ndarray | None:
    """Route a rounded baseline inside this range's connected highland.

Four-neighbour centre steps keep every quadratic corner in the actual
support cells. Nearby ranges cannot contribute points or PCA weights.
    """
    from scipy import ndimage, sparse
    from scipy.sparse.csgraph import dijkstra

    component = int(components[feature.row, feature.column])
    if component <= 0:
        return None
    rows, columns = np.nonzero(components == component)
    if len(rows) < 3:
        return None
    offsets = (columns - feature.column + grid.shape[1] // 2) % grid.shape[1] - grid.shape[1] // 2
    xs = feature.column + offsets
    coordinates = np.column_stack((xs, rows)).astype(np.float64)
    centered = coordinates - coordinates.mean(axis=0)
    _eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    axis = eigenvectors[:, -1]
    if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
        axis = -axis
    projection = centered @ axis
    across = np.abs(centered @ np.asarray((-axis[1], axis[0])))
    local_rows = rows - int(rows.min())
    local_columns = xs - int(xs.min())
    support = np.zeros((int(local_rows.max()) + 1, int(local_columns.max()) + 1), dtype=bool)
    support[local_rows, local_columns] = True
    depth = ndimage.distance_transform_edt(np.pad(support, 1))[1:-1, 1:-1]
    interior = depth[local_rows, local_columns]
    endpoints = [int(np.argmin(np.abs(projection - float(np.quantile(projection, quantile)))
                               + .35 * across - .22 * interior)) for quantile in (.12, .88)]
    if endpoints[0] == endpoints[1]:
        endpoints = [int(np.argmin(projection)), int(np.argmax(projection))]
    if endpoints[0] == endpoints[1]:
        return None
    indices = np.full(support.shape, -1, dtype=np.int32)
    indices[local_rows, local_columns] = np.arange(len(rows), dtype=np.int32)
    source = []
    target = []
    for first, second in ((indices[:, :-1], indices[:, 1:]), (indices[:-1], indices[1:])):
        valid = (first >= 0) & (second >= 0)
        source.append(first[valid])
        target.append(second[valid])
    source = np.concatenate(source)
    target = np.concatenate(target)
    resistance = 1.0 + 4.0 / (1.0 + interior)
    costs = (resistance[source] + resistance[target]) * .5
    graph = sparse.csr_matrix((costs, (source, target)), shape=(len(rows), len(rows)))
    distance, predecessor = dijkstra(graph, directed=False, indices=endpoints[0], return_predecessors=True)
    if not np.isfinite(distance[endpoints[1]]):
        return None
    route = [endpoints[1]]
    while route[-1] != endpoints[0]:
        route.append(int(predecessor[route[-1]]))
    return _orient_label_path(coordinates[np.asarray(route[::-1])] + .5)


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


def _geographic_symbol_overlay(grid: WorldGrid, thematic: ThematicLayers, society: SocietyLayers, *, raw_elevation_m: np.ndarray) -> str:
    """A point and model-sea-level height for each supported named summit.

    Wetland, snow and arid relief are expressed by the separately clipped
    regional texture layer, rather than hundreds of unrelated point marks.
    """
    raw = np.asarray(raw_elevation_m)
    if raw.shape != grid.shape or not np.isfinite(raw).all():
        raise ValueError("geographic heights require the accepted raw metre DEM")
    parts = ['<g id="geographic-symbols" aria-label="山峰高程点（模型海面基准，米）">', symbol_definitions()]
    for feature in society.geographic_features:
        if feature.feature_type != "peak":
            continue
        x, y = feature.column + .5, feature.row + .5
        height_m = float(raw[feature.row, feature.column])
        if height_m <= 0 or grid.water[feature.row, feature.column] != 0:
            raise ValueError("named summit height must be supported by actual land metres")
        parts.append(f'<g data-geographic-symbol="peak" data-feature-id="{html.escape(feature.identifier, quote=True)}" '
                     f'data-map-x="{x}" data-map-y="{y}" data-min-visible-scale="4" data-max-visible-scale="8193" '
                     f'transform="translate({x} {y})"><use href="#mark-peak"/>'
                     f'<text x="2" y="1.5" font-size="5" fill="#665448" stroke="#f5f0df" stroke-width=".65" '
                     f'paint-order="stroke" data-height-m="{height_m:.8f}" data-height-datum="model-sea-level">'
                     f'{round(height_m)}</text></g>')
    return ''.join(parts) + '</g>'


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
        "desert": ("#916d37", 12.0, "letter-spacing:1px"),
        "wetland": ("#386c70", 11.0, "letter-spacing:0.5px"),
    }
    styles.update({kind: (style["color"], 10.5, "") for kind, style in LANDFORM_STYLES.items()})

    def scale_window(feature: Any) -> tuple[float, float]:
        if feature.feature_type in LANDFORM_STYLES:
            style = LANDFORM_STYLES[feature.feature_type]
            return style["min_scale"], style["max_scale"]
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
        elif feature.feature_type in {"plain", "plateau", "basin", "desert", "wetland"}:
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
    range_components = mountain_components(grid)
    mountain_path_lengths: dict[str, float] = {}
    for feature in society.geographic_features:
        if feature.feature_type == "mountain":
            path = _mountain_label_path(grid, feature, range_components)
            if path is not None:
                path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                natural_text_width = len(feature.name) * styles["mountain"][1] + 1.2 * (len(feature.name) - 1)
                if path_length * .92 >= natural_text_width:
                    curved_paths[feature.identifier] = path
                    mountain_path_lengths[feature.identifier] = path_length
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
        if feature.identifier in mountain_path_lengths:
            # Below this scale, readable glyphs would extend beyond their own
            # range. Compact massifs use the ordinary label at their real anchor.
            available_width = mountain_path_lengths[feature.identifier] * .92 - 1.2 * (len(feature.name) - 1)
            minimum_visible_scale = max(minimum_visible_scale, len(feature.name) * size / available_width)
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
            f'fill="#9b402f" font-size="{size:.2f}" text-anchor="middle" '
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
    land_surface,
    title: str,
    partition_id: str,
    data_attribute: str,
    zones: Sequence[tuple[str, str]],
    extra_overlay: str = "",
    zone_outline_color: str | None = None,
    zone_outline_width: float = 0.0,
    fill_opacity: float = 0.58,
) -> str:
    """Package working paint with its own authoritative physical shore clip."""

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
        'shape-rendering="geometricPrecision" data-surface-contract="shared-land-clip">'
        '<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
        f'<path d="{geometry_path_data(land_surface)}" clip-rule="evenodd" />'
        '</clipPath></defs><g clip-path="url(#land-silhouette-clip)">'
        f'<g id="{partition_id}" data-{data_attribute}="partition" '
        'data-partition="mutually-exclusive" '
        f'aria-label="{html.escape(title)}分区">{"".join(zone_groups)}</g>'
        f'{extra_overlay}</g></svg>'
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
            level_attribute = f' data-level="{float(path_levels[index])}"'
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


def _river_paths(grid: WorldGrid, shoreline_paths: Sequence[np.ndarray], *, raw_elevation_m) -> list[np.ndarray]:
    """Turn flow_to links into edge/reach chains, not cell polygons.

    A reach stops at every indegree junction so each tributary owns the
    shared confluence point and the downstream reach starts at that point.
    """

    active = grid.river_order.reshape(-1) > 0
    if not np.any(active):
        return []
    downstream = grid.flow_to.reshape(-1)
    outlet_targets = hydrologic_outlet_targets(grid, raw_elevation_m)
    cell_count = active.size
    indegree = np.zeros(cell_count, dtype=np.int32)
    for source in np.flatnonzero(active):
        target = int(downstream[source])
        if 0 <= target < cell_count and active[target]:
            indegree[target] += 1

    starts = [int(cell) for cell in np.flatnonzero(active) if indegree[cell] != 1]
    paths: list[np.ndarray] = []
    visited_edges: set[tuple[int, int]] = set()
    shore = shapely.STRtree([shapely.LineString(path) for path in shoreline_paths])

    def water_boundary_point(
        current: int,
        target: int,
    ) -> tuple[float, float] | None:
        row, column = divmod(current, grid.shape[1])
        if target < 0 or target >= cell_count:
            return None
        water_row, water_column = divmod(target, grid.shape[1])
        if (grid.water[water_row, water_column] == 0
                or max(abs(water_row - row), abs(water_column - column)) != 1):
            return None
        origin = shapely.Point(column + .5, row + .5)
        outflow = shapely.LineString((origin.coords[0], (water_column + .5, water_row + .5)))
        candidates = shore.query(outflow, predicate="intersects")
        if not len(candidates):
            raise WorldGridRenderError("native dry-to-wet river outflow must cross the visible shoreline")
        crossings = shapely.intersection(outflow, shapely.union_all(shore.geometries[candidates]))
        # The first encountered shore belongs to this river's landward bank.
        # A nearest projection of the grid midpoint can instead move sideways
        # onto another cove or the far shore of a narrow inlet.
        mouth = shapely.shortest_line(origin, crossings)
        return tuple(mouth.coords[-1])

    def follow(start: int) -> None:
        points: list[tuple[float, float]] = []
        current = start
        while 0 <= current < cell_count and active[current]:
            row, column = divmod(current, grid.shape[1])
            points.append((column + 0.5, row + 0.5))
            target = outlet_targets.get(current, int(downstream[current]))
            if target < 0 or target >= cell_count or not active[target]:
                boundary = water_boundary_point(
                    current,
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


def _map_theme_palettes(society: SocietyLayers) -> dict:
    return {
        "climate": _CLIMATE_ZONES, "biome": _BIOME_ZONES, "watershed": _WATERSHED_ZONES,
        "potential": _LAND_POTENTIAL_ZONES, "habitability": _HABITABILITY_ZONES,
        "vegetation": _VEGETATION_ZONES, "population": _POPULATION_ZONES,
        "civilizations": _culture_zones(society), "languages": _language_zones(society),
        "religions": _religion_zones(society), "political": _political_zones(society),
        "provinces": _province_zones(society),
    }


def _write_map_previews(output: Path, grid: WorldGrid, thematic: ThematicLayers,
                        society: SocietyLayers, vegetation_fraction: np.ndarray,
                        density: np.ndarray) -> None:
    from .atlas_previews import write_theme_previews

    values = {
        "climate": thematic.climate.koppen_code,
        "biome": thematic.biome_zone,
        "watershed": thematic.major_basin_rank,
        "potential": thematic.land_potential_band,
        "habitability": thematic.habitability_band,
        "vegetation": np.digitize(vegetation_fraction, VEGETATION_THRESHOLDS).astype(np.uint8),
        "population": np.where(density > 0, np.digitize(density, POPULATION_DENSITY_THRESHOLDS), 0).astype(np.uint8),
        "civilizations": _civilization_display_values(grid, thematic, society),
        "languages": society.cultures.language_id,
        "religions": _religion_display_values(society),
        "political": society.politics.state_id,
        "provinces": society.provinces.province_id,
    }
    write_theme_previews(output, water=grid.water, values=values, palettes=_map_theme_palettes(society))


def refresh_review_interface(grid: WorldGrid, directory: Path, *, society: SocietyLayers,
                             thematic: ThematicLayers, vegetation_fraction: np.ndarray,
                             density: np.ndarray, physical_source: ProceduralSurface) -> None:
    """Rebuild an interface around its verified saved scene and factual places."""
    from .review_interface import load_interface_snapshot
    from .atlas_ui import write_map_app_assets
    from .svg_groups import replace_group

    overlay, presentation = load_interface_snapshot(
        directory, grid_digest=grid.content_digest(), society_digest=society_content_digest(society))
    from .city_map_assets import write_city_map_assets
    from .terrain_refinement import terrain_from_source
    terrain_field=terrain_from_source(grid,physical_source)
    locations=dict(presentation['settlementLocations'])
    transport=json.loads((directory/'transport-crossings.json').read_text(encoding='utf-8'))
    harbors={}
    grid_digest=grid.content_digest()
    for city in society.settlements:
        saved=json.loads((directory/'city-maps'/f'{city.identifier}.json').read_text(encoding='utf-8'))
        if saved['gridDigest']!=grid_digest:
            raise ValueError('saved harbor belongs to a different physical world')
        if saved['recipe']['harbor']:
            harbors[city.identifier]=saved['recipe']['harbor']
    overlay=replace_group(overlay,'id','city-layer',_city_overlay(society,locations),required=True)
    document = _html_document(grid, grid.content_digest(), overlay, society=society,
        tectonic_diagnostics=presentation['tectonicDiagnostics'],territorial_qa=presentation['territorialQa'],
        width=presentation['width'],height=presentation['height'],viewbox_width=presentation['viewboxWidth'],viewbox_height=presentation['viewboxHeight'])
    if len(document.encode('utf-8'))-len(overlay.encode('utf-8'))>128*1024:
        raise WorldGridRenderError('map interface shell byte budget exceeded')
    write_interface_snapshot(directory,overlay,grid_digest=grid.content_digest(),society_digest=society_content_digest(society),
        width=presentation['width'],height=presentation['height'],viewbox_width=presentation['viewboxWidth'],viewbox_height=presentation['viewboxHeight'],
        tectonic_diagnostics=presentation['tectonicDiagnostics'],territorial_qa=presentation['territorialQa'],settlement_locations=locations)
    write_map_app_assets(directory,grid=grid,society=society,settlement_locations=locations)
    physical_paths=[(p['mode'],p['importance'],p['geometry']['coordinates']) for p in transport['drawnTransportPaths']]
    write_city_map_assets(directory, grid, society, locations,terrain_field=terrain_field,harbors=harbors,physical_paths=physical_paths)
    _write_map_previews(directory, grid, thematic, society, vegetation_fraction, density)
    (directory / "globe.js").write_bytes((Path(__file__).parent / "web/globe.js").read_bytes())
    (directory / "index.html").write_text(document, encoding="utf-8", newline="\n")
    qa = json.loads((directory / "qa.json").read_text(encoding="utf-8"))
    interface_files = [directory / name for name in (
        "index.html", "place-index.json", "atlas-ui.css", "atlas-ui.js", "atlas-tiles.js", "atlas-overview.js",
        "atlas-interaction.js", "atlas-ruler.js", "atlas-navigation.js", "atlas-navigation-worker.js", "navigation-network.json", "travel-profile.json",
        "city-character.js", "city-detail.js", "city-site.js", "city-map.js", "globe.js")]
    interface_files.extend((directory / "city-maps").glob("*.json"))
    interface_files.extend((directory / "map-previews").glob("*.png"))
    qa["artifacts"].update({file.relative_to(directory).as_posix(): file.stat().st_size for file in interface_files})
    (directory / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, sort_keys=True,
                                                 separators=(",", ":")) + "\n", encoding="utf-8")


def _html_document(
    grid: WorldGrid,
    digest: str,
    overlay: str,
    *,
    society: SocietyLayers,
    tectonic_diagnostics: Mapping[str, object],
    territorial_qa: Mapping[str, int | float],
    width: int,
    height: int,
    viewbox_width: int,
    viewbox_height: int,
) -> str:
    from .atlas_ui import ui_markup
    from .svg_groups import replace_group

    # Source geometry is a separately persisted asset. The overview manager
    # keeps its parsed tree detached from the interactive document.
    if 'id="overview-source"' in overlay:
        overlay=replace_group(overlay,'id','overview-source','<g id="overview-source"></g>',required=True)

    safe_digest = html.escape(digest, quote=True)
    world_name = str(grid.metadata["worldProfile"]["name"])
    page_title = html.escape(f"{world_name} · 世界地图")
    polar_scene_band = 0
    crs = grid.metadata["coordinateReferenceSystem"]
    if (crs["contractId"], crs["reviewProjection"], crs["storageGridMapping"], crs["readerProjection"]) != (
        "EIR-GEOG-1", "equirectangular", "plate-carree", "equal-earth"
    ):
        raise WorldGridRenderError("unsupported review projection contract")
    radius_km = float(grid.metadata["planet"]["radiusKm"])
    if not math.isfinite(radius_km) or radius_km <= 0:
        raise WorldGridRenderError("a saved planet needs its actual positive radius")
    extents = grid.metadata["extents"]
    west, north = extents["west"], extents["north"]
    longitude_span = extents["east"] - west
    latitude_span = north - extents["south"]
    palettes = _map_theme_palettes(society)
    legends = {key: '<div class="legend">' + ''.join(
        f'<div class="legend-row"><span class="swatch" style="background:{color}"></span><span>{html.escape(label)}</span></div>'
        for label, color in zones
    ) + '</div>' for key, zones in palettes.items() if key not in ("political", "provinces")}
    legends["none"] = '<div class="legend"><div class="legend-row"><span class="river-line"></span>常年河</div><div class="legend-row"><span class="river-line seasonal"></span>季节河</div><div class="legend-row"><span class="river-line" style="border-color:#195d84"></span>可通航河段（交通开启时）</div><div class="legend-row"><span class="swatch" style="background:#effafa"></span>海洋与湖泊</div></div>'
    country_key = '<div class="legend-row"><svg class="symbol-sample" viewBox="0 0 40 10" aria-hidden="true"><path d="M0 5H40" fill="none" stroke="#46413b" stroke-width="1.4" stroke-dasharray="7 3 1.3 3"/></svg><span>实际国界 · 长划点线</span></div>'
    province_key = '<div class="legend-row"><svg class="symbol-sample" viewBox="0 0 40 10" aria-hidden="true"><path d="M0 5H40" fill="none" stroke="#685f54" stroke-width=".95" stroke-dasharray="4 2 1 2"/></svg><span>省界 · 短划点线</span></div>'
    legends["political"] = '<div class="legend">' + country_key + '</div><p>颜色表示国家；斜线表示无常设政权区。</p>'
    legends["provinces"] = '<div class="legend">' + country_key + province_key + '</div><p>颜色表示省份；无常设国家治理的地区不划省。</p>'
    legends["monsoon"] = '<div class="legend"><div class="legend-row"><span class="swatch precip-scale"></span>降水：少 → 多</div><div class="legend-row"><span class="river-line seasonal"></span>季节河</div></div><p>风向箭头与降水均显示当前季节；四季使用同一色标。</p>'
    legends["tectonic"] = '<div class="legend"><div class="legend-row"><span class="route-line" style="border-color:#405d5d"></span>碰撞／俯冲</div><div class="legend-row"><span class="route-line" style="border-color:#b94e43"></span>分离／扩张</div><div class="legend-row"><span class="route-line" style="border-color:#bd7a35"></span>转换断层</div></div>'
    population_label = _format_population_range(society.population.population_min, society.population.population_max)
    summary = f"{len(society.politics.states)} 个国家 · {len(society.provinces.provinces)} 个省份 · {len(society.settlements)} 处聚落"
    info_html = (
        f'<p>{html.escape(world_name)} · {summary}</p><p>总人口区间：{population_label}。星球半径：{radius_km:g} km。</p>'
        '<p>搜索地点或点击地图上的聚落查看详情。滚轮、触控板或双指缩放；拖动平移。测距可连接多个地点。</p>'
        '<details><summary>地理与坐标说明</summary><p>远景显示世界轮廓，放大后加载更细的地形与海岸。地形、河流与专题共用同一海陆边界。</p>'
        '<p>城市周边深度放大时显示更细的坡面与谷地，淡细等高线的高差间隔为100米；偏远荒野保留较粗地形。</p>'
        '<p>本页使用等距圆柱投影，高纬地区的面积被放大。可结合球体视图阅读；测距按这颗星球的球面计算。</p>'
        f'<p>板块：{tectonic_diagnostics["plateCount"]} 个。自然国界贴合率：{100*float(territorial_qa["stateBoundaryAlignment"]):.1f}%。</p>'
        f'<p class="digest">数据指纹：{safe_digest}</p></details>'
    )
    interface = ui_markup(world_name=world_name, legends=legends, summary=summary, info_html=info_html)
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="grid-digest" content="{safe_digest}">
  <meta name="theme-color" content="#ffffff">
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="atlas-ui.css">
  <title>{page_title}</title>
</head>
<body>
  <main aria-label="{page_title}">
    <section id="viewport" aria-label="世界地图，拖动平移，滚轮缩放" tabindex="0">
      <div class="map-scale" aria-label="当地纬度比例尺"><output id="scale-distance"></output><span id="scale-rule"></span></div>
      <svg id="map-canvas" aria-label="世界地图"><g id="atlas-camera">
        <svg id="overlay" x="0" y="{polar_scene_band}" width="{width}" height="{height}" viewBox="0 0 {viewbox_width} {viewbox_height}" preserveAspectRatio="none" data-render-mode="viewport-tiles" role="img" aria-label="地图图层">{overlay}</svg>
      </g></svg>
    </section>
    {interface}
  </main>
  <script src="atlas-tiles.js"></script>
  <script src="atlas-overview.js"></script>
  <script src="atlas-ruler.js"></script>
  <script src="atlas-navigation.js"></script>
  <script src="atlas-interaction.js"></script>
  <script src="city-character.js"></script>
  <script src="city-detail.js"></script>
  <script src="city-site.js"></script>
  <script src="city-map.js"></script>
  <script src="atlas-ui.js"></script>
  <script>
    const viewport = document.getElementById('viewport');
    let scale = 1;
    let translateX = 0;
    let translateY = 0;
    let dragging = false;
    let cameraInteracted = false;
    let adaptiveUpdateTimer = 0;
    let cameraFrame = 0;
    let minimumScale = 0.2;
    const maximumScale = 8192;
    // Crossing symbols are local details, even when their roads span the world.
    const bridgeThresholds = {{ trunk: 8, regional: 16, local: 32 }};
    const seasons = ['vernal', 'june', 'autumnal', 'december'];
    const themes = ['none','climate','biome','watershed','potential','habitability','vegetation','population','civilizations','languages','religions','political','provinces'];
    const query = new URLSearchParams(window.location.search);
    document.getElementById('toggle-nominal-realms').checked=query.get('nominal')==='1';
    let activeView = ['monsoon','tectonic'].includes(query.get('view')) ? query.get('view') : 'physical';
    let activeTheme = themes.includes(query.get('theme')) ? query.get('theme') : 'none';
    let activeSeason = seasons.includes(query.get('season')) ? query.get('season') : 'vernal';
    const polarSceneBand = {polar_scene_band};
    const atlasScaleX = {width} / {viewbox_width};
    const atlasScaleY = {height} / {viewbox_height};

    let ui = null, selectedPlace = null, selectedSettlementId = null, placeIndexPromise = null;
    const showMapError = message => {{ document.getElementById('map-status').textContent = message; }};
    function loadPlaceIndex() {{
      if (!placeIndexPromise) placeIndexPromise = fetch('place-index.json').then(response => {{
        if (!response.ok) throw new Error('无法载入地点索引');
        return response.json();
      }}).catch(error => {{placeIndexPromise=null; throw error;}});
      return placeIndexPromise;
    }}
    function availableMapArea() {{
      if (viewport.clientWidth <= 700) return {{left:12,top:78,right:viewport.clientWidth-70,bottom:viewport.clientHeight*.54}};
      const navigationPanel=document.getElementById('navigation-panel'),legend=document.getElementById('map-legend').getBoundingClientRect();
      return {{left:document.getElementById('place-panel').getBoundingClientRect().right+24,
        top:90,right:navigationPanel.hidden?viewport.clientWidth-90:navigationPanel.getBoundingClientRect().left-24,
        bottom:Math.min(viewport.clientHeight-70,legend.top>viewport.clientHeight/2?legend.top-24:viewport.clientHeight-70)}};
    }}
    function focusPlace(place) {{
      const area=availableMapArea(), availableWidth=Math.max(180,area.right-area.left),
        availableHeight=Math.max(120,area.bottom-area.top);
      let x=place.native[0],y=place.native[1];
      if (place.bounds) {{
        const [left,top,right,bottom]=place.bounds;
        x=(left+right)/2;y=(top+bottom)/2;
        scale=Math.min(availableWidth/Math.max(1,(right-left)*atlasScaleX)*.88,
          availableHeight/Math.max(1,(bottom-top)*atlasScaleY)*.88);
      }} else {{
        const kmPerColumn=2*Math.PI*{radius_km}*Math.max(.08,Math.cos(place.latitude*Math.PI/180))/{viewbox_width};
        scale=availableWidth/(600/kmPerColumn*atlasScaleX);
      }}
      scale=Math.max(minimumScale,Math.min(maximumScale,scale));
      translateX=(area.left+area.right)/2-x*atlasScaleX*scale;
      translateY=(area.top+area.bottom)/2-(y*atlasScaleY+polarSceneBand)*scale;
      cameraInteracted=true;constrainCamera();applyTransform();scheduleAdaptiveUpdate(true);saveCamera();
    }}
    function updateSelectedMarker() {{
      document.getElementById('selected-place-marker')?.remove();
      if (!selectedPlace) return;
      const namespace='http://www.w3.org/2000/svg', group=document.createElementNS(namespace,'g');
      group.id='selected-place-marker';group.setAttribute('pointer-events','none');
      const title=document.createElementNS(namespace,'title');title.textContent=selectedPlace.name;group.append(title);
      for (const [color,width] of [['#ffffff',16],['#087bff',11],['#ffffff',3]]) {{
        const path=document.createElementNS(namespace,'path');
        for (const [key,value] of Object.entries({{d:'M'+selectedPlace.native.join(' ')+'h0',fill:'none',stroke:color,
          'stroke-width':width,'stroke-linecap':'round','vector-effect':'non-scaling-stroke'}})) path.setAttribute(key,String(value));
        group.append(path);
      }}
      document.getElementById('overlay').append(group);
    }}
    function selectMapSettlement(identifier) {{
      ui.selectPlace('city:'+identifier,{{center:false}}).catch(error=>showMapError(error.message));
    }}
    document.getElementById('city-layer').addEventListener('click',event=>{{
      const target=event.target.closest('[data-settlement-id]');
      if (event.detail===0 && target && !ruler.active) selectMapSettlement(target.dataset.settlementId);
    }});
    const staticNodeCache = new Map();
    function updateNominalRealms() {{
      const enabled=activeView==='physical' && document.getElementById('toggle-nominal-realms').checked;
      document.getElementById('nominal-legend').hidden=!enabled;
    }}
    function staticNodes(selector) {{
      if (!staticNodeCache.has(selector)) staticNodeCache.set(selector,Array.from(viewport.querySelectorAll(selector)));
      return staticNodeCache.get(selector);
    }}
    let previousStrokeScale = NaN;
    let orderedGeographicLabels = null;
    const overviewManager = WorldAtlasOverview.create({{
      sourceURL:'overview-source.svg',
      container:document.getElementById('overview-world'),
      width:{viewbox_width},height:{viewbox_height},
      loadError:message => {{showMapError(message);}},
    }});
    const tileManager = WorldAtlasTiles.create({{
      container:document.getElementById('detail-world'),
      onMounted:() => {{
        updateTilePresentation();updateRiverAppearance();updateTransportAppearance();
        updateMapCoverage();
      }},
      loadError:message => {{showMapError(message);}},
    }});
    function tileLayerVisibility() {{
      const physical=activeView==='physical', tectonic=activeView==='tectonic';
      const administrative=physical && ['political','provinces'].includes(activeTheme);
      const layers={{'surface-backfill':!tectonic,'polar-land-surface':!tectonic,
        'transport-network':physical && document.getElementById('toggle-transport').checked,
        'civilization-boundaries':physical && ['civilizations','languages'].includes(activeTheme),
        'language-boundaries':physical && activeTheme==='languages',
        'language-civilization-boundaries':physical && activeTheme==='languages',
        'religion-boundaries':physical && activeTheme==='religions',
        'nominal-realms':physical && document.getElementById('toggle-nominal-realms').checked,
        'geographic-textures':physical && physicalToggleEnabled('geographic-symbols'),
        'province-boundaries':physical && activeTheme==='provinces',
        'state-boundaries':administrative || physical && activeTheme==='none' && document.getElementById('toggle-state-outline').checked,
        'rivers':physical ? physicalToggleEnabled('rivers') : !tectonic && document.getElementById('toggle-seasonal-rivers').checked}};
      for (const name of ['elevation-bands','bathymetry-bands','coast','lakes','sea-ice']) layers[name]=!tectonic && physicalToggleEnabled(name);
      for (const name of ['elevation-contours','snow']) layers[name]=physical && physicalToggleEnabled(name);
      return layers;
    }}
    function updateTilePresentation() {{
      const thematic=activeView==='physical' && activeTheme!=='none';
      const administrative=thematic && ['political','provinces'].includes(activeTheme);
      const quantitative=thematic && ['potential','habitability','vegetation','population'].includes(activeTheme);
      const visibility=tileLayerVisibility();
      for (const node of viewport.querySelectorAll('[data-tile-layer]')) {{
        const layer=node.dataset.tileLayer;
        setElementVisible(node,visibility[layer]!==false);
        if (layer==='elevation-bands') node.setAttribute('opacity',quantitative ? '.12' : administrative ? '.16' : thematic ? '.34' : '1');
        if (layer==='bathymetry-bands') node.setAttribute('opacity',administrative ? '.72' : thematic ? '.62' : '1');
        if (layer==='elevation-contours') node.setAttribute('opacity',quantitative ? '.24' : administrative ? '.30' : thematic ? '.60' : '1');
        if (layer==='geographic-textures') node.setAttribute('opacity',quantitative ? '.28' : administrative ? '.60' : thematic ? '.75' : '1');
      }}
    }}
    function updateMapCoverage() {{
      const detailed=scale>=8 && activeView!=='tectonic';
      setElementVisible(document.getElementById('overview-world'),activeView!=='tectonic'
        && (!detailed || !tileManager.stats().baseTilesReady));
      setElementVisible(document.getElementById('detail-world'),detailed);
    }}
    function updateViewportTiles() {{
      const detailed=scale>=8 && activeView!=='tectonic';
      const preload=scale>=6 && activeView!=='tectonic';
      const worldBounds=[-translateX/scale/atlasScaleX,(-translateY/scale-polarSceneBand)/atlasScaleY,
        (viewport.clientWidth-translateX)/scale/atlasScaleX,((viewport.clientHeight-translateY)/scale-polarSceneBand)/atlasScaleY];
      tileManager.update({{
        scale:preload ? Math.max(scale,8) : Math.min(scale,3.99),
        worldBounds,
        theme:activeView==='physical' ? activeTheme : 'none',season:activeSeason,layerVisibility:tileLayerVisibility(),
      }});
      updateMapCoverage();
      if (activeView!=='tectonic' && (!detailed || !tileManager.stats().baseTilesReady)) overviewManager.update({{
        theme:activeView==='physical' ? activeTheme : 'none',
        season:activeSeason,view:activeView,layerVisibility:tileLayerVisibility(),
        scale:scale*Math.max(atlasScaleX,atlasScaleY),worldBounds,pixelRatio:window.devicePixelRatio,
        refine:!detailed,
      }});
    }}

    function updateAdaptiveStrokes() {{
      updateSelectedMarker();
      const centerRow=((viewport.clientHeight/2-translateY)/scale-polarSceneBand)/atlasScaleY;
      const centerLatitude=Math.max(-89,Math.min(89,90-centerRow/{viewbox_height}*180));
      const kmPerScreenPixel=2*Math.PI*{radius_km}*Math.cos(centerLatitude*Math.PI/180)/{viewbox_width}/atlasScaleX/scale;
      const targetKm=110*kmPerScreenPixel;
      const magnitude=Math.pow(10,Math.floor(Math.log10(targetKm)));
      const niceKm=([5,2,1].find(value=>value*magnitude<=targetKm)||1)*magnitude;
      document.getElementById('scale-distance').textContent=niceKm<1 ? `${{Math.round(niceKm*1000)}} m` : `${{Number(niceKm.toPrecision(3))}} km`;
      document.getElementById('scale-rule').style.width=`${{niceKm/kmPerScreenPixel}}px`;
      staticNodes('[data-geographic-symbol]').forEach(symbol => {{
        const x=Number(symbol.dataset.mapX),y=Number(symbol.dataset.mapY);
        const sx=x*atlasScaleX*scale+translateX, sy=(y*atlasScaleY+polarSceneBand)*scale+translateY;
        const visible=scale>=Number(symbol.dataset.minVisibleScale) && scale<Number(symbol.dataset.maxVisibleScale) && sx>-16 && sx<viewport.clientWidth+16 && sy>-16 && sy<viewport.clientHeight+16;
        symbol.style.display=visible ? '' : 'none';
        if (visible) symbol.setAttribute('transform',`translate(${{x}} ${{y}}) scale(${{Math.min(1,2/scale).toFixed(5)}})`);
      }});
      const widthFactor = 1 / Math.max(0.00001, scale * atlasScaleX);
      if (previousStrokeScale !== scale) staticNodes('[data-base-stroke]:not([vector-effect="non-scaling-stroke"])').forEach((path) => {{
        const base = Number(path.dataset.baseStroke);
        if (Number.isFinite(base)) {{
          const minimum = path.closest('[data-tile-layer="elevation-contours"]') ? 0.45 : 0.95;
          path.setAttribute('stroke-width', (Math.max(minimum, base) * widthFactor).toFixed(7));
        }}
      }});
      const screenMapFactor = widthFactor;
      const detailLevel = scale < 0.85 ? 'world' : scale < 2.4 ? 'regional' : scale < 6 ? 'local' : 'close';
      document.documentElement.dataset.mapDetail = detailLevel;
      if (previousStrokeScale !== scale) staticNodes('[data-screen-stroke]:not([vector-effect="non-scaling-stroke"])').forEach((path) => {{
        const screenWidth = Number(path.dataset.screenStroke);
        if (Number.isFinite(screenWidth)) {{
          const mapWidth = Math.max(0.95, screenWidth) * screenMapFactor;
          path.setAttribute('stroke-width', mapWidth.toFixed(7));
        }}
        const screenDash = (path.dataset.screenDash || '')
          .trim().split(/\\s+/).map(Number);
        if (screenDash.length && screenDash.every(Number.isFinite)) {{
          path.setAttribute(
            'stroke-dasharray',
            screenDash.map((length) => (length * screenMapFactor).toFixed(7)).join(' '),
          );
        }}
      }});
      previousStrokeScale=scale;
      const placed = [];
      const labelQueue = [];
      const placeLabel = (label, eligible, horizontalMargin = 3, verticalMargin = 2) => {{
        if (eligible && label.hasAttribute('x') && label.hasAttribute('y')) {{
          const x=Number(label.getAttribute('x'))*atlasScaleX*scale+translateX;
          const y=(Number(label.getAttribute('y'))*atlasScaleY+polarSceneBand)*scale+translateY;
          eligible=x>-120 && x<viewport.clientWidth+120 && y>-60 && y<viewport.clientHeight+60;
        }}
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
            : label.dataset.cityLabelTier ? {{metropolis:20,city:17,town:14,site:15}}[label.dataset.cityLabelTier] : fontSize * 3.0;
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
        labelQueue.push({{label, horizontalMargin, verticalMargin}});
      }};

      const positionCityLabel = label => {{
        const x = Number(label.dataset.mapX), y = Number(label.dataset.mapY);
        const offsetX = Number(label.dataset.baseOffsetX), offsetY = Number(label.dataset.baseOffsetY);
        const markerScale = Math.min(1, 1.8 / Math.max(.01,scale*atlasScaleX));
        if ([x, y, offsetX, offsetY].every(Number.isFinite)) {{
          label.setAttribute('x', (x + offsetX * markerScale).toFixed(5));
          label.setAttribute('y', (y + offsetY * markerScale).toFixed(5));
        }}
      }};
      const physical = activeView === 'physical';
      const labelsEnabled = physical && (
        document.getElementById('toggle-cities').checked ||
        document.getElementById('toggle-transport').checked
      );
      const cityLabels = staticNodes('#city-labels [data-city-label-tier]');
      const focusedCityLabel = cityLabels.find(label => label.dataset.settlementId === selectedSettlementId);
      // The requested place owns its label space before surrounding map labels.
      if (focusedCityLabel) {{
        positionCityLabel(focusedCityLabel);
        placeLabel(focusedCityLabel, labelsEnabled);
      }}
      const stateOutlineEnabled = physical && (['political','provinces'].includes(activeTheme) || activeTheme === 'none' && document.getElementById('toggle-state-outline').checked);
      if (stateOutlineEnabled) {{
        const thresholds = activeTheme === 'provinces'
          ? {{ major: 0, medium: 0.60, small: 1.20 }}
          : {{ major: 0, medium: 0.46, small: 0.90 }};
        const stateLabelMaximum = activeTheme === 'provinces' ? 5.4 : 10.5;
        staticNodes('#state-labels [data-state-size]').forEach((label) => {{
          placeLabel(
            label,
            scale >= thresholds[label.dataset.stateSize] && scale < stateLabelMaximum,
            5,
            3,
          );
        }});
        const frontierThresholds = {{ major: 0, secondary: 1.42 }};
        staticNodes('#frontier-group-labels [data-frontier-tier]').forEach((label) => {{
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
        viewport.querySelectorAll(`#${{thematicLabelId}}-labels [data-theme-label]`).forEach((label) => {{
          placeLabel(label, scale >= minimumScale && scale < maximumScale, 5, 3);
        }});
      }}

      // Province names are the primary reading layer in the province theme.
      // Reserve their deep-interior anchors before geographic and city labels
      // compete for the remaining screen space.
      const provinceLabels = staticNodes('#province-labels [data-province-id]');
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
      const geographicLabels = orderedGeographicLabels ||= staticNodes('#geographic-labels text[data-feature-tier]').sort((first, second) => {{
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
      cityLabels.forEach(label => {{
        if (label === focusedCityLabel) return;
        positionCityLabel(label);
        placeLabel(label, labelsEnabled && scale >= thresholds[label.dataset.cityLabelTier]);
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
      staticNodes('#city-symbols [data-city-symbol-tier]').forEach((symbol) => {{
        const tier = symbol.dataset.citySymbolTier;
        const sx=Number(symbol.dataset.mapX)*atlasScaleX*scale+translateX;
        const sy=(Number(symbol.dataset.mapY)*atlasScaleY+polarSceneBand)*scale+translateY;
        const eligible = labelsEnabled && scale >= symbolThresholds[tier] && sx>-24 && sx<viewport.clientWidth+24 && sy>-24 && sy<viewport.clientHeight+24;
        symbol.style.display = eligible ? '' : 'none';
        if (!eligible) return;
        const x = Number(symbol.dataset.mapX);
        const y = Number(symbol.dataset.mapY);
        const symbolScale = Math.min(1, 1.5 / Math.max(.01,scale*atlasScaleX));
        symbol.setAttribute(
          'transform',
          `translate(${{x.toFixed(5)}} ${{y.toFixed(5)}}) scale(${{symbolScale.toFixed(5)}})`,
        );
      }});
      updateTransportAppearance();
      staticNodes('#bridge-layer [data-bridge-importance]').forEach((symbol) => {{
        const sx=Number(symbol.dataset.mapX)*atlasScaleX*scale+translateX;
        const sy=(Number(symbol.dataset.mapY)*atlasScaleY+polarSceneBand)*scale+translateY;
        const eligible = transportEnabled && scale >= bridgeThresholds[symbol.dataset.bridgeImportance] && sx>-24 && sx<viewport.clientWidth+24 && sy>-24 && sy<viewport.clientHeight+24;
        symbol.style.display = eligible ? '' : 'none';
        if (!eligible) return;
        const x = Number(symbol.dataset.mapX);
        const y = Number(symbol.dataset.mapY);
        const angle = Number(symbol.dataset.baseAngle);
        const span = symbol.querySelector('.bridge-physical-span');
        const projectedLength = span.getTotalLength() * atlasScaleX * scale;
        const close = projectedLength >= 12;
        const reading = symbol.querySelector('.bridge-far');
        reading.style.display = close ? 'none' : '';
        symbol.querySelector('.bridge-close').style.display = close ? '' : 'none';
        const symbolScale = 14 / (4.68 * atlasScaleX * scale);
        reading.setAttribute('transform', `rotate(${{angle.toFixed(8)}}) scale(${{symbolScale}})`);
        symbol.setAttribute(
          'transform',
          `translate(${{x.toFixed(8)}} ${{y.toFixed(8)}})`,
        );
      }});
      const measuredLabels = labelQueue.map(item => ({{...item, rect:item.label.getBoundingClientRect()}}));
      for (const {{label, rect, horizontalMargin, verticalMargin}} of measuredLabels) {{
        const collision = placed.some(other => !(
          rect.right + horizontalMargin < other.left || rect.left - horizontalMargin > other.right ||
          rect.bottom + verticalMargin < other.top || rect.top - verticalMargin > other.bottom
        ));
        if (collision) label.style.display='none';
        else placed.push(rect);
      }}
    }}

    function scheduleAdaptiveUpdate(immediate = false) {{
      window.clearTimeout(adaptiveUpdateTimer);
      if (dragging && !immediate) return;
      if (immediate) {{
        updateAdaptiveStrokes();
        return;
      }}
      adaptiveUpdateTimer = window.setTimeout(updateAdaptiveStrokes, 120);
    }}

    function flushCameraTransform() {{
      cameraFrame = 0;
      document.getElementById('atlas-camera').setAttribute('transform',`translate(${{translateX}} ${{translateY}}) scale(${{scale}})`);
      // Hide immediately while zooming out, before the deferred symbol update.
      document.getElementById('bridge-layer').style.visibility = scale < bridgeThresholds.trunk ? 'hidden' : '';
      updateViewportTiles();
    }}

    function requestCameraTransform() {{
      if (!cameraFrame) cameraFrame = window.requestAnimationFrame(flushCameraTransform);
    }}

    function applyTransform() {{
      if (cameraFrame) window.cancelAnimationFrame(cameraFrame);
      flushCameraTransform();
      window.navigation?.updateScale(scale*atlasScaleX);
      const levelNames = {{ world: '全图', regional: '区域', local: '地方', close: '近景' }};
      const detailLevel = scale < 0.85 ? 'world' : scale < 2.4 ? 'regional' : scale < 6 ? 'local' : 'close';
      if (document.documentElement.dataset.mapDetail !== detailLevel) document.documentElement.dataset.mapDetail = detailLevel;
      const zoomLevel = document.getElementById('zoom-level');
      if (zoomLevel.value !== levelNames[detailLevel]) zoomLevel.value = levelNames[detailLevel];
      const zoomTitle = `当前缩放 ${{Math.round(scale * 100)}}%`;
      if (zoomLevel.title !== zoomTitle) zoomLevel.title = zoomTitle;
      scheduleAdaptiveUpdate(false);
    }}


    function constrainCamera() {{
      if ({viewbox_width}*atlasScaleX*scale<=viewport.clientWidth) translateX=(viewport.clientWidth-{viewbox_width}*atlasScaleX*scale)/2;
      else translateX=Math.max(viewport.clientWidth-{viewbox_width}*atlasScaleX*scale,Math.min(0,translateX));
      const sceneHeight={viewbox_height}*atlasScaleY+2*polarSceneBand;
      if (sceneHeight*scale<=viewport.clientHeight) translateY=(viewport.clientHeight-sceneHeight*scale)/2;
      else translateY=Math.max(viewport.clientHeight-sceneHeight*scale,Math.min(0,translateY));
    }}
    function screenToNative(clientX,clientY) {{
      const bounds=viewport.getBoundingClientRect();
      return [(clientX-bounds.left-translateX)/scale/atlasScaleX,
        ((clientY-bounds.top-translateY)/scale-polarSceneBand)/atlasScaleY];
    }}
    function saveCamera() {{
      const center=screenToNative(viewport.clientWidth/2,viewport.clientHeight/2), url=new URL(location.href);
      url.searchParams.set('zoom',Number(scale.toFixed(7)));
      url.searchParams.set('x',Number(center[0].toFixed(5)));
      url.searchParams.set('y',Number(center[1].toFixed(5)));
      history.replaceState({{...history.state,atlasCamera:{{scale,translateX,translateY}}}},'',url.pathname+url.search+url.hash);
    }}
    function restoreCamera() {{
      const query=new URLSearchParams(location.search);
      if (!['zoom','x','y'].every(key=>query.has(key))) return false;
      const zoom=Number(query.get('zoom')),x=Number(query.get('x')),y=Number(query.get('y'));
      if (![zoom,x,y].every(Number.isFinite) || zoom<=0 || x<0 || x>{viewbox_width} || y<-polarSceneBand/atlasScaleY || y>{viewbox_height}+polarSceneBand/atlasScaleY) return false;
      scale=Math.max(minimumScale,Math.min(maximumScale,zoom));
      translateX=viewport.clientWidth/2-x*atlasScaleX*scale;
      translateY=viewport.clientHeight/2-(y*atlasScaleY+polarSceneBand)*scale;
      cameraInteracted=true;constrainCamera();return true;
    }}
    function zoomAt(factor,clientX,clientY) {{
      const bounds=viewport.getBoundingClientRect();
      const x=(clientX-bounds.left-translateX)/scale,y=(clientY-bounds.top-translateY)/scale;
      const nextScale=Math.max(minimumScale,Math.min(maximumScale,scale*factor));
      translateX=clientX-bounds.left-x*nextScale;translateY=clientY-bounds.top-y*nextScale;
      scale=nextScale;cameraInteracted=true;constrainCamera();applyTransform();
    }}
    function zoomAtCenter(factor) {{
      const bounds=viewport.getBoundingClientRect();
      zoomAt(factor,bounds.left+bounds.width/2,bounds.top+bounds.height/2);
      saveCamera();
    }}
    function resetView() {{
      ui?.closePlace({{history:false}});
      const url=new URL(location.href);url.searchParams.delete('place');
      history.replaceState(null,'',url.pathname+url.search+url.hash);
      cameraInteracted=false;fitScene();applyTransform();saveCamera();
    }}
    function fitScene() {{
      scale=Math.min(viewport.clientWidth/({viewbox_width}*atlasScaleX),viewport.clientHeight/({viewbox_height}*atlasScaleY));
      minimumScale=scale;
      translateX=(viewport.clientWidth-{viewbox_width}*atlasScaleX*scale)/2;
      translateY=(viewport.clientHeight-{viewbox_height}*atlasScaleY*scale)/2-polarSceneBand*scale;
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
      const riverGroups = viewport.querySelectorAll('[data-tile-layer="rivers"]');
      if (!riverGroups.length) return;
      if (activeView === 'tectonic') {{
        riverGroups.forEach(group => setElementVisible(group, false));
        return;
      }}
      const seasonIndex = seasons.indexOf(activeSeason);
      const seasonalEnabled = document.getElementById('toggle-seasonal-rivers').checked;
      const physical = activeView === 'physical';
      riverGroups.forEach(group => setElementVisible(group, physical ? physicalToggleEnabled('rivers') : seasonalEnabled));
      viewport.querySelectorAll('[data-tile-layer="rivers"] path').forEach((path) => {{
        const navigable=physical && document.getElementById('toggle-transport').checked && path.dataset.navigable==='true';
        const riverColor=navigable ? path.dataset.navigationColor : path.dataset.riverColor;
        if(riverColor)path.setAttribute(path.classList.contains('river-channel') ? 'fill' : 'stroke',riverColor);
        const strengths = (path.dataset.seasonalStrengths || '0,0,0,0').split(',').map(Number);
        const strength = Number.isFinite(strengths[seasonIndex]) ? strengths[seasonIndex] : 0;
        const disappears = strengths.some((value) => value <= 0);
        path.style.display = physical || strength > 0 ? '' : 'none';
        path.setAttribute('opacity', physical ? '1' : strength >= 48 ? '0.96' : '0.68');
        if (path.classList.contains('river-readable-line')) {{
          path.dataset.seasonalDash = String(disappears);
          path.setAttribute('stroke-dasharray', disappears
            ? (path.getAttribute('vector-effect')==='non-scaling-stroke' ? '3 2' : `${{3/scale/atlasScaleX}} ${{2/scale/atlasScaleX}}`) : 'none');
          path.setAttribute('stroke-width',path.getAttribute('vector-effect')==='non-scaling-stroke' ? '1.10' : (1.10 / scale / atlasScaleX).toFixed(7));
        }}
      }});
    }}

    function updateTransportAppearance() {{
      const enabled=activeView==='physical' && document.getElementById('toggle-transport').checked;
      const thresholds=activeTheme==='provinces' ? {{trunk:0,regional:.96,local:1.82}}
        : activeTheme==='political' ? {{trunk:0,regional:1.08,local:2.18}}
        : {{trunk:0,regional:1.22,local:2.48}};
      viewport.querySelectorAll('[data-route-importance]').forEach(path => {{
        path.style.display=enabled && scale>=thresholds[path.dataset.routeImportance] ? '' : 'none';
      }});
    }}

    function updateThematicState() {{
      const monsoon = activeView === 'monsoon';
      const tectonic = activeView === 'tectonic';
      const physical = activeView === 'physical';
      document.getElementById('physical-controls').hidden = !physical;
      document.getElementById('monsoon-controls').hidden = !monsoon;
      document.getElementById('tectonic-controls').hidden = !tectonic;
      document.querySelectorAll('[data-view-button]').forEach((button) => {{
        button.setAttribute('aria-pressed', String(button.dataset.viewButton === activeView));
      }});
      document.querySelectorAll('[data-season-button]').forEach((button) => {{
        button.setAttribute('aria-pressed', String(button.dataset.seasonButton === activeSeason));
      }});

      const precipitationEnabled = document.getElementById('toggle-monsoon-precipitation').checked;
      viewport.querySelectorAll('.monsoon-precipitation').forEach((image) => {{
        setElementVisible(image, monsoon && precipitationEnabled && image.dataset.season === activeSeason);
      }});
      const windEnabled = document.getElementById('toggle-monsoon-wind').checked;
      viewport.querySelectorAll('.seasonal-wind').forEach((group) => {{
        setElementVisible(group, monsoon && windEnabled && group.dataset.season === activeSeason);
      }});
      viewport.querySelectorAll('[data-theme-map]').forEach((image) => {{
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
      const quantitativeActive = thematicActive && ['potential','habitability','vegetation','population'].includes(activeTheme);
      document.documentElement.dataset.activeTheme = physical ? activeTheme : activeView;
      const elevationBands = document.getElementById('elevation-bands');
      const bathymetryBands = document.getElementById('bathymetry-bands');
      const elevationContours = document.getElementById('elevation-contours');
      if (elevationBands) elevationBands.setAttribute('opacity', quantitativeActive ? '0.12' : administrativeActive ? '0.16' : thematicActive ? '0.34' : '1');
      if (bathymetryBands) bathymetryBands.setAttribute('opacity', administrativeActive ? '0.72' : thematicActive ? '0.62' : '1');
      if (elevationContours) elevationContours.setAttribute('opacity', quantitativeActive ? '0.24' : administrativeActive ? '0.18' : thematicActive ? '0.36' : '0.60');

      const citiesEnabled = physical && document.getElementById('toggle-cities').checked;
      const transportEnabled = physical && document.getElementById('toggle-transport').checked;
      setElementVisible(document.getElementById('city-layer'), citiesEnabled || transportEnabled);
      setElementVisible(document.getElementById('transport-network'), transportEnabled);
      setElementVisible(document.getElementById('bridge-layer'), transportEnabled);
      setElementVisible(document.getElementById('civilization-labels'), physical && activeTheme === 'civilizations');
      setElementVisible(document.getElementById('language-labels'), physical && activeTheme === 'languages');
      setElementVisible(document.getElementById('religion-labels'), physical && activeTheme === 'religions');
      setElementVisible(document.getElementById('civilization-boundaries'), physical && ['civilizations','languages'].includes(activeTheme));
      setElementVisible(document.getElementById('language-boundaries'), physical && activeTheme === 'languages');
      setElementVisible(document.getElementById('language-civilization-boundaries'), physical && activeTheme === 'languages');
      setElementVisible(document.getElementById('religion-boundaries'), physical && activeTheme === 'religions');
      const stateOutlineEnabled = administrativeActive || physical && activeTheme === 'none' && document.getElementById('toggle-state-outline').checked;
      setElementVisible(document.getElementById('state-labels'), stateOutlineEnabled);
      setElementVisible(document.getElementById('province-labels'), provinceActive);
      setElementVisible(document.getElementById('frontier-group-labels'), administrativeActive);
      setElementVisible(document.getElementById('province-boundaries'), provinceActive);
      setElementVisible(document.getElementById('state-boundaries'), stateOutlineEnabled);
      updateNominalRealms();
      setElementVisible(document.getElementById('surface-backfill'), !tectonic);

      for (const layerId of ['elevation-bands','bathymetry-bands','graticule','coast','lakes','inland-seas','sea-ice','polar-references']) {{
        setElementVisible(document.getElementById(layerId), !tectonic && physicalToggleEnabled(layerId));
      }}
      for (const layerId of ['elevation-contours','snow','geographic-symbols']) {{
        setElementVisible(document.getElementById(layerId), physical && physicalToggleEnabled(layerId));
      }}
      document.getElementById('landform-legend').hidden = !physical || !physicalToggleEnabled('geographic-symbols');
      document.getElementById('settlement-legend').hidden = !physical || !document.getElementById('toggle-cities').checked;
      setElementVisible(
        document.getElementById('polar-labels'),
        physical && physicalToggleEnabled('polar-references'),
      );
      setElementVisible(
        document.getElementById('geographic-labels'),
        physical && physicalToggleEnabled('geographic-labels'),
      );
      updateRiverAppearance();
      updateTilePresentation();updateViewportTiles();
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
      if(document.getElementById('toggle-nominal-realms').checked)nextQuery.set('nominal','1');
      else nextQuery.delete('nominal');
      const suffix = nextQuery.toString();
      window.history.replaceState(history.state, '', `${{window.location.pathname}}${{suffix ? '?' + suffix : ''}}`);
      ui?.sync({{view:activeView,theme:activeTheme,season:activeSeason}});
      document.getElementById('globe-link').href='globe.html?theme='+encodeURIComponent(activeView==='physical' && activeTheme!=='none' ? activeTheme : 'terrain');
    }}


    const ruler=WorldAtlasRuler.create({{svgOverlay:document.getElementById('overlay'),worldWidth:{viewbox_width},radiusKm:{radius_km},
      mapNativeToGeo:([x,y])=>[{west}+x/{viewbox_width}*{longitude_span},{north}-y/{viewbox_height}*{latitude_span}],
      onChange:state=>{{viewport.classList.toggle('measuring',state.active);ui?.setRulerState(state);}}}});
    const navigation=WorldAtlasNavigation.create({{overlay:document.getElementById('overlay'),
      onPicking:active=>{{if(active)ruler.finish();viewport.classList.toggle('navigating',active);}},
      onFit:points=>{{if(!points.length)return;const bounds=points.reduce((b,p)=>[Math.min(b[0],p[0]),Math.min(b[1],p[1]),Math.max(b[2],p[0]),Math.max(b[3],p[1])],[Infinity,Infinity,-Infinity,-Infinity]);
        focusPlace({{bounds,native:points[0]}});
        navigation.updateScale(scale*atlasScaleX);}},
    }});
    window.navigation=navigation;
    window.addEventListener('popstate',()=>{{
      const url=new URLSearchParams(location.search);
      activeView=['monsoon','tectonic'].includes(url.get('view')) ? url.get('view') : 'physical';
      activeTheme=themes.includes(url.get('theme')) ? url.get('theme') : 'none';
      activeSeason=seasons.includes(url.get('season')) ? url.get('season') : 'vernal';
      document.getElementById('toggle-nominal-realms').checked=url.get('nominal')==='1';
      if (!restoreCamera()) {{cameraInteracted=false;fitScene();}}
      updateThematicState();applyTransform();
    }});
    const cityMap=WorldAtlasCityMap.create({{viewport:document.getElementById('city-map-viewport'),container:document.getElementById('city-fabric-layer')}});
    window.cityMap=cityMap;
    ui=WorldAtlasUI.create({{places:loadPlaceIndex,
      onRegenerateCity:async place=>{{const recipe=await cityMap.open(place);if(recipe)ui.setCityDetails(recipe);}},
      onSelectPlace:async (place,{{restore,center=true,generate=true}})=>{{
        selectedPlace=place;selectedSettlementId=place.kind==='city' ? place.sourceId : null;updateSelectedMarker();
        if (place.kind==='city') document.getElementById('toggle-cities').checked=true;
        if (center && !(restore && queryHasCamera())) focusPlace(place);
        updateThematicState();scheduleAdaptiveUpdate(true);
        if(place.kind==='city'){{
          if(generate){{const recipe=await cityMap.open(place);if(recipe)ui.setCityDetails(recipe);}}
        }}else cityMap.clear();
      }},
      onClosePlace:()=>{{cityMap.clear();selectedPlace=null;selectedSettlementId=null;updateSelectedMarker();scheduleAdaptiveUpdate(true);}},
      onViewChange:view=>{{activeView=view;updateThematicState();}},
      onThemeChange:theme=>{{activeView='physical';activeTheme=theme;updateThematicState();}},
      onSeasonChange:season=>{{activeSeason=season;updateThematicState();}},
      onLayerChange:()=>updateThematicState(),
      onZoom:action=>{{if(action==='reset')resetView();else zoomAtCenter(action==='in' ? 2 : .5);}},
      onRulerAction:action=>{{
        navigation.cancelPicking();
        if (action==='start') {{if (!ruler.active && ruler.state().stations.length) ruler.clear();ruler.start();}}
        else ruler[action]();
      }},
    }});
    function queryHasCamera() {{const url=new URLSearchParams(location.search);return ['zoom','x','y'].every(key=>url.has(key));}}
    const interaction=WorldAtlasInteraction.attach({{viewport,
      getCamera:()=>({{scale,translateX,translateY,minimumScale,maximumScale}}),
      setCamera:next=>{{scale=next.scale;translateX=next.translateX;translateY=next.translateY;cameraInteracted=true;constrainCamera();requestCameraTransform();}},
      zoomAt,zoomBy:zoomAtCenter,reset:resetView,
      onGestureStart:()=>{{cameraInteracted=true;dragging=true;window.clearTimeout(adaptiveUpdateTimer);}},
      onCommit:()=>{{dragging=false;applyTransform();scheduleAdaptiveUpdate(true);saveCamera();}},
      onTap:event=>{{
        if (navigation.picking) {{navigation.addPoint(screenToNative(event.clientX,event.clientY),event.target.closest('[data-settlement-id]')?.dataset.settlementId);}}
        else if (ruler.active) {{
          const point=screenToNative(event.clientX,event.clientY),last=ruler.state().stations.at(-1)?.native;
          if (!last || Math.hypot((last[0]-point[0])*atlasScaleX,(last[1]-point[1])*atlasScaleY)*scale>5) ruler.addNativePoint(point);
        }} else {{
          const city=event.target.closest('[data-settlement-id]');
          if(city)selectMapSettlement(city.dataset.settlementId);else ui.closePlace();
        }}
      }},
      onDoubleClick:()=>{{if(!ruler.active)return false;ruler.finish();return true;}},
      onEscape:()=>{{if(navigation.picking){{navigation.cancelPicking();return true;}}if(!ruler.active)return false;ruler.finish();return true;}},
    }});
    fitScene();restoreCamera();updateThematicState();applyTransform();
    ui.setRulerState(ruler.state());
    window.addEventListener('resize',()=>{{
      const oldWidth=Number(viewport.dataset.previousWidth)||viewport.clientWidth,
        oldHeight=Number(viewport.dataset.previousHeight)||viewport.clientHeight;
      const center=[(oldWidth/2-translateX)/scale,(oldHeight/2-translateY)/scale];
      const fit=Math.min(viewport.clientWidth/({viewbox_width}*atlasScaleX),viewport.clientHeight/({viewbox_height}*atlasScaleY));
      minimumScale=fit;
      if (!cameraInteracted) fitScene();else {{
        scale=Math.max(minimumScale,scale);translateX=viewport.clientWidth/2-center[0]*scale;translateY=viewport.clientHeight/2-center[1]*scale;constrainCamera();
      }}
      viewport.dataset.previousWidth=viewport.clientWidth;viewport.dataset.previousHeight=viewport.clientHeight;
      applyTransform();saveCamera();
    }});
    viewport.dataset.previousWidth=viewport.clientWidth;viewport.dataset.previousHeight=viewport.clientHeight;
  </script>
</body>
</html>
'''


def render_review(grid: WorldGrid, output_dir: str | Path, *, physical_source: ProceduralSurface,
                  society: SocietyLayers | None = None, travel_capabilities: tuple[str,...] = ()) -> dict[str, Any]:
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
    contours_path = output_dir / "elevation-contours.svgz"
    index_path = output_dir / "index.html"
    qa_path = output_dir / "qa.json"
    climate_path = output_dir / "climate.svg"
    biome_path = output_dir / "biome.svg"
    watersheds_path = output_dir / "watersheds.svg"
    land_potential_path = output_dir / "land-potential.svg"
    habitability_path = output_dir / "habitability.svg"
    vegetation_path = output_dir / "vegetation-cover.svg"
    population_path = output_dir / "population.png"
    population_vector_path = output_dir / "population.svg"
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

    logger = logging.getLogger(__name__)
    logger.info("Reconstructing the accepted physical ground")
    terrain_field = terrain_from_source(grid, physical_source)
    logger.info("Extracting the shared physical shoreline")
    land_surface = continuous_land_surface(grid, terrain_field=terrain_field)
    land_surface_paths = _geometry_filled_paths(land_surface)
    coast_paths = _surface_outline_paths(land_surface_paths, grid.shape)
    lake_fill_paths = _geometry_filled_paths(_lake_surface(grid, land_surface))
    lake_paths = _surface_outline_paths(lake_fill_paths, grid.shape)
    # Every water boundary is already a physical coast. Separate lake and
    # inland-sea outlines used to draw misaligned duplicates of that shore.
    inland_sea_paths: list[np.ndarray] = []
    logger.info("Extracting continuous physical relief")
    relief_bands, contour_paths, contour_path_levels = physical_relief_paths(
        terrain_field, _elevation_thresholds(grid),
        checkpoint_directory=output_dir.parent / "physical-contours",
        source_identity={"gridDigest": digest,
                         "rawElevationSha256": hashlib.sha256(
                             np.ascontiguousarray(physical_source.relative_elevation_m).tobytes()).hexdigest(),
                         "physicalDiagnostics": dict(physical_source.diagnostics)})
    # Base colours cover the common land clip. Repeating its half-million
    # vertices in both base bands adds no terrain information.
    frame_paths = _geometry_filled_paths(shapely.box(0, 0, grid.shape[1], grid.shape[0]))
    elevation_band_paths = [frame_paths] + [
        frame_paths if band is None else band for band in relief_bands]
    snow_fill_paths = _filled_mask_paths(grid.snow & (grid.water == 0))
    snow_paths = _surface_outline_paths(snow_fill_paths, grid.shape)
    polar_land_mask = polar_continent_mask(grid)
    polar_land_fill_paths = _filled_mask_paths(polar_land_mask)
    polar_land_paths = _surface_outline_paths(polar_land_fill_paths, grid.shape)
    sea_ice_fill_paths = _filled_mask_paths(grid.sea_ice)
    sea_ice_paths = _surface_outline_paths(sea_ice_fill_paths, grid.shape)
    # A channel's banks use native ground widths. Minimum overview ink is a
    # separate centreline, never a replacement for the river's actual surface.
    logger.info("Extracting the physical river network")
    river_source_paths = _river_paths(grid, coast_paths, raw_elevation_m=physical_source.relative_elevation_m)
    river_seasonal_strengths = _river_seasonal_strengths(grid, river_source_paths)
    river_orders = []
    for path in river_source_paths:
        columns = np.clip(np.floor(path[:, 0]).astype(int), 0, grid.shape[1] - 1)
        rows = np.clip(np.floor(path[:, 1]).astype(int), 0, grid.shape[0] - 1)
        river_orders.append(int(np.max(grid.river_order[rows, columns], initial=0)))
    river_path_attributes = [
        {
            "seasonal-strengths": ",".join(str(value) for value in strengths),
            "river-order": river_orders[index],
        }
        for index, strengths in enumerate(river_seasonal_strengths)
    ]
    river_paths = terrain_channel_paths(grid, river_source_paths, terrain_field=terrain_field)
    wind_overlay, wind_arrow_counts = _wind_arrow_groups(grid)
    graticule_paths, graticule_major_flags, graticule_labels = _graticule_paths(grid)
    elevation_palette, _ = _elevation_palette_for(grid)
    water_palette, _ = _bathymetry_palette_for(grid)
    # Ordinal bands leave small gaps at mixed categorical edges. A shallow
    # water base and complementary land/water clips keep those coastal gaps
    # in their correct surface family, using the same visible shoreline.
    # Trace the saved numeric depth surface before quantizing colours.
    # A class raster cannot retain where a shelf threshold crosses a cell.
    maritime = np.isin(grid.water, (1, 3))
    bathymetry_band_paths = scalar_band_paths(
        physical_source.bathymetry, maritime,
        np.arange(1, len(water_palette), dtype=float) / len(water_palette))
    inland_sea_fill_paths: list[tuple[np.ndarray, np.ndarray]] = []
    ecological_sources = FreshwaterCorridors.from_surfaces(
        grid.shape, river_paths, river_orders, _lake_surface(grid, land_surface))
    thematic = derive_thematic_layers(grid, ecological_sources=ecological_sources)
    tectonics = derive_tectonic_review(grid, physical_source=physical_source)
    tectonic_svg = render_tectonic_svg(grid, tectonics)
    name_source, society_request = _society_generation_request(grid)
    if society is None:
        society = derive_society_layers(grid, thematic, name_source,
                                        raw_elevation_m=physical_source.relative_elevation_m,
                                        **society_request)
    navigation_attributes = river_navigation_attributes(grid, river_source_paths, society.transport.routes)
    navigation_sections = river_navigation_segments(grid, river_source_paths, river_paths, society.transport.routes)
    river_path_attributes = [{**attrs, **navigation} for attrs, navigation
                             in zip(river_path_attributes, navigation_attributes, strict=True)]
    river_tile_features = []
    river_markup = _river_channel_markup(grid, river_source_paths, river_paths, river_path_attributes,
                                       tile_features=river_tile_features, navigation_segments=navigation_sections)
    river_channel_geometry = shapely.intersection(shapely.union_all([
        feature.geometry for feature in river_tile_features
        if feature.attributes.get("class") == "river-channel"
    ]), land_surface)
    (output_dir / "river-channels.svg").write_text(
        _line_overlay_svg_document(grid, river_markup, title="实际河道与最小阅读线"),
        encoding="utf-8",
    )
    (output_dir / "river-navigation.json").write_text(json.dumps({
        "schema": "physical-reach-navigation-v1",
        "routes": [{"identifier": route.identifier, "importance": route.importance,
                    "sourceSettlementId": route.source_settlement_id,
                    "targetSettlementId": route.target_settlement_id, "path": route.path}
                   for route in society.transport.routes if route.mode == "river"],
    }, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    capital_ids = {item.core_settlement_id for item in society.politics.states}
    territorial_qa = _territorial_quality_metrics(grid, thematic, society)
    climate_zones = thematic.climate.koppen_code
    land_mask = grid.water == 0
    climate_zone_paths = _categorical_partition_paths(
        climate_zones,
        land_mask,
        category_count=len(_CLIMATE_ZONES),
        land_surface=land_surface,
    )
    numeric_working_surface = shapely.union_all([
        geometry for band in climate_zone_paths
        for geometry in filled_geometries(band)
    ])
    biome_zone_paths = _categorical_partition_paths(
        thematic.biome_zone,
        land_mask,
        category_count=BIOME_COUNT,
        land_surface=land_surface,
    )
    watershed_zone_paths = _categorical_partition_paths(
        thematic.major_basin_rank,
        land_mask,
        category_count=len(_WATERSHED_ZONES),
        land_surface=land_surface,
    )
    potential_zone_paths = derive_land_potential_field(grid,thematic.climate,ecological_sources=ecological_sources).band_paths(
        LAND_POTENTIAL_THRESHOLDS,working_surface=_scalar_working_surface(numeric_working_surface))
    habitability_zone_paths = derive_habitability_field(grid,thematic.climate,ecological_sources=ecological_sources).band_paths(
        HABITABILITY_THRESHOLDS,working_surface=_scalar_working_surface(numeric_working_surface))
    vegetation = derive_vegetation_cover(grid,thematic.climate,ecological_sources=ecological_sources)
    vegetation_zone_paths = derive_vegetation_field(grid,thematic.climate,ecological_sources=ecological_sources).band_paths(
        VEGETATION_THRESHOLDS,working_surface=_scalar_working_surface(numeric_working_surface))
    population_band_count = int(
        np.unique(society.population.population_band[land_mask]).size
    )
    population_pixels = _categorical_overlay_pixels(
        society.population.population_band,
        land_mask,
        _POPULATION_ZONES,
        opacity=0.74,
    )
    density = population_density(grid, society.population)
    population_zone_paths = _population_zone_paths(
        density, land_mask, working_surface=numeric_working_surface)
    civilization_zones = _culture_zones(society)
    civilization_display = _civilization_display_values(grid, thematic, society)
    civilization_partition = _coastal_partition_topology(
        civilization_display,
        land_mask,
        category_count=len(civilization_zones),
        land_surface=land_surface,
    )
    civilization_faces, civilization_face_ids = civilization_partition.faces, civilization_partition.labels
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
    language_partition = _coastal_partition_topology(
        society.cultures.language_id,
        land_mask,
        category_count=len(language_zones),
        land_surface=land_surface,
    )
    language_faces, language_face_ids = language_partition.faces, language_partition.labels
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
        civilization_partition.visible_faces,
        civilization_partition.visible_labels,
    )
    language_boundary_paths = _shared_topology_boundary_paths(
        language_partition.visible_faces,
        language_partition.visible_labels,
    )
    religion_zones = _religion_zones(society)
    religion_display = _religion_display_values(society)
    religion_partition = _coastal_partition_topology(
        religion_display,
        land_mask,
        category_count=len(religion_zones),
        land_surface=land_surface,
    )
    religion_faces, religion_face_ids = religion_partition.faces, religion_partition.labels
    _validate_partition_inventory(
        religion_display,
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
        religion_partition.visible_faces,
        religion_partition.visible_labels,
    )
    political_zones = _political_zones(society)
    province_zones = _province_zones(society)
    from .society.administrations import administrative_source, administrative_paint_coverage
    administrative_front = administrative_source(grid, thematic, society)
    administrative_faces, province_face_ids = administrative_paint_coverage(
        administrative_front, land_mask)
    administrative_visible_faces, administrative_visible_ids = clip_partition_to_surface(
        administrative_faces, province_face_ids, land_surface)
    administrative_partition = CoastalPartition(administrative_faces, province_face_ids,
        administrative_visible_faces, administrative_visible_ids)
    administrative_faces, province_face_ids = administrative_partition.faces, administrative_partition.labels
    _governance, _governance_report, nominal_features = write_governance_overlay(output_dir, grid, society, administrative_partition.visible_faces,
                             administrative_partition.visible_labels)
    _validate_partition_inventory(
        society.provinces.province_id,
        land_mask,
        province_face_ids,
        layer_name="province",
    )
    province_to_state = np.zeros(len(province_zones), dtype=np.int32)
    for province in society.provinces.provinces:
        province_to_state[province.identifier] = province.state_identifier
    (
        _visible_state_face_ids,
        state_boundary_paths,
        province_boundary_paths,
    ) = _administrative_boundary_paths(
        administrative_partition.visible_faces,
        administrative_partition.visible_labels,
        province_to_state,
    )
    state_face_ids = province_to_state[province_face_ids]
    political_zone_paths = _shared_topology_zone_paths(
        administrative_faces,
        state_face_ids,
        category_count=len(political_zones),
        include_zero=True,
    )
    province_zone_paths = _shared_topology_zone_paths(
        administrative_faces,
        province_face_ids,
        category_count=len(province_zones),
        include_zero=True,
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
    # The ground carrier and main ink overlay are separate SVG assets. Count
    # their geometry in their own document budgets instead of combining them.
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
            [points for points, _ in lake_fill_paths],
            [points for points, _ in snow_fill_paths],
            [points for points, _ in sea_ice_fill_paths],
        ),
        path_layers=(
            coast_paths,
            lake_paths,
            inland_sea_paths,
            snow_paths,
            polar_land_paths,
            sea_ice_paths,
            river_paths,
            [points for points, _ in lake_fill_paths],
            [points for points, _ in snow_fill_paths],
            [points for points, _ in sea_ice_fill_paths],
        ),
    )
    # Complete physical fills and contours are offline exports and globe
    # composition inputs. The flat viewer only loads bounded viewport tiles;
    # its budgets belong to each tile, the overview and actual interaction.
    _validate_svg_budget((graticule_paths,))
    for partition_paths in (
        climate_zone_paths,
        biome_zone_paths,
        watershed_zone_paths,
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
    # Continuous numeric themes, like full relief, are offline geometry.
    # Their published bytes and each mounted tile retain strict limits;
    # a whole-world vertex ceiling would truncate valid native observations.
    shallow_ocean_ordinal = _shallow_ocean_ordinal(grid, len(water_palette))
    shallow_water_color = _rgb_hex(water_palette[shallow_ocean_ordinal])
    lowland_color = _rgb_hex(elevation_palette[0])
    climate_svg = _partition_overlay_svg_document(
        grid,
        climate_zone_paths,
        land_surface=land_surface,
        title="世界气候图",
        partition_id="climate-zones",
        data_attribute="climate-zone",
        zones=_CLIMATE_ZONES,
        fill_opacity=0.82,
    )
    biome_svg = _partition_overlay_svg_document(
        grid,
        biome_zone_paths,
        land_surface=land_surface,
        title="世界生物群系图",
        partition_id="biome-zones",
        data_attribute="biome-zone",
        zones=_BIOME_ZONES,
        fill_opacity=0.80,
    )
    watersheds_svg = _partition_overlay_svg_document(
        grid,
        watershed_zone_paths,
        land_surface=land_surface,
        title="世界水文流域图",
        partition_id="major-watersheds",
        data_attribute="basin-rank",
        zones=_WATERSHED_ZONES,
        fill_opacity=0.70,
    )
    land_potential_svg = _partition_overlay_svg_document(
        grid,
        potential_zone_paths,
        land_surface=land_surface,
        title="世界农业潜力图",
        partition_id="land-potential-bands",
        data_attribute="potential-band",
        zones=_LAND_POTENTIAL_ZONES,
        fill_opacity=0.88,
    )
    habitability_svg = _partition_overlay_svg_document(
        grid, habitability_zone_paths, land_surface=land_surface, title="世界宜居度图",
        partition_id="habitability-bands", data_attribute="habitability-band",
        zones=_HABITABILITY_ZONES, fill_opacity=0.88,
    )
    vegetation_svg = _partition_overlay_svg_document(
        grid, vegetation_zone_paths, land_surface=land_surface, title="世界植被覆盖图",
        partition_id="vegetation-bands", data_attribute="vegetation-band",
        zones=_VEGETATION_ZONES, fill_opacity=0.94,
    )
    civilizations_svg = _partition_overlay_svg_document(
        grid,
        civilization_zone_paths,
        land_surface=land_surface,
        title="世界文明区图",
        partition_id="civilization-regions",
        data_attribute="civilization",
        zones=civilization_zones,
        zone_outline_width=0.0,
        fill_opacity=0.68,
    )
    languages_svg = _partition_overlay_svg_document(
        grid,
        language_zone_paths,
        land_surface=land_surface,
        title="世界语言分布图",
        partition_id="language-regions",
        data_attribute="language",
        zones=language_zones,
        zone_outline_width=0.0,
        fill_opacity=0.64,
    )
    religions_svg = _partition_overlay_svg_document(
        grid,
        religion_zone_paths,
        land_surface=land_surface,
        title="世界宗教分布图",
        partition_id="religion-regions",
        data_attribute="religion",
        zones=religion_zones,
        zone_outline_width=0.0,
        fill_opacity=0.66,
    )
    political_svg = _partition_overlay_svg_document(
        grid,
        political_zone_paths,
        land_surface=land_surface,
        title="世界国家政区图层",
        partition_id="political-regions",
        data_attribute="state",
        zones=political_zones,
        extra_overlay=_frontier_overlay(frontier_fill_paths),
        zone_outline_width=0.0,
        fill_opacity=0.91,
    )
    provinces_svg = _partition_overlay_svg_document(
        grid,
        province_zone_paths,
        land_surface=land_surface,
        title="世界省份政区图层",
        partition_id="province-regions",
        data_attribute="province",
        zones=province_zones,
        extra_overlay=_frontier_overlay(frontier_fill_paths),
        zone_outline_width=0.0,
        fill_opacity=0.89,
    )
    population_svg = _partition_overlay_svg_document(
        grid, population_zone_paths, land_surface=land_surface, title="世界人口分布图",
        partition_id="population-regions", data_attribute="population-band",
        zones=_POPULATION_ZONES, fill_opacity=0.88,
    )
    contour_width_by_level = {
        level: 0.90 if (index + 1) % 4 == 0 else 0.45
        for index, level in enumerate(_elevation_thresholds(grid))
    }
    land_shape_data = " ".join(
        _filled_path_data(points, codes) for points, codes in land_surface_paths
    )
    contours_body = (
        '<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
        f'<path d="{land_shape_data}" clip-rule="evenodd" /></clipPath></defs>'
        '<g clip-path="url(#land-silhouette-clip)">'
    ) + _svg_group(
            "elevation-contour-lines",
            contour_paths,
            _CONTOUR_COLOR,
            stroke_width=0.45,
            stroke_widths=[contour_width_by_level[level] for level in contour_path_levels],
            path_levels=contour_path_levels,
        ) + '</g>'
    contours_svg = _line_overlay_svg_document(
        grid,
        contours_body,
        title="世界等高线",
    )
    thematic_documents = {
        contours_path.name: contours_svg,
        climate_path.name: climate_svg,
        biome_path.name: biome_svg,
        watersheds_path.name: watersheds_svg,
        land_potential_path.name: land_potential_svg,
        habitability_path.name: habitability_svg,
        vegetation_path.name: vegetation_svg,
        population_vector_path.name: population_svg,
        civilizations_path.name: civilizations_svg,
        languages_path.name: languages_svg,
        religions_path.name: religions_svg,
        political_path.name: political_svg,
        provinces_path.name: provinces_svg,
        tectonic_path.name: tectonic_svg,
    }
    contours_encoded = encode_svgz(contours_svg)
    export_sizes = {
        name: len(contours_encoded) if name == contours_path.name else len(document.encode("utf-8"))
        for name, document in thematic_documents.items()
    }
    oversized = {name: size for name, size in export_sizes.items() if size > _MAX_EXPORT_SVG_BYTES}
    if oversized:
        raise WorldGridRenderError(f"offline SVG export byte budget exceeded: {oversized}")
    water_shape_data = (
        f'M0,0 H{grid.shape[1]} V{grid.shape[0]} H0 Z ' + land_shape_data
    )
    land_silhouette_clip = (
        '<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
        + "".join(
            f'<path d="{_filled_path_data(points, codes)}" fill-rule="evenodd" clip-rule="evenodd" />'
            for points, codes in land_surface_paths
        )
        + '</clipPath><clipPath id="water-silhouette-clip" clipPathUnits="userSpaceOnUse">'
        + f'<path d="{water_shape_data}" clip-rule="evenodd" />'
        + '</clipPath></defs>'
    )
    surface_backfill = (
        land_silhouette_clip
        + '<g id="surface-backfill" data-layer="surface-backfill" '
        'aria-label="海陆不透底色">'
        f'<rect x="0" y="0" width="{grid.shape[1]}" height="{grid.shape[0]}" '
        f'fill="{shallow_water_color}" />'
        + f'<rect x="0" y="0" width="{grid.shape[1]}" height="{grid.shape[0]}" '
        f'fill="{lowland_color}" clip-path="url(#land-silhouette-clip)" />'
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
                ("potential", land_potential_path.name, "农业潜力叠加"),
                ("habitability", habitability_path.name, "宜居度叠加"),
                ("vegetation", vegetation_path.name, "植被覆盖叠加"),
                ("population", population_vector_path.name, "人口分布叠加"),
                ("civilizations", civilizations_path.name, "文明区叠加"),
                ("languages", languages_path.name, "语言区叠加"),
                ("religions", religions_path.name, "宗教分布叠加"),
                ("political", political_path.name, "国家政区叠加"),
                ("provinces", provinces_path.name, "省份政区叠加"),
            )
        )
        + '</g>'
    )
    city_locations = {
        settlement.identifier: (settlement.row + 0.5, settlement.column + 0.5)
        for settlement in society.settlements
    }
    from .city_harbors import derive_harbors, connect_harbor_routes
    harbors = derive_harbors(grid, society, city_locations, terrain_field,
        road_surface=land_surface.difference(river_channel_geometry),land_surface=land_surface)
    society=replace(society,transport=replace(society.transport,
        routes=connect_harbor_routes(society.transport.routes,harbors,grid,land_surface=land_surface)))
    logger.info("Resolving shared roads, river crossings and bridge facilities")
    display_river_geometry = shapely.MultiLineString(river_paths)
    transport_geometry = prepare_transport_geometry(
        grid, society.transport.routes, society.transport.bridges,
        locations=city_locations, land_surface=land_surface,
        river_source_geometry=shapely.MultiLineString(river_source_paths),
        river_geometry=display_river_geometry,
        river_channel_geometry=river_channel_geometry,
        raw_elevation_m=physical_source.relative_elevation_m,
        terrain_field=terrain_field,
    )
    from .transport_artifacts import write_transport_sources
    display_roads,transport_proof=write_transport_sources(output_dir,grid,transport_geometry,
        river_geometry=display_river_geometry,channel_geometry=river_channel_geometry)
    road_errors=transport_proof['checks']['bridgesOffRoad']
    river_errors=transport_proof['checks']['bridgesOffRiver']
    territorial_qa["sourceBridgeRasterMismatches"] = territorial_qa.pop("bridgesOffRoadOrRiver")
    territorial_qa["bridgesOffRoadOrRiver"] = road_errors + river_errors
    from .navigation import write_navigation_assets,travel_profile
    write_navigation_assets(output_dir,grid,society,display_roads,terrain_field=terrain_field,
        settlement_locations=city_locations,profile=travel_profile(grid.metadata['worldProfile']['technologyEra'],capabilities=travel_capabilities),
        sea_paths=[points for mode,importance,points in transport_geometry.paths if mode=='sea'],
        land_surface=land_surface,harbors=harbors)
    transport_tile_features = []
    transport_markup = _transport_overlay(
        transport_geometry, technology_era=str(grid.metadata["worldProfile"]["technologyEra"]),
        tile_features=transport_tile_features,
    )
    physical_surface_markup = "".join(
        (
            surface_backfill,
            '<g clip-path="url(#water-silhouette-clip)">'
            + _svg_filled_band_group(
                "bathymetry-bands",
                "海深色带",
                bathymetry_band_paths,
                water_palette,
                reverse_bands=True,
            )
            + '</g>',
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
        )
    )
    physical_surface_encoded = encode_svgz(
        _line_overlay_svg_document(grid, physical_surface_markup, title="完整展示地表导出")
    )
    if len(physical_surface_encoded) > _MAX_EXPORT_SVG_BYTES:
        raise WorldGridRenderError("offline physical SVG export byte budget exceeded")
    (output_dir / "physical-surface.svgz").write_bytes(physical_surface_encoded)
    # Every display level clips a complete numeric working coverage to its
    # own authoritative land surface. A fine-shore-clipped face cannot cover
    # a regional shore summary that expands beyond that fine shoreline.
    numeric_working_paths = {
        "potential": potential_zone_paths,
        "habitability": habitability_zone_paths,
        "vegetation": vegetation_zone_paths,
        "population": population_zone_paths,
    }
    # All detail comes from these spatial records; the browser never mounts
    # their complete world geometry. Overview and tiles share the same source.
    tile_features = [
        TileFeature(shapely.box(0, 0, grid.shape[1], grid.shape[0]),
                    {"fill": shallow_water_color, "stroke": "none"}, "surface-backfill", section="surface"),
        TileFeature(shapely.box(0, 0, grid.shape[1], grid.shape[0]),
                    {"fill": lowland_color, "stroke": "none", "clip": "land"},
                    "surface-backfill", section="surface"),
    ]
    for band in reversed(range(len(bathymetry_band_paths))):
        tile_features.extend(filled_features(bathymetry_band_paths[band], "bathymetry-bands",
                             _rgb_hex(water_palette[band]), clip="water"))
    for band, paths in enumerate(elevation_band_paths):
        tile_features.extend(filled_features(paths, "elevation-bands", _rgb_hex(elevation_palette[band]), clip="land"))
    tile_features.extend(filled_features(polar_land_fill_paths, "polar-land-surface", _POLAR_LAND_COLOR))
    tile_features.extend(filled_features(lake_fill_paths, "lakes", shallow_water_color))
    for theme, paths, zones, opacity, attribute in (
        ("climate", climate_zone_paths, _CLIMATE_ZONES, .82, "climate-zone"),
        ("biome", biome_zone_paths, _BIOME_ZONES, .80, "biome-zone"),
        ("watershed", watershed_zone_paths, _WATERSHED_ZONES, .70, "basin-rank"),
        ("potential", numeric_working_paths["potential"], _LAND_POTENTIAL_ZONES, .88, "potential-band"),
        ("habitability", numeric_working_paths["habitability"], _HABITABILITY_ZONES, .88, "habitability-band"),
        ("vegetation", numeric_working_paths["vegetation"], _VEGETATION_ZONES, .94, "vegetation-band"),
        ("population", numeric_working_paths["population"], _POPULATION_ZONES, .88, "population-band"),
        ("civilizations", civilization_zone_paths, civilization_zones, .68, "civilization"),
        ("languages", language_zone_paths, language_zones, .64, "language"),
        ("religions", religion_zone_paths, religion_zones, .66, "religion"),
        ("political", political_zone_paths, political_zones, .91, "state"),
        ("provinces", province_zone_paths, province_zones, .89, "province"),
    ):
        for identifier, ((label, color), faces) in enumerate(zip(zones, paths, strict=True)):
            tile_features.extend(filled_features(faces, "theme-fill", color, section="theme", theme=theme,
                                 opacity=opacity, clip="land", attributes={f"data-{attribute}": str(identifier),
                                                                          "aria-label": label}))
    landform_inventory = derive_landform_inventory(grid, thematic,
        raw_elevation_m=physical_source.relative_elevation_m)
    landform_tile_features = landform_features(grid, landform_inventory)
    tile_features.extend(landform_tile_features)
    landform_document = _line_overlay_svg_document(grid,
        '<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
        f'<path d="{land_shape_data}" clip-rule="evenodd" /></clipPath></defs>'
        + landform_svg_body(landform_tile_features),title="地貌范围与实际坡向")
    if len(landform_document.encode("utf-8")) > _MAX_EXPORT_SVG_BYTES:
        raise WorldGridRenderError("landform SVG export byte budget exceeded")
    (output_dir / "landform-regions.svg").write_text(landform_document,encoding="utf-8")
    (output_dir / "landform-display.json").write_text(json.dumps(landform_inventory.diagnostics,
        ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tile_features.extend(filled_features(snow_fill_paths, "snow", _SNOW_COLOR, section="ink", opacity=.44, clip="land"))
    tile_features.extend(filled_features(sea_ice_fill_paths, "sea-ice", _SEA_ICE_COLOR, section="ink", opacity=.88, clip="water"))
    tile_features.extend(line_features(coast_paths, "coast", _COAST_COLOR, 1.10, attributes={"data-base-stroke": "1.10"}))
    tile_features.extend(river_tile_features)
    for path, level in zip(contour_paths, contour_path_levels, strict=True):
        tile_features.extend(line_features([path], "elevation-contours", _CONTOUR_COLOR,
                             contour_width_by_level[level], opacity=.60, clip="land",
                             attributes={"data-base-stroke": str(contour_width_by_level[level]), "data-level": str(level)}))
    for layer, paths, color, stroke_width, opacity, dash in (
        ("civilization-boundaries", civilization_boundary_paths, "#4c5360", .82, .92, None),
        ("language-boundaries", language_boundary_paths, "#625f59", .95, .82, None),
        ("language-civilization-boundaries", civilization_boundary_paths, "#f8f3e8", .95, .92, None),
        ("religion-boundaries", religion_boundary_paths, "#6f5550", .95, .78, "3.2 2.3"),
        ("state-boundaries", state_boundary_paths, "#f4eee1", 2.80, .44, "7 3 1.3 3"),
        ("state-boundaries", state_boundary_paths, "#46413b", 1.40, .88, "7 3 1.3 3"),
        ("province-boundaries", province_boundary_paths, "#685f54", .95, .72, "4 2 1 2"),
    ):
        tile_features.extend(line_features(paths, layer, color, stroke_width, opacity=opacity, dash=dash, clip="land",
                             attributes={"data-screen-stroke": str(stroke_width),
                                         **({"data-screen-dash": dash} if dash else {})}))
    tile_features.extend(transport_tile_features)
    tile_features.extend(nominal_features)
    tile_levels = []
    for level_id, minimum_scale, tolerance in (("regional", 8, .08), ("local", 32, .025), ("detail", 64, 0)):
        level_land = generalize_display_surface(land_surface, frame_shape=grid.shape,
                                               tolerance=tolerance) if tolerance else land_surface
        level_coast = shapely.difference(level_land.boundary, shapely.box(0, 0, grid.shape[1], grid.shape[0]).boundary)
        features, coast_written = [], False
        for feature in tile_features:
            if feature.layer == "coast":
                if coast_written:
                    continue
                coast_written = True
                features.extend(replace(feature, geometry=part) for part in shapely.get_parts(level_coast)
                                if not part.is_empty)
                continue
            features.append(feature)
        tile_levels.append(TileLevel(level_id, minimum_scale, level_land, features))
    overview_land = tile_levels[0].land_surface
    overview_features = _administrative_overview_features(
        tile_features, administrative_faces, province_face_ids, province_to_state,
        frame_shape=grid.shape)
    overview_surface = overview_markup(overview_features, overview_land, grid.shape[1], grid.shape[0], section="surface")
    overview_ink = overview_markup(overview_features, overview_land, grid.shape[1], grid.shape[0], section="ink")
    # Theme assets are paint records assembled under the physical overview's
    # authoritative clips. Own that geometry once in the base, rather than
    # repeating definitions that the browser previously discarded on import.
    overview_bytes, oversized_overviews = {}, {}
    for theme in dict.fromkeys(feature.theme for feature in tile_features if feature.theme):
        body = overview_markup(overview_features, overview_land, grid.shape[1], grid.shape[0], section="theme", theme=theme)
        document = _line_overlay_svg_document(grid, body, title=f"{theme} 全图轮廓")
        overview_bytes[theme] = len(document.encode("utf-8"))
        logger.info("Overview %s: %s bytes", theme, overview_bytes[theme])
        if overview_bytes[theme] > _MAX_THEMATIC_SVG_BYTES:
            oversized_overviews[theme] = overview_bytes[theme]
        (output_dir / f"overview-{theme}.svg").write_text(document, encoding="utf-8")
    if oversized_overviews:
        raise WorldGridRenderError(f"overview SVG byte budget exceeded: {oversized_overviews}")
    if sum(overview_bytes.values()) > _MAX_THEMATIC_SVG_TOTAL_BYTES:
        raise WorldGridRenderError(f"combined overview SVG byte budget exceeded: {sum(overview_bytes.values())}")
    # Detailed scalar faces and physical channels share each block. Keep the
    # request footprint small without simplifying any delivered geometry.
    manifest = write_atlas_tiles(output_dir, grid.shape[1], grid.shape[0], tile_levels, tile_size=16)
    city_relief = derive_city_relief(grid, terrain_field, society.settlements,
        _elevation_thresholds(grid),
        settlement_locations=city_locations,tile_size=manifest['tileSize'])
    (output_dir / "city-contours.json").write_text(json.dumps({
        "schema": "accepted-physical-minor-curves-v1",
        "coordinateSpace": "native-cell",
        "gridDigest": digest,
        "curves": [{"heightM": float(feature.attributes["data-height-m"]),
                    "geometry": shapely.geometry.mapping(feature.geometry)}
                   for feature in city_relief.features],
    }, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = write_city_relief_tiles(output_dir, city_relief)
    thematic_overlay_images = re.sub(r'data-source="([^"]+)"',
        lambda match: 'data-source="overview-' + {
            "watersheds.svg": "watershed", "land-potential.svg": "potential",
            "vegetation-cover.svg": "vegetation",
        }.get(match[1], match[1].removesuffix(".svg")) + '.svg"', thematic_overlay_images)
    overlay = "".join((
        '<svg id="overview-source-svg" style="display:none" aria-hidden="true"><g id="overview-source">',
        '<g id="physical-surface">' + overview_surface + '</g>',
        thematic_overlay_images,
        overview_ink,
        '</g></svg><g id="overview-world" aria-label="世界概览"></g><g id="detail-world" aria-label="当前视窗地块"></g>',
        precipitation_overlay,
        _graticule_svg(grid, graticule_paths, graticule_major_flags, graticule_labels),
        _polar_reference_svg(grid),
        _bridge_overlay(transport_geometry),
        _geographic_symbol_overlay(grid, thematic, society, raw_elevation_m=physical_source.relative_elevation_m),
        _toponymy_overlay(grid, society),
        _polar_label_overlay(grid),
        _culture_label_overlay(society, kind="civilization"),
        _culture_label_overlay(society, kind="language"),
        _religion_label_overlay(society),
        _political_label_overlay(society),
        _province_label_overlay(society),
        _frontier_group_label_overlay(society),
        _city_overlay(society, city_locations),
        '<g id="seasonal-rivers" data-layer="seasonal-rivers" aria-label="季节河流动态样式"></g>',
        wind_overlay,
        '<image id="tectonic-map" class="tectonic-foundation-map" '
        f'data-source="{tectonic_path.name}" x="0" y="0" '
        f'width="{grid.shape[1]}" height="{grid.shape[0]}" '
        'preserveAspectRatio="none" aria-label="板块构造基础图" hidden />',
    ))
    html_kwargs = {
        "society": society,
        "tectonic_diagnostics": tectonics.diagnostics,
        "territorial_qa": territorial_qa,
        "width": display_width,
        "height": display_height,
        "viewbox_width": grid.shape[1],
        "viewbox_height": grid.shape[0],
    }
    write_interface_snapshot(
        output_dir, overlay, grid_digest=digest,
        society_digest=society_content_digest(society),
        tectonic_diagnostics=tectonics.diagnostics,
        territorial_qa=territorial_qa,
        width=display_width, height=display_height,
        viewbox_width=grid.shape[1], viewbox_height=grid.shape[0],
        settlement_locations=city_locations,
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
        temporary_population = _temporary_file(
            output_dir, population_path.name, ".png"
        )
        temporary_files.append(
            (temporary_population, _file_identity(temporary_population))
        )
        temporary_climate = _temporary_file(output_dir, climate_path.name, ".svg")
        temporary_files.append((temporary_climate, _file_identity(temporary_climate)))
        temporary_thematic: dict[Path, Path] = {climate_path: temporary_climate}
        for final_path in (
            contours_path,
            biome_path,
            watersheds_path,
            land_potential_path,
            habitability_path,
            vegetation_path,
            population_vector_path,
            civilizations_path,
            languages_path,
            religions_path,
            political_path,
            provinces_path,
            tectonic_path,
        ):
            temporary_path = _temporary_file(output_dir, final_path.name, final_path.suffix)
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

        Image.fromarray(_terrain_pixels(grid, terrain_field=terrain_field, bathymetry=physical_source.bathymetry)).save(
            temporary_terrain, format="PNG", optimize=True
        )
        Image.fromarray(population_pixels, mode="RGBA").save(
            temporary_population, format="PNG", optimize=True
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
            if final_path == contours_path:
                temporary_path.write_bytes(contours_encoded)
            else:
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
        # Physical water is valid at high latitude; only human activity is
        # constrained by the current society domain. Do not truncate rivers.
        polar_human_activity = {name: count for name, count in polar_activity.items()
                                if name not in {"lakeCells", "riverCells"}}
        if any(polar_human_activity.values()):
            raise WorldGridRenderError(
                "human activity outside society domain: "
                + ", ".join(
                    f"{name}={count}"
                    for name, count in polar_human_activity.items()
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
            "elevationContourLevelCount": len(set(contour_path_levels)),
            "terrainSource": {
                "units": "model-metres-above-sea-level",
                "sampling": "shared-native-ground-with-procedural-subgrid-landforms",
                "refinement": terrain_field.diagnostics,
                "datumMeters": terrain_field.sea_level_m,
                "elevationScaleMeters": terrain_field.elevation_scale_m,
                "elevationExponent": terrain_field.elevation_exponent,
                "nativeHeightRangeMeters": [float(terrain_field.native_m.min()),
                                            float(terrain_field.native_m.max())],
                "nativeSamples": int(terrain_field.native_m.size),
                "independentCoastNoise": False,
                "fixedCoastalGrade": False,
            },
            "viewportTiles": manifest,
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
                "habitability-bands": int(sum(bool(paths) for paths in habitability_zone_paths)),
                "vegetation-bands": int(sum(bool(paths) for paths in vegetation_zone_paths)),
                "population-bands": population_band_count,
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
                    for mode in ("road", "rail", "river", "sea")
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
                population_path.name: temporary_population.stat().st_size,
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
                (temporary_population, population_path),
                (temporary_society_npz, society_npz_path),
                (temporary_society_json, society_json_path),
                *((temporary_path, final_path) for final_path, temporary_path in temporary_thematic.items()),
            )
            + tuple(zip(temporary_precipitation, precipitation_paths, strict=True))
        )
    finally:
        for temporary, identity in temporary_files:
            _unlink_owned_file(temporary, identity)

    from .atlas_ui import write_map_app_assets
    write_map_app_assets(output_dir, grid=grid, society=society, settlement_locations=city_locations)
    from .city_map_assets import write_city_map_assets
    write_city_map_assets(output_dir, grid, society, city_locations,terrain_field=terrain_field,harbors=harbors,
                         physical_paths=transport_geometry.paths)
    _write_map_previews(output_dir, grid, thematic, society, vegetation.fraction, density)

    # Use the same authoritative vector surfaces as the flat atlas. Enlarging
    # the native-grid PNG would preserve the coarse coastal and border pixels.
    globe_surface = "".join((
        surface_backfill,
        '<g clip-path="url(#water-silhouette-clip)">'
        + _svg_filled_band_group("bathymetry-bands", "海深色带", bathymetry_band_paths,
                                 water_palette, reverse_bands=True) + '</g>',
        '<g clip-path="url(#land-silhouette-clip)">'
        + _svg_filled_band_group("elevation-bands", "高程色带", elevation_band_paths,
                                 elevation_palette) + '</g>',
        _svg_filled_surface_group("polar-land-surface", "极地灰白陆面", polar_land_fill_paths,
                                 _POLAR_LAND_COLOR, polar_land_paths, _POLAR_LAND_COLOR, 0.0, 1.0),
        _svg_filled_surface_group("lakes", "湖泊", lake_fill_paths, shallow_water_color,
                                 lake_paths, _COAST_COLOR, 0.35),
    ))
    globe_surface_cover = "".join((
        _svg_filled_surface_group("snow", "积雪", snow_fill_paths, _SNOW_COLOR,
                                 snow_paths, _SNOW_COLOR, 0.0, 0.44),
        _svg_filled_surface_group("sea-ice", "多年海冰", sea_ice_fill_paths, _SEA_ICE_COLOR,
                                 sea_ice_paths, _SEA_ICE_EDGE_COLOR, 0.35, 0.88),
    ))
    globe_lines = _svg_group("rivers", river_paths, _RIVER_COLOR, stroke_width=0.38)
    globe_lines += _svg_group("coast", coast_paths, _COAST_COLOR, stroke_width=0.35)
    globe_base_document = _line_overlay_svg_document(
        grid, globe_surface, title="World terrain surface",
    )
    globe_standalone_themes = dict((("climate", climate_svg),
                            ("biome", biome_svg), ("watershed", watersheds_svg),
                            ("potential", land_potential_svg), ("habitability", habitability_svg),
                            ("vegetation", vegetation_svg), ("population", population_svg),
                            ("civilizations", civilizations_svg), ("languages", languages_svg),
                            ("religions", religions_svg), ("political", political_svg),
                            ("provinces", provinces_svg)))
    globe_state_ink = _partition_boundary_overlay(
        "state-boundaries", state_boundary_paths, color="#46413b", width=0.40,
    )
    globe_textures = globe_theme_documents(
        globe_base_document, globe_standalone_themes,
        boundary_bodies={
            "political": globe_state_ink,
            "provinces": globe_state_ink + _partition_boundary_overlay(
                "province-boundaries", province_boundary_paths, color="#685f54",
                width=0.22, opacity=0.72,
            ),
        },
    )
    write_globe(output_dir,
        surface=globe_base_document,
        ink=_line_overlay_svg_document(grid, globe_surface_cover + globe_lines, title="World surface ink"),
        textures=globe_textures, grid_digest=digest,
        world_name=str(grid.metadata.get("worldProfile", {}).get("name", "")))

    return {
        "terrain": terrain_path,
        "elevationContours": contours_path,
        "index": index_path,
        "qa": qa_path,
        "climate": climate_path,
        "biome": biome_path,
        "watersheds": watersheds_path,
        "landPotential": land_potential_path,
        "habitability": habitability_path,
        "vegetation": vegetation_path,
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
