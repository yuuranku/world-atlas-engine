"""Derive and render the standalone tectonic foundation map.

The canonical world grid is the evidence source.  Plate interiors are grown
from continental, oceanic, and polar cores while relief belts, active-margin
trenches, and ocean-floor highs act as expensive barriers.  A plate core is a
seed tendency, not a crust mask: one plate may contain continental crust and
the adjacent ocean floor, so passive continental margins remain inside a
plate instead of being mistaken for plate boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import html
import hashlib
import io
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from PIL import Image

from world_atlas.tectonic_foundation import derive_tectonic_foundation, foundation_from_causal_fields

from .model import WorldGrid
from .procedural_planet import load_surface_bundle
from .society.world_identity import NameRegistry


_PLATE_COLORS = (
    "#d5a96f", "#78aeb8", "#b79bc2", "#9fbe79", "#d58f86",
    "#86b893", "#d5c67c", "#8ca7cf", "#c79f82", "#8fb9ae",
    "#c4a8cf", "#b1bd76",
)
_BOUNDARY_ZH = {
    "convergent": "碰撞／俯冲边界",
    "divergent": "分离／扩张边界",
    "transform": "转换断层",
}


@dataclass(frozen=True, slots=True)
class TectonicPlate:
    identifier: int
    name: str
    kind: str
    row: int
    column: int
    area_cells: int
    velocity_east_cm_per_year: float
    velocity_north_cm_per_year: float
    color: str


@dataclass(frozen=True, slots=True)
class TectonicBoundary:
    plate_ids: tuple[int, int]
    classification: str
    paths: tuple[tuple[tuple[float, float], ...], ...]
    evidence: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class TectonicReview:
    plate_id: np.ndarray
    plates: tuple[TectonicPlate, ...]
    boundaries: tuple[TectonicBoundary, ...]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        labels = np.array(self.plate_id, dtype=np.int16, copy=True)
        labels.setflags(write=False)
        object.__setattr__(self, "plate_id", labels)
        object.__setattr__(self, "plates", tuple(self.plates))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def _normalise(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    sample = values if mask is None else values[mask]
    if sample.size == 0:
        return np.zeros(values.shape, dtype=np.float64)
    low, high = np.quantile(sample.astype(np.float64), (0.08, 0.94))
    if high <= low + 1.0e-12:
        return np.zeros(values.shape, dtype=np.float64)
    return np.clip((values.astype(np.float64) - low) / (high - low), 0.0, 1.0)


def _pair_normal(
    first: tuple[int, int, str],
    second: tuple[int, int, str],
    width: int,
) -> tuple[float, float]:
    delta_x = float(second[1] - first[1])
    if delta_x > width / 2:
        delta_x -= width
    elif delta_x < -width / 2:
        delta_x += width
    delta_y = float(second[0] - first[0])
    length = max(1.0e-9, math.hypot(delta_x, delta_y))
    return delta_x / length, delta_y / length


def derive_tectonic_review(grid: WorldGrid) -> TectonicReview:
    """Render the same geography-first tectonic model used by island generation."""

    if not isinstance(grid, WorldGrid):
        raise TypeError("derive_tectonic_review requires a WorldGrid")
    source_metadata = grid.metadata.get("proceduralPhysicalSource")
    requested_plate_count = (
        source_metadata.get("plateCount")
        if isinstance(source_metadata, Mapping)
        else None
    )
    bundle = source_metadata.get("fieldBundle") if isinstance(source_metadata, Mapping) else None
    if bundle is not None:
        source = Path(bundle["path"])
        if hashlib.sha256(source.read_bytes()).hexdigest().upper() != bundle["sha256"].upper():
            raise ValueError("tectonic source bundle hash mismatch")
        surface = load_surface_bundle(source)
        if surface.land_mask.shape != grid.shape or not np.array_equal(surface.land_mask, grid.water == 0):
            raise ValueError("tectonic source does not match the accepted land mask")
        foundation = foundation_from_causal_fields(
            surface.plate_id, surface.boundary_class, surface.land_mask,
            surface.elevation, surface.bathymetry,
            surface.plate_velocity_east_cm_per_year, surface.plate_velocity_north_cm_per_year,
        )
    else:
        foundation = derive_tectonic_foundation(
            grid.water == 0, grid.elevation, np.maximum(grid.bathymetry_band, 0),
            plate_count=requested_plate_count,
        )
    request = grid.metadata.get("societyGeneration", {})
    registry = NameRegistry(
        request.get("namingSeed", 0), request.get("forbiddenNames", ())
    )
    plates: list[TectonicPlate] = []
    for plate in foundation.plates:
        name = registry.name(f"plate:{plate.identifier}", plate.identifier % 12, suffix="板块")
        plates.append(
            TectonicPlate(
                plate.identifier,
                name,
                plate.kind,
                plate.row,
                plate.column,
                plate.area_cells,
                plate.velocity_east_cm_per_year,
                plate.velocity_north_cm_per_year,
                _PLATE_COLORS[plate.identifier % len(_PLATE_COLORS)],
            )
        )
    boundaries = tuple(
        TectonicBoundary(
            boundary.plate_ids,
            boundary.classification,
            boundary.paths,
            boundary.evidence,
        )
        for boundary in foundation.boundaries
    )
    diagnostics = dict(foundation.diagnostics)
    diagnostics["boundaryPairCount"] = len(
        {boundary.plate_ids for boundary in foundation.boundaries}
    )
    if bundle is None and foundation.plate_id.size >= 10_000:
        if float(diagnostics["coastBoundaryFraction"]) > 0.60:
            raise ValueError("tectonic QA failed: plate boundaries track coastlines too closely")
        if float(diagnostics["convergentEvidenceFraction"]) < 0.90:
            raise ValueError("tectonic QA failed: convergent boundaries lack orogenic or trench evidence")
        if float(diagnostics["divergentOceanSupportFraction"]) < 0.82:
            raise ValueError("tectonic QA failed: divergent boundaries are not predominantly oceanic")
    return TectonicReview(
        foundation.plate_id,
        tuple(plates),
        boundaries,
        diagnostics,
    )


def _hex_rgb(value: str) -> np.ndarray:
    return np.asarray([int(value[index:index + 2], 16) for index in (1, 3, 5)], dtype=np.float64)


def _base_data_url(grid: WorldGrid, tectonics: TectonicReview) -> str:
    plate_colors = np.asarray([_hex_rgb(plate.color) for plate in tectonics.plates])
    expanded = plate_colors[tectonics.plate_id]
    land = grid.water == 0
    elevation = _normalise(grid.elevation, land)
    bathymetry = np.clip(np.maximum(grid.bathymetry_band, 0) / 7.0, 0.0, 1.0)
    physical = np.zeros_like(expanded)
    land_tone = 235.0 - elevation[..., None] * np.asarray((96.0, 78.0, 45.0))
    physical[land] = land_tone[land]
    ocean_rgb = np.stack((206.0 - 30.0 * bathymetry, 221.0 - 30.0 * bathymetry, 226.0 - 22.0 * bathymetry), axis=-1)
    physical[~land] = ocean_rgb[~land]
    pixels = np.clip(expanded * 0.56 + physical * 0.44, 0.0, 255.0)
    coast = land != np.roll(land, 1, axis=1)
    coast |= land != np.roll(land, -1, axis=1)
    coast[1:] |= land[1:] != land[:-1]
    pixels[coast] = pixels[coast] * 0.40 + np.asarray((47.0, 65.0, 73.0)) * 0.60
    image = Image.fromarray(pixels.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _path_data(points: tuple[tuple[float, float], ...]) -> str:
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _samples_along_path(
    points: tuple[tuple[float, float], ...],
    *,
    spacing: float,
) -> tuple[tuple[float, float, float, float], ...]:
    segments: list[tuple[float, float, float, float, float]] = []
    total = 0.0
    for left, right in zip(points, points[1:]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        length = math.hypot(dx, dy)
        if length <= 1.0e-9:
            continue
        segments.append((total, length, left[0], left[1], math.atan2(dy, dx)))
        total += length
    if total < spacing * 0.72:
        return ()
    targets = np.arange(spacing * 0.58, total, spacing)
    result: list[tuple[float, float, float, float]] = []
    segment_index = 0
    for target in targets:
        while segment_index + 1 < len(segments) and target >= segments[segment_index][0] + segments[segment_index][1]:
            segment_index += 1
        start, length, x, y, angle = segments[segment_index]
        fraction = np.clip((target - start) / length, 0.0, 1.0)
        result.append((
            float(x + math.cos(angle) * length * fraction),
            float(y + math.sin(angle) * length * fraction),
            math.cos(angle),
            math.sin(angle),
        ))
    return tuple(result)


def _arrow(
    start_x: float,
    start_y: float,
    direction_x: float,
    direction_y: float,
    length: float,
    css_class: str,
) -> str:
    norm = max(1.0e-9, math.hypot(direction_x, direction_y))
    end_x = start_x + direction_x / norm * length
    end_y = start_y + direction_y / norm * length
    return (
        f'<path class="{css_class}" d="M{start_x:.2f},{start_y:.2f} '
        f'L{end_x:.2f},{end_y:.2f}" marker-end="url(#motion-arrow)" />'
    )


def render_tectonic_svg(grid: WorldGrid, tectonics: TectonicReview) -> str:
    """Return one self-contained SVG foundation map."""

    if not isinstance(grid, WorldGrid) or not isinstance(tectonics, TectonicReview):
        raise TypeError("render_tectonic_svg requires WorldGrid and TectonicReview")
    height, width = tectonics.plate_id.shape
    symbol_scale = max(1.0, width / 544.0, height / 272.0)
    base = _base_data_url(grid, tectonics)
    lines: list[str] = []
    relative_arrows: list[str] = []
    teeth: list[str] = []
    for boundary in tectonics.boundaries:
        css_class = f"boundary {boundary.classification}"
        lines.extend(f'<path class="{css_class}" d="{_path_data(path)}" />' for path in boundary.paths)
        longest = max(boundary.paths, key=len)
        middle = longest[len(longest) // 2]
        first, second = boundary.plate_ids
        nx, ny = _pair_normal(
            (tectonics.plates[first].row, tectonics.plates[first].column, tectonics.plates[first].kind),
            (tectonics.plates[second].row, tectonics.plates[second].column, tectonics.plates[second].kind),
            width,
        )
        offset = 3.2 * symbol_scale
        length = 5.4 * symbol_scale
        if boundary.classification == "convergent":
            relative_arrows.append(_arrow(middle[0] - nx * offset, middle[1] - ny * offset, nx, ny, length, "relative convergent"))
            relative_arrows.append(_arrow(middle[0] + nx * offset, middle[1] + ny * offset, -nx, -ny, length, "relative convergent"))
            first_plate = tectonics.plates[first]
            second_plate = tectonics.plates[second]
            if first_plate.kind == "continental" and second_plate.kind != "continental":
                overriding = first_plate
            elif second_plate.kind == "continental" and first_plate.kind != "continental":
                overriding = second_plate
            else:
                overriding = max((first_plate, second_plate), key=lambda plate: plate.area_cells)
            for path in boundary.paths:
                for x, y, tx, ty in _samples_along_path(
                    path,
                    spacing=8.6 * symbol_scale,
                ):
                    px, py = -ty, tx
                    seed_dx = float(overriding.column - x)
                    if seed_dx > width / 2:
                        seed_dx -= width
                    elif seed_dx < -width / 2:
                        seed_dx += width
                    seed_dy = float(overriding.row - y)
                    if px * seed_dx + py * seed_dy < 0.0:
                        px, py = -px, -py
                    teeth.append(
                        f'<path class="tooth" d="M{x - tx * 1.25 * symbol_scale:.2f},{y - ty * 1.25 * symbol_scale:.2f} '
                        f'L{x + px * 2.25 * symbol_scale:.2f},{y + py * 2.25 * symbol_scale:.2f} '
                        f'L{x + tx * 1.25 * symbol_scale:.2f},{y + ty * 1.25 * symbol_scale:.2f} Z" />'
                    )
        elif boundary.classification == "divergent":
            relative_arrows.append(_arrow(middle[0] - nx * offset, middle[1] - ny * offset, -nx, -ny, length, "relative divergent"))
            relative_arrows.append(_arrow(middle[0] + nx * offset, middle[1] + ny * offset, nx, ny, length, "relative divergent"))
        else:
            tx, ty = -ny, nx
            relative_arrows.append(_arrow(middle[0] - nx * offset, middle[1] - ny * offset, tx, ty, length, "relative transform"))
            relative_arrows.append(_arrow(middle[0] + nx * offset, middle[1] + ny * offset, -tx, -ty, length, "relative transform"))

    plate_labels: list[str] = []
    plate_arrows: list[str] = []
    for plate in tectonics.plates:
        label_x = float(
            np.clip(
                plate.column + 0.5,
                24.0 * symbol_scale,
                width - 24.0 * symbol_scale,
            )
        )
        label_y = float(
            np.clip(
                plate.row + 0.5,
                12.0 * symbol_scale,
                height - 11.0 * symbol_scale,
            )
        )
        speed = math.hypot(plate.velocity_east_cm_per_year, plate.velocity_north_cm_per_year)
        arrow_length = symbol_scale * (7.0 + min(10.0, speed * 2.1))
        plate_arrows.append(_arrow(
            label_x,
            label_y + 4.0 * symbol_scale,
            plate.velocity_east_cm_per_year,
            -plate.velocity_north_cm_per_year,
            arrow_length,
            "plate-motion",
        ))
        kind_zh = {
            "continental": "含大陆地壳",
            "oceanic": "以海洋地壳为主",
            "polar": "含极地大陆地壳",
        }[plate.kind]
        plate_labels.append(
            f'<g class="plate-label" transform="translate({label_x:.2f} {label_y:.2f})">'
            f'<text class="plate-name" text-anchor="middle">{html.escape(plate.name)}</text>'
            f'<text class="plate-meta" y="{3.8 * symbol_scale:.2f}" text-anchor="middle">{kind_zh} · {speed:.1f} cm/年</text></g>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{grid.shape[1]}" height="{grid.shape[0]}" viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" aria-label="世界板块构造图">
  <title>世界板块构造基础图</title>
  <defs>
    <marker id="motion-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="{3.2 * symbol_scale:.2f}" markerHeight="{3.2 * symbol_scale:.2f}" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 Z" fill="context-stroke" /></marker>
    <filter id="label-halo" x="-30%" y="-40%" width="160%" height="180%"><feMorphology in="SourceAlpha" operator="dilate" radius="{0.75 * symbol_scale:.2f}" result="dilate"/><feFlood flood-color="#f6f2e7" flood-opacity="0.92"/><feComposite in2="dilate" operator="in"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <style>
    .boundary{{fill:none;stroke-linecap:round;stroke-linejoin:round}}
    .boundary.convergent{{stroke:#405d5d;stroke-width:{1.6 * symbol_scale:.2f}}}
    .boundary.divergent{{stroke:#b94e43;stroke-width:{1.35 * symbol_scale:.2f}}}
    .boundary.transform{{stroke:#bd7a35;stroke-width:{1.25 * symbol_scale:.2f};stroke-dasharray:{3.2 * symbol_scale:.2f} {2.1 * symbol_scale:.2f}}}
    .tooth{{fill:#405d5d;stroke:none}}
    .relative{{fill:none;stroke-width:{0.95 * symbol_scale:.2f};stroke-linecap:round;opacity:.94}}
    .relative.convergent{{stroke:#405d5d}} .relative.divergent{{stroke:#b94e43}} .relative.transform{{stroke:#9b642f}}
    .plate-motion{{fill:none;stroke:#263e59;stroke-width:{1.25 * symbol_scale:.2f};stroke-linecap:round}}
    .plate-name{{font:700 {5.8 * symbol_scale:.2f}px "Microsoft YaHei",sans-serif;fill:#263447;filter:url(#label-halo)}}
    .plate-meta{{font:600 {2.5 * symbol_scale:.2f}px "Microsoft YaHei",sans-serif;fill:#405164;filter:url(#label-halo)}}
    .legend-title{{font:700 {4.0 * symbol_scale:.2f}px "Microsoft YaHei",sans-serif;fill:#263447}}
    .legend-text{{font:600 {2.7 * symbol_scale:.2f}px "Microsoft YaHei",sans-serif;fill:#35465b}}
  </style>
  <image id="tectonic-plates" data-layer="tectonic-plates" href="{base}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="none" />
  <g id="tectonic-boundaries" data-layer="tectonic-boundaries">{''.join(lines)}{''.join(teeth)}</g>
  <g id="tectonic-relative-motion" data-layer="tectonic-relative-motion">{''.join(relative_arrows)}</g>
  <g id="tectonic-plate-motion" data-layer="tectonic-plate-motion">{''.join(plate_arrows)}</g>
  <g id="tectonic-plate-labels" data-layer="tectonic-plate-labels">{''.join(plate_labels)}</g>
  <g id="tectonic-legend" transform="translate({6 * symbol_scale:.2f} {7 * symbol_scale:.2f})">
    <rect x="{-2.5 * symbol_scale:.2f}" y="{-4.5 * symbol_scale:.2f}" width="{75 * symbol_scale:.2f}" height="{20.5 * symbol_scale:.2f}" rx="{1.5 * symbol_scale:.2f}" fill="#f7f3e9" fill-opacity=".91" stroke="#66747e" stroke-width="{0.35 * symbol_scale:.2f}" />
    <text class="legend-title" x="0" y="0">板块边界与相对运动</text>
    <path class="boundary convergent" d="M0,{5 * symbol_scale:.2f} L{13 * symbol_scale:.2f},{5 * symbol_scale:.2f}"/><path class="tooth" d="M{4 * symbol_scale:.2f},{5 * symbol_scale:.2f} L{5.5 * symbol_scale:.2f},{7 * symbol_scale:.2f} L{7 * symbol_scale:.2f},{5 * symbol_scale:.2f} Z"/><text class="legend-text" x="{16 * symbol_scale:.2f}" y="{6 * symbol_scale:.2f}">{_BOUNDARY_ZH['convergent']}</text>
    <path class="boundary divergent" d="M0,{10 * symbol_scale:.2f} L{13 * symbol_scale:.2f},{10 * symbol_scale:.2f}"/><text class="legend-text" x="{16 * symbol_scale:.2f}" y="{11 * symbol_scale:.2f}">{_BOUNDARY_ZH['divergent']}</text>
    <path class="boundary transform" d="M{38 * symbol_scale:.2f},{10 * symbol_scale:.2f} L{51 * symbol_scale:.2f},{10 * symbol_scale:.2f}"/><text class="legend-text" x="{54 * symbol_scale:.2f}" y="{11 * symbol_scale:.2f}">{_BOUNDARY_ZH['transform']}</text>
  </g>
</svg>
'''


__all__ = [
    "TectonicBoundary",
    "TectonicPlate",
    "TectonicReview",
    "derive_tectonic_review",
    "render_tectonic_svg",
]
