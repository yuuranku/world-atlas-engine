"""Refresh administrative drawing without repeating a completed atlas render."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import shapely

from .. import __version__
from ..timing import measure_stage
from .cartographic_features import overview_markup
from .cartographic_generalization import generalize_display_surface
from .cartographic_surface import continuous_land_surface
from .cartographic_tiles import refresh_atlas_tile_themes
from .coastal_partition import clip_partition_to_surface
from .ecological_sources import derive_ecological_sources
from .globe_assets import THEME_EXPORT_FILENAMES, globe_theme_documents
from .governance_render import write_governance_overlay
from .presentation import society_content_digest
from .review_interface import load_interface_snapshot, write_interface_snapshot
from .society.administrations import administrative_paint_coverage, administrative_source
from .society.administrative_display import administrative_display_coverage
from .svg_groups import extract_group, replace_group
from .terrain_refinement import terrain_from_source
from .thematic import derive_thematic_layers


_INK_LAYERS = ("state-boundaries", "province-boundaries", "nominal-realms")


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _same_administrations(first, second):
    for layer, field in (("politics", "state_id"), ("provinces", "province_id")):
        if not np.array_equal(getattr(getattr(first, layer), field),
                              getattr(getattr(second, layer), field)):
            raise ValueError("administrative drawing refresh cannot change native ownership")
    first_parents = [(p.identifier, p.state_identifier) for p in first.provinces.provinces]
    second_parents = [(p.identifier, p.state_identifier) for p in second.provinces.provinces]
    if (first_parents != second_parents
            or [s.identifier for s in first.politics.states]
            != [s.identifier for s in second.politics.states]):
        raise ValueError("administrative drawing refresh cannot change the saved hierarchy")


def refresh_review_administrations(grid, source_dir, target_dir, *, administrative_society,
                                   rendered_society, physical_source):
    """Publish one new review with only its shared administrative drawing rebuilt.

    The administrative society supplies the recorded simulation seats and routes;
    the rendered society supplies the unchanged scene contract and delivered city
    locations. Neither society nor any canonical array is written or mutated.
    """
    from . import render

    source_dir, target_dir = Path(source_dir).resolve(), Path(target_dir).resolve()
    if target_dir.exists():
        raise FileExistsError("administrative review target must be a new directory")
    if target_dir.is_relative_to(source_dir) or source_dir.is_relative_to(target_dir):
        raise ValueError("administrative review source and target must not nest")
    _same_administrations(administrative_society, rendered_society)
    grid_digest = grid.content_digest()
    rendered_digest = society_content_digest(rendered_society)
    overlay, presentation = load_interface_snapshot(source_dir,
        grid_digest=grid_digest, society_digest=rendered_digest)
    if (presentation["viewboxWidth"], presentation["viewboxHeight"]) != (grid.shape[1], grid.shape[0]):
        raise ValueError("administrative review dimensions differ from the saved physical world")
    qa = json.loads((source_dir / "qa.json").read_text(encoding="utf-8"))
    if qa["gridDigest"] != grid_digest:
        raise ValueError("administrative review QA belongs to a different physical world")
    record = {
        "schema": "world-atlas-administrative-review-v1", "status": "building",
        "engineVersion": __version__, "sourceReview": str(source_dir),
        "sourceSceneSha256": presentation["sceneSha256"],
        "sourceTileManifestSha256": _sha256(source_dir / "atlas-manifest.json"),
        "sourcePhysicalSurfaceSha256": _sha256(source_dir / "physical-surface.svgz"),
        "physicalElevationSha256": hashlib.sha256(
            np.ascontiguousarray(physical_source.relative_elevation_m).tobytes()).hexdigest(),
        "gridDigest": grid_digest, "renderedSocietyDigest": rendered_digest,
        "administrativeSocietyDigest": society_content_digest(administrative_society),
        "nativeOwnershipChanged": False,
        "codeSha256": {module: _sha256(Path(__file__).parent / filename)
            for module, filename in (("administrative-review", "administrative_review.py"),
                ("administrative-source", "society/administrations.py"),
                ("administrative-display", "society/administrative_display.py"),
                ("render", "render.py"))},
    }
    target_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        with measure_stage(target_dir, "administrative-review"):
            with measure_stage(target_dir, "saved-review-copy"):
                shutil.copytree(source_dir, target_dir, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("tiles", "timing.json", "timing.lock"))
            _write_json(target_dir / "administrative-refresh.json", record)
            with measure_stage(target_dir, "administrative-inputs"):
                terrain = terrain_from_source(grid, physical_source)
                land_surface = continuous_land_surface(grid, terrain_field=terrain)
                thematic = derive_thematic_layers(grid,
                    ecological_sources=derive_ecological_sources(grid, physical_source))
            with measure_stage(target_dir, "administrative-geometry"):
                front = administrative_source(grid, thematic, administrative_society)
                land = grid.water == 0
                for values, owners in ((administrative_society.politics.state_id, front.countries.owner),
                                       (administrative_society.provinces.province_id, front.provinces.owner)):
                    if not np.array_equal(values, np.where(land, np.maximum(owners, 0), -1)):
                        raise ValueError("administrative source no longer reproduces the saved native ownership")
                raw_faces, province_ids = administrative_paint_coverage(front, land)
                faces = administrative_display_coverage(raw_faces, frame_shape=grid.shape)
                visible_faces, visible_ids = clip_partition_to_surface(faces, province_ids, land_surface)
                political_zones = render._political_zones(rendered_society)
                province_zones = render._province_zones(rendered_society)
                parents = np.zeros(len(province_zones), dtype=np.int32)
                for province in rendered_society.provinces.provinces:
                    parents[province.identifier] = province.state_identifier
                encoded_faces = shapely.to_wkb(faces)
                geometry_bytes = b"".join(encoded_faces)
                lengths = np.asarray([len(item) for item in encoded_faces], dtype=np.int64)
                np.savez_compressed(target_dir / "administrative-display.npz",
                    wkb=np.frombuffer(geometry_bytes, dtype=np.uint8), offsets=np.r_[0, np.cumsum(lengths)],
                    province_ids=province_ids, province_to_state=parents)
                _, state_paths, province_paths = render._administrative_boundary_paths(
                    visible_faces, visible_ids, parents)
                state_ids = parents[province_ids]
                political_paths = render._shared_topology_zone_paths(faces, state_ids,
                    category_count=len(political_zones), include_zero=True)
                province_zone_paths = render._shared_topology_zone_paths(faces, province_ids,
                    category_count=len(province_zones), include_zero=True)
                frontier_paths = render._geometry_filled_paths(shapely.union_all(
                    [face for face, owner in zip(faces, state_ids, strict=True) if owner == 0]))
                _, governance_check, nominal_features = write_governance_overlay(
                    target_dir, grid, rendered_society, visible_faces, visible_ids)
                theme_features = render._administrative_theme_features(political_paths,
                    province_zone_paths, political_zones, province_zones)
                ink_features = [*render._administrative_ink_features(state_paths, province_paths),
                                *nominal_features]
                record["geometry"] = {
                    "provinceFaces": len(faces), "visibleProvinceFaces": len(visible_faces),
                    "sourceCoverageSha256": hashlib.sha256(b"".join(shapely.to_wkb(raw_faces))).hexdigest(),
                    "coverageSha256": hashlib.sha256(geometry_bytes).hexdigest(),
                    "paintedGeometryChangedCount": int(np.count_nonzero(~shapely.equals(raw_faces, faces))),
                    "stateBoundaryPaths": len(state_paths), "provinceBoundaryPaths": len(province_paths),
                    "governanceCheck": governance_check,
                }
            with measure_stage(target_dir, "administrative-exports"):
                for theme, paths, zones, partition, attribute, title, opacity in (
                    ("political", political_paths, political_zones, "political-regions", "state",
                     "世界国家政区图层", .91),
                    ("provinces", province_zone_paths, province_zones, "province-regions", "province",
                     "世界省份政区图层", .89),
                ):
                    document = render._partition_overlay_svg_document(grid, paths,
                        land_surface=land_surface, title=title, partition_id=partition,
                        data_attribute=attribute, zones=zones,
                        extra_overlay=render._frontier_overlay(frontier_paths), fill_opacity=opacity)
                    if len(document.encode("utf-8")) > render._MAX_EXPORT_SVG_BYTES:
                        raise ValueError(f"administrative SVG export byte budget exceeded: {theme}")
                    (target_dir / f"{theme}.svg").write_text(document, encoding="utf-8")
                overview_land = generalize_display_surface(land_surface,
                    frame_shape=grid.shape, tolerance=.08)
                overview_features = render._administrative_overview_features(
                    [*theme_features, *ink_features], faces, province_ids, parents)
                for theme in ("political", "provinces"):
                    body = overview_markup(overview_features, overview_land,
                        grid.shape[1], grid.shape[0], section="theme", theme=theme)
                    document = render._line_overlay_svg_document(grid, body, title=f"{theme} 全图轮廓")
                    if len(document.encode("utf-8")) > render._MAX_EXPORT_SVG_BYTES:
                        raise ValueError(f"administrative overview byte budget exceeded: {theme}")
                    (target_dir / f"overview-{theme}.svg").write_text(document, encoding="utf-8")
                overview_bytes = sum(path.stat().st_size for path in target_dir.glob("overview-*.svg")
                                     if path.name != "overview-source.svg")
                if overview_bytes > render._MAX_THEMATIC_SVG_TOTAL_BYTES:
                    raise ValueError(f"combined overview SVG byte budget exceeded: {overview_bytes}")
                overview_group = extract_group(overlay, "id", "overview-source")
                opening_end = overview_group.index(">") + 1
                body = overview_group[opening_end:-len("</g>")]
                for layer in _INK_LAYERS:
                    ink = overview_markup([f for f in overview_features if f.layer == layer],
                        overview_land, grid.shape[1], grid.shape[0], section="ink")
                    body = replace_group(body, "data-tile-layer", layer, ink,
                        before="transport-network")
                overview_group = overview_group[:opening_end] + body + "</g>"
                overlay = replace_group(overlay, "id", "overview-source", overview_group, required=True)
                write_interface_snapshot(target_dir, overlay, grid_digest=grid_digest,
                    society_digest=rendered_digest, width=presentation["width"], height=presentation["height"],
                    viewbox_width=presentation["viewboxWidth"], viewbox_height=presentation["viewboxHeight"],
                    tectonic_diagnostics=presentation["tectonicDiagnostics"],
                    territorial_qa=presentation["territorialQa"],
                    settlement_locations=presentation["settlementLocations"])
            with measure_stage(target_dir, "administrative-tiles"):
                manifest = refresh_atlas_tile_themes(source_dir, target_dir, theme_features,
                    ink_layers={layer: [f for f in ink_features if f.layer == layer] for layer in _INK_LAYERS})
            with measure_stage(target_dir, "administrative-globe"):
                base = gzip.decompress((source_dir / "physical-surface.svgz").read_bytes()).decode("utf-8")
                documents = {theme: (target_dir / filename).read_text(encoding="utf-8")
                             for theme, filename in THEME_EXPORT_FILENAMES.items()}
                state_ink = render._partition_boundary_overlay("state-boundaries", state_paths,
                    color="#46413b", width=.40)
                textures = globe_theme_documents(base, documents, boundary_bodies={
                    "political": state_ink,
                    "provinces": state_ink + render._partition_boundary_overlay("province-boundaries",
                        province_paths, color="#685f54", width=.22, opacity=.72),
                })
                for theme in ("political", "provinces"):
                    (target_dir / f"globe-theme-{theme}.svg").write_text(textures[theme], encoding="utf-8")
            qa["viewportTiles"] = manifest
            for filename in qa["artifacts"]:
                qa["artifacts"][filename] = (target_dir / filename).stat().st_size
            qa["administrativeDrawing"] = record["geometry"]
            _write_json(target_dir / "qa.json", qa)
        record.update(status="complete", elapsedSeconds=round(time.perf_counter() - started, 3),
            sceneSha256=_sha256(target_dir / "atlas-scene.svg"),
            phases=json.loads((target_dir / "timing.json").read_text(encoding="utf-8")))
    except BaseException as error:
        record.update(status="failed", elapsedSeconds=round(time.perf_counter() - started, 3),
                      failure=f"{type(error).__name__}: {error}")
        _write_json(target_dir / "administrative-refresh.json", record)
        raise
    _write_json(target_dir / "administrative-refresh.json", record)
    return record
