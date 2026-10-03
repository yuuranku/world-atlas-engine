"""Shared geometry records for the overview and viewport-loaded atlas."""

from dataclasses import replace
import html

import numpy as np
import shapely

from .cartographic_tiles import TileFeature, feature_markup, geometry_path_data
from .cartographic_generalization import _constrained_ring
from .svg_paths import COORDINATE_SCALE


def shared_display_coverage(geometries):
    """Node the delivered precision once, retaining actual category ownership."""
    source = np.asarray(geometries, dtype=object)
    coordinates = shapely.get_coordinates(source)
    canonical = np.array_equal(coordinates, np.rint(coordinates*COORDINATE_SCALE)/COORDINATE_SCALE)
    if (canonical and bool(np.all(shapely.is_valid(source)))
            and bool(shapely.coverage_is_valid(source))):
        # An existing exact delivered coverage is already the common graph.
        # Never round independent source faces in order to qualify here.
        return source
    # Contour intervals can describe a common straight edge with different
    # intermediate vertices. A shared noded arrangement makes these T nodes
    # explicit on the delivery grid. Floating noding followed by independent
    # polygon snapping can give opposite owners different intersection nodes.
    # Independently rounding a T node before noding can also move it off its
    # shared straight segment. Snap-round the original complete graph first.
    graph = shapely.union_all(shapely.boundary(source), grid_size=1e-8)
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(graph)))
    owners = np.full(len(faces), -1, dtype=np.int32)
    samples = shapely.point_on_surface(faces)
    shapely.prepare(source)
    for index, geometry in enumerate(source):
        owners[shapely.covers(geometry, samples)] = index
    # Dissolve internal source edges before rounding the outer coverage.
    # Rounding polygon inputs separately can invent the same T-node hole
    # that the common linework above prevents.
    covered = shapely.set_precision(shapely.union_all(source), 1e-8)
    # Polygonize produced disjoint faces of one noded graph. Dissolving a
    # subset therefore needs no second intersection overlay.
    unowned = shapely.coverage_union_all(faces[owners == -1])
    if shapely.intersection(unowned, covered, grid_size=1e-8).area != 0:
        raise ValueError("display noding must assign every originally painted face")
    result = shapely.set_precision(np.asarray([
        shapely.coverage_union_all(faces[owners == index]) for index in range(len(source))
    ], dtype=object), 1e-8)
    if not bool(shapely.coverage_is_valid(result)):
        raise ValueError("display category boundaries must form a shared exact coverage")
    if not shapely.symmetric_difference(covered, shapely.coverage_union_all(
            result[~shapely.is_empty(result)]), grid_size=1e-8).is_empty:
        raise ValueError("display noding must retain the complete delivered coverage")
    return result


def _protect_native_coverage(original, simplified, frame_shape):
    """Keep shared original chords whenever their sweep crosses a cell centre."""
    def preserve_ring(first, last):
        if shapely.is_ccw(first) != shapely.is_ccw(last):
            last = shapely.LinearRing(list(last.coords)[::-1])
        return _constrained_ring(first, last, frame_shape=frame_shape)

    result = []
    for geometry, summary in zip(original, simplified, strict=True):
        if geometry.is_empty:
            result.append(geometry)
            continue
        parts, summaries = shapely.get_parts(geometry), shapely.get_parts(summary)
        if len(parts) != len(summaries):
            raise ValueError("overview simplification must retain all category components")
        preserved = []
        for part, candidate in zip(parts, summaries, strict=True):
            if len(part.interiors) != len(candidate.interiors):
                raise ValueError("overview simplification must retain category holes")
            preserved.append(shapely.Polygon(
                preserve_ring(part.exterior, candidate.exterior),
                [preserve_ring(first, last)
                 for first, last in zip(part.interiors, candidate.interiors, strict=True)],
            ))
        result.append(preserved[0] if geometry.geom_type == "Polygon" else shapely.MultiPolygon(preserved))
    result = np.asarray(result, dtype=object)
    if not bool(np.all(shapely.is_valid(result))):
        result = _restore_shared_topology_chords(original, result)
    if not bool(np.all(shapely.is_valid(result))) or not bool(shapely.coverage_is_valid(result)):
        raise ValueError("native-preserving overview must retain shared coverage topology")
    return result


def _restore_shared_topology_chords(original, protected):
    """Restore intersecting summary chords on both sides of a shared edge.

    Restoring a native-sensitive subchain can intersect a neighbouring chord
    that was safe in the fully simplified ring. Keep those original subchains
    as well, rather than repairing, buffering or filling the resulting shape.
    Every iteration removes at least one chord and introduces no new chord.
    """
    rings, parts, owners = [], [], []
    for owner, (source, geometry) in enumerate(zip(original, protected, strict=True)):
        for first, last in zip(shapely.get_parts(source), shapely.get_parts(geometry), strict=True):
            indices = []
            for before, after in zip((first.exterior, *first.interiors),
                                     (last.exterior, *last.interiors), strict=True):
                indices.append(len(rings))
                rings.append([np.asarray(before.coords)[:-1], np.asarray(after.coords)[:-1]])
            parts.append(indices)
            owners.append(owner)

    def assemble():
        polygons = [shapely.Polygon(rings[indices[0]][1],
                                    [rings[index][1] for index in indices[1:]]) for indices in parts]
        result = []
        for owner, geometry in enumerate(original):
            selected = [polygon for index, polygon in enumerate(polygons) if owners[index] == owner]
            result.append(selected[0] if geometry.geom_type == "Polygon" else shapely.MultiPolygon(selected))
        return polygons, np.asarray(result, dtype=object)

    def chord_key(first, last):
        return tuple(sorted((tuple(first), tuple(last))))

    lookups = {}
    def original_interval(ring, first, last):
        points = rings[ring][0]
        if ring not in lookups:
            lookups[ring] = {tuple(point): index for index, point in enumerate(points)}
        start, end = lookups[ring][tuple(first)], lookups[ring][tuple(last)]
        return start, (end - start) % len(points)

    polygons, result = assemble()
    while not bool(np.all(shapely.is_valid(result))):
        restore = set()
        for polygon, indices in zip(polygons, parts, strict=True):
            if polygon.is_valid:
                continue
            coordinates, edge_rings = [], []
            for ring in indices:
                points = rings[ring][1]
                coordinates.append(np.stack((points, np.roll(points, -1, axis=0)), axis=1))
                edge_rings.extend([ring] * len(points))
            coordinates = np.concatenate(coordinates)
            edges = shapely.linestrings(coordinates)
            pairs = shapely.STRtree(edges).query(edges, predicate="intersects")
            pairs = pairs[:, pairs[0] < pairs[1]]
            for first, last in pairs.T:
                first_points, last_points = coordinates[first], coordinates[last]
                common_endpoint = any(np.array_equal(a, b) for a in first_points for b in last_points)
                if common_endpoint and shapely.intersection(edges[first], edges[last]).geom_type == "Point":
                    continue
                for edge in (first, last):
                    _, distance = original_interval(edge_rings[edge], *coordinates[edge])
                    if distance > 1:
                        restore.add(chord_key(*coordinates[edge]))
        if not restore:
            raise ValueError("invalid overview topology must be attributable to a replaceable summary chord")
        for ring, (points, current) in enumerate(rings):
            kept = []
            for first, last in zip(current, np.roll(current, -1, axis=0), strict=True):
                if chord_key(first, last) in restore:
                    start, distance = original_interval(ring, first, last)
                    kept.append(points[(start + np.arange(distance)) % len(points)])
                else:
                    kept.append(first[None, :])
            rings[ring][1] = np.concatenate(kept)
        polygons, result = assemble()
    return result


def filled_geometries(paths):
    for points, codes in paths:
        starts = np.flatnonzero(codes == 1)
        rings = [points[first:last] for first, last in
                 zip(starts, (*starts[1:], len(points)), strict=True)]
        if rings:
            geometry = shapely.Polygon(rings[0], rings[1:])
            # Contour extraction can produce point-touching rings. Normalize
            # their polygonal faces before spatial slicing; line remnants do
            # not represent painted area.
            if not shapely.is_valid(geometry):
                geometry = shapely.make_valid(geometry)
            pending = [geometry]
            while pending:
                part = pending.pop()
                if part.geom_type == "Polygon" and not part.is_empty:
                    yield part
                elif hasattr(part, "geoms"):
                    pending.extend(reversed(part.geoms))


def filled_features(paths, layer, color, *, section="surface", theme=None,
                    opacity=1.0, clip=None, attributes=None):
    attrs = {"fill": color, "fill-opacity": str(opacity),
             "fill-rule": "evenodd", "stroke": "none", **(attributes or {})}
    if clip:
        attrs["clip"] = clip
    return [TileFeature(geometry, attrs, layer, theme=theme, section=section)
            for geometry in filled_geometries(paths)]


def line_features(paths, layer, color, width, *, opacity=1.0, dash=None,
                  clip=None, attributes=None):
    attrs = {"fill": "none", "stroke": color, "stroke-width": str(width),
             "stroke-opacity": str(opacity), "stroke-linecap": "round",
             "stroke-linejoin": "round", "vector-effect": "non-scaling-stroke",
             **(attributes or {})}
    if dash:
        attrs["stroke-dasharray"] = dash
    if clip:
        attrs["clip"] = clip
    features = []
    for path in paths:
        if len(path) < 2:
            continue
        geometry = shapely.LineString(path)
        # Eight-decimal adjacent duplicates encode no drawable line. This
        # tests the sole encoder without changing any surviving source node.
        if geometry_path_data(geometry):
            features.append(TileFeature(geometry, attrs, layer, section="ink"))
    return features


def overview_coverage(geometries, *, frame_shape):
    """Summarize one atomic coverage before deriving its grouped maps and ink."""
    original = shared_display_coverage(geometries)
    return _protect_native_coverage(original, shapely.coverage_simplify(
        original, .36, simplify_boundary=True,
    ), frame_shape)


def overview_markup(features, land_surface, width, height, *, section, theme=None):
    """Draw the supplied common overview silhouette and summarized features.

    The overview contains no minor contours, physical channel faces, or local
    routes. Administrative fills and ink retain their common source arcs.
    Other polygonal themes use shared coverage simplification, while the
    shore clip is common to every overview layer.
    """
    land = land_surface
    clip_ids = {"land": "land-silhouette-clip", "water": "water-silhouette-clip"}
    selected = [feature for feature in features
                if feature.section == section and feature.theme == theme
                and not feature.geometry.is_empty]
    if section == "theme" and selected:
        # A categorical overview needs one paint operation per category,
        # rather than repeating identical attributes on every disconnected
        # fragment. Combining equal categories preserves their exact area.
        grouped = {}
        for feature in selected:
            key = (feature.layer, tuple(sorted(feature.attributes.items())))
            if key not in grouped:
                grouped[key] = (feature, [])
            grouped[key][1].append(feature.geometry)
        selected = [replace(feature, geometry=shapely.union_all(geometries))
                    for feature, geometries in grouped.values()]
        if theme not in {"political", "provinces"}:
            geometries = overview_coverage([feature.geometry for feature in selected],
                                          frame_shape=(height, width))
            selected = [replace(feature, geometry=geometry)
                        for feature, geometry in zip(selected, geometries, strict=True)]
    groups = []
    active_layer = None
    coast_written = False
    for feature in selected:
        if feature.layer == "rivers" and (
                feature.attributes.get("class") == "river-channel" or
                int(feature.attributes.get("data-river-order", 0)) < 4):
            continue
        if feature.layer == "elevation-contours":
            continue
        if feature.attributes.get("data-landform-texture") == "true":
            continue
        if feature.layer == "transport-network" and (
                feature.attributes.get("data-route-importance") != "trunk"):
            continue
        geometry = feature.geometry
        if feature.layer == "coast":
            if coast_written:
                continue
            coast_written = True
            geometry = shapely.difference(land.boundary, shapely.box(0, 0, width, height).boundary)
        elif (feature.path_data is None and section != "theme"
              and feature.layer not in {"rivers", "transport-network",
                                        "state-boundaries", "province-boundaries"}
              and "data-landform-support" not in feature.attributes):
            # Supported landform areas already carry the field's extracted
            # contours. An absolute overview tolerance can replace a small
            # genuine peak by a triangle while retaining its native centre.
            # Their regional tint therefore uses the same boundary as detail.
            # River and road vertices also carry a jointly checked embedding:
            # independent summary chords can move their crossing facilities.
            # Administrative ink and both hierarchy fills likewise consume
            # the same province arcs. Independent theme or line reduction
            # would detach their boundary even if native ownership survives.
            geometry = shapely.simplify(geometry, .12, preserve_topology=True)
        if active_layer != feature.layer:
            if active_layer is not None:
                groups.append("</g>")
            active_layer = feature.layer
            groups.append(f'<g id="{html.escape(active_layer)}" data-tile-layer="{html.escape(active_layer)}">')
        groups.append(feature_markup(feature, geometry=geometry, clip_ids=clip_ids))
    if active_layer is not None:
        groups.append("</g>")
    body = "".join(groups)
    if section == "surface":
        land_data = geometry_path_data(land)
        water_data = geometry_path_data(shapely.difference(shapely.box(0, 0, width, height), land))
        body = (
            '<defs><clipPath id="land-silhouette-clip" clipPathUnits="userSpaceOnUse">'
            f'<path d="{land_data}" clip-rule="evenodd" /></clipPath>'
            '<clipPath id="water-silhouette-clip" clipPathUnits="userSpaceOnUse">'
            f'<path d="{water_data}" clip-rule="evenodd" /></clipPath></defs>' + body
        )
    return body
