"""Shared deterministic spatial primitives for all society partitions."""

from __future__ import annotations

import heapq
import math
from collections import Counter, deque
from collections.abc import Sequence

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as graph_connected_components

from ..model import WorldGrid
from ..polar import polar_continent_mask


_EIGHT_NEIGHBORS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def categorical_components(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label equal positive values once, including the longitude seam.

    ``scipy.ndimage.label`` labels any adjacent non-zero values as one feature,
    which is unsuitable for an ownership raster.  Building only equal-owner
    east/south edges creates a sparse graph over occupied cells instead.  This
    is one native-resolution pass for the whole partition, not one full-grid
    component pass per country or province.
    """

    labels = np.asarray(values)
    if labels.ndim != 2:
        raise ValueError("administrative ownership must be two-dimensional")
    positive = labels > 0
    nodes = np.flatnonzero(positive.ravel())
    if nodes.size == 0:
        return (
            nodes.astype(np.int64, copy=False),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    owner_by_node = labels.ravel()[nodes].astype(np.int32, copy=False)
    node_lookup = np.full(labels.size, -1, dtype=np.int32)
    node_lookup[nodes] = np.arange(nodes.size, dtype=np.int32)
    node_lookup = node_lookup.reshape(labels.shape)

    east_equal = positive & (labels == np.roll(labels, -1, axis=1))
    east_source = node_lookup[east_equal]
    east_target = np.roll(node_lookup, -1, axis=1)[east_equal]
    south_equal = (
        positive[:-1]
        & positive[1:]
        & (labels[:-1] == labels[1:])
    )
    south_source = node_lookup[:-1][south_equal]
    south_target = node_lookup[1:][south_equal]
    rows = np.concatenate((east_source, south_source)).astype(np.int32, copy=False)
    columns = np.concatenate((east_target, south_target)).astype(
        np.int32,
        copy=False,
    )
    graph = coo_matrix(
        (
            np.ones(rows.size, dtype=np.uint8),
            (rows, columns),
        ),
        shape=(nodes.size, nodes.size),
    ).tocsr()
    _count, component_by_node = graph_connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    return (
        nodes.astype(np.int64, copy=False),
        np.asarray(component_by_node, dtype=np.int32),
        owner_by_node,
    )


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Label wrapped-longitude four-neighbor components in deterministic order."""

    allowed = np.asarray(mask, dtype=bool)
    if allowed.ndim != 2:
        raise ValueError("connected_components expects a two-dimensional mask")
    labels, count = ndimage.label(allowed, output=np.int32)
    # Native four-neighbour labelling has finite edges. Merge only matching
    # longitude edges, then keep the first row-major cell's numbering.
    parents = np.arange(count + 1, dtype=np.int32)
    if allowed.shape[1]:
        for left, right in zip(labels[:, 0], labels[:, -1], strict=True):
            if not left or not right:
                continue
            while parents[left] != left:
                left = parents[left]
            while parents[right] != right:
                right = parents[right]
            parents[max(left, right)] = min(left, right)
    for identifier in range(1, count + 1):
        parents[identifier] = parents[parents[identifier]]
    roots, remap = np.unique(parents, return_inverse=True)
    labels = remap[labels].astype(np.int32)
    sizes = tuple(int(value) for value in np.bincount(labels.ravel(), minlength=len(roots))[1:])
    labels.setflags(write=False)
    return labels, sizes


def society_domain_mask(grid: WorldGrid) -> np.ndarray:
    """Return land where the configured world permits human-layer growth.

    Polar continents in this world deliberately end at the biome layer.  The
    rule is activated by the canonical physical metadata and uses the planet's
    axial tilt to derive the polar circles, rather than a fixed row count.
    """

    return (grid.water == 0) & ~polar_continent_mask(grid)


def reduce_field(values: np.ndarray, *, step: int, mode: str = "mean") -> np.ndarray:
    source = np.asarray(values)
    if source.ndim != 2:
        raise ValueError("reduce_field expects a two-dimensional array")
    if isinstance(step, bool) or int(step) < 1:
        raise ValueError("step must be a positive integer")
    step = int(step)
    height = int(math.ceil(source.shape[0] / step))
    width = int(math.ceil(source.shape[1] / step))
    if mode == "max":
        if source.dtype == np.bool_:
            fill = False
        elif np.issubdtype(source.dtype, np.integer):
            fill = np.iinfo(source.dtype).min
        else:
            fill = -np.inf
        padded = np.full((height * step, width * step), fill, dtype=source.dtype)
        padded[: source.shape[0], : source.shape[1]] = source
        return padded.reshape(height, step, width, step).max(axis=(1, 3))
    if mode == "min":
        if source.dtype == np.bool_:
            fill = True
        elif np.issubdtype(source.dtype, np.integer):
            fill = np.iinfo(source.dtype).max
        else:
            fill = np.inf
        padded = np.full((height * step, width * step), fill, dtype=source.dtype)
        padded[: source.shape[0], : source.shape[1]] = source
        return padded.reshape(height, step, width, step).min(axis=(1, 3))
    if mode != "mean":
        raise ValueError("mode must be mean, max, or min")
    numeric = np.asarray(source, dtype=np.float64)
    padded = np.zeros((height * step, width * step), dtype=np.float64)
    valid = np.zeros_like(padded)
    padded[: source.shape[0], : source.shape[1]] = numeric
    valid[: source.shape[0], : source.shape[1]] = 1.0
    block_sum = padded.reshape(height, step, width, step).sum(axis=(1, 3))
    block_count = valid.reshape(height, step, width, step).sum(axis=(1, 3))
    return np.divide(block_sum, block_count, where=block_count > 0.0)


def select_spaced_seeds(
    score: np.ndarray,
    valid: np.ndarray,
    *,
    count: int,
    minimum_distance: float,
) -> tuple[tuple[int, int], ...]:
    values = np.asarray(score, dtype=np.float64)
    allowed = np.asarray(valid, dtype=bool)
    if values.shape != allowed.shape:
        raise ValueError("score and valid must have matching shapes")
    if count < 1:
        return ()
    flat = np.flatnonzero(allowed & np.isfinite(values))
    order = flat[np.lexsort((flat, -values.reshape(-1)[flat]))]
    return select_spaced_candidates(order, values.shape, count=count, minimum_distance=minimum_distance)


def select_spaced_candidates(order, shape, *, count, minimum_distance, occupied=()):
    """Filter ranked cells by exact wrapped spacing using local buckets."""
    if count < 1:
        return ()
    chosen: list[tuple[int, int]] = []
    width = shape[1]
    bucket_size = max(float(minimum_distance), 1.0)
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, column in occupied:
        buckets.setdefault((math.floor(row / bucket_size), math.floor(column / bucket_size)), []).append((row, column))
    for index in order:
        row, column = np.unravel_index(int(index), shape)
        row, column = int(row), int(column)
        separated = True
        bucket_row = math.floor(row / bucket_size)
        nearby = []
        # Query all three longitude images; this also handles a partial bin
        # at the seam, without changing the exact wrapped distance test.
        column_bins = {math.floor((column + shift) / bucket_size) + offset
                       for shift in (-width, 0, width) for offset in (-1, 0, 1)}
        for bin_row in (bucket_row - 1, bucket_row, bucket_row + 1):
            for bin_column in column_bins:
                nearby.extend(buckets.get((bin_row, bin_column), ()))
        for other_row, other_column in nearby:
            dx = abs(column - other_column)
            dx = min(dx, width - dx)
            if math.hypot(row - other_row, dx) < minimum_distance:
                separated = False
                break
        if separated:
            chosen.append((int(row), int(column)))
            buckets.setdefault((bucket_row, math.floor(column / bucket_size)), []).append((row, column))
            if len(chosen) == count:
                break
    return tuple(chosen)


def allocate_regions_by_proximity(
    friction: np.ndarray,
    valid: np.ndarray,
    *,
    seeds: Sequence[tuple[int, int]],
    seed_strengths: Sequence[float] | None = None,
    label_friction: np.ndarray | None = None,
    radial_centers: Sequence[tuple[int, int]] | None = None,
    radial_scale: float | None = None,
    radial_weight: float = 0.0,
    components: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate native-resolution regions without a Python edge traversal.

    Every output cell is evaluated at its authored resolution.  Broad realm
    shape comes from wrapped distance, seed strength and the label's local
    environmental suitability; full-resolution boundary regrowth then places
    the final seam on rivers, ridges, watersheds and route crossings.  This is
    intentionally a contiguous-front approximation rather than an independent
    shortest-path map for every label, which also prevents a culture from
    tunnelling through a rival and reappearing as an unexplained enclave.
    """

    resistance = np.asarray(friction, dtype=np.float32)
    allowed = np.asarray(valid, dtype=bool)
    if resistance.ndim != 2 or resistance.shape != allowed.shape:
        raise ValueError("friction and valid must have matching two-dimensional shapes")
    if not seeds:
        return (
            np.full(resistance.shape, -1, dtype=np.int32),
            np.full(resistance.shape, np.inf, dtype=np.float32),
        )
    strengths = np.ones(len(seeds), dtype=np.float32)
    if seed_strengths is not None:
        strengths = np.asarray(tuple(seed_strengths), dtype=np.float32)
        if strengths.shape != (len(seeds),) or np.any(strengths <= 0.0):
            raise ValueError("seed_strengths must be positive and correspond to seeds")
    label_resistance = None
    if label_friction is not None:
        label_resistance = np.asarray(label_friction, dtype=np.float32)
        if label_resistance.shape != (len(seeds), *resistance.shape):
            raise ValueError("label_friction must have shape (N, H, W)")
    centers = tuple(seeds) if radial_centers is None else tuple(radial_centers)
    if len(centers) != len(seeds):
        raise ValueError("radial_centers must correspond to seeds")
    if radial_weight > 0.0 and (
        radial_scale is None or not math.isfinite(radial_scale) or radial_scale <= 0.0
    ):
        raise ValueError("radial_scale must be finite and positive")
    component_labels = None if components is None else np.asarray(components)
    if component_labels is not None and component_labels.shape != resistance.shape:
        raise ValueError("components must align with the allocation lattice")

    height, width = resistance.shape
    row_grid = np.arange(height, dtype=np.float32)[:, None]
    column_grid = np.arange(width, dtype=np.float32)[None, :]
    best_cost = np.full(resistance.shape, np.inf, dtype=np.float32)
    labels = np.full(resistance.shape, -1, dtype=np.int32)
    for index, ((seed_row, seed_column), (center_row, center_column)) in enumerate(
        zip(seeds, centers, strict=True)
    ):
        if not (0 <= seed_row < height and 0 <= seed_column < width):
            raise ValueError("seed lies outside the allocation lattice")
        if not allowed[seed_row, seed_column]:
            raise ValueError("seed must lie on a valid cell")
        row_distance = row_grid - float(seed_row)
        column_distance = np.abs(column_grid - float(seed_column))
        column_distance = np.minimum(column_distance, width - column_distance)
        distance = np.hypot(row_distance, column_distance).astype(np.float32)
        active = resistance if label_resistance is None else label_resistance[index]
        seed_resistance = max(0.20, float(active[seed_row, seed_column]))
        environmental_factor = np.sqrt(
            np.clip(active / seed_resistance, 0.16, 25.0)
        ).astype(np.float32)
        candidate = distance * environmental_factor / strengths[index]
        if radial_weight > 0.0:
            center_row_distance = row_grid - float(center_row)
            center_column_distance = np.abs(column_grid - float(center_column))
            center_column_distance = np.minimum(
                center_column_distance,
                width - center_column_distance,
            )
            center_distance = np.hypot(
                center_row_distance,
                center_column_distance,
            ).astype(np.float32)
            candidate *= 1.0 + float(radial_weight) * np.power(
                center_distance / float(radial_scale),
                1.35,
            )
        candidate_domain = allowed
        if component_labels is not None:
            seed_component = int(component_labels[seed_row, seed_column])
            candidate_domain = candidate_domain & (component_labels == seed_component)
        candidate = np.where(candidate_domain, candidate, np.inf)
        better = candidate < best_cost - 1.0e-6
        labels[better] = index + 1
        best_cost[better] = candidate[better]
    for identifier, (row, column) in enumerate(seeds, start=1):
        labels[row, column] = identifier
        best_cost[row, column] = 0.0
    return labels, best_cost


def allocate_regions(
    friction: np.ndarray,
    valid: np.ndarray,
    *,
    seeds: Sequence[tuple[int, int]],
    maximum_cost: float | None = None,
    transition_penalty: np.ndarray | None = None,
    seed_strengths: Sequence[float] | None = None,
    label_friction: np.ndarray | None = None,
    radial_centers: Sequence[tuple[int, int]] | None = None,
    radial_scale: float | None = None,
    radial_weight: float = 0.0,
    allow_diagonal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate one label per valid cell using wrapped eight-neighbor Dijkstra."""

    resistance = np.asarray(friction, dtype=np.float64)
    allowed = np.asarray(valid, dtype=bool)
    if resistance.shape != allowed.shape:
        raise ValueError("friction and valid must have matching shapes")
    if np.any(~np.isfinite(resistance[allowed])) or np.any(resistance[allowed] <= 0.0):
        raise ValueError("valid friction must be finite and positive")
    transitions = None
    if transition_penalty is not None:
        transitions = np.asarray(transition_penalty, dtype=np.float64)
        if transitions.shape != (8, *resistance.shape):
            raise ValueError("transition_penalty must have shape (8, H, W)")
        if np.any(~np.isfinite(transitions)) or np.any(transitions < 0.0):
            raise ValueError("transition_penalty must be finite and non-negative")
    strengths = np.ones(len(seeds), dtype=np.float64)
    if seed_strengths is not None:
        strengths = np.asarray(tuple(seed_strengths), dtype=np.float64)
        if strengths.shape != (len(seeds),):
            raise ValueError("seed_strengths must correspond to seeds")
        if np.any(~np.isfinite(strengths)) or np.any(strengths <= 0.0):
            raise ValueError("seed_strengths must be finite and positive")
    label_resistance = None
    if label_friction is not None:
        label_resistance = np.asarray(label_friction, dtype=np.float64)
        if label_resistance.shape != (len(seeds), *resistance.shape):
            raise ValueError("label_friction must have shape (N, H, W)")
        if np.any(~np.isfinite(label_resistance[:, allowed])) or np.any(
            label_resistance[:, allowed] <= 0.0
        ):
            raise ValueError("valid label_friction must be finite and positive")
    centers = None
    if radial_centers is not None:
        centers = tuple((int(row), int(column)) for row, column in radial_centers)
        if len(centers) != len(seeds):
            raise ValueError("radial_centers must correspond to seeds")
        if radial_scale is None or not math.isfinite(radial_scale) or radial_scale <= 0.0:
            raise ValueError("radial_scale must be finite and positive")
        if not math.isfinite(radial_weight) or radial_weight < 0.0:
            raise ValueError("radial_weight must be finite and non-negative")
    labels = np.full(resistance.shape, -1, dtype=np.int32)
    costs = np.full(resistance.shape, np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int, int]] = []
    seed_owner: dict[tuple[int, int], int] = {}
    for label, (row, column) in enumerate(seeds, start=1):
        if not (0 <= row < resistance.shape[0] and 0 <= column < resistance.shape[1]):
            raise ValueError("seed lies outside the allocation lattice")
        if not allowed[row, column]:
            raise ValueError("seed must lie on a valid cell")
        costs[row, column] = 0.0
        labels[row, column] = label
        seed_owner[(row, column)] = label
        heapq.heappush(heap, (0.0, label, row, column))
    height, width = resistance.shape
    if not isinstance(allow_diagonal, bool):
        raise TypeError("allow_diagonal must be boolean")
    neighbors = tuple(
        (direction, dy, dx, distance)
        for direction, (dy, dx, distance) in enumerate(
            (
                (-1, -1, math.sqrt(2.0)),
                (-1, 0, 1.0),
                (-1, 1, math.sqrt(2.0)),
                (0, -1, 1.0),
                (0, 1, 1.0),
                (1, -1, math.sqrt(2.0)),
                (1, 0, 1.0),
                (1, 1, math.sqrt(2.0)),
            )
        )
        if allow_diagonal or direction in (1, 3, 4, 6)
    )
    while heap:
        cost, label, row, column = heapq.heappop(heap)
        if cost != costs[row, column] or label != labels[row, column]:
            continue
        for direction, dy, dx, distance in neighbors:
            next_row = row + dy
            if next_row < 0 or next_row >= height:
                continue
            next_column = (column + dx) % width
            if not allowed[next_row, next_column]:
                continue
            owner = seed_owner.get((next_row, next_column))
            if owner is not None and owner != label:
                continue
            active_resistance = (
                resistance
                if label_resistance is None
                else label_resistance[label - 1]
            )
            edge = 0.5 * (
                active_resistance[row, column]
                + active_resistance[next_row, next_column]
            )
            if centers is not None and radial_weight > 0.0:
                center_row, center_column = centers[label - 1]
                center_dx = abs(next_column - center_column)
                center_dx = min(center_dx, width - center_dx)
                center_distance = math.hypot(next_row - center_row, center_dx)
                edge *= 1.0 + radial_weight * math.pow(
                    center_distance / float(radial_scale),
                    1.35,
                )
            if transitions is not None:
                edge += transitions[direction, row, column]
            next_cost = cost + edge * distance / strengths[label - 1]
            if maximum_cost is not None and next_cost > maximum_cost:
                continue
            current_cost = costs[next_row, next_column]
            current_label = int(labels[next_row, next_column])
            if next_cost < current_cost - 1.0e-12 or (
                abs(next_cost - current_cost) <= 1.0e-12 and label < current_label
            ):
                costs[next_row, next_column] = next_cost
                labels[next_row, next_column] = label
                heapq.heappush(heap, (next_cost, label, next_row, next_column))
    return labels, costs


def fill_unreachable_components(
    labels: np.ndarray,
    valid: np.ndarray,
    seeds: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Attach disconnected valid islands to the nearest existing seed label.

    Cost allocation cannot cross water. A remote island without its own seed
    still belongs to a language or polity through its nearest cultural/port
    core; assigning the whole component preserves the water barrier instead of
    drawing a fictional land corridor.
    """

    result = np.asarray(labels, dtype=np.int32).copy()
    allowed = np.asarray(valid, dtype=bool)
    if result.shape != allowed.shape:
        raise ValueError("labels and valid must have matching shapes")
    components, sizes = connected_components(allowed)
    width = allowed.shape[1]
    for identifier in range(1, len(sizes) + 1):
        region = components == identifier
        if not np.any(region) or np.any(result[region] > 0):
            continue
        rows, columns = np.nonzero(region)
        row_center = float(np.mean(rows))
        angles = columns.astype(np.float64) * (2.0 * np.pi / width)
        column_center = float(
            (math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
            % (2.0 * np.pi))
            * width
            / (2.0 * np.pi)
        )
        nearest = min(
            range(len(seeds)),
            key=lambda index: (
                math.hypot(
                    seeds[index][0] - row_center,
                    min(
                        abs(seeds[index][1] - column_center),
                        width - abs(seeds[index][1] - column_center),
                    ),
                ),
                index,
            ),
        )
        result[region] = nearest + 1
    return result


def physical_transition_penalties(
    elevation: np.ndarray,
    river_order: np.ndarray,
    *,
    land_mask: np.ndarray,
) -> np.ndarray:
    """Return directional costs for crossing natural political frontiers.

    Travel *along* a valley or river stays comparatively cheap.  Crossing a
    measured crest, slope break, or either bank of a large river is costly.
    Keeping this as an edge field (rather than another per-cell terrain cost)
    is important: it lets a territory occupy a whole basin while placing its
    frontier on the basin rim instead of merely avoiding all rugged land.
    """

    heights = np.asarray(elevation, dtype=np.float64)
    rivers = np.asarray(river_order)
    dry = np.asarray(land_mask, dtype=bool)
    if rivers.shape != heights.shape or dry.shape != heights.shape:
        raise ValueError("physical boundary fields must have matching shapes")
    if heights.ndim != 2:
        raise ValueError("physical boundary fields must be two-dimensional")
    # Eight full-resolution edge fields are the largest temporary society
    # allocation.  Float32 is ample for costs measured in single-cell travel
    # units and halves peak memory during native-resolution frontier regrowth.
    penalties = np.zeros((8, *heights.shape), dtype=np.float32)
    major_river = rivers >= 3
    secondary_river = rivers >= 2
    for direction, (dy, dx) in enumerate(_EIGHT_NEIGHBORS):
        target_elevation = np.roll(heights, shift=(-dy, -dx), axis=(0, 1))
        target_dry = np.roll(dry, shift=(-dy, -dx), axis=(0, 1))
        target_river_order = np.roll(rivers, shift=(-dy, -dx), axis=(0, 1))
        target_major_river = target_river_order >= 3
        target_secondary_river = target_river_order >= 2
        relief = np.abs(target_elevation - heights)
        ridge = np.maximum(target_elevation, heights)
        # An outlet ID is a D8 routing result, not measured ridge height.  In
        # particular, priority-flood flats can have long rectangular outlet
        # partitions.  Require the terrain on BOTH sides of this crossing to
        # descend away from the edge.  Two support distances distinguish a
        # broad crest from a one-cell peak without introducing a wave field.
        def terrain_neighbor(field: np.ndarray, steps: int) -> np.ndarray:
            delta_y, delta_x = dy * steps, dx * steps
            values = np.roll(field, shift=(-delta_y, -delta_x), axis=(0, 1))
            if delta_y > 0:
                values[-delta_y:, :] = np.roll(field[-1, :], -delta_x)
            elif delta_y < 0:
                values[:-delta_y, :] = np.roll(field[0, :], -delta_x)
            return values

        crest = np.zeros(heights.shape, dtype=np.float64)
        for support in (1, 2):
            before = terrain_neighbor(heights, -support)
            after = terrain_neighbor(heights, support + 1)
            supported = np.minimum(heights - before, target_elevation - after)
            actual_land = dry & target_dry & terrain_neighbor(dry, -support) & terrain_neighbor(dry, support + 1)
            crest = np.maximum(crest, np.where(actual_land, np.maximum(0.0, supported) / support, 0.0))
        river_bank = major_river ^ target_major_river
        secondary_bank = secondary_river ^ target_secondary_river
        river_strength = np.maximum(rivers, target_river_order).astype(np.float64)
        ridge_prominence = np.clip(ridge - 0.40, 0.0, 0.60)
        penalties[direction] = (
            34.0 * np.power(relief, 1.20)
            + 52.0 * crest * (1.0 + 2.0 * ridge_prominence)
            + river_bank
            * (5.0 + 2.8 * np.clip(river_strength - 2.0, 0.0, 3.0))
            + secondary_bank * 1.8
        )
        penalties[direction, ~(dry & target_dry)] = 0.0
        if dy < 0:
            penalties[direction, 0, :] = 0.0
        elif dy > 0:
            penalties[direction, -1, :] = 0.0
    return penalties


def natural_compartment_ids(
    valid: np.ndarray,
    transition_penalty: np.ndarray,
    *,
    domain: np.ndarray | None = None,
    barrier_threshold: float = 14.0,
) -> np.ndarray:
    """Label land compartments enclosed by strong geographic edge barriers.

    The operation stays on the native grid and cuts adjacency, not pixels.
    Rivers, drainage divides and ridges therefore separate two nearby places
    without turning either feature into an artificial band of empty land.
    An optional positive ``domain`` is a hard cultural parent boundary.
    """

    allowed = np.asarray(valid, dtype=bool)
    transitions = np.asarray(transition_penalty, dtype=np.float32)
    if allowed.ndim != 2 or transitions.shape != (8, *allowed.shape):
        raise ValueError("natural-compartment fields must align")
    if not np.isfinite(barrier_threshold) or barrier_threshold <= 0.0:
        raise ValueError("barrier_threshold must be finite and positive")
    if domain is None:
        domains = np.ones(allowed.shape, dtype=np.int32)
    else:
        domains = np.asarray(domain, dtype=np.int32)
        if domains.shape != allowed.shape:
            raise ValueError("natural-compartment domain must align")
        allowed = allowed & (domains > 0)

    labels = np.zeros(allowed.shape, dtype=np.int32)
    height, width = allowed.shape
    neighbors = (
        (1, -1, 0, 6),
        (3, 0, -1, 4),
        (4, 0, 1, 3),
        (6, 1, 0, 1),
    )
    identifier = 0
    for start_row, start_column in zip(*np.nonzero(allowed), strict=True):
        if labels[start_row, start_column] > 0:
            continue
        identifier += 1
        component_domain = int(domains[start_row, start_column])
        labels[start_row, start_column] = identifier
        queue = deque(((int(start_row), int(start_column)),))
        while queue:
            row, column = queue.popleft()
            for direction, dy, dx, reverse_direction in neighbors:
                next_row = row + dy
                if next_row < 0 or next_row >= height:
                    continue
                next_column = (column + dx) % width
                if (
                    not allowed[next_row, next_column]
                    or labels[next_row, next_column] > 0
                    or int(domains[next_row, next_column]) != component_domain
                ):
                    continue
                barrier = max(
                    float(transitions[direction, row, column]),
                    float(transitions[reverse_direction, next_row, next_column]),
                )
                if barrier >= barrier_threshold:
                    continue
                labels[next_row, next_column] = identifier
                queue.append((next_row, next_column))
    labels.setflags(write=False)
    return labels


def snap_partition_to_natural_regions(
    labels: np.ndarray,
    valid: np.ndarray,
    transition_penalty: np.ndarray,
    anchors: dict[tuple[int, int], int],
    *,
    owner_field: np.ndarray | None = None,
    label_owner: np.ndarray | None = None,
    barrier_threshold: float = 14.0,
    anchored_support: float = 0.45,
    unanchored_majority: float = 0.64,
) -> np.ndarray:
    """Keep an unopposed watershed or valley side inside one partition.

    Least-cost growth must draw a bisector when two cores compete.  This pass
    treats strong physical edges as natural-region rims.  A lone core may only
    claim the whole region when its existing territory already supplies broad
    support; otherwise one river-valley city could annex an arbitrarily long
    corridor.  Unanchored regions likewise require a decisive majority.
    Regions with competing cores stay divided because the internal boundary
    then represents real settlement competition.  Optional parent ownership
    keeps state-level regions inside civilizations and provinces inside their
    state before any political aggregation occurs.
    """

    result = np.asarray(labels, dtype=np.int32).copy()
    allowed = np.asarray(valid, dtype=bool)
    transitions = np.asarray(transition_penalty, dtype=np.float64)
    if result.shape != allowed.shape or transitions.shape != (8, *result.shape):
        raise ValueError("natural-region partition fields must align")
    if not np.isfinite(barrier_threshold) or barrier_threshold <= 0.0:
        raise ValueError("barrier_threshold must be finite and positive")
    if not 0.0 < anchored_support <= 1.0:
        raise ValueError("anchored_support must lie in (0, 1]")
    if not 0.5 < unanchored_majority <= 1.0:
        raise ValueError("unanchored_majority must lie in (0.5, 1]")
    maximum_label = int(result.max(initial=0))
    if owner_field is None and label_owner is None:
        cell_owner = np.ones(result.shape, dtype=np.int32)
        owner_by_label = np.ones(maximum_label + 1, dtype=np.int32)
        owner_by_label[0] = 0
    elif owner_field is None or label_owner is None:
        raise ValueError("owner_field and label_owner must be supplied together")
    else:
        cell_owner = np.asarray(owner_field, dtype=np.int32)
        owner_by_label = np.asarray(label_owner, dtype=np.int32)
        if cell_owner.shape != result.shape:
            raise ValueError("owner_field must align with natural-region labels")
        if owner_by_label.ndim != 1 or len(owner_by_label) <= maximum_label:
            raise ValueError("label_owner must cover every positive label")
        positive = allowed & (result > 0)
        authored_parent = positive & (cell_owner > 0)
        if np.any(
            owner_by_label[result[authored_parent]] != cell_owner[authored_parent]
        ):
            raise ValueError("partition labels must begin inside their parent domain")
        # A zero parent is not a second domain.  It is neutral land between
        # authored culture fields, so keep its current seed-derived domain
        # during snapping.  This prevents a neutral corridor from joining two
        # distinct parent domains while still allowing ordinary hinterland to
        # cover not-yet-classified cells.
        neutral = positive & (cell_owner == 0)
        cell_owner = cell_owner.copy()
        cell_owner[neutral] = owner_by_label[result[neutral]]
    height, width = result.shape
    visited = np.zeros(result.shape, dtype=bool)
    neighbors = (
        (1, -1, 0, 6),
        (3, 0, -1, 4),
        (4, 0, 1, 3),
        (6, 1, 0, 1),
    )
    for start_row, start_column in zip(*np.nonzero(allowed), strict=True):
        if visited[start_row, start_column]:
            continue
        queue: deque[tuple[int, int]] = deque(
            ((int(start_row), int(start_column)),)
        )
        visited[start_row, start_column] = True
        component_owner = int(cell_owner[start_row, start_column])
        cells: list[tuple[int, int]] = []
        counts: Counter[int] = Counter()
        owners: set[int] = set()
        while queue:
            row, column = queue.popleft()
            cells.append((row, column))
            identifier = int(result[row, column])
            if identifier > 0:
                counts[identifier] += 1
            owner = int(anchors.get((row, column), 0))
            if owner > 0:
                owners.add(owner)
            for direction, dy, dx, reverse_direction in neighbors:
                next_row = row + dy
                if next_row < 0 or next_row >= height:
                    continue
                next_column = (column + dx) % width
                if visited[next_row, next_column] or not allowed[next_row, next_column]:
                    continue
                if int(cell_owner[next_row, next_column]) != component_owner:
                    continue
                barrier = max(
                    float(transitions[direction, row, column]),
                    float(
                        transitions[
                            reverse_direction,
                            next_row,
                            next_column,
                        ]
                    ),
                )
                if barrier >= barrier_threshold:
                    continue
                visited[next_row, next_column] = True
                queue.append((next_row, next_column))
        replacement = 0
        if len(owners) == 1:
            candidate = next(iter(owners))
            if (
                int(owner_by_label[candidate]) == component_owner
                and counts[candidate] / max(1, len(cells)) >= anchored_support
            ):
                replacement = candidate
        elif not owners and counts:
            eligible = {
                identifier: count
                for identifier, count in counts.items()
                if int(owner_by_label[identifier]) == component_owner
            }
            if not eligible:
                continue
            candidate = min(
                eligible,
                key=lambda identifier: (-counts[identifier], identifier),
            )
            if counts[candidate] / max(1, sum(counts.values())) >= unanchored_majority:
                replacement = candidate
        if replacement <= 0:
            continue
        rows, columns = zip(*cells, strict=True)
        result[rows, columns] = replacement
    for cell, identifier in anchors.items():
        result[cell] = identifier
    return result


def refine_partition_boundaries(
    labels: np.ndarray,
    valid: np.ndarray,
    transition_penalty: np.ndarray,
    anchors: dict[tuple[int, int], int],
    *,
    owner_field: np.ndarray | None = None,
    label_owner: np.ndarray | None = None,
    friction: np.ndarray | None = None,
    band_radius: int = 7,
) -> np.ndarray:
    """Regrow only a partition's frontier on the full-resolution landscape.

    This pass freezes every interior, opens a narrow band on both sides of each border,
    and lets the neighbouring labels compete again at native resolution.
    Directional river and ridge costs therefore decide the actual
    seam. No independent coordinate noise changes administrative ownership.

    ``owner_field`` and ``label_owner`` constrain nested partitions: states
    cannot cross civilization domains and provinces cannot cross countries.
    """

    # Local import avoids the spatial/component import cycle. Formation and
    # its final seam regrowth share one continuous arrival-time primitive.
    from .territorial_simulation import _continuous_front_time, _FRONT_PEERS

    source = np.asarray(labels)
    allowed = np.asarray(valid, dtype=bool)
    transitions = np.asarray(transition_penalty)
    if source.ndim != 2 or source.shape != allowed.shape:
        raise ValueError("partition labels and valid mask must align")
    if transitions.shape != (8, *source.shape):
        raise ValueError("transition_penalty must have shape (8, H, W)")
    if np.any(~np.isfinite(transitions)) or np.any(transitions < 0.0):
        raise ValueError("transition_penalty must be finite and non-negative")
    band_radius = int(band_radius)
    if band_radius < 1:
        raise ValueError("band_radius must be positive")

    maximum_label = int(source.max(initial=0))
    if owner_field is None and label_owner is None:
        cell_owner = np.ones(source.shape, dtype=np.int32)
        owner_by_label = np.ones(maximum_label + 1, dtype=np.int32)
        owner_by_label[0] = 0
    elif owner_field is None or label_owner is None:
        raise ValueError("owner_field and label_owner must be supplied together")
    else:
        cell_owner = np.asarray(owner_field)
        owner_by_label = np.asarray(label_owner)
        if cell_owner.shape != source.shape:
            raise ValueError("owner_field must align with partition labels")
        if owner_by_label.ndim != 1 or len(owner_by_label) <= maximum_label:
            raise ValueError("label_owner must cover every positive label")

    resistance = (
        np.ones(source.shape, dtype=np.float32)
        if friction is None
        else np.asarray(friction, dtype=np.float32)
    )
    if resistance.shape != source.shape:
        raise ValueError("friction must align with partition labels")
    if np.any(~np.isfinite(resistance[allowed])) or np.any(resistance[allowed] <= 0.0):
        raise ValueError("valid friction must be finite and positive")

    result = source.astype(np.int32, copy=True)
    positive = allowed & (result > 0)
    mapped_owner = np.zeros(source.shape, dtype=np.int32)
    mapped_owner[positive] = owner_by_label[result[positive]]
    boundary = np.zeros(source.shape, dtype=bool)
    height, width = source.shape

    def shifted(values: np.ndarray, dy: int, dx: int, fill: int | bool) -> np.ndarray:
        output = np.roll(values, shift=(-dy, -dx), axis=(0, 1))
        if dy < 0:
            output[-dy - 1 :: -1, :] = fill
        elif dy > 0:
            output[height - dy :, :] = fill
        return output

    for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        neighbor_label = shifted(result, dy, dx, -1)
        neighbor_positive = shifted(positive, dy, dx, False)
        neighbor_owner = shifted(mapped_owner, dy, dx, 0)
        boundary |= (
            positive
            & neighbor_positive
            & (result != neighbor_label)
            & (mapped_owner == neighbor_owner)
        )
    if not np.any(boundary):
        return result.astype(source.dtype, copy=False)

    band = boundary.copy()
    for _ in range(band_radius):
        expanded = band.copy()
        for dy, dx in _EIGHT_NEIGHBORS:
            expanded |= shifted(band, dy, dx, False)
        band = expanded & positive

    anchor_labels = np.zeros(source.shape, dtype=np.int32)
    for (row, column), identifier in anchors.items():
        if not (0 <= row < height and 0 <= column < width):
            raise ValueError("anchor lies outside the partition lattice")
        if identifier <= 0 or identifier > maximum_label:
            raise ValueError("anchor label must be a positive partition label")
        anchor_labels[row, column] = identifier

    seed_mask = np.zeros(source.shape, dtype=bool)
    frozen = positive & ~band
    for dy, dx in _EIGHT_NEIGHBORS:
        neighbor_frozen = shifted(frozen, dy, dx, False)
        neighbor_label = shifted(result, dy, dx, -1)
        seed_mask |= band & neighbor_frozen & (neighbor_label == result)
    seed_mask |= band & (anchor_labels > 0)

    costs = np.full(source.shape, np.inf, dtype=np.float64)
    owners = np.zeros(source.shape, dtype=np.int32)
    accepted = np.zeros(source.shape, dtype=bool)
    heap: list[tuple[float, int, int, int]] = []
    for row, column in zip(*np.nonzero(seed_mask), strict=True):
        identifier = int(anchor_labels[row, column] or result[row, column])
        costs[row, column] = 0.0
        owners[row, column] = identifier
        heapq.heappush(heap, (0.0, identifier, int(row), int(column)))

    directions = tuple(
        (direction, dy, dx, math.sqrt(2.0) if dy and dx else 1.0)
        for direction, (dy, dx) in enumerate(_EIGHT_NEIGHBORS)
    )
    while heap:
        cost, identifier, row, column = heapq.heappop(heap)
        if cost != costs[row, column] or identifier != owners[row, column]:
            continue
        if accepted[row, column]:
            continue
        accepted[row, column] = True
        label_domain = int(owner_by_label[identifier])
        for direction, dy, dx, distance in directions:
            next_row = row + dy
            if next_row < 0 or next_row >= height:
                continue
            next_column = (column + dx) % width
            if accepted[next_row, next_column] or not band[next_row, next_column]:
                continue
            if int(cell_owner[next_row, next_column]) != label_domain:
                continue
            forced = int(anchor_labels[next_row, next_column])
            if forced > 0 and forced != identifier:
                continue
            reverse_direction = 7 - direction
            barrier = max(
                float(transitions[direction, row, column]),
                float(transitions[reverse_direction, next_row, next_column]),
            )
            edge = (
                0.5 * (float(resistance[row, column]) + float(resistance[next_row, next_column]))
                + barrier
            ) * distance
            next_cost = cost + edge
            if direction in _FRONT_PEERS:
                next_cost = _continuous_front_time(
                    next_row, next_column, direction, identifier, cost, edge, 1.,
                    resistance, transitions, costs, owners, accepted, symmetric_crossings=True,
                )
            current_cost = float(costs[next_row, next_column])
            current_owner = int(owners[next_row, next_column])
            if next_cost < current_cost - 1.0e-12 or (
                abs(next_cost - current_cost) <= 1.0e-12
                and (current_owner <= 0 or identifier < current_owner)
            ):
                costs[next_row, next_column] = next_cost
                owners[next_row, next_column] = identifier
                heapq.heappush(
                    heap,
                    (next_cost, identifier, next_row, next_column),
                )

    reachable = band & (owners > 0)
    result[reachable] = owners[reachable]
    for cell, identifier in anchors.items():
        result[cell] = identifier
    return result.astype(source.dtype, copy=False)
