"""Continuous, reproducible morphology controls for tectonic planets.

The generator samples a position in a geological morphospace.  Human-facing
world descriptions are produced afterwards from measured map properties, so
there is no closed list of map templates hidden behind the seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class WorldMorphology:
    cycle_position: float
    continental_aggregation: float
    crust_fragmentation: float
    microplate_share: float
    island_arc_activity: float
    hotspot_activity: float
    latitude_bias: float
    hemisphere_asymmetry: float
    ocean_basin_openness: float

    def __post_init__(self) -> None:
        for name, value in vars_from_slots(self).items():
            lower = -1.0 if name == "latitude_bias" else 0.0
            if not math.isfinite(value) or not lower <= value <= 1.0:
                raise ValueError(f"{name} must be in [{lower}, 1]")


def vars_from_slots(value: WorldMorphology) -> dict[str, float]:
    return {
        field: float(getattr(value, field))
        for field in value.__slots__
    }


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(min(upper, max(lower, value)))


def _cycle_bump(position: float, centre: float, width: float) -> float:
    distance = abs(position - centre)
    distance = min(distance, 1.0 - distance)
    return math.exp(-0.5 * (distance / width) ** 2)


def sample_world_morphology(seed: int) -> WorldMorphology:
    """Sample correlated geological controls from one deterministic seed.

    ``cycle_position`` is circular: assembly approaches zero, maximum
    aggregation lies near 0.25, breakup near 0.5, and maximum dispersal near
    0.75.  The remaining controls vary continuously around that cycle instead
    of selecting a named preset.
    """

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    mixed = (int(seed) ^ 0xD1B54A32D192ED03) & 0xFFFFFFFFFFFFFFFF
    rng = np.random.Generator(np.random.PCG64(mixed))
    cycle = float(rng.random())
    jitter = lambda scale: float(rng.normal(0.0, scale))

    aggregation = 0.50 + 0.38 * math.cos(math.tau * (cycle - 0.25))
    breakup = _cycle_bump(cycle, 0.50, 0.13)
    dispersal = _cycle_bump(cycle, 0.75, 0.18)
    assembly = _cycle_bump(cycle, 0.02, 0.17)
    fragmentation = 0.16 + 0.57 * breakup + 0.23 * dispersal
    microplates = 0.10 + 0.38 * dispersal + 0.32 * assembly
    island_arcs = 0.12 + 0.55 * assembly + 0.23 * dispersal
    basin_openness = 0.18 + 0.62 * dispersal + 0.18 * breakup

    return WorldMorphology(
        cycle_position=cycle,
        continental_aggregation=_clamp(aggregation + jitter(0.07)),
        crust_fragmentation=_clamp(fragmentation + jitter(0.08)),
        microplate_share=_clamp(microplates + jitter(0.09)),
        island_arc_activity=_clamp(island_arcs + jitter(0.08)),
        hotspot_activity=_clamp(float(rng.beta(2.0, 4.2))),
        latitude_bias=_clamp(float(rng.normal(0.0, 0.46)), -1.0, 1.0),
        hemisphere_asymmetry=_clamp(float(rng.beta(2.1, 2.8))),
        ocean_basin_openness=_clamp(basin_openness + jitter(0.08)),
    )


def representative_morphology_seeds(
    root_seed: int,
    count: int,
    *,
    candidate_count: int = 256,
) -> tuple[int, ...]:
    """Select a reproducible, well-spaced sample of the continuous axes."""

    if isinstance(root_seed, bool) or not isinstance(root_seed, (int, np.integer)):
        raise TypeError("root_seed must be an integer")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError("count must be an integer")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, (int, np.integer)):
        raise TypeError("candidate_count must be an integer")
    count = int(count)
    candidate_count = int(candidate_count)
    if count <= 0 or candidate_count < count:
        raise ValueError("candidate_count must be at least the positive requested count")

    rng = np.random.Generator(
        np.random.PCG64((int(root_seed) ^ 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF)
    )
    candidates = rng.integers(0, 2**32, size=candidate_count, dtype=np.uint64)
    vectors = np.asarray(
        [list(vars_from_slots(sample_world_morphology(int(seed))).values()) for seed in candidates],
        dtype=np.float64,
    )
    vectors[:, 6] = (vectors[:, 6] + 1.0) * 0.5
    centre = np.mean(vectors, axis=0)
    first = int(np.argmax(np.sum(np.square(vectors - centre), axis=1)))
    selected = [first]
    nearest = np.sum(np.square(vectors - vectors[first]), axis=1)
    nearest[first] = -1.0
    while len(selected) < count:
        chosen = int(np.argmax(nearest))
        selected.append(chosen)
        candidate_distance = np.sum(np.square(vectors - vectors[chosen]), axis=1)
        nearest = np.minimum(nearest, candidate_distance)
        nearest[selected] = -1.0
    return tuple(int(candidates[index]) for index in selected)


_METRIC_NAMES = frozenset(
    {
        "largest_landmass_share",
        "second_landmass_share",
        "land_component_count",
        "island_area_share",
        "internal_divergent_fraction",
        "internal_convergent_fraction",
        "active_margin_fraction",
        "enclosed_sea_fraction",
        "shelf_fraction",
        "mean_absolute_land_latitude",
    }
)


def _metrics(values: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError("metrics must be a mapping")
    missing = _METRIC_NAMES.difference(values)
    if missing:
        raise ValueError("metrics missing: " + ", ".join(sorted(missing)))
    result: dict[str, float] = {}
    for name in _METRIC_NAMES:
        try:
            number = float(values[name])
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be numeric") from error
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        result[name] = number
    return result


def classify_world_morphology(metrics: Mapping[str, float]) -> str:
    """Compose an open-ended description from observable map metrics."""

    value = _metrics(metrics)
    convergent = value["internal_convergent_fraction"]
    divergent = value["internal_divergent_fraction"]
    largest = value["largest_landmass_share"]
    second = value["second_landmass_share"]
    components = int(round(value["land_component_count"]))

    if convergent >= max(0.10, divergent * 1.30):
        stage = "聚合期"
    elif divergent >= max(0.10, convergent * 1.30):
        stage = "裂解期"
    elif largest >= 0.62:
        stage = "稳定超大陆期"
    else:
        stage = "最大离散期"

    if largest >= 0.72:
        structure = "单一超大陆"
    elif largest >= 0.48 and second >= 0.16:
        structure = "双核心大陆"
    elif largest >= 0.48:
        structure = "主大陆—外围陆块"
    elif components >= 7 and value["island_area_share"] >= 0.18:
        structure = "群岛—微大陆"
    elif components >= 5:
        structure = "分散多大陆"
    else:
        structure = "少数大型大陆"

    modifiers: list[str] = []
    if value["island_area_share"] >= 0.18:
        modifiers.append("岛弧密集")
    if value["enclosed_sea_fraction"] >= 0.09:
        modifiers.append("内海发达")
    if value["active_margin_fraction"] >= 0.55:
        modifiers.append("活动边缘占优")
    elif value["shelf_fraction"] >= 0.35:
        modifiers.append("宽陆架")
    if value["mean_absolute_land_latitude"] >= 0.55:
        modifiers.append("高纬陆地偏重")

    return "·".join((stage, structure, *modifiers))


__all__ = [
    "WorldMorphology",
    "classify_world_morphology",
    "representative_morphology_seeds",
    "sample_world_morphology",
    "vars_from_slots",
]
