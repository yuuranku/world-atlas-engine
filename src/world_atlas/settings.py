"""Strict, reproducible settings for the post-terrain world pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping


_PLANET_KEYS = {
    "solarConstantWm2",
    "radiusKm",
    "gravityMps2",
    "rotationPeriodHours",
    "rotationDirection",
    "orbitalPeriodDays",
    "axialTiltDegrees",
}
_SOCIETY_INTEGER_KEYS = {
    "settlementCount",
    "civilizationCount",
    "minimumCivilizationCount",
    "languageCount",
    "religionCount",
    "stateCount",
    "populationMin",
    "populationMax",
}
_SOCIETY_KEYS = _SOCIETY_INTEGER_KEYS | {"frontierTargetShare"}
_TECHNOLOGY_ERAS = frozenset({"tribal", "ancient", "medieval", "early-modern", "preindustrial", "industrial", "contemporary"})


def _strict_mapping(value: object, *, name: str, keys: set[str]) -> dict:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(keys))}")
    return dict(value)


def _finite_number(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError(f"{name} must be an unsigned 32-bit integer")
    return value


@dataclass(frozen=True, slots=True)
class WorldSettings:
    human_seed: int
    naming_seed: int
    planet: Mapping[str, float | str]
    society: Mapping[str, int | float]
    technology_era: str
    travel_capabilities: tuple[str, ...]

    def document(self) -> dict[str, object]:
        return {
            "schema": "world-atlas-world-settings-v3",
            "humanSeed": self.human_seed,
            "namingSeed": self.naming_seed,
            "planet": dict(self.planet),
            "society": dict(self.society),
            "technologyEra": self.technology_era,
            "travelCapabilities": list(self.travel_capabilities),
        }


def load_world_settings(path: str | Path) -> WorldSettings:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    root = _strict_mapping(
        document,
        name="world settings",
        keys={
            "schema",
            "humanSeed",
            "namingSeed",
            "planet",
            "society",
            "technologyEra",
            "travelCapabilities",
        },
    )
    if root["schema"] != "world-atlas-world-settings-v3":
        raise ValueError("unknown world settings schema")
    technology_era = root["technologyEra"]
    capabilities = root['travelCapabilities']
    if (not isinstance(capabilities, list) or any(not isinstance(value,str) or value not in {'magic-flight','dragon'} for value in capabilities)
            or len(set(capabilities)) != len(capabilities)):
        raise ValueError('travelCapabilities must list unique supported capabilities: magic-flight, dragon')
    if technology_era not in _TECHNOLOGY_ERAS:
        raise ValueError(
            "technologyEra must be one of: "
            + ", ".join(sorted(_TECHNOLOGY_ERAS))
        )

    planet = _strict_mapping(root["planet"], name="planet", keys=_PLANET_KEYS)
    direction = planet["rotationDirection"]
    if direction not in {"prograde", "retrograde"}:
        raise ValueError("planet.rotationDirection must be prograde or retrograde")
    normalized_planet: dict[str, float | str] = {
        "solarConstantWm2": _finite_number(
            planet["solarConstantWm2"], name="planet.solarConstantWm2", positive=True
        ),
        "radiusKm": _finite_number(planet["radiusKm"], name="planet.radiusKm", positive=True),
        "gravityMps2": _finite_number(
            planet["gravityMps2"], name="planet.gravityMps2", positive=True
        ),
        "rotationPeriodHours": _finite_number(
            planet["rotationPeriodHours"], name="planet.rotationPeriodHours", positive=True
        ),
        "rotationDirection": direction,
        "orbitalPeriodDays": _finite_number(
            planet["orbitalPeriodDays"], name="planet.orbitalPeriodDays", positive=True
        ),
        "axialTiltDegrees": _finite_number(
            planet["axialTiltDegrees"], name="planet.axialTiltDegrees"
        ),
    }
    if not 0.0 <= float(normalized_planet["axialTiltDegrees"]) <= 90.0:
        raise ValueError("planet.axialTiltDegrees must lie in [0, 90]")

    society = _strict_mapping(root["society"], name="society", keys=_SOCIETY_KEYS)
    normalized_society: dict[str, int | float] = {
        key: _positive_integer(society[key], name=f"society.{key}")
        for key in _SOCIETY_INTEGER_KEYS
    }
    frontier_share = _finite_number(
        society["frontierTargetShare"], name="society.frontierTargetShare"
    )
    if not 0.0 <= frontier_share < 1.0:
        raise ValueError("society.frontierTargetShare must lie in [0, 1)")
    normalized_society["frontierTargetShare"] = frontier_share
    if normalized_society["minimumCivilizationCount"] > normalized_society["civilizationCount"]:
        raise ValueError("minimumCivilizationCount cannot exceed civilizationCount")
    if normalized_society["populationMin"] > normalized_society["populationMax"]:
        raise ValueError("populationMin cannot exceed populationMax")

    return WorldSettings(
        human_seed=_seed(root["humanSeed"], name="humanSeed"),
        naming_seed=_seed(root["namingSeed"], name="namingSeed"),
        planet=normalized_planet,
        society=normalized_society,
        technology_era=technology_era,
        travel_capabilities=tuple(capabilities),
    )
