"""Strict immutable data contracts for downstream society generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def immutable_array(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PopulationLayers:
    """Normalized population support and six-level display partition."""

    population_weight: np.ndarray
    population_band: np.ndarray
    population_min: int = 450_000_000
    population_max: int = 600_000_000

    def __post_init__(self) -> None:
        weights = np.asarray(self.population_weight)
        bands = np.asarray(self.population_band)
        if weights.ndim != 2:
            raise ValueError("population_weight must be a two-dimensional array")
        if bands.shape != weights.shape:
            raise ValueError("population_band must match population_weight shape")
        if weights.dtype != np.float32:
            raise ValueError("population_weight must have dtype float32")
        if bands.dtype != np.uint8:
            raise ValueError("population_band must have dtype uint8")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("population_weight must contain finite non-negative values")
        if np.any(bands > 6):
            raise ValueError("population_band values must be in the range 0..6")
        if isinstance(self.population_min, bool) or isinstance(self.population_max, bool):
            raise ValueError("population bounds must be integers")
        if int(self.population_min) < 0 or int(self.population_min) > int(self.population_max):
            raise ValueError("population_min must not exceed population_max")
        object.__setattr__(
            self,
            "population_weight",
            immutable_array(weights, dtype=np.dtype(np.float32)),
        )
        object.__setattr__(
            self,
            "population_band",
            immutable_array(bands, dtype=np.dtype(np.uint8)),
        )


@dataclass(frozen=True, slots=True)
class Settlement:
    """One physically resolved population and transport node."""

    identifier: str
    name: str
    row: int
    column: int
    tier: str
    site_type: str
    score: float
    population_min: int
    population_max: int
    holy_religion_identifier: int | None = None

    def __post_init__(self) -> None:
        if not self.identifier.strip() or not self.name.strip():
            raise ValueError("settlement identifiers and names must be non-empty")
        if self.row < 0 or self.column < 0:
            raise ValueError("settlement coordinates must be non-negative")
        if self.tier not in {"metropolis", "city", "town", "site"}:
            raise ValueError("unsupported settlement tier")
        if self.site_type not in {
            "river-city",
            "port",
            "lake-port",
            "market",
            "pass",
            "oasis",
            "fortress",
            "island-port",
        }:
            raise ValueError("unsupported settlement site type")
        if self.population_min < 0 or self.population_min > self.population_max:
            raise ValueError("invalid settlement population range")
        if (
            self.holy_religion_identifier is not None
            and self.holy_religion_identifier < 1
        ):
            raise ValueError("holy religion identifier must be positive")


@dataclass(frozen=True, slots=True)
class TransportRoute:
    """One causally routed connection between settlement nodes."""

    identifier: str
    mode: str
    importance: str
    source_settlement_id: str
    target_settlement_id: str | None
    path: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.identifier.strip() or not self.source_settlement_id.strip():
            raise ValueError("transport route identifiers must be non-empty")
        if self.mode not in {"road", "river", "sea"}:
            raise ValueError("unsupported transport mode")
        if self.importance not in {"trunk", "regional", "local"}:
            raise ValueError("unsupported transport importance")
        if self.target_settlement_id is not None and not self.target_settlement_id.strip():
            raise ValueError("target settlement identifier must be non-empty")
        normalized = tuple((float(column), float(row)) for column, row in self.path)
        if len(normalized) < 2 or any(
            not np.isfinite(column) or not np.isfinite(row)
            for column, row in normalized
        ):
            raise ValueError("transport paths require at least two finite points")
        object.__setattr__(self, "path", normalized)


@dataclass(frozen=True, slots=True)
class Bridge:
    """One explicit engineered crossing on a represented river and road."""

    identifier: str
    row: int
    column: int
    route_identifier: str
    river_order: int
    importance: str

    def __post_init__(self) -> None:
        if not self.identifier.strip() or not self.route_identifier.strip():
            raise ValueError("bridge identifiers must be non-empty")
        if self.row < 0 or self.column < 0:
            raise ValueError("bridge coordinates must be non-negative")
        if self.river_order < 1:
            raise ValueError("bridges must cross a represented river")
        if self.importance not in {"trunk", "regional", "local"}:
            raise ValueError("unsupported bridge importance")


@dataclass(frozen=True, slots=True)
class TransportLayers:
    """Physical transport graph and normalized accessibility field."""

    accessibility: np.ndarray
    routes: tuple[TransportRoute, ...]
    bridges: tuple[Bridge, ...] = ()

    def __post_init__(self) -> None:
        accessibility = np.asarray(self.accessibility)
        if accessibility.ndim != 2 or accessibility.dtype != np.float32:
            raise ValueError("transport accessibility must be a two-dimensional float32 array")
        if np.any(~np.isfinite(accessibility)) or np.any(
            (accessibility < 0.0) | (accessibility > 1.0)
        ):
            raise ValueError("transport accessibility values must be in the range 0..1")
        identifiers = [route.identifier for route in self.routes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("transport route identifiers must be unique")
        route_by_identifier = {route.identifier: route for route in self.routes}
        bridge_identifiers = [bridge.identifier for bridge in self.bridges]
        if len(bridge_identifiers) != len(set(bridge_identifiers)):
            raise ValueError("bridge identifiers must be unique")
        for bridge in self.bridges:
            if bridge.row >= accessibility.shape[0] or bridge.column >= accessibility.shape[1]:
                raise ValueError("bridge lies outside transport grid")
            route = route_by_identifier.get(bridge.route_identifier)
            if route is None or route.mode != "road":
                raise ValueError("bridges must reference a road route")
        object.__setattr__(
            self,
            "accessibility",
            immutable_array(accessibility, dtype=np.dtype(np.float32)),
        )


@dataclass(frozen=True, slots=True)
class Civilization:
    identifier: int
    name: str
    name_family: str
    core_settlement_id: str
    internal_diversity: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.identifier < 1:
            raise ValueError("civilization identifier must be positive")
        if not self.name.strip():
            raise ValueError("civilization name must be non-empty")
        if not self.name_family.strip():
            raise ValueError("civilization name_family must be non-empty")
        if len(self.internal_diversity) < 3:
            raise ValueError("civilizations require at least three internal contrasts")


@dataclass(frozen=True, slots=True)
class Language:
    identifier: int
    name: str
    name_family: str
    family_identifier: int
    core_settlement_id: str

    def __post_init__(self) -> None:
        if self.identifier < 1 or self.family_identifier < 1:
            raise ValueError("language identifiers must be positive")
        if not self.name.strip():
            raise ValueError("language name must be non-empty")
        if not self.name_family.strip():
            raise ValueError("language name_family must be non-empty")


@dataclass(frozen=True, slots=True)
class CultureLayers:
    civilization_id: np.ndarray
    civilization_influence: np.ndarray
    language_family_id: np.ndarray
    language_id: np.ndarray
    language_contact: np.ndarray
    civilizations: tuple[Civilization, ...]
    languages: tuple[Language, ...]

    def __post_init__(self) -> None:
        civilization = np.asarray(self.civilization_id)
        expected_shape = civilization.shape
        arrays = {
            "civilization_id": (civilization, np.dtype(np.int16)),
            "civilization_influence": (
                np.asarray(self.civilization_influence),
                np.dtype(np.uint8),
            ),
            "language_family_id": (np.asarray(self.language_family_id), np.dtype(np.int16)),
            "language_id": (np.asarray(self.language_id), np.dtype(np.int16)),
            "language_contact": (np.asarray(self.language_contact), np.dtype(np.bool_)),
        }
        for name, (value, dtype) in arrays.items():
            if value.shape != expected_shape or value.dtype != dtype:
                raise ValueError(f"{name} must have shape {expected_shape} and dtype {dtype}")
            object.__setattr__(self, name, immutable_array(value, dtype=dtype))
        if np.any(np.asarray(self.civilization_influence) > 3):
            raise ValueError("civilization_influence values must be in the range 0..3")


@dataclass(frozen=True, slots=True)
class Religion:
    """One organized religious tradition with a concrete pilgrimage hearth."""

    identifier: int
    name: str
    tradition: str
    holy_settlement_id: str
    origin_civilization_identifier: int
    worldview: str
    moral_ideal: str
    institution: str

    def __post_init__(self) -> None:
        if self.identifier < 1 or self.origin_civilization_identifier < 1:
            raise ValueError("religion identifiers must be positive")
        if any(
            not value.strip()
            for value in (
                self.name,
                self.tradition,
                self.holy_settlement_id,
                self.worldview,
                self.moral_ideal,
                self.institution,
            )
        ):
            raise ValueError("religion records require non-empty text")


@dataclass(frozen=True, slots=True)
class ReligionLayers:
    religion_id: np.ndarray
    religions: tuple[Religion, ...]

    def __post_init__(self) -> None:
        identifiers = {item.identifier for item in self.religions}
        if identifiers != set(range(1, len(self.religions) + 1)):
            raise ValueError("religion identifiers must be contiguous")
        holy_settlements = [item.holy_settlement_id for item in self.religions]
        if len(holy_settlements) != len(set(holy_settlements)):
            raise ValueError("each religion requires a distinct holy city")
        values = np.asarray(self.religion_id)
        if values.ndim != 2 or values.dtype != np.int16:
            raise ValueError("religion_id must be a two-dimensional int16 array")
        positive = set(int(value) for value in np.unique(values[values > 0]))
        if positive != identifiers:
            raise ValueError("every religion must own map cells")
        object.__setattr__(
            self,
            "religion_id",
            immutable_array(values, dtype=np.dtype(np.int16)),
        )


@dataclass(frozen=True, slots=True)
class NameLexicon:
    major_country_names: tuple[str, ...]
    major_country_roles: tuple[str, ...]
    minor_country_names: tuple[str, ...]
    place_roots: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.major_country_names) != 19:
            raise ValueError("name lexicon requires exactly 19 major state names")
        if len(self.minor_country_names) < 120:
            raise ValueError("name lexicon does not contain enough minor state names")
        if len(self.major_country_roles) != len(self.major_country_names):
            raise ValueError("major country names and placement roles must align")
        for values in (
            self.major_country_names,
            self.major_country_roles,
            self.minor_country_names,
            self.place_roots,
        ):
            if not values or any(not item.strip() for item in values):
                raise ValueError("name lexicon entries must be non-empty")
@dataclass(frozen=True, slots=True)
class GeographicFeature:
    """One world-scale named physical feature anchored to the canonical grid."""

    identifier: str
    feature_type: str
    name: str
    row: int
    column: int
    language_identifier: int
    tier: str

    def __post_init__(self) -> None:
        if self.feature_type not in {
            "mountain",
            "peak",
            "river",
            "lake",
            "sea",
            "inland-sea",
            "bay",
            "strait",
            "island",
            "island-group",
            "plain",
            "plateau",
            "basin",
        }:
            raise ValueError("unsupported geographic feature type")
        if self.row < 0 or self.column < 0:
            raise ValueError("geographic feature coordinates must be non-negative")
        if self.language_identifier < 1:
            raise ValueError("geographic features require a positive language identifier")
        if self.tier not in {"major", "secondary", "detail"}:
            raise ValueError("invalid geographic feature tier")
        if not self.identifier.strip() or not self.name.strip():
            raise ValueError("geographic feature identifiers and names must be non-empty")


@dataclass(frozen=True, slots=True)
class State:
    """Country identity and territory, deliberately free of government form."""

    identifier: int
    name: str
    core_settlement_id: str
    size_class: str
    population_min: int
    population_max: int
    civilization_identifier: int
    language_identifier: int

    def __post_init__(self) -> None:
        if self.identifier < 1:
            raise ValueError("state identifier must be positive")
        if self.size_class not in {"major", "medium", "small"}:
            raise ValueError("invalid state size class")
        if self.population_min < 0 or self.population_min > self.population_max:
            raise ValueError("invalid state population range")


@dataclass(frozen=True, slots=True)
class GovernmentForm:
    """Reusable institution template that is not owned by any country."""

    identifier: int
    key: str
    name: str
    description: str

    def __post_init__(self) -> None:
        if self.identifier < 1:
            raise ValueError("government-form identifier must be positive")
        if not self.key.strip() or not self.name.strip() or not self.description.strip():
            raise ValueError("government-form text must be non-empty")


@dataclass(frozen=True, slots=True)
class PoliticalEntity:
    """Current combination of one country identity and one government form."""

    identifier: int
    country_identifier: int
    government_form_identifier: int
    formal_name: str

    def __post_init__(self) -> None:
        if min(self.identifier, self.country_identifier, self.government_form_identifier) < 1:
            raise ValueError("political-entity identifiers must be positive")
        if not self.formal_name.strip():
            raise ValueError("political-entity formal_name must be non-empty")


@dataclass(frozen=True, slots=True)
class FrontierGroup:
    """A mobile or locally rooted people outside stable state administration.

    The point is a label anchor, not a territorial claim.  Frontier peoples
    therefore remain visible without inventing a second set of hard borders.
    """

    identifier: str
    name: str
    row: int
    column: int
    livelihood: str
    organization: str
    civilization_identifier: int
    language_identifier: int
    tier: str

    def __post_init__(self) -> None:
        if not self.identifier.strip() or not self.name.strip():
            raise ValueError("frontier-group identifiers and names must be non-empty")
        if self.row < 0 or self.column < 0:
            raise ValueError("frontier-group coordinates must be non-negative")
        if self.livelihood not in {
            "pastoral",
            "foraging",
            "maritime",
            "agropastoral",
            "highland",
            "riverine",
        }:
            raise ValueError("unsupported frontier-group livelihood")
        if self.organization not in {
            "khan-court",
            "confederacy",
            "tribes",
            "sea-clans",
            "boat-people",
            "island-clans",
            "hunters",
            "forest-clans",
            "village-league",
            "hill-tribes",
            "clan-league",
            "river-clans",
            "fishers",
        }:
            raise ValueError("unsupported frontier-group organization")
        if min(self.civilization_identifier, self.language_identifier) < 1:
            raise ValueError("frontier groups require positive cultural references")
        if self.tier not in {"major", "secondary"}:
            raise ValueError("unsupported frontier-group tier")


@dataclass(frozen=True, slots=True)
class PoliticalLayers:
    state_id: np.ndarray
    frontier: np.ndarray
    states: tuple[State, ...]
    government_forms: tuple[GovernmentForm, ...]
    political_entities: tuple[PoliticalEntity, ...]
    frontier_groups: tuple[FrontierGroup, ...]

    def __post_init__(self) -> None:
        state = np.asarray(self.state_id)
        frontier = np.asarray(self.frontier)
        if state.ndim != 2 or state.dtype != np.int16:
            raise ValueError("state_id must be a two-dimensional int16 array")
        if frontier.shape != state.shape or frontier.dtype != np.bool_:
            raise ValueError("frontier must be a matching bool array")
        identifiers = {item.identifier for item in self.states}
        if identifiers != set(range(1, len(self.states) + 1)):
            raise ValueError("state identifiers must be contiguous")
        government_identifiers = {item.identifier for item in self.government_forms}
        if government_identifiers != set(range(1, len(self.government_forms) + 1)):
            raise ValueError("government-form identifiers must be contiguous")
        entity_identifiers = {item.identifier for item in self.political_entities}
        if entity_identifiers != identifiers:
            raise ValueError("every country requires exactly one current political entity")
        if {item.country_identifier for item in self.political_entities} != identifiers:
            raise ValueError("political entities must each combine one generated country")
        if any(
            item.government_form_identifier not in government_identifiers
            for item in self.political_entities
        ):
            raise ValueError("political entities must reference a reusable government form")
        frontier_group_identifiers = [item.identifier for item in self.frontier_groups]
        frontier_group_names = [item.name for item in self.frontier_groups]
        if len(frontier_group_identifiers) != len(set(frontier_group_identifiers)):
            raise ValueError("frontier-group identifiers must be unique")
        if len(frontier_group_names) != len(set(frontier_group_names)):
            raise ValueError("frontier-group names must be unique")
        for item in self.frontier_groups:
            if item.row >= state.shape[0] or item.column >= state.shape[1]:
                raise ValueError("frontier-group anchor lies outside political grid")
            if not bool(frontier[item.row, item.column]):
                raise ValueError("frontier-group anchor must lie in frontier land")
        object.__setattr__(self, "state_id", immutable_array(state, dtype=np.dtype(np.int16)))
        object.__setattr__(self, "frontier", immutable_array(frontier, dtype=np.dtype(np.bool_)))


@dataclass(frozen=True, slots=True)
class Province:
    """One institutional country subdivision around a settlement core."""

    identifier: int
    name: str
    state_identifier: int
    core_settlement_id: str
    region_type: str
    administrative_system: str
    administrative_function: str
    administrative_rank: str
    population_density_class: str
    area_cells: int

    def __post_init__(self) -> None:
        if self.identifier < 1 or self.state_identifier < 1:
            raise ValueError("province identifiers must be positive")
        if not self.name.strip() or not self.core_settlement_id.strip():
            raise ValueError("province names and core settlements must be non-empty")
        if self.region_type not in {
            "archipelago",
            "coastal",
            "lake",
            "river",
            "oasis",
            "highland",
            "plain",
            "interior",
        }:
            raise ValueError("unsupported province region type")
        if self.administrative_system not in {
            "central-bureaucracy",
            "feudal-vassalage",
            "civic-administration",
            "confederal-territory",
            "estate-administration",
        }:
            raise ValueError("unsupported province administrative system")
        if self.administrative_function not in {
            "capital",
            "civil",
            "military",
            "frontier",
            "maritime",
            "vassal",
            "crown",
            "pastoral",
            "civic",
        }:
            raise ValueError("unsupported province administrative function")
        if not self.administrative_rank.strip():
            raise ValueError("province administrative rank must be non-empty")
        if self.population_density_class not in {"dense", "settled", "sparse"}:
            raise ValueError("unsupported province population-density class")
        if self.area_cells < 1:
            raise ValueError("province area_cells must be positive")


@dataclass(frozen=True, slots=True)
class ProvinceLayers:
    province_id: np.ndarray
    provinces: tuple[Province, ...]

    def __post_init__(self) -> None:
        province = np.asarray(self.province_id)
        if province.ndim != 2 or province.dtype != np.int32:
            raise ValueError("province_id must be a two-dimensional int32 array")
        identifiers = {item.identifier for item in self.provinces}
        if identifiers != set(range(1, len(self.provinces) + 1)):
            raise ValueError("province identifiers must be contiguous")
        positive = set(int(value) for value in np.unique(province[province > 0]))
        if positive != identifiers:
            raise ValueError("every province record must own map cells")
        object.__setattr__(
            self,
            "province_id",
            immutable_array(province, dtype=np.dtype(np.int32)),
        )


@dataclass(frozen=True, slots=True)
class SocietyLayers:
    """Complete downstream society model derived from one WorldGrid."""

    population: PopulationLayers
    settlements: tuple[Settlement, ...]
    transport: TransportLayers
    cultures: CultureLayers
    religions: ReligionLayers
    geographic_features: tuple[GeographicFeature, ...]
    politics: PoliticalLayers
    provinces: ProvinceLayers

    def __post_init__(self) -> None:
        shape = self.population.population_weight.shape
        if self.cultures.civilization_id.shape != shape:
            raise ValueError("culture layers must match population shape")
        if self.religions.religion_id.shape != shape:
            raise ValueError("religion layers must match population shape")
        if self.transport.accessibility.shape != shape:
            raise ValueError("transport layers must match population shape")
        if self.politics.state_id.shape != shape:
            raise ValueError("political layers must match population shape")
        if self.provinces.province_id.shape != shape:
            raise ValueError("province layers must match population shape")
        state_by_province = {
            item.identifier: item.state_identifier for item in self.provinces.provinces
        }
        for identifier, state_identifier in state_by_province.items():
            if np.any(
                self.politics.state_id[self.provinces.province_id == identifier]
                != state_identifier
            ):
                raise ValueError("province cells must remain inside their parent country")
