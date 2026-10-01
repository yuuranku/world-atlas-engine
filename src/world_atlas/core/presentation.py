"""Deterministic, lazy presentation assets for atlas review clients.

The simulation layers remain the authority for where settlements can exist.  This
module only turns already-resolved settlement nodes into compact *recipes* that a
browser can expand into a detailed city after the user has zoomed in.  It never
creates DOM, SVG, or geometry for every city up front.

The output is deliberately independent from the HTML renderer so a native client,
a Three.js globe, or a static map renderer can all consume the same recipes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from hashlib import blake2b, sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy import ndimage

from .model import WorldGrid
from .city_transport import CityCorridor, CityTransportIndex, refined_land_routes
from .society.model import SocietyLayers, Settlement


PRESENTATION_SCHEMA = "world-atlas-presentation-v1"
PRESENTATION_VERSION = 1
CITY_RECIPES_GLOBAL = "WorldAtlasCityRecipes"


@dataclass(frozen=True, slots=True)
class _EraProfile:
    """Density and street-form assumptions for one technology era."""

    density_per_km2: float
    core_bonus: int
    grid_bias: float
    landmark: str


_ERA_PROFILES: dict[str, _EraProfile] = {
    "tribal": _EraProfile(780.0, 0, 0.03, "gathering-ground"),
    "ancient": _EraProfile(1_550.0, 0, 0.12, "forum"),
    "medieval": _EraProfile(2_250.0, 0, 0.17, "market-hall"),
    "early-modern": _EraProfile(2_850.0, 1, 0.33, "exchange-hall"),
    "preindustrial": _EraProfile(3_200.0, 0, 0.21, "market-square"),
    "industrial": _EraProfile(3_450.0, 1, 0.52, "rail-terminal"),
    "contemporary": _EraProfile(4_100.0, 2, 0.65, "civic-centre"),
}
_ERA_CLUSTER_THRESHOLDS = {
    "tribal": 42.0,
    "ancient": 54.0,
    "medieval": 64.0,
    "early-modern": 70.0,
    "preindustrial": 76.0,
    "industrial": 120.0,
    "contemporary": 145.0,
}
_WATER_KIND_BY_CODE = {1: "ocean", 2: "lake", 3: "inland-sea"}
_CULTURAL_STYLE_FAMILIES = (
    "courtyard",
    "arcaded",
    "terraced",
    "timber-frame",
    "stone-masonry",
    "canal-side",
    "walled-compound",
    "vernacular-mixed",
)


@dataclass(frozen=True, slots=True)
class CityRecipe:
    """A compact, physical-context-aware specification for one city drawing."""

    identifier: str
    name: str
    row: int
    column: int
    center_row: float
    center_column: float
    seed: int
    tier: str
    site_type: str
    era: str
    population_min: int
    population_max: int
    population_estimate: int
    footprint_km2: float
    footprint_grid_cells: float
    radius_km: float
    radius_rows: float
    radius_columns: float
    grid_cell_height_km: float
    grid_cell_width_km: float
    footprint_bounds: tuple[float, float, float, float]
    core_count: int
    core_zones: tuple["CityCoreZone", ...]
    morphology: str
    street_pattern: str
    building_pattern: str
    orientation_degrees: int
    block_aspect: float
    street_irregularity: float
    water_kind: str
    nearest_water_kind: str | None
    water_distance_cells: float | None
    river_distance_cells: float | None
    river_order: int
    water_bearing_degrees: int | None
    river_bearing_degrees: int | None
    slope_class: str
    slope: float
    elevation: float
    maximum_buildable_slope: float
    buildable_cells: tuple[tuple[int, int], ...]
    blocked_cells: tuple[tuple[int, int], ...]
    buildable_bounds: tuple[int, int, int, int]
    civilization_identifier: int
    language_identifier: int
    religion_identifier: int
    state_identifier: int
    cultural_style: str
    landmarks: tuple[str, ...]
    road_connections: int
    rail_connections: int
    river_connections: int
    sea_connections: int
    bridge_count: int
    transport_interfaces: tuple[str, ...]
    transport_corridors: tuple[CityCorridor, ...]
    cluster_identifier: str | None = None
    cluster_member_count: int = 1
    cluster_role: str = "standalone"

    def document(self) -> dict[str, object]:
        """Return only JSON-native values for a browser or an API client."""

        return {
            "id": self.identifier,
            "name": self.name,
            # Raster identity remains in anchorCell. All displayed geometry
            # shares this refined dry-cell location, including coastal access.
            "location": {"row": self.center_row, "column": self.center_column},
            "anchorCell": {"row": self.row, "column": self.column},
            "seed": self.seed,
            "tier": self.tier,
            "siteType": self.site_type,
            "era": self.era,
            "population": {
                "minimum": self.population_min,
                "maximum": self.population_max,
                "estimate": self.population_estimate,
            },
            "urban": {
                # This is a planning target, not a claim about the eventual
                # visible land area.  The browser must intersect its urban
                # fabric with terrain.buildable.allowedCells before drawing.
                "targetFootprintKm2": self.footprint_km2,
                "targetEquivalentGridCells": self.footprint_grid_cells,
                "areaSemantics": "target-before-native-terrain-mask",
                "radiusKm": self.radius_km,
                "coordinateSpace": "world-grid-cells",
                "bounds": {
                    "north": self.footprint_bounds[0],
                    "west": self.footprint_bounds[1],
                    "south": self.footprint_bounds[2],
                    "east": self.footprint_bounds[3],
                },
                "radiusRows": self.radius_rows,
                "radiusColumns": self.radius_columns,
                "gridCellKilometres": {
                    "row": self.grid_cell_height_km,
                    "column": self.grid_cell_width_km,
                },
                "streetWidthsMetres": _street_widths_metres(self.era, self.tier),
                "clipToLand": True,
                "coreCount": self.core_count,
                "coreZones": [zone.document() for zone in self.core_zones],
                "cluster": {
                    "id": self.cluster_identifier,
                    "memberCount": self.cluster_member_count,
                    "role": self.cluster_role,
                },
            },
            "morphology": {
                "kind": self.morphology,
                "streetPattern": self.street_pattern,
                "buildingPattern": self.building_pattern,
                "orientationDegrees": self.orientation_degrees,
                "blockAspect": self.block_aspect,
                "streetIrregularity": self.street_irregularity,
            },
            "water": {
                "kind": self.water_kind,
                "nearestWaterKind": self.nearest_water_kind,
                "distanceCells": self.water_distance_cells,
                "riverDistanceCells": self.river_distance_cells,
                "riverOrder": self.river_order,
                "waterBearingDegrees": self.water_bearing_degrees,
                "riverBearingDegrees": self.river_bearing_degrees,
            },
            "terrain": {
                "slopeClass": self.slope_class,
                "slope": self.slope,
                "elevation": self.elevation,
                "constraints": {
                    "avoidWater": True,
                    "avoidRiverChannel": True,
                    "maximumBuildableSlope": self.maximum_buildable_slope,
                },
                "buildable": {
                    "coordinateSpace": "world-grid-cells",
                    "precision": "native-grid-cell",
                    "bounds": {
                        "north": self.buildable_bounds[0],
                        "west": self.buildable_bounds[1],
                        "south": self.buildable_bounds[2],
                        "east": self.buildable_bounds[3],
                    },
                    "allowedCells": [list(cell) for cell in self.buildable_cells],
                    "blockedCells": [list(cell) for cell in self.blocked_cells],
                    "allowedCellCount": len(self.buildable_cells),
                    "blockedCellCount": len(self.blocked_cells),
                    "clipCityGeometry": True,
                },
            },
            "culture": {
                "civilizationId": self.civilization_identifier,
                "languageId": self.language_identifier,
                "religionId": self.religion_identifier,
                "stateId": self.state_identifier,
                "style": self.cultural_style,
            },
            "landmarks": list(self.landmarks),
            "transport": {
                "roadConnections": self.road_connections,
                "railConnections": self.rail_connections,
                "riverConnections": self.river_connections,
                "seaConnections": self.sea_connections,
                "bridgeCount": self.bridge_count,
                "interfaces": list(self.transport_interfaces),
                "corridors": [corridor.document() for corridor in self.transport_corridors],
            },
        }


@dataclass(frozen=True, slots=True)
class CityCoreZone:
    """One small, bounded centre inside a city footprint in world-grid space."""

    identifier: str
    role: str
    row: float
    column: float
    radius_km: float

    def document(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "role": self.role,
            "row": self.row,
            "column": self.column,
            "radiusKm": self.radius_km,
        }


@dataclass(frozen=True, slots=True)
class UrbanCluster:
    """A compact agglomeration record for viewports that draw grouped cities."""

    identifier: str
    core_settlement_id: str
    member_settlement_ids: tuple[str, ...]
    population_estimate: int

    def document(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "coreSettlementId": self.core_settlement_id,
            "memberSettlementIds": list(self.member_settlement_ids),
            "populationEstimate": self.population_estimate,
        }


@dataclass(frozen=True, slots=True)
class PresentationAssets:
    """Reusable manifest and lazy city recipe payload for one world build."""

    grid_digest: str
    society_digest: str
    human_seed: int
    world_seed: int
    era: str
    engine_version: str
    world_name: str
    city_recipes: tuple[CityRecipe, ...]
    urban_clusters: tuple[UrbanCluster, ...]

    def manifest_document(self) -> dict[str, object]:
        """Return the authoritative JSON manifest, including every recipe."""

        return {
            "schema": PRESENTATION_SCHEMA,
            "version": PRESENTATION_VERSION,
            "engineVersion": self.engine_version,
            "gridDigest": self.grid_digest,
            "societyDigest": self.society_digest,
            "humanSeed": self.human_seed,
            "worldSeed": self.world_seed,
            "era": self.era,
            "worldName": self.world_name,
            "cityRecipes": [recipe.document() for recipe in self.city_recipes],
            "urbanClusters": [cluster.document() for cluster in self.urban_clusters],
            "lazyAssets": {
                "cityRecipesScript": "city-recipes.js",
                "global": CITY_RECIPES_GLOBAL,
                "strategy": "load-on-zoom",
                "minimumScreenDiameter": 4,
                "preRenderedCityDom": False,
            },
        }

    def javascript_document(self) -> str:
        """Return a browser script containing data only, not generated city DOM."""

        payload = {
            "schema": PRESENTATION_SCHEMA,
            "version": PRESENTATION_VERSION,
            "gridDigest": self.grid_digest,
            "societyDigest": self.society_digest,
            "humanSeed": self.human_seed,
            "era": self.era,
            "lazy": {"strategy": "load-on-zoom", "minimumScreenDiameter": 4},
            "recipes": [recipe.document() for recipe in self.city_recipes],
            "urbanClusters": [cluster.document() for cluster in self.urban_clusters],
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return (
            "/* Lazy data payload; render city geometry only after a zoom request. */\n"
            "(function (global) {\n"
            "  \"use strict\";\n"
            f"  global.{CITY_RECIPES_GLOBAL} = Object.freeze({serialized});\n"
            "})(typeof window !== \"undefined\" ? window : globalThis);\n"
        )


@dataclass(frozen=True, slots=True)
class _GridContext:
    water_distance: np.ndarray
    river_distance: np.ndarray | None
    normalized_slope: np.ndarray
    cell_area_km2: np.ndarray


def resolve_city_era(grid: WorldGrid, era: str | None = None) -> str:
    """Resolve a stable presentation era from an explicit request or metadata."""

    requested = era
    if requested is None:
        profile = grid.metadata.get("worldProfile")
        requested = profile.get("technologyEra") if isinstance(profile, Mapping) else None
    if requested not in _ERA_PROFILES:
        raise ValueError(
            "city presentation requires a strict technologyEra; choose one of: "
            + ", ".join(sorted(_ERA_PROFILES))
        )
    return requested


def society_content_digest(society: SocietyLayers) -> str:
    """Hash every society field that changes city presentation semantics."""

    digest = sha256()
    metadata = {
        "settlements": [asdict(item) for item in society.settlements],
        "transportRoutes": [asdict(item) for item in society.transport.routes],
        "bridges": [asdict(item) for item in society.transport.bridges],
        "civilizations": [asdict(item) for item in society.cultures.civilizations],
        "languages": [asdict(item) for item in society.cultures.languages],
        "religions": [asdict(item) for item in society.religions.religions],
        "geographicFeatures": [asdict(item) for item in society.geographic_features],
        "states": [asdict(item) for item in society.politics.states],
        "governmentForms": [asdict(item) for item in society.politics.government_forms],
        "politicalEntities": [asdict(item) for item in society.politics.political_entities],
        "frontierGroups": [asdict(item) for item in society.politics.frontier_groups],
        "provinces": [asdict(item) for item in society.provinces.provinces],
        "populationRange": [
            society.population.population_min,
            society.population.population_max,
        ],
    }
    digest.update(b"society-metadata\0")
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for name, array in (
        ("populationWeight", society.population.population_weight),
        ("populationBand", society.population.population_band),
        ("civilizationId", society.cultures.civilization_id),
        ("civilizationInfluence", society.cultures.civilization_influence),
        ("languageFamilyId", society.cultures.language_family_id),
        ("languageId", society.cultures.language_id),
        ("languageContact", society.cultures.language_contact),
        ("religionId", society.religions.religion_id),
        ("stateId", society.politics.state_id),
        ("provinceId", society.provinces.province_id),
        ("frontier", society.politics.frontier),
        ("transportAccessibility", society.transport.accessibility),
    ):
        value = np.ascontiguousarray(np.asarray(array))
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def derive_presentation_assets(
    grid: WorldGrid,
    society: SocietyLayers,
    *,
    era: str | None = None,
    engine_version: str = "world-atlas-engine",
    settlement_locations: Mapping[str, tuple[float, float]] | None = None,
) -> PresentationAssets:
    """Derive deterministic, lazy client assets without changing simulation data.

    ``humanSeed`` and the settlement identifier are the only entropy source for
    the per-city visual variation.  Physical measurements choose viable urban
    forms, while culture and political arrays provide their semantic context.
    Supplied displayed anchors control all city geometry and transport access;
    a client must pass them when sharing an existing atlas scene.
    """

    human_seed = _human_seed(grid)
    resolved_era = resolve_city_era(grid, era)
    if not isinstance(engine_version, str) or not engine_version.strip():
        raise ValueError("engine_version must be a non-empty string")
    _validate_shapes(grid, society)
    context = _derive_grid_context(grid)
    capital_ids = {item.core_settlement_id for item in society.politics.states}
    identifiers = [settlement.identifier for settlement in society.settlements]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("city presentation requires unique settlement identifiers")
    if settlement_locations is not None:
        if set(settlement_locations) != set(identifiers):
            raise ValueError("city presentation requires a displayed anchor for every settlement")
        for settlement in society.settlements:
            location = settlement_locations[settlement.identifier]
            if (len(location) != 2 or not all(math.isfinite(value) for value in location)
                    or not 0<=location[0]<grid.shape[0] or not 0<=location[1]<grid.shape[1]):
                raise ValueError('displayed settlement anchors must lie within the world')
            if not settlement.row<=location[0]<settlement.row+1 or not settlement.column<=location[1]<settlement.column+1:
                if (settlement.site_type not in {'port','island-port','lake-port'}
                    or math.hypot(location[0]-settlement.row-.5,location[1]-settlement.column-.5)>3
                    or (society.politics.state_id[int(location[0]),int(location[1])]!=society.politics.state_id[settlement.row,settlement.column]
                        and not (society.politics.state_id[int(location[0]),int(location[1])]<=0 and grid.water[int(location[0]),int(location[1])]>0))):
                    raise ValueError('only a verified coastal port may refine its anchor outside its native land cell within the same polity')
    raw_recipes = tuple(
        _derive_city_recipe(
            grid,
            society,
            settlement,
            context,
            human_seed=human_seed,
            era=resolved_era,
            is_capital=settlement.identifier in capital_ids,
            location=settlement_locations[settlement.identifier] if settlement_locations is not None else None,
        )
        for settlement in sorted(
            society.settlements,
            key=lambda item: (item.identifier, item.row, item.column),
        )
    )
    locations = {recipe.identifier: (recipe.center_row, recipe.center_column) for recipe in raw_recipes}
    transport_index = CityTransportIndex(society.transport, grid.shape[1], corridors=(
        CityCorridor(route.identifier, route.mode, route.path)
        for route in refined_land_routes(society.transport.routes, locations)))
    connected_recipes = []
    for recipe in raw_recipes:
        summary = _transport_summary(transport_index, recipe.identifier, water_kind=recipe.water_kind)
        connected_recipes.append(replace(recipe,
            road_connections=summary["road"], rail_connections=summary["rail"],
            river_connections=summary["river"], sea_connections=summary["sea"],
            bridge_count=summary["bridges"], transport_interfaces=summary["interfaces"],
            transport_corridors=transport_index.corridors(recipe.footprint_bounds)))
    recipes, clusters = _assign_urban_clusters(grid, tuple(connected_recipes), era=resolved_era)
    profile = grid.metadata.get("worldProfile", {})
    world_name = str(profile.get("name", "Unnamed World")) if isinstance(profile, Mapping) else "Unnamed World"
    world_seed = _world_seed(grid, human_seed)
    return PresentationAssets(
        grid_digest=grid.content_digest(),
        society_digest=society_content_digest(society),
        human_seed=human_seed,
        world_seed=world_seed,
        era=resolved_era,
        engine_version=engine_version.strip(),
        world_name=world_name,
        city_recipes=recipes,
        urban_clusters=clusters,
    )


def write_presentation_assets(
    output_directory: str | Path,
    grid: WorldGrid,
    society: SocietyLayers,
    *,
    era: str | None = None,
    engine_version: str = "world-atlas-engine",
) -> PresentationAssets:
    """Write ``presentation.json`` and lazy ``city-recipes.js`` atomically."""

    assets = derive_presentation_assets(
        grid,
        society,
        era=era,
        engine_version=engine_version,
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.dumps(
        assets.manifest_document(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _atomic_write_text(output / "presentation.json", manifest)
    _atomic_write_text(output / "city-recipes.js", assets.javascript_document())
    return assets


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _human_seed(grid: WorldGrid) -> int:
    request = grid.metadata.get("societyGeneration")
    if not isinstance(request, Mapping):
        raise ValueError("city presentation requires societyGeneration metadata")
    seed = request.get("humanSeed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("city presentation requires an unsigned 32-bit humanSeed")
    return int(seed)


def _world_seed(grid: WorldGrid, fallback: int) -> int:
    profile = grid.metadata.get("worldProfile")
    if not isinstance(profile, Mapping):
        return fallback
    seed = profile.get("seed", fallback)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        return fallback
    return int(seed)


def _validate_shapes(grid: WorldGrid, society: SocietyLayers) -> None:
    shape = grid.shape
    for name, array in (
        ("water", grid.water),
        ("river_order", grid.river_order),
        ("elevation", grid.elevation),
        ("population", society.population.population_weight),
        ("civilization", society.cultures.civilization_id),
        ("language", society.cultures.language_id),
        ("religion", society.religions.religion_id),
        ("state", society.politics.state_id),
    ):
        if np.asarray(array).shape != shape:
            raise ValueError(f"{name} array must match the WorldGrid shape")


def _derive_grid_context(grid: WorldGrid) -> _GridContext:
    water = np.asarray(grid.water) != 0
    river = np.asarray(grid.river_order) > 0
    water_distance = _wrapped_distance_to(water)
    river_distance = _wrapped_distance_to(river) if bool(np.any(river)) else None
    elevation = np.asarray(grid.elevation, dtype=np.float64)
    row_gradient = np.empty_like(elevation)
    row_gradient[1:-1] = (elevation[2:] - elevation[:-2]) * 0.5
    row_gradient[0] = elevation[1] - elevation[0]
    row_gradient[-1] = elevation[-1] - elevation[-2]
    column_gradient = (np.roll(elevation, -1, axis=1) - np.roll(elevation, 1, axis=1)) * 0.5
    raw_slope = np.hypot(row_gradient, column_gradient)
    land_slope = raw_slope[~water]
    scale = float(np.percentile(land_slope, 98.0)) if land_slope.size else 0.0
    normalized_slope = (
        np.clip(raw_slope / scale, 0.0, 1.0) if scale > 1e-9 else np.zeros_like(raw_slope)
    )
    return _GridContext(
        water_distance=water_distance,
        river_distance=river_distance,
        normalized_slope=normalized_slope,
        cell_area_km2=_cell_area_km2(grid),
    )


def _wrapped_distance_to(mask: np.ndarray) -> np.ndarray:
    """Distance-to-mask in grid cells, with the longitude seam wrapped."""

    values = np.asarray(mask, dtype=bool)
    if not bool(np.any(values)):
        return np.full(values.shape, np.inf, dtype=np.float64)
    width = values.shape[1]
    wrapped = np.concatenate((values, values, values), axis=1)
    return ndimage.distance_transform_edt(~wrapped)[:, width : width * 2]


def _cell_area_km2(grid: WorldGrid) -> np.ndarray:
    height, width = grid.shape
    planet = grid.metadata.get("planet", {})
    radius = float(planet.get("radiusKm", 6371.0)) if isinstance(planet, Mapping) else 6371.0
    if not math.isfinite(radius) or radius <= 0.0:
        radius = 6371.0
    latitudes = 90.0 - (np.arange(height, dtype=np.float64) + 0.5) * 180.0 / height
    delta_latitude = math.pi / height
    delta_longitude = 2.0 * math.pi / width
    areas = radius * radius * delta_latitude * delta_longitude * np.cos(np.deg2rad(latitudes))
    return np.broadcast_to(areas[:, np.newaxis], (height, width)).copy()


def _derive_city_recipe(
    grid: WorldGrid,
    society: SocietyLayers,
    settlement: Settlement,
    context: _GridContext,
    *,
    human_seed: int,
    era: str,
    is_capital: bool,
    location: tuple[float, float] | None,
) -> CityRecipe:
    height, width = grid.shape
    row, column = settlement.row, settlement.column
    if not (0 <= row < height and 0 <= column < width):
        raise ValueError(f"settlement {settlement.identifier} lies outside the WorldGrid")
    if int(grid.water[row, column]) != 0:
        raise ValueError(f"settlement {settlement.identifier} lies on water")
    if int(grid.river_order[row, column]) > 0:
        raise ValueError(f"settlement {settlement.identifier} lies on a represented river channel")

    seed = _city_seed(human_seed, settlement.identifier)
    profile = _ERA_PROFILES[era]
    water_distance, water_bearing, water_code = _nearest_context(
        np.asarray(grid.water) != 0,
        row,
        column,
        maximum_radius=18,
        wrap_width=width,
        value_field=np.asarray(grid.water),
    )
    river_distance, river_bearing, nearest_river_order = _nearest_context(
        np.asarray(grid.river_order) > 0,
        row,
        column,
        maximum_radius=18,
        wrap_width=width,
        value_field=np.asarray(grid.river_order),
    )
    measured_water_distance = _finite_distance(context.water_distance[row, column])
    measured_river_distance = (
        _finite_distance(context.river_distance[row, column])
        if context.river_distance is not None
        else None
    )
    if measured_water_distance is None:
        measured_water_distance = water_distance
    if measured_river_distance is None:
        measured_river_distance = river_distance

    slope = float(context.normalized_slope[row, column])
    slope_class = _slope_class(slope)
    water_kind = _water_kind(
        settlement.site_type,
        water_code,
        measured_water_distance,
        measured_river_distance,
    )
    morphology, street_pattern = _morphology(
        settlement,
        era=era,
        profile=profile,
        water_kind=water_kind,
        water_distance=measured_water_distance,
        river_distance=measured_river_distance,
        slope_class=slope_class,
        variation=_city_random(seed, 1),
    )
    estimate = _population_estimate(settlement, seed)
    footprint_km2, footprint_cells, radius_km = _urban_footprint(
        settlement,
        estimate,
        profile,
        slope=slope,
        water_kind=water_kind,
        cell_area_km2=float(context.cell_area_km2[row, column]),
        variation=_city_random(seed, 2),
    )
    (
        radius_rows,
        radius_columns,
        footprint_bounds,
        grid_cell_height_km,
        grid_cell_width_km,
    ) = _footprint_geometry(
        grid,
        row,
        column,
        radius_km,
    )
    if location is None:
        center_row, center_column = _waterfront_location(grid, row, column,
            radius_rows=radius_rows, radius_columns=radius_columns, water_kind=water_kind,
            water_bearing_degrees=water_bearing)
    else:
        center_row, center_column = location
    delta_row, delta_column = center_row - row - 0.5, center_column - column - 0.5
    footprint_bounds = tuple(round(value + (delta_row if index % 2 == 0 else delta_column), 5)
                             for index, value in enumerate(footprint_bounds))
    (
        maximum_buildable_slope,
        buildable_cells,
        blocked_cells,
        buildable_bounds,
    ) = _buildable_city_cells(
        grid,
        context,
        row=row,
        column=column,
        footprint_bounds=footprint_bounds,
        slope=slope,
        slope_class=slope_class,
    )
    core_count = _core_count(
        settlement,
        estimate,
        profile,
        water_kind=water_kind,
        is_capital=is_capital,
        variation=_city_random(seed, 3),
    )
    core_zones = _core_zones(
        settlement.identifier,
        row=center_row,
        column=center_column,
        count=core_count,
        radius_km=radius_km,
        radius_rows=radius_rows,
        radius_columns=radius_columns,
        water_bearing=water_bearing,
        seed=seed,
        is_capital=is_capital,
        is_holy=settlement.holy_religion_identifier is not None,
        is_port=water_kind in {"seaport", "island-harbor", "lakefront", "riverfront"},
    )
    allowed_core_cells = set(buildable_cells)
    safe_cores = []
    for zone in core_zones:
        core_row, core_column = zone.row, zone.column
        for _ in range(16):
            if (math.floor(core_row), math.floor(core_column) % width) in allowed_core_cells:
                break
            core_row = (core_row + center_row) * 0.5
            core_column = (core_column + center_column) * 0.5
        safe_cores.append(replace(zone, row=round(core_row, 5), column=round(core_column, 5)))
    core_zones = tuple(safe_cores)
    civilization = int(society.cultures.civilization_id[row, column])
    language = int(society.cultures.language_id[row, column])
    religion = int(society.religions.religion_id[row, column])
    state = int(society.politics.state_id[row, column])
    cultural_style = _CULTURAL_STYLE_FAMILIES[
        _style_index(human_seed, civilization, language, religion)
    ]
    landmarks = _landmarks(
        settlement,
        era=era,
        profile=profile,
        water_kind=water_kind,
        is_capital=is_capital,
        core_count=core_count,
    )
    return CityRecipe(
        identifier=settlement.identifier,
        name=settlement.name,
        row=row,
        column=column,
        center_row=center_row,
        center_column=center_column,
        seed=seed,
        tier=settlement.tier,
        site_type=settlement.site_type,
        era=era,
        population_min=settlement.population_min,
        population_max=settlement.population_max,
        population_estimate=estimate,
        footprint_km2=footprint_km2,
        footprint_grid_cells=footprint_cells,
        radius_km=radius_km,
        radius_rows=radius_rows,
        radius_columns=radius_columns,
        grid_cell_height_km=grid_cell_height_km,
        grid_cell_width_km=grid_cell_width_km,
        footprint_bounds=footprint_bounds,
        core_count=core_count,
        core_zones=core_zones,
        morphology=morphology,
        street_pattern=street_pattern,
        building_pattern=_building_pattern(era, water_kind, slope_class),
        orientation_degrees=_urban_orientation(
            seed,
            morphology=morphology,
            water_bearing=water_bearing,
            river_bearing=river_bearing,
        ),
        block_aspect=round(0.65 + 1.20 * _city_random(seed, 4), 3),
        street_irregularity=round(0.12 + 0.78 * _city_random(seed, 5), 3),
        water_kind=water_kind,
        nearest_water_kind=_WATER_KIND_BY_CODE.get(water_code),
        water_distance_cells=_rounded_optional(measured_water_distance),
        river_distance_cells=_rounded_optional(measured_river_distance),
        river_order=int(nearest_river_order or 0),
        water_bearing_degrees=water_bearing,
        river_bearing_degrees=river_bearing,
        slope_class=slope_class,
        slope=round(slope, 4),
        elevation=round(float(grid.elevation[row, column]), 4),
        maximum_buildable_slope=maximum_buildable_slope,
        buildable_cells=buildable_cells,
        blocked_cells=blocked_cells,
        buildable_bounds=buildable_bounds,
        civilization_identifier=civilization,
        language_identifier=language,
        religion_identifier=religion,
        state_identifier=state,
        cultural_style=cultural_style,
        landmarks=landmarks,
        road_connections=0,
        rail_connections=0,
        river_connections=0,
        sea_connections=0,
        bridge_count=0,
        transport_interfaces=(),
        transport_corridors=(),
    )


def _waterfront_location(grid, row, column, *, radius_rows, radius_columns, water_kind, water_bearing_degrees):
    """Place synthetic coastal fabric against an actual shared water edge.

    The accepted native land cell remains authoritative. This subcell choice
    does not invent a coastline or move an inland town to a distant sea. River
    channels keep their existing centreline rather than treating them as seas.
    """
    center_row, center_column = row + 0.5, column + 0.5
    if water_kind not in {"seaport", "island-harbor", "lakefront"}:
        return center_row, center_column
    height, width = grid.shape
    bearing = math.radians(water_bearing_degrees or 0)
    directions = sorted(((0, 1), (0, -1), (-1, 0), (1, 0)),
                        key=lambda offset: -(offset[1] * math.cos(bearing) - offset[0] * math.sin(bearing)))
    for dr, dc in directions:
        nr, nc = row + dr, (column + dc) % width
        if 0 <= nr < height and int(grid.water[nr, nc]) > 0:
            center_row += dr * max(0.0, 0.5 - radius_rows * 0.55)
            center_column += dc * max(0.0, 0.5 - radius_columns * 0.55)
            break
    return round(center_row, 5), round(center_column, 5)


def _nearest_context(
    mask: np.ndarray,
    row: int,
    column: int,
    *,
    maximum_radius: int,
    wrap_width: int,
    value_field: np.ndarray | None = None,
) -> tuple[float | None, int | None, int | None]:
    """Find a nearby water/river bearing without materialising city geometry."""

    values = np.asarray(mask, dtype=bool)
    height, width = values.shape
    maximum = min(maximum_radius, max(1, width // 2), max(1, height - 1))
    best: tuple[float, int, int] | None = None
    for row_offset in range(-maximum, maximum + 1):
        candidate_row = row + row_offset
        if not 0 <= candidate_row < height:
            continue
        for column_offset in range(-maximum, maximum + 1):
            if row_offset == 0 and column_offset == 0:
                continue
            if max(abs(row_offset), abs(column_offset)) > maximum:
                continue
            candidate_column = (column + column_offset) % wrap_width
            if not values[candidate_row, candidate_column]:
                continue
            distance = math.hypot(row_offset, column_offset)
            candidate = (distance, row_offset, column_offset)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None, None, None
    distance, row_offset, column_offset = best
    bearing = int(round(math.degrees(math.atan2(-row_offset, column_offset)) % 360.0)) % 360
    value = (
        int(value_field[(row + row_offset), (column + column_offset) % wrap_width])
        if value_field is not None
        else None
    )
    return distance, bearing, value


def _finite_distance(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _rounded_optional(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _city_seed(human_seed: int, settlement_identifier: str) -> int:
    material = f"world-atlas-city-recipe-v1\0{human_seed}\0{settlement_identifier}".encode("utf-8")
    return int.from_bytes(blake2b(material, digest_size=4).digest(), "big")


def _city_random(seed: int, channel: int) -> float:
    digest = blake2b(f"{seed}\0{channel}".encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _style_index(human_seed: int, civilization: int, language: int, religion: int) -> int:
    material = f"city-culture-style-v1\0{human_seed}\0{civilization}\0{language}\0{religion}".encode("ascii")
    return int.from_bytes(blake2b(material, digest_size=2).digest(), "big") % len(_CULTURAL_STYLE_FAMILIES)


def _slope_class(slope: float) -> str:
    if slope < 0.10:
        return "flat"
    if slope < 0.25:
        return "gentle"
    if slope < 0.48:
        return "rolling"
    if slope < 0.74:
        return "steep"
    return "escarpment"


def _water_kind(
    site_type: str,
    water_code: int | None,
    water_distance: float | None,
    river_distance: float | None,
) -> str:
    if site_type == "island-port":
        return "island-harbor"
    if site_type == "lake-port":
        return "lakefront"
    if site_type == "port":
        return "seaport"
    if site_type == "river-city" or (river_distance is not None and river_distance <= 3.0):
        return "riverfront"
    if site_type == "oasis":
        return "oasis-water"
    if water_distance is not None and water_distance <= 3.0:
        return "coastal-" + _WATER_KIND_BY_CODE.get(water_code, "water")
    return "inland"


def _morphology(
    settlement: Settlement,
    *,
    era: str,
    profile: _EraProfile,
    water_kind: str,
    water_distance: float | None,
    river_distance: float | None,
    slope_class: str,
    variation: float,
) -> tuple[str, str]:
    if settlement.site_type in {"fortress", "pass"}:
        return "fortified-ridge", "contour-following"
    if settlement.site_type == "oasis":
        return "oasis-radial", "radial-grove"
    if water_kind in {"seaport", "island-harbor"}:
        return "waterfront-harbor", "quay-linear"
    if water_kind == "lakefront":
        return "lakefront-crescent", "shore-parallel"
    if water_kind == "riverfront" or (river_distance is not None and river_distance <= 3.0):
        return "river-corridor", "bank-parallel"
    if slope_class in {"steep", "escarpment"}:
        return "fortified-ridge", "contour-following"
    if slope_class == "rolling":
        return "hillside-organic", "contour-following"
    if water_distance is not None and water_distance <= 4.0:
        return "coastal-terrace", "shore-parallel"
    if variation < profile.grid_bias:
        return "planned-grid", "orthogonal"
    return "plain-organic", "organic"


def _population_estimate(settlement: Settlement, seed: int) -> int:
    lower, upper = settlement.population_min, settlement.population_max
    if lower == upper:
        return lower
    # Individual settlements should not all land at the mid-point of their band.
    fraction = 0.30 + 0.45 * _city_random(seed, 0)
    return int(round(lower + (upper - lower) * fraction))


def _urban_footprint(
    settlement: Settlement,
    population: int,
    profile: _EraProfile,
    *,
    slope: float,
    water_kind: str,
    cell_area_km2: float,
    variation: float,
) -> tuple[float, float, float]:
    tier_density = {"metropolis": 0.82, "city": 0.98, "town": 1.18, "site": 0.76}[settlement.tier]
    water_density = 1.13 if water_kind in {"seaport", "island-harbor", "riverfront", "lakefront"} else 1.0
    terrain_density = max(0.55, 1.0 - 0.32 * slope)
    density = profile.density_per_km2 * tier_density * water_density * terrain_density
    density *= 0.88 + 0.24 * variation
    minimum = {"metropolis": 18.0, "city": 4.0, "town": 0.55, "site": 0.12}[settlement.tier]
    footprint = max(minimum, population / max(1.0, density))
    footprint = round(float(footprint), 3)
    equivalent_cells = round(footprint / max(0.001, cell_area_km2), 5)
    radius = round(math.sqrt(footprint / math.pi), 3)
    return footprint, equivalent_cells, radius


def _footprint_geometry(
    grid: WorldGrid,
    row: int,
    column: int,
    radius_km: float,
) -> tuple[float, float, tuple[float, float, float, float], float, float]:
    """Convert a physical footprint into the map's native grid coordinates."""

    height, width = grid.shape
    planet = grid.metadata.get("planet", {})
    radius = float(planet.get("radiusKm", 6371.0)) if isinstance(planet, Mapping) else 6371.0
    radius = radius if math.isfinite(radius) and radius > 0.0 else 6371.0
    cell_height_km = math.pi * radius / height
    latitude = 90.0 - (row + 0.5) * 180.0 / height
    cell_width_km = 2.0 * math.pi * radius * max(0.08, math.cos(math.radians(latitude))) / width
    radius_rows = round(radius_km / cell_height_km, 5)
    radius_columns = round(radius_km / cell_width_km, 5)
    center_row = row + 0.5
    center_column = column + 0.5
    bounds = (
        round(max(0.0, center_row - radius_rows), 5),
        round(center_column - radius_columns, 5),
        round(min(float(height), center_row + radius_rows), 5),
        round(center_column + radius_columns, 5),
    )
    return (
        radius_rows,
        radius_columns,
        bounds,
        round(cell_height_km, 5),
        round(cell_width_km, 5),
    )


def _core_zones(
    settlement_identifier: str,
    *,
    row: float,
    column: float,
    count: int,
    radius_km: float,
    radius_rows: float,
    radius_columns: float,
    water_bearing: int | None,
    seed: int,
    is_capital: bool,
    is_holy: bool,
    is_port: bool,
) -> tuple[CityCoreZone, ...]:
    """Place only semantic centres; detailed streets are generated lazily later."""

    roles = [
        "government-core" if is_capital else "sacred-core" if is_holy else "harbor-core" if is_port else "central-core"
    ]
    if is_port and len(roles) < count:
        roles.append("waterfront-core")
    if is_holy and len(roles) < count:
        roles.append("pilgrimage-core")
    for index in range(len(roles), count):
        roles.append("market-core" if index == 1 else "district-core")
    result: list[CityCoreZone] = []
    inland_bearing = ((water_bearing or 0) + 180) % 360
    for index, role in enumerate(roles):
        if index == 0:
            offset_rows = 0.0
            offset_columns = 0.0
        else:
            spread = (0.28 + 0.36 * _city_random(seed, 30 + index))
            if water_bearing is None:
                bearing = int((_city_random(seed, 40 + index) * 360.0) % 360.0)
            else:
                bearing = int((inland_bearing + (_city_random(seed, 40 + index) - 0.5) * 125.0) % 360.0)
            radians = math.radians(bearing)
            offset_rows = -math.sin(radians) * radius_rows * spread
            offset_columns = math.cos(radians) * radius_columns * spread
        zone_radius = max(0.18, radius_km * (0.19 + 0.12 * _city_random(seed, 50 + index)))
        result.append(
            CityCoreZone(
                identifier=f"{settlement_identifier}-core-{index + 1}",
                role=role,
                row=round(row + offset_rows, 5),
                column=round(column + offset_columns, 5),
                radius_km=round(zone_radius, 3),
            )
        )
    return tuple(result)


def _transport_summary(
    transport_index: CityTransportIndex,
    settlement_identifier: str,
    *,
    water_kind: str,
) -> dict[str, Any]:
    """Expose transport interfaces without inventing roads, rails, or bridges."""

    counts = transport_index.connections[settlement_identifier]
    bridge_count = counts["bridges"]
    interfaces: list[str] = []
    if counts["road"]:
        interfaces.append("road-gate")
    if counts["rail"]:
        interfaces.append("rail-terminal")
    if counts["river"] or water_kind == "riverfront":
        interfaces.append("river-landing")
    if counts["sea"] or water_kind in {"seaport", "island-harbor", "lakefront"}:
        interfaces.append("port-quay")
    if bridge_count:
        interfaces.append("bridge-approach")
    return {**counts, "bridges": int(bridge_count), "interfaces": tuple(interfaces)}


def _assign_urban_clusters(
    grid: WorldGrid,
    recipes: tuple[CityRecipe, ...],
    *,
    era: str,
) -> tuple[tuple[CityRecipe, ...], tuple[UrbanCluster, ...]]:
    """Group proximate major centres without turning all settlements into a blob."""

    candidates = [
        index
        for index, recipe in enumerate(recipes)
        # Population bands are deliberately scaled to each generated world;
        # tier is therefore the stable signal here, not an Earth-specific
        # absolute population threshold.
        if recipe.tier in {"city", "metropolis"}
    ]
    if len(candidates) < 2:
        return recipes, ()
    parent = {index: index for index in candidates}

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first, second = root(first), root(second)
        if first != second:
            parent[max(first, second)] = min(first, second)

    base_threshold = _ERA_CLUSTER_THRESHOLDS[era]
    for offset, first_index in enumerate(candidates):
        first = recipes[first_index]
        for second_index in candidates[offset + 1 :]:
            second = recipes[second_index]
            threshold = base_threshold + min(48.0, 2.2 * (first.radius_km + second.radius_km))
            if _grid_distance_km(grid, first.row, first.column, second.row, second.column) > threshold:
                continue
            if _segment_crosses_water(grid, first.row, first.column, second.row, second.column):
                continue
            union(first_index, second_index)
    grouped: dict[int, list[int]] = {}
    for index in candidates:
        grouped.setdefault(root(index), []).append(index)
    clusters: list[UrbanCluster] = []
    replacements: dict[int, CityRecipe] = {}
    for member_indices in grouped.values():
        if len(member_indices) < 2:
            continue
        members = sorted((recipes[index] for index in member_indices), key=lambda item: item.identifier)
        core = max(members, key=lambda item: (item.population_estimate, item.identifier))
        identifier_material = "\0".join(item.identifier for item in members).encode("utf-8")
        identifier = "urban-cluster-" + blake2b(identifier_material, digest_size=6).hexdigest()
        cluster = UrbanCluster(
            identifier=identifier,
            core_settlement_id=core.identifier,
            member_settlement_ids=tuple(item.identifier for item in members),
            population_estimate=sum(item.population_estimate for item in members),
        )
        clusters.append(cluster)
        for index in member_indices:
            recipe = recipes[index]
            replacements[index] = replace(
                recipe,
                cluster_identifier=identifier,
                cluster_member_count=len(members),
                cluster_role="core" if recipe.identifier == core.identifier else "member",
            )
    return tuple(replacements.get(index, recipe) for index, recipe in enumerate(recipes)), tuple(sorted(clusters, key=lambda item: item.identifier))


def _grid_distance_km(
    grid: WorldGrid,
    first_row: int,
    first_column: int,
    second_row: int,
    second_column: int,
) -> float:
    height, width = grid.shape
    planet = grid.metadata.get("planet", {})
    radius = float(planet.get("radiusKm", 6371.0)) if isinstance(planet, Mapping) else 6371.0
    radius = radius if math.isfinite(radius) and radius > 0.0 else 6371.0
    row_delta = (second_row - first_row) * math.pi * radius / height
    column_delta = abs(second_column - first_column)
    column_delta = min(column_delta, width - column_delta)
    latitude = 90.0 - ((first_row + second_row) * 0.5 + 0.5) * 180.0 / height
    column_distance = column_delta * 2.0 * math.pi * radius * math.cos(math.radians(latitude)) / width
    return math.hypot(row_delta, column_distance)


def _segment_crosses_water(
    grid: WorldGrid,
    first_row: int,
    first_column: int,
    second_row: int,
    second_column: int,
) -> bool:
    """Reject a cluster link that would visibly leap across sea water."""

    height, width = grid.shape
    delta_column = second_column - first_column
    if abs(delta_column) > width / 2:
        delta_column -= int(math.copysign(width, delta_column))
    steps = max(abs(second_row - first_row), abs(delta_column), 1) * 2
    for step in range(1, int(steps)):
        fraction = step / steps
        row = int(round(first_row + (second_row - first_row) * fraction))
        column = int(round(first_column + delta_column * fraction)) % width
        if not 0 <= row < height or int(grid.water[row, column]) != 0:
            return True
    return False


def _maximum_buildable_slope(slope_class: str) -> float:
    return {
        "flat": 0.35,
        "gentle": 0.42,
        "rolling": 0.52,
        "steep": 0.64,
        "escarpment": 0.72,
    }[slope_class]


def _buildable_city_cells(
    grid: WorldGrid,
    context: _GridContext,
    *,
    row: int,
    column: int,
    footprint_bounds: tuple[float, float, float, float],
    slope: float,
    slope_class: str,
) -> tuple[float, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[int, int, int, int]]:
    """Return the native cells that may contain city geometry.

    The browser intentionally receives this conservative source-grid mask instead
    of invented high-resolution terrain.  A recipe can therefore draw a detailed
    cartographic footprint while its fill, streets, cores, and transport markers
    are clipped away from both water and represented river channels.
    """

    height, width = grid.shape
    north, west, south, east = footprint_bounds
    row_start = max(0, int(math.floor(north)) - 1)
    row_end = min(height - 1, int(math.ceil(south)) + 1)
    column_start = int(math.floor(west)) - 1
    column_end = int(math.ceil(east)) + 1
    # A fortified or terraced settlement can occupy its accepted anchor cell,
    # but still cannot spill into steeper neighbours.  This preserves the
    # existing settlement authority without pretending all escarpment cells are
    # ordinary building land.
    maximum_slope = min(
        1.0,
        max(_maximum_buildable_slope(slope_class), float(slope) + 0.035),
    )
    allowed: list[tuple[int, int]] = []
    blocked: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = set()
    water = np.asarray(grid.water)
    river_order = np.asarray(grid.river_order)
    for candidate_row in range(row_start, row_end + 1):
        for unwrapped_column in range(column_start, column_end + 1):
            candidate_column = unwrapped_column % width
            candidate = (candidate_row, candidate_column)
            if candidate in visited:
                continue
            visited.add(candidate)
            permitted = (
                int(water[candidate]) == 0
                and int(river_order[candidate]) == 0
                and float(context.normalized_slope[candidate]) <= maximum_slope
            )
            (allowed if permitted else blocked).append(candidate)
    if (row, column) not in allowed:
        raise ValueError(
            "settlement %s has no native land cell eligible for its city footprint"
            % (f"at row {row}, column {column}",)
        )
    return (
        round(maximum_slope, 4),
        tuple(sorted(allowed)),
        tuple(sorted(blocked)),
        (row_start, column_start, row_end, column_end),
    )


def _core_count(
    settlement: Settlement,
    population: int,
    profile: _EraProfile,
    *,
    water_kind: str,
    is_capital: bool,
    variation: float,
) -> int:
    tier_base = {"site": 1, "town": 1, "city": 2, "metropolis": 3}[settlement.tier]
    population_bonus = max(0, int(math.log10(max(1, population)) - 5.35))
    water_bonus = int(water_kind in {"seaport", "island-harbor", "riverfront"} and population >= 180_000)
    capital_bonus = int(is_capital and population >= 100_000)
    variation_bonus = int(variation > 0.80 and settlement.tier in {"city", "metropolis"})
    return min(7, max(1, tier_base + profile.core_bonus + population_bonus + water_bonus + capital_bonus + variation_bonus))


def _building_pattern(era: str, water_kind: str, slope_class: str) -> str:
    if water_kind in {"seaport", "island-harbor", "lakefront", "riverfront"}:
        return "waterfront-mixed-use"
    if slope_class in {"steep", "escarpment"}:
        return "terraced-compounds"
    if era == "tribal":
        return "dispersed-compounds"
    if era == "ancient":
        return "courtyard-blocks"
    if era == "medieval":
        return "walled-lots"
    if era == "early-modern":
        return "mercantile-blocks"
    if era in {"industrial", "contemporary"}:
        return "mixed-density-blocks"
    return "perimeter-blocks"


def _urban_orientation(
    seed: int,
    *,
    morphology: str,
    water_bearing: int | None,
    river_bearing: int | None,
) -> int:
    """Align shore and river cities with their actual local interface."""

    bearing = (
        river_bearing
        if morphology == "river-corridor" and river_bearing is not None
        else water_bearing
        if morphology in {"waterfront-harbor", "lakefront-crescent", "coastal-terrace"}
        else None
    )
    if bearing is None:
        return int(seed % 360)
    # Street grids and waterfront quays normally run roughly parallel to the
    # bank, while the small deterministic deviation prevents cloned cities.
    variation = int(round((_city_random(seed, 67) - 0.5) * 18.0))
    return (int(bearing) + 90 + variation) % 360


def _street_widths_metres(era: str, tier: str) -> dict[str, float]:
    """Cartographic street classes expressed in real-world metres.

    Browsers convert these to local grid units using the authored cell size;
    they do not use a fixed screen-pixel icon width.
    """

    arterial, collector, local = {
        "tribal": (4.0, 2.4, 1.4),
        "ancient": (8.0, 4.5, 2.4),
        "medieval": (7.0, 3.8, 2.1),
        "early-modern": (12.0, 6.2, 3.1),
        "preindustrial": (11.0, 5.6, 3.0),
        "industrial": (20.0, 9.0, 4.8),
        "contemporary": (28.0, 13.0, 6.5),
    }[era]
    scale = {"site": 0.68, "town": 0.82, "city": 1.0, "metropolis": 1.16}[tier]
    return {
        "arterial": round(arterial * scale, 1),
        "collector": round(collector * scale, 1),
        "local": round(local * scale, 1),
    }


def _landmarks(
    settlement: Settlement,
    *,
    era: str,
    profile: _EraProfile,
    water_kind: str,
    is_capital: bool,
    core_count: int,
) -> tuple[str, ...]:
    values: list[str] = []
    if is_capital:
        values.append("government-seat")
    if settlement.holy_religion_identifier is not None:
        values.append("temple-complex")
    if settlement.site_type in {"fortress", "pass"}:
        values.append("citadel")
    if water_kind in {"seaport", "island-harbor"}:
        values.extend(("harbor", "lighthouse"))
    elif water_kind in {"riverfront", "lakefront"}:
        values.append("river-port")
    if era in {"industrial", "contemporary"} and settlement.tier in {"city", "metropolis"}:
        values.append(profile.landmark)
    elif settlement.tier in {"city", "metropolis"}:
        values.append(profile.landmark)
    if core_count >= 4:
        values.append("secondary-market")
    return tuple(dict.fromkeys(values))


__all__ = [
    "CITY_RECIPES_GLOBAL",
    "PRESENTATION_SCHEMA",
    "PRESENTATION_VERSION",
    "CityRecipe",
    "PresentationAssets",
    "derive_presentation_assets",
    "resolve_city_era",
    "society_content_digest",
    "write_presentation_assets",
]
