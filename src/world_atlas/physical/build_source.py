"""Build physical atlas fields and editable geometry from authored colour artwork.

The input is deliberately treated as a coloured artwork, not a DEM.  The output
therefore contains authored-relative ordinal elevation and bathymetry bands plus
derived hydrography.  The output contract and planetary CRS come from the
reviewed context; the result is not assumed to be GeoJSON or a
political-boundary source.  Rivers and perennial snow are derived from the
continuous relative surface rather than copied from independent authored
centerlines.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from contourpy import contour_generator
from PIL import Image
from shapely import make_valid
from shapely.affinity import scale as affine_scale
from shapely.affinity import translate as affine_translate
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import nearest_points, polygonize, unary_union

from world_atlas.physical.climate import (
    HYDROCLIMATE_MODEL_NAME,
    SNOWLINE_MODEL_NAME,
    MonsoonHydroclimateResult,
    SnowlineConfig,
    compute_monsoon_hydroclimate,
    compute_perennial_snowline,
)
from world_atlas.physical.hydrology import HydrologyConfig, HydrologyResult, compute_hydrology
from world_atlas.physical.island_generation import (
    LakeIslandField,
    ShelfArchipelagoField,
    classify_inland_seas,
    derive_inland_sea_bathymetry,
    generate_inland_water_islands,
    generate_shelf_archipelagos,
)
from world_atlas.physical.ocean_floor import complete_ocean_bathymetry
from world_atlas.physical.planetary_grid import PlanetConfig
from world_atlas.physical.polar_terrain import generate_polar_terrain
from world_atlas.physical.physiographic_provinces import shape_physiographic_provinces
from world_atlas.physical.terrain_refinement import incise_drainage, refine_continuous_terrain


SOURCE_NAME = "eirenor-physical-reference.png"
SOURCE_ARTIFACT = "src/assets/eirenor-physical-reference.png"
SOURCE_WIDTH = 1591
SOURCE_HEIGHT = 906
BOARD_WIDTH = 1812
BOARD_HEIGHT = 906
LEFT_PADDING = 111
RIGHT_PADDING = 110
TOP_PADDING = 0
BOTTOM_PADDING = 0
GRID_STEP = 2
GRID_WIDTH = math.ceil(BOARD_WIDTH / GRID_STEP)
GRID_HEIGHT = math.ceil(BOARD_HEIGHT / GRID_STEP)
GRID_X_STEP = BOARD_WIDTH / GRID_WIDTH
GRID_Y_STEP = BOARD_HEIGHT / GRID_HEIGHT
LONGITUDE_MIN = -180.0
LONGITUDE_MAX = 180.0
LATITUDE_MIN = -90.0
LATITUDE_MAX = 90.0
ELEVATION_LEVELS = 16
BATHYMETRY_LEVELS = 8
SIMPLIFY_TOLERANCE = 0.85
COASTLINE_TOLERANCE = 0.35
COORDINATE_DECIMALS = 5
SMOOTHING_ITERATIONS = 2
BATHYMETRY_SMOOTHING_ITERATIONS = 6
RIVER_SMOOTHING_ITERATIONS = 2
ELEVATION_REGULARIZATION_SIGMA_GRID_UNITS = 1.5
ELEVATION_REGULARIZATION_PASSES = 2
ELEVATION_CONTOUR_SIMPLIFY_TOLERANCE = 0.25

ELEVATION_REGULARIZATION_ALGORITHM = "masked-normalized-gaussian"
ELEVATION_SURFACE_METHOD = (
    "palette-position-inpaint-gaussian-physiographic-provinces-two-pass-hydrology"
)

# These are deliberately fractions of the sampled land-cell count rather than
# physical discharge values.  The lower stream threshold is appropriate for
# the roughly 161k land cells in this artwork while still leaving room for
# higher-order tributaries and a readable mainstem class.
HYDROLOGY_CONFIG = HydrologyConfig(
    stream_burn_depth=0.035,
    stream_threshold_fraction=0.00004,
    tributary_threshold_fraction=0.001,
    mainstem_threshold_fraction=0.01,
    minimum_headwater_length=6,
    minimum_stream_span_factor=0.04,
    valley_support_threshold=0.08,
    lowland_headwater_flow_multiplier=1.25,
)
SNOWLINE_CONFIG = SnowlineConfig()

LAND_PALETTE = np.array(
    [
        (0x69, 0xBD, 0xA9),
        (0x8F, 0xD2, 0xA4),
        (0xB6, 0xE2, 0xA1),
        (0xD7, 0xEF, 0x9F),
        (0xEE, 0xF8, 0xA8),
        (0xFB, 0xF8, 0xB0),
        (0xFE, 0xEB, 0x9F),
        (0xFE, 0xD4, 0x83),
        (0xFD, 0xB7, 0x6A),
        (0xF9, 0x94, 0x56),
        (0xF0, 0x70, 0x4A),
        (0xE0, 0x50, 0x4A),
    ],
    dtype=np.float64,
)

# These are the eight recurring blue source colours, ordered from the pale
# coastal fringe to the deep authoring-board ocean.  They remain ordinal source
# references; they are not a bathymetric metre scale.
BATHYMETRY_PALETTE = np.array(
    [
        (0x9F, 0xB8, 0xE7),
        (0x99, 0xB3, 0xE4),
        (0x91, 0xAD, 0xE1),
        (0x99, 0xAF, 0xD1),
        (0x8C, 0xA5, 0xCB),
        (0x7C, 0x98, 0xC3),
        (0x6B, 0x8B, 0xBC),
        (0x67, 0x87, 0xB8),
    ],
    dtype=np.float64,
)


class SourceBuildError(RuntimeError):
    """Raised when an invariant would make the generated source unsafe."""


@dataclass(frozen=True, slots=True)
class RasterCleanupPolicy:
    """Semantic, scale-free gates for removing raster components and holes.

    The policy is evaluated on connected-component area, bounding-box aspect,
    filled ratio, and the widest local row/column span.  It never names a
    source coordinate: edge-connected components are preserved generically so
    a narrow strait or board-touching ocean remains registered.  A component
    that overlaps ``persistence_mask`` is also retained, allowing an ordinal
    feature that persists into another layer to survive a small-speck filter.
    """

    semantic: str
    connectivity: int = 8
    min_component_cells: int = 1
    min_component_width: int = 1
    min_component_fill: float = 0.0
    # A finite default keeps the audit record strict-JSON serialisable while
    # being effectively unbounded for a sampled board.
    max_component_aspect: float = 1.0e9
    min_component_area_fraction: float | None = None
    min_component_width_fraction: float | None = None
    min_hole_area_fraction: float | None = None
    min_hole_width_fraction: float | None = None
    min_hole_cells: int = 0
    min_hole_width: int = 0
    preserve_edge_components: bool = True

    def __post_init__(self) -> None:
        if not self.semantic.strip():
            raise SourceBuildError("raster cleanup semantic must be non-empty")
        if self.connectivity not in {4, 8}:
            raise SourceBuildError("raster cleanup connectivity must be 4 or 8")
        for name in (
            "min_component_cells",
            "min_component_width",
            "min_hole_cells",
            "min_hole_width",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SourceBuildError(f"raster cleanup {name} must be a non-negative integer")
        for name in ("min_component_fill", "max_component_aspect"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise SourceBuildError(f"raster cleanup {name} must be finite and non-negative")
        if self.min_component_fill > 1.0:
            raise SourceBuildError("raster cleanup min_component_fill cannot exceed one")
        for name in (
            "min_component_area_fraction",
            "min_component_width_fraction",
            "min_hole_area_fraction",
            "min_hole_width_fraction",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise SourceBuildError(
                    f"raster cleanup {name} must be finite and non-negative when supplied"
                )

    def as_record(self) -> dict[str, Any]:
        """Return an auditable JSON-safe policy record."""

        return {
            "semantic": self.semantic,
            "connectivity": self.connectivity,
            "minComponentCells": self.min_component_cells,
            "minComponentWidth": self.min_component_width,
            "minComponentFill": self.min_component_fill,
            "maxComponentAspect": self.max_component_aspect,
            "minComponentAreaFraction": self.min_component_area_fraction,
            "minComponentWidthFraction": self.min_component_width_fraction,
            "minHoleCells": self.min_hole_cells,
            "minHoleWidth": self.min_hole_width,
            "minHoleAreaFraction": self.min_hole_area_fraction,
            "minHoleWidthFraction": self.min_hole_width_fraction,
            "preserveEdgeComponents": self.preserve_edge_components,
        }


@dataclass(frozen=True, slots=True)
class RasterCleanupStats:
    """Deterministic counts emitted alongside one cleanup pass."""

    semantic: str
    connectivity: int
    input_components: int
    retained_components: int
    removed_components: int
    filled_holes: int
    input_cells: int
    output_cells: int
    resolved_min_component_cells: int
    resolved_min_component_width: int
    resolved_min_hole_cells: int
    resolved_min_hole_width: int
    removed_component_bboxes: tuple[tuple[int, int, int, int], ...] = ()
    filled_hole_bboxes: tuple[tuple[int, int, int, int], ...] = ()

    def as_record(self, *, include_components: bool = False) -> dict[str, Any]:
        """Return a JSON-safe audit record."""

        record: dict[str, Any] = {
            "semantic": self.semantic,
            "connectivity": self.connectivity,
            "inputComponents": self.input_components,
            "retainedComponents": self.retained_components,
            "removedComponents": self.removed_components,
            "filledHoles": self.filled_holes,
            "inputCells": self.input_cells,
            "outputCells": self.output_cells,
            "resolvedMinComponentCells": self.resolved_min_component_cells,
            "resolvedMinComponentWidth": self.resolved_min_component_width,
            "resolvedMinHoleCells": self.resolved_min_hole_cells,
            "resolvedMinHoleWidth": self.resolved_min_hole_width,
        }
        if include_components:
            record["removedComponentBboxes"] = [list(bbox) for bbox in self.removed_component_bboxes]
            record["filledHoleBboxes"] = [list(bbox) for bbox in self.filled_hole_bboxes]
        return record


@dataclass(frozen=True, slots=True)
class GeometryArtifactCleanupStats:
    """Audit one post-nesting geometry artifact cleanup pass.

    The thresholds are expressed in the sampled authoring grid's units.  A
    report keeps polygon and hole removals separate because dropping a tiny
    polygon reduces coverage while dropping a tiny hole restores coverage.
    """

    minimum_area: float
    minimum_hole_area: float
    minimum_clear_width: float
    input_part_count: int
    output_part_count: int
    removed_polygon_count: int
    removed_polygon_area: float
    input_hole_count: int
    output_hole_count: int
    removed_hole_count: int
    removed_hole_area: float
    input_area: float
    output_area: float
    performed: bool = True

    def as_record(self) -> dict[str, Any]:
        """Return a JSON-safe provenance/QA record."""

        return {
            "performed": self.performed,
            "rule": "relative-grid-area-and-area-span-artifact-cleanup",
            "minimumAreaGrid2": self.minimum_area,
            "minimumHoleAreaGrid2": self.minimum_hole_area,
            "minimumClearWidthGrid": self.minimum_clear_width,
            "inputPartCount": self.input_part_count,
            "outputPartCount": self.output_part_count,
            "removedPolygonCount": self.removed_polygon_count,
            "removedPolygonAreaGrid2": self.removed_polygon_area,
            "inputHoleCount": self.input_hole_count,
            "outputHoleCount": self.output_hole_count,
            "removedHoleCount": self.removed_hole_count,
            "removedHoleAreaGrid2": self.removed_hole_area,
            "inputAreaGrid2": self.input_area,
            "outputAreaGrid2": self.output_area,
        }


@dataclass(frozen=True, slots=True)
class PhysicalFields:
    """The canonical unsmoothed raster fields shared by all map builders.

    The arrays use NumPy ``(row, column)`` / ``(y, x)`` indexing.  They are
    deliberately kept in memory rather than serialized into the geographic
    source: the source JSON publishes review geometry, while settlement
    placement consumes these exact classified cells.  ``continuous_elevation``
    is the continuous authored palette position; ``elevation_ordinal`` is its
    configured ordinal display classification.  Neither is a metric DEM.
    """

    source_path: Path
    source_sha256: str
    source_dimensions: tuple[int, int]
    board_dimensions: tuple[int, int]
    grid_step_px: int
    left_padding_px: int
    land_mask: np.ndarray
    lake_mask: np.ndarray
    inland_sea_mask: np.ndarray
    generated_lake_island_mask: np.ndarray
    generated_ocean_island_mask: np.ndarray
    generated_polar_land_mask: np.ndarray
    generated_north_polar_land_mask: np.ndarray
    generated_south_polar_land_mask: np.ndarray
    inland_water_mask: np.ndarray
    ocean_mask: np.ndarray
    water_candidate_mask: np.ndarray
    raster_cleanup: dict[str, Any]
    continuous_elevation: np.ndarray
    elevation_ordinal: np.ndarray
    bathymetry: np.ndarray
    linear_constraint_mask: np.ndarray
    authored_river_constraint: np.ndarray
    hydroclimate: MonsoonHydroclimateResult
    hydrology: HydrologyResult
    perennial_snow_mask: np.ndarray
    inland_components: tuple[tuple[tuple[int, int], ...], ...]
    lake_component_indices: frozenset[int]
    runtime_parameters: dict[str, Any]
    context: "BuildContext"

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Return ``(height, width)`` for every raster field."""

        return tuple(int(value) for value in self.land_mask.shape)

    @property
    def river_channel_mask(self) -> np.ndarray:
        """Return the active extracted river-channel cells."""

        return self.hydrology.stream_mask

    @property
    def physical_version(self) -> str:
        """Stable version label for the canonical field preparation chain."""

        return f"{self.context.output_contract_id}/physical-fields-v12"

    @property
    def generation_parameters(self) -> dict[str, Any]:
        """Return the exact classification parameters used to make fields."""
        parameters = dict(self.runtime_parameters)
        parameters["surfaceRule"] = (
            "nearest ordered authored palette; exterior connected water component is ocean"
        )
        parameters["elevationRule"] = (
            "piecewise-linear-RGB-segment-projection, water-line inpaint, masked "
            "normalized Gaussian regularization, then ordinal classes from that shared DEM"
        )
        parameters["elevationContourRule"] = (
            "cell-centre marching-square threshold contours with linear subcell crossings, "
            "followed by topology-preserving simplification and Chaikin smoothing"
        )
        parameters["bathymetryRule"] = (
            "piecewise-linear-RGB-segment-projection to continuous palette position, "
            "masked normalized 3x3 smoothing, then authored ordinal classes; "
            "board-edge ocean is deepest"
        )
        parameters["bathymetrySmoothingRule"] = (
            "authored palette-position [0,1] surface; normalized masked 3x3 box "
            "passes; explicit longitude wrap only; floor/clip quantization"
        )
        parameters["denoiseRule"] = (
            "semantic connected-component area/width/fill/aspect cleanup with persistence and pinhole filling; no coordinate exceptions"
        )
        parameters["rasterCleanup"] = self.raster_cleanup
        return parameters


# Raster cleanup is deliberately expressed with both a small-cell floor and a
# relative fraction.  The floor keeps synthetic/very small boards useful;
# the fraction scales the semantic gate when the authoring resolution changes.
WATER_CLEANUP_POLICY = RasterCleanupPolicy(
    semantic="surface-water-candidate",
    min_component_cells=2,
    min_component_width=1,
    min_component_fill=0.28,
    max_component_aspect=4.5,
    min_component_area_fraction=0.000012,
    min_component_width_fraction=0.003,
    preserve_edge_components=True,
)
ELEVATION_CLEANUP_POLICY = RasterCleanupPolicy(
    semantic="elevation-cumulative-band",
    min_component_cells=1,
    min_component_width=1,
    min_component_fill=0.18,
    max_component_aspect=8.0,
    min_component_area_fraction=0.000012,
    min_component_width_fraction=0.002,
    min_hole_cells=2,
    min_hole_width=1,
    preserve_edge_components=False,
)
BATHYMETRY_CLEANUP_POLICY = RasterCleanupPolicy(
    semantic="bathymetry-cumulative-band",
    min_component_cells=1,
    min_component_width=1,
    min_component_fill=0.18,
    max_component_aspect=8.0,
    min_component_area_fraction=0.000012,
    min_component_width_fraction=0.003,
    min_hole_cells=2,
    min_hole_width=1,
    preserve_edge_components=True,
)

# Geometry coordinates are authored in sampled grid units.  Keep these gates
# shared by pre-vectorization and post-nesting cleanup so a nested repair does
# not silently use a different sliver definition.
GEOMETRY_ARTIFACT_MIN_AREA_GRID2 = 0.025
GEOMETRY_ARTIFACT_MIN_CLEAR_WIDTH_GRID = 0.055
# A lake must occupy a scale-relative patch of the sampled surface.  This is
# deliberately separate from the broad water-candidate cleanup gate: thin
# authored blue marks remain available as river evidence, while tiny compact
# colour specks cannot become dozens of visible inland lakes after upsampling.
LAKE_COMPONENT_AREA_FRACTION = 0.00002
LAKE_ISLAND_MINIMUM_AREA_FRACTION = 0.0005
LAKE_ISLAND_MINIMUM_AREA_CELLS = 256
LAKE_ISLAND_MAXIMUM_COUNT = 4
LAKE_ISLAND_SHORE_CLEARANCE_CELLS = 3
INLAND_SEA_MINIMUM_AREA_FRACTION = 0.005
INLAND_SEA_MINIMUM_AREA_CELLS = 2_048
SHELF_ARCHIPELAGO_MAXIMUM_GROUPS = 10
SHELF_ARCHIPELAGO_MAXIMUM_ISLANDS_PER_GROUP = 10
SHELF_ARCHIPELAGO_MINIMUM_SHORE_DISTANCE = 4
SHELF_ARCHIPELAGO_MAXIMUM_SHORE_DISTANCE = 34

HYDROLOGY_ALGORITHM = "priority-flood-d8-climate-runoff-headwater-network-v14"

HYDROLOGY_PROVENANCE = {
    "algorithm": HYDROLOGY_ALGORITHM,
    "routingScore": "greatest-descent-slope; relative drop divided by ground distance",
    "groundDistance": "D8 plate-carree step with latitude-aware cosine horizontal factor and polar floor 0.05",
    "flatRouting": "lexicographic physical cost-to-spill on corrected flats; raw ascent is exact outside authored valleys and uncertainty-tolerant inside them, then valley evidence and D8-distance travel through deterministic low-frequency unresolved alluvial relief bend ruler-straight flatland routes without permitting resolved uphill flow",
    "mfdSelection": "fractional downslope accumulation over physical-slope weights and normalized runoff-source yield; D8 remains the published topology",
    "runoffSource": "latitude/coast/orographic monsoon dry-season yield, increasing upland yield, and perennial-snow meltwater bonus; weights normalize to one land-cell equivalent on average",
    "valleySupport": "concavity against an automatically scaled local box; positive-depth 90th-percentile normalization and local-relief floor, then downstream propagation",
    "parallelSuppression": "bidirectional deterministic corridor competition with confluence buffering; closure is followed by a second suppression pass; authored and valley evidence are priority signals, never absolute exemptions",
    "thresholdModel": "global threshold=max(1, ceil(landCells * streamThresholdFraction), ceil(sqrt(landCells) * minimumStreamSpanFactor)); valley, upland-valley, snow, and lake-draining basins apply bounded local reductions before pruning; tributary/mainstem classes use configured fractions",
    "lakeHeadwaterExtension": "lake-bound headwaters may be extended along reverse-D8 predecessors before minimum-length pruning, consuming a bounded share of the normalized closure budget; no lateral route is invented",
    "boundedClosure": "closure follows existing D8 paths only, atomically bounded by normalized total-cell budget and maximum chain length; over-limit chains remain unselected and dangling cells are discarded",
    "headwaterPruning": "iterative minimum-length pruning followed by catchment provenance gating: upstream upland, perennial snow, or a substantially larger valley-supported lowland flow is required; the minimum-length invariant is reapplied after every late topology-changing prune",
    "constraintModel": "credible thin/linear authored water-palette components expand into a five-cell valley floor with graded two-cell shoulders; compact/noise components are inpaint-only; finite burn depth is applied before depression correction",
}


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Immutable parameters for one physical-source build pass.

    Raster and geometry helpers receive this context explicitly.  Keeping
    source dimensions, palettes, and policies out of module globals means
    separate field-preparation passes cannot change one another's classification rules.
    """

    source_name: str
    source_artifact: str
    source_dimensions: tuple[int, int]
    board_dimensions: tuple[int, int]
    padding: tuple[int, int, int, int]
    sampling_step_px: int
    grid_dimensions: tuple[int, int]
    longitude_extent: tuple[float, float]
    latitude_extent: tuple[float, float]
    elevation_levels: int
    bathymetry_levels: int
    land_palette: tuple[tuple[int, int, int], ...]
    bathymetry_palette: tuple[tuple[int, int, int], ...]
    simplify_tolerance: float
    coastline_tolerance: float
    coordinate_decimals: int
    elevation_regularization_sigma_grid_units: float
    elevation_regularization_passes: int
    elevation_contour_simplify_tolerance: float
    smoothing_iterations: int
    bathymetry_smoothing_iterations: int
    river_smoothing_iterations: int
    water_cleanup_policy: RasterCleanupPolicy
    elevation_cleanup_policy: RasterCleanupPolicy
    bathymetry_cleanup_policy: RasterCleanupPolicy
    hydrology_config: HydrologyConfig
    snowline_config: SnowlineConfig
    planet: PlanetConfig
    output_format: str
    output_contract_id: str
    output_schema_version: int
    projection: str
    coordinate_reference_system_id: str
    coordinate_reference_system_name: str
    datum: str
    coordinate_order: str
    longitude_positive_direction: str
    latitude_reference: str
    prime_meridian_degrees: float
    coordinate_unit: str
    wgs84: bool
    axis: str
    edge_policy: str
    profile_id: str
    canonical: bool
    complete_globe: bool

    @property
    def source_width(self) -> int:
        return self.source_dimensions[0]

    @property
    def source_height(self) -> int:
        return self.source_dimensions[1]

    @property
    def board_width(self) -> int:
        return self.board_dimensions[0]

    @property
    def board_height(self) -> int:
        return self.board_dimensions[1]

    @property
    def left_padding(self) -> int:
        return self.padding[0]

    @property
    def right_padding(self) -> int:
        return self.padding[1]

    @property
    def top_padding(self) -> int:
        return self.padding[2]

    @property
    def bottom_padding(self) -> int:
        return self.padding[3]

    @property
    def grid_width(self) -> int:
        return self.grid_dimensions[0]

    @property
    def grid_height(self) -> int:
        return self.grid_dimensions[1]

    @property
    def grid_x_step(self) -> float:
        return self.board_width / self.grid_width

    @property
    def grid_y_step(self) -> float:
        return self.board_height / self.grid_height

    @classmethod
    def reviewed(cls) -> "BuildContext":
        """Return reviewed defaults for direct library consumers."""

        return cls(
            source_name=SOURCE_NAME,
            source_artifact=SOURCE_ARTIFACT,
            source_dimensions=(SOURCE_WIDTH, SOURCE_HEIGHT),
            board_dimensions=(BOARD_WIDTH, BOARD_HEIGHT),
            padding=(LEFT_PADDING, RIGHT_PADDING, TOP_PADDING, BOTTOM_PADDING),
            sampling_step_px=GRID_STEP,
            grid_dimensions=(GRID_WIDTH, GRID_HEIGHT),
            longitude_extent=(LONGITUDE_MIN, LONGITUDE_MAX),
            latitude_extent=(LATITUDE_MIN, LATITUDE_MAX),
            elevation_levels=ELEVATION_LEVELS,
            bathymetry_levels=BATHYMETRY_LEVELS,
            land_palette=tuple(
                tuple(int(channel) for channel in colour) for colour in LAND_PALETTE
            ),
            bathymetry_palette=tuple(
                tuple(int(channel) for channel in colour) for colour in BATHYMETRY_PALETTE
            ),
            simplify_tolerance=SIMPLIFY_TOLERANCE,
            coastline_tolerance=COASTLINE_TOLERANCE,
            coordinate_decimals=COORDINATE_DECIMALS,
            elevation_regularization_sigma_grid_units=ELEVATION_REGULARIZATION_SIGMA_GRID_UNITS,
            elevation_regularization_passes=ELEVATION_REGULARIZATION_PASSES,
            elevation_contour_simplify_tolerance=ELEVATION_CONTOUR_SIMPLIFY_TOLERANCE,
            smoothing_iterations=SMOOTHING_ITERATIONS,
            bathymetry_smoothing_iterations=BATHYMETRY_SMOOTHING_ITERATIONS,
            river_smoothing_iterations=RIVER_SMOOTHING_ITERATIONS,
            water_cleanup_policy=WATER_CLEANUP_POLICY,
            elevation_cleanup_policy=ELEVATION_CLEANUP_POLICY,
            bathymetry_cleanup_policy=BATHYMETRY_CLEANUP_POLICY,
            hydrology_config=HYDROLOGY_CONFIG,
            snowline_config=SNOWLINE_CONFIG,
            planet=PlanetConfig(
                radius_km=6371.0,
                gravity_mps2=9.80665,
                rotation_period_hours=24.0,
                rotation_direction="prograde",
                axial_tilt_degrees=23.44,
                orbital_period_days=365.25,
                orbital_eccentricity=0.0167,
                longitude_of_periapsis_degrees=102.9,
                solar_constant_wm2=1361.0,
                surface_pressure_pa=101325.0,
                ocean_heat_capacity_factor=4.0,
            ),
            output_format="eirenor-physical-geographic-source",
            output_contract_id="EIR-GEOG-1",
            output_schema_version=1,
            projection="plate-carree",
            coordinate_reference_system_id="EIR-GEOG-1",
            coordinate_reference_system_name="Eirenor custom planetary geographic coordinates",
            datum="eir-custom-planetary",
            coordinate_order="[lon,lat]",
            longitude_positive_direction="east",
            latitude_reference="planetary-center",
            prime_meridian_degrees=0.0,
            coordinate_unit="degree",
            wgs84=False,
            axis="x-east,y-south-display",
            edge_policy="closed-board-with-longitude-wrap-at-antimeridian",
            profile_id="eirenor-reviewed-physical",
            canonical=False,
            complete_globe=False,
        )


def reviewed_context() -> BuildContext:
    """Create the reviewed Eirenor context for an explicit caller."""

    return BuildContext.reviewed()


def _runtime_parameters(context: BuildContext) -> dict[str, Any]:
    """Return one immutable context's values for field provenance."""

    return {
        "sourceArtifact": context.source_artifact,
        "completeGlobe": context.complete_globe,
        "sourceWidthPx": context.source_width,
        "sourceHeightPx": context.source_height,
        "boardWidthPx": context.board_width,
        "boardHeightPx": context.board_height,
        "leftPaddingPx": context.left_padding,
        "rightPaddingPx": context.right_padding,
        "topPaddingPx": context.top_padding,
        "bottomPaddingPx": context.bottom_padding,
        "gridStepPx": context.sampling_step_px,
        "gridWidth": context.grid_width,
        "gridHeight": context.grid_height,
        "gridXStepPx": context.grid_x_step,
        "gridYStepPx": context.grid_y_step,
        "longitudeExtent": list(context.longitude_extent),
        "latitudeExtent": list(context.latitude_extent),
        "elevationLevels": context.elevation_levels,
        "bathymetryLevels": context.bathymetry_levels,
        "elevationSemantics": "ordinal-palette",
        "continuousSurfaceMethod": ELEVATION_SURFACE_METHOD,
        "elevationRegularization": {
            "algorithm": ELEVATION_REGULARIZATION_ALGORITHM,
            "sigmaGridUnits": _round_relative(
                context.elevation_regularization_sigma_grid_units
            ),
            "passes": int(context.elevation_regularization_passes),
            "validMask": "land-mask",
            "wrapColumns": _edge_policy_wraps_longitude(context.edge_policy),
                "wrapRows": False,
                "input": "palette-position-after-water-inpaint",
                "output": "regularized-input-to-structured-relief-and-two-pass-hydrology",
        },
        "elevationContour": {
            "algorithm": "cell-center-marching-squares",
            "interpolation": "linear-cell-center-threshold-crossings",
            "simplifyToleranceGridUnits": _round_relative(
                context.elevation_contour_simplify_tolerance
            ),
            "smoothingIterations": int(context.smoothing_iterations),
        },
        "simplifyToleranceGridUnits": context.simplify_tolerance,
        "coastlineToleranceGridUnits": context.coastline_tolerance,
        "coordinateDecimals": context.coordinate_decimals,
        "smoothingIterations": context.smoothing_iterations,
        "bathymetrySmoothingIterations": context.bathymetry_smoothing_iterations,
        "riverSmoothingIterations": context.river_smoothing_iterations,
        "landPalette": _hex_palette(np.asarray(context.land_palette, dtype=np.float64)),
        "bathymetryPalette": _hex_palette(
            np.asarray(context.bathymetry_palette, dtype=np.float64)
        ),
        "cleanupPolicies": {
            "waterCandidate": context.water_cleanup_policy.as_record(),
            "elevationBand": context.elevation_cleanup_policy.as_record(),
            "bathymetryBand": context.bathymetry_cleanup_policy.as_record(),
        },
        "lakeRecognition": {
            "minimumAreaFraction": LAKE_COMPONENT_AREA_FRACTION,
            "minimumAbsoluteCells": 4,
            "rule": "compact interior water components below the scale-relative area gate remain hydrology evidence only",
        },
        "lakeIslands": {
            "algorithm": "deterministic-inland-water-organic-islands-v2",
            "minimumAreaFraction": LAKE_ISLAND_MINIMUM_AREA_FRACTION,
            "minimumAbsoluteCells": LAKE_ISLAND_MINIMUM_AREA_CELLS,
            "maximumIslandCountPerLake": LAKE_ISLAND_MAXIMUM_COUNT,
            "shoreClearanceCells": LAKE_ISLAND_SHORE_CLEARANCE_CELLS,
            "elevationRule": "continuous relative relief 0.015 at shore to at most 0.28 at island core",
        },
        "inlandSea": {
            "minimumAreaFraction": INLAND_SEA_MINIMUM_AREA_FRACTION,
            "minimumAbsoluteCells": INLAND_SEA_MINIMUM_AREA_CELLS,
            "rule": "closed inland water above the scale-relative gate shares the maritime bathymetry system and retains an inland-sea identity",
        },
        "shelfArchipelagos": {
            "algorithm": "hierarchical-microcontinent-shelf-arc-hotspot-system-v4",
            "maximumGroups": SHELF_ARCHIPELAGO_MAXIMUM_GROUPS,
            "maximumIslandsPerGroup": SHELF_ARCHIPELAGO_MAXIMUM_ISLANDS_PER_GROUP,
            "minimumShoreDistanceCells": SHELF_ARCHIPELAGO_MINIMUM_SHORE_DISTANCE,
            "maximumShoreDistanceCells": SHELF_ARCHIPELAGO_MAXIMUM_SHORE_DISTANCE,
        },
        "hydrology": {
            **HYDROLOGY_PROVENANCE,
            "streamBurnDepth": _round_relative(context.hydrology_config.stream_burn_depth),
            "streamThresholdFraction": _round_relative(
                context.hydrology_config.stream_threshold_fraction
            ),
            "tributaryThresholdFraction": _round_relative(
                context.hydrology_config.tributary_threshold_fraction
            ),
            "mainstemThresholdFraction": _round_relative(
                context.hydrology_config.mainstem_threshold_fraction
            ),
            "minimumHeadwaterLength": int(context.hydrology_config.minimum_headwater_length),
            "closureBudgetFraction": _round_relative(
                context.hydrology_config.closure_budget_fraction
            ),
            "closureMaxChainFraction": _round_relative(
                context.hydrology_config.closure_max_chain_fraction
            ),
            "minimumStreamSpanFactor": _round_relative(
                context.hydrology_config.minimum_stream_span_factor
            ),
            "mfdExponent": _round_relative(context.hydrology_config.mfd_exponent),
            "valleyWindowFraction": _round_relative(
                context.hydrology_config.valley_window_fraction
            ),
            "valleyDepthFraction": _round_relative(
                context.hydrology_config.valley_depth_fraction
            ),
            "valleySupportThreshold": _round_relative(
                context.hydrology_config.valley_support_threshold
            ),
            "valleySupportDistanceFactor": _round_relative(
                context.hydrology_config.valley_support_distance_factor
            ),
            "parallelSearchRadiusFactor": _round_relative(
                context.hydrology_config.parallel_search_radius_factor
            ),
            "parallelPriorityWeight": _round_relative(
                context.hydrology_config.parallel_priority_weight
            ),
            "runoffWeightFloor": _round_relative(
                context.hydrology_config.runoff_weight_floor
            ),
            "snowmeltRunoffBonus": _round_relative(
                context.hydrology_config.snowmelt_runoff_bonus
            ),
            "headwaterUplandQuantile": _round_relative(
                context.hydrology_config.headwater_upland_quantile
            ),
            "lowlandHeadwaterFlowMultiplier": _round_relative(
                context.hydrology_config.lowland_headwater_flow_multiplier
            ),
            "majorRiversPerContinent": int(
                context.hydrology_config.major_rivers_per_continent
            ),
            "majorRiverTributaries": int(
                context.hydrology_config.major_river_tributaries
            ),
            "continentMinimumLandFraction": _round_relative(
                context.hydrology_config.continent_minimum_land_fraction
            ),
            "majorRiverMinimumSpanFactor": _round_relative(
                context.hydrology_config.major_river_minimum_span_factor
            ),
        },
        "snowline": {
            "equatorThreshold": _round_relative(context.snowline_config.equator_threshold),
            "poleThreshold": _round_relative(context.snowline_config.pole_threshold),
            "exponent": _round_relative(context.snowline_config.exponent),
            "maxOffset": _round_relative(context.snowline_config.max_offset),
            "polarFullSnowLatitude": _round_relative(
                context.snowline_config.polar_full_snow_latitude
            ),
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _generation_software() -> dict[str, Any]:
    """Hash the exact physical-field implementation used by this build."""

    source_directory = Path(__file__).resolve().parent
    sources = {
        name: sha256(source_directory / name)
        for name in (
            "build_source.py",
            "climate.py",
            "hydrology.py",
            "polar_terrain.py",
        )
    }
    bundle_payload = "".join(f"{name}:{sources[name]}\n" for name in sorted(sources))
    return {
        "sources": sources,
        "bundleSha256": hashlib.sha256(bundle_payload.encode("utf-8")).hexdigest().upper(),
    }


def _connected_components(
    mask: np.ndarray,
    connectivity: int = 8,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """Return connected-component labels and deterministic cell lists."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise SourceBuildError("connected-component mask must be a two-dimensional array")
    if connectivity not in {4, 8}:
        raise SourceBuildError("connected-component connectivity must be 4 or 8")
    height, width = mask.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    components: list[list[tuple[int, int]]] = []
    neighbours = (
        ((-1, 0), (1, 0), (0, -1), (0, 1))
        if connectivity == 4
        else tuple(
            (row_delta, column_delta)
            for row_delta in (-1, 0, 1)
            for column_delta in (-1, 0, 1)
            if row_delta or column_delta
        )
    )

    for row in range(height):
        for column in range(width):
            if not mask[row, column] or labels[row, column] >= 0:
                continue
            component_id = len(components)
            queue: deque[tuple[int, int]] = deque([(row, column)])
            labels[row, column] = component_id
            cells: list[tuple[int, int]] = []
            while queue:
                current_row, current_column = queue.popleft()
                cells.append((current_row, current_column))
                for row_delta, column_delta in neighbours:
                    next_row = current_row + row_delta
                    next_column = current_column + column_delta
                    if not (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and mask[next_row, next_column]
                        and labels[next_row, next_column] < 0
                    ):
                        continue
                    labels[next_row, next_column] = component_id
                    queue.append((next_row, next_column))
            components.append(cells)
    return labels, components


def _component_metrics(
    cells: Sequence[tuple[int, int]],
) -> tuple[float, float, float, int]:
    """Return area, aspect, fill, and the narrowest local occupied width."""

    area, aspect, fill = _component_shape(cells)
    row_counts = Counter(row for row, _ in cells)
    column_counts = Counter(column for _, column in cells)
    local_width = min(
        max(row_counts.values(), default=0),
        max(column_counts.values(), default=0),
    )
    return area, aspect, fill, int(local_width)


def _component_bbox(cells: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return an exclusive ``(min_column, min_row, max_column, max_row)`` box."""

    if not cells:
        raise SourceBuildError("cannot calculate a bounding box for an empty component")
    rows = [row for row, _ in cells]
    columns = [column for _, column in cells]
    return (min(columns), min(rows), max(columns) + 1, max(rows) + 1)


def _edge_component_ids(labels: np.ndarray) -> set[int]:
    """Return labels touching any board edge without coordinate assumptions."""

    if labels.ndim != 2:
        raise SourceBuildError("component labels must be two-dimensional")
    height, width = labels.shape
    edge_labels = np.concatenate(
        (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
    )
    return {int(component_id) for component_id in edge_labels if int(component_id) >= 0}


def _resolve_cleanup_thresholds(
    mask: np.ndarray,
    policy: RasterCleanupPolicy,
    reference_cell_count: int | None,
) -> tuple[int, int, int, int]:
    """Resolve scale-relative gates against this sampled board.

    Explicit integer floors remain useful for small synthetic tests, while a
    fractional policy scales with the authored land/ocean population when a
    grid resolution changes.  The caller may provide a shared population
    (for example full land count for every elevation threshold) so high bands
    do not get progressively weaker merely because they contain fewer cells.
    """

    if reference_cell_count is not None and (
        isinstance(reference_cell_count, bool) or reference_cell_count <= 0
    ):
        raise SourceBuildError("cleanup reference_cell_count must be a positive integer")
    population = max(
        1,
        int(reference_cell_count)
        if reference_cell_count is not None
        else int(np.count_nonzero(mask)),
    )
    board_span = max(1, min(int(mask.shape[0]), int(mask.shape[1])))

    def resolve_cells(floor: int, fraction: float | None) -> int:
        if fraction is None:
            return int(floor)
        return max(int(floor), int(math.ceil(float(fraction) * population)))

    def resolve_width(floor: int, fraction: float | None) -> int:
        if fraction is None:
            return int(floor)
        return max(int(floor), int(math.ceil(float(fraction) * board_span)))

    return (
        resolve_cells(policy.min_component_cells, policy.min_component_area_fraction),
        resolve_width(policy.min_component_width, policy.min_component_width_fraction),
        resolve_cells(policy.min_hole_cells, policy.min_hole_area_fraction),
        resolve_width(policy.min_hole_width, policy.min_hole_width_fraction),
    )


def cleanup_raster_mask(
    mask: np.ndarray,
    policy: RasterCleanupPolicy,
    *,
    persistence_mask: np.ndarray | None = None,
    reference_cell_count: int | None = None,
) -> tuple[np.ndarray, RasterCleanupStats]:
    """Clean semantic raster components before polygonization.

    Components are retained when they are board-connected, overlap an
    authored feature that persists into a neighbouring ordinal layer, or pass
    all configured area/width/fill/aspect gates.  Interior complement
    components that are tiny or needle-thin are filled as pinholes.  Every
    decision is deterministic and based only on raster topology and relative
    shape metrics; no map coordinate or source-specific exception is used.
    """

    raw = np.asarray(mask, dtype=bool)
    if raw.ndim != 2:
        raise SourceBuildError("raster cleanup mask must be a two-dimensional array")
    if not isinstance(policy, RasterCleanupPolicy):
        raise SourceBuildError("raster cleanup requires a RasterCleanupPolicy")
    if persistence_mask is not None:
        persistence = np.asarray(persistence_mask, dtype=bool)
        if persistence.shape != raw.shape:
            raise SourceBuildError("raster cleanup persistence mask shape must match mask")
    else:
        persistence = None

    min_component_cells, min_component_width, min_hole_cells, min_hole_width = (
        _resolve_cleanup_thresholds(raw, policy, reference_cell_count)
    )
    labels, components = _connected_components(raw, policy.connectivity)
    edge_ids = _edge_component_ids(labels) if policy.preserve_edge_components else set()
    cleaned = np.zeros_like(raw, dtype=bool)
    retained_components = 0
    removed_component_bboxes: list[tuple[int, int, int, int]] = []

    for component_id, cells in enumerate(components):
        area, aspect, fill, local_width = _component_metrics(cells)
        component_mask = labels == component_id
        persistent = persistence is not None and bool(np.any(component_mask & persistence))
        keep = (
            component_id in edge_ids
            or persistent
            or (
                area >= min_component_cells
                and local_width >= min_component_width
                and fill >= policy.min_component_fill
                and aspect <= policy.max_component_aspect
            )
        )
        if keep:
            cleaned[component_mask] = True
            retained_components += 1
        else:
            removed_component_bboxes.append(_component_bbox(cells))

    # A complement component is a hole only when it does not touch the board.
    # Tiny islands of complement noise and one-cell cracks are removed, while
    # a broad interior water body remains untouched.
    filled_holes = 0
    filled_hole_bboxes: list[tuple[int, int, int, int]] = []
    if min_hole_cells > 0 or min_hole_width > 0:
        inverse_labels, inverse_components = _connected_components(
            ~cleaned,
            policy.connectivity,
        )
        inverse_edge_ids = _edge_component_ids(inverse_labels)
        for component_id, cells in enumerate(inverse_components):
            if component_id in inverse_edge_ids:
                continue
            area, _aspect, _fill, local_width = _component_metrics(cells)
            if area <= min_hole_cells or (
                min_hole_width > 0 and local_width < min_hole_width
            ):
                cleaned[inverse_labels == component_id] = True
                filled_holes += 1
                filled_hole_bboxes.append(_component_bbox(cells))

    report = RasterCleanupStats(
        semantic=policy.semantic,
        connectivity=policy.connectivity,
        input_components=len(components),
        retained_components=retained_components,
        removed_components=len(components) - retained_components,
        filled_holes=filled_holes,
        input_cells=int(np.count_nonzero(raw)),
        output_cells=int(np.count_nonzero(cleaned)),
        resolved_min_component_cells=min_component_cells,
        resolved_min_component_width=min_component_width,
        resolved_min_hole_cells=min_hole_cells,
        resolved_min_hole_width=min_hole_width,
        removed_component_bboxes=tuple(removed_component_bboxes),
        filled_hole_bboxes=tuple(filled_hole_bboxes),
    )
    return cleaned, report


def _exterior_component(mask: np.ndarray) -> np.ndarray:
    """Return water connected to the sampled world edge through a true channel.

    Surface-water topology uses four-neighbour connectivity: two blue cells
    meeting only at a corner do not form a navigable strait.
    """

    labels, components = _connected_components(mask, connectivity=4)
    height, width = mask.shape
    exterior_ids: set[int] = set()
    for row in range(height):
        for column in (0, width - 1):
            component_id = int(labels[row, column])
            if component_id >= 0:
                exterior_ids.add(component_id)
    for column in range(width):
        for row in (0, height - 1):
            component_id = int(labels[row, column])
            if component_id >= 0:
                exterior_ids.add(component_id)
    return np.isin(labels, sorted(exterior_ids))


def _promote_narrowly_connected_basins(
    water: np.ndarray,
    broad_ocean: np.ndarray,
    *,
    minimum_basin_area: int,
) -> np.ndarray:
    """Restore real seas behind narrow straits without turning rivers into bays.

    The broad-ocean pass intentionally removes one-cell blue lines so authored
    rivers cannot tear marine corridors through a continent.  A compact water
    basin attached to that ocean is different: the basin proves that the thin
    reach is a strait, not a river.  Promote the whole connected residual basin
    in that case, including its neck.
    """

    surface_water = np.asarray(water, dtype=bool)
    ocean = np.asarray(broad_ocean, dtype=bool)
    if surface_water.shape != ocean.shape or surface_water.ndim != 2:
        raise SourceBuildError("surface water and broad ocean must share a 2D shape")
    if isinstance(minimum_basin_area, bool) or int(minimum_basin_area) < 1:
        raise SourceBuildError("minimum basin area must be a positive integer")

    result = ocean.copy()
    residual = surface_water & ~ocean
    _labels, components = _connected_components(residual, connectivity=4)
    height, width = surface_water.shape
    for component in components:
        if not _is_lake(component, minimum_area=int(minimum_basin_area)):
            continue
        touches_ocean = False
        for row, column in component:
            for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row = row + row_delta
                next_column = column + column_delta
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and ocean[next_row, next_column]
                ):
                    touches_ocean = True
                    break
            if touches_ocean:
                break
        if touches_ocean:
            for row, column in component:
                result[row, column] = True
    return result


def _enclosed_ocean_mask(ocean: np.ndarray) -> np.ndarray:
    """Return marine cells cut off from the world ocean by generated land."""

    water = np.asarray(ocean, dtype=bool)
    if water.ndim != 2:
        raise SourceBuildError("ocean mask must be two-dimensional")
    return water & ~_exterior_component(water)


def _wide_exterior_water(
    mask: np.ndarray,
    *,
    minimum_corridor_width: int = 3,
    wrap_columns: bool = False,
) -> np.ndarray:
    """Return board-connected water without admitting one-cell line evidence.

    Palette-blue marks in an authored physical reference can represent rivers
    or valley cues as well as open water.  A plain edge flood-fill mistakes a
    line that touches the ocean for a sea inlet and tears it through the land
    mask.  An opening on the sampled water domain retains water corridors at
    least ``minimum_corridor_width`` cells wide, then restores their original
    shoreline footprint.  Removed thin branches remain in ``water_candidate``
    and are handled later as hydrology constraints, never as ocean.
    """

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise SourceBuildError("wide exterior water mask must be two-dimensional")
    if (
        isinstance(minimum_corridor_width, bool)
        or not isinstance(minimum_corridor_width, int)
        or minimum_corridor_width < 1
        or minimum_corridor_width % 2 == 0
    ):
        raise SourceBuildError("minimum ocean corridor width must be a positive odd integer")
    if not isinstance(wrap_columns, bool):
        raise SourceBuildError("wide exterior water wrap_columns must be boolean")

    radius = (minimum_corridor_width - 1) // 2
    if radius == 0:
        return _exterior_component(source)

    def neighbourhood_reduce(values: np.ndarray, *, require_all: bool) -> np.ndarray:
        row_padding = np.ones((radius, values.shape[1]), dtype=bool) if require_all else np.zeros((radius, values.shape[1]), dtype=bool)
        vertical = np.concatenate((row_padding, values, row_padding), axis=0)
        if wrap_columns:
            horizontal = np.concatenate(
                (vertical[:, -radius:], vertical, vertical[:, :radius]), axis=1
            )
        else:
            column_padding = np.ones((vertical.shape[0], radius), dtype=bool) if require_all else np.zeros((vertical.shape[0], radius), dtype=bool)
            horizontal = np.concatenate((column_padding, vertical, column_padding), axis=1)
        result = np.ones_like(values, dtype=bool) if require_all else np.zeros_like(values, dtype=bool)
        for row_offset in range(2 * radius + 1):
            for column_offset in range(2 * radius + 1):
                window = horizontal[
                    row_offset : row_offset + values.shape[0],
                    column_offset : column_offset + values.shape[1],
                ]
                if require_all:
                    result &= window
                else:
                    result |= window
        return result

    eroded = neighbourhood_reduce(source, require_all=True)
    exterior_core = _exterior_component(eroded)
    # Re-expand only from the broad exterior core and clip back to original
    # water evidence.  This retains genuine coastal fringes but cannot cross a
    # narrow river-like branch that did not survive erosion.
    return neighbourhood_reduce(exterior_core, require_all=False) & source


def _boundary_segments(
    mask: np.ndarray,
    adjacent: np.ndarray | None = None,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Trace the boundary of a categorical raster with run-length edges.

    Each edge is emitted once.  Internal edges between two true cells are not
    emitted, so the resulting line network is suitable for polygonize/linemerge.
    If ``adjacent`` is supplied, only edges next to that second mask are kept;
    this is used to isolate the ocean coastline from lake boundaries.
    """

    height, width = mask.shape
    segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def add(start: tuple[int, int], end: tuple[int, int]) -> None:
        ordered = (start, end) if start <= end else (end, start)
        segments.add(ordered)

    for row, column in zip(*np.where(mask), strict=True):
        row = int(row)
        column = int(column)
        if row == 0 or not mask[row - 1, column]:
            if adjacent is None or (row > 0 and adjacent[row - 1, column]):
                add((column, row), (column + 1, row))
        if row == height - 1 or not mask[row + 1, column]:
            if adjacent is None or (row + 1 < height and adjacent[row + 1, column]):
                add((column + 1, row + 1), (column, row + 1))
        if column == 0 or not mask[row, column - 1]:
            if adjacent is None or (column > 0 and adjacent[row, column - 1]):
                add((column, row + 1), (column, row))
        if column == width - 1 or not mask[row, column + 1]:
            if adjacent is None or (
                column + 1 < width and adjacent[row, column + 1]
            ):
                add((column + 1, row), (column + 1, row + 1))
    return _run_length_segments(sorted(segments))


def _run_length_segments(
    segments: Sequence[tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Coalesce adjacent axial pixel edges before polygonization."""

    horizontal: dict[int, list[int]] = {}
    vertical: dict[int, list[int]] = {}
    for start, end in segments:
        if start[1] == end[1]:
            horizontal.setdefault(start[1], []).append(min(start[0], end[0]))
        elif start[0] == end[0]:
            vertical.setdefault(start[0], []).append(min(start[1], end[1]))
        else:
            raise SourceBuildError(f"non-axial boundary edge: {start} -> {end}")

    merged: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def merge_runs(groups: dict[int, list[int]], horizontal_axis: bool) -> None:
        for fixed, starts in groups.items():
            ordered = sorted(set(starts))
            if not ordered:
                continue
            run_start = run_end = ordered[0]
            for next_start in ordered[1:]:
                if next_start == run_end + 1:
                    run_end = next_start
                    continue
                if horizontal_axis:
                    merged.append(((run_start, fixed), (run_end + 1, fixed)))
                else:
                    merged.append(((fixed, run_start), (fixed, run_end + 1)))
                run_start = run_end = next_start
            if horizontal_axis:
                merged.append(((run_start, fixed), (run_end + 1, fixed)))
            else:
                merged.append(((fixed, run_start), (fixed, run_end + 1)))

    merge_runs(horizontal, horizontal_axis=True)
    merge_runs(vertical, horizontal_axis=False)
    return sorted(merged)


def _sampled_polygonize(mask: np.ndarray) -> Polygon | MultiPolygon:
    """Polygonize a raster mask while retaining holes and disconnected islands."""

    segments = _boundary_segments(mask)
    if not segments:
        return MultiPolygon()
    faces = polygonize([LineString([start, end]) for start, end in segments])
    polygons: list[Polygon] = []
    height, width = mask.shape
    for face in faces:
        point = face.representative_point()
        column = min(width - 1, max(0, int(np.floor(point.x))))
        row = min(height - 1, max(0, int(np.floor(point.y))))
        if mask[row, column]:
            polygons.append(face)
    if not polygons:
        # Polygonize represents the unbounded side of a board-touching mask
        # from the opposite face.  The ocean therefore has a valid outer face
        # whose representative sample is land and would be rejected above.
        # Vectorize the bounded complement, then subtract it from the board so
        # edge-connected ocean and narrow straits remain a real polygon.
        touches_board = bool(
            mask[0].any()
            or mask[-1].any()
            or mask[:, 0].any()
            or mask[:, -1].any()
        )
        if touches_board:
            complement = _sampled_polygonize(~mask)
            board = box(0, 0, width, height)
            if complement.is_empty:
                return board
            return _polygon_only(make_valid(board.difference(complement)))
        # 8-connected diagonal components do not share an axial edge, so the
        # run-length line network has no bounded face even though every cell
        # is an accepted component cell.  Preserve those cells as independent
        # polygons rather than silently dropping the component.
        return _polygon_only(
            make_valid(
                unary_union(
                    [
                        box(column, row, column + 1, row + 1)
                        for row, column in zip(*np.where(mask), strict=True)
                    ]
                )
            )
        )
    try:
        geometry = make_valid(unary_union(polygons))
    except Exception:
        # A coarse diagonal saddle can create a non-noded face network. The
        # deterministic cell union is the topology-safe recovery path.
        geometry = MultiPolygon()
    # Diagonal one-cell contacts can make polygonize faces ambiguous at a
    # coarse authoring step. Fall back to a cell union when its area audit
    # disagrees; the primary path remains run-length boundary polygonization.
    if abs(float(geometry.area) - float(mask.sum())) > 1e-6:
        geometry = unary_union(
            [
                box(column, row, column + 1, row + 1)
                for row, column in zip(*np.where(mask), strict=True)
            ]
        )
    return _polygon_only(geometry)


def _polygon_only(geometry: Any) -> Polygon | MultiPolygon:
    if geometry.is_empty:
        return MultiPolygon()
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = [part for part in geometry.geoms if isinstance(part, Polygon)]
        return MultiPolygon(polygons) if polygons else MultiPolygon()
    return MultiPolygon()


def _line_only(geometry: Any) -> LineString | MultiLineString:
    if geometry.is_empty:
        return MultiLineString()
    if isinstance(geometry, LineString):
        return geometry
    if isinstance(geometry, MultiLineString):
        return geometry
    if isinstance(geometry, GeometryCollection):
        lines = [part for part in geometry.geoms if isinstance(part, LineString)]
        return MultiLineString(lines) if lines else MultiLineString()
    return MultiLineString()


def _exterior_lines(geometry: Polygon | MultiPolygon) -> MultiLineString:
    """Return closed outer coast rings without including inland-water holes."""

    geometry = _polygon_only(geometry)
    polygons = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    rings: list[LineString] = []
    for polygon in polygons:
        ring = LineString(polygon.exterior.coords)
        if not ring.is_empty and ring.length > 0:
            rings.append(ring)
    return MultiLineString(rings) if rings else MultiLineString()


def _chaikin_closed_coordinates(
    coordinates: Sequence[Sequence[float]],
    iterations: int,
) -> list[tuple[float, float]]:
    """Round a closed ring while keeping its first vertex as its closure."""

    points = [(float(x), float(y)) for x, y in coordinates[:-1]]
    if len(points) < 3:
        return [(float(x), float(y)) for x, y in coordinates]
    for _ in range(iterations):
        rounded: list[tuple[float, float]] = []
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            rounded.extend(
                (
                    (
                        0.75 * point[0] + 0.25 * next_point[0],
                        0.75 * point[1] + 0.25 * next_point[1],
                    ),
                    (
                        0.25 * point[0] + 0.75 * next_point[0],
                        0.25 * point[1] + 0.75 * next_point[1],
                    ),
                )
            )
        points = rounded
    return [*points, points[0]]


def _polygon_parts(geometry: Polygon | MultiPolygon) -> list[Polygon]:
    geometry = _polygon_only(geometry)
    return [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)


def _smooth_polygon_geometry(
    geometry: Polygon | MultiPolygon,
    context: BuildContext,
    iterations: int | None = None,
    preserve_board_edges: bool = False,
) -> Polygon | MultiPolygon:
    """Chaikin-smooth every ring and repair the result without dropping islands."""

    if iterations is None:
        iterations = context.smoothing_iterations
    rounded_parts: list[Polygon] = []
    for polygon in _polygon_parts(geometry):
        exterior = list(polygon.exterior.coords)
        touches_board_edge = preserve_board_edges and any(
            x in {0.0, float(context.grid_width)}
            or y in {0.0, float(context.grid_height)}
            for x, y in exterior
        )
        shell = exterior if touches_board_edge else _chaikin_closed_coordinates(exterior, iterations)
        holes = [
            _chaikin_closed_coordinates(list(interior.coords), iterations)
            for interior in polygon.interiors
        ]
        candidate = Polygon(shell, holes)
        if candidate.is_empty or candidate.area <= 0:
            candidate = polygon
        rounded_parts.append(candidate)
    if not rounded_parts:
        return MultiPolygon()
    repaired = make_valid(unary_union(rounded_parts))
    return _polygon_only(repaired)


def _resize_masked_surface(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    factor: int,
    wrap_columns: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Upsample a masked surface with normalized bilinear interpolation.

    ``Image.Resampling.BILINEAR`` supplies deterministic subcell interpolation
    without adding a numerical dependency.  The value and mask weights are
    resized separately and divided, so an ocean sentinel cannot enter a land
    contour.  The returned nearest mask is deliberately kept conservative: an
    interpolated contour may move within a land cell but never creates land
    outside the original land domain.
    """

    source = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if source.ndim != 2 or source.shape != mask.shape:
        raise SourceBuildError("masked surface resize arrays must share a 2D shape")
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)):
        raise SourceBuildError("masked surface resize factor must be a positive integer")
    factor = int(factor)
    if factor < 1:
        raise SourceBuildError("masked surface resize factor must be a positive integer")
    if not isinstance(wrap_columns, bool):
        raise SourceBuildError("masked surface resize wrap_columns must be boolean")
    if factor == 1:
        return source.copy(), mask.copy()
    if bool(np.any(mask & ~np.isfinite(source))):
        raise SourceBuildError("masked surface resize valid values must be finite")

    height, width = source.shape

    def resize_float(array: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        return np.asarray(
            Image.fromarray(np.asarray(array, dtype=np.float32), mode="F").resize(
                (target_width, target_height),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float64,
        )

    def resize_mask(array: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        return np.asarray(
            Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").resize(
                (target_width, target_height),
                Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        ) > 0

    weighted = np.where(mask, source, 0.0)
    mask_values = mask.astype(np.float64)
    if wrap_columns:
        weighted = np.concatenate((weighted[:, -1:], weighted, weighted[:, :1]), axis=1)
        mask_values = np.concatenate((mask_values[:, -1:], mask_values, mask_values[:, :1]), axis=1)
        resized_width = (width + 2) * factor
        resized_weighted = resize_float(weighted, resized_width, height * factor)
        resized_weights = resize_float(mask_values, resized_width, height * factor)
        start = factor
        end = start + width * factor
        resized_weighted = resized_weighted[:, start:end]
        resized_weights = resized_weights[:, start:end]
        expanded_mask_source = np.concatenate(
            (mask[:, -1:], mask, mask[:, :1]),
            axis=1,
        )
        expanded_mask = resize_mask(
            expanded_mask_source,
            resized_width,
            height * factor,
        )[:, start:end]
    else:
        resized_weighted = resize_float(weighted, width * factor, height * factor)
        resized_weights = resize_float(mask_values, width * factor, height * factor)
        expanded_mask = resize_mask(mask, width * factor, height * factor)

    result = np.zeros(resized_weighted.shape, dtype=np.float64)
    np.divide(
        resized_weighted,
        resized_weights,
        out=result,
        where=resized_weights > 1.0e-12,
    )
    return np.clip(result, 0.0, 1.0), expanded_mask


class _ContinuousContourEngine:
    """Reusable C++ contour extractor for one masked continuous field."""

    def __init__(
        self,
        surface: np.ndarray,
        valid: np.ndarray,
        *,
        upsample_factor: int,
        wrap_columns: bool,
    ) -> None:
        values = np.asarray(surface, dtype=np.float64)
        valid_mask = np.asarray(valid, dtype=bool)
        if values.ndim != 2 or values.shape != valid_mask.shape:
            raise SourceBuildError("continuous contour engine arrays must share a 2D shape")
        if isinstance(upsample_factor, bool) or not isinstance(
            upsample_factor, (int, np.integer)
        ):
            raise SourceBuildError("continuous contour engine upsample factor must be an integer")
        self.factor = int(upsample_factor)
        if self.factor < 1:
            raise SourceBuildError("continuous contour engine upsample factor must be positive")
        self.source_shape = values.shape
        self.base_geometry = _sampled_polygonize(valid_mask)
        if self.factor == 1:
            self.surface = values.copy()
            self.valid = valid_mask.copy()
        else:
            self.surface, self.valid = _resize_masked_surface(
                values,
                valid_mask,
                factor=self.factor,
                wrap_columns=wrap_columns,
            )
        height, width = self.surface.shape
        x = (np.arange(width, dtype=np.float64) + 0.5) / self.factor
        y = (np.arange(height, dtype=np.float64) + 0.5) / self.factor
        masked_surface = np.ma.array(self.surface, mask=~self.valid)
        self.generator = contour_generator(
            x=x,
            y=y,
            z=masked_surface,
            name="serial",
            line_type="Separate",
            corner_mask=True,
        )

    def _expanded_gate(self, mask: np.ndarray) -> np.ndarray:
        gate = np.asarray(mask, dtype=bool)
        if gate.shape != self.source_shape:
            raise SourceBuildError("continuous contour gate shape does not match its engine")
        if self.factor == 1:
            return gate.copy()
        return np.asarray(
            Image.fromarray(gate.astype(np.uint8), mode="L").resize(
                (
                    gate.shape[1] * self.factor,
                    gate.shape[0] * self.factor,
                ),
                Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        ) > 0

    def threshold_geometry(
        self,
        threshold_mask: np.ndarray,
        threshold: float,
    ) -> Polygon | MultiPolygon:
        """Return a cumulative polygon for one threshold without rewalking cells."""

        if not math.isfinite(float(threshold)):
            raise SourceBuildError("continuous contour threshold must be finite")
        expanded_gate = self._expanded_gate(threshold_mask)
        high = self.valid & expanded_gate
        if not bool(np.any(high)):
            return MultiPolygon()
        if bool(np.all(expanded_gate[self.valid])):
            return self.base_geometry

        line_strings: list[LineString] = []
        for coordinates in self.generator.lines(float(threshold)):
            if len(coordinates) < 2:
                continue
            line_strings.append(LineString(np.asarray(coordinates, dtype=np.float64)))
        if not line_strings:
            return _polygon_only(
                make_valid(_sampled_polygonize(np.asarray(threshold_mask, dtype=bool)))
            )

        boundary = self.base_geometry.boundary
        boundary_links: list[LineString] = []
        for line in line_strings:
            if line.is_ring:
                continue
            for endpoint in (Point(line.coords[0]), Point(line.coords[-1])):
                nearest_boundary, nearest_endpoint = nearest_points(boundary, endpoint)
                if nearest_boundary.distance(nearest_endpoint) <= 1.0e-9:
                    continue
                boundary_links.append(LineString([endpoint, nearest_boundary]))
        linework = unary_union([boundary, *line_strings, *boundary_links])
        faces = list(polygonize(linework))
        if not faces:
            return _polygon_only(
                make_valid(_sampled_polygonize(np.asarray(threshold_mask, dtype=bool)))
            )

        height, width = self.surface.shape
        selected: list[Polygon] = []
        for face in faces:
            representative = face.representative_point()
            column = min(
                width - 1,
                max(0, int(math.floor(float(representative.x) * self.factor))),
            )
            row = min(
                height - 1,
                max(0, int(math.floor(float(representative.y) * self.factor))),
            )
            if (
                expanded_gate[row, column]
                and self.valid[row, column]
                and float(self.surface[row, column]) >= float(threshold)
                and self.base_geometry.covers(representative)
            ):
                selected.append(face)
        if not selected:
            return _polygon_only(
                make_valid(_sampled_polygonize(np.asarray(threshold_mask, dtype=bool)))
            )
        return _polygon_only(
            make_valid(unary_union(selected).intersection(self.base_geometry))
        )


def _continuous_component_geometry(
    cells: Sequence[tuple[int, int]],
    shape: tuple[int, int],
    *,
    upsample_factor: int = 4,
) -> Polygon | MultiPolygon:
    """Extract one raster component as a bounded continuous shoreline.

    The component cells remain the semantic source of truth.  A padded local
    occupancy field is bilinearly upsampled, lightly regularized at sub-cell
    scale, and contoured at an area-matched threshold.  This removes raster
    stair steps without inventing a new lake or moving its shore by several
    authoring cells.
    """

    if (
        len(shape) != 2
        or isinstance(shape[0], bool)
        or isinstance(shape[1], bool)
        or int(shape[0]) <= 0
        or int(shape[1]) <= 0
    ):
        raise SourceBuildError("continuous component shape must be positive and two-dimensional")
    if isinstance(upsample_factor, bool) or not isinstance(
        upsample_factor, (int, np.integer)
    ):
        raise SourceBuildError("continuous component upsample factor must be an integer")
    upsample_factor = int(upsample_factor)
    if upsample_factor < 2:
        raise SourceBuildError("continuous component upsample factor must be at least two")
    normalized_cells = [(int(row), int(column)) for row, column in cells]
    if not normalized_cells:
        return MultiPolygon()
    height, width = int(shape[0]), int(shape[1])
    if any(not (0 <= row < height and 0 <= column < width) for row, column in normalized_cells):
        raise SourceBuildError("continuous component cell lies outside its raster shape")

    rows = [row for row, _ in normalized_cells]
    columns = [column for _, column in normalized_cells]
    padding = 3
    row_start = max(0, min(rows) - padding)
    row_end = min(height, max(rows) + padding + 1)
    column_start = max(0, min(columns) - padding)
    column_end = min(width, max(columns) + padding + 1)
    # Always keep an explicit dry rim, including components that touch the
    # authoring board, so the extracted ring cannot close against the crop.
    local_height = row_end - row_start + 2
    local_width = column_end - column_start + 2
    occupancy = np.zeros((local_height, local_width), dtype=np.float64)
    for row, column in normalized_cells:
        occupancy[row - row_start + 1, column - column_start + 1] = 1.0

    valid = np.ones(occupancy.shape, dtype=bool)
    expanded, expanded_valid = _resize_masked_surface(
        occupancy,
        valid,
        factor=upsample_factor,
        wrap_columns=False,
    )
    regularized = _masked_gaussian_smooth(
        expanded,
        expanded_valid,
        sigma=max(1.0, upsample_factor * 0.32),
        passes=1,
        wrap_columns=False,
    )
    target_area = float(len(normalized_cells) * upsample_factor * upsample_factor)
    low = 0.25
    high = 0.75
    best_geometry: Polygon | MultiPolygon = MultiPolygon()
    best_error = math.inf
    for _ in range(12):
        threshold = (low + high) * 0.5
        threshold_mask = regularized >= threshold
        candidate = _marching_square_threshold_geometry(
            regularized,
            expanded_valid,
            threshold_mask,
            threshold,
        )
        candidate = _polygon_only(make_valid(candidate))
        if candidate.is_empty:
            high = threshold
            continue
        error = abs(float(candidate.area) - target_area)
        if error < best_error:
            best_error = error
            best_geometry = candidate
        if float(candidate.area) > target_area:
            low = threshold
        else:
            high = threshold
    if best_geometry.is_empty:
        raise SourceBuildError("continuous component contour extraction produced no geometry")

    geometry = affine_scale(
        best_geometry,
        xfact=1.0 / upsample_factor,
        yfact=1.0 / upsample_factor,
        origin=(0.0, 0.0),
    )
    geometry = affine_translate(
        geometry,
        xoff=float(column_start - 1),
        yoff=float(row_start - 1),
    )
    geometry = _polygon_only(make_valid(geometry))
    if geometry.is_empty or not geometry.is_valid:
        raise SourceBuildError("continuous component contour extraction produced invalid geometry")
    return geometry


def _marching_square_threshold_geometry(
    surface: np.ndarray,
    valid: np.ndarray,
    threshold_mask: np.ndarray,
    threshold: float,
) -> Polygon | MultiPolygon:
    """Refine a coarse mask with cell-centre marching-square contours.

    ``threshold_mask`` remains the authoritative raster-cleaned cell-domain
    gate.  Only the boundaries between neighbouring valid cell-centre samples
    are refined; the valid-land polygon boundary is retained as a coastline
    topology guard.  This avoids constructing a full-resolution boolean raster
    while still placing contour vertices at linearly interpolated subcell
    positions.
    """

    values = np.asarray(surface, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=bool)
    coarse_mask = np.asarray(threshold_mask, dtype=bool)
    if (
        values.ndim != 2
        or values.shape != valid_mask.shape
        or values.shape != coarse_mask.shape
    ):
        raise SourceBuildError("marching-square arrays must share a 2D shape")
    if not math.isfinite(float(threshold)):
        raise SourceBuildError("marching-square threshold must be finite")
    base_geometry = _sampled_polygonize(valid_mask)
    if base_geometry.is_empty:
        return MultiPolygon()
    high_mask = valid_mask & coarse_mask
    if not bool(np.any(high_mask)):
        return MultiPolygon()
    if bool(np.all(coarse_mask[valid_mask])):
        return base_geometry

    def coarse_geometry() -> Polygon | MultiPolygon:
        """Return the cleaned raster gate without ever expanding it."""

        return _polygon_only(
            make_valid(_sampled_polygonize(high_mask).intersection(base_geometry))
        )

    height, width = values.shape
    if height < 2 or width < 2:
        return coarse_geometry()

    segments: list[LineString] = []
    corner_valid = (
        valid_mask[:-1, :-1]
        & valid_mask[:-1, 1:]
        & valid_mask[1:, 1:]
        & valid_mask[1:, :-1]
    )
    corner_values = np.stack(
        (
            values[:-1, :-1],
            values[:-1, 1:],
            values[1:, 1:],
            values[1:, :-1],
        ),
        axis=0,
    )
    corner_high = corner_values >= float(threshold)
    crossing = corner_valid & (corner_high.sum(axis=0) > 0) & (corner_high.sum(axis=0) < 4)
    crossing_rows, crossing_columns = np.where(crossing)

    def edge_intersection(
        first: tuple[float, float],
        second: tuple[float, float],
        first_value: float,
        second_value: float,
    ) -> tuple[float, float]:
        difference = second_value - first_value
        fraction = 0.5 if abs(difference) <= 1.0e-15 else (float(threshold) - first_value) / difference
        fraction = min(1.0, max(0.0, fraction))
        return (
            round(first[0] + fraction * (second[0] - first[0]), 12),
            round(first[1] + fraction * (second[1] - first[1]), 12),
        )

    for row_value, column_value in zip(crossing_rows, crossing_columns, strict=True):
        row = int(row_value)
        column = int(column_value)
        corner_points = (
            (column + 0.5, row + 0.5),
            (column + 1.5, row + 0.5),
            (column + 1.5, row + 1.5),
            (column + 0.5, row + 1.5),
        )
        values_for_cell = tuple(float(corner_values[index, row, column]) for index in range(4))
        intersections: list[tuple[float, float] | None] = []
        for first_index, second_index in ((0, 1), (1, 2), (2, 3), (3, 0)):
            first_high = values_for_cell[first_index] >= float(threshold)
            second_high = values_for_cell[second_index] >= float(threshold)
            intersections.append(
                edge_intersection(
                    corner_points[first_index],
                    corner_points[second_index],
                    values_for_cell[first_index],
                    values_for_cell[second_index],
                )
                if first_high != second_high
                else None
            )
        crossing_points = [point for point in intersections if point is not None]
        if len(crossing_points) == 2:
            segments.append(LineString(crossing_points))
        elif len(crossing_points) == 4:
            # The bilinear centre is the deterministic asymptotic decider for
            # a saddle.  The selected pairing keeps the higher side together.
            centre_high = sum(values_for_cell) / 4.0 >= float(threshold)
            if centre_high:
                pairs = ((0, 1), (2, 3))
            else:
                pairs = ((1, 2), (3, 0))
            for first_index, second_index in pairs:
                first_point = intersections[first_index]
                second_point = intersections[second_index]
                if first_point is not None and second_point is not None:
                    segments.append(LineString([first_point, second_point]))

    if not segments:
        return coarse_geometry()
    boundary = base_geometry.boundary
    # A contour which reaches the edge of a valid domain has one endpoint in
    # its crossing-cell linework rather than a second contour segment.  Close
    # those open chains against the actual valid boundary before polygonize;
    # otherwise polygonize sees only the boundary and can return the complete
    # land domain for an otherwise half-domain threshold.
    endpoint_counts: Counter[tuple[float, float]] = Counter()
    for segment in segments:
        start, end = segment.coords[0], segment.coords[-1]
        endpoint_counts[(float(start[0]), float(start[1]))] += 1
        endpoint_counts[(float(end[0]), float(end[1]))] += 1
    boundary_links: list[LineString] = []
    for endpoint, count in endpoint_counts.items():
        if count != 1:
            continue
        nearest_boundary, nearest_endpoint = nearest_points(
            boundary,
            Point(endpoint[0], endpoint[1]),
        )
        if nearest_boundary.distance(nearest_endpoint) <= 1.0e-12:
            continue
        boundary_links.append(
            LineString(
                [
                    endpoint,
                    (
                        round(float(nearest_boundary.x), 12),
                        round(float(nearest_boundary.y), 12),
                    ),
                ]
            )
        )
    linework = unary_union([boundary, *segments, *boundary_links])
    faces = list(polygonize(linework))
    if not faces:
        return coarse_geometry()

    def nearest_sample(point_x: float, point_y: float) -> tuple[int, int]:
        sample_column = min(width - 1, max(0, int(math.floor(point_x))))
        sample_row = min(height - 1, max(0, int(math.floor(point_y))))
        return sample_row, sample_column

    def bilinear_value(point_x: float, point_y: float) -> float:
        left = min(width - 1, max(0, int(math.floor(point_x - 0.5))))
        top = min(height - 1, max(0, int(math.floor(point_y - 0.5))))
        right = min(width - 1, left + 1)
        bottom = min(height - 1, top + 1)
        horizontal = min(1.0, max(0.0, point_x - (left + 0.5)))
        vertical = min(1.0, max(0.0, point_y - (top + 0.5)))
        corners = (
            (top, left, (1.0 - horizontal) * (1.0 - vertical)),
            (top, right, horizontal * (1.0 - vertical)),
            (bottom, right, horizontal * vertical),
            (bottom, left, (1.0 - horizontal) * vertical),
        )
        weighted = 0.0
        weight_total = 0.0
        for row, column, weight in corners:
            if not valid_mask[row, column]:
                continue
            weighted += float(values[row, column]) * weight
            weight_total += weight
        if weight_total <= 1.0e-12:
            row, column = nearest_sample(point_x, point_y)
            return float(values[row, column])
        return weighted / weight_total

    selected: list[Polygon] = []
    for face in faces:
        representative = face.representative_point()
        if not base_geometry.covers(representative):
            continue
        sample_row, sample_column = nearest_sample(representative.x, representative.y)
        # Keep raster cleanup/persistence authoritative while using the
        # interpolated field to place the actual boundary inside a cell.
        if (
            coarse_mask[sample_row, sample_column]
            and bilinear_value(representative.x, representative.y) >= float(threshold)
        ):
            selected.append(face)
    if not selected:
        return coarse_geometry()
    refined = _polygon_only(make_valid(unary_union(selected).intersection(base_geometry)))
    return refined if not refined.is_empty else coarse_geometry()


def _repair_nested_geometry(
    geometry: Polygon | MultiPolygon,
    container: Polygon | MultiPolygon,
    *,
    return_stats: bool = False,
) -> (
    Polygon
    | MultiPolygon
    | tuple[Polygon | MultiPolygon, GeometryArtifactCleanupStats]
):
    """Clip, repair, and clean a smoothed layer against its lower layer.

    Intersection can create a new sliver where two otherwise healthy rings
    meet.  The artifact pass deliberately runs after ``make_valid`` so the
    same relative grid thresholds used by vectorization also apply to every
    nested repair.  With ``return_stats=True`` the caller receives the
    provenance needed to audit what that post-nesting pass removed.
    """

    repaired = _polygon_only(make_valid(geometry.intersection(container)))
    cleaned, report = _remove_geometry_artifacts_with_stats(repaired)
    if not cleaned.is_empty:
        if not cleaned.is_valid:
            raise SourceBuildError("nested geometry cleanup produced invalid geometry")
        outside = cleaned.difference(container).area
        if outside > 1e-6:
            raise SourceBuildError(
                f"nested geometry cleanup escaped its container by {outside}"
            )
    if return_stats:
        return cleaned, report
    return cleaned


def _vectorize_polygon_mask(
    mask: np.ndarray,
    context: BuildContext,
    simplify_tolerance: float | None = None,
    preserve_board_edges: bool = False,
    continuous_surface: np.ndarray | None = None,
    threshold: float | None = None,
    continuous_valid: np.ndarray | None = None,
    continuous_upsample_factor: int = 1,
    continuous_engine: _ContinuousContourEngine | None = None,
) -> Polygon | MultiPolygon:
    """Trace, simplify and smooth a raster mask as editable polygons.

    The grid only supplies classification samples.  No grid cell rectangles
    are emitted: run-length boundary tracing creates the source geometry,
    followed by topology-preserving simplification and corner-cut passes.
    When ``continuous_surface`` and ``threshold`` are provided, a masked
    cell-centre marching-square subcell contours are traced before the
    coordinates are returned in grid units.
    """

    if simplify_tolerance is None:
        simplify_tolerance = context.simplify_tolerance
    mask = np.asarray(mask, dtype=bool)
    if (continuous_surface is None) != (threshold is None):
        raise SourceBuildError(
            "continuous contour vectorization requires both surface and threshold"
        )
    if continuous_surface is not None and continuous_valid is None:
        raise SourceBuildError(
            "continuous contour vectorization requires an explicit valid mask"
        )
    if isinstance(continuous_upsample_factor, bool) or not isinstance(
        continuous_upsample_factor, (int, np.integer)
    ):
        raise SourceBuildError("continuous contour upsample factor must be an integer")
    continuous_upsample_factor = int(continuous_upsample_factor)
    if continuous_upsample_factor < 1:
        raise SourceBuildError("continuous contour upsample factor must be positive")
    if continuous_surface is None and continuous_upsample_factor != 1:
        raise SourceBuildError("raster mask vectorization cannot request continuous upsampling")
    if continuous_surface is None:
        raw_geometry = _sampled_polygonize(mask)
    else:
        surface = np.asarray(continuous_surface, dtype=np.float64)
        if surface.shape != mask.shape:
            raise SourceBuildError(
                "continuous contour surface and threshold mask must share a shape"
            )
        valid = np.asarray(continuous_valid, dtype=bool)
        if valid.shape != mask.shape:
            raise SourceBuildError(
                "continuous contour valid mask and threshold mask must share a shape"
            )
        engine = continuous_engine
        if engine is None:
            engine = _ContinuousContourEngine(
                surface,
                valid,
                upsample_factor=continuous_upsample_factor,
                wrap_columns=_edge_policy_wraps_longitude(context.edge_policy),
            )
        elif engine.factor != continuous_upsample_factor:
            raise SourceBuildError(
                "continuous contour engine factor does not match vectorization request"
            )
        raw_geometry = engine.threshold_geometry(mask, float(threshold))
    if raw_geometry.is_empty:
        return MultiPolygon()
    effective_simplify_tolerance = (
        simplify_tolerance * 0.35
        if continuous_surface is not None and continuous_upsample_factor > 1
        else simplify_tolerance
    )
    simplified = _polygon_only(
        make_valid(
            raw_geometry.simplify(
                effective_simplify_tolerance,
                preserve_topology=True,
            )
        )
    )
    if simplified.is_empty:
        simplified = raw_geometry
    smoothed = _smooth_polygon_geometry(
        simplified,
        preserve_board_edges=preserve_board_edges,
        context=context,
        iterations=(1 if continuous_surface is not None and continuous_upsample_factor > 1 else None),
    )
    if smoothed.is_empty:
        smoothed = simplified
    final_geometry = _polygon_only(
        make_valid(
            smoothed.simplify(
                effective_simplify_tolerance
                * (
                    0.2
                    if continuous_surface is not None and continuous_upsample_factor > 1
                    else 0.55
                ),
                preserve_topology=True,
            )
        )
    )
    final_geometry = final_geometry if not final_geometry.is_empty else smoothed
    return _remove_geometry_artifacts(final_geometry)


def _thin_geometry_artifact(
    geometry: Polygon,
    *,
    minimum_area: float,
    minimum_clear_width: float,
) -> bool:
    """Identify a near-zero or needle-thin polygon using relative geometry."""

    if geometry.area < minimum_area:
        return True
    min_x, min_y, max_x, max_y = geometry.bounds
    span = max(max_x - min_x, max_y - min_y)
    if span <= 0.0:
        return True
    # Area/span is a conservative thickness proxy.  A normal one-cell island
    # remains wider than this threshold after one corner-cut pass, whereas a
    # smoothing-generated needle or zero-area sliver does not.
    return geometry.area / span < minimum_clear_width


def _remove_geometry_artifacts(
    geometry: Polygon | MultiPolygon,
    *,
    minimum_area: float = GEOMETRY_ARTIFACT_MIN_AREA_GRID2,
    minimum_hole_area: float = GEOMETRY_ARTIFACT_MIN_AREA_GRID2,
    minimum_clear_width: float = GEOMETRY_ARTIFACT_MIN_CLEAR_WIDTH_GRID,
) -> Polygon | MultiPolygon:
    """Drop only near-zero/needle slivers and pinholes after vector cleanup.

    Raster cleanup is the authoritative semantic pass.  This final guard is
    deliberately conservative and scale-relative to a sampled grid cell; it
    strips invalid repair leftovers without imposing a coordinate exception or
    erasing a regular small island/lake.
    """

    cleaned, _report = _remove_geometry_artifacts_with_stats(
        geometry,
        minimum_area=minimum_area,
        minimum_hole_area=minimum_hole_area,
        minimum_clear_width=minimum_clear_width,
    )
    return cleaned


def _remove_geometry_artifacts_with_stats(
    geometry: Polygon | MultiPolygon,
    *,
    minimum_area: float = GEOMETRY_ARTIFACT_MIN_AREA_GRID2,
    minimum_hole_area: float = GEOMETRY_ARTIFACT_MIN_AREA_GRID2,
    minimum_clear_width: float = GEOMETRY_ARTIFACT_MIN_CLEAR_WIDTH_GRID,
    performed: bool = True,
) -> tuple[Polygon | MultiPolygon, GeometryArtifactCleanupStats]:
    """Run geometry artifact cleanup and return its auditable report."""

    normalized = _polygon_only(make_valid(geometry))
    input_parts = _polygon_parts(normalized)
    input_hole_count = sum(len(polygon.interiors) for polygon in input_parts)
    retained: list[Polygon] = []
    removed_polygon_count = 0
    removed_polygon_area = 0.0
    removed_hole_count = 0
    removed_hole_area = 0.0
    for polygon in input_parts:
        if _thin_geometry_artifact(
            polygon,
            minimum_area=minimum_area,
            minimum_clear_width=minimum_clear_width,
        ):
            removed_polygon_count += 1
            removed_polygon_area += float(polygon.area)
            continue
        holes: list[list[tuple[float, float]]] = []
        for interior in polygon.interiors:
            hole = Polygon(interior)
            if _thin_geometry_artifact(
                hole,
                minimum_area=minimum_hole_area,
                minimum_clear_width=minimum_clear_width,
            ):
                removed_hole_count += 1
                removed_hole_area += float(hole.area)
                continue
            holes.append([(float(x), float(y)) for x, y in interior.coords])
        candidate = Polygon(polygon.exterior.coords, holes)
        if not candidate.is_empty and candidate.area > 0.0:
            retained.append(candidate)
        else:
            removed_polygon_count += 1
            removed_polygon_area += float(polygon.area)
    cleaned = (
        MultiPolygon()
        if not retained
        else _polygon_only(make_valid(unary_union(retained)))
    )
    output_parts = _polygon_parts(cleaned)
    output_hole_count = sum(len(polygon.interiors) for polygon in output_parts)
    report = GeometryArtifactCleanupStats(
        minimum_area=float(minimum_area),
        minimum_hole_area=float(minimum_hole_area),
        minimum_clear_width=float(minimum_clear_width),
        input_part_count=len(input_parts),
        output_part_count=len(output_parts),
        removed_polygon_count=removed_polygon_count,
        removed_polygon_area=removed_polygon_area,
        input_hole_count=input_hole_count,
        output_hole_count=output_hole_count,
        removed_hole_count=removed_hole_count,
        removed_hole_area=removed_hole_area,
        input_area=float(normalized.area),
        output_area=float(cleaned.area),
        performed=performed,
    )
    return cleaned, report


def _edge_mask(shape: tuple[int, int]) -> np.ndarray:
    """Return the outermost cells of a sampled board for edge registration."""

    height, width = shape
    result = np.zeros(shape, dtype=bool)
    result[0, :] = True
    result[-1, :] = True
    result[:, 0] = True
    result[:, -1] = True
    return result


def _edge_policy_wraps_longitude(edge_policy: str) -> bool:
    """Return whether an authoring edge policy opts into longitude wrapping."""

    return "longitude-wrap" in edge_policy


def _water_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    # Surface water is continuous only through an edge.  Treating corner-touching
    # cells as one lake lets isolated one-cell colour flecks hitchhike onto a
    # larger basin and later reappear as apparent colour overflow in SVG.  The
    # general raster cleanup policy keeps its configurable connectivity; this
    # semantic water partition intentionally uses the stricter 4-neighbour rule.
    labels, components = _connected_components(mask, connectivity=4)
    edge_ids = _edge_component_ids(labels)
    inland = np.zeros_like(mask, dtype=bool)
    for component_id, cells in enumerate(components):
        if component_id not in edge_ids and len(cells) >= 2:
            for row, column in cells:
                inland[row, column] = True
    return labels, [cells for index, cells in enumerate(components) if index not in edge_ids and len(cells) >= 2]


def _component_mask(cells: Sequence[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for row, column in cells:
        mask[row, column] = True
    return mask


def _component_shape(cells: Sequence[tuple[int, int]]) -> tuple[float, float, float]:
    rows = [row for row, _ in cells]
    columns = [column for _, column in cells]
    height = max(rows) - min(rows) + 1
    width = max(columns) - min(columns) + 1
    area = len(cells)
    aspect = max(width, height) / max(1, min(width, height))
    fill = area / (width * height)
    return float(area), float(aspect), float(fill)


def _is_lake(
    cells: Sequence[tuple[int, int]],
    *,
    minimum_area: int = 4,
) -> bool:
    """Recognise a compact authored water body, not a thin river fragment."""

    if isinstance(minimum_area, bool) or int(minimum_area) < 1:
        raise SourceBuildError("lake minimum area must be a positive integer")

    area, aspect, fill, local_width = _component_metrics(cells)
    compact = aspect <= 4.5 and fill >= 0.30
    elongated_basin = local_width >= 3 and aspect <= 9.0 and fill >= 0.20
    return area >= int(minimum_area) and local_width >= 2 and (
        compact or elongated_basin
    )


def _is_linear_constraint(cells: Sequence[tuple[int, int]]) -> bool:
    """Return whether a non-lake component is credible line evidence."""

    area, aspect, fill, local_width = _component_metrics(cells)
    # A compact 2x2 (or larger compact) water-coloured patch is not allowed to burn a
    # valley.  Long/high-aspect strokes and sparse diagonal chains are useful
    # authored river evidence; isolated one/two-cell specks are inpaint-only.
    return (
        area >= 3
        and (
            aspect >= 2.0
            or fill <= 0.65
            or (local_width <= 1 and area >= 4)
        )
    )


def _authored_river_constraint_mask(
    components: Sequence[Sequence[tuple[int, int]]],
    lake_components: set[int],
    land: np.ndarray,
) -> np.ndarray:
    """Expand authored water-palette strokes into graded valley corridors.

    The source artwork's thin water-coloured components are evidence for preferred
    valleys, not a finished one-cell river network.  A five-cell valley floor
    with two graded shoulder cells lets the hydrology core choose a local
    thalweg from terrain and alluvial relief while retaining the authored
    valley's location.  Compact
    lake/noise components are inpaint-only and cannot become a stream.  Lake
    components are intentionally excluded so a water lake outline cannot
    become a stream.
    """

    constraint = np.zeros(land.shape, dtype=np.float64)
    height, width = land.shape
    for component_index, component in enumerate(components):
        if component_index in lake_components or not _is_linear_constraint(component):
            continue
        for row, column in component:
            for row_delta in range(-4, 5):
                for column_delta in range(-4, 5):
                    next_row = row + row_delta
                    next_column = column + column_delta
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and land[next_row, next_column]
                    ):
                        distance = max(abs(row_delta), abs(column_delta))
                        strength = (1.0, 1.0, 1.0, 0.65, 0.30)[distance]
                        constraint[next_row, next_column] = max(
                            float(constraint[next_row, next_column]),
                            strength,
                        )
    return constraint


def _cell_center(cell: Sequence[int | float]) -> tuple[float, float]:
    """Map a row/column cell to its internal grid x/y center."""

    row, column = int(cell[0]), int(cell[1])
    return (column + 0.5, row + 0.5)


def _node_id(cell: Sequence[int | float]) -> str:
    """Return a stable source-node identifier for a grid cell."""

    row, column = int(cell[0]), int(cell[1])
    return f"node-{row:03d}-{column:03d}"


def _chaikin_open_line(line: LineString, *, iterations: int) -> LineString:
    """Cut open-line corners for display while preserving both endpoints.

    This helper intentionally accepts only a line already expressed in the
    target publication coordinate system.  It never sees hydrology cells and
    cannot alter the D8 path, node metadata, or endpoint ownership.  A
    two-point line has no corner to cut and is returned unchanged.
    """

    if not isinstance(line, LineString):
        raise SourceBuildError("Chaikin open-line smoothing expects a LineString")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)):
        raise SourceBuildError(
            "Chaikin open-line smoothing iterations must be a non-negative integer"
        )
    iterations = int(iterations)
    if iterations < 0:
        raise SourceBuildError(
            "Chaikin open-line smoothing iterations must be a non-negative integer"
        )

    coordinates = list(line.coords)
    if iterations == 0 or len(coordinates) < 3:
        return LineString(coordinates)

    for _ in range(iterations):
        if len(coordinates) < 3:
            break
        refined: list[tuple[float, ...]] = [tuple(coordinates[0])]
        for start, end in zip(coordinates[:-1], coordinates[1:], strict=True):
            refined.append(
                tuple(0.75 * first + 0.25 * second for first, second in zip(start, end, strict=True))
            )
            refined.append(
                tuple(0.25 * first + 0.75 * second for first, second in zip(start, end, strict=True))
            )
        refined.append(tuple(coordinates[-1]))
        coordinates = refined
    return LineString(coordinates)


def _smooth_open_line_mapping(
    geometry: Mapping[str, Any],
    *,
    iterations: int,
) -> dict[str, Any]:
    """Apply cartographic open-line smoothing to serialized line parts."""

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "LineString":
        parts = [coordinates]
    elif geometry_type == "MultiLineString":
        parts = coordinates
    else:
        raise SourceBuildError(
            "cartographic river smoothing expects LineString or MultiLineString"
        )
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        raise SourceBuildError("cartographic river smoothing coordinates must be a list")

    smoothed_parts: list[list[list[float]]] = []
    for part in parts:
        if not isinstance(part, Sequence) or isinstance(part, (str, bytes)):
            raise SourceBuildError("cartographic river smoothing line part must be a list")
        smoothed = _chaikin_open_line(
            LineString(part),
            iterations=iterations,
        )
        smoothed_parts.append([list(point) for point in smoothed.coords])
    if geometry_type == "LineString":
        return {"type": geometry_type, "coordinates": smoothed_parts[0]}
    return {"type": geometry_type, "coordinates": smoothed_parts}


def _network_line_geometry(
    record: dict[str, Any],
    coastline: LineString | MultiLineString,
    context: BuildContext,
    waterbody_geometries: Mapping[str, Polygon | MultiPolygon] | None = None,
) -> LineString | MultiLineString:
    """Convert a hydrology cell path into a direction-preserving grid line.

    Hydrology stores an ocean outlet as an in-grid land cell.  Append the
    nearest point on the derived coastline so a published mouth reaches the
    land/ocean interface instead of stopping half a cell inland.  Lake outlets
    are snapped to the shoreline of their resolved waterbody; confluence and
    inland-sink endpoints remain on their explicit grid node cells.
    """

    cells = [tuple(cell) for cell in record.get("cells", ())]
    if not cells:
        return MultiLineString()
    coordinates = [_cell_center(cell) for cell in cells]
    endpoint = tuple(record.get("to", cells[-1]))
    endpoint_center = _cell_center(endpoint)
    end_type = str(record.get("end_type", "outlet"))
    outlet_type = str(record.get("outlet_type", "inland_sink"))
    if end_type == "confluence":
        if coordinates[-1] != endpoint_center:
            coordinates.append(endpoint_center)
    elif outlet_type == "ocean":
        final_point = LineString([coordinates[-1], coordinates[-1]])
        if not coastline.is_empty:
            # Shapely returns the nearest point on each geometry in argument
            # order.  The first point belongs to the coastline; using the
            # second one silently left every ocean mouth at its inland cell.
            mouth, _ = nearest_points(coastline, final_point)
            mouth_coordinate = (float(mouth.x), float(mouth.y))
            if mouth_coordinate != coordinates[-1]:
                coordinates.append(mouth_coordinate)
        elif endpoint_center != coordinates[-1]:
            coordinates.append(endpoint_center)
    elif outlet_type == "lake" and end_type == "outlet":
        waterbody_id = str(
            record.get("waterbody_id") or record.get("waterbodyId") or ""
        )
        if not waterbody_id or waterbody_geometries is None:
            raise SourceBuildError(
                f"lake terminal {record.get('id', '<unknown>')} has no waterbody geometry"
            )
        waterbody = waterbody_geometries.get(waterbody_id)
        if waterbody is None or waterbody.is_empty:
            raise SourceBuildError(
                f"lake terminal {record.get('id', '<unknown>')} references unknown waterbody {waterbody_id}"
            )
        boundary = waterbody.boundary
        if boundary.is_empty:
            raise SourceBuildError(
                f"lake waterbody {waterbody_id} has no drawable shoreline"
            )
        mouth, _ = nearest_points(
            boundary,
            LineString([coordinates[-1], coordinates[-1]]),
        )
        mouth_coordinate = (float(mouth.x), float(mouth.y))
        if mouth_coordinate != coordinates[-1]:
            coordinates.append(mouth_coordinate)
    elif endpoint_center != coordinates[-1]:
        # For an explicit inland sink, retain the legal terminal cell in the
        # line so the endpoint metadata and geometry agree.
        coordinates.append(endpoint_center)
    if len(coordinates) < 2:
        return MultiLineString()
    line = LineString(coordinates)
    return line if line.length > 0 else MultiLineString()


def _hydrology_features(
    result: HydrologyResult,
    coastline: LineString | MultiLineString,
    context: BuildContext,
    waterbody_geometries: Mapping[str, Polygon | MultiPolygon] | None = None,
    lake_cell_waterbody_ids: Mapping[tuple[int, int], str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Publish direction-oriented reaches and stable parent relationships."""

    provisional: list[dict[str, Any]] = []
    for record in result.network:
        record_for_geometry = dict(record)
        waterbody_id: str | None = None
        # ``outlet_type`` describes the basin's eventual outlet and is copied
        # onto every reach in that basin.  Only the terminal reach itself
        # needs a waterbody ID/shoreline snap; confluence reaches still end at
        # their explicit grid node and must not be resolved against a lake
        # polygon.
        if (
            str(record.get("outlet_type", "")) == "lake"
            and str(record.get("end_type", "outlet")) == "outlet"
        ):
            raw_waterbody_id = record.get("waterbody_id") or record.get("waterbodyId")
            if raw_waterbody_id:
                waterbody_id = str(raw_waterbody_id)
            elif lake_cell_waterbody_ids:
                endpoint = record.get("to")
                if isinstance(endpoint, Sequence) and len(endpoint) == 2:
                    waterbody_id = lake_cell_waterbody_ids.get(
                        (int(endpoint[0]), int(endpoint[1]))
                    )
            elif waterbody_geometries:
                endpoint = record.get("to")
                if endpoint is not None:
                    endpoint_point = Point(*_cell_center(endpoint))
                    matches = sorted(
                        waterbody_id_value
                        for waterbody_id_value, waterbody in waterbody_geometries.items()
                        if waterbody.covers(endpoint_point)
                    )
                    if len(matches) == 1:
                        waterbody_id = matches[0]
            if not waterbody_id or waterbody_geometries is None:
                raise SourceBuildError(
                    f"lake terminal {record.get('id', '<unknown>')} cannot resolve waterbodyId"
                )
            if waterbody_id not in waterbody_geometries:
                raise SourceBuildError(
                    f"lake terminal {record.get('id', '<unknown>')} references unknown waterbody {waterbody_id}"
                )
            record_for_geometry["waterbody_id"] = waterbody_id
        line = _network_line_geometry(
            record_for_geometry,
            coastline,
            context=context,
            waterbody_geometries=waterbody_geometries,
        )
        if line.is_empty or line.length <= 0:
            continue
        serialized_geometry = _smooth_open_line_mapping(
            _line_mapping(line, context=context),
            iterations=context.river_smoothing_iterations,
        )
        cells = [list(cell) for cell in record["cells"]]
        endpoint = list(record["to"])
        if cells[-1] != endpoint:
            cells_for_length = [*cells, endpoint]
        else:
            cells_for_length = cells
        from_node = _node_id(record["from"])
        to_node = _node_id(endpoint)
        feature = {
            "id": str(record["id"]),
            "kind": "river",
            "geometry": serialized_geometry,
            "sourceRule": "derived-d8-accumulation-network",
            "fromNodeId": from_node,
            "toNodeId": to_node,
            "fromCell": list(record["from"]),
            "toCell": endpoint,
            "order": int(record["order"]),
            "accumulationClass": str(record["accumulation_class"]),
            "maxAccumulation": _round_relative(float(record["accumulation"])),
            "outletType": str(record["outlet_type"]),
            "endType": str(record["end_type"]),
            "outletGeometryRule": (
                "ocean-mouth-snapped-to-derived-coastline"
                if str(record["outlet_type"]) == "ocean"
                else (
                    "lake-mouth-snapped-to-waterbody-shoreline"
                    if str(record["outlet_type"]) == "lake"
                    else "terminal-grid-cell"
                )
            ),
            "lengthCells": len(cells_for_length),
            "parentReachIds": [],
            "downstreamReachIds": [],
        }
        if waterbody_id is not None:
            feature["waterbodyId"] = waterbody_id
        provisional.append(feature)

    incoming: dict[str, list[str]] = {}
    outgoing: dict[str, list[str]] = {}
    for feature in provisional:
        incoming.setdefault(feature["toNodeId"], []).append(feature["id"])
        outgoing.setdefault(feature["fromNodeId"], []).append(feature["id"])
    for feature in provisional:
        feature["parentReachIds"] = sorted(
            reach_id
            for reach_id in incoming.get(feature["fromNodeId"], [])
            if reach_id != feature["id"]
        )
        feature["downstreamReachIds"] = sorted(
            reach_id
            for reach_id in outgoing.get(feature["toNodeId"], [])
            if reach_id != feature["id"]
        )

    node_reach_ids = {
        node_id
        for feature in provisional
        for node_id in (feature["fromNodeId"], feature["toNodeId"])
    }
    confluence_nodes = sum(
        len(incoming.get(node_id, ())) >= 2
        and bool(outgoing.get(node_id))
        for node_id in node_reach_ids
    )
    ocean_draining_reaches = sum(
        feature["outletType"] == "ocean" for feature in provisional
    )
    ocean_outlet_reaches = sum(
        feature["outletType"] == "ocean" and feature["endType"] == "outlet"
        for feature in provisional
    )
    lake_draining_reaches = sum(
        feature["outletType"] == "lake" for feature in provisional
    )
    lake_outlet_reaches = sum(
        feature["outletType"] == "lake" and feature["endType"] == "outlet"
        for feature in provisional
    )
    inland_sink_outlet_reaches = sum(
        feature["outletType"] == "inland_sink" and feature["endType"] == "outlet"
        for feature in provisional
    )
    stats = {
        "reachCount": len(provisional),
        "nodeCount": len(node_reach_ids),
        "confluenceNodeCount": int(confluence_nodes),
        # ``*DrainingReachCount`` includes upstream reaches whose eventual
        # basin outlet is that type.  ``*OutletReachCount`` counts only the
        # terminal reach ending at the outlet itself.
        "oceanDrainingReachCount": int(ocean_draining_reaches),
        "oceanOutletReachCount": int(ocean_outlet_reaches),
        "oceanOutletCount": int(ocean_outlet_reaches),
        "lakeDrainingReachCount": int(lake_draining_reaches),
        "lakeOutletReachCount": int(lake_outlet_reaches),
        "lakeOutletCount": int(lake_outlet_reaches),
        "inlandSinkOutletReachCount": int(inland_sink_outlet_reaches),
        "inlandSinkOutletCount": int(inland_sink_outlet_reaches),
    }
    return provisional, stats


def _grid_to_lonlat(
    x: float,
    y: float,
    context: BuildContext,
    wrap_right_edge: bool = False,
) -> list[float]:
    board_x = x * context.grid_x_step
    board_y = y * context.grid_y_step
    longitude_min, longitude_max = context.longitude_extent
    latitude_min, latitude_max = context.latitude_extent
    longitude_span = longitude_max - longitude_min
    latitude_span = latitude_max - latitude_min
    longitude = longitude_min + board_x / context.board_width * longitude_span
    if wrap_right_edge and x >= context.grid_width - 1e-9:
        # A global polygon has two board edges at the antimeridian. Keep its
        # right edge just inside the half-open longitude domain so planar
        # validators do not collapse both edges onto -180 degrees.
        longitude = longitude_max - 10 ** (-context.coordinate_decimals)
    latitude = latitude_max - board_y / context.board_height * latitude_span
    if longitude >= longitude_max:
        longitude = longitude_min
    if abs(longitude) < 0.0000005:
        longitude = 0.0
    if abs(latitude) < 0.0000005:
        latitude = 0.0
    return [round(longitude, context.coordinate_decimals), round(latitude, context.coordinate_decimals)]


def _map_coordinates(
    value: Any,
    context: BuildContext,
    wrap_right_edge: bool = False,
) -> Any:
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and all(isinstance(number, (int, float, np.number)) for number in value):
            return _grid_to_lonlat(
                float(value[0]),
                float(value[1]),
                wrap_right_edge=wrap_right_edge,
                context=context,
            )
        return [
            _map_coordinates(child, wrap_right_edge=wrap_right_edge, context=context)
            for child in value
        ]
    return value


def _geometry_mapping(
    geometry: Any,
    context: BuildContext,
    wrap_right_edge: bool = False,
) -> dict[str, Any]:
    mapped = dict(geometry.__geo_interface__)
    mapped["coordinates"] = _map_coordinates(
        mapped["coordinates"],
        wrap_right_edge=wrap_right_edge,
        context=context,
    )
    return {"type": mapped["type"], "coordinates": mapped["coordinates"]}


def _line_mapping(
    geometry: LineString | MultiLineString,
    context: BuildContext,
) -> dict[str, Any]:
    geometry = _line_only(geometry)
    if isinstance(geometry, LineString):
        geometry = MultiLineString([geometry])
    return _geometry_mapping(geometry, context=context)


def _polygon_mapping(
    geometry: Polygon | MultiPolygon,
    context: BuildContext,
    wrap_right_edge: bool = False,
) -> dict[str, Any]:
    geometry = _polygon_only(geometry)
    if isinstance(geometry, Polygon):
        geometry = MultiPolygon([geometry])
    return _geometry_mapping(geometry, wrap_right_edge=wrap_right_edge, context=context)


def _round_relative(value: float) -> float:
    return round(float(value), 12)


def _palette_projection(
    values: np.ndarray,
    palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered-palette position and squared RGB residual.

    The artwork uses flat fills plus antialiased blends.  A nearest-colour
    lookup would throw the blends away, so each pixel is projected onto every
    adjacent palette segment and the closest segment wins.  The returned
    position is continuous from 0 (first source colour) to 1 (last source
    colour).  The residual lets the surface classifier compare arbitrary
    authored land and water palettes without assuming that water is blue.
    """

    if values.ndim != 2 or values.shape[1] != 3:
        raise SourceBuildError("palette projection expects an N x 3 colour array")
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) < 2:
        raise SourceBuildError("palette projection expects at least two RGB stops")
    best_distance = np.full(values.shape[0], np.inf, dtype=np.float64)
    best_position = np.zeros(values.shape[0], dtype=np.float64)
    segment_count = len(palette) - 1
    for index, (start, end) in enumerate(zip(palette[:-1], palette[1:], strict=True)):
        vector = end - start
        denominator = float(np.dot(vector, vector))
        if denominator <= 0:
            continue
        projection = np.clip(((values - start) * vector).sum(axis=1) / denominator, 0.0, 1.0)
        projected = start + projection[:, None] * vector
        distance = ((values - projected) ** 2).sum(axis=1)
        take = distance < best_distance
        best_distance[take] = distance[take]
        best_position[take] = (index + projection[take]) / segment_count
    if not bool(np.all(np.isfinite(best_distance))):
        raise SourceBuildError("palette projection requires at least one distinct RGB segment")
    return np.clip(best_position, 0.0, 1.0), best_distance


def _palette_position(values: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Project colours onto an ordered palette without area rebalancing."""

    position, _ = _palette_projection(values, palette)
    return position


def _palette_distance(values: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Return squared RGB distance to the nearest ordered palette segment."""

    _, distance = _palette_projection(values, palette)
    return distance


def _masked_box_smooth(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    iterations: int,
    wrap_columns: bool = False,
) -> np.ndarray:
    """Smooth a continuous surface with a normalized, mask-aware 3x3 box.

    Only cells selected by ``valid`` are updated. Invalid cells remain
    byte-for-byte values from the copied input and never contribute to a
    neighbour average, which keeps a coastline from bleeding into the ocean.
    Rows are always bounded; ``wrap_columns`` explicitly enables longitude
    wrap for a closed authoring board. The returned array is a new float
    surface and the valid domain remains in authored-relative ``[0, 1]``.
    """

    source = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if source.ndim != 2 or mask.ndim != 2 or source.shape != mask.shape:
        raise SourceBuildError("masked box surface arrays must share a two-dimensional shape")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)):
        raise SourceBuildError("masked box smoothing iterations must be a non-negative integer")
    iterations = int(iterations)
    if iterations < 0:
        raise SourceBuildError("masked box smoothing iterations must be a non-negative integer")
    if not isinstance(wrap_columns, bool):
        raise SourceBuildError("masked box smoothing wrap_columns must be boolean")
    if bool(np.any(mask & ~np.isfinite(source))):
        raise SourceBuildError("masked box smoothing valid values must be finite")
    if bool(np.any(mask & ((source < 0.0) | (source > 1.0)))):
        raise SourceBuildError("masked box smoothing valid values must be within [0, 1]")

    result = source.copy()
    if iterations == 0 or not bool(mask.any()):
        return result

    height, width = result.shape
    for _ in range(iterations):
        numerator = np.zeros(result.shape, dtype=np.float64)
        denominator = np.zeros(result.shape, dtype=np.int16)
        padded_values = np.pad(
            result,
            ((1, 1), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        padded_mask = np.pad(
            mask,
            ((1, 1), (0, 0)),
            mode="constant",
            constant_values=False,
        )
        for row_delta in (-1, 0, 1):
            row_values = padded_values[1 + row_delta : 1 + row_delta + height]
            row_mask = padded_mask[1 + row_delta : 1 + row_delta + height]
            for column_delta in (-1, 0, 1):
                if wrap_columns:
                    neighbour_values = np.roll(
                        row_values,
                        -column_delta,
                        axis=1,
                    )
                    neighbour_mask = np.roll(
                        row_mask,
                        -column_delta,
                        axis=1,
                    )
                else:
                    padded_column_values = np.pad(
                        row_values,
                        ((0, 0), (1, 1)),
                        mode="constant",
                        constant_values=0.0,
                    )
                    padded_column_mask = np.pad(
                        row_mask,
                        ((0, 0), (1, 1)),
                        mode="constant",
                        constant_values=False,
                    )
                    start = 1 + column_delta
                    neighbour_values = padded_column_values[:, start : start + width]
                    neighbour_mask = padded_column_mask[:, start : start + width]
                numerator += np.where(neighbour_mask, neighbour_values, 0.0)
                denominator += neighbour_mask
        updated = result.copy()
        updated[mask] = np.clip(
            numerator[mask] / denominator[mask],
            0.0,
            1.0,
        )
        result = updated
    return result


def _masked_gaussian_smooth(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    sigma: float,
    passes: int,
    wrap_columns: bool = False,
) -> np.ndarray:
    """Apply deterministic normalized Gaussian passes inside a boolean mask.

    The numerator and denominator are convolved separately, so invalid cells
    never contribute a value.  Rows are always bounded; columns wrap only when
    the caller's authoring edge policy explicitly describes a longitude seam.
    The input is copied and masked-out sentinels are preserved byte-for-byte.
    """

    source = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if source.ndim != 2 or mask.ndim != 2 or source.shape != mask.shape:
        raise SourceBuildError(
            "masked Gaussian surface arrays must share a two-dimensional shape"
        )
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float, np.number)):
        raise SourceBuildError("masked Gaussian sigma must be a positive finite number")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise SourceBuildError("masked Gaussian sigma must be a positive finite number")
    if isinstance(passes, bool) or not isinstance(passes, (int, np.integer)):
        raise SourceBuildError("masked Gaussian passes must be a non-negative integer")
    passes = int(passes)
    if passes < 0:
        raise SourceBuildError("masked Gaussian passes must be a non-negative integer")
    if not isinstance(wrap_columns, bool):
        raise SourceBuildError("masked Gaussian wrap_columns must be boolean")
    if bool(np.any(mask & ~np.isfinite(source))):
        raise SourceBuildError("masked Gaussian valid values must be finite")
    if bool(np.any(mask & ((source < 0.0) | (source > 1.0)))):
        raise SourceBuildError("masked Gaussian valid values must be within [0, 1]")

    result = source.copy()
    if passes == 0 or not bool(mask.any()):
        return result

    height, width = result.shape
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= float(weights.sum())
    for _ in range(passes):
        numerator = np.zeros(result.shape, dtype=np.float64)
        denominator = np.zeros(result.shape, dtype=np.float64)
        padded_values = np.pad(
            result,
            ((radius, radius), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        padded_mask = np.pad(
            mask,
            ((radius, radius), (0, 0)),
            mode="constant",
            constant_values=False,
        )
        for row_index, row_delta in enumerate(range(-radius, radius + 1)):
            row_values = padded_values[
                radius + row_delta : radius + row_delta + height
            ]
            row_mask = padded_mask[
                radius + row_delta : radius + row_delta + height
            ]
            for column_index, column_delta in enumerate(range(-radius, radius + 1)):
                weight = float(weights[row_index] * weights[column_index])
                if wrap_columns:
                    neighbour_values = np.roll(
                        row_values,
                        -column_delta,
                        axis=1,
                    )
                    neighbour_mask = np.roll(
                        row_mask,
                        -column_delta,
                        axis=1,
                    )
                else:
                    padded_column_values = np.pad(
                        row_values,
                        ((0, 0), (radius, radius)),
                        mode="constant",
                        constant_values=0.0,
                    )
                    padded_column_mask = np.pad(
                        row_mask,
                        ((0, 0), (radius, radius)),
                        mode="constant",
                        constant_values=False,
                    )
                    start = radius + column_delta
                    neighbour_values = padded_column_values[:, start : start + width]
                    neighbour_mask = padded_column_mask[:, start : start + width]
                numerator += np.where(neighbour_mask, neighbour_values, 0.0) * weight
                denominator += neighbour_mask * weight
        updated = result.copy()
        updated[mask] = np.clip(
            numerator[mask] / denominator[mask],
            0.0,
            1.0,
        )
        result = updated
    return result


def _ordinal_classes(
    values: np.ndarray,
    palette: np.ndarray,
    level_count: int,
) -> np.ndarray:
    """Return authored ordinal classes from a continuous palette position."""

    if level_count < 2:
        raise SourceBuildError("ordinal palette classification needs at least two levels")
    position = _palette_position(values, palette)
    return np.minimum(level_count - 1, np.floor(position * level_count).astype(np.int16))


def _inpaint_continuous_elevation(
    values: np.ndarray,
    trusted_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Replace authored water-line/noise samples with nearby land medians.

    Thin water-coloured marks are hydrographic evidence, not low-elevation samples.  A
    deterministic expanding Chebyshev ring takes the median of the nearest
    trusted land values for each target cell.  The operation is local,
    resolution-aware, and independent of source coordinates; hydrology later
    applies its configured finite stream burn separately.
    """

    source = np.asarray(values, dtype=np.float64)
    trusted = np.asarray(trusted_mask, dtype=bool)
    targets = np.asarray(target_mask, dtype=bool)
    if source.ndim != 2 or trusted.shape != source.shape or targets.shape != source.shape:
        raise SourceBuildError("continuous-elevation inpaint arrays must share a 2D shape")
    result = source.copy()
    if not bool(targets.any()):
        return result
    trusted_values = source[trusted]
    fallback = float(np.median(trusted_values)) if trusted_values.size else 0.5
    height, width = source.shape
    for row, column in zip(*np.where(targets), strict=True):
        row = int(row)
        column = int(column)
        for radius in range(1, max(height, width)):
            row_start = max(0, row - radius)
            row_end = min(height, row + radius + 1)
            column_start = max(0, column - radius)
            column_end = min(width, column + radius + 1)
            local_trusted = trusted[row_start:row_end, column_start:column_end]
            if not bool(local_trusted.any()):
                continue
            local_values = source[row_start:row_end, column_start:column_end][local_trusted]
            result[row, column] = float(np.median(local_values))
            break
        else:
            result[row, column] = fallback
    return np.clip(result, 0.0, 1.0)


def _hex_palette(palette: np.ndarray) -> list[str]:
    return [
        "#" + "".join(f"{int(channel):02x}" for channel in colour)
        for colour in palette
    ]


def _majority_denoise(
    classes: np.ndarray,
    valid: np.ndarray,
    class_count: int,
) -> np.ndarray:
    """Remove isolated colour speckles without changing the surface mask."""

    result = classes.copy()
    padded = np.pad(classes, 1, mode="constant", constant_values=-1)
    for row, column in zip(*np.where(valid), strict=True):
        neighbourhood = padded[row : row + 3, column : column + 3]
        values = neighbourhood[neighbourhood >= 0]
        counts = np.bincount(values, minlength=class_count)
        winner = int(counts.argmax())
        if counts[winner] >= 5 and winner != int(classes[row, column]):
            result[row, column] = winner
    return result


_SURFACE_NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, 1, 1.0),
    (1, 1, math.sqrt(2.0)),
    (1, 0, 1.0),
    (1, -1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (-1, -1, math.sqrt(2.0)),
)


def _distance_inside_mask(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Return deterministic eight-neighbour distance within ``mask``."""

    active = np.asarray(mask, dtype=bool)
    sources = np.asarray(seeds, dtype=bool) & active
    if active.ndim != 2 or sources.shape != active.shape:
        raise SourceBuildError("surface distance masks must share a 2D shape")
    distance = np.full(active.shape, np.inf, dtype=np.float64)
    queue: list[tuple[float, int, int]] = []
    for row, column in zip(*np.where(sources), strict=True):
        row = int(row)
        column = int(column)
        distance[row, column] = 0.0
        queue.append((0.0, row, column))
    heapq.heapify(queue)
    height, width = active.shape
    while queue:
        current_distance, row, column = heapq.heappop(queue)
        if current_distance != float(distance[row, column]):
            continue
        for row_delta, column_delta, step in _SURFACE_NEIGHBOURS:
            neighbour_row = row + row_delta
            neighbour_column = column + column_delta
            if not (
                0 <= neighbour_row < height
                and 0 <= neighbour_column < width
                and active[neighbour_row, neighbour_column]
            ):
                continue
            candidate = current_distance + step
            if candidate + 1.0e-12 >= float(distance[neighbour_row, neighbour_column]):
                continue
            distance[neighbour_row, neighbour_column] = candidate
            heapq.heappush(queue, (candidate, neighbour_row, neighbour_column))
    return distance


def _continuous_ordinal_surface(
    authored_position: np.ndarray,
    ordinal_classes: np.ndarray,
    land_mask: np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    """Interpolate a continuous relative surface inside discrete colour bands.

    A hypsometric colour band only constrains an interval, not one constant
    elevation.  Each connected band component is therefore interpolated from
    its distance to lower and higher neighbouring bands.  Components with only
    one bounded side grade toward their interior extremum.  This is the raster
    analogue of contour-to-DEM interpolation and removes the broad terraces
    that otherwise create arbitrary parallel drainage paths.
    """

    authored = np.asarray(authored_position, dtype=np.float64)
    classes = np.asarray(ordinal_classes, dtype=np.int16)
    land = np.asarray(land_mask, dtype=bool)
    if authored.ndim != 2 or classes.shape != authored.shape or land.shape != authored.shape:
        raise SourceBuildError("continuous-surface arrays must share a 2D shape")
    if class_count < 2:
        raise SourceBuildError("continuous surface requires at least two ordinal classes")
    if np.any(land & ((classes < 0) | (classes >= class_count))):
        raise SourceBuildError("land ordinal class is outside the configured range")

    result = np.zeros(authored.shape, dtype=np.float64)
    height, width = authored.shape
    for ordinal in range(class_count):
        band = land & (classes == ordinal)
        if not bool(band.any()):
            continue
        lower_boundary = np.zeros(band.shape, dtype=bool)
        upper_boundary = np.zeros(band.shape, dtype=bool)
        for row, column in zip(*np.where(band), strict=True):
            row = int(row)
            column = int(column)
            for row_delta, column_delta, _ in _SURFACE_NEIGHBOURS:
                neighbour_row = row + row_delta
                neighbour_column = column + column_delta
                if not (0 <= neighbour_row < height and 0 <= neighbour_column < width):
                    continue
                if not land[neighbour_row, neighbour_column]:
                    if ordinal == 0:
                        lower_boundary[row, column] = True
                    continue
                neighbour_class = int(classes[neighbour_row, neighbour_column])
                if neighbour_class < ordinal:
                    lower_boundary[row, column] = True
                elif neighbour_class > ordinal:
                    upper_boundary[row, column] = True

        lower_distance = _distance_inside_mask(band, lower_boundary)
        upper_distance = _distance_inside_mask(band, upper_boundary)
        labels, components = _connected_components(band, 8)
        for component_id, cells in enumerate(components):
            component_rows = np.fromiter((cell[0] for cell in cells), dtype=np.int64)
            component_columns = np.fromiter((cell[1] for cell in cells), dtype=np.int64)
            has_lower = bool(np.any(lower_boundary[component_rows, component_columns]))
            has_upper = bool(np.any(upper_boundary[component_rows, component_columns]))
            if has_lower and has_upper:
                lower_values = lower_distance[component_rows, component_columns]
                upper_values = upper_distance[component_rows, component_columns]
                fraction = (lower_values + 0.5) / (
                    lower_values + upper_values + 1.0
                )
            elif has_lower:
                lower_values = lower_distance[component_rows, component_columns]
                maximum = float(np.max(lower_values, initial=0.0))
                fraction = (lower_values + 0.5) / (maximum + 1.0)
            elif has_upper:
                upper_values = upper_distance[component_rows, component_columns]
                maximum = float(np.max(upper_values, initial=0.0))
                fraction = 1.0 - (upper_values + 0.5) / (maximum + 1.0)
            else:
                original = authored[component_rows, component_columns]
                fraction = np.mod(original * class_count, 1.0)
                if bool(np.allclose(fraction, fraction[0], atol=1.0e-12)):
                    fraction = np.full(fraction.shape, 0.5, dtype=np.float64)
            fraction = np.clip(fraction, 1.0e-6, 1.0 - 1.0e-6)
            result[component_rows, component_columns] = (
                ordinal + fraction
            ) / class_count
            if np.any(labels[component_rows, component_columns] != component_id):
                raise SourceBuildError("continuous-surface component labelling drifted")
    return np.clip(result, 0.0, 1.0)


def _surface_flatness_stats(values: np.ndarray, valid: np.ndarray) -> dict[str, int | float]:
    """Return compact evidence that terrace reconstruction changed the field."""

    samples = np.round(np.asarray(values, dtype=np.float64)[valid], 8)
    if samples.size == 0:
        return {"sampleCount": 0, "uniqueRoundedValueCount": 0, "largestFlatFraction": 0.0}
    _, counts = np.unique(samples, return_counts=True)
    return {
        "sampleCount": int(samples.size),
        "uniqueRoundedValueCount": int(counts.size),
        "largestFlatFraction": float(np.max(counts) / samples.size),
    }


def _ordinal_adjacency_stats(classes: np.ndarray, valid: np.ndarray) -> dict[str, int | float]:
    """Measure local ordinal jumps on the shared land raster."""

    values = np.asarray(classes, dtype=np.int16)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or values.shape != mask.shape:
        raise SourceBuildError("ordinal adjacency arrays must share a 2D shape")
    total_edges = 0
    jump_count = 0
    maximum_jump = 0
    for row_delta, column_delta in ((1, 0), (0, 1)):
        first = values[:-row_delta or None, :-column_delta or None]
        second = values[row_delta:, column_delta:]
        edge_mask = (
            mask[:-row_delta or None, :-column_delta or None]
            & mask[row_delta:, column_delta:]
        )
        deltas = np.abs(first[edge_mask] - second[edge_mask])
        total_edges += int(deltas.size)
        jump_count += int(np.count_nonzero(deltas > 1))
        maximum_jump = max(maximum_jump, int(deltas.max(initial=0)))
    return {
        "edgeCount": total_edges,
        "jumpGreaterThanOneCount": jump_count,
        "jumpGreaterThanOneFraction": (
            float(jump_count / total_edges) if total_edges else 0.0
        ),
        "maximumAbsoluteOrdinalJump": maximum_jump,
    }


def _polygon_geometry_stats(
    geometries: Sequence[Polygon | MultiPolygon],
    *,
    max_reasonable_segment_grid_units: float,
) -> dict[str, int | float]:
    """Summarize published contour complexity in grid units."""

    if (
        not math.isfinite(max_reasonable_segment_grid_units)
        or max_reasonable_segment_grid_units <= 0.0
    ):
        raise SourceBuildError("contour segment QA limit must be positive and finite")
    polygons = [polygon for geometry in geometries for polygon in _polygon_parts(geometry)]
    parts = len(polygons)
    holes = sum(len(polygon.interiors) for polygon in polygons)
    vertex_count = 0
    segment_count = 0
    long_segment_count = 0
    perimeter = 0.0
    acute_corner_threshold_degrees = 60.0
    valid_corner_count = 0
    acute_corner_count = 0
    for polygon in polygons:
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            coordinates = list(ring.coords)
            vertex_count += max(0, len(coordinates) - 1)
            for start, end in zip(coordinates[:-1], coordinates[1:], strict=True):
                length = math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
                perimeter += length
                segment_count += 1
                if length > max_reasonable_segment_grid_units:
                    long_segment_count += 1
            unique_coordinates = coordinates[:-1] if len(coordinates) > 1 else []
            if len(unique_coordinates) < 3:
                continue
            signed_area_twice = sum(
                float(unique_coordinates[index][0]) * float(
                    unique_coordinates[(index + 1) % len(unique_coordinates)][1]
                )
                - float(unique_coordinates[(index + 1) % len(unique_coordinates)][0])
                * float(unique_coordinates[index][1])
                for index in range(len(unique_coordinates))
            )
            if abs(signed_area_twice) <= 1.0e-12:
                continue
            orientation = 1.0 if signed_area_twice > 0.0 else -1.0
            for index, current in enumerate(unique_coordinates):
                previous = unique_coordinates[index - 1]
                following = unique_coordinates[(index + 1) % len(unique_coordinates)]
                incoming = (
                    float(current[0]) - float(previous[0]),
                    float(current[1]) - float(previous[1]),
                )
                outgoing = (
                    float(following[0]) - float(current[0]),
                    float(following[1]) - float(current[1]),
                )
                incoming_length = math.hypot(*incoming)
                outgoing_length = math.hypot(*outgoing)
                if incoming_length <= 1.0e-12 or outgoing_length <= 1.0e-12:
                    continue
                valid_corner_count += 1
                cosine = (
                    (-incoming[0]) * outgoing[0]
                    + (-incoming[1]) * outgoing[1]
                ) / (incoming_length * outgoing_length)
                smaller_angle = math.degrees(
                    math.acos(max(-1.0, min(1.0, cosine)))
                )
                turn_cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
                interior_angle = (
                    smaller_angle
                    if turn_cross * orientation > 0.0
                    else 360.0 - smaller_angle
                )
                if interior_angle < acute_corner_threshold_degrees:
                    acute_corner_count += 1
    return {
        "partCount": parts,
        "holeCount": holes,
        "vertexCount": vertex_count,
        "segmentCount": segment_count,
        "perimeterGridUnits": _round_relative(perimeter),
        "vertexDensityPerGridUnit": _round_relative(
            vertex_count / perimeter if perimeter > 0.0 else 0.0
        ),
        "maxReasonableSegmentGridUnits": _round_relative(
            max_reasonable_segment_grid_units
        ),
        "longStraightSegmentCount": long_segment_count,
        "longStraightSegmentFraction": _round_relative(
            long_segment_count / segment_count if segment_count else 0.0
        ),
        "acuteCornerCount": acute_corner_count,
        "acuteCornerFraction": _round_relative(
            acute_corner_count / valid_corner_count if valid_corner_count else 0.0
        ),
        "acuteCornerThresholdDegrees": acute_corner_threshold_degrees,
    }


def _polygon_axis_artifact_stats(
    geometries: Sequence[Polygon | MultiPolygon],
    *,
    long_segment_grid_units: float,
) -> dict[str, int | float]:
    """Measure raster-axis lines and near-right corners visible at high zoom."""

    if (
        not math.isfinite(float(long_segment_grid_units))
        or float(long_segment_grid_units) <= 0.0
    ):
        raise SourceBuildError("axis-artifact segment threshold must be positive and finite")
    segment_count = 0
    long_axis_segment_count = 0
    corner_count = 0
    near_right_angle_count = 0
    axis_tolerance_degrees = 2.0
    right_angle_tolerance_degrees = 10.0
    for geometry in geometries:
        for polygon in _polygon_parts(geometry):
            for ring in (polygon.exterior, *polygon.interiors):
                coordinates = list(ring.coords)
                unique = coordinates[:-1] if len(coordinates) > 1 else []
                for start, end in zip(coordinates[:-1], coordinates[1:], strict=True):
                    delta_x = float(end[0]) - float(start[0])
                    delta_y = float(end[1]) - float(start[1])
                    length = math.hypot(delta_x, delta_y)
                    if length <= 1.0e-12:
                        continue
                    segment_count += 1
                    angle = abs(math.degrees(math.atan2(delta_y, delta_x))) % 180.0
                    axis_distance = min(angle, abs(90.0 - angle), abs(180.0 - angle))
                    if (
                        length >= float(long_segment_grid_units)
                        and axis_distance <= axis_tolerance_degrees
                    ):
                        long_axis_segment_count += 1
                if len(unique) < 3:
                    continue
                for index, current in enumerate(unique):
                    previous = unique[index - 1]
                    following = unique[(index + 1) % len(unique)]
                    first = (
                        float(previous[0]) - float(current[0]),
                        float(previous[1]) - float(current[1]),
                    )
                    second = (
                        float(following[0]) - float(current[0]),
                        float(following[1]) - float(current[1]),
                    )
                    first_length = math.hypot(*first)
                    second_length = math.hypot(*second)
                    if first_length <= 1.0e-12 or second_length <= 1.0e-12:
                        continue
                    corner_count += 1
                    cosine = (
                        first[0] * second[0] + first[1] * second[1]
                    ) / (first_length * second_length)
                    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                    if abs(angle - 90.0) <= right_angle_tolerance_degrees:
                        near_right_angle_count += 1
    return {
        "segmentCount": segment_count,
        "longAxisAlignedSegmentCount": long_axis_segment_count,
        "longAxisAlignedSegmentFraction": _round_relative(
            long_axis_segment_count / segment_count if segment_count else 0.0
        ),
        "longAxisAlignedThresholdGridUnits": _round_relative(
            float(long_segment_grid_units)
        ),
        "axisToleranceDegrees": axis_tolerance_degrees,
        "cornerCount": corner_count,
        "nearRightAngleCount": near_right_angle_count,
        "nearRightAngleFraction": _round_relative(
            near_right_angle_count / corner_count if corner_count else 0.0
        ),
        "rightAngleToleranceDegrees": right_angle_tolerance_degrees,
    }


def _feature_sort_key(feature: tuple[float, float, float, Any]) -> tuple[float, float, float]:
    area, aspect, fill, _ = feature
    return (-area, aspect, -fill)


def prepare_physical_fields(
    source_path: Path,
    *,
    include_debug: bool = False,
    context: BuildContext,
    procedural_surface: Any | None = None,
) -> PhysicalFields:
    """Prepare the canonical classified raster used by every map stage.

    This is the only PNG-to-raster entry point.  It preserves the existing
    source chain exactly: board padding, BOX reduction, colour classification,
    ordinal denoising, authored stream burning, and deterministic hydrology.
    Callers must use the returned arrays directly; they must not reconstruct
    terrain from smoothed SVG/JSON geometry.
    """

    source_path = Path(source_path).resolve()
    if not isinstance(context, BuildContext):
        raise SourceBuildError("prepare_physical_fields context must be a BuildContext")
    with Image.open(source_path) as image:
        source = image.convert("RGB")
    if source.size != context.source_dimensions:
        raise SourceBuildError(
            "context source dimensions do not match source image: "
            f"expected {context.source_width}x{context.source_height}, got {source.size}"
        )
    if (
        context.board_width < context.source_width + context.left_padding + context.right_padding
        or context.board_height < context.source_height + context.top_padding + context.bottom_padding
    ):
        raise SourceBuildError("context board must contain the source image and declared padding")

    source_array = np.asarray(source, dtype=np.uint8)
    ocean_color = source_array[0, 0]
    board = np.empty((context.board_height, context.board_width, 3), dtype=np.uint8)
    board[:] = ocean_color
    board[
        context.top_padding : context.top_padding + context.source_height,
        context.left_padding : context.left_padding + context.source_width,
    ] = source_array
    reduced = np.asarray(
        Image.fromarray(board, mode="RGB").resize(
            (context.grid_width, context.grid_height), Image.Resampling.BOX
        ),
        dtype=np.float64,
    )
    if procedural_surface is not None:
        procedural_land = np.asarray(procedural_surface.land_mask, dtype=bool)
        procedural_elevation = np.asarray(
            procedural_surface.elevation, dtype=np.float64
        )
        procedural_bathymetry = np.asarray(
            procedural_surface.bathymetry, dtype=np.float64
        )
        if any(
            value.shape != reduced.shape[:2]
            for value in (
                procedural_land,
                procedural_elevation,
                procedural_bathymetry,
            )
        ):
            raise SourceBuildError(
                "procedural surface fields must match the configured grid"
            )
        if (
            not np.all(np.isfinite(procedural_elevation))
            or not np.all(np.isfinite(procedural_bathymetry))
            or np.any((procedural_elevation < 0.0) | (procedural_elevation > 1.0))
            or np.any((procedural_bathymetry < 0.0) | (procedural_bathymetry > 1.0))
        ):
            raise SourceBuildError(
                "procedural surface elevation and bathymetry must be finite in [0, 1]"
            )
    land_palette = np.asarray(context.land_palette, dtype=np.float64)
    bathymetry_palette = np.asarray(context.bathymetry_palette, dtype=np.float64)
    flattened_colours = reduced.reshape(-1, 3)
    land_distance = _palette_distance(flattened_colours, land_palette).reshape(
        reduced.shape[:2]
    )
    water_distance = _palette_distance(flattened_colours, bathymetry_palette).reshape(
        reduced.shape[:2]
    )

    # Surface semantics come entirely from the authored palettes.  The
    # exterior connected water component becomes ocean; compact interior
    # components can become lakes and thin interior evidence remains available
    # to hydrology.  No channel-colour or Eirenor coordinate is hard-coded.
    water_candidate = (
        ~procedural_land
        if procedural_surface is not None
        else water_distance < land_distance
    )
    raw_water_labels, raw_inland_components = _water_components(water_candidate)
    raw_lake_minimum_area = max(
        4,
        int(
            math.ceil(
                LAKE_COMPONENT_AREA_FRACTION
                * max(1, int(np.count_nonzero(water_candidate)))
            )
        ),
    )
    raw_lake_component_indices = {
        component_index
        for component_index, component in enumerate(raw_inland_components)
        if _is_lake(component, minimum_area=raw_lake_minimum_area)
    }
    if procedural_surface is not None:
        # A canonical DEM/mask is already generated geography, not a scanned
        # colour artwork. Never interpret small bays/islets as paint noise.
        cleaned_water_candidate = water_candidate.copy()
        component_count = int(raw_water_labels.max(initial=-1)) + 1
        water_cells = int(np.count_nonzero(water_candidate))
        water_cleanup_stats = RasterCleanupStats(
            "accepted-surface", 4, component_count, component_count, 0, 0,
            water_cells, water_cells, 0, 0, 0, 0,
        )
    else:
        cleaned_water_candidate, water_cleanup_stats = cleanup_raster_mask(
            water_candidate,
            context.water_cleanup_policy,
            reference_cell_count=int(np.count_nonzero(water_candidate)),
        )
    cleaned_lake_minimum_area = max(
        4,
        int(
            math.ceil(
                LAKE_COMPONENT_AREA_FRACTION
                * max(1, int(np.count_nonzero(cleaned_water_candidate)))
            )
        ),
    )
    ocean = _wide_exterior_water(
        cleaned_water_candidate,
        minimum_corridor_width=3,
        wrap_columns=_edge_policy_wraps_longitude(context.edge_policy),
    )
    ocean = _promote_narrowly_connected_basins(
        cleaned_water_candidate,
        ocean,
        minimum_basin_area=cleaned_lake_minimum_area,
    )
    # Classify only topologically closed components as inland water.  A compact
    # basin whose narrow neck reaches the world ocean was restored above and is
    # therefore a sea, while an isolated blue line remains hydrology evidence.
    _, inland_components = _water_components(cleaned_water_candidate & ~ocean)
    inland_water = np.zeros_like(water_candidate, dtype=bool)
    lake_component_indices = {
        component_index
        for component_index, component in enumerate(inland_components)
        if _is_lake(component, minimum_area=cleaned_lake_minimum_area)
    }
    lake_mask = np.zeros_like(water_candidate, dtype=bool)
    for component_index, component in enumerate(inland_components):
        # A compact accepted lake is a surface exclusion.  Retained linear
        # water evidence remains available through ``water_candidate_mask``
        # and the raw authored river constraint, but must not carve a false
        # inland-water hole out of the land mask.
        if component_index not in lake_component_indices:
            continue
        for row, column in component:
            inland_water[row, column] = True
            lake_mask[row, column] = True
    if procedural_surface is not None:
        # All source water stays water, including narrow straits and tiny
        # closed ponds. Connectivity distinguishes ocean from enclosed water.
        ocean = water_candidate & np.isin(raw_water_labels, tuple(_edge_component_ids(raw_water_labels)))
        lake_mask = water_candidate & ~ocean
        inland_water = lake_mask.copy()
    lake_island_minimum_area = max(
        LAKE_ISLAND_MINIMUM_AREA_CELLS,
        int(math.ceil(lake_mask.size * LAKE_ISLAND_MINIMUM_AREA_FRACTION)),
    )
    lake_islands: LakeIslandField = (
        LakeIslandField(np.zeros_like(lake_mask), np.zeros_like(procedural_elevation), 0, 0)
        if procedural_surface is not None
        else generate_inland_water_islands(
            lake_mask,
            minimum_lake_area=lake_island_minimum_area,
            maximum_islands=LAKE_ISLAND_MAXIMUM_COUNT,
            shore_clearance=LAKE_ISLAND_SHORE_CLEARANCE_CELLS,
        )
    )
    generated_lake_island_mask = lake_islands.island_mask
    if bool(generated_lake_island_mask.any()):
        lake_mask &= ~generated_lake_island_mask
        inland_water &= ~generated_lake_island_mask
        cleaned_water_candidate = cleaned_water_candidate.copy()
        cleaned_water_candidate[generated_lake_island_mask] = False
        inland_components = [
            [
                (row, column)
                for row, column in component
                if not generated_lake_island_mask[row, column]
            ]
            for component in inland_components
        ]
    inland_sea_minimum_area = max(
        INLAND_SEA_MINIMUM_AREA_CELLS,
        int(math.ceil(lake_mask.size * INLAND_SEA_MINIMUM_AREA_FRACTION)),
    )
    inland_sea_mask = classify_inland_seas(
        lake_mask,
        minimum_area=inland_sea_minimum_area,
    )
    latitude_rows = context.latitude_extent[1] - (
        (np.arange(context.grid_height, dtype=np.float64) + 0.5)
        * context.grid_y_step
        / context.board_height
        * (context.latitude_extent[1] - context.latitude_extent[0])
    )
    polar_circle_latitude = 90.0 - context.planet.axial_tilt_degrees
    polar_latitude_mask = np.broadcast_to(
        np.abs(latitude_rows[:, None]) >= polar_circle_latitude,
        lake_mask.shape,
    )
    # Polar regions have no inland water in this world contract.  Remove any
    # authored high-latitude lake evidence before terrain reconstruction so it
    # becomes ice-covered continental surface, never a hidden lake or river.
    polar_inland_water = inland_water & polar_latitude_mask
    if bool(polar_inland_water.any()):
        inland_water = inland_water.copy()
        inland_water &= ~polar_inland_water
        lake_mask = lake_mask.copy()
        lake_mask &= ~polar_latitude_mask
        inland_sea_mask = inland_sea_mask.copy()
        inland_sea_mask &= ~polar_latitude_mask
        if procedural_surface is not None:
            ocean = ocean | polar_inland_water
        else:
            water_candidate = water_candidate.copy()
            water_candidate[polar_inland_water] = False
            cleaned_water_candidate = cleaned_water_candidate.copy()
            cleaned_water_candidate[polar_inland_water] = False
        inland_components = [
            [
                (row, column)
                for row, column in component
                if not polar_latitude_mask[row, column]
            ]
            for component in inland_components
        ]
        inland_components = [
            component for component in inland_components if len(component) >= 2
        ]
        lake_component_indices = {
            component_index
            for component_index, component in enumerate(inland_components)
            if _is_lake(component, minimum_area=cleaned_lake_minimum_area)
        }
    preliminary_land = ~ocean & ~inland_water
    preliminary_bathymetry = np.zeros(reduced.shape[:2], dtype=np.float64)
    preliminary_bathymetry[ocean] = (
        procedural_bathymetry[ocean]
        if procedural_surface is not None
        else _palette_position(reduced[ocean], bathymetry_palette)
    )
    preliminary_bathymetry[inland_sea_mask] = derive_inland_sea_bathymetry(
        inland_sea_mask,
        maximum_depth=0.78,
    )[inland_sea_mask]
    # The authored board intentionally omits both polar caps.  Generate the
    # missing continents here, before archipelagos, terrain refinement,
    # hydrology, snow and climate, so they participate in the same physical
    # chain as every authored continent.  Existing authored land always wins
    # if the two sources meet at a marginal cell.
    polar_terrain = generate_polar_terrain(
        reduced.shape[:2],
        context.longitude_extent,
        context.latitude_extent,
        ~ocean & ~inland_water,
    )
    complete_procedural_globe = context.complete_globe
    if complete_procedural_globe:
        generated_polar_land_mask = np.zeros_like(ocean, dtype=bool)
        generated_north_polar_land_mask = np.zeros_like(ocean, dtype=bool)
        generated_south_polar_land_mask = np.zeros_like(ocean, dtype=bool)
    else:
        generated_polar_land_mask = polar_terrain.land_mask & ocean
        generated_north_polar_land_mask = polar_terrain.north_land & ocean
        generated_south_polar_land_mask = polar_terrain.south_land & ocean
    if bool(generated_polar_land_mask.any()):
        ocean = ocean.copy()
        ocean &= ~generated_polar_land_mask
        water_candidate = water_candidate.copy()
        water_candidate[generated_polar_land_mask] = False
        cleaned_water_candidate = cleaned_water_candidate.copy()
        cleaned_water_candidate[generated_polar_land_mask] = False
        preliminary_bathymetry[generated_polar_land_mask] = 0.0
    polar_shelf = (
        np.zeros_like(ocean, dtype=bool)
        if complete_procedural_globe
        else polar_terrain.continental_shelf & ocean
    )
    preliminary_bathymetry[polar_shelf] = np.minimum(
        preliminary_bathymetry[polar_shelf],
        polar_terrain.shelf_bathymetry[polar_shelf],
    )
    preliminary_land = ~ocean & ~inland_water
    preliminary_relief = np.zeros(reduced.shape[:2], dtype=np.float64)
    authored_land = preliminary_land & ~generated_polar_land_mask
    preliminary_relief[authored_land] = (
        procedural_elevation[authored_land]
        if procedural_surface is not None
        else _palette_position(reduced[authored_land], land_palette)
    )
    preliminary_relief[generated_polar_land_mask] = (
        polar_terrain.relative_elevation[generated_polar_land_mask]
    )
    if procedural_surface is None:
        preliminary_bathymetry = complete_ocean_bathymetry(
            ocean,
            preliminary_land,
            preliminary_bathymetry,
        )
    if complete_procedural_globe:
        empty_mask = np.zeros_like(ocean, dtype=bool)
        empty_surface = np.zeros_like(preliminary_bathymetry, dtype=np.float64)
        shelf_archipelagos = ShelfArchipelagoField(
            island_mask=empty_mask,
            large_island_mask=empty_mask,
            shelf_island_mask=empty_mask,
            island_arc_mask=empty_mask,
            hotspot_island_mask=empty_mask,
            seamount_mask=empty_mask,
            shelf_mask=empty_mask,
            shelf_bathymetry=empty_surface,
            trench_mask=empty_mask,
            relative_elevation=empty_surface,
            island_count=0,
            group_count=0,
        )
    else:
        shelf_archipelagos = generate_shelf_archipelagos(
            ocean,
            preliminary_land,
            preliminary_bathymetry,
            preliminary_relief,
            maximum_groups=SHELF_ARCHIPELAGO_MAXIMUM_GROUPS,
            maximum_islands_per_group=SHELF_ARCHIPELAGO_MAXIMUM_ISLANDS_PER_GROUP,
            minimum_shore_distance=SHELF_ARCHIPELAGO_MINIMUM_SHORE_DISTANCE,
            maximum_shore_distance=SHELF_ARCHIPELAGO_MAXIMUM_SHORE_DISTANCE,
        )
    generated_shelf_mask = shelf_archipelagos.shelf_mask & ocean
    preliminary_bathymetry[generated_shelf_mask] = np.minimum(
        preliminary_bathymetry[generated_shelf_mask],
        shelf_archipelagos.shelf_bathymetry[generated_shelf_mask],
    )
    generated_trench_mask = shelf_archipelagos.trench_mask & ocean
    preliminary_bathymetry[generated_trench_mask] = np.maximum(
        preliminary_bathymetry[generated_trench_mask],
        0.96,
    )
    generated_seamount_mask = shelf_archipelagos.seamount_mask & ocean
    preliminary_bathymetry[generated_seamount_mask] = np.minimum(
        preliminary_bathymetry[generated_seamount_mask],
        0.72,
    )
    generated_ocean_island_mask = shelf_archipelagos.island_mask
    if bool(generated_ocean_island_mask.any()):
        ocean = ocean.copy()
        ocean &= ~generated_ocean_island_mask
        cleaned_water_candidate = cleaned_water_candidate.copy()
        cleaned_water_candidate[generated_ocean_island_mask] = False
        preliminary_bathymetry[generated_ocean_island_mask] = 0.0
    # Procedural polar land and archipelagos can close a former marine embayment.
    # Re-evaluate the finished shoreline so those enclosed cells become lakes
    # instead of retaining a stale ocean code and ocean bathymetry.
    # Four-neighbour diagonal cracks within the generated polar ice margin are
    # still polar marine channels, never lakes.  The world contract excludes
    # all polar lakes, so only non-polar enclosed water is reclassified.
    enclosed_ocean = _enclosed_ocean_mask(ocean) & ~polar_latitude_mask
    if bool(enclosed_ocean.any()):
        ocean = ocean.copy()
        ocean &= ~enclosed_ocean
        inland_water = inland_water.copy()
        inland_water |= enclosed_ocean
        lake_mask = lake_mask.copy()
        lake_mask |= enclosed_ocean
        inland_sea_mask = inland_sea_mask.copy()
        inland_sea_mask &= ~enclosed_ocean
        preliminary_bathymetry[enclosed_ocean] = 0.0
    # From this point onward ``lake_mask`` is the final semantic water
    # classification.  Rebuild its components after shoreline-generating
    # polar land and archipelagos have been applied; retaining the earlier
    # component list would leave newly enclosed lakes without a published
    # waterbody ID, even though hydrology correctly drains into them.
    _, final_lake_components = _connected_components(lake_mask, connectivity=4)
    inland_components = [list(component) for component in final_lake_components]
    lake_component_indices = set(range(len(inland_components)))
    land = ~ocean & ~inland_water
    if procedural_surface is not None and not np.array_equal(land, procedural_land):
        raise SourceBuildError("accepted land/water mask changed during physical import")
    if not bool(land.any()):
        raise SourceBuildError("surface classification produced no land")

    # Recover the continuous authored colour position and water-line inpaint
    # before any ordinal quantization. This is palette interpolation, not a
    # metric DEM inversion.
    continuous_elevation = np.zeros(reduced.shape[:2], dtype=np.float64)
    continuous_elevation[land] = (
        procedural_elevation[land]
        if procedural_surface is not None
        else _palette_position(reduced[land], land_palette)
    )
    continuous_elevation[generated_lake_island_mask] = (
        lake_islands.relative_elevation[generated_lake_island_mask]
    )
    continuous_elevation[generated_ocean_island_mask] = (
        shelf_archipelagos.relative_elevation[generated_ocean_island_mask]
    )
    continuous_elevation[generated_polar_land_mask] = (
        polar_terrain.relative_elevation[generated_polar_land_mask]
    )
    # Fine water-coloured marks that were not accepted as compact lakes are authored
    # river/valley evidence (including isolated colour noise).  They are not
    # observed lowlands: inpaint them from the nearest trusted land median so
    # only the finite hydrology burn can attract flow into those valleys.
    linear_constraint_mask = (
        land
        & water_candidate
        & ~lake_mask
        & ~generated_lake_island_mask
        & ~generated_ocean_island_mask
    )
    trusted_elevation_mask = land & ~linear_constraint_mask
    continuous_elevation = _inpaint_continuous_elevation(
        continuous_elevation,
        trusted_elevation_mask,
        linear_constraint_mask,
    )
    authored_surface_stats = _surface_flatness_stats(continuous_elevation, land)
    if procedural_surface is None:
        continuous_elevation = _masked_gaussian_smooth(
            continuous_elevation,
            land,
            sigma=context.elevation_regularization_sigma_grid_units,
            passes=context.elevation_regularization_passes,
            wrap_columns=_edge_policy_wraps_longitude(context.edge_policy),
        )
    regularized_surface_stats = _surface_flatness_stats(continuous_elevation, land)
    terrain_refinement = None
    physiographic_provinces = None
    if procedural_surface is None:
        terrain_refinement = refine_continuous_terrain(
            continuous_elevation, land, ocean, lake_mask, levels=context.elevation_levels,
        )
        continuous_elevation = terrain_refinement.elevation
        physiographic_provinces = shape_physiographic_provinces(
            continuous_elevation, land, ocean, inland_sea_mask, preliminary_bathymetry,
            levels=context.elevation_levels,
        )
        continuous_elevation = physiographic_provinces.elevation

    maritime_water = ocean | inland_sea_mask
    continuous_bathymetry = preliminary_bathymetry.copy()
    if procedural_surface is None:
        continuous_bathymetry = _masked_box_smooth(
            continuous_bathymetry,
            maritime_water,
            iterations=context.bathymetry_smoothing_iterations,
            wrap_columns=_edge_policy_wraps_longitude(context.edge_policy),
        )
    bathymetry_grid = np.full(reduced.shape[:2], -1, dtype=np.int16)
    bathymetry_grid[maritime_water] = np.clip(
        np.floor(
            continuous_bathymetry[maritime_water] * context.bathymetry_levels
        ),
        0,
        context.bathymetry_levels - 1,
    ).astype(np.int16)
    # The padded board edge is an authoring registration edge, not an observed
    # shallow-water sample. Explicitly close it as the deepest source class.
    edge_ocean = ocean & _edge_mask(ocean.shape)
    bathymetry_grid[edge_ocean] = context.bathymetry_levels - 1
    bathymetry_grid = _majority_denoise(
        bathymetry_grid, maritime_water, class_count=context.bathymetry_levels
    )

    # Keep the fine authored water-palette fragments as hydrology evidence, while the
    # cleaned water mask controls land/lake geometry.  This separation avoids
    # turning each one-pixel river mark into a polygonal inland-water hole.
    authored_river_constraint = (
        np.zeros_like(land)
        if procedural_surface is not None
        else _authored_river_constraint_mask(raw_inland_components, raw_lake_component_indices, land)
    )
    linear_constraint_components = [
        component
        for index, component in enumerate(raw_inland_components)
        if index not in raw_lake_component_indices and _is_linear_constraint(component)
    ]
    polar_land_hydrology_exclusion = land & polar_latitude_mask
    if bool(np.any(lake_mask & polar_latitude_mask)):
        raise SourceBuildError("polar regions must not contain lakes")
    preliminary_snow_mask = compute_perennial_snowline(
        continuous_elevation,
        land,
        latitude_rows,
        config=context.snowline_config,
    ).perennial_snow_mask
    preliminary_hydroclimate = compute_monsoon_hydroclimate(
        continuous_elevation,
        land,
        ocean,
        latitude_rows,
        planet=context.planet,
    )
    # Polar continents deliberately end at climate and biome.  Exclude them
    # from hydrology at the input boundary so their area and relief cannot
    # change channel thresholds or drainage topology on the inhabited
    # continents.  They remain canonical land for terrain, snow and climate.
    hydrology_land = land & ~polar_land_hydrology_exclusion
    hydrology_ocean = ocean | polar_land_hydrology_exclusion
    preliminary_hydrology = compute_hydrology(
        continuous_elevation,
        hydrology_land,
        hydrology_ocean,
        lake_mask=lake_mask,
        river_constraint_mask=authored_river_constraint,
        perennial_snow_mask=preliminary_snow_mask & hydrology_land,
        runoff_source_yield=preliminary_hydroclimate.perennial_runoff_yield,
        seasonality_index=preliminary_hydroclimate.monsoon_index,
        latitude_degrees=latitude_rows,
        config=context.hydrology_config,
    )
    drainage_incision = incise_drainage(
        continuous_elevation,
        hydrology_land,
        preliminary_hydrology.stream_mask,
        preliminary_hydrology.downstream_index,
        preliminary_hydrology.mfd_accumulation,
    )
    incised_snow_mask = compute_perennial_snowline(
        drainage_incision.elevation,
        land,
        latitude_rows,
        config=context.snowline_config,
    ).perennial_snow_mask
    incised_hydroclimate = compute_monsoon_hydroclimate(
        drainage_incision.elevation,
        land,
        ocean,
        latitude_rows,
        planet=context.planet,
    )
    hydrology = compute_hydrology(
        drainage_incision.elevation,
        hydrology_land,
        hydrology_ocean,
        lake_mask=lake_mask,
        river_constraint_mask=authored_river_constraint,
        perennial_snow_mask=incised_snow_mask & hydrology_land,
        runoff_source_yield=incised_hydroclimate.perennial_runoff_yield,
        seasonality_index=incised_hydroclimate.monsoon_index,
        latitude_degrees=latitude_rows,
        config=context.hydrology_config,
    )
    hydrologic_clamp_cell_count = int(
        np.count_nonzero(
            land
            & (
                (hydrology.corrected_elevation < 0.0)
                | (hydrology.corrected_elevation > 1.0)
            )
        )
    )
    continuous_elevation = hydrology.corrected_elevation.copy()
    continuous_elevation[land] = np.clip(
        continuous_elevation[land],
        1.0 / (context.elevation_levels * 200.0),
        1.0,
    )
    continuous_elevation[~land] = 0.0
    hydrology = replace(
        hydrology,
        corrected_elevation=continuous_elevation.copy(),
    )
    # Polar continents intentionally stop at terrain, snow, simple climate and
    # biome.  Hydrology never enters them; clear every published field again at
    # this contract boundary so downstream consumers cannot mistake polar land
    # for a drainage surface.
    polar_hydrology_mask = polar_land_hydrology_exclusion
    if bool(polar_hydrology_mask.any()):
        def cleared(values: np.ndarray, fill: Any) -> np.ndarray:
            result = np.array(values, copy=True)
            result[polar_hydrology_mask] = fill
            return result

        retained_network: list[dict[str, Any]] = []
        for record in hydrology.network:
            if any(
                polar_hydrology_mask[int(row), int(column)]
                for row, column in record.get("cells", ())
            ):
                continue
            retained_network.append(dict(record))
        for sequence, record in enumerate(retained_network, start=1):
            record["id"] = f"river-{sequence:04d}"
        diagnostics = dict(hydrology.diagnostics)
        diagnostics["polarHydrology"] = {
            "published": False,
            "excludedSurfaceCells": int(np.count_nonzero(polar_hydrology_mask)),
            "removedStreamCells": int(
                np.count_nonzero(hydrology.stream_mask & polar_hydrology_mask)
            ),
            "removedNetworkRecords": len(hydrology.network) - len(retained_network),
            "reason": "polar-continents-stop-at-climate-and-biome",
        }
        hydrology = replace(
            hydrology,
            flow_direction=cleared(hydrology.flow_direction, -1),
            accumulation=cleared(hydrology.accumulation, 0.0),
            mfd_accumulation=cleared(hydrology.mfd_accumulation, 0.0),
            valley_score=cleared(hydrology.valley_score, 0.0),
            valley_support=cleared(hydrology.valley_support, 0.0),
            valley_supported=cleared(hydrology.valley_supported, False),
            constraint_strength=cleared(hydrology.constraint_strength, 0.0),
            flat_distance=cleared(hydrology.flat_distance, -1),
            stream_mask=cleared(hydrology.stream_mask, False),
            stream_order=cleared(hydrology.stream_order, 0),
            outlet_type=cleared(hydrology.outlet_type, ""),
            downstream_index=cleared(hydrology.downstream_index, -1),
            network=tuple(retained_network),
            diagnostics=diagnostics,
        )
    reconstructed_surface_stats = _surface_flatness_stats(continuous_elevation, land)
    perennial_snow_mask = compute_perennial_snowline(
        continuous_elevation,
        land,
        latitude_rows,
        config=context.snowline_config,
    ).perennial_snow_mask
    hydroclimate = compute_monsoon_hydroclimate(
        continuous_elevation,
        land,
        ocean,
        latitude_rows,
        planet=context.planet,
    )
    class_grid = np.full(reduced.shape[:2], -1, dtype=np.int16)
    class_grid[land] = np.clip(
        np.floor(continuous_elevation[land] * context.elevation_levels),
        0,
        context.elevation_levels - 1,
    ).astype(np.int16)

    if procedural_surface is not None:
        polar_choices = procedural_surface.diagnostics["polarContinents"]
        generated_north_polar_land_mask = land & polar_latitude_mask & (latitude_rows[:, None] > 0) & polar_choices["north"]
        generated_south_polar_land_mask = land & polar_latitude_mask & (latitude_rows[:, None] < 0) & polar_choices["south"]
        generated_polar_land_mask = generated_north_polar_land_mask | generated_south_polar_land_mask

    return PhysicalFields(
        source_path=source_path,
        source_sha256=sha256(source_path),
        source_dimensions=context.source_dimensions,
        board_dimensions=context.board_dimensions,
        grid_step_px=context.sampling_step_px,
        left_padding_px=context.left_padding,
        land_mask=land,
        lake_mask=lake_mask,
        inland_sea_mask=inland_sea_mask,
        generated_lake_island_mask=generated_lake_island_mask,
        generated_ocean_island_mask=generated_ocean_island_mask,
        generated_polar_land_mask=generated_polar_land_mask,
        generated_north_polar_land_mask=generated_north_polar_land_mask,
        generated_south_polar_land_mask=generated_south_polar_land_mask,
        inland_water_mask=inland_water,
        ocean_mask=ocean,
        water_candidate_mask=cleaned_water_candidate,
        raster_cleanup={
            "waterCandidate": {
                "policy": context.water_cleanup_policy.as_record(),
                "stats": water_cleanup_stats.as_record(include_components=include_debug),
                "rawCandidateCells": int(np.count_nonzero(water_candidate)),
                "cleanedCandidateCells": int(np.count_nonzero(cleaned_water_candidate)),
                "rawInlandComponentCount": len(raw_inland_components),
                "cleanedInlandComponentCount": len(inland_components),
                "rawLakeComponentCount": len(raw_lake_component_indices),
                "cleanedLakeComponentCount": len(lake_component_indices),
                "rawLakeMinimumAreaCells": raw_lake_minimum_area,
                "cleanedLakeMinimumAreaCells": cleaned_lake_minimum_area,
                "linearConstraintCellCount": int(np.count_nonzero(linear_constraint_mask)),
                "burnEligibleLinearComponentCount": len(linear_constraint_components),
                "burnEligibleLinearCellCount": sum(
                    len(component) for component in linear_constraint_components
                ),
                "linearConstraintRule": "raw authored-water-palette non-lake cells are deterministic nearest-ring median inpaint evidence; finite stream burn remains the only hydrology attraction",
            },
            "lakeIslands": {
                "algorithm": "deterministic-inland-water-organic-islands-v2",
                "minimumLakeAreaCells": lake_island_minimum_area,
                "modifiedLakeCount": lake_islands.modified_lake_count,
                "islandCount": lake_islands.island_count,
                "islandCellCount": int(
                    np.count_nonzero(generated_lake_island_mask)
                ),
                "shoreClearanceCells": LAKE_ISLAND_SHORE_CLEARANCE_CELLS,
            },
            "inlandSea": {
                "algorithm": "closed-basin-distance-bathymetry-v2",
                "minimumAreaCells": inland_sea_minimum_area,
                "cellCount": int(np.count_nonzero(inland_sea_mask)),
                "componentCount": len(_water_components(inland_sea_mask)[1]),
                "bathymetryBandCount": int(
                    np.unique(bathymetry_grid[inland_sea_mask]).size
                ),
            },
            "shelfArchipelagos": {
                "algorithm": "geologic-microcontinent-shelf-subduction-hotspot-system-v5",
                "groupCount": shelf_archipelagos.group_count,
                "islandCount": shelf_archipelagos.island_count,
                "islandCellCount": int(np.count_nonzero(generated_ocean_island_mask)),
                "largeIslandCellCount": int(
                    np.count_nonzero(shelf_archipelagos.large_island_mask)
                ),
                "shelfIslandCellCount": int(
                    np.count_nonzero(shelf_archipelagos.shelf_island_mask)
                ),
                "islandArcCellCount": int(
                    np.count_nonzero(shelf_archipelagos.island_arc_mask)
                ),
                "hotspotIslandCellCount": int(
                    np.count_nonzero(shelf_archipelagos.hotspot_island_mask)
                ),
                "seamountCellCount": int(
                    np.count_nonzero(shelf_archipelagos.seamount_mask)
                ),
                "generatedShelfCellCount": int(np.count_nonzero(generated_shelf_mask)),
                "generatedTrenchCellCount": int(np.count_nonzero(generated_trench_mask)),
            },
            "polarContinents": {
                **polar_terrain.diagnostics,
                "generatedLandCells": int(
                    np.count_nonzero(generated_polar_land_mask)
                ),
                "generatedNorthLandCells": int(
                    np.count_nonzero(generated_north_polar_land_mask)
                ),
                "generatedSouthLandCells": int(
                    np.count_nonzero(generated_south_polar_land_mask)
                ),
                "authoredOverlapCells": int(
                    np.count_nonzero(polar_terrain.land_mask & ~generated_polar_land_mask)
                ),
                "suppressedAuthoredInlandWaterCells": int(
                    np.count_nonzero(polar_inland_water)
                ),
                "lakePolicy": "no-polar-lakes",
                "feedsClimateAndBiome": True,
                "feedsHydrology": False,
            },
            "terrainRefinement": {
                **(terrain_refinement.diagnostics if terrain_refinement is not None else {"sourceTerrainPreserved": True}),
                **drainage_incision.diagnostics,
                "hydrologicClampCellCount": hydrologic_clamp_cell_count,
                "algorithm": ("accepted-terrain-hydrology-only-v1" if procedural_surface is not None
                              else "shore-gradient-grouped-rolling-hills-two-pass-hydrology-v3"),
                "preservesSurfaceMasks": True,
            },
            "hydroclimate": {
                "model": HYDROCLIMATE_MODEL_NAME,
                "axialTiltDegrees": _round_relative(
                    context.planet.axial_tilt_degrees
                ),
                "monsoonCellCount": int(np.count_nonzero(hydroclimate.monsoon_index > 0.35)),
                "meanAnnualPrecipitation": _round_relative(
                    float(np.mean(hydroclimate.annual_precipitation[land]))
                ),
                "meanPerennialRunoffYield": _round_relative(
                    float(np.mean(hydroclimate.perennial_runoff_yield[land]))
                ),
            },
            "physiographicProvinces": {
                **(physiographic_provinces.diagnostics if physiographic_provinces is not None else {"sourceTerrainPreserved": True}),
                "algorithm": ("accepted-tectonic-surface" if procedural_surface is not None
                              else "orogenic-axis-multiscale-ranges-active-margin-inland-basin-v2"),
                "preservesExistingWaterAfterIslandGeneration": True,
            },
            "continuousSurface": {
                "method": ELEVATION_SURFACE_METHOD,
                "authored": authored_surface_stats,
                "regularized": regularized_surface_stats,
                "reconstructed": reconstructed_surface_stats,
                "preservesLandMask": True,
                "regularization": {
                    "algorithm": ELEVATION_REGULARIZATION_ALGORITHM,
                    "sigmaGridUnits": _round_relative(
                        context.elevation_regularization_sigma_grid_units
                    ),
                    "passes": int(context.elevation_regularization_passes),
                    "wrapColumns": _edge_policy_wraps_longitude(context.edge_policy),
                    "wrapRows": False,
                },
                "displayQuantization": "floor(clamp(regularizedContinuousElevation * levels, 0, levels - 1))",
            },
        },
        continuous_elevation=continuous_elevation,
        elevation_ordinal=class_grid,
        bathymetry=bathymetry_grid,
        linear_constraint_mask=linear_constraint_mask,
        authored_river_constraint=authored_river_constraint,
        hydroclimate=hydroclimate,
        hydrology=hydrology,
        perennial_snow_mask=perennial_snow_mask,
        inland_components=tuple(tuple(component) for component in inland_components),
        lake_component_indices=frozenset(lake_component_indices),
        runtime_parameters=_runtime_parameters(context),
        context=context,
    )


def _build_source(
    source_path: Path,
    *,
    include_debug: bool = False,
    context: BuildContext,
) -> dict[str, Any]:
    fields = prepare_physical_fields(
        source_path,
        include_debug=include_debug,
        context=context,
    )
    context = fields.context
    # These local aliases make the metadata section easy to audit while
    # keeping every value tied to this immutable build pass.
    source_name = context.source_name
    source_artifact = context.source_artifact
    source_width = context.source_width
    source_height = context.source_height
    board_width = context.board_width
    board_height = context.board_height
    left_padding = context.left_padding
    right_padding = context.right_padding
    top_padding = context.top_padding
    bottom_padding = context.bottom_padding
    grid_step = context.sampling_step_px
    grid_width = context.grid_width
    grid_height = context.grid_height
    grid_x_step = context.grid_x_step
    grid_y_step = context.grid_y_step
    longitude_min, longitude_max = context.longitude_extent
    latitude_min, latitude_max = context.latitude_extent
    elevation_levels = context.elevation_levels
    bathymetry_levels = context.bathymetry_levels
    simplify_tolerance = context.simplify_tolerance
    coastline_tolerance = context.coastline_tolerance
    coordinate_decimals = context.coordinate_decimals
    smoothing_iterations = context.smoothing_iterations
    elevation_contour_simplify_tolerance = context.elevation_contour_simplify_tolerance
    bathymetry_smoothing_iterations = context.bathymetry_smoothing_iterations
    river_smoothing_iterations = context.river_smoothing_iterations
    land_palette = np.asarray(context.land_palette, dtype=np.float64)
    bathymetry_palette = np.asarray(context.bathymetry_palette, dtype=np.float64)
    water_cleanup_policy = context.water_cleanup_policy
    elevation_cleanup_policy = context.elevation_cleanup_policy
    bathymetry_cleanup_policy = context.bathymetry_cleanup_policy
    hydrology_config = context.hydrology_config
    snowline_config = context.snowline_config
    # Upper-case aliases are local to this function and cannot leak one
    # preparation pass's values into another.
    ELEVATION_LEVELS = elevation_levels
    BATHYMETRY_LEVELS = bathymetry_levels
    GRID_WIDTH = grid_width
    GRID_HEIGHT = grid_height
    GRID_X_STEP = grid_x_step
    GRID_Y_STEP = grid_y_step
    BOARD_WIDTH = board_width
    BOARD_HEIGHT = board_height
    LATITUDE_MIN = latitude_min
    LATITUDE_MAX = latitude_max
    LONGITUDE_MIN = longitude_min
    LONGITUDE_MAX = longitude_max
    SIMPLIFY_TOLERANCE = simplify_tolerance
    COASTLINE_TOLERANCE = coastline_tolerance
    COORDINATE_DECIMALS = coordinate_decimals
    SMOOTHING_ITERATIONS = smoothing_iterations
    ELEVATION_CONTOUR_SIMPLIFY_TOLERANCE = elevation_contour_simplify_tolerance
    BATHYMETRY_SMOOTHING_ITERATIONS = bathymetry_smoothing_iterations
    RIVER_SMOOTHING_ITERATIONS = river_smoothing_iterations
    LAND_PALETTE = land_palette
    BATHYMETRY_PALETTE = bathymetry_palette
    ELEVATION_CLEANUP_POLICY = elevation_cleanup_policy
    BATHYMETRY_CLEANUP_POLICY = bathymetry_cleanup_policy
    WATER_CLEANUP_POLICY = water_cleanup_policy
    HYDROLOGY_CONFIG = hydrology_config
    SNOWLINE_CONFIG = snowline_config
    longitude_span = longitude_max - longitude_min
    latitude_span = latitude_max - latitude_min
    x_center = "x + 0.5" if left_padding == 0 else f"x + {left_padding} + 0.5"
    y_center = "y + 0.5" if top_padding == 0 else f"y + {top_padding} + 0.5"
    longitude_offset = (
        f"+ {longitude_min:g}" if longitude_min >= 0 else f"- {abs(longitude_min):g}"
    )
    longitude_formula = (
        f"(({x_center}) / {board_width}) * {longitude_span:g} {longitude_offset}"
    )
    latitude_formula = (
        f"{latitude_max:g} - (({y_center}) / {board_height}) * {latitude_span:g}"
    )
    longitude_edge_formula = (
        f"xBoard / {board_width} * {longitude_span:g} {longitude_offset}"
    )
    latitude_edge_formula = (
        f"{latitude_max:g} - yBoard / {board_height} * {latitude_span:g}"
    )
    land = fields.land_mask
    lake_mask = fields.lake_mask
    inland_water = fields.inland_water_mask
    ocean = fields.ocean_mask
    water_candidate = fields.water_candidate_mask
    continuous_elevation = fields.continuous_elevation
    class_grid = fields.elevation_ordinal
    bathymetry_grid = fields.bathymetry
    authored_river_constraint = fields.authored_river_constraint
    hydrology = fields.hydrology
    inland_components = [list(component) for component in fields.inland_components]
    lake_component_indices = fields.lake_component_indices

    # Each layer is a cumulative threshold: layer 01 is the entire authored
    # land base, and each following layer keeps cells at or above its ordinal
    # threshold.  Components are cleaned before polygonization and then
    # clipped to the preceding mask so rounded SVG faces remain nested.
    cumulative_bands: list[Polygon | MultiPolygon] = []
    elevation_masks: list[np.ndarray] = []
    elevation_cleanup_records: list[dict[str, Any]] = []
    authored_elevation_counts = [
        int(np.count_nonzero(land & (class_grid == index)))
        for index in range(context.elevation_levels)
    ]
    for index in range(ELEVATION_LEVELS):
        raw_mask = land & (class_grid >= index)
        if index == 0:
            # The base mask is the authored land/island registration.  It is
            # never area-filtered, so a genuine small island cannot disappear
            # merely because it is small at this authoring resolution.
            cleaned_mask = raw_mask.copy()
            labels, components = _connected_components(
                raw_mask,
                ELEVATION_CLEANUP_POLICY.connectivity,
            )
            min_cells, min_width, min_hole_cells, min_hole_width = (
                _resolve_cleanup_thresholds(
                    raw_mask,
                    ELEVATION_CLEANUP_POLICY,
                    int(np.count_nonzero(land)),
                )
            )
            cleanup_report = RasterCleanupStats(
                semantic=ELEVATION_CLEANUP_POLICY.semantic,
                connectivity=ELEVATION_CLEANUP_POLICY.connectivity,
                input_components=len(components),
                retained_components=len(components),
                removed_components=0,
                filled_holes=0,
                input_cells=int(np.count_nonzero(raw_mask)),
                output_cells=int(np.count_nonzero(cleaned_mask)),
                resolved_min_component_cells=min_cells,
                resolved_min_component_width=min_width,
                resolved_min_hole_cells=min_hole_cells,
                resolved_min_hole_width=min_hole_width,
            )
        else:
            persistence = land & (class_grid >= index + 1) if index + 1 < ELEVATION_LEVELS else None
            cleaned_mask, cleanup_report = cleanup_raster_mask(
                raw_mask,
                ELEVATION_CLEANUP_POLICY,
                persistence_mask=persistence,
                reference_cell_count=int(np.count_nonzero(land)),
            )
            cleaned_mask &= elevation_masks[-1]
        elevation_masks.append(cleaned_mask)
        elevation_cleanup_records.append(
            {
                "ordinal": index + 1,
                "policy": ELEVATION_CLEANUP_POLICY.as_record(),
                "stats": cleanup_report.as_record(include_components=include_debug),
                "postNestedCells": int(np.count_nonzero(cleaned_mask)),
            }
        )

    cleaned_elevation_counts = [
        int(np.count_nonzero(elevation_masks[index]))
        - (
            int(np.count_nonzero(elevation_masks[index + 1]))
            if index + 1 < ELEVATION_LEVELS
            else 0
        )
        for index in range(ELEVATION_LEVELS)
    ]
    elevation_contour_engine = _ContinuousContourEngine(
        continuous_elevation,
        land,
        upsample_factor=4,
        wrap_columns=_edge_policy_wraps_longitude(context.edge_policy),
    )
    for index in range(ELEVATION_LEVELS):
        threshold_mask = elevation_masks[index]
        smoothed_geometry = _vectorize_polygon_mask(
            threshold_mask,
            simplify_tolerance=ELEVATION_CONTOUR_SIMPLIFY_TOLERANCE,
            context=context,
            continuous_surface=(continuous_elevation if index > 0 else None),
            threshold=(index / ELEVATION_LEVELS if index > 0 else None),
            continuous_valid=(land if index > 0 else None),
            continuous_upsample_factor=(4 if index > 0 else 1),
            continuous_engine=(elevation_contour_engine if index > 0 else None),
        )
        if index > 0:
            smoothed_geometry, nested_cleanup_report = _repair_nested_geometry(
                smoothed_geometry,
                cumulative_bands[index - 1],
                return_stats=True,
            )
            elevation_cleanup_records[index]["postNestedGeometry"] = (
                nested_cleanup_report.as_record()
            )
        if smoothed_geometry.is_empty or not smoothed_geometry.is_valid:
            raise SourceBuildError(f"elevation-{index + 1:02d} smoothing produced invalid geometry")
        cumulative_bands.append(smoothed_geometry)
    for index in range(1, len(cumulative_bands)):
        outside = cumulative_bands[index].difference(cumulative_bands[index - 1]).area
        if outside > 1e-6:
            raise SourceBuildError(
                f"elevation-{index + 1:02d} escapes its lower cumulative layer by {outside}"
            )

    cumulative_bathymetry: list[Polygon | MultiPolygon] = []
    bathymetry_masks: list[np.ndarray] = []
    bathymetry_cleanup_records: list[dict[str, Any]] = []
    maritime_water = ocean | fields.inland_sea_mask
    authored_bathymetry_counts = [
        int(np.count_nonzero(maritime_water & (bathymetry_grid == index)))
        for index in range(context.bathymetry_levels)
    ]
    for index in range(BATHYMETRY_LEVELS):
        raw_mask = maritime_water & (bathymetry_grid >= index)
        if index == 0:
            cleaned_mask = raw_mask.copy()
            labels, components = _connected_components(
                raw_mask,
                BATHYMETRY_CLEANUP_POLICY.connectivity,
            )
            min_cells, min_width, min_hole_cells, min_hole_width = (
                _resolve_cleanup_thresholds(
                    raw_mask,
                    BATHYMETRY_CLEANUP_POLICY,
                    int(np.count_nonzero(maritime_water)),
                )
            )
            cleanup_report = RasterCleanupStats(
                semantic=BATHYMETRY_CLEANUP_POLICY.semantic,
                connectivity=BATHYMETRY_CLEANUP_POLICY.connectivity,
                input_components=len(components),
                retained_components=len(components),
                removed_components=0,
                filled_holes=0,
                input_cells=int(np.count_nonzero(raw_mask)),
                output_cells=int(np.count_nonzero(cleaned_mask)),
                resolved_min_component_cells=min_cells,
                resolved_min_component_width=min_width,
                resolved_min_hole_cells=min_hole_cells,
                resolved_min_hole_width=min_hole_width,
            )
        else:
            persistence = maritime_water & (bathymetry_grid >= index + 1) if index + 1 < BATHYMETRY_LEVELS else None
            cleaned_mask, cleanup_report = cleanup_raster_mask(
                raw_mask,
                BATHYMETRY_CLEANUP_POLICY,
                persistence_mask=persistence,
                reference_cell_count=int(np.count_nonzero(maritime_water)),
            )
            cleaned_mask &= bathymetry_masks[-1]
        bathymetry_masks.append(cleaned_mask)
        bathymetry_cleanup_records.append(
            {
                "ordinal": index + 1,
                "policy": BATHYMETRY_CLEANUP_POLICY.as_record(),
                "stats": cleanup_report.as_record(include_components=include_debug),
                "postNestedCells": int(np.count_nonzero(cleaned_mask)),
            }
        )
    cleaned_bathymetry_counts = [
        int(np.count_nonzero(bathymetry_masks[index]))
        - (
            int(np.count_nonzero(bathymetry_masks[index + 1]))
            if index + 1 < BATHYMETRY_LEVELS
            else 0
        )
        for index in range(BATHYMETRY_LEVELS)
    ]
    for index in range(BATHYMETRY_LEVELS):
        threshold_mask = bathymetry_masks[index]
        smoothed_geometry = _vectorize_polygon_mask(
            threshold_mask,
            preserve_board_edges=True,
            context=context,
        )
        if index > 0:
            smoothed_geometry, nested_cleanup_report = _repair_nested_geometry(
                smoothed_geometry,
                cumulative_bathymetry[index - 1],
                return_stats=True,
            )
            bathymetry_cleanup_records[index]["postNestedGeometry"] = (
                nested_cleanup_report.as_record()
            )
        if smoothed_geometry.is_empty or not smoothed_geometry.is_valid:
            raise SourceBuildError(
                f"bathymetry-{index + 1:02d} smoothing produced invalid geometry"
            )
        cumulative_bathymetry.append(smoothed_geometry)
    for index in range(1, len(cumulative_bathymetry)):
        outside = cumulative_bathymetry[index].difference(
            cumulative_bathymetry[index - 1]
        ).area
        if outside > 1e-6:
            raise SourceBuildError(
                f"bathymetry-{index + 1:02d} escapes its lower cumulative layer by {outside}"
            )

    debug_removed_features: list[dict[str, Any]] = []
    debug_filled_features: list[dict[str, Any]] = []
    if include_debug:

        def append_debug_components(
            stats_record: Mapping[str, Any],
            semantic: str,
            ordinal: int,
        ) -> None:
            for bbox_key, kind, target in (
                ("removedComponentBboxes", "removed-component", debug_removed_features),
                ("filledHoleBboxes", "filled-hole", debug_filled_features),
            ):
                bboxes = stats_record.get(bbox_key, ())
                if not isinstance(bboxes, list):
                    continue
                for component_index, bbox in enumerate(bboxes):
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    min_column, min_row, max_column, max_row = (
                        int(value) for value in bbox
                    )
                    if max_column <= min_column or max_row <= min_row:
                        continue
                    target.append(
                        {
                            "id": f"cleanup-{kind}-{semantic}-{ordinal:02d}-{component_index:04d}",
                            "kind": f"cleanup-{kind}",
                            "ordinal": ordinal,
                            "sourceRule": semantic,
                            "geometry": _polygon_mapping(
                                box(min_column, min_row, max_column, max_row),
                                context=context,
                            ),
                        }
                    )

        water_record = fields.raster_cleanup.get("waterCandidate", {})
        if isinstance(water_record, Mapping):
            water_stats = water_record.get("stats", {})
            if isinstance(water_stats, Mapping):
                append_debug_components(water_stats, "surface-water-candidate", 0)
        for record in elevation_cleanup_records:
            append_debug_components(
                record.get("stats", {}),
                "elevation-cumulative-band",
                int(record["ordinal"]),
            )
        for record in bathymetry_cleanup_records:
            append_debug_components(
                record.get("stats", {}),
                "bathymetry-cumulative-band",
                int(record["ordinal"]),
            )

    # Taking the smoothed base exteriors keeps every island/coast ring closed;
    # the independent lake layer owns inland-water shorelines.
    coastline = _exterior_lines(cumulative_bands[0])
    if coastline.is_empty:
        raise SourceBuildError("surface classification produced no coastline")

    latitude_rows = LATITUDE_MAX - (
        (np.arange(GRID_HEIGHT, dtype=np.float64) + 0.5)
        * GRID_Y_STEP
        / BOARD_HEIGHT
        * (LATITUDE_MAX - LATITUDE_MIN)
    )
    snow_geometry = _vectorize_polygon_mask(
        fields.perennial_snow_mask,
        simplify_tolerance=0.35,
        context=context,
    )
    snow_geometry, snow_nested_cleanup_report = _repair_nested_geometry(
        snow_geometry,
        cumulative_bands[0],
        return_stats=True,
    )
    if not snow_geometry.is_empty and not snow_geometry.is_valid:
        raise SourceBuildError("perennial snowline mask produced invalid land geometry")

    lake_entries: list[dict[str, Any]] = []
    accepted_lake_component_count = 0
    ordered_components = sorted(
        inland_components,
        key=lambda cells: (
            -len(cells),
            min(row for row, _ in cells),
            min(column for _, column in cells),
        ),
    )
    for component in ordered_components:
        area, aspect, fill = _component_shape(component)
        component_mask = _component_mask(component, water_candidate.shape)
        # ``inland_components`` was rebuilt from the final semantic lake mask
        # in ``prepare_physical_fields``.  Shape heuristics belong to the
        # earlier water-classification stage and must not be re-applied here:
        # long tectonic lakes and tiny residual lakes still need shoreline
        # geometry and stable IDs for river terminals.
        if component:
            accepted_lake_component_count += 1
            raw_lake = _polygon_only(make_valid(_sampled_polygonize(component_mask)))
            lake = _continuous_component_geometry(
                component,
                water_candidate.shape,
                upsample_factor=4,
            )
            lake = _polygon_only(make_valid(lake))
            if lake.is_empty or lake.area < 1.0:
                raise SourceBuildError(
                    "accepted lake component could not be published as a valid area"
                )
            area_drift_fraction = abs(float(lake.area) - float(raw_lake.area)) / float(
                raw_lake.area
            )
            boundary_displacement = float(
                lake.boundary.hausdorff_distance(raw_lake.boundary)
            )
            if area_drift_fraction > 0.03:
                raise SourceBuildError(
                    "continuous lake shoreline exceeded the three-percent area gate"
                )
            if boundary_displacement > 1.5:
                raise SourceBuildError(
                    "continuous lake shoreline exceeded the 1.5-cell displacement gate"
                )
            lake_entries.append(
                {
                    "id": f"lake-{len(lake_entries) + 1:03d}",
                    "area": area,
                    "aspect": aspect,
                    "fill": fill,
                    "geometry": lake,
                    "cells": tuple(component),
                    "geometryQuality": {
                        "method": "area-matched-upsampled-occupancy-contour",
                        "upsampleFactor": 4,
                        "areaDriftFraction": _round_relative(area_drift_fraction),
                        "maxBoundaryDisplacementGridUnits": _round_relative(
                            boundary_displacement
                        ),
                        "axisArtifacts": _polygon_axis_artifact_stats(
                            [lake],
                            long_segment_grid_units=1.5,
                        ),
                    },
                }
            )
    if len(lake_entries) != accepted_lake_component_count:
        raise SourceBuildError(
            "accepted lake component count does not match published lake geometry count"
        )

    elevation_features: list[dict[str, Any]] = []
    for index, geometry in enumerate(cumulative_bands, start=1):
        ordinal = index - 1
        # The published cumulative mask is ``surface >= ordinal / levels``;
        # keep metadata tied to that same DEM threshold rather than spreading
        # the ordinal labels over ``levels - 1`` intervals.
        threshold = ordinal / ELEVATION_LEVELS
        elevation_features.append(
            {
                "id": f"elevation-{index:02d}",
                "kind": "elevation-band",
                "ordinal": index,
                "threshold": _round_relative(threshold),
                "minOrdinal": ordinal,
                "nestedWithin": None if index == 1 else f"elevation-{index - 1:02d}",
                "authoredRelativeElevation": {
                    "min": _round_relative(threshold),
                    "max": 1.0,
                },
                "sourceOrdinalCellCount": authored_elevation_counts[ordinal],
                "cleanedOrdinalCellCount": cleaned_elevation_counts[ordinal],
            "geometry": _polygon_mapping(geometry, context=context),
            }
        )

    bathymetry_features: list[dict[str, Any]] = []
    for index, geometry in enumerate(cumulative_bathymetry, start=1):
        ordinal = index - 1
        threshold = ordinal / (BATHYMETRY_LEVELS - 1)
        feature: dict[str, Any] = {
            "id": f"bathymetry-{index:02d}",
            "kind": "bathymetry-band",
            "ordinal": index,
            "threshold": _round_relative(threshold),
            "minOrdinal": ordinal,
            "nestedWithin": None if index == 1 else f"bathymetry-{index - 1:02d}",
            "authoredRelativeBathymetry": {
                "min": _round_relative(threshold),
                "max": 1.0,
            },
            "sourceOrdinalCellCount": authored_bathymetry_counts[ordinal],
            "cleanedOrdinalCellCount": cleaned_bathymetry_counts[ordinal],
            "geometry": _polygon_mapping(
                geometry, wrap_right_edge=True, context=context
            ),
        }
        if index == BATHYMETRY_LEVELS:
            feature["edgeRule"] = "authoring-board-edge-ocean-forced-deepest"
        bathymetry_features.append(feature)

    lake_features = [
        {
            "id": entry["id"],
            "kind": "lake",
            "geometry": _polygon_mapping(entry["geometry"], context=context),
            "sourceRule": "inland-water-component-area-matched-continuous-shoreline",
            "geometryQuality": entry["geometryQuality"],
        }
        for entry in lake_entries
    ]
    waterbody_geometries = {
        entry["id"]: entry["geometry"] for entry in lake_entries
    }
    lake_cell_waterbody_ids = {
        cell: str(entry["id"])
        for entry in lake_entries
        for cell in entry["cells"]
    }
    river_features, hydrology_stats = _hydrology_features(
        hydrology,
        coastline,
        context=context,
        waterbody_geometries=waterbody_geometries,
        lake_cell_waterbody_ids=lake_cell_waterbody_ids,
    )
    snow_cells = np.argwhere(fields.perennial_snow_mask)
    snow_latitudes = [float(latitude_rows[int(row)]) for row, _ in snow_cells]
    snow_latitude_range = (
        [
            _round_relative(min(snow_latitudes)),
            _round_relative(max(snow_latitudes)),
        ]
        if snow_latitudes
        else None
    )
    snow_features = []
    if not snow_geometry.is_empty:
        snow_features.append(
            {
                "id": "snowline-perennial",
                "kind": "perennialSnow",
                "modelName": SNOWLINE_MODEL_NAME,
                "relativeUnits": "authored-relative-elevation-[0,1]-not-metres",
                "parameters": {
                    "equatorThreshold": _round_relative(SNOWLINE_CONFIG.equator_threshold),
                    "poleThreshold": _round_relative(SNOWLINE_CONFIG.pole_threshold),
                    "exponent": _round_relative(SNOWLINE_CONFIG.exponent),
                    "maxOffset": _round_relative(SNOWLINE_CONFIG.max_offset),
                    "polarFullSnowLatitude": _round_relative(
                        SNOWLINE_CONFIG.polar_full_snow_latitude
                    ),
                },
                "sourceRule": "derived-from-continuous-relative-elevation-and-latitude",
                "sourceSnowCellCount": int(fields.perennial_snow_mask.sum()),
                "latitudeRange": snow_latitude_range,
                "geometry": _polygon_mapping(snow_geometry, context=context),
            }
        )

    result = {
        "format": context.output_format,
        "contractId": context.output_contract_id,
        "schemaVersion": context.output_schema_version,
        "canonical": context.canonical,
        "profileId": context.profile_id,
        "generationSoftware": _generation_software(),
        "coordinatePrecisionStatus": "provisional",
        "sourceMethod": "georeferenced-artwork",
        "coordRefSys": context.coordinate_reference_system_id,
        "coordinateReferenceSystem": {
            "id": context.coordinate_reference_system_id,
            "name": context.coordinate_reference_system_name,
            "datum": context.datum,
            "coordinateOrder": context.coordinate_order,
            "longitudeDomain": f"[{longitude_min:g},{longitude_max:g})",
            "latitudeDomain": f"[{latitude_min:g},{latitude_max:g}]",
            "longitudePositiveDirection": context.longitude_positive_direction,
            "latitudeReference": context.latitude_reference,
            "primeMeridianDegrees": context.prime_meridian_degrees,
            "unit": context.coordinate_unit,
            "projection": context.projection,
            "wgs84": context.wgs84,
            "worldRadiusKm": context.planet.radius_km,
        },
        "authoringGrid": {
            "projection": context.projection,
            "widthPx": board_width,
            "heightPx": board_height,
            "aspectRatio": _round_relative(board_width / board_height),
            "origin": "top-left",
            "axis": context.axis,
            "edgePolicy": context.edge_policy,
            "longitudeExtent": [longitude_min, longitude_max],
            "latitudeExtent": [latitude_min, latitude_max],
            "paddingPx": {
                "left": left_padding,
                "right": right_padding,
                "top": top_padding,
                "bottom": bottom_padding,
            },
            "pixelToLonLat": {
                "longitude": longitude_formula,
                "latitude": latitude_formula,
                "edgeVertices": f"{longitude_edge_formula}; {latitude_edge_formula}",
            },
            "nominalSamplingDegrees": round(
                (longitude_max - longitude_min) / grid_width, 4
            ),
            "nominalSamplingIsNotPositionAccuracy": True,
        },
        "source": {
            "name": source_name,
            "artifact": source_artifact,
            "sha256": sha256(source_path),
            "widthPx": source_width,
            "heightPx": source_height,
            "paddingMethod": (
                f"left-{left_padding}-right-{right_padding}-top-{top_padding}-"
                f"bottom-{bottom_padding}-ocean-padding-no-vertical-rescale"
            ),
            "coordinatePrecisionStatus": "provisional",
            "sourceMethod": "georeferenced-artwork",
            "limitations": [
                "The coloured artwork is not a DEM.",
                "Relative elevation and bathymetry are authored ordinal layers; no metre or physical contour value is implied.",
                "The continuous relative elevation, hydrology and perennial snowline are illustrative derived layers; no metric discharge, snow depth or snowline altitude is implied.",
                "Coastline and lakes are extracted as editable review geometry; rivers and snowline are derived from the sampled surface and require author review.",
                "The nominal sampling interval is an internal authoring-grid interval, not positional accuracy.",
            ],
        },
        "classification": {
            "profileId": context.profile_id,
            "projection": context.projection,
            "edgePolicy": context.edge_policy,
            "elevationSemantics": "ordinal-palette",
            "continuousSurfaceMethod": ELEVATION_SURFACE_METHOD,
            "surfaceRule": "nearest ordered authored land/bathymetry palette; exterior connected water component is ocean",
            "elevationRule": f"piecewise-linear projection onto the ordered {len(land_palette)}-stop land palette, water-line inpaint, masked Gaussian regularization, structured sub-band relief, bounded valley incision, and final hydrologic correction, then subdivided into {elevation_levels} solid ordinal levels from the shared final surface; no metric elevation or area thresholds",
            "elevationRegularization": {
                "algorithm": ELEVATION_REGULARIZATION_ALGORITHM,
                "input": "palette-position-after-water-inpaint",
                "validMask": "land-mask",
                "sigmaGridUnits": _round_relative(
                    context.elevation_regularization_sigma_grid_units
                ),
                "passes": int(context.elevation_regularization_passes),
                "wrapColumns": _edge_policy_wraps_longitude(context.edge_policy),
                "wrapRows": False,
                "quantization": "floor(clamp(sharedFinalContinuousElevation * levels, 0, levels - 1))",
                "output": "regularized-input-to-structured-relief-and-two-pass-hydrology",
                "sharedSurface": True,
                "consumers": [
                    "display-ordinal-bands",
                    "hydrology",
                    "perennial-relative-snowline",
                    "settlement-features",
                ],
            },
            "elevationContour": {
                "algorithm": "cell-center-marching-squares",
                "interpolation": "linear-cell-center-threshold-crossings",
                "input": "shared-final-hydrologically-corrected-continuous-elevation",
                "validMask": "land-mask-and-cleaned-persistent-threshold-mask",
                "simplifyToleranceGridUnits": _round_relative(
                    context.elevation_contour_simplify_tolerance
                ),
                "smoothingIterations": int(smoothing_iterations),
                "wrapColumns": _edge_policy_wraps_longitude(context.edge_policy),
                "wrapRows": False,
                "thresholdRule": "ordinal / elevation-level-count",
                "continuousUpsampleFactor": 4,
                "detailRule": "upsample-continuous-field-before-contouring; reduced-vector-simplification-after-contouring",
            },
            "bathymetryRule": f"piecewise-linear projection onto the ordered {len(bathymetry_palette)}-stop bathymetry palette from shallow fringe to deep ocean, masked normalized 3x3 smoothing for {bathymetry_smoothing_iterations} iteration(s), then subdivided into {bathymetry_levels} authored ordinal levels; board-edge ocean is forced to the deepest ordinal",
            "bathymetrySmoothingIterations": BATHYMETRY_SMOOTHING_ITERATIONS,
            "bathymetrySmoothing": {
                "algorithm": "masked-normalized-box-3x3",
                "input": "continuous-authored-palette-position-[0,1]",
                "validMask": "ocean-mask",
                "iterations": BATHYMETRY_SMOOTHING_ITERATIONS,
                "wrapColumns": _edge_policy_wraps_longitude(context.edge_policy),
                "wrapRows": False,
                "quantization": "floor(clamp(smoothedPosition * levels, 0, levels - 1))",
                "preservesOceanMask": True,
            },
            "riverSmoothingIterations": RIVER_SMOOTHING_ITERATIONS,
            "riverSmoothing": {
                "algorithm": "chaikin-open-line-corner-cutting",
                "scope": "cartographic-only",
                "geometry": "published-river-linework-after-coordinate-conversion",
                "iterations": RIVER_SMOOTHING_ITERATIONS,
                "endpointRule": "from-to-confluence-lake-and-coast-mouth-coordinates-preserved",
                "topologyRule": "sourceRule-remains-derived-d8-accumulation-network",
                "topologyUnchanged": True,
                "gridEvidence": "hydrology-cells-and-node-metadata-unchanged",
                "twoPointLines": "unchanged",
            },
            "continuousRelativeElevation": {
                "rule": "palette projection, water-line inpaint, masked Gaussian regularization, structured sub-band relief, bounded drainage incision, and final hydrologic correction",
                "range": [0.0, 1.0],
                "units": "authored-relative-not-metres",
                "sourceCellCount": int(land.sum()),
                "retainedFor": [
                    "display-ordinal-bands",
                    "hydrology",
                    "perennial-relative-snowline",
                    "settlement-features",
                ],
                "terraceDiagnostics": fields.raster_cleanup["continuousSurface"],
            },
            "denoiseRule": "semantic connected-component area/width/fill/aspect cleanup with ordinal persistence and pinhole filling; authored river evidence remains a finite hydrology constraint",
            "rasterCleanup": {
                "waterCandidate": fields.raster_cleanup["waterCandidate"],
                "elevationBands": elevation_cleanup_records,
                "bathymetryBands": bathymetry_cleanup_records,
                "snowlinePostNestedGeometry": snow_nested_cleanup_report.as_record(),
                "baseMaskRule": "ordinal base masks remain unfiltered to preserve authored small islands and board registration",
            },
            "gridStepPx": GRID_STEP,
            "gridDimensions": {"width": grid_width, "height": grid_height},
            "gridCellSizePx": {
                "x": round(GRID_X_STEP, 9),
                "y": round(GRID_Y_STEP, 9),
            },
            "sampledAngularSizeDegrees": {
                "longitude": round((longitude_max - longitude_min) / grid_width, 9),
                "latitude": round((latitude_max - latitude_min) / grid_height, 9),
            },
            "antimeridianRule": "bathymetry right board edge serialized at 179.99999 to preserve the half-open longitude domain",
            "simplifyToleranceGridUnits": simplify_tolerance,
            "coastlineToleranceGridUnits": coastline_tolerance,
            "landPalette": {
                "colors": _hex_palette(LAND_PALETTE),
                "ordering": "source-authored-low-to-high",
                "interpolation": "piecewise-linear-RGB-segment-projection",
                "sourceStrata": len(land_palette),
                "outputOrdinalLevels": elevation_levels,
            },
            "bathymetryPalette": {
                "colors": _hex_palette(BATHYMETRY_PALETTE),
                "ordering": "source-authored-shallow-to-deep",
                "interpolation": "piecewise-linear-RGB-segment-projection",
                "sourceStrata": len(bathymetry_palette),
                "outputOrdinalLevels": bathymetry_levels,
            },
            "latitudeProvenance": {
                "status": "derived-for-latitude-aware-hydrology-and-snowline",
                "axis": "authoring-grid-y-to-planetary-center-latitude",
                "formula": latitude_formula,
                "intendedUses": [
                    "latitude-aware hydrology",
                    "illustrative perennial relative snowline",
                ],
                "geometryGenerated": True,
            },
            "hydrology": {
                "algorithm": HYDROLOGY_ALGORITHM,
                **HYDROLOGY_PROVENANCE,
                "flowDirection": "D8",
                "depressionCorrection": "priority-flood-with-explicit-lake-terminals",
                "constraintRule": "finite-stream-burning-of-adjacent-land-from-authored-water-palette-components",
                "thresholdUnits": "fraction-of-sampled-land-cells",
                "accumulationUnits": "land-cell-equivalents-not-physical-discharge",
                "parameters": {
                    "streamBurnDepth": _round_relative(HYDROLOGY_CONFIG.stream_burn_depth),
                    "streamThresholdFraction": _round_relative(HYDROLOGY_CONFIG.stream_threshold_fraction),
                    "tributaryThresholdFraction": _round_relative(HYDROLOGY_CONFIG.tributary_threshold_fraction),
                    "mainstemThresholdFraction": _round_relative(HYDROLOGY_CONFIG.mainstem_threshold_fraction),
                    "minimumHeadwaterLength": int(HYDROLOGY_CONFIG.minimum_headwater_length),
                    "closureBudgetFraction": _round_relative(
                        HYDROLOGY_CONFIG.closure_budget_fraction
                    ),
                    "closureMaxChainFraction": _round_relative(
                        HYDROLOGY_CONFIG.closure_max_chain_fraction
                    ),
                    "minimumStreamSpanFactor": _round_relative(
                        HYDROLOGY_CONFIG.minimum_stream_span_factor
                    ),
                    "mfdExponent": _round_relative(HYDROLOGY_CONFIG.mfd_exponent),
                    "valleyWindowFraction": _round_relative(
                        HYDROLOGY_CONFIG.valley_window_fraction
                    ),
                    "valleyDepthFraction": _round_relative(
                        HYDROLOGY_CONFIG.valley_depth_fraction
                    ),
                    "valleySupportThreshold": _round_relative(
                        HYDROLOGY_CONFIG.valley_support_threshold
                    ),
                    "valleySupportDistanceFactor": _round_relative(
                        HYDROLOGY_CONFIG.valley_support_distance_factor
                    ),
                    "parallelSearchRadiusFactor": _round_relative(
                        HYDROLOGY_CONFIG.parallel_search_radius_factor
                    ),
                    "parallelPriorityWeight": _round_relative(
                        HYDROLOGY_CONFIG.parallel_priority_weight
                    ),
                    "runoffWeightFloor": _round_relative(
                        HYDROLOGY_CONFIG.runoff_weight_floor
                    ),
                    "snowmeltRunoffBonus": _round_relative(
                        HYDROLOGY_CONFIG.snowmelt_runoff_bonus
                    ),
                    "headwaterUplandQuantile": _round_relative(
                        HYDROLOGY_CONFIG.headwater_upland_quantile
                    ),
                    "lowlandHeadwaterFlowMultiplier": _round_relative(
                        HYDROLOGY_CONFIG.lowland_headwater_flow_multiplier
                    ),
                    "majorRiversPerContinent": int(
                        HYDROLOGY_CONFIG.major_rivers_per_continent
                    ),
                    "majorRiverTributaries": int(
                        HYDROLOGY_CONFIG.major_river_tributaries
                    ),
                    "continentMinimumLandFraction": _round_relative(
                        HYDROLOGY_CONFIG.continent_minimum_land_fraction
                    ),
                    "majorRiverMinimumSpanFactor": _round_relative(
                        HYDROLOGY_CONFIG.major_river_minimum_span_factor
                    ),
                },
                "sampledLandCellCount": int(land.sum()),
                "sampledLakeCellCount": int(lake_mask.sum()),
                "authoredRiverConstraintCellCount": int(np.count_nonzero(authored_river_constraint)),
                "streamCellCount": int(hydrology.stream_mask.sum()),
                "diagnostics": hydrology.diagnostics,
                **hydrology_stats,
            },
            "snowline": {
                "modelName": SNOWLINE_MODEL_NAME,
                "relativeUnits": "authored-relative-elevation-[0,1]-not-metres",
                "formula": "clamp(polar-cap-or-pole + (equator - pole) * cos(abs(latitude))^exponent + offset, 0, 1)",
                "parameters": {
                    "equatorThreshold": _round_relative(SNOWLINE_CONFIG.equator_threshold),
                    "poleThreshold": _round_relative(SNOWLINE_CONFIG.pole_threshold),
                    "exponent": _round_relative(SNOWLINE_CONFIG.exponent),
                    "maxOffset": _round_relative(SNOWLINE_CONFIG.max_offset),
                    "polarFullSnowLatitude": _round_relative(
                        SNOWLINE_CONFIG.polar_full_snow_latitude
                    ),
                },
                "sourceSnowCellCount": int(fields.perennial_snow_mask.sum()),
                "latitudeRange": snow_latitude_range,
                "oceanExcluded": True,
            },
            "smoothing": {
                "algorithm": "chaikin-corner-cutting",
                "iterations": SMOOTHING_ITERATIONS,
                "topologyRepair": "make_valid-and-intersection-with-lower-layer-then-relative-grid-artifact-cleanup",
                "closedRingEndpoints": "preserved",
                "riverEndpointsAndBranches": "preserved-by-open-line-endpoints",
            },
            "protectedFeatures": [
                "ocean-padding-boundary",
                "inland-water-components",
                "coastline-island-components",
                "small-islands-retained-in-base-land-mask",
                "persistent-ordinal-components",
                "semantic-pinhole-and-needle-cleanup",
                "river-endpoints-and-branch-points",
                "lake-islands-and-holes",
                "authored-river-constraint-as-finite-burn-only",
            ],
            "sampledSurfaceCellCount": int(fields.grid_shape[0] * fields.grid_shape[1]),
            "sampledLandCellCount": int(land.sum()),
            "sampledOceanCellCount": int(ocean.sum()),
            "sampledInlandWaterCellCount": int(inland_water.sum()),
        },
        "layers": {
            "elevation": {
                "coverageRule": "nested-cumulative-authored-elevation",
                "features": elevation_features,
            },
            "bathymetry": {
                "coverageRule": "nested-cumulative-authored-relative-bathymetry",
                "features": bathymetry_features,
            },
            "coastline": {
                "coverageRule": "ocean-adjacent-land-interface-only",
                "features": [
                    {
                        "id": "coastline-main",
                        "kind": "coastline",
                        "geometry": _line_mapping(coastline, context=context),
                    }
                ],
            },
            "lakes": {
                "coverageRule": "independent-inland-water-polygons",
                "features": lake_features,
            },
            "rivers": {
                "coverageRule": "derived-d8-accumulation-network",
                "features": river_features,
            },
            "snowline": {
                "coverageRule": "derived-illustrative-perennial-relative-snowline",
                "features": snow_features,
            },
        },
    }
    if include_debug:
        result["debugLayers"] = {
            "removed-components": debug_removed_features,
            "filled-holes": debug_filled_features,
        }
    return result


def build(
    source_path: Path,
    output_path: Path,
    *,
    include_debug: bool = False,
) -> dict[str, Any]:
    source = _build_source(
        source_path,
        include_debug=include_debug,
        context=reviewed_context(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(source, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return source


__all__ = [
    "BATHYMETRY_LEVELS",
    "BATHYMETRY_SMOOTHING_ITERATIONS",
    "RIVER_SMOOTHING_ITERATIONS",
    "BOARD_HEIGHT",
    "BOARD_WIDTH",
    "ELEVATION_LEVELS",
    "GRID_HEIGHT",
    "GRID_STEP",
    "GRID_WIDTH",
    "LEFT_PADDING",
    "BuildContext",
    "PhysicalFields",
    "reviewed_context",
    "SourceBuildError",
    "build",
    "prepare_physical_fields",
]
