"""Pure crust and ocean-cooling primitives for the tectonic relief stage.

This module deliberately stops at the physical inputs needed by the later
relief kernels.  It does not read a source file, alter a DEM, or infer a
plate boundary.  Every public operation allocates its result and validates
the masks and units it receives.

The ocean-cooling curve is a bounded square-root response::

    f(age) = sqrt(age / tau) / (1 + sqrt(age / tau))
    depth  = ridge_depth + (mature_depth - ridge_depth) * f(age)

It starts at the ridge depth and approaches, without crossing, the mature
ocean depth as crustal age grows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


# The labels are values, not display strings.  Keeping them as uint8 makes
# the raster contract explicit and keeps later layer composition compact.
CRUST_OCEANIC: np.uint8 = np.uint8(0)
CRUST_TRANSITIONAL: np.uint8 = np.uint8(1)
CRUST_CONTINENTAL: np.uint8 = np.uint8(2)

_CRUST_VALUES = np.array(
    [CRUST_OCEANIC, CRUST_TRANSITIONAL, CRUST_CONTINENTAL],
    dtype=np.uint8,
)
_CONVERGENT_CLASSIFICATIONS = frozenset({"convergent", "oblique-convergent"})
_BOUNDARY_CLASSIFICATIONS = frozenset(
    {
        "stable",
        "transform",
        "divergent",
        "oblique-divergent",
        "convergent",
        "oblique-convergent",
    }
)
_DECISION_REASONS_WITHOUT_PLATES = frozenset({"not-convergent", "continental-collision"})
_DECISION_REASONS_WITH_PLATES = frozenset(
    {
        "oceanic-under-continental",
        "older-oceanic",
        "stronger-convergence-drive",
        "stable-plate-id",
    }
)
_DECISION_REASONS = _DECISION_REASONS_WITHOUT_PLATES | _DECISION_REASONS_WITH_PLATES
_PROFILE_DOMAINS = frozenset({"oceanic", "non-oceanic", "any"})
_PROFILE_CODES = frozenset(
    {
        "ridge-shoulder",
        "axial-rift",
        "rift-shoulder",
        "central-graben",
        "boundary-basin",
        "trench",
        "forearc",
        "island-arc",
        "backarc-basin",
        "continental-arc",
        "fold-thrust-belt",
        "inland-foreland-basin",
        "collision-orogen",
        "broad-plateau",
        "foreland-basin",
        "strike-slip-valley",
        "pull-apart-basin",
        "restraining-ridge",
    }
)


def _numeric_array(value: Any, *, name: str, allow_scalar: bool) -> np.ndarray:
    """Return a real float array without accepting coercion surprises."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real numeric array") from error
    if not allow_scalar and raw.ndim == 0:
        raise ValueError(f"{name} must be an array")
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain real numeric values")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    return array


def _boolean_mask(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a boolean mask") from error
    if array.ndim == 0 or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean mask")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")
    return array


def _finite_nonnegative_scalar(value: Any, *, name: str, strictly_positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if raw.ndim != 0 or raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number) or number < 0.0 or (strictly_positive and number <= 0.0):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return number


def _finite_scalar(value: Any, *, name: str) -> float:
    """Return a finite real scalar without imposing a sign."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if raw.ndim != 0 or raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_distance(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    allow_nan: bool = False,
) -> np.ndarray:
    array = _numeric_array(value, name=name, allow_scalar=False)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} must not contain infinity")
    if not allow_nan and np.any(np.isnan(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(np.isfinite(array) & (array < 0.0)):
        raise ValueError(f"{name} must be non-negative")
    return array


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    if value.ndim == 0:
        return float(value)
    return value


@dataclass(frozen=True, slots=True)
class CrustFields:
    """A validated crust-kind raster paired with oceanic crust age.

    ``ocean_age_myr`` is meaningful only for oceanic cells.  Continental
    and transitional cells carry ``NaN`` so that an accidental use of an
    ocean-age value on land is visible rather than silently plausible.
    """

    kind: np.ndarray
    ocean_age_myr: np.ndarray

    def __post_init__(self) -> None:
        try:
            raw_kind = np.asarray(self.kind)
        except (TypeError, ValueError) as error:
            raise ValueError("kind must be an integer array") from error
        if raw_kind.ndim == 0 or raw_kind.dtype.kind not in {"i", "u"}:
            raise ValueError("kind must be a non-scalar integer array")
        try:
            kind_int = np.asarray(self.kind, dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("kind must be an integer array") from error
        if not np.all(np.isin(kind_int, _CRUST_VALUES.astype(np.int64))):
            raise ValueError("kind contains an unknown crust label")

        ages = _numeric_array(self.ocean_age_myr, name="ocean_age_myr", allow_scalar=False)
        if ages.shape != raw_kind.shape:
            raise ValueError(
                f"ocean_age_myr shape {ages.shape} does not match kind shape {raw_kind.shape}"
            )
        if np.any(np.isinf(ages)):
            raise ValueError("ocean_age_myr must not contain infinity")
        oceanic = kind_int == int(CRUST_OCEANIC)
        if np.any(oceanic & (~np.isfinite(ages) | (ages < 0.0))):
            raise ValueError("oceanic cells require finite non-negative ocean_age_myr")
        if np.any((~oceanic) & ~np.isnan(ages)):
            raise ValueError("non-oceanic cells require NaN ocean_age_myr")

        object.__setattr__(self, "kind", _readonly_copy(kind_int, dtype=np.dtype(np.uint8)))
        object.__setattr__(self, "ocean_age_myr", _readonly_copy(ages, dtype=np.dtype(np.float64)))


def classify_crust(
    solid_land_mask: Any,
    ocean_mask: Any,
    distance_to_coast_km: Any,
    shelf_width_km: Any,
) -> np.ndarray:
    """Classify continental, shelf-transition, and oceanic crust.

    ``solid_land_mask`` is the solid-surface domain and may include an
    inpainted lake floor.  The separate lake identity remains outside this
    function.  Ocean cells at or inside the explicit shelf width are
    transitional; deeper ocean cells are oceanic.
    """

    land = _boolean_mask(solid_land_mask, name="solid_land_mask")
    ocean = _boolean_mask(ocean_mask, name="ocean_mask", shape=land.shape)
    if np.any(land & ocean):
        raise ValueError("solid_land_mask and ocean_mask must be mutually exclusive")
    if not np.all(land | ocean):
        raise ValueError("solid_land_mask and ocean_mask must cover every cell")
    distance = _validate_distance(
        distance_to_coast_km,
        name="distance_to_coast_km",
        shape=land.shape,
    )
    shelf_width = _finite_nonnegative_scalar(shelf_width_km, name="shelf_width_km")

    kind = np.full(land.shape, CRUST_OCEANIC, dtype=np.uint8)
    kind[land] = CRUST_CONTINENTAL
    kind[ocean & (distance <= shelf_width)] = CRUST_TRANSITIONAL
    return kind


def compute_ocean_age_myr(
    distance_to_ridge_km: Any,
    half_spreading_rate_cm_per_year: Any,
    oceanic_mask: Any,
) -> np.ndarray:
    """Convert ridge distance and half-spreading rate into ocean age.

    The unit conversion is explicit: ``1 cm/year = 10 km/Myr``.  Values
    outside the oceanic mask are not interpreted and are returned as NaN;
    NaN is accepted there as an undefined-distance sentinel.
    """

    mask = _boolean_mask(oceanic_mask, name="oceanic_mask")
    # Distance outside oceanic cells is undefined and deliberately ignored;
    # callers may carry NaN, infinity, or another sentinel there.  Only the
    # selected oceanic cells participate in the physical validation.
    distance = _numeric_array(
        distance_to_ridge_km,
        name="distance_to_ridge_km",
        allow_scalar=False,
    )
    if distance.shape != mask.shape:
        raise ValueError(
            f"distance_to_ridge_km shape {distance.shape} does not match {mask.shape}"
        )
    if np.any(mask & (~np.isfinite(distance) | (distance < 0.0))):
        raise ValueError("oceanic distance_to_ridge_km must be finite")
    rate = _finite_nonnegative_scalar(
        half_spreading_rate_cm_per_year,
        name="half_spreading_rate_cm_per_year",
        strictly_positive=True,
    )
    age = np.full(mask.shape, np.nan, dtype=np.float64)
    age[mask] = distance[mask] / (rate * 10.0)
    return age


def ocean_depth_from_age(
    age_myr: Any,
    ridge_depth_m: Any,
    mature_depth_m: Any,
    cooling_tau_myr: Any,
) -> float | np.ndarray:
    """Return a bounded square-root cooling depth for oceanic crust.

    ``NaN`` ages remain ``NaN``.  Ridge and mature depths are negative, and
    mature depth must be deeper than ridge depth.  The output approaches the
    mature depth asymptotically and never changes the coastline mask.
    """

    age = _numeric_array(age_myr, name="age_myr", allow_scalar=True)
    if np.any(np.isinf(age)) or np.any(np.isfinite(age) & (age < 0.0)):
        raise ValueError("age_myr must contain non-negative finite values or NaN")
    if isinstance(ridge_depth_m, (bool, np.bool_)) or isinstance(mature_depth_m, (bool, np.bool_)):
        raise ValueError("ridge_depth_m and mature_depth_m must be finite non-positive numbers")
    try:
        ridge = float(ridge_depth_m)
        mature = float(mature_depth_m)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("ridge_depth_m and mature_depth_m must be finite numbers") from error
    if not math.isfinite(ridge) or not math.isfinite(mature) or ridge > 0.0 or mature > 0.0:
        raise ValueError("ridge_depth_m and mature_depth_m must be finite non-positive numbers")
    if mature >= ridge:
        raise ValueError("mature_depth_m must be deeper than ridge_depth_m")
    tau = _finite_nonnegative_scalar(
        cooling_tau_myr,
        name="cooling_tau_myr",
        strictly_positive=True,
    )

    square_root_age = np.sqrt(age / tau)
    fraction = square_root_age / (1.0 + square_root_age)
    depth = ridge + (mature - ridge) * fraction
    return _scalar_or_array(np.asarray(depth, dtype=np.float64))


def _plate_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty plate id")
    return value


def _crust_scalar(value: Any, *, name: str) -> np.uint8:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a crust label")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a crust label") from error
    if raw.ndim != 0 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a crust label")
    try:
        numeric = int(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a crust label") from error
    if numeric not in {int(CRUST_OCEANIC), int(CRUST_TRANSITIONAL), int(CRUST_CONTINENTAL)}:
        raise ValueError(f"{name} contains an unknown crust label")
    return np.uint8(numeric)


def _age_scalar(value: Any, *, name: str, required: bool) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative number or NaN")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite non-negative number or NaN") from error
    if raw.ndim != 0 or raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must be a finite non-negative number or NaN")
    number = float(raw)
    if required:
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be finite and non-negative for oceanic crust")
    elif not math.isnan(number):
        raise ValueError(f"{name} must be NaN for non-oceanic crust")
    return number


@dataclass(frozen=True, slots=True)
class SubductionDecision:
    """The deterministic result of one convergent boundary comparison."""

    subducting_plate_id: str | None
    overriding_plate_id: str | None
    reason: str

    def __post_init__(self) -> None:
        subducting = self.subducting_plate_id
        overriding = self.overriding_plate_id
        if subducting is not None:
            _plate_id(subducting, name="subducting_plate_id")
        if overriding is not None:
            _plate_id(overriding, name="overriding_plate_id")
        if (subducting is None) != (overriding is None):
            raise ValueError("subducting and overriding plate ids must be both set or both None")
        if subducting is not None and subducting == overriding:
            raise ValueError("subduction plate ids must be distinct")
        if not isinstance(self.reason, str) or self.reason not in _DECISION_REASONS:
            raise ValueError("reason is not a recognized subduction decision code")
        if self.reason in _DECISION_REASONS_WITHOUT_PLATES and (
            subducting is not None or overriding is not None
        ):
            raise ValueError(f"{self.reason} requires both plate ids to be None")
        if self.reason in _DECISION_REASONS_WITH_PLATES and (
            subducting is None or overriding is None
        ):
            raise ValueError(f"{self.reason} requires two plate ids")


@dataclass(frozen=True, slots=True)
class TectonicProfileConfig:
    """Explicit amplitudes and length scales for one boundary profile."""

    distance_scale_km: float
    reference_speed_cm_per_year: float
    max_speed_multiplier: float
    ridge_uplift_m: float
    rift_shoulder_uplift_m: float
    trench_depth_m: float
    arc_uplift_m: float
    collision_uplift_m: float
    basin_depth_m: float
    transform_valley_depth_m: float

    def __post_init__(self) -> None:
        for field_name in (
            "distance_scale_km",
            "reference_speed_cm_per_year",
            "max_speed_multiplier",
            "ridge_uplift_m",
            "rift_shoulder_uplift_m",
            "trench_depth_m",
            "arc_uplift_m",
            "collision_uplift_m",
            "basin_depth_m",
            "transform_valley_depth_m",
        ):
            value = _finite_nonnegative_scalar(
                getattr(self, field_name),
                name=field_name,
                strictly_positive=True,
            )
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class ProfileFeature:
    """One smooth signed Gaussian lobe in a boundary-normal profile."""

    code: str
    domain: str
    center_km: float
    width_km: float
    amplitude_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or self.code not in _PROFILE_CODES:
            raise ValueError("code is not a recognized profile feature code")
        if not isinstance(self.domain, str) or self.domain not in _PROFILE_DOMAINS:
            raise ValueError("domain must be oceanic, non-oceanic, or any")
        center = _finite_scalar(self.center_km, name="center_km")
        width = _finite_nonnegative_scalar(self.width_km, name="width_km", strictly_positive=True)
        amplitude = _finite_scalar(self.amplitude_m, name="amplitude_m")
        if amplitude == 0.0:
            raise ValueError("amplitude_m must be non-zero")
        object.__setattr__(self, "center_km", center)
        object.__setattr__(self, "width_km", width)
        object.__setattr__(self, "amplitude_m", amplitude)


def choose_subduction_side(
    boundary_classification: Any,
    plate_a_id: Any,
    crust_a: Any,
    age_a_myr: Any,
    convergence_drive_a: Any,
    plate_b_id: Any,
    crust_b: Any,
    age_b_myr: Any,
    convergence_drive_b: Any,
) -> SubductionDecision:
    """Choose the subducting side using the shared relief-stage rule.

    Transitional crust is treated as continental for this decision.  An
    oblique-convergent boundary still has a convergent component and enters
    the same rule; all other boundary classes return no subduction.
    """

    if not isinstance(boundary_classification, str) or boundary_classification not in _BOUNDARY_CLASSIFICATIONS:
        raise ValueError("boundary_classification is not a recognized tectonic class")
    plate_a = _plate_id(plate_a_id, name="plate_a_id")
    plate_b = _plate_id(plate_b_id, name="plate_b_id")
    if plate_a == plate_b:
        raise ValueError("plate ids must be distinct")
    kind_a = _crust_scalar(crust_a, name="crust_a")
    kind_b = _crust_scalar(crust_b, name="crust_b")
    age_a = _age_scalar(age_a_myr, name="age_a_myr", required=kind_a == CRUST_OCEANIC)
    age_b = _age_scalar(age_b_myr, name="age_b_myr", required=kind_b == CRUST_OCEANIC)
    drive_a = _finite_nonnegative_scalar(convergence_drive_a, name="convergence_drive_a")
    drive_b = _finite_nonnegative_scalar(convergence_drive_b, name="convergence_drive_b")

    if boundary_classification not in _CONVERGENT_CLASSIFICATIONS:
        return SubductionDecision(None, None, "not-convergent")

    ocean_a = kind_a == CRUST_OCEANIC
    ocean_b = kind_b == CRUST_OCEANIC
    continental_a = not ocean_a
    continental_b = not ocean_b
    if ocean_a and continental_b:
        return SubductionDecision(plate_a, plate_b, "oceanic-under-continental")
    if ocean_b and continental_a:
        return SubductionDecision(plate_b, plate_a, "oceanic-under-continental")
    if continental_a and continental_b:
        return SubductionDecision(None, None, "continental-collision")

    # Both sides are oceanic.  Older crust is colder and denser.  Exact ties
    # then use the explicit convergence drive and stable lexical IDs.
    if age_a > age_b:
        return SubductionDecision(plate_a, plate_b, "older-oceanic")
    if age_b > age_a:
        return SubductionDecision(plate_b, plate_a, "older-oceanic")
    if drive_a > drive_b:
        return SubductionDecision(plate_a, plate_b, "stronger-convergence-drive")
    if drive_b > drive_a:
        return SubductionDecision(plate_b, plate_a, "stronger-convergence-drive")
    if plate_a < plate_b:
        return SubductionDecision(plate_a, plate_b, "stable-plate-id")
    return SubductionDecision(plate_b, plate_a, "stable-plate-id")


def _profile_plate_ids(value: Any) -> tuple[str, str]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("plate_ids must be a two-item tuple")
    first = _plate_id(value[0], name="plate_ids[0]")
    second = _plate_id(value[1], name="plate_ids[1]")
    if first == second:
        raise ValueError("plate_ids must contain two distinct ids")
    return first, second


def _profile_speed_factor(speed: float, config: TectonicProfileConfig) -> float:
    return min(abs(speed) / config.reference_speed_cm_per_year, config.max_speed_multiplier)


def _profile_feature(
    code: str,
    domain: str,
    center_ratio: float,
    width_ratio: float,
    amplitude_m: float,
    speed_factor: float,
    config: TectonicProfileConfig,
) -> ProfileFeature:
    """Construct a feature from the shared profile scale and speed factor."""

    return ProfileFeature(
        code,
        domain,
        round(center_ratio * config.distance_scale_km, 12),
        round(width_ratio * config.distance_scale_km, 12),
        amplitude_m * speed_factor,
    )


def _validate_profile_decision(
    boundary_classification: str,
    plate_ids: tuple[str, str],
    crust_a: np.uint8,
    crust_b: np.uint8,
    decision: Any,
) -> None:
    if not isinstance(decision, SubductionDecision):
        raise TypeError("subduction_decision must be a SubductionDecision")
    plate_a, plate_b = plate_ids
    convergent = boundary_classification in _CONVERGENT_CLASSIFICATIONS
    if not convergent:
        if decision.reason != "not-convergent":
            raise ValueError("non-convergent profile requires a not-convergent decision")
        return

    ocean_a = crust_a == CRUST_OCEANIC
    ocean_b = crust_b == CRUST_OCEANIC
    if not ocean_a and not ocean_b:
        if decision.reason != "continental-collision" or decision.subducting_plate_id is not None:
            raise ValueError("continental convergence requires a continental-collision decision")
        return

    if ocean_a != ocean_b:
        expected_subducting = plate_a if ocean_a else plate_b
        expected_overriding = plate_b if ocean_a else plate_a
        if (
            decision.reason != "oceanic-under-continental"
            or decision.subducting_plate_id != expected_subducting
            or decision.overriding_plate_id != expected_overriding
        ):
            raise ValueError("mixed convergence decision does not match crust and plate sides")
        return

    if decision.reason not in {
        "older-oceanic",
        "stronger-convergence-drive",
        "stable-plate-id",
    }:
        raise ValueError("ocean-ocean convergence requires an oceanic subduction decision")
    if {decision.subducting_plate_id, decision.overriding_plate_id} != {plate_a, plate_b}:
        raise ValueError("subduction decision plate ids do not match profile plate ids")


def build_boundary_profile(
    boundary_classification: Any,
    plate_ids: Any,
    crust_a: Any,
    crust_b: Any,
    subduction_decision: Any,
    normal_velocity_cm_per_year: Any,
    tangent_velocity_cm_per_year: Any,
    bend_sense: Any,
    config: Any,
) -> tuple[ProfileFeature, ...]:
    """Build one deterministic, signed Gaussian profile around a boundary.

    The profile coordinate is signed: plate A occupies the negative side and
    plate B the positive side.  Convergent profiles are emitted from the
    subducting side toward the overriding side, which makes a side reversal
    visible as a sign reversal without changing feature order.
    """

    if not isinstance(boundary_classification, str) or boundary_classification not in _BOUNDARY_CLASSIFICATIONS:
        raise ValueError("boundary_classification is not a recognized tectonic class")
    plate_pair = _profile_plate_ids(plate_ids)
    kind_a = _crust_scalar(crust_a, name="crust_a")
    kind_b = _crust_scalar(crust_b, name="crust_b")
    if not isinstance(config, TectonicProfileConfig):
        raise TypeError("config must be a TectonicProfileConfig")
    _validate_profile_decision(
        boundary_classification,
        plate_pair,
        kind_a,
        kind_b,
        subduction_decision,
    )
    normal = _finite_scalar(normal_velocity_cm_per_year, name="normal_velocity_cm_per_year")
    tangent = _finite_scalar(tangent_velocity_cm_per_year, name="tangent_velocity_cm_per_year")
    bend = _finite_scalar(bend_sense, name="bend_sense")
    if not -1.0 <= bend <= 1.0:
        raise ValueError("bend_sense must lie within [-1, 1]")

    if boundary_classification in {"divergent", "oblique-divergent"} and normal <= 0.0:
        raise ValueError("divergent profiles require positive normal velocity")
    if boundary_classification in _CONVERGENT_CLASSIFICATIONS and normal >= 0.0:
        raise ValueError("convergent profiles require negative normal velocity")

    # Ridge, trench, arc, basin, and collision strength are driven by the
    # convergent/divergent normal component.  Tangential motion belongs only
    # to the pure transform valley branch below; it must not inflate a main
    # convergent or divergent relief profile.
    normal_factor = _profile_speed_factor(normal, config)
    tangent_factor = _profile_speed_factor(tangent, config)
    features: list[ProfileFeature] = []

    if boundary_classification == "stable":
        return ()

    if boundary_classification in {"divergent", "oblique-divergent"}:
        ocean_a = kind_a == CRUST_OCEANIC
        ocean_b = kind_b == CRUST_OCEANIC
        if ocean_a and ocean_b:
            shoulder_amplitude = config.ridge_uplift_m
            shoulder_code = "ridge-shoulder"
            center_code = "axial-rift"
            left_domain = right_domain = "oceanic"
        elif not ocean_a and not ocean_b:
            shoulder_amplitude = config.rift_shoulder_uplift_m
            shoulder_code = "rift-shoulder"
            center_code = "central-graben"
            left_domain = right_domain = "non-oceanic"
        else:
            shoulder_code = "ridge-shoulder"
            center_code = "boundary-basin"
            if ocean_a:
                left_domain, right_domain = "oceanic", "non-oceanic"
            else:
                left_domain, right_domain = "non-oceanic", "oceanic"
            shoulder_amplitude = config.ridge_uplift_m
        if ocean_a != ocean_b:
            left_amplitude = config.ridge_uplift_m if left_domain == "oceanic" else config.rift_shoulder_uplift_m
            right_amplitude = config.ridge_uplift_m if right_domain == "oceanic" else config.rift_shoulder_uplift_m
            features.extend(
                (
                    _profile_feature("ridge-shoulder" if left_domain == "oceanic" else "rift-shoulder", left_domain, -0.45, 0.24, left_amplitude, normal_factor, config),
                    _profile_feature("boundary-basin", "any", 0.0, 0.22, -config.basin_depth_m, normal_factor, config),
                    _profile_feature("ridge-shoulder" if right_domain == "oceanic" else "rift-shoulder", right_domain, 0.45, 0.24, right_amplitude, normal_factor, config),
                )
            )
        else:
            features.extend(
                (
                    _profile_feature(shoulder_code, left_domain, -0.55, 0.25, shoulder_amplitude, normal_factor, config),
                    _profile_feature(center_code, "oceanic" if ocean_a and ocean_b else "non-oceanic", 0.0, 0.20, -config.basin_depth_m, normal_factor, config),
                    _profile_feature(shoulder_code, right_domain, 0.55, 0.25, shoulder_amplitude, normal_factor, config),
                )
            )
        return tuple(features)

    if boundary_classification in _CONVERGENT_CLASSIFICATIONS:
        if subduction_decision.reason == "continental-collision":
            features.extend(
                (
                    _profile_feature("foreland-basin", "non-oceanic", -0.95, 0.28, -config.basin_depth_m, normal_factor, config),
                    _profile_feature("collision-orogen", "non-oceanic", 0.0, 0.28, config.collision_uplift_m, normal_factor, config),
                    _profile_feature("broad-plateau", "non-oceanic", 0.0, 0.70, config.collision_uplift_m * 0.75, normal_factor, config),
                    _profile_feature("foreland-basin", "non-oceanic", 0.95, 0.28, -config.basin_depth_m, normal_factor, config),
                )
            )
            return tuple(features)

        plate_a, _ = plate_pair
        subducting_sign = -1.0 if subduction_decision.subducting_plate_id == plate_a else 1.0
        overriding_sign = -subducting_sign
        subducting_is_oceanic = (
            (subducting_sign < 0.0 and kind_a == CRUST_OCEANIC)
            or (subducting_sign > 0.0 and kind_b == CRUST_OCEANIC)
        )
        if kind_a == CRUST_OCEANIC and kind_b == CRUST_OCEANIC:
            features.extend(
                (
                    _profile_feature("trench", "oceanic", subducting_sign * 0.12, 0.14, -config.trench_depth_m, normal_factor, config),
                    _profile_feature("forearc", "oceanic", overriding_sign * 0.28, 0.18, config.arc_uplift_m * 0.35, normal_factor, config),
                    _profile_feature("island-arc", "oceanic", overriding_sign * 0.52, 0.22, config.arc_uplift_m, normal_factor, config),
                    _profile_feature("backarc-basin", "oceanic", overriding_sign * 0.82, 0.25, -config.basin_depth_m, normal_factor, config),
                )
            )
        elif subducting_is_oceanic:
            features.extend(
                (
                    _profile_feature("trench", "oceanic", subducting_sign * 0.12, 0.14, -config.trench_depth_m, normal_factor, config),
                    _profile_feature("forearc", "non-oceanic", overriding_sign * 0.28, 0.18, config.arc_uplift_m * 0.35, normal_factor, config),
                    _profile_feature("continental-arc", "non-oceanic", overriding_sign * 0.50, 0.20, config.arc_uplift_m, normal_factor, config),
                    _profile_feature("fold-thrust-belt", "non-oceanic", overriding_sign * 0.68, 0.24, config.collision_uplift_m * 0.55, normal_factor, config),
                    _profile_feature("inland-foreland-basin", "non-oceanic", overriding_sign * 0.95, 0.28, -config.basin_depth_m, normal_factor, config),
                )
            )
        return tuple(features)

    if boundary_classification == "transform":
        if tangent_factor <= 0.0:
            raise ValueError("transform profiles require a non-zero tangent velocity")
        features.append(
            _profile_feature(
                "strike-slip-valley",
                "any",
                0.0,
                0.24,
                -config.transform_valley_depth_m,
                tangent_factor,
                config,
            )
        )
        if bend < 0.0:
            features.append(
                _profile_feature(
                    "pull-apart-basin",
                    "any",
                    0.0,
                    0.34,
                    -config.basin_depth_m,
                    tangent_factor,
                    config,
                )
            )
        elif bend > 0.0:
            features.append(
                _profile_feature(
                    "restraining-ridge",
                    "any",
                    0.0,
                    0.34,
                    config.ridge_uplift_m,
                    tangent_factor,
                    config,
                )
            )
        return tuple(features)

    raise ValueError("unsupported boundary classification")


def _validate_sample_crust(value: Any, *, shape: tuple[int, ...]) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("sample_crust_kind must be an integer array") from error
    if raw.shape != shape or raw.ndim == 0 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError("sample_crust_kind must be an integer array matching distance shape")
    try:
        sample = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("sample_crust_kind must be an integer array") from error
    if not np.all(np.isin(sample, _CRUST_VALUES.astype(np.int64))):
        raise ValueError("sample_crust_kind contains an unknown crust label")
    return sample


def evaluate_boundary_profile(
    features: Any,
    signed_distance_km: Any,
    sample_crust_kind: Any,
) -> float | np.ndarray:
    """Evaluate matching profile lobes at signed boundary distances."""

    if not isinstance(features, tuple):
        raise ValueError("features must be a tuple of ProfileFeature")
    if any(not isinstance(feature, ProfileFeature) for feature in features):
        raise ValueError("features must contain only ProfileFeature values")
    distance = _numeric_array(signed_distance_km, name="signed_distance_km", allow_scalar=True)
    if np.any(~np.isfinite(distance)):
        raise ValueError("signed_distance_km must be finite")
    if distance.ndim == 0:
        try:
            sample_raw = np.asarray(sample_crust_kind)
        except (TypeError, ValueError) as error:
            raise ValueError("sample_crust_kind must be a crust label") from error
        if sample_raw.ndim != 0:
            raise ValueError("sample_crust_kind shape must match signed_distance_km")
        sample = _crust_scalar(sample_crust_kind, name="sample_crust_kind")
        result = 0.0
        for feature in features:
            if feature.domain == "any" or (
                feature.domain == "oceanic" and sample == CRUST_OCEANIC
            ) or (
                feature.domain == "non-oceanic" and sample != CRUST_OCEANIC
            ):
                offset = (float(distance) - feature.center_km) / feature.width_km
                result += feature.amplitude_m * math.exp(-0.5 * offset * offset)
        return np.float64(result)

    sample = _validate_sample_crust(sample_crust_kind, shape=distance.shape)
    result = np.zeros(distance.shape, dtype=np.float64)
    oceanic = sample == int(CRUST_OCEANIC)
    for feature in features:
        if feature.domain == "oceanic":
            domain = oceanic
        elif feature.domain == "non-oceanic":
            domain = ~oceanic
        else:
            domain = np.ones(distance.shape, dtype=bool)
        offset = (distance - feature.center_km) / feature.width_km
        result[domain] += feature.amplitude_m * np.exp(-0.5 * offset[domain] * offset[domain])
    return result


__all__ = [
    "CRUST_CONTINENTAL",
    "CRUST_OCEANIC",
    "CRUST_TRANSITIONAL",
    "CrustFields",
    "ProfileFeature",
    "SubductionDecision",
    "TectonicProfileConfig",
    "build_boundary_profile",
    "choose_subduction_side",
    "classify_crust",
    "compute_ocean_age_myr",
    "evaluate_boundary_profile",
    "ocean_depth_from_age",
]
