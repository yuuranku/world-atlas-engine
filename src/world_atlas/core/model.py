"""The canonical, cell-indexed physical data model for the world grid."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np


_ARRAY_DTYPES = {
    "elevation": np.dtype("float32"),
    "elevation_band": np.dtype("uint8"),
    "water": np.dtype("uint8"),
    "bathymetry_band": np.dtype("int8"),
    "flow_to": np.dtype("int32"),
    "discharge": np.dtype("float32"),
    "river_order": np.dtype("uint8"),
    "snow": np.dtype("bool"),
    "sea_ice": np.dtype("bool"),
    "seasonal_precipitation": np.dtype("uint8"),
    "seasonal_wind_east": np.dtype("int8"),
    "seasonal_wind_north": np.dtype("int8"),
    "seasonal_river_strength": np.dtype("uint8"),
}
_ARRAY_NAMES = tuple(_ARRAY_DTYPES)
_SPATIAL_ARRAY_NAMES = _ARRAY_NAMES[:9]
_SEASONAL_ARRAY_NAMES = _ARRAY_NAMES[9:]
_SEASON_COUNT = 4
_SEASON_IDS = ("vernal", "june", "autumnal", "december")
_SEASON_SOLAR_LONGITUDES = (0.0, 90.0, 180.0, 270.0)
_REQUIRED_METADATA = frozenset(
    {
        "format",
        "schema",
        "inputFingerprint",
        "planet",
        "extents",
        "elevation",
        "bathymetry",
        "climate",
    }
)
_WORLD_GRID_FORMAT = "eirenor-world-grid"
_WORLD_GRID_SCHEMA = "world-grid-v3"
_BUNDLE_FIELD = "bundle"
_CONTENT_DIGEST_FIELD = "contentDigest"


def _freeze_json(value: Any, ancestors: frozenset[int] = frozenset()) -> Any:
    """Deep-copy JSON values into immutable containers and reject unsafe data."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata numeric values must be finite")
        return value
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in ancestors:
            raise ValueError("metadata cannot contain reference cycles")
        next_ancestors = ancestors | {marker}
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("metadata object keys must be strings")
            frozen[key] = _freeze_json(item, next_ancestors)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in ancestors:
            raise ValueError("metadata cannot contain reference cycles")
        next_ancestors = ancestors | {marker}
        return tuple(_freeze_json(item, next_ancestors) for item in value)
    raise TypeError(f"metadata value is not JSON-safe: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    """Convert recursively frozen metadata back to JSON-native containers."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _is_reparse_point(path: Path) -> bool:
    """Return whether an existing path is a link, junction, or reparse point."""

    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
        attributes = os.lstat(path).st_file_attributes
    except (AttributeError, FileNotFoundError):
        return False
    except OSError:
        return True
    return bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT


def _contains_reparse_point(path: Path) -> bool:
    """Return whether *path* or an existing ancestor is a reparse point."""

    candidate = Path(os.path.abspath(path))
    while True:
        if os.path.lexists(candidate) and _is_reparse_point(candidate):
            return True
        if candidate.parent == candidate:
            return False
        candidate = candidate.parent


def _validate_save_paths(output_directory: Path) -> Path:
    """Validate the directory and two target paths before creating anything."""

    output_directory = Path(os.path.abspath(output_directory))
    if _contains_reparse_point(output_directory):
        raise ValueError(f"WorldGrid output path contains a reparse point: {output_directory}")
    if os.path.lexists(output_directory) and not output_directory.is_dir():
        raise ValueError(f"WorldGrid output path must be a directory: {output_directory}")

    if output_directory.is_dir():
        for root, directories, files in os.walk(
            output_directory, topdown=True, followlinks=False
        ):
            for name in (*directories, *files):
                child = Path(root) / name
                if _is_reparse_point(child):
                    raise ValueError(f"WorldGrid output tree contains a reparse point: {child}")

        if any(output_directory.iterdir()):
            raise ValueError(
                "WorldGrid output directory must be nonexistent or empty"
            )

    return output_directory


def _temporary_file(directory: Path, name: str, suffix: str) -> Path:
    fd, path = tempfile.mkstemp(
        prefix=f".{name}.", suffix=suffix, dir=directory
    )
    os.close(fd)
    return Path(path)


def _file_identity(path: Path) -> tuple[int, int] | None:
    try:
        stat_result = os.stat(path, follow_symlinks=False)
    except OSError:
        return None
    return stat_result.st_dev, stat_result.st_ino


def _same_file_identity(path: Path, identity: tuple[int, int] | None) -> bool:
    return identity is not None and _file_identity(path) == identity


def _unlink_owned_file(path: Path, identity: tuple[int, int] | None) -> None:
    if not os.path.lexists(path) or not _same_file_identity(path, identity):
        return
    if _is_reparse_point(path) or not path.is_file():
        raise ValueError(f"refusing to remove an unsafe transaction file: {path}")
    path.unlink()


def _atomic_replace_files(
    replacements: tuple[tuple[Path, Path], ...],
) -> None:
    """Publish fully-written siblings, requiring fresh targets.

    A caller that needs a complete multi-file bundle should stage the parent
    directory and publish that directory as one unit (as ``generate`` does).
    """

    temporary_identities = [
        (temporary, target, _file_identity(temporary))
        for temporary, target in replacements
    ]
    try:
        if any(os.path.lexists(target) for _temporary, target, _ in temporary_identities):
            raise FileExistsError(
                "atomic output targets must be fresh and absent: "
                + ", ".join(str(target) for _temporary, target, _ in temporary_identities)
            )
        for temporary, target, temporary_identity in temporary_identities:
            if temporary_identity is None:
                raise FileNotFoundError(f"transaction file disappeared: {temporary}")
            temporary.replace(target)
    except BaseException:
        # All targets were validated as fresh, so remove only files matching
        # this transaction's temporary identities after a partial replace.
        for _temporary, target, temporary_identity in temporary_identities:
            if _same_file_identity(target, temporary_identity):
                _unlink_owned_file(target, temporary_identity)
        raise
    finally:
        for temporary, _target, temporary_identity in temporary_identities:
            _unlink_owned_file(temporary, temporary_identity)


def _immutable_array(value: np.ndarray) -> np.ndarray:
    """Copy through immutable bytes so writeability cannot be re-enabled."""

    raw_bytes = value.tobytes(order="C")
    return np.frombuffer(raw_bytes, dtype=value.dtype, count=value.size).reshape(
        value.shape, order="C"
    )


def _metadata_band_levels(metadata: Mapping[str, Any], key: str) -> int:
    record = metadata.get(key)
    if not isinstance(record, Mapping):
        raise TypeError(f"metadata.{key} must be a mapping")
    levels = record.get("levels")
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1:
        raise ValueError(f"metadata.{key}.levels must be a positive integer")
    return levels


def _validate_climate_metadata(metadata: Mapping[str, Any]) -> None:
    record = metadata.get("climate")
    if not isinstance(record, Mapping):
        raise TypeError("metadata.climate must be a mapping")
    if tuple(record.get("seasons", ())) != _SEASON_IDS:
        raise ValueError(
            "metadata.climate.seasons must be vernal, june, autumnal, december"
        )
    try:
        solar_longitudes = tuple(
            float(value) for value in record.get("solarLongitudeDegrees", ())
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "metadata.climate.solarLongitudeDegrees must be numeric"
        ) from error
    if solar_longitudes != _SEASON_SOLAR_LONGITUDES:
        raise ValueError(
            "metadata.climate.solarLongitudeDegrees must be 0, 90, 180, 270"
        )
    for key in (
        "precipitationQuantization",
        "windQuantization",
        "riverQuantization",
        "model",
    ):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata.climate.{key} must be a non-empty string")


@dataclass(frozen=True)
class WorldGrid:
    """Immutable canonical arrays and metadata for one sampled planet grid.

    Water codes are ``0`` for land, ``1`` for ocean, ``2`` for lake, and
    ``3`` for an inland sea.
    ``flow_to`` uses a flattened C-order cell index and ``-1`` for no
    downstream cell.  ``bathymetry_band == -1`` denotes land or lake; ocean
    and inland-sea cells use a valid metadata-bounded ordinal.
    """

    elevation: np.ndarray
    elevation_band: np.ndarray
    water: np.ndarray
    bathymetry_band: np.ndarray
    flow_to: np.ndarray
    discharge: np.ndarray
    river_order: np.ndarray
    snow: np.ndarray
    sea_ice: np.ndarray
    seasonal_precipitation: np.ndarray
    seasonal_wind_east: np.ndarray
    seasonal_wind_north: np.ndarray
    seasonal_river_strength: np.ndarray
    metadata: Mapping[str, Any]

    _ARRAY_NAMES: ClassVar[tuple[str, ...]] = _ARRAY_NAMES

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name in self._ARRAY_NAMES:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray")
            expected_dtype = _ARRAY_DTYPES[name]
            if value.dtype != expected_dtype:
                raise TypeError(
                    f"{name} must have dtype {expected_dtype}, got {value.dtype}"
                )
            expected_dimensions = 3 if name in _SEASONAL_ARRAY_NAMES else 2
            if value.ndim != expected_dimensions:
                dimension_text = "three" if expected_dimensions == 3 else "two"
                raise ValueError(f"{name} must be a {dimension_text}-dimensional array")
            immutable_value = _immutable_array(value)
            object.__setattr__(self, name, immutable_value)
            arrays[name] = immutable_value

        shape = arrays[_SPATIAL_ARRAY_NAMES[0]].shape
        if any(arrays[name].shape != shape for name in _SPATIAL_ARRAY_NAMES[1:]):
            raise ValueError("all spatial WorldGrid arrays must have the same shape")
        seasonal_shape = (_SEASON_COUNT, *shape)
        for name in _SEASONAL_ARRAY_NAMES:
            if arrays[name].shape != seasonal_shape:
                raise ValueError(f"{name} must have shape {seasonal_shape}")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        metadata = _freeze_json(self.metadata)
        missing = _REQUIRED_METADATA.difference(metadata)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"metadata is missing required keys: {missing_text}")
        fingerprint = metadata["inputFingerprint"]
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("inputFingerprint must be a non-empty string")
        object.__setattr__(self, "metadata", metadata)
        _validate_climate_metadata(metadata)

        water = self.water
        if np.any((water < 0) | (water > 3)):
            raise ValueError(
                "water codes must be 0 (land), 1 (ocean), 2 (lake), or 3 (inland sea)"
            )

        if np.any(self.bathymetry_band < -1):
            raise ValueError("bathymetry_band must use -1 or a non-negative band")

        elevation_levels = _metadata_band_levels(metadata, "elevation")
        if np.any(self.elevation_band >= elevation_levels):
            raise ValueError(
                "elevation_band contains values outside metadata.elevation.levels"
            )

        bathymetry_levels = _metadata_band_levels(metadata, "bathymetry")
        maritime_water = (water == 1) | (water == 3)
        if np.any(maritime_water & (self.bathymetry_band < 0)):
            raise ValueError(
                "ocean and inland-sea bathymetry_band must be a non-negative band"
            )
        if np.any((~maritime_water) & (self.bathymetry_band != -1)):
            raise ValueError("land and lake bathymetry_band must be -1")
        if np.any(maritime_water & (self.bathymetry_band >= bathymetry_levels)):
            raise ValueError(
                "bathymetry_band contains values outside metadata.bathymetry.levels"
            )

        flow_to = self.flow_to
        cell_count = int(np.prod(shape, dtype=np.int64))
        if np.any((flow_to < -1) | (flow_to >= cell_count)):
            raise ValueError("flow_to must contain -1 or an in-range flattened index")

        if np.any((water == 1) & (self.river_order > 0)):
            raise ValueError("river cells cannot be located on ocean cells")
        if np.any(self.sea_ice & (water != 1)):
            raise ValueError("sea_ice cells must be located on open-ocean cells")

        if not np.isfinite(self.elevation).all():
            raise ValueError("elevation values must be finite")
        if not np.isfinite(self.discharge).all():
            raise ValueError("discharge values must be finite")
        if np.any(self.discharge < 0):
            raise ValueError("discharge values must be non-negative")

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    def content_digest(self) -> str:
        """Return a deterministic digest of metadata and all canonical arrays."""

        digest = sha256()
        metadata_bytes = json.dumps(
            _thaw_json(self.metadata),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(b"metadata\0")
        digest.update(metadata_bytes)
        for name in self._ARRAY_NAMES:
            value = np.ascontiguousarray(getattr(self, name))
            digest.update(name.encode("ascii"))
            digest.update(b"\0")
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(value.shape).encode("ascii"))
            digest.update(b"\0")
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    def save(self, output_directory: str | Path) -> None:
        """Write the two-file WorldGrid bundle without pickle serialization."""

        output_directory = _validate_save_paths(Path(output_directory))
        output_directory.mkdir(parents=True, exist_ok=True)
        npz_path = output_directory / "world-grid.npz"
        metadata_path = output_directory / "world-grid.json"
        temporary_npz: Path | None = None
        temporary_json: Path | None = None
        try:
            temporary_npz = _temporary_file(output_directory, npz_path.name, ".npz")
            temporary_json = _temporary_file(
                output_directory, metadata_path.name, ".json"
            )
            np.savez_compressed(
                temporary_npz,
                **{name: getattr(self, name) for name in self._ARRAY_NAMES},
            )
            metadata_document = _thaw_json(self.metadata)
            metadata_document[_BUNDLE_FIELD] = {
                _CONTENT_DIGEST_FIELD: self.content_digest()
            }
            with temporary_json.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    metadata_document,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
            _atomic_replace_files(
                ((temporary_npz, npz_path), (temporary_json, metadata_path))
            )
        finally:
            for temporary in (temporary_npz, temporary_json):
                if temporary is not None and os.path.lexists(temporary):
                    identity = _file_identity(temporary)
                    if identity is not None:
                        _unlink_owned_file(temporary, identity)

    @classmethod
    def load(cls, output_directory: str | Path) -> "WorldGrid":
        """Load and validate a two-file WorldGrid bundle."""

        output_directory = Path(output_directory)
        npz_path = output_directory / "world-grid.npz"
        metadata_path = output_directory / "world-grid.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            raise ValueError("world-grid.json must contain an object")
        if document.get("format") != _WORLD_GRID_FORMAT:
            raise ValueError("world-grid.json format is unsupported")
        if document.get("schema") != _WORLD_GRID_SCHEMA:
            raise ValueError("world-grid.json schema is unsupported")
        bundle = document.get(_BUNDLE_FIELD)
        if not isinstance(bundle, dict):
            raise ValueError("world-grid.json bundle is missing")
        expected_digest = bundle.get(_CONTENT_DIGEST_FIELD)
        if not isinstance(expected_digest, str) or not expected_digest.strip():
            raise ValueError("world-grid.json content digest is missing")
        metadata = dict(document)
        metadata.pop(_BUNDLE_FIELD)
        with np.load(npz_path, allow_pickle=False) as bundle:
            missing = set(cls._ARRAY_NAMES).difference(bundle.files)
            unexpected = set(bundle.files).difference(cls._ARRAY_NAMES)
            if missing or unexpected:
                raise ValueError(
                    f"world-grid.npz fields mismatch; missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected)}"
                )
            arrays = {name: bundle[name] for name in cls._ARRAY_NAMES}
        grid = cls(metadata=metadata, **arrays)
        if grid.content_digest() != expected_digest:
            raise ValueError("world-grid content digest mismatch")
        return grid
