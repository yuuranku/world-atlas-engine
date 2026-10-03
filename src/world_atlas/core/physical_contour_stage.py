"""Persist continuous-root cartographic height graphs before map artwork."""
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import scipy
import shapely
import contourpy

from .cartographic_contours import CARTOGRAPHIC_CONTOUR_CONTRACT, cartographic_curve_batches


_SCHEMA = "physical-cartographic-height-graphs-v2"
_CURVE_MODULES = (
    "continuous_terrain", "continuous_scalar", "continuous_pchip", "hypsometry", "terrain_refinement", "river_network",
    "implicit_terrain", "implicit_terrain_bounds", "implicit_curves",
    "implicit_pchip", "implicit_warp_events", "implicit_river_events",
    "implicit_terrain_parallel", "physical_contour_stage",
    "cartographic_contours",
)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_bytes(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json(path, value):
    _atomic_bytes(path, (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())


def _array_binding(value):
    if value is None:
        return None
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def _refinement_binding(field):
    from .terrain_refinement import RefinedTerrainField
    if not isinstance(field, RefinedTerrainField):
        return None
    drainage, warps = field._height, field._landforms
    return {
        "seed": field.seed, "radiusKm": field.radius_km,
        "amplitude": _array_binding(field._amplitude),
        "drainage": {
            "curves": _array_binding(drainage.curves),
            "breadth": _array_binding(drainage.breadth),
            "strength": _array_binding(drainage.strength),
            "profileNative": _array_binding(drainage.pchip.native_m),
            "metric": [drainage.radius_km, drainage.cell_x_km, drainage.cell_y_km,
                       drainage.maximum_breadth, drainage.edge_count],
        },
        "warps": {
            "centres": _array_binding(warps.centres),
            "radius": _array_binding(warps.radius),
            "displacement": _array_binding(warps.displacement),
            "maximumDisplacement": warps.maximum_displacement,
        },
        "riverCoordinates": _array_binding(shapely.get_coordinates(field._rivers.geometries)
                                           if field._rivers is not None else None),
        "riverSupportDistance": _array_binding(field._river_support_distance),
        "anchors": _array_binding(field._anchors.data if field._anchors is not None else None),
    }


def _extraction_runtime_binding():
    root = Path(__file__).parent
    return {
        "sourceModulesSha256": {
            **{name: _sha(root / (name + ".py")) for name in _CURVE_MODULES},
            "physical/hydrology": _sha(root.parent / "physical" / "hydrology.py"),
        },
        "runtime": {"python": list(sys.version_info[:3]), "numpy": np.__version__,
                    "scipy": scipy.__version__, "shapely": shapely.__version__,
                    "geos": shapely.geos_version_string,"contourpy":contourpy.__version__},
    }


def _binding(field, levels, identity):
    if not isinstance(identity, dict) or not identity:
        raise ValueError("source height graphs require an explicit physical input identity")
    return {
        "physicalInput": json.loads(json.dumps(identity)),
        "fieldClass": type(field).__name__,
        "shape": [field.height, field.width],
        "nativeHeightSha256": hashlib.sha256(np.ascontiguousarray(field.native_m).tobytes()).hexdigest(),
        "refinement": _refinement_binding(field),
        "datum": [float(field.sea_level_m), float(field.elevation_scale_m), float(field.elevation_exponent)],
        "heightLevelsHex": [float(level).hex() for level in levels],
        "cartographicContract": dict(CARTOGRAPHIC_CONTOUR_CONTRACT),
        **_extraction_runtime_binding(),
    }


def _write_graph(directory, index, level, fingerprint, paths):
    points = [np.asarray(path, dtype=np.float64) for path in paths]
    if any(path.ndim != 2 or path.shape[1] != 2 or len(path) < 2
           or not np.isfinite(path).all() for path in points):
        raise ValueError("a source height graph must contain finite complete paths")
    offsets = np.r_[0, np.cumsum([len(path) for path in points], dtype=np.int64)]
    flat = np.concatenate(points) if points else np.empty((0, 2), dtype=np.float64)
    path = directory / f"level-{index:03d}.npz"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=path.name,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, level=np.asarray(level, dtype=np.float64), points=flat, offsets=offsets)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    record = {"schema": _SCHEMA, "fingerprint": fingerprint, "index": index,
              "heightMHex": float(level).hex(), "sha256": _sha(path),
              "paths": len(points), "vertices": len(flat)}
    _json(directory / f"level-{index:03d}.json", record)
    return record


def _read_graph(directory, index, level, fingerprint):
    header = directory / f"level-{index:03d}.json"
    record = json.loads(header.read_text(encoding="utf-8"))
    if (set(record) != {"schema", "fingerprint", "index", "heightMHex", "sha256", "paths", "vertices"}
            or record["schema"] != _SCHEMA or record["fingerprint"] != fingerprint
            or record["index"] != index or record["heightMHex"] != float(level).hex()):
        raise ValueError("source height graph does not match this exact physical extraction")
    path = directory / f"level-{index:03d}.npz"
    if _sha(path) != record["sha256"]:
        raise ValueError("saved source height graph bytes changed")
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != {"level", "points", "offsets"}:
            raise ValueError("source height graph requires its current array schema")
        stored_level, points, offsets = bundle["level"], bundle["points"], bundle["offsets"]
    if (stored_level.dtype != np.dtype("float64") or stored_level.shape != ()
            or float(stored_level).hex() != float(level).hex()
            or points.dtype != np.dtype("float64") or points.ndim != 2 or points.shape[1] != 2
            or not np.isfinite(points).all() or offsets.dtype != np.dtype("int64")
            or offsets.ndim != 1 or not len(offsets) or offsets[0] != 0
            or offsets[-1] != len(points) or np.any(np.diff(offsets) < 2)
            or len(offsets) - 1 != record["paths"] or len(points) != record["vertices"]):
        raise ValueError("saved source height graph has invalid coordinates or path boundaries")
    return [points[begin:end] for begin, end in zip(offsets[:-1], offsets[1:], strict=True)]


def current_height_graphs(directory):
    """Check reusable bytes/code before copying; the field binding follows.

    This never rewrites an extraction identity. A new output must still enter
    ``staged_height_curves`` with its actual field and physical source identity.
    """
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        binding = manifest["binding"]
        current = _extraction_runtime_binding()
        if (manifest["schema"] != _SCHEMA
                or binding["cartographicContract"] != CARTOGRAPHIC_CONTOUR_CONTRACT
                or any(binding[key] != value for key, value in current.items())):
            return False
        fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if manifest["fingerprint"] != fingerprint:
            return False
        levels = np.asarray([float.fromhex(value) for value in binding["heightLevelsHex"]])
        if not np.isfinite(levels).all() or np.any(np.diff(levels) <= 0):
            return False
        completed = 0
        for index, level in enumerate(levels):
            if (directory / f"level-{index:03d}.json").exists():
                _read_graph(directory, index, level, fingerprint)
                completed += 1
        return completed > 0
    except (OSError, ValueError, TypeError, KeyError):
        return False


def staged_height_curves(field, levels, directory, *, source_identity):
    """Resume only cartographic graphs bound to identical inputs and contract.

    A changed extraction is rejected. There is no older-format reader or
    substituted geometry. Failed tasks never commit a complete graph header.
    """
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or not np.isfinite(levels).all() or np.any(np.diff(levels) <= 0):
        raise ValueError("source height graphs require increasing finite heights")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binding = _binding(field, levels, source_identity)
    fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = directory / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("schema") != _SCHEMA or previous.get("binding") != binding:
            raise ValueError("obsolete physical contour stage: remove it before extracting the changed source")
    result = [None] * len(levels)
    pending = []
    for index, level in enumerate(levels):
        if (directory / f"level-{index:03d}.json").exists():
            result[index] = _read_graph(directory, index, level, fingerprint)
        else:
            pending.append(index)
    completed = len(levels) - len(pending)
    record = {"schema": _SCHEMA, "binding": binding, "fingerprint": fingerprint,
              "status": "building", "completedHeights": completed, "totalHeights": len(levels)}
    _json(manifest, record)
    logger = logging.getLogger(__name__)
    if pending:
        logger.info("Extracting %s cartographic height graphs from one shared quarter-cell sample; %s source graphs already saved", len(pending), completed)
        current_index = None
        try:
            graphs = iter(cartographic_curve_batches(field, levels[pending]))
            for index in pending:
                current_index = index
                started = time.monotonic()
                paths = next(graphs)
                item = _write_graph(directory, index, levels[index], fingerprint, paths)
                elapsed = time.monotonic()-started
                completed += 1
                logger.info("Height graph complete %s/%s: %.12g m, %s paths, %s vertices, %.1f s",
                            completed, len(levels), levels[index], item["paths"], item["vertices"], elapsed)
                record["completedHeights"] = completed
                _json(manifest, record)
        except BaseException as error:
            record.update(status="failed", failure=repr(error), failureType=type(error).__name__,
                          failedHeightIndex=current_index,
                          failedHeightM=None if current_index is None else float(levels[current_index]))
            _json(manifest, record)
            logger.exception("Cartographic height graph extraction failed at height index %s; complete source graphs remain saved",
                             current_index)
            raise
        for index in pending:
            result[index] = _read_graph(directory, index, levels[index], fingerprint)
    if _binding(field, levels, source_identity) != binding:
        raise ValueError("physical extraction inputs or source changed while height graphs were running")
    record["status"] = "complete"
    _json(manifest, record)
    return result
