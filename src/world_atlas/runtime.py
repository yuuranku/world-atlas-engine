"""Explicit native/Node rendering dependency checks, before long simulation."""
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

DEPENDENCIES = {"numpy": "2.3.5", "contourpy": "1.3.3", "Pillow": "12.3.0", "shapely": "2.1.2"}


def require_renderer(mapshaper: str | Path | None = None) -> tuple[str, Path]:
    entry = mapshaper if mapshaper is not None else os.environ.get("WORLD_ATLAS_MAPSHAPER")
    if not entry:
        raise ValueError("Mapshaper 0.7.56 required: pass --mapshaper <runtime/node_modules/mapshaper/bin/mapshaper>")
    path = Path(entry).resolve()
    if not path.is_file():
        raise ValueError(f"Mapshaper entry does not exist: {path}")
    package = path.parent.parent / "package.json"
    if not package.is_file():
        raise ValueError(f"Mapshaper package.json not found beside entry: {path}")
    metadata = json.loads(package.read_text(encoding="utf-8"))
    if metadata.get("name") != "mapshaper" or metadata.get("version") != "0.7.56":
        raise ValueError("Mapshaper must be the locked version 0.7.56")
    node = shutil.which("node")
    if not node:
        raise ValueError("Node.js is required for the Mapshaper renderer")
    return node, path


def doctor(mapshaper: str | Path | None = None) -> dict:
    dependencies = {}
    for name, expected in DEPENDENCIES.items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = None
        dependencies[name] = {"required": expected, "installed": actual, "ok": actual == expected}
    try:
        node, entry = require_renderer(mapshaper)
        check = subprocess.run([node, str(entry), "-v"], text=True, capture_output=True, timeout=30, check=True)
        renderer = {"ok": True, "entry": str(entry), "version": check.stdout.strip()}
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        renderer = {"ok": False, "error": str(error)}
    python_ok = sys.version_info[:2] == (3, 14)
    return {"ok": python_ok and renderer["ok"] and all(d["ok"] for d in dependencies.values()),
            "python": sys.version.split()[0], "pythonSupported": python_ok, "dependencies": dependencies,
            "renderer": renderer, "requiresAI": False}
