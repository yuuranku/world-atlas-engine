"""Persist the rendered map independently of its browser interface."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET


def write_interface_snapshot(
    directory: Path,
    overlay: str,
    *,
    grid_digest: str,
    society_digest: str,
    width: int,
    height: int,
    viewbox_width: int,
    viewbox_height: int,
    tectonic_diagnostics: Mapping[str, object],
    territorial_qa: Mapping[str, int | float],
    settlement_locations: Mapping[str, tuple[float, float]],
) -> None:
    # The browser overlay uses HTML boolean attributes. A standalone SVG
    # needs their explicit XML spelling; geometry and attribute values stay
    # untouched when the same scene is reused by an interface rebuild.
    overlay = re.sub(
        r"<[^>]+>",
        lambda match: re.sub(r"\s(hidden)(?=\s|/?>)", r' \1=""', match[0]),
        overlay,
    )
    scene = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {viewbox_width} {viewbox_height}">'
        + overlay + '</svg>'
    )
    encoded = scene.encode("utf-8")
    ET.fromstring(scene)
    presentation = {
        "contract": "world-atlas-interface-1",
        "gridDigest": grid_digest,
        "societyDigest": society_digest,
        "sceneSha256": hashlib.sha256(encoded).hexdigest(),
        "width": width,
        "height": height,
        "viewboxWidth": viewbox_width,
        "viewboxHeight": viewbox_height,
        "tectonicDiagnostics": dict(tectonic_diagnostics),
        "territorialQa": dict(territorial_qa),
        "settlementLocations": dict(settlement_locations),
    }
    (directory / "atlas-scene.svg").write_bytes(encoded)
    if 'id="overview-source"' in overlay:
        from .svg_groups import extract_group
        source=extract_group(overlay,'id','overview-source')
        (directory/'overview-source.svg').write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {viewbox_width} {viewbox_height}">{source}</svg>',encoding='utf-8')
    (directory / "interface-presentation.json").write_text(
        json.dumps(presentation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_interface_snapshot(
    directory: Path, *, grid_digest: str, society_digest: str
) -> tuple[str, dict[str, Any]]:
    presentation = json.loads(
        (directory / "interface-presentation.json").read_text(encoding="utf-8")
    )
    if presentation["contract"] != "world-atlas-interface-1":
        raise ValueError("unsupported map interface snapshot")
    if (presentation["gridDigest"], presentation["societyDigest"]) != (
        grid_digest, society_digest
    ):
        raise ValueError("map interface snapshot belongs to a different world")
    encoded = (directory / "atlas-scene.svg").read_bytes()
    if hashlib.sha256(encoded).hexdigest() != presentation["sceneSha256"]:
        raise ValueError("map interface scene content changed")
    scene = encoded.decode("utf-8")
    root = ET.fromstring(scene)
    expected_viewbox = (
        f'0 0 {presentation["viewboxWidth"]} {presentation["viewboxHeight"]}'
    )
    if root.tag != "{http://www.w3.org/2000/svg}svg" or root.get("viewBox") != expected_viewbox:
        raise ValueError("map interface scene dimensions do not match")
    # The persisted scene is the direct SVG writer's output. Keep its exact
    # geometry and attributes when rebuilding only the surrounding interface.
    return scene[scene.index(">") + 1:scene.rindex("</svg>")], presentation
