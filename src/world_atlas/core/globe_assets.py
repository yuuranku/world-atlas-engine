"""Build globe paint documents owned by the one shared physical surface."""
from __future__ import annotations

from collections.abc import Mapping
import math
import xml.etree.ElementTree as ET


_SVG = "http://www.w3.org/2000/svg"
_LAND_CLIP = "land-silhouette-clip"
_LAND_REFERENCE = f"url(#{_LAND_CLIP})"
THEME_EXPORT_FILENAMES = {
    "climate": "climate.svg", "biome": "biome.svg",
    "watershed": "watersheds.svg", "potential": "land-potential.svg",
    "habitability": "habitability.svg", "vegetation": "vegetation-cover.svg",
    "population": "population.svg", "civilizations": "civilizations.svg",
    "languages": "languages.svg", "religions": "religions.svg",
    "political": "political.svg", "provinces": "provinces.svg",
}
ET.register_namespace("", _SVG)


def _document(source: str):
    try:
        root = ET.fromstring(source)
        box = tuple(float(value) for value in root.attrib["viewBox"].split())
    except (ET.ParseError, KeyError, ValueError) as error:
        raise ValueError("globe paint requires current full-world SVG documents") from error
    if (root.tag != f"{{{_SVG}}}svg" or len(box) != 4
            or not all(math.isfinite(value) for value in box)
            or box[:2] != (0.0, 0.0) or box[3] <= 0 or box[2] != 2 * box[3]):
        raise ValueError("globe paint requires the full-world 2:1 viewBox")
    return root, box


def _clip_data(clip):
    if clip.attrib != {"id": _LAND_CLIP, "clipPathUnits": "userSpaceOnUse"}:
        raise ValueError("globe paint requires the authoritative user-space land clip")
    if not len(clip) or any(
        node.tag != f"{{{_SVG}}}path" or node.attrib.get("clip-rule") != "evenodd"
        or not node.attrib.get("d") for node in clip
    ):
        raise ValueError("globe paint requires the complete even-odd land paths")
    return " ".join(node.attrib["d"] for node in clip)


def globe_theme_documents(
    base_surface: str,
    standalone_themes: Mapping[str, str],
    *,
    boundary_bodies: Mapping[str, str],
) -> dict[str, str]:
    """Repackage current standalone paint without copying its clip definition.

    Standalone exports own a physical clip. Globe overlays consume that exact
    clip from ``base_surface`` instead. All paint nodes, coordinates, styles and
    shared junctions are copied unchanged; this operation draws no new geometry.
    """
    if set(standalone_themes) != set(THEME_EXPORT_FILENAMES):
        raise ValueError("the globe requires all twelve current standalone themes")
    if set(boundary_bodies) != {"political", "provinces"}:
        raise ValueError("globe administrative ink must declare both thematic owners")
    base, box = _document(base_surface)
    clips = base.findall(f".//{{{_SVG}}}clipPath[@id='{_LAND_CLIP}']")
    if len(clips) != 1:
        raise ValueError("the globe base must own exactly one physical land clip")
    clip_data = _clip_data(clips[0])
    result = {}
    for theme in ("terrain", *THEME_EXPORT_FILENAMES):
        document = ET.Element(f"{{{_SVG}}}svg", {
            "viewBox": base.attrib["viewBox"], "preserveAspectRatio": "none",
            "role": "img", "data-surface-contract": "globe-owned-land-clip",
        })
        ET.SubElement(document, f"{{{_SVG}}}title").text = f"{theme} globe surface"
        if theme != "terrain":
            source, source_box = _document(standalone_themes[theme])
            if (source_box != box
                    or source.attrib.get("data-surface-contract") != "shared-land-clip"
                    or len(source) != 2
                    or source[0].tag != f"{{{_SVG}}}defs"
                    or len(source[0]) != 1
                    or source[0][0].tag != f"{{{_SVG}}}clipPath"
                    or _clip_data(source[0][0]) != clip_data):
                raise ValueError("standalone globe paint must use the exact base physical clip")
            paint = source[1]
            if (paint.tag != f"{{{_SVG}}}g"
                    or paint.attrib != {"clip-path": _LAND_REFERENCE}
                    or len(paint.findall(f".//{{{_SVG}}}g[@data-partition='mutually-exclusive']")) != 1
                    or paint.findall(f".//{{{_SVG}}}clipPath")
                    or paint.findall(f".//{{{_SVG}}}svg")):
                raise ValueError("globe paint requires the current owned partition group")
            document.append(paint)
        if theme in boundary_bodies:
            boundary = ET.fromstring(f'<svg xmlns="{_SVG}">{boundary_bodies[theme]}</svg>')
            if any(node.tag not in {f"{{{_SVG}}}g", f"{{{_SVG}}}path"}
                   for child in boundary for node in child.iter()):
                raise ValueError("globe administrative ink must contain only source vector ink")
            document.extend(boundary)
        result[theme] = ET.tostring(document, encoding="unicode")
    return result
