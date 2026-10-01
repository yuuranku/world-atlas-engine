"""Publish a terrain-only review from the canonical continuous surface bundle."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np

from world_atlas.core.procedural_planet import load_surface_bundle
from world_atlas.core.continuous_terrain import PhysicalTerrainField
from world_atlas.core.implicit_terrain import terrain_level_curves
from world_atlas.core.svg_paths import COORDINATE_SCALE, integer_subpath_data


def publish_terrain_review(
    source_directory: Path,
    output_directory: Path,
    *,
    version: str,
) -> dict[str, object]:
    source = source_directory.resolve()
    output = output_directory.resolve()
    surface = load_surface_bundle(source / "physical-fields.npz")
    height, width = surface.land_mask.shape
    output.mkdir(parents=True, exist_ok=True)
    image_source = source / "physical-reference.procedural.png"
    image_target = output / "terrain.png"
    shutil.copy2(image_source, image_target)
    shutil.copy2(source / "provenance.json", output / "provenance.json")

    diagnostics = surface.diagnostics
    terrain = PhysicalTerrainField(surface.relative_elevation_m,
        land_mask=surface.land_mask, sea_level_m=diagnostics["seaLevelMeters"],
        elevation_scale_m=diagnostics["elevationScaleMeters"],
        elevation_exponent=diagnostics["elevationExponent"])
    palette_levels = np.linspace(0.06, 0.96, 19)
    metre_levels = terrain.contour_height_m(palette_levels)
    curves = terrain_level_curves(terrain, metre_levels)
    paths = []
    for palette, metres, branches in zip(palette_levels, metre_levels, curves, strict=True):
        for points in branches:
            delivered = np.rint(points*COORDINATE_SCALE).astype(np.int64)
            keep = np.r_[True, np.any(delivered[1:] != delivered[:-1], axis=1)]
            data = integer_subpath_data(map(tuple, delivered[keep]),
                closed=bool(np.array_equal(points[0], points[-1])))
            if data:
                paths.append(f'<path data-elevation-level="{palette:.17g}" '
                    f'data-elevation-m="{metres:.17g}" d="{data}"/>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<g fill="none" stroke="#43544b" stroke-opacity=".21" '
        'stroke-width=".45" stroke-linecap="round" stroke-linejoin="round">'
        + "".join(paths) + '</g></svg>'
    )
    (output / "contours.svg").write_text(svg, encoding="utf-8")
    title = html.escape(f"架空世界地形图 {version}")
    image_digest = hashlib.sha256(image_target.read_bytes()).hexdigest()
    document = _VIEWER.replace("__TITLE__", title).replace("__WIDTH__", str(width))
    document = document.replace("__HEIGHT__", str(height)).replace("__HASH__", image_digest[:16])
    document = document.replace("__SEED__", str(surface.diagnostics["seed"]))
    (output / "index.html").write_text(document, encoding="utf-8")
    record = {
        "version": version,
        "width": width,
        "height": height,
        "sourceSha256": image_digest,
        "contourCount": len(paths),
        "contourModel": "source-tensor-pchip-native-roots-adaptive-curves",
        "contourBytes": len(svg.encode("utf-8")),
        "review": str(output / "index.html"),
    }
    (output / "review.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return record


_VIEWER = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;font-family:"Microsoft YaHei UI","Noto Sans CJK SC",sans-serif}
*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden;background:#162235}
#viewport{position:absolute;inset:0;overflow:hidden;cursor:grab;touch-action:none;user-select:none}
#viewport.dragging{cursor:grabbing}
#scene{position:absolute;left:0;top:0;width:__WIDTH__px;height:__HEIGHT__px;transform-origin:0 0;will-change:transform;pointer-events:none}
#scene img{position:absolute;inset:0;width:100%;height:100%;max-width:none}
#contours{opacity:0;transition:opacity .15s}
.bar{position:fixed;z-index:2;top:18px;left:18px;display:flex;align-items:center;gap:8px;padding:8px;border:1px solid #ffffff26;border-radius:10px;background:#111d2dec;box-shadow:0 8px 30px #00000030}
.title{padding:0 10px 0 6px;font-size:14px;font-weight:700;color:#eef4fb}
button{min-width:40px;height:34px;padding:0 12px;border:1px solid #ffffff24;border-radius:7px;color:#e9f1f8;background:#ffffff14;font:inherit;font-size:13px;cursor:pointer}
button:hover,button[aria-pressed=true]{background:#ffffff2b}button:focus-visible{outline:2px solid #b9d6ed;outline-offset:2px}
#scale{min-width:54px;text-align:center;color:#bfcddd;font-size:13px;font-variant-numeric:tabular-nums}
.hint{position:fixed;z-index:2;right:18px;bottom:18px;padding:8px 11px;border-radius:7px;color:#bfcddd;background:#111d2dd9;font-size:12px;pointer-events:none}
#error{position:fixed;inset:40% 20%;text-align:center;z-index:3;color:#f4d6c9;background:#162235;padding:25px;border:1px solid #f4d6c9;border-radius:10px}
@media(max-width:720px){.bar{top:8px;left:8px;right:8px;gap:4px;flex-wrap:wrap}.title{width:100%;padding:3px 5px}.hint{right:8px;bottom:8px;font-size:11px}}
@media(prefers-reduced-motion:reduce){#contours{transition:none}}
</style>
</head>
<body>
<main id="viewport" aria-label="可缩放的架空世界地形图">
 <div id="scene">
  <img id="map" src="terrain.png?v=__HASH__" width="__WIDTH__" height="__HEIGHT__" alt="__TITLE__">
  <img id="contours" src="contours.svg?v=__HASH__" width="__WIDTH__" height="__HEIGHT__" alt="" aria-hidden="true">
 </div>
</main>
<nav class="bar" aria-label="地图控制">
 <span class="title">__TITLE__</span>
 <button id="minus" type="button" aria-label="缩小">−</button>
 <span id="scale" aria-live="off">100%</span>
 <button id="plus" type="button" aria-label="放大">＋</button>
 <button id="fit" type="button">全图</button>
 <button id="actual" type="button">原大</button>
 <button id="toggle-contours" type="button" aria-pressed="true">等高线</button>
</nav>
<div class="hint">种子 __SEED__ · 滚轮缩放 · 按住拖动</div>
<div id="error" role="alert" hidden>地图文件未能载入，请重新打开完整的 review 文件夹中的 index.html。</div>
<script>
const viewport=document.querySelector('#viewport'),scene=document.querySelector('#scene');
const map=document.querySelector('#map'),contours=document.querySelector('#contours');
const scaleLabel=document.querySelector('#scale'),contourButton=document.querySelector('#toggle-contours');
const imageWidth=__WIDTH__,imageHeight=__HEIGHT__;
let scale=1,x=0,y=0,drag=null,frame=0,showContours=true;
const fitScale=()=>Math.min(innerWidth/imageWidth,innerHeight/imageHeight);
function render(){frame=0;scene.style.transform=`translate3d(${x}px,${y}px,0) scale(${scale})`;scaleLabel.textContent=`${Math.round(scale*100)}%`;contours.style.opacity=showContours?Math.min(1,Math.max(0,(scale-.45)/.75)):0}
function schedule(){if(!frame)frame=requestAnimationFrame(render)}
function clamp(){const w=imageWidth*scale,h=imageHeight*scale;x=w<=innerWidth?(innerWidth-w)/2:Math.min(0,Math.max(innerWidth-w,x));y=h<=innerHeight?(innerHeight-h)/2:Math.min(0,Math.max(innerHeight-h,y))}
function fit(){scale=fitScale();x=(innerWidth-imageWidth*scale)/2;y=(innerHeight-imageHeight*scale)/2;schedule()}
function zoomAt(next,px=innerWidth/2,py=innerHeight/2){next=Math.min(6,Math.max(fitScale()*.8,next));const imageX=(px-x)/scale,imageY=(py-y)/scale;scale=next;x=px-imageX*scale;y=py-imageY*scale;clamp();schedule()}
viewport.addEventListener('wheel',event=>{event.preventDefault();zoomAt(scale*Math.exp(-event.deltaY*.0012),event.clientX,event.clientY)},{passive:false});
viewport.addEventListener('pointerdown',event=>{if(event.button!==0)return;viewport.setPointerCapture(event.pointerId);viewport.classList.add('dragging');drag={x:event.clientX,y:event.clientY,ox:x,oy:y}});
viewport.addEventListener('pointermove',event=>{if(!drag)return;x=drag.ox+event.clientX-drag.x;y=drag.oy+event.clientY-drag.y;clamp();schedule()});
function endDrag(){drag=null;viewport.classList.remove('dragging')}
for(const event of ['pointerup','pointercancel','lostpointercapture'])viewport.addEventListener(event,endDrag);
document.querySelector('#minus').addEventListener('click',()=>zoomAt(scale/1.3));
document.querySelector('#plus').addEventListener('click',()=>zoomAt(scale*1.3));
document.querySelector('#fit').addEventListener('click',fit);
document.querySelector('#actual').addEventListener('click',()=>zoomAt(1));
contourButton.addEventListener('click',()=>{showContours=!showContours;contourButton.setAttribute('aria-pressed',String(showContours));schedule()});
addEventListener('resize',()=>{clamp();schedule()});
map.addEventListener('error',()=>{document.querySelector('#error').hidden=false});
fit();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(json.dumps(publish_terrain_review(
        args.source_directory, args.output_directory, version=args.version,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
