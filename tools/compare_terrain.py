"""Compare generated physical fields after aligning their ocean-map seam."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from world_atlas.core.procedural_planet import load_surface_bundle


def compare(before: Path, after: Path) -> dict:
    old = load_surface_bundle(before / 'source/physical-fields.npz')
    new = load_surface_bundle(after / 'source/physical-fields.npz')
    before_recipe = json.loads((before/'source/provenance.json').read_text(encoding='utf-8'))['recipe']
    after_recipe = json.loads((after/'source/provenance.json').read_text(encoding='utf-8'))['recipe']
    shift = old.diagnostics['mapSeam']['longitudeRollColumns'] - new.diagnostics['mapSeam']['longitudeRollColumns']
    mask = np.roll(new.land_mask, shift, axis=1)
    weight = np.cos(np.radians(90 - (np.arange(mask.shape[0]) + .5) * 180 / mask.shape[0]))[:, None]
    weighted = lambda field: float(np.sum(field * weight))
    checks = {
        'sameSeed': old.diagnostics['seed'] == new.diagnostics['seed'],
        'sameRecipe': before_recipe == after_recipe,
        'samePlanetSettings': json.loads((before/'worldgen.json').read_text(encoding='utf-8'))['planet'] == json.loads((after/'worldgen.json').read_text(encoding='utf-8'))['planet'],
        'sameShape': old.land_mask.shape == new.land_mask.shape,
        'samePlateOwnership': bool(np.array_equal(old.plate_id, np.roll(new.plate_id, shift, axis=1))),
        'samePlateVelocity': all(np.array_equal(getattr(old,field), np.roll(getattr(new,field), shift, axis=1)) for field in ('plate_velocity_east_cm_per_year','plate_velocity_north_cm_per_year')),
        'samePlateBoundaries': bool(np.array_equal(old.boundary_class, np.roll(new.boundary_class, shift, axis=1))),
        'oceanAtBothEdges': not bool(new.land_mask[:, [0,-1]].any()),
        'finiteElevation': bool(np.isfinite(new.elevation).all()),
        'finiteBathymetry': bool(np.isfinite(new.bathymetry).all()),
        'oneSurface': bool(np.array_equal(new.signed_height > 0, new.land_mask)),
        'dryElevationOnly': bool(np.all(new.elevation[~new.land_mask] == 0)),
        'wetDepthOnly': bool(np.all(new.bathymetry[new.land_mask] == 0)),
        'landFractionWithinTolerance': abs(new.diagnostics['landFraction'] - before_recipe['land_fraction']) < .002,
    }
    report = {
        'before': str(before), 'after': str(after), 'checks': checks,
        'ok': all(checks.values()), 'alignmentRollColumns': shift,
        'sameRecipe': checks['sameRecipe'],
        'landIntersectionOverUnion': weighted(mask & old.land_mask) / weighted(mask | old.land_mask),
        'changedSurfaceFraction': weighted(mask != old.land_mask) / (weighted(np.ones(mask.shape, dtype=bool))),
        'beforeScaleProfile': old.diagnostics['coastlineScaleProfile'],
        'afterScaleProfile': new.diagnostics['coastlineScaleProfile'],
        'beforeLandFraction': old.diagnostics['landFraction'],
        'afterLandFraction': new.diagnostics['landFraction'],
        'afterCoastalEvolution': new.diagnostics['coastalEvolution'],
        'beforeSourceSha256': hashlib.sha256((before/'source/physical-fields.npz').read_bytes()).hexdigest(),
        'afterSourceSha256': hashlib.sha256((after/'source/physical-fields.npz').read_bytes()).hexdigest(),
    }
    (after / 'coastline-comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    # Derived from physical masks, not an edited or AI-generated picture.
    for name, field in [('before',old.land_mask),('after',mask)]:
        pixels = np.where(field[...,None], np.array([47,66,75],dtype=np.uint8),
                          np.array([239,244,242],dtype=np.uint8))
        Image.fromarray(pixels).save(after / 'review' / f'coast-{name}.png')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>同种子海岸轮廓对照</title>
<style>body{margin:0;padding:20px;background:#e6ece9;font:16px "Microsoft YaHei UI",sans-serif;color:#243840}h1{font-size:22px}section{max-width:1440px;margin:16px auto}img{width:100%;display:block;border:1px solid #b5c5c5}h2{font-size:16px}p{font-size:13px}</style>
<section><h1>同一种子 · 只看海陆轮廓</h1><p>上下图已对齐中央经线；深色为陆地，浅色为海洋。</p><h2>修改前</h2><img src="coast-before.png"><h2>修改后</h2><img src="coast-after.png"><p><a href="index.html">返回地形图</a></p></section></html>'''
    (after/'review/comparison.html').write_text(page,encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    args = parser.parse_args()
    report = compare(args.before, args.after)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['ok'] else 1)
