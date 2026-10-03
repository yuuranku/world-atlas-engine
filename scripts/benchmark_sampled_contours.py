"""Measure a cartographic sampling option without changing atlas outputs."""
import argparse
import json
from pathlib import Path
import time

from contourpy import contour_generator
import numpy as np

from world_atlas.core.model import WorldGrid
from world_atlas.core.procedural_planet import load_surface_bundle
from world_atlas.core.terrain_refinement import terrain_from_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bounds", type=float, nargs=4, default=(1500.,300.,1756.,556.))
    parser.add_argument("--subdivision", type=int, default=4)
    args = parser.parse_args()
    start = time.perf_counter()
    world = Path(args.world)
    grid = WorldGrid.load(world / "grid")
    source = load_surface_bundle(world / "source/physical-fields.npz")
    field = terrain_from_source(grid, source)
    construction = time.perf_counter()-start
    levels = [float.fromhex(h) for h in json.loads((world / "physical-contours/manifest.json").read_text())["binding"]["heightLevelsHex"]]
    west,north,east,south = args.bounds
    x = np.linspace(west,east,round((east-west)*args.subdivision)+1)
    y = np.linspace(north,south,round((south-north)*args.subdivision)+1)
    values = np.empty((len(y),len(x)))
    start = time.perf_counter()
    for begin in range(0,len(y),32):
        values[begin:begin+32] = field.sample_rect(x,y[begin:begin+32])
    sampling = time.perf_counter()-start
    start = time.perf_counter()
    generator = contour_generator(x=x,y=y,z=values,name="serial",line_type="Separate")
    graphs = [generator.lines(level) for level in levels]
    tracing = time.perf_counter()-start
    errors=[]
    for level,paths in zip(levels,graphs,strict=True):
        points=np.concatenate(paths) if paths else np.empty((0,2))
        if len(points)>2000:points=points[np.linspace(0,len(points)-1,2000,dtype=int)]
        height=field.sample_points(points[:,0],points[:,1]) if len(points) else np.empty(0)
        error=abs(height-level)
        errors.append({"heightM":level,"witnesses":len(points),
                       "maximumResidualM":float(error.max()) if len(error) else 0.,
                       "rmsResidualM":float(np.sqrt(np.mean(error*error))) if len(error) else 0.})
    result={"world":args.world,"bounds":args.bounds,"subdivision":args.subdivision,
            "sampleShape":list(values.shape),"sampleCount":values.size,"sampleBytes":values.nbytes,
            "constructionSeconds":construction,"samplingSeconds":sampling,"tracing22HeightsSeconds":tracing,
            "paths":[len(p) for p in graphs],"vertices":[sum(map(len,p)) for p in graphs],
            "vertexHeightWitnesses":errors,
            "precisionContract":"cartographic sampled isolines; source physical field unchanged; hidden sub-sample peaks are not certified"}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
