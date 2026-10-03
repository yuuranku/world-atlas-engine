"""Measure exact ecology using original lines versus their unchanged segments."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import shapely

from benchmark_numeric_ecology import synthetic_inputs
from world_atlas.core.continuous_ecology import ContinuousEcologyField, FreshwaterCorridors


def original_index(field):
    field._trees=[(shapely.STRtree(parts),radius,parts,np.arange(len(parts)))
                  for parts,radius in field._sources if len(parts)]
    return field


def query_case():
    height,width=512,1024
    paths=[]
    xx=np.linspace(2.,width-2.,300)
    for index in range(32):
        yy=8.+index*15.5+3.*np.sin(xx*.023+index*.9)
        paths.append(shapely.LineString(np.column_stack((xx,yy))))
    sources=FreshwaterCorridors(width,height,(3,),(shapely.MultiLineString(paths),),
                               shapely.box(747.,188.,775.,218.))
    def make():
        return ContinuousEcologyField(np.full((height,width),.1),np.ones((height,width)),
            np.ones((height,width),bool),sources,supply_capacity=.72,
            river_radii=(3.6,),lake_radius=3.4)
    old,new=original_index(make()),make()
    y,x=np.indices((height,width),dtype=float)
    started=time.perf_counter();before=old.supply_points(x+.5,y+.5)
    before_seconds=time.perf_counter()-started
    started=time.perf_counter();after=new.supply_points(x+.5,y+.5)
    after_seconds=time.perf_counter()-started
    max_difference=float(np.max(abs(before-after)))
    byte_identical=np.array_equal(old.native,new.native)
    result={"shape":[height,width],"queryCount":int(x.size),"riverCount":len(paths),
            "verticesPerRiver":len(xx),"originalQuerySeconds":before_seconds,
            "segmentQuerySeconds":after_seconds,"querySpeedup":before_seconds/after_seconds,
            "supplyMaxAbsDifference":max_difference,"nativeByteIdentical":byte_identical,
            "originalIndexPartCount":sum(len(tree.geometries) for tree,*_ in old._trees),
            "segmentIndexPartCount":sum(len(tree.geometries) for tree,*_ in new._trees)}
    print(json.dumps(result),flush=True)
    if max_difference>2e-14 or not byte_identical:raise ValueError("exact supply changed")
    return result


def band_case():
    surface,lake,land,rivers,inputs=synthetic_inputs()
    sources=FreshwaterCorridors.from_surfaces(land.shape,rivers,(1,2,4),lake)
    results=[]
    for index,(background,ceiling,cuts) in enumerate(inputs):
        def make():
            return ContinuousEcologyField(background,ceiling,land,sources,
                supply_capacity=(.72,.81,.78)[index],river_radii=(1.8,2.5,3.6),lake_radius=3.4)
        old,new=original_index(make()),make()
        started=time.perf_counter();before=old.class_regions(cuts,working_surface=surface)
        before_seconds=time.perf_counter()-started
        started=time.perf_counter();after=new.class_regions(cuts,working_surface=surface)
        after_seconds=time.perf_counter()-started
        equal=bool(np.array_equal(shapely.to_wkb(before),shapely.to_wkb(after)))
        native_identical=np.array_equal(old.native,new.native)
        result={"themeIndex":index,"cutCount":len(cuts),"originalExactBandSeconds":before_seconds,
                "segmentExactBandSeconds":after_seconds,"bandSpeedup":before_seconds/after_seconds,
                "nativeByteIdentical":native_identical,"bandsWKBIdentical":equal,
                "nativeSHA256":hashlib.sha256(new.native.tobytes()).hexdigest()}
        print(json.dumps(result),flush=True)
        if not equal or not native_identical:raise ValueError("exact native values or display bands changed")
        results.append(result)
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report={"queryCase":query_case(),"exactBands":band_case(),
            "scope":"524288 native queries on 32 long source reaches; three fixed 256x128 exact ecological drawing fields. Not a full-world timing."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":main()
