"""Draw sampled topology with roots on the unchanged continuous terrain.

The cartographic grid has four intervals per native cell. ContourPy discovers
display branches on those samples; their edge vertices are solved on the
original field, rather than its linear sampled approximation. Features wholly
between samples are not certified. Physical terrain and shore extraction keep
their independent model and precision contracts.
"""
import numpy as np
import shapely
from contourpy import contour_generator
from scipy.optimize import elementwise

from .continuous_terrain import PhysicalTerrainField
from .implicit_terrain import _ROOT_TOLERANCES
from .terrain_refinement import RefinedTerrainField


_SUBDIVISION = 4
CARTOGRAPHIC_CONTOUR_CONTRACT = {
    "schema": "sampled-topology-continuous-roots-v1",
    "subdivisionPerNativeCell": _SUBDIVISION,
    "topology": "ContourPy marching squares on shared quarter-cell terrain samples",
    "vertices": "original continuous terrain roots on the sampled edge brackets",
    "hiddenFeatures": "features wholly between quarter-cell samples are not certified",
    "physicalResolutionChanged": False,
}


def _axes(field, bounds):
    west,north,east,south = bounds
    first = np.floor(np.asarray((west,north))*_SUBDIVISION).astype(int)
    last = np.ceil(np.asarray((east,south))*_SUBDIVISION).astype(int)
    return (np.arange(first[0],last[0]+1,dtype=float)/_SUBDIVISION,
            np.arange(first[1],last[1]+1,dtype=float)/_SUBDIVISION)


def _samplers(field, query_bounds):
    if query_bounds is None:
        x,y = _axes(field,(0,0,field.width,field.height))
        values = np.empty((len(y),len(x)))
        for begin in range(0,len(y),32):
            values[begin:begin+32] = field.sample_rect(x,y[begin:begin+32])
        return [contour_generator(x=x,y=y,z=values,name="serial",line_type="Separate")]
    bounds = np.asarray(query_bounds,dtype=float).reshape(-1,4)
    if (not np.isfinite(bounds).all() or np.any(bounds[:,:2]>=bounds[:,2:])
            or np.any(bounds[:,:2]<0) or np.any(bounds[:,2]>field.width)
            or np.any(bounds[:,3]>field.height)):
        raise ValueError("cartographic terrain queries require ordered rectangles inside the world")
    axes = [_axes(field,bound) for bound in np.unique(bounds,axis=0)]
    if not axes:return []
    sizes = [len(x)*len(y) for x,y in axes]
    coordinates = np.concatenate([np.column_stack((np.tile(x,len(y)),np.repeat(y,len(x)))) for x,y in axes])
    values = np.empty(len(coordinates))
    for begin in range(0,len(values),16384):
        points = coordinates[begin:begin+16384]
        values[begin:begin+16384] = field.sample_points(points[:,0],points[:,1])
    samplers=[];offset=0
    for (x,y),size in zip(axes,sizes,strict=True):
        samplers.append(contour_generator(x=x,y=y,z=values[offset:offset+size].reshape(len(y),len(x)),
                                          name="serial",line_type="Separate"))
        offset+=size
    return samplers


def _true_edge_paths(field, level, paths):
    if not paths:return []
    sizes = [len(path) for path in paths]
    points,inverse = np.unique(np.concatenate(paths),axis=0,return_inverse=True)
    solved = points.copy()
    station=points*_SUBDIVISION
    separation=abs(station-np.rint(station))
    moving_axes=np.argmax(separation,axis=1)
    for axis in (0,1):
        fixed_axis=1-axis
        fixed = station[:,fixed_axis]
        moving = station[:,axis]
        # A corner already is an original sampled value at the cut. Every
        # other ContourPy vertex has exactly one grid-aligned coordinate.
        # ContourPy interpolates both coordinates, so even the fixed edge
        # coordinate can acquire one rounding unit. Snap that coordinate to
        # its model-owned grid node, then solve only the varying coordinate.
        selected = np.flatnonzero((moving_axes==axis) & np.any(separation!=0,axis=1))
        for begin in range(0,len(selected),16384):
            ids=selected[begin:begin+16384]
            lower=np.floor(moving[ids])/_SUBDIVISION
            upper=lower+1/_SUBDIVISION
            constant=np.rint(fixed[ids])/_SUBDIVISION
            def residual(position,fixed):
                return (field.sample_points(position,fixed) if axis==0
                        else field.sample_points(fixed,position))-level
            root=elementwise.find_root(residual,(lower,upper),args=(constant,),
                                       tolerances=_ROOT_TOLERANCES,maxiter=100)
            if not np.all(root.success):
                raise ValueError("cartographic edge roots must converge on the original terrain")
            solved[ids,axis]=root.x
            solved[ids,fixed_axis]=constant
    result=[];offset=0
    for size in sizes:
        result.append(solved[inverse[offset:offset+size]])
        offset+=size
    return result


def cartographic_curve_batches(field, levels, *, query_bounds=None):
    """Sample once and yield each completed height's true-root display graph."""
    levels=np.asarray(levels,dtype=float)
    if levels.ndim!=1 or not np.isfinite(levels).all() or np.any(np.diff(levels)<=0):
        raise ValueError("cartographic contours require increasing finite heights")
    if not isinstance(field,(PhysicalTerrainField,RefinedTerrainField)):
        raise ValueError("cartographic contours require the accepted physical ground model")
    if not len(levels):return
    samplers=_samplers(field,query_bounds)
    for level in levels:
        cut=float(np.nextafter(level,-np.inf))
        paths=_true_edge_paths(field,cut,[path for sampler in samplers for path in sampler.lines(cut)])
        if len(samplers)>1 and paths:
            geometry=shapely.line_merge(shapely.union_all(shapely.MultiLineString(paths)))
            paths=[shapely.get_coordinates(part) for part in shapely.get_parts(geometry)]
        yield paths


def cartographic_level_curves(field, levels, *, query_bounds=None):
    return list(cartographic_curve_batches(field,levels,query_bounds=query_bounds))
