"""River-corridor supply participates directly in quantitative map fields.

The scalar identity max(background, min(ceiling, supply)) makes every
superlevel an exact Boolean combination of the three source superlevels.
River distance is evaluated on complete continuous drainage lines; it is
never sampled to a raster and interpolated back into a different field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import shapely

from .continuous_scalar import ContinuousScalarField, scalar_band_paths
from .implicit_curves import polygon_path


ECOLOGY_MODEL = "continuous-river-corridor-max-background-min-ceiling-supply-v1"


def _band_geometry(paths):
    polygons = []
    for points, codes in paths:
        points, codes = np.asarray(points), np.asarray(codes)
        starts = np.flatnonzero(codes == 1)
        rings = [points[first:last] for first,last in zip(starts,(*starts[1:],len(points)),strict=True)]
        polygons.append(shapely.Polygon(rings[0],rings[1:]))
    # These faces belong to one noded scalar arrangement. Coverage union can
    # dissolve their shared edges without rebuilding that arrangement.
    return shapely.coverage_union_all(polygons)


@dataclass(frozen=True)
class FreshwaterCorridors:
    """Continuous accepted drainage lines and native freshwater lake areas."""

    width: int
    height: int
    river_orders: tuple[int,...]
    river_geometries: tuple
    lake_geometry: object

    @classmethod
    def from_surfaces(cls, shape, river_paths, river_orders, lake_surface):
        """Bind already accepted continuous reaches and their shared lake shore."""
        height,width = shape
        if (height<1 or width<1 or len(river_paths)!=len(river_orders)
                or not shapely.is_valid(lake_surface)
                or (not lake_surface.is_empty and shapely.get_type_id(lake_surface) not in (3,6))):
            raise ValueError("river ecology requires ordered accepted reaches and a valid continuous lake surface")
        native_orders=np.asarray(river_orders,dtype=int)
        if np.any(native_orders<1):
            raise ValueError("accepted ecological river reaches need positive source order")
        paths=[]
        for path in river_paths:
            points=np.asarray(path,dtype=float)
            if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or np.any(~np.isfinite(points)):
                raise ValueError("accepted ecological river reaches need finite native coordinates")
            paths.append(shapely.LineString(points))
        orders = tuple(int(value) for value in np.unique(native_orders))
        geometries = []
        for value in orders:
            geometries.append(shapely.line_merge(shapely.union_all(
                [path for path,order in zip(paths,native_orders,strict=True) if order==value])))
        return cls(width,height,orders,tuple(geometries),lake_surface)

    def periodic_parts(self, geometry):
        from shapely.affinity import translate
        return np.asarray([part for offset in (-self.width,0,self.width)
                           for part in shapely.get_parts(translate(geometry,xoff=offset))],dtype=object)


class ContinuousEcologyField:
    """One source evaluator, native inventory, and vector superlevel model."""

    def __init__(self, background, ceiling, land_mask, sources:FreshwaterCorridors,
                 *, supply_capacity:float, river_radii, lake_radius:float):
        background,ceiling = np.asarray(background,dtype=float),np.asarray(ceiling,dtype=float)
        land = np.asarray(land_mask,dtype=bool)
        if (background.shape!=land.shape or ceiling.shape!=land.shape
                or background.shape!=(sources.height,sources.width)
                or np.any(~np.isfinite(background)) or np.any(~np.isfinite(ceiling))
                or np.any((background<0)|(background>ceiling)|(ceiling>1))):
            raise ValueError("ecological background and ceiling must be aligned scores in [0,1]")
        radii = tuple(float(value) for value in river_radii)
        if (len(radii)!=len(sources.river_orders) or any(not np.isfinite(value) or value<=0 for value in radii)
                or not np.isfinite(lake_radius) or lake_radius<=0
                or not np.isfinite(supply_capacity) or not 0<supply_capacity<=1):
            raise ValueError("continuous ecological supply needs positive finite source radii and capacity")
        self.width,self.height = sources.width,sources.height
        background=background.astype(np.float32).astype(float)
        ceiling=ceiling.astype(np.float32).astype(float)
        self.background = ContinuousScalarField(background,land)
        self.ceiling = ContinuousScalarField(ceiling,land)
        self.land_mask = land.copy();self.land_mask.flags.writeable=False
        self.sources = sources
        self.supply_capacity = float(np.float32(supply_capacity))
        self.river_radii = radii
        self.lake_radius = float(lake_radius)
        self._sources = [(sources.periodic_parts(geometry),radius)
                         for geometry,radius in zip(sources.river_geometries,radii,strict=True)]
        if not sources.lake_geometry.is_empty:
            self._sources.append((sources.periodic_parts(sources.lake_geometry),self.lake_radius))
        self._trees = []
        for parts,radius in self._sources:
            if not len(parts):continue
            line_indices = np.flatnonzero(shapely.get_type_id(parts)==1)
            polygon_indices = np.flatnonzero(shapely.get_type_id(parts)!=1)
            coordinates,owner = shapely.get_coordinates(parts[line_indices],return_index=True)
            same_line = owner[:-1]==owner[1:]
            segments = shapely.linestrings(np.stack(
                (coordinates[:-1][same_line],coordinates[1:][same_line]),axis=1))
            original_owner = np.concatenate((line_indices[owner[:-1][same_line]],polygon_indices))
            # Every original segment is unchanged. Smaller bounding boxes make
            # nearest queries local; source parts still own all buffer arcs.
            tree = shapely.STRtree(np.concatenate((segments,parts[polygon_indices])))
            self._trees.append((tree,radius,parts,original_owner))
        self._native = None
        self._native_supply = None
        self._native_y, self._native_x = np.where(self.land_mask)
        self._native_y.flags.writeable = self._native_x.flags.writeable = False
        identity = hashlib.sha256()
        identity.update(ECOLOGY_MODEL.encode())
        identity.update(background.tobytes());identity.update(ceiling.tobytes());identity.update(land.tobytes())
        identity.update(json.dumps((self.supply_capacity,self.river_radii,self.lake_radius),separators=(',',':')).encode())
        for parts,radius in self._sources:
            identity.update(shapely.to_wkb(shapely.GeometryCollection(parts)))
        self.fingerprint = identity.hexdigest()

    def supply_points(self,x,y):
        x,y = np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            raise ValueError("ecological supply queries must be finite")
        shape=x.shape;x=x.ravel()%self.width;y=y.ravel()
        result=np.zeros(x.size,dtype=float)
        for begin in range(0,x.size,16384):
            stop=min(x.size,begin+16384)
            points=shapely.points(x[begin:stop],y[begin:stop])
            local=np.zeros(stop-begin,dtype=float)
            for tree,radius,_parts,_owner in self._trees:
                indices,distance=tree.query_nearest(points,max_distance=radius,return_distance=True,all_matches=False)
                kernel=np.square(1.-np.minimum(distance/radius,1.)**2)*self.supply_capacity
                np.maximum.at(local,indices[0],kernel)
            result[begin:stop]=local
        return result.reshape(shape)

    def sample_points(self,x,y):
        return np.maximum(self.background.sample_points(x,y),
                          np.minimum(self.ceiling.sample_points(x,y),self.supply_points(x,y)))

    @property
    def native(self):
        if self._native is None:
            y,x=self._native_y,self._native_x
            supply=self._supply_observations()
            values=np.zeros((self.height,self.width),dtype=float)
            values[y,x]=np.maximum(self.background.native[y,x],np.minimum(self.ceiling.native[y,x],supply))
            values=values.astype(np.float32);values.flags.writeable=False
            self._native=values
        return self._native

    def _supply_observations(self):
        if self._native_supply is None:
            y,x=self._native_y,self._native_x
            self._native_supply=self.supply_points(x+.5,y+.5)
            self._native_supply.flags.writeable=False
        return self._native_supply

    def supply_superlevel(self,level):
        if not np.isfinite(level):
            raise ValueError("ecological supply cut must be finite")
        if level<=0:
            return shapely.box(0,0,self.width,self.height)
        if level>self.supply_capacity:
            return shapely.GeometryCollection()
        factor=np.sqrt(1.-np.sqrt(level/self.supply_capacity))
        y,x=self._native_y,self._native_x
        expected=self._supply_observations()>=level
        # GEOS encodes circular source arcs as inscribed chords. Refine that
        # same distance contour until its chords retain every actual native
        # witness. The model, radius and cut stay unchanged; no classified
        # cell or painted patch participates in constructing the geometry.
        groups=[]
        for tree,radius,parts,owner in self._trees:
            groups.append((parts,radius*factor,tree,owner,
                parts.copy() if factor==0 else shapely.buffer(parts,radius*factor,quad_segs=16)))
        for resolution in (16,64,256,1024,4096):
            contour=shapely.union_all(np.concatenate([geometry for _,_,_,_,geometry in groups])) if groups else shapely.GeometryCollection()
            shapely.prepare(contour)
            observed=shapely.intersects_xy(contour,x+.5,y+.5)
            if np.array_equal(observed,expected):
                return contour
            witnesses=shapely.points(x[expected&~observed]+.5,y[expected&~observed]+.5)
            # Only the original source parts responsible for an unresolved
            # witness need finer circular arcs. All other accepted chords
            # retain their sufficient precision, avoiding global oversampling.
            for parts,radius,tree,owner,geometry in groups:
                if radius<=0 or not len(witnesses):continue
                pairs=tree.query_nearest(witnesses,max_distance=radius,all_matches=True)
                indices=np.unique(owner[pairs[1]])
                if len(indices):geometry[indices]=shapely.buffer(parts[indices],radius,quad_segs=resolution*4)
        raise ValueError("continuous supply arcs must preserve every source native threshold witness")

    def _superlevels(self,thresholds):
        thresholds=np.asarray(thresholds,dtype=float)
        if thresholds.ndim!=1 or np.any(~np.isfinite(thresholds)) or np.any(np.diff(thresholds)<=0):
            raise ValueError("ecological classes require ordered finite cuts")
        background=[_band_geometry(paths) for paths in scalar_band_paths(self.background.native,self.land_mask,thresholds)]
        ceiling=[_band_geometry(paths) for paths in scalar_band_paths(self.ceiling.native,self.land_mask,thresholds)]
        return [shapely.union_all((shapely.coverage_union_all([band for band in background[index+1:] if not band.is_empty]),
                    shapely.intersection(shapely.coverage_union_all([band for band in ceiling[index+1:] if not band.is_empty]),self.supply_superlevel(float(level)))))
                for index,level in enumerate(thresholds)]

    def superlevel(self,level):
        return self._superlevels((level,))[0]

    def class_regions(self,thresholds,*,working_surface):
        from .cartographic_features import shared_display_coverage
        levels=self._superlevels(thresholds)
        frame=shapely.box(0,0,self.width,self.height)
        previous=frame
        regions=[]
        for level in levels:
            regions.append(shapely.difference(previous,level));previous=level
        regions.append(previous)
        return shared_display_coverage(tuple(shapely.intersection(region,working_surface) for region in regions))

    def band_paths(self,thresholds,*,working_surface):
        return [[polygon_path(polygon) for polygon in shapely.get_parts(region)]
                for region in self.class_regions(thresholds,working_surface=working_surface)]
