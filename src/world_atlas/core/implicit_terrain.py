"""Trace the existing physical model with roots owned by that model.

Native PCHIP intervals provide exact base-field topology. Refined ground uses
conservative ranges to discover subgrid features and derivative intervals to
certify each regular branch before encoding its true curve. No sampled raster
is substituted for the refined model.
"""

import numpy as np
import shapely
from scipy.optimize import elementwise

from .continuous_scalar import _threshold_segments, pchip_curve_sections
from .continuous_terrain import PhysicalTerrainField
from .implicit_curves import adaptive_curve_paths
from .implicit_pchip import pchip_switch_abscissae, split_pchip_branches
from .terrain_refinement import RefinedTerrainField


_ROOT_TOLERANCES = {"xatol":1e-14,"xrtol":4*np.finfo(float).eps,"fatol":0.,"frtol":0.}
_CHART_DIRECTIONS=np.array(((1.,0.),(0.,1.),(1.,1.),(1.,-1.)))


def _native_axes(field):
    return (np.r_[0.,np.arange(field.width)+.5,float(field.width)],
            np.r_[0.,np.arange(field.height)+.5,float(field.height)])


def _joined_paths(paths):
    if not paths:return []
    return [shapely.get_coordinates(part) for part in
            shapely.get_parts(shapely.line_merge(shapely.MultiLineString(paths)))]


def _query_cells(field, query_bounds):
    """Unique complete native intervals intersecting the explicit queries."""
    bounds=np.asarray(query_bounds,dtype=float)
    if bounds.shape==(4,):bounds=bounds[None,:]
    if (bounds.ndim!=2 or bounds.shape[1:]!=(4,) or not np.all(np.isfinite(bounds))
            or np.any(bounds[:,:2]>=bounds[:,2:]) or np.any(bounds[:,:2]<0)
            or np.any(bounds[:,2]>field.width) or np.any(bounds[:,3]>field.height)):
        raise ValueError("terrain queries require finite ordered rectangles inside the world frame")
    x,y=_native_axes(field)
    selected=[]
    for west,north,east,south in bounds:
        c0=max(0,int(np.searchsorted(x,west,side='right'))-1)
        c1=min(len(x)-1,int(np.searchsorted(x,east,side='left')))
        r0=max(0,int(np.searchsorted(y,north,side='right'))-1)
        r1=min(len(y)-1,int(np.searchsorted(y,south,side='left')))
        selected.extend((np.arange(r0,r1)[:,None]*(field.width+1)+np.arange(c0,c1)).ravel())
    indices=np.unique(np.asarray(selected,dtype=np.int64))
    return np.column_stack((indices//(field.width+1),indices%(field.width+1)))


def _cell_boxes(field,cells):
    x,y=_native_axes(field)
    rows,columns=cells.T
    return (np.column_stack((x[columns],y[rows])),
            np.column_stack((x[columns+1],y[rows+1])))


def _native_cell_segments(field,cells,level):
    """The existing native PCHIP graph restricted to selected whole cells."""
    lower,upper=_cell_boxes(field,cells)
    if not len(cells):return np.empty((0,2,2)),lower,upper
    corners=_corner_points(lower,upper)
    above=field.sample_points(corners[:,:,0],corners[:,:,1])>level
    crossings=above!=np.roll(above,-1,axis=1)
    owners,edges=np.nonzero(crossings)
    if not len(owners):return np.empty((0,2,2)),lower[:0],upper[:0]
    points=_SharedPorts(field,level).solve(corners[owners,edges],corners[owners,(edges+1)%4])
    point_index=np.full(crossings.shape,-1,dtype=int)
    point_index[owners,edges]=np.arange(len(points))
    segments,selected=[],[]
    for owner in np.flatnonzero(crossings.any(axis=1)):
        ids=point_index[owner];active=ids[ids>=0]
        if len(active)==2:pairs=(active,)
        elif len(active)==4:
            north,east,south,west=ids
            pairs=((west,north),(south,east)) if points[north,0]<=points[south,0] else ((west,south),(north,east))
        else:raise ValueError("a physical native interval requires two or four contour ports")
        for pair in pairs:
            segments.append(points[np.asarray(pair)]);selected.append(owner)
    return np.asarray(segments),lower[selected],upper[selected]


def _base_level_curves(field, levels, cells=None):
    x,y=_native_axes(field)
    if cells is None:
        sampled=np.empty((len(y),len(x)))
        for begin in range(0,len(y),64):
            sampled[begin:begin+64]=field.sample_rect(x,y[begin:begin+64])
    result=[]
    for level in levels:
        # Display classes include exact threshold plateaus. A representable
        # preceding cut traces their perimeter without changing the model.
        level=float(np.nextafter(level,-np.inf))
        segments,lower,upper=(_threshold_segments(field,x,y,sampled,float(level)) if cells is None
                              else _native_cell_segments(field,cells,float(level)))
        if not len(segments):
            result.append([])
            continue
        segments,lower,upper=split_pchip_branches(field,field.native_m,segments,lower,upper,level)
        def sections(starts,ends,owner):
            return pchip_curve_sections(field,field.native_m,starts,ends,
                                        lower[owner],upper[owner],float(level))
        result.append(_joined_paths(adaptive_curve_paths(segments[:,0],segments[:,1],sections)))
    return result


def _selected_ranges(field,cells):
    """The same source range proof, evaluated only on selected native boxes."""
    lower,upper=_cell_boxes(field,cells)
    rows,columns=cells[:,0]-1,cells[:,1]-1
    low=np.full(len(cells),np.inf);high=np.full(len(cells),-np.inf)
    for dr in (0,1):
        for dc in (0,1):
            value=field.native_m[np.clip(rows+dr,0,field.height-1),(columns+dc)%field.width]
            np.minimum(low,value,out=low);np.maximum(high,value,out=high)
    # Half-cells at the longitude frame also interpolate its periodic neighbour.
    frame=(lower[:,0]==0)|(upper[:,0]==field.width)
    if frame.any():
        points=_corner_points(lower[frame],upper[frame])
        value=field.sample_points(points[:,:,0],points[:,:,1]) if isinstance(field,PhysicalTerrainField) else field.base.sample_points(points[:,:,0],points[:,:,1])
        low[frame]=np.minimum(low[frame],value.min(axis=1));high[frame]=np.maximum(high[frame],value.max(axis=1))
    if isinstance(field,PhysicalTerrainField):return low,high
    if field._landforms.maximum_displacement>=1:
        raise ValueError("implicit terrain coarse bounds require subcell ground deformation")
    if field._landforms.count and len(cells):
        pairs=field._landforms._tree.query(shapely.box(lower[:,0],lower[:,1],upper[:,0],upper[:,1]))
        warped=np.unique(pairs[0])
        for dr in (-1,0,1,2):
            for dc in (-1,0,1,2):
                value=field.native_m[np.clip(rows[warped]+dr,0,field.height-1),(columns[warped]+dc)%field.width]
                low[warped]=np.minimum(low[warped],value);high[warped]=np.maximum(high[warped],value)
    amplitude=np.zeros(len(cells))
    for dr in (0,1):
        for dc in (0,1):
            np.maximum(amplitude,field._amplitude[np.clip(rows+dr,0,field.height-1),(columns+dc)%field.width],out=amplitude)
    minimum_abs=np.where((low<=0)&(high>=0),0.,np.minimum(abs(low),abs(high)))
    gain=.45*np.tanh(amplitude/(minimum_abs+30))
    ids=np.flatnonzero(gain!=0)
    if len(ids):
        zero=_zero_detail_boxes(field,rows[ids],columns[ids],lower[ids],upper[ids])
        gain[ids[zero]]=0.
    products=np.stack((low*(1-gain),low*(1+gain),high*(1-gain),high*(1+gain)))
    low, high = products.min(axis=0), products.max(axis=0)
    if field._river_bed is not None and len(field._river_bed.segments):
        bed = field._river_bed
        radius = bed.maximum_radius
        boxes = shapely.box(lower[:,0]-radius, lower[:,1]-radius,
                            upper[:,0]+radius, upper[:,1]+radius)
        owner, segment = bed._tree.query(boxes)
        np.minimum.at(low, owner, bed.beds[segment].min(axis=1))
    return low, high


def terrain_height_range(field,query_bounds):
    """Conservative actual-model height range over explicit native queries."""
    if not isinstance(field,(PhysicalTerrainField,RefinedTerrainField)):
        raise ValueError("terrain height ranges require the accepted physical ground model")
    cells=_query_cells(field,query_bounds)
    if not len(cells):raise ValueError("terrain height ranges require at least one query rectangle")
    low,high=_selected_ranges(field,cells)
    return float(low.min()),float(high.max())


def _zero_detail_boxes(field,rows,columns,lower,upper):
    """Certify the unchanged drainage detail is identically zero in a box."""
    drainage=field._height
    zero=np.ones(len(rows),dtype=bool)
    for dr in (0,1):
        for dc in (0,1):
            zero &= drainage.pchip.native_m[
                np.clip(rows+dr,0,field.height-1),(columns+dc)%field.width]==0
    # Shape-preserving PCHIP between two zero horizontal rows is exactly zero.
    # Its full source relief is also zero only outside all compact profiles.
    ids=np.flatnonzero(zero)
    if drainage.edge_count and len(ids):
        for begin in range(0,len(ids),16384):
            selected=ids[begin:begin+16384];a,b=lower[selected],upper[selected]
            cosine=np.maximum(np.minimum(np.cos(np.pi/2-a[:,1]*np.pi/field.height),
                np.cos(np.pi/2-b[:,1]*np.pi/field.height)),.08)
            margin_x=np.nextafter(drainage.maximum_breadth/
                (drainage.cell_x_km*np.nextafter(cosine,-np.inf)),np.inf)
            margin_y=np.nextafter(drainage.maximum_breadth/drainage.cell_y_km,np.inf)
            query=shapely.box(np.nextafter(a[:,0]-margin_x,-np.inf),
                np.nextafter(a[:,1]-margin_y,-np.inf),np.nextafter(b[:,0]+margin_x,np.inf),
                np.nextafter(b[:,1]+margin_y,np.inf))
            touched=drainage._tree.query(query)
            if touched.shape[1]:zero[selected[np.unique(touched[0])]]=False
    return zero


def _coarse_ranges(field):
    """Enclose each initial native box, including the actual warp and gain."""
    native=field.native_m
    rows=np.r_[-1,np.arange(field.height)]
    columns=np.r_[-1,np.arange(field.width)]
    displacement=field._landforms.maximum_displacement
    # Current compact patches move less than a quarter native cell. A larger
    # future model must supply correspondingly larger source support here.
    if displacement >= 1:
        raise ValueError("implicit terrain coarse bounds require subcell ground deformation")
    low=np.full((len(rows),len(columns)),np.inf)
    high=np.full_like(low,-np.inf)
    for dr in (0,1):
        for dc in (0,1):
            values=native[np.clip(rows+dr,0,field.height-1)[:,None],
                          ((columns+dc)%field.width)[None,:]]
            np.minimum(low,values,out=low);np.maximum(high,values,out=high)
    if displacement:
        x,y=_native_axes(field)
        warped=np.zeros_like(low,dtype=bool)
        patches=field._landforms
        for centre,radius in zip(patches.centres,patches.radius,strict=True):
            west=max(0,int(np.searchsorted(x,centre[0]-radius,side='right'))-1)
            east=min(len(columns)-1,int(np.searchsorted(x,centre[0]+radius,side='left')))
            north=max(0,int(np.searchsorted(y,centre[1]-radius,side='right'))-1)
            south=min(len(rows)-1,int(np.searchsorted(y,centre[1]+radius,side='left')))
            if west<=east and north<=south:warped[north:south+1,west:east+1]=True
        rr,cc=np.nonzero(warped)
        for dr in (-1,0,1,2):
            for dc in (-1,0,1,2):
                values=native[np.clip(rows[rr]+dr,0,field.height-1),
                              (columns[cc]+dc)%field.width]
                low[rr,cc]=np.minimum(low[rr,cc],values)
                high[rr,cc]=np.maximum(high[rr,cc],values)
    amplitude=np.zeros_like(low)
    for dr in (0,1):
        for dc in (0,1):
            values=field._amplitude[np.clip(rows+dr,0,field.height-1)[:,None],
                                    ((columns+dc)%field.width)[None,:]]
            np.maximum(amplitude,values,out=amplitude)
    minimum_abs=np.where((low<=0)&(high>=0),0.,np.minimum(abs(low),abs(high)))
    gain=.45*np.tanh(amplitude/(minimum_abs+30))
    rr,cc=np.nonzero(gain!=0)
    if len(rr):
        x,y=_native_axes(field)
        lower=np.column_stack((x[cc],y[rr]));upper=np.column_stack((x[cc+1],y[rr+1]))
        zero=_zero_detail_boxes(field,rows[rr],columns[cc],lower,upper)
        gain[rr[zero],cc[zero]]=0.
    products=np.stack((low*(1-gain),low*(1+gain),high*(1-gain),high*(1+gain)))
    return products.min(axis=0),products.max(axis=0)


def _corner_points(lower,upper):
    return np.stack((lower,np.column_stack((upper[:,0],lower[:,1])),upper,
                     np.column_stack((lower[:,0],upper[:,1]))),axis=1)


def _monotone(gradient_low,gradient_high):
    return (gradient_low>0)|(gradient_high<0)


def _chart_parameters(directions):
    parameters=np.column_stack((directions[:,1],-directions[:,0]))
    reverse=(parameters[:,0]<0)|((parameters[:,0]==0)&(parameters[:,1]<0))
    parameters[reverse]*=-1
    return parameters


def _gradient_directions(gradient_low,gradient_high):
    """A bounded four-point proposal, conditioned before distance arithmetic.

    Outside the gradient hull its nearest edge occurs among these six chords.
    An interior hull may propose another direction; only the complete model
    derivative certificate can accept it. GEOS nearest-point arithmetic can
    underflow on finite subnormal gradients, so this tiny candidate operation
    uses scaled segment projections, independently of map geometry.
    """
    low=np.asarray(gradient_low).reshape(-1,4,2)
    high=np.asarray(gradient_high).reshape(-1,4,2)
    if not np.isfinite(low).all()or not np.isfinite(high).all():
        raise ValueError("terrain direction proposals require finite point gradient intervals")
    scale=np.maximum(np.max(abs(low),axis=(1,2)),np.max(abs(high),axis=(1,2)))[:,None,None]
    a=np.divide(low,scale,out=np.zeros_like(low),where=scale!=0)
    b=np.divide(high,scale,out=np.zeros_like(high),where=scale!=0)
    gradients=(a+b)*.5
    first,last=np.triu_indices(4,1)
    a,b=gradients[:,first],gradients[:,last]
    delta=b-a;length=np.max(abs(delta),axis=2)
    direction=np.divide(delta,length[:,:,None],out=np.zeros_like(delta),where=length[:,:,None]!=0)
    denominator=np.sum(direction**2,axis=2)
    station=np.divide(-np.sum(a*direction,axis=2),denominator,
                       out=np.zeros_like(denominator),where=denominator!=0)
    station=np.minimum(np.maximum(station,0.),length)
    closest=a+station[:,:,None]*direction
    index=np.argmin(np.hypot(closest[:,:,0],closest[:,:,1]),axis=1)
    normals=closest[np.arange(len(closest)),index]
    scale=np.max(abs(normals),axis=1)
    normals=np.divide(normals,scale[:,None],out=np.zeros_like(normals),where=scale[:,None]!=0)
    reverse=(normals[:,0]<0)|((normals[:,0]==0)&(normals[:,1]<0))
    normals[reverse]*=-1
    return normals


def _separating_directions(field,lower,upper,bounds):
    """Point intervals propose a direction; the whole-box oracle certifies it."""
    points=_corner_points(lower,upper).reshape(-1,2)
    _,_,glo,ghi=bounds(field,points,points)
    return _gradient_directions(glo,ghi)


def _direction_sections(field,level,points,lower,upper,directions):
    """Roots on a signed-direction chart inside its original model box."""
    centre=(lower+upper)*.5
    parameter=_chart_parameters(directions)
    t=np.sum((points-centre)*parameter,axis=1)/np.sum(parameter**2,axis=1)
    base=centre+t[:,None]*parameter
    first=(lower-base)/directions;last=(upper-base)/directions
    minimum=np.minimum(first,last).max(axis=1);maximum=np.maximum(first,last).min(axis=1)
    def residual(position,bx,by,dx,dy):
        return field.sample_points(bx+dx*position,by+dy*position)-level
    result=elementwise.find_root(residual,(minimum,maximum),
        args=(base[:,0],base[:,1],directions[:,0],directions[:,1]),
        tolerances=_ROOT_TOLERANCES,maxiter=100)
    if not bool(np.all(result.success)):
        raise ValueError("certified directional physical branch sections must converge")
    return base+result.x[:,None]*directions


class _SharedPorts:
    """Reuse a true edge root regardless of neighbouring subdivision depth."""
    def __init__(self,field,level):
        self.field,self.level=field,float(level)
        self.roots={}
        self.model_roots={}
        self.warped_roots={}

    def solve(self,first,last):
        axes=np.argmax(abs(last-first),axis=1)
        points=np.empty_like(first)
        pending=[]
        for i,(a,b,axis) in enumerate(zip(first,last,axes,strict=True)):
            fixed=float(a[1-axis]);start,stop=sorted((float(a[axis]),float(b[axis])))
            canonical_fixed=fixed%self.field.width if axis==1 else fixed
            key=(int(axis),canonical_fixed)
            roots=self.roots.get(key,())
            found=[v for v in roots if start<=v<=stop]
            if len(found)>1:
                raise ValueError("a certified monotone edge contains multiple known terrain roots")
            if found:
                point=a.copy();point[axis]=found[0];points[i]=point
            else:
                pending.append((i,int(axis),fixed,start,stop))
        for axis in (0,1):
            selected=[item for item in pending if item[1]==axis]
            if not selected:continue
            ids=np.array([p[0]for p in selected])
            fixed=np.array([p[2]for p in selected])
            a=np.array([p[3]for p in selected]);b=np.array([p[4]for p in selected])
            def residual(t,constant):
                return (self.field.sample_points(t,constant) if axis==0
                        else self.field.sample_points(constant,t))-self.level
            result=elementwise.find_root(residual,(a,b),args=(fixed,),
                                         tolerances=_ROOT_TOLERANCES,maxiter=100)
            if not bool(np.all(result.success)):
                raise ValueError("certified physical edge roots must converge")
            points[ids,axis]=result.x;points[ids,1-axis]=fixed
            for item,value in zip(selected,result.x,strict=True):
                key=(axis,item[2]%self.field.width if axis==1 else item[2])
                known=[v for v in self.roots.get(key,())if item[3]<=v<=item[4]]
                if len(known)>1:
                    raise ValueError("a certified monotone edge contains multiple periodic terrain roots")
                if known:
                    points[item[0],axis]=known[0]
                else:
                    self.roots.setdefault(key,set()).add(float(value))
        return points

    def solve_model_lines(self,first,last,normals,offsets,line_keys,periods,canonical_offsets):
        """Reuse a true root of a recorded source-model switching line."""
        solved=np.argmax(abs(normals),axis=1);free=1-solved
        points=np.empty_like(first);pending=[]
        translations=np.asarray(periods)*self.field.width
        for i,(key,axis)in enumerate(zip(line_keys,free,strict=True)):
            start,stop=sorted((first[i,axis],last[i,axis]))
            if axis==0:start-=translations[i];stop-=translations[i]
            key=tuple(key)
            known=[v for v in self.model_roots.get(key,())if start<=v<=stop]
            if len(known)>1:
                raise ValueError("a certified model line contains multiple known terrain roots")
            if known:
                points[i,axis]=known[0]
            else:pending.append((i,key,start,stop))
        if pending:
            ids=np.array([item[0]for item in pending]);axes=free[ids];other=solved[ids]
            a=np.array([item[2]for item in pending]);b=np.array([item[3]for item in pending])
            selected_normals=normals[ids];selected_offsets=canonical_offsets[ids]
            rows=np.arange(len(ids))
            def residual(t,nfree,nother,offset,axis):
                fixed=(offset-nfree*t)/nother
                x=np.where(axis==0,t,fixed);y=np.where(axis==0,fixed,t)
                return self.field.sample_points(x,y)-self.level
            result=elementwise.find_root(residual,(a,b),
                args=(selected_normals[rows,axes],selected_normals[rows,other],selected_offsets,axes),
                tolerances=_ROOT_TOLERANCES,maxiter=100)
            if not bool(np.all(result.success)):
                raise ValueError("certified physical model-line roots must converge")
            points[ids,axes]=result.x
            for item,value in zip(pending,result.x,strict=True):
                known=[v for v in self.model_roots.get(item[1],())if item[2]<=v<=item[3]]
                if len(known)>1:
                    raise ValueError("a certified model line contains multiple periodic terrain roots")
                if known:points[item[0],free[item[0]]]=known[0]
                else:self.model_roots.setdefault(item[1],set()).add(float(value))
        rows=np.arange(len(first))
        points[rows,free]+=np.where(free==0,translations,0.)
        points[rows,solved]=(offsets-normals[rows,free]*points[rows,free])/normals[rows,solved]
        return points

    def solve_warped_events(self,first,last,source_x,canonical_x):
        """Solve the original height on an exact inverse-W source event."""
        parameters=np.empty(len(first));pending=[]
        for i,(a,b,key)in enumerate(zip(first,last,canonical_x,strict=True)):
            known=[v for v in self.warped_roots.get(float(key),())if a<=v<=b]
            if len(known)>1:
                raise ValueError("a certified warped event contains multiple known terrain roots")
            if known:parameters[i]=known[0]
            else:pending.append((i,float(key),a,b))
        if pending:
            ids=np.array([item[0]for item in pending])
            def residual(t,x):
                px,py=self.field.inverse_ground_coordinates(x,t)
                return self.field.sample_points(px,py)-self.level
            result=elementwise.find_root(residual,(first[ids],last[ids]),args=(source_x[ids],),
                                         tolerances=_ROOT_TOLERANCES,maxiter=100)
            if not bool(np.all(result.success)):
                raise ValueError("certified warped physical event roots must converge")
            for item,value in zip(pending,result.x,strict=True):
                known=[v for v in self.warped_roots.get(item[1],())if item[2]<=v<=item[3]]
                if len(known)>1:
                    raise ValueError("a certified warped event contains multiple periodic terrain roots")
                parameters[item[0]]=known[0]if known else value
                if not known:self.warped_roots.setdefault(item[1],set()).add(float(value))
        return np.column_stack(self.field.inverse_ground_coordinates(source_x,parameters))


class _TerrainBatchBounds:
    """Reuse identical model certificates across cuts inside one native batch."""
    def __init__(self,field):
        self.field=field
        self.records={}
        self.hits=0
        self.misses=0

    def __call__(self,field,lower,upper):
        from .implicit_terrain_bounds import field_range_gradient_bounds
        if field is not self.field:
            raise ValueError("a native batch certificate belongs to exactly one physical model")
        boxes=np.column_stack((lower,upper))
        keys=[tuple(box)for box in boxes]
        unknown={}
        for index,key in enumerate(keys):
            if key not in self.records:unknown.setdefault(key,index)
        if unknown:
            ids=np.fromiter(unknown.values(),dtype=int)
            lo,hi,glo,ghi=field_range_gradient_bounds(field,lower[ids],upper[ids])
            values=np.column_stack((lo,hi,glo,ghi))
            self.records.update(zip(unknown,values,strict=True))
        self.misses+=len(unknown);self.hits+=len(keys)-len(unknown)
        values=np.array([self.records[key]for key in keys]).reshape(-1,6)
        return values[:,0],values[:,1],values[:,2:4],values[:,4:6]


def _edge_roots(field,level,first,last,ports,bounds):
    """Isolate all transverse native-line roots, including hidden pairs."""
    owner=np.arange(len(first))
    roots=[[]for _ in first]
    for _ in range(40):
        if not len(first):break
        axes=np.argmax(abs(last-first),axis=1)
        lo,hi,glo,ghi=bounds(field,np.minimum(first,last),np.maximum(first,last))
        active=(lo<=level)&(hi>=level)&(lo!=hi)
        first,last,owner,glo,ghi,axes=(v[active]for v in(first,last,owner,glo,ghi,axes))
        if not len(first):break
        signed=_monotone(glo,ghi)[np.arange(len(first)),axes]
        values=field.sample_points(np.column_stack((first[:,0],last[:,0])),
                                  np.column_stack((first[:,1],last[:,1])))
        crossing=(values[:,0]>level)!=(values[:,1]>level)
        selected=np.flatnonzero(signed&crossing)
        if len(selected):
            points=ports.solve(first[selected],last[selected])
            for index,point in zip(owner[selected],points,strict=True):roots[int(index)].append(point)
        unresolved=~signed
        first,last,owner=first[unresolved],last[unresolved],owner[unresolved]
        if not len(first):break
        middle=(first+last)*.5
        axes=np.argmax(abs(last-first),axis=1)
        if np.any(middle[np.arange(len(first)),axes]==first[np.arange(len(first)),axes])or np.any(middle[np.arange(len(first)),axes]==last[np.arange(len(first)),axes]):
            raise ValueError("a physical model event cannot isolate its transverse root at binary64 precision")
        first,last=np.concatenate((first,middle)),np.concatenate((middle,last))
        owner=np.tile(owner,2)
    else:
        if len(first):raise ValueError("physical model event roots must converge")
    return roots


def _model_line_roots(field,level,segments,normals,offsets,line_keys,periods,canonical_offsets,ports,bounds):
    """Isolate all transverse roots on actual model lines, without sampling seeds."""
    from .implicit_terrain_bounds import directional_bounds
    first,last=segments[:,0],segments[:,1]
    owner=np.arange(len(first));roots=[[]for _ in first]
    for _ in range(40):
        if not len(first):break
        lower,upper=np.minimum(first,last),np.maximum(first,last)
        lo,hi,_,_=bounds(field,lower,upper)
        active=(lo<=level)&(hi>=level)&(lo!=hi)
        first,last,owner=(v[active]for v in(first,last,owner))
        if not len(first):break
        normal=normals[owner];offset=offsets[owner]
        direction=np.column_stack((-normal[:,1],normal[:,0]))
        matrix=np.stack((direction,normal),axis=2)
        dlo,dhi=directional_bounds(field,np.minimum(first,last),np.maximum(first,last),matrix)
        signed=_monotone(dlo,dhi)[:,0]
        values=field.sample_points(np.column_stack((first[:,0],last[:,0])),
                                  np.column_stack((first[:,1],last[:,1])))
        crossing=(values[:,0]>level)!=(values[:,1]>level)
        ids=np.flatnonzero(signed&crossing)
        if len(ids):
            source_ids=owner[ids]
            points=ports.solve_model_lines(first[ids],last[ids],normal[ids],offset[ids],
                [line_keys[identifier]for identifier in source_ids],periods[source_ids],canonical_offsets[source_ids])
            for identifier,point in zip(owner[ids],points,strict=True):roots[int(identifier)].append(point)
        unresolved=~signed
        first,last,owner=(v[unresolved]for v in(first,last,owner))
        if not len(first):break
        normal=normals[owner];offset=offsets[owner]
        solved=np.argmax(abs(normal),axis=1);free=1-solved;rows=np.arange(len(first))
        middle=np.empty_like(first)
        middle[rows,free]=(first[rows,free]+last[rows,free])*.5
        middle[rows,solved]=(offset-normal[rows,free]*middle[rows,free])/normal[rows,solved]
        if np.any(middle[rows,free]==first[rows,free])or np.any(middle[rows,free]==last[rows,free]):
            raise ValueError("a model switching line cannot isolate its root at binary64 precision")
        first,last=np.concatenate((first,middle)),np.concatenate((middle,last));owner=np.tile(owner,2)
        if len(first)>1000000:
            raise ValueError("physical switching-line root isolation did not contract")
    else:
        if len(first):raise ValueError("physical model switching-line roots must converge")
    return roots


def _warped_event_roots(field,level,events,lower,upper,ports):
    """Isolate roots on transported source events using their true 1D oracle."""
    from .implicit_terrain_bounds import warped_pchip_event_bounds
    first=np.array([event['sourceYLower']for event in events])
    last=np.array([event['sourceYUpper']for event in events])
    source_x=np.array([event['sourceX']for event in events])
    canonical_x=np.array([event['canonicalSourceX']for event in events])
    branch=np.array([event['owner']for event in events],dtype=int)
    owner=np.arange(len(events));roots=[[]for _ in events]
    for _ in range(40):
        if not len(first):break
        a,b,lo,hi,dlo,dhi=warped_pchip_event_bounds(field,first,last,source_x[owner])
        active=(lo<=level)&(hi>=level)&(lo!=hi)
        active &= np.all((b>=lower[branch[owner]])&(a<=upper[branch[owner]]),axis=1)
        first,last,owner,dlo,dhi=(v[active]for v in(first,last,owner,dlo,dhi))
        if not len(first):break
        signed=(dlo>0)|(dhi<0)
        xx=np.column_stack((source_x[owner],source_x[owner]))
        px,py=field.inverse_ground_coordinates(xx,np.column_stack((first,last)))
        values=field.sample_points(px,py)
        crossing=(values[:,0]>level)!=(values[:,1]>level)
        ids=np.flatnonzero(signed&crossing)
        if len(ids):
            points=ports.solve_warped_events(first[ids],last[ids],source_x[owner[ids]],canonical_x[owner[ids]])
            for identifier,point in zip(owner[ids],points,strict=True):
                index=branch[identifier]
                if np.all(point>=lower[index])and np.all(point<=upper[index]):roots[int(identifier)].append(point)
        unresolved=~signed
        first,last,owner=(v[unresolved]for v in(first,last,owner))
        if not len(first):break
        middle=first+(last-first)*.5
        if np.any(middle==first)or np.any(middle==last):
            raise ValueError("a transported PCHIP event cannot isolate its root at binary64 precision")
        first,last=np.r_[first,middle],np.r_[middle,last];owner=np.tile(owner,2)
        if len(first)>1000000:
            raise ValueError("transported PCHIP event isolation did not contract")
    else:
        if len(first):raise ValueError("transported PCHIP model-event roots must converge")
    return roots


def _refined_event_sections(field,level,starts,ends,lower,upper,axes,directions,ports,bounds):
    """Insert true ports at the original model's explicit switching events."""
    from .implicit_river_events import interior_river_medials, interior_medial_is_nearest
    from .implicit_warp_events import warped_pchip_events
    height_events=pchip_switch_abscissae(field._height.pchip,lower,upper)
    base_events=pchip_switch_abscissae(field.base,lower,upper)
    events=[np.unique(np.r_[height,base])for height,base in zip(height_events,base_events,strict=True)]
    owners,first,last,height_switch=[],[],[],[]
    for owner,xvalues in enumerate(events):
        if axes[owner]==1:
            a,b=sorted((starts[owner,0],ends[owner,0]))
            xvalues=xvalues[(xvalues>a)&(xvalues<b)]
        for x in xvalues:
            owners.append(owner);first.append((x,lower[owner,1]));last.append((x,upper[owner,1]))
            height_switch.append(bool(np.any(height_events[owner]==x)))
    markers=[[]for _ in starts]
    def append_marker(owner,point,is_height):
        if not is_height and field._landforms.count:
            patches=field._landforms
            ids=patches._tree.query(shapely.Point(point))
            if len(ids)and np.any(np.sum((point-patches.centres[ids])**2,axis=1)<patches.radius[ids]**2):
                return
        markers[owner].append(point)
    if owners:
        owners=np.asarray(owners,dtype=int);first,last=np.asarray(first),np.asarray(last)
        height_switch=np.asarray(height_switch)
        direct=np.flatnonzero(axes[owners]==1)
        if len(direct):
            points=ports.solve(first[direct],last[direct])
            for owner,point,is_height in zip(owners[direct],points,height_switch[direct],strict=True):
                append_marker(owner,point,is_height)
        indirect=np.flatnonzero(axes[owners]!=1)
        if len(indirect):
            points=_edge_roots(field,level,first[indirect],last[indirect],ports,bounds)
            for owner,part,is_height in zip(owners[indirect],points,height_switch[indirect],strict=True):
                for point in part:append_marker(owner,point,is_height)
    medials=interior_river_medials(field,lower,upper)
    if medials:
        segments=np.array([event['segment']for event in medials])
        normals=np.array([event['normal']for event in medials]);offsets=np.array([event['offset']for event in medials])
        keys=[event['canonicalLineKey']for event in medials]
        periods=np.array([event['longitudePeriod']for event in medials])
        canonical_offsets=np.array([event['canonicalOffset']for event in medials])
        points=_model_line_roots(field,level,segments,normals,offsets,keys,periods,canonical_offsets,ports,bounds)
        for event,part in zip(medials,points,strict=True):
            markers[event['owner']].extend(point for point in part if interior_medial_is_nearest(field,event,point))
    warped=warped_pchip_events(field,lower,upper)
    if warped:
        points=_warped_event_roots(field,level,warped,lower,upper,ports)
        for event,part in zip(warped,points,strict=True):markers[event['owner']].extend(part)
    segments,new_owners=[],[]
    for owner,part in enumerate(markers):
        parameter=_chart_parameters(directions[owner:owner+1])[0]
        a,b=sorted((starts[owner]@parameter,ends[owner]@parameter))
        part=np.unique(np.asarray(part).reshape(-1,2),axis=0)
        part=[point for point in part if a<point@parameter<b]
        part.sort(key=lambda point:point@parameter,reverse=starts[owner]@parameter>ends[owner]@parameter)
        points=np.vstack((starts[owner],*part,ends[owner]))
        segments.extend(np.stack((points[:-1],points[1:]),axis=1));new_owners.extend([owner]*(len(points)-1))
    segments,new_owners=np.asarray(segments),np.asarray(new_owners)
    return (segments[:,0],segments[:,1],lower[new_owners],upper[new_owners],
            axes[new_owners],directions[new_owners])


def _refined_branch_requests(field,level,lower,upper,ports,bounds):
    """Trace one true-port graph, batching its model queries across cuts."""
    starts,ends,branch_low,branch_high,branch_axes,branch_directions=[],[],[],[],[],[]
    terminal_paths=[]
    depth=0
    # Certification controls termination; a native interval can require more
    # than forty subdivisions before its conservative range excludes a level.
    while len(lower):
        value_low,value_high,gradient_low,gradient_high=yield ("range",lower,upper)
        active=(value_low<=level)&(value_high>=level)&(value_low!=value_high)
        lower,upper,gradient_low,gradient_high=(a[active]for a in
                                                (lower,upper,gradient_low,gradient_high))
        if not len(lower):break
        corners=_corner_points(lower,upper)
        values=field.sample_points(corners[:,:,0],corners[:,:,1])
        above=values>level
        crossing=above!=np.roll(above,-1,axis=1)
        count=crossing.sum(axis=1)
        monotone=_monotone(gradient_low,gradient_high)
        # Both coordinate derivatives certify all four boundary edges at once.
        edge_ok=np.all(monotone,axis=1)
        need_edges=~edge_ok & np.any(monotone,axis=1)
        ids=np.flatnonzero(need_edges)
        if len(ids):
            edge_first=corners[ids].reshape(-1,2)
            edge_last=np.roll(corners[ids],-1,axis=1).reshape(-1,2)
            edge_lower=np.minimum(edge_first,edge_last);edge_upper=np.maximum(edge_first,edge_last)
            elo,ehi,eglo,eghi=yield ("range",edge_lower,edge_upper)
            axes=np.argmax(edge_upper-edge_lower,axis=1)
            edge_monotone=_monotone(eglo,eghi)[np.arange(len(axes)),axes]
            excluded=(elo>level)|(ehi<level)
            edge_ok[ids]=np.all((edge_monotone|excluded).reshape(-1,4),axis=1)
        regular=edge_ok & np.any(monotone,axis=1) & ((count==0)|(count==2))
        midpoint=lower+(upper-lower)*.5
        splittable=(midpoint>lower)&(midpoint<upper)
        # Adjacent binary64 coordinates have no queryable interior. Their
        # four values and shared edge roots exhaust the represented cell;
        # interval rounding can otherwise keep it active indefinitely.
        terminal=~np.any(splittable,axis=1)
        regular|=terminal & (count==0)
        for identifier in np.flatnonzero(terminal & (count==2)):
            edges=np.flatnonzero(crossing[identifier])
            points=ports.solve(corners[identifier,edges],corners[identifier,(edges+1)%4])
            if not np.array_equal(points[0],points[1]):terminal_paths.append(points)
            regular[identifier]=True
        diagonal_ids=np.flatnonzero(~regular & ~np.any(monotone,axis=1))
        if len(diagonal_ids):
            dlo,dhi=yield ("direction",lower[diagonal_ids],upper[diagonal_ids],
                           np.array(((1.,1.),(1.,-1.))))
            signed=_monotone(dlo,dhi)
            candidates=np.broadcast_to(_CHART_DIRECTIONS[2:],(len(diagonal_ids),2,2)).copy()
            proposed_ids=np.flatnonzero(~signed.any(axis=1))
            if len(proposed_ids):
                ids=diagonal_ids[proposed_ids]
                points=_corner_points(lower[ids],upper[ids]).reshape(-1,2)
                _,_,glo,ghi=yield ("range",points,points)
                proposed=_gradient_directions(glo,ghi)
                valid=np.any(proposed!=0,axis=1)
                plo,phi=np.zeros(len(ids)),np.zeros(len(ids))
                if valid.any():
                    matrix=np.stack((proposed[valid],_chart_parameters(proposed[valid])),axis=2)
                    lo,hi=yield ("direction",lower[ids[valid]],upper[ids[valid]],matrix)
                    plo[valid],phi[valid]=lo[:,0],hi[:,0]
                extra=np.zeros((len(diagonal_ids),2));extra[proposed_ids]=proposed
                candidates=np.concatenate((candidates,extra[:,None]),axis=1)
                low=np.zeros(len(diagonal_ids));high=np.zeros(len(diagonal_ids))
                low[proposed_ids],high[proposed_ids]=plo,phi
                dlo=np.column_stack((dlo,low));dhi=np.column_stack((dhi,high))
                signed=_monotone(dlo,dhi)
            selected=np.flatnonzero(signed.any(axis=1))
            if len(selected):
                owners=diagonal_ids[selected]
                edge_first=corners[owners].reshape(-1,2)
                edge_last=np.roll(corners[owners],-1,axis=1).reshape(-1,2)
                edge_ports=_edge_roots(field,level,edge_first,edge_last,ports,bounds)
                for index,identifier in enumerate(owners):
                    points=[]
                    for part in edge_ports[index*4:index*4+4]:
                        for point in part:
                            if not any(np.array_equal(point,known)for known in points):points.append(point)
                    if not points:
                        regular[identifier]=True
                    elif len(points)==2:
                        points=np.asarray(points)
                        magnitude=np.where(dlo[selected[index]]>0,dlo[selected[index]],-dhi[selected[index]])
                        parameters=_chart_parameters(candidates[selected[index]])
                        usable=signed[selected[index]] & (
                            points[0]@parameters.T != points[1]@parameters.T)
                        if not usable.any():
                            raise ValueError("a certified directional terrain branch has a collapsed independent coordinate")
                        direction=candidates[selected[index],int(np.argmax(np.where(usable,magnitude,-np.inf)))]
                        starts.append(points[0]);ends.append(points[1])
                        branch_low.append(lower[identifier]);branch_high.append(upper[identifier]);branch_axes.append(2)
                        branch_directions.append(direction)
                        regular[identifier]=True
        accepted=np.flatnonzero(regular & ~terminal & (count==2) & np.any(monotone,axis=1))
        if len(accepted):
            first,last=[],[]
            for identifier in accepted:
                for edge in np.flatnonzero(crossing[identifier]):
                    first.append(corners[identifier,edge]);last.append(corners[identifier,(edge+1)%4])
            true_ports=ports.solve(np.asarray(first),np.asarray(last)).reshape(-1,2,2)
            for k,identifier in enumerate(accepted):
                if bool(np.all(true_ports[k,0]==true_ports[k,1])):
                    continue
                usable=monotone[identifier] & (true_ports[k,0,::-1]!=true_ports[k,1,::-1])
                magnitude=np.where(gradient_low[identifier]>0,gradient_low[identifier],
                                   -gradient_high[identifier])
                axis=int(np.argmax(np.where(usable,magnitude,-np.inf)))
                # A branch chart uses the other coordinate as its parameter.
                if not bool(np.any(usable)):
                    raise ValueError("certified terrain branch has a collapsed independent coordinate: "
                                     f"box={[lower[identifier].tolist(),upper[identifier].tolist()]}, "
                                     f"ports={true_ports[k].tolist()}, axis={axis}, "
                                     f"gradient={[gradient_low[identifier].tolist(),gradient_high[identifier].tolist()]}, "
                                     f"values={field.sample_points(true_ports[k,:,0],true_ports[k,:,1]).tolist()}")
                starts.append(true_ports[k,0]);ends.append(true_ports[k,1])
                branch_low.append(lower[identifier]);branch_high.append(upper[identifier]);branch_axes.append(axis)
                branch_directions.append(_CHART_DIRECTIONS[axis])
        unresolved=~regular
        lower,upper,monotone,splittable=(a[unresolved]for a in(lower,upper,monotone,splittable))
        if not len(lower):break
        middle=lower+(upper-lower)*.5
        if np.any(~np.any(splittable,axis=1)):
            ids=np.flatnonzero(~np.any(splittable,axis=1))[:3]
            vlo,vhi,glo,ghi=bounds(field,lower[ids],upper[ids])
            points=_corner_points(lower[ids],upper[ids])
            raise ValueError(f"physical contour unresolved at binary64 critical cell, level={level}, "
                             f"depth={depth},boxes={list(zip(lower[ids].tolist(),upper[ids].tolist()))}, "
                             f"ranges={[vlo.tolist(),vhi.tolist()]},gradient={[glo.tolist(),ghi.tolist()]}, "
                             f"values={field.sample_points(points[:,:,0],points[:,:,1]).tolist()}")
        child_low,child_high=[],[]
        # Retain a proven root bracket. Splitting its solved coordinate adds
        # artificial boundaries arbitrarily close to a real model corner.
        # Refine only the independent parameter when one chart is certified.
        one_axis=np.sum(monotone,axis=1)==1
        split=splittable & ~monotone
        split[~one_axis]=splittable[~one_axis]
        if np.any(~np.any(split,axis=1)):
            raise ValueError(f"physical contour parameter unresolved at binary64, level={level}")
        for solved_axis in (0,1):
            ids=np.flatnonzero(~split[:,solved_axis] & split[:,1-solved_axis])
            if not len(ids):continue
            axis=1-solved_axis
            first_high,last_low=upper[ids].copy(),lower[ids].copy()
            first_high[:,axis],last_low[:,axis]=middle[ids,axis],middle[ids,axis]
            child_low.extend((lower[ids],last_low));child_high.extend((first_high,upper[ids]))
        ids=np.flatnonzero(np.all(split,axis=1))
        if len(ids):
            a,b,m=lower[ids],upper[ids],middle[ids]
            child_low.extend((a,np.column_stack((m[:,0],a[:,1])),m,np.column_stack((a[:,0],m[:,1]))))
            child_high.extend((m,np.column_stack((b[:,0],m[:,1])),b,np.column_stack((m[:,0],b[:,1]))))
        lower,upper=np.concatenate(child_low),np.concatenate(child_high)
        if len(lower)>1000000:
            raise ValueError(f"physical contour critical-cell isolation did not contract inside native batch, "
                             f"level={level},depth={depth},first={lower[0].tolist()}")
        depth+=1
    if not starts:return terminal_paths
    starts,ends,branch_low,branch_high=(np.asarray(a)for a in(starts,ends,branch_low,branch_high))
    axes=np.asarray(branch_axes)
    directions=np.asarray(branch_directions)
    starts,ends,branch_low,branch_high,axes,directions=_refined_event_sections(
        field,level,starts,ends,branch_low,branch_high,axes,directions,ports,bounds)
    def sections(first,last,owner):
        points=(first[:,None,:]+(last-first)[:,None,:]*np.array((.25,.5,.75))[None,:,None]).reshape(-1,2)
        owner=np.repeat(owner,3);axis=axes[owner]
        for solved_axis in (0,1):
            ids=np.flatnonzero(axis==solved_axis)
            if not len(ids):continue
            fixed=points[ids,1-solved_axis]
            def residual(t,constant):
                return (field.sample_points(t,constant)if solved_axis==0
                        else field.sample_points(constant,t))-level
            result=elementwise.find_root(residual,(branch_low[owner[ids],solved_axis],
                                                   branch_high[owner[ids],solved_axis]),
                                         args=(fixed,),tolerances=_ROOT_TOLERANCES,maxiter=100)
            if not bool(np.all(result.success)):
                raise ValueError("certified physical branch sections must converge")
            points[ids,solved_axis]=result.x
        ids=np.flatnonzero(axis==2)
        if len(ids):
            selected=owner[ids]
            points[ids]=_direction_sections(field,level,points[ids],branch_low[selected],
                                            branch_high[selected],directions[selected])
        return points.reshape(-1,3,2)
    return terminal_paths+adaptive_curve_paths(starts,ends,sections)


def _resolve_branch_requests(field,generators,bounds):
    """One oracle batch for all simultaneous levels, with separate roots.

    Native source boxes, derivative certificates, branch choices and each
    height's shared-port encounter order stay unchanged. Only independent
    array queries are combined; hidden peaks still require the same proofs.
    """
    from .implicit_terrain_bounds import directional_bounds
    output=[None]*len(generators)
    pending=[]
    for index,generator in enumerate(generators):
        try:request=next(generator)
        except StopIteration as finished:output[index]=finished.value
        else:pending.append((index,generator,request))
    while pending:
        following=[]
        for kind in ("range","direction"):
            selected=[entry for entry in pending if entry[2][0]==kind]
            if not selected:continue
            lower=np.concatenate([entry[2][1]for entry in selected])
            upper=np.concatenate([entry[2][2]for entry in selected])
            if kind=="range":
                answers=bounds(field,lower,upper)
            else:
                directions=np.concatenate([np.broadcast_to(entry[2][3],(len(entry[2][1]),2,2))
                                           for entry in selected])
                answers=directional_bounds(field,lower,upper,directions)
            offset=0
            for index,generator,request in selected:
                stop=offset+len(request[1])
                answer=tuple(values[offset:stop]for values in answers)
                offset=stop
                try:next_request=generator.send(answer)
                except StopIteration as finished:output[index]=finished.value
                else:following.append((index,generator,next_request))
        pending=following
    return output


def _refined_curves(field,levels,lower,upper,coarse_low,coarse_high):
    """Share model certificates while each level retains its own true-port graph."""
    cuts=np.nextafter(levels,-np.inf)
    ports=[_SharedPorts(field,cut)for cut in cuts]
    paths=[[]for _ in levels]
    for begin in range(0,len(lower),2048):
        a,b=lower[begin:begin+2048],upper[begin:begin+2048]
        low,high=coarse_low[begin:begin+2048],coarse_high[begin:begin+2048]
        bounds=_TerrainBatchBounds(field)
        generators=[];owners=[]
        for index,(level,cut)in enumerate(zip(levels,cuts,strict=True)):
            selected=(low<=level)&(high>=level)
            if selected.any():
                generators.append(_refined_branch_requests(field,float(cut),a[selected],b[selected],ports[index],bounds))
                owners.append(index)
        for index,parts in zip(owners,_resolve_branch_requests(field,generators,bounds),strict=True):
            paths[index].extend(parts)
    return [_joined_paths(parts)for parts in paths]


def _refined_curve_boxes(field,levels,*,cells=None):
    """Prepare level-independent native source boxes once for all cuts."""
    if cells is not None:
        low,high=_selected_ranges(field,cells)
        lower,upper=_cell_boxes(field,cells)
    else:
        x,y=_native_axes(field)
        coarse_low,coarse_high=_coarse_ranges(field)
        needed=np.zeros_like(coarse_low,dtype=bool)
        for level in levels:needed|=(coarse_low<=level)&(coarse_high>=level)
        rows,columns=np.nonzero(needed)
        lower=np.column_stack((x[columns],y[rows]))
        upper=np.column_stack((x[columns+1],y[rows+1]))
        low,high=coarse_low[rows,columns],coarse_high[rows,columns]
    return lower,upper,low,high


def terrain_level_curves(field,metre_levels,*,query_bounds=None):
    """Return true curves for the supported saved physical models only."""
    levels=np.asarray(metre_levels,dtype=float)
    if levels.ndim!=1 or not np.all(np.isfinite(levels))or np.any(np.diff(levels)<=0):
        raise ValueError("physical contours require increasing finite metre levels")
    if not isinstance(field,(PhysicalTerrainField,RefinedTerrainField)):
        raise ValueError("physical contours require the accepted physical ground model")
    if (isinstance(field, RefinedTerrainField) and field._river_bed is not None
            and len(field._river_bed.segments)):
        raise ValueError("certified terrain contours do not support the continuous river-bed model")
    cells=_query_cells(field,query_bounds) if query_bounds is not None else None
    if isinstance(field,PhysicalTerrainField):
        return _base_level_curves(field,levels,cells)
    if not field._height.edge_count and not field._landforms.count:
        return _base_level_curves(field.base,levels,cells)
    lower,upper,low,high=_refined_curve_boxes(field,levels,cells=cells)
    if len(levels)>1 and len(lower)>2048:
        from .implicit_terrain_parallel import parallel_height_curves
        return parallel_height_curves(field,levels,lower,upper,low,high)
    return _refined_curves(field,levels,lower,upper,low,high)
