"""Shared subcell administrative faces from travel-time competitors.

Native points carry arrival times, rather than one-hot category indicators.
The planar lower envelope is built once and supplies both fill and border ink.
"""
from __future__ import annotations
import numpy as np
import math
import shapely
from ..raster_topology import _triangle_boundaries
from .administrative_front import AdministrativeFront
from .territorial_simulation import _NEIGHBORS


def _local_fields(front, identifiers, row_indices, column_indices):
    """Extend a missing competitor only by an evidenced neighbouring edge.

    A discarded third arrival cannot be lower than the accepted runner-up.
    This keeps all native winners while giving each shared vertex one common
    travel-time potential, independent of its incident polygon or triangle.
    Zero is the explicitly excluded administrative domain, not a town seed.
    """
    height,width=front.valid.shape
    rows=np.asarray(row_indices);columns=np.asarray(column_indices)%width
    rr,cc=np.meshgrid(rows,columns,indexing='ij')
    first_owner=front.owners[0,rr,cc]
    first_time=front.arrivals[0,rr,cc]
    second_time=front.arrivals[1,rr,cc]
    active=front.valid[rr,cc]&(first_owner>0)
    first_time=np.where(active,first_time,0.)
    # This finite value denotes absence; it exceeds every finite arrival plus
    # every legal one-cell traversal, and is never a geometric perturbation.
    maximum=front.absent_time
    result=[]
    for identifier in identifiers:
        if identifier==0:
            distance=np.full(rr.shape,maximum)
            for direction,(dy,dx,length) in enumerate(_NEIGHBORS):
                nr=np.clip(rr+dy,0,height-1);nc=(cc+dx)%width
                inside=(rr+dy>=0)&(rr+dy<height)
                excluded=inside&((front.owners[0,nr,nc]<=0)|~front.valid[nr,nc])
                distance=np.minimum(distance,np.where(excluded,front.edge_costs[direction,rr,cc]*length,maximum))
            result.append(np.where(active,distance,0.))
            continue
        known_first=first_owner==identifier
        known_second=front.owners[1,rr,cc]==identifier
        known=known_first|known_second
        reached=known_first.copy()
        original=np.where(known_first,front.arrivals[0,rr,cc],front.arrivals[1,rr,cc])
        extension=np.full(rr.shape,np.inf)
        excluded=np.full(rr.shape,np.inf)
        for direction,(dy,dx,distance) in enumerate(_NEIGHBORS):
            nr=np.clip(rr+dy,0,height-1);nc=(cc+dx)%width
            reachable=(rr+dy>=0)&(rr+dy<height)
            present_first=front.owners[0,nr,nc]==identifier
            present_second=front.owners[1,nr,nc]==identifier
            present=reachable&(present_first|present_second)
            reached|=reachable&present_first
            neighbour=np.where(present_first,front.arrivals[0,nr,nc],front.arrivals[1,nr,nc])
            edge=front.edge_costs[7-direction,nr,nc]*distance
            extension=np.minimum(extension,np.where(present,neighbour+edge,np.inf))
            excluded=np.minimum(excluded,np.where(present,edge,np.inf))
        extended=np.maximum(extension,np.where(np.isfinite(second_time),second_time,first_time))
        time=np.where(known,original,extended)
        margins=np.where(active,np.maximum(0.,time-first_time),excluded)
        if front.parent is not None:
            parent=front.parent[rr,cc]
            margins=np.where(active&(parent>0)&(parent!=front.domains[identifier]),maximum,margins)
        # A competitor's travel potential can cross occupied territory. Its
        # administrative front can only enter this interpolation slab from
        # an actual neighbouring winner. Remote runner-ups cannot materialise
        # as a detached country between two unrelated native observations.
        result.append(np.where(reached&np.isfinite(margins),margins,maximum))
    return np.asarray(result)


def _cell_samples(front, west, north, east, south):
    height,width=front.valid.shape
    # Half-cell exterior slabs repeat the nearest native observation. Longitude
    # periodicity remains in the metric; the exported planar frame has two sides.
    x=np.unique(np.r_[west,np.arange(max(0,math.ceil(west-.5)),min(width-1,math.floor(east-.5))+1)+.5,east])
    y=np.unique(np.r_[north,np.arange(max(0,math.ceil(north-.5)),min(height-1,math.floor(south-.5))+1)+.5,south])
    rows=np.clip(np.floor(y).astype(int),0,height-1)
    columns=np.clip(np.floor(x).astype(int),0,width-1)
    halo_rows=np.arange(max(0,int(rows.min())-1),min(height,int(rows.max())+2))
    halo_columns=np.arange(int(columns.min())-1,int(columns.max())+2)%width
    identifiers=np.unique(front.owners[:,halo_rows[:,None],halo_columns])
    identifiers=np.unique(np.r_[0,identifiers[identifiers>0]])
    values=_local_fields(front,identifiers,rows,columns)
    winners=np.argmin(values,axis=0)
    mixed=(winners[:-1,:-1]!=winners[:-1,1:])|(winners[:-1,:-1]!=winners[1:,1:])|(winners[:-1,:-1]!=winners[1:,:-1])
    row,column=np.nonzero(mixed)
    corner_rows=np.stack((row,row,row+1,row+1),axis=1)
    corner_columns=np.stack((column,column+1,column+1,column),axis=1)
    corners=np.stack((x[corner_columns],y[corner_rows]),axis=2)
    corner_values=values[:,corner_rows,corner_columns].transpose(1,0,2)
    points=np.concatenate((corners,corners.mean(axis=1)[:,None]),axis=1)
    # A common positive affine rescaling preserves the exact lower envelope;
    # dimensionless unit scores avoid cancellation of large travel times in
    # the generic shared-envelope clipping primitive.
    scores=1-np.concatenate((corner_values,corner_values.mean(axis=2)[:,:,None]),axis=2)/(front.absent_time+1)
    fan=np.array(((0,1,4),(1,2,4),(2,3,4),(3,0,4)))
    segments=_triangle_boundaries(points[:,fan].reshape(-1,3,2),
                  scores[:,:,fan].transpose(0,2,1,3).reshape(-1,len(identifiers),3))
    return segments


def _face_owner(front, face):
    point=face.representative_point();height,width=front.valid.shape
    west=max(0.,np.floor(point.x-.5)+.5);east=min(float(width),west+1.)
    north=max(0.,np.floor(point.y-.5)+.5);south=min(float(height),north+1.)
    if point.x<.5:west,east=0.,.5
    if point.x>width-.5:west,east=width-.5,float(width)
    if point.y<.5:north,south=0.,.5
    if point.y>height-.5:north,south=height-.5,float(height)
    rows=np.clip(np.floor((north,south)).astype(int),0,height-1)
    cols=np.clip(np.floor((west,east)).astype(int),0,width-1)
    hr=np.arange(max(0,rows.min()-1),min(height,rows.max()+2))
    hc=np.arange(cols.min()-1,cols.max()+2)%width
    ids=np.unique(np.r_[0,front.owners[:,hr[:,None],hc].ravel()])
    ids=ids[ids>=0]
    values=_local_fields(front,ids,rows,cols)[:,(0,0,1,1),(0,1,1,0)]
    sx=(point.x-west)/(east-west);sy=(point.y-north)/(south-north)
    distances=(sy,1-sx,1-sy,sx);edge=int(np.argmin(distances))
    first,last=((0,1),(1,2),(2,3),(3,0))[edge]
    centre_weight=2*distances[edge];along=(sx,sy,1-sx,1-sy)[edge]
    last_weight=along-centre_weight*.5;first_weight=1-centre_weight-last_weight
    return int(ids[np.argmin(values[:,first]*first_weight+values[:,last]*last_weight+values.mean(axis=1)*centre_weight)])


def administrative_coverage(front: AdministrativeFront):
    """Return full-frame atomic polygons and their administrative owner IDs."""
    height,width=front.valid.shape
    parts=[]
    # Every block boundary is a native midpoint knot, not an extra observation.
    x=np.r_[0.,np.arange(63.5,width,64),float(width)]
    y=np.r_[0.,np.arange(63.5,height,64),float(height)]
    for north,south in zip(y,y[1:]):
        for west,east in zip(x,x[1:]):
            segments=_cell_samples(front,west,north,east,south)
            if len(segments):parts.append(segments)
    frame=shapely.box(0,0,width,height)
    if not parts:return (frame,),np.array([_face_owner(front,frame)],dtype=np.int32)
    graph=shapely.union_all([*shapely.linestrings(np.concatenate(parts)),frame.boundary],grid_size=1e-8)
    polygons,cuts,dangles,invalid=shapely.polygonize_full(shapely.get_parts(graph))
    if not cuts.is_empty or not dangles.is_empty or not invalid.is_empty:
        raise ValueError('administrative arrival graph has unresolved shared edges')
    faces=tuple(shapely.set_precision(shapely.get_parts(polygons),0.))
    labels=np.array([_face_owner(front,face) for face in faces],dtype=np.int32)
    if not shapely.coverage_is_valid(faces) or not np.all(shapely.is_valid(faces)):
        raise ValueError('administrative arrival graph must be one valid shared coverage')
    return faces,labels
