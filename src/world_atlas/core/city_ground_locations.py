"""Refine coarse settlement addresses on the shared continuous dry ground."""

import numpy as np
import shapely


def _bank_side(line, point):
    point=np.asarray(point,dtype=float)
    station=float(line.project(shapely.Point(point)))
    centre=np.asarray(line.interpolate(station).coords[0])
    before=np.asarray(line.interpolate(max(0.,station-.0001)).coords[0])
    after=np.asarray(line.interpolate(min(line.length,station+.0001)).coords[0])
    tangent=after-before
    offset=point-centre
    return float(tangent[0]*offset[1]-tangent[1]*offset[0])


def refine_city_ground_locations(grid,society,locations,terrain_field,*,
                                 land_surface,river_source_geometry,
                                 river_geometry,river_channel_geometry):
    """Keep each source address in its native cell and on its source river bank.

    Rivers can move within their continuous valley support past a raster cell
    centre. The address still owns that cell; its published centre must use
    the corresponding actual dry bank, rather than creating a new crossing.
    A centre on the native flow line has no specified source bank and chooses
    usable dry ground in that same cell.
    """
    source=shapely.get_parts(river_source_geometry)
    display=shapely.get_parts(river_geometry)
    if len(source)!=len(display):
        raise ValueError('settlement bank refinement requires shared river reach identities')
    movements=shapely.hausdorff_distance(source,display)
    source_index=shapely.STRtree(source)
    dry=land_surface.difference(river_channel_geometry)
    shapely.prepare(dry)
    origins=np.array([(city.column+.5,city.row+.5) for city in society.settlements])
    covered=shapely.covers(dry,shapely.points(origins))
    elevations=terrain_field.sample_points(origins[:,0],origins[:,1])
    reach_radius=float(np.max(movements,initial=0.))
    for city,origin,is_dry,elevation in zip(society.settlements,origins,covered,elevations,strict=True):
        neighbours=source_index.query(shapely.Point(origin),predicate='dwithin',distance=reach_radius)
        constraints=[]
        for index in neighbours:
            if source[index].distance(shapely.Point(origin))>movements[index]+1e-8:
                continue
            station=float(source[index].project(shapely.Point(origin)))
            if station<=1e-8 or station>=source[index].length-1e-8:
                # A river endpoint's ray does not divide the dry plane into
                # two banks. Do not extend it into an artificial barrier.
                continue
            side=_bank_side(source[index],origin)
            if abs(side)>1e-10:
                constraints.append((int(index),side))
        def matches_bank(point):
            return all(side*_bank_side(display[index],point)>0 for index,side in constraints)
        if is_dry and elevation>0 and matches_bank(origin):
            locations[city.identifier]=(float(origin[1]),float(origin[0]))
            continue
        cell=shapely.box(city.column,city.row,city.column+1,city.row+1)
        pieces=[part for part in shapely.get_parts(dry.intersection(cell))
                if part.geom_type=='Polygon' and not part.is_empty]
        candidates=[np.asarray(part.representative_point().coords[0]) for part in pieces]
        fractions=np.array((.125,.375,.625,.875))
        x,y=np.meshgrid(city.column+fractions,city.row+fractions)
        candidates.extend(np.column_stack((x.ravel(),y.ravel())))
        candidates=np.asarray(candidates,dtype=float)
        valid=shapely.covers(dry,shapely.points(candidates))
        valid &= terrain_field.sample_points(candidates[:,0],candidates[:,1])>0
        choices=[point for point,usable in zip(candidates,valid,strict=True)
                 if usable and matches_bank(point)]
        if not choices:
            raise ValueError(f'{city.identifier}: source cell has no dry ground on its corresponding river bank')
        anchor=min(choices,key=lambda point:float(np.sum((point-origin)**2)))
        locations[city.identifier]=(float(anchor[1]),float(anchor[0]))
    return locations
