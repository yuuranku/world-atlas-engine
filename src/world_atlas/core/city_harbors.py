"""Harbor sites on the world's continuous shoreline, shared by both maps."""

import math

import numpy as np
import shapely
from dataclasses import replace


def derive_harbors(grid, society, locations, terrain_field, *, road_surface):
    """Choose a shore reachable from the city on the shared dry road surface."""
    shapely.prepare(road_surface)
    result = {}
    for city in society.settlements:
        if city.site_type not in {'port', 'island-port', 'lake-port'}:
            continue
        row, column = locations[city.identifier]
        radius = float(grid.metadata['planet']['radiusKm'])
        row_km = math.pi*radius/grid.shape[0]
        column_km = 2*math.pi*radius/grid.shape[1] * max(.08, math.cos(math.radians(90-row/grid.shape[0]*180)))
        angles = np.linspace(0., 2*np.pi, 128, endpoint=False)
        distances = np.linspace(0., row_km*2.5, 129)
        x = column + np.cos(angles[:, None])*distances/column_km
        y = row + np.sin(angles[:, None])*distances/row_km
        inside = (y>=0)&(y<=grid.shape[0])
        heights = terrain_field.sample_points(x, np.clip(y,0,grid.shape[0]))
        wet = (heights <= 0)&inside
        # Only the first shoreline encountered from the dry city is eligible.
        # A ray cannot leap over an island to a second, unrelated coast.
        first = np.argmax(wet, axis=1)
        eligible = wet.any(axis=1) & (first > 0)
        if not eligible.any():
            raise ValueError(f'{city.identifier}: port has no reachable continuous shoreline')
        population=(city.population_min+city.population_max)/2
        harbor_scale=min(3., max(.6, .8*math.sqrt(population/15000)))
        # Compare nearby continuous shoreline sites by exposure as well as
        # distance. Land around the seaward half-circle shelters an inlet.
        options=np.flatnonzero(eligible)
        low_many=distances[first[options]-1].copy()
        high_many=distances[first[options]].copy()
        direction_many=np.column_stack((np.cos(angles[options])/column_km,np.sin(angles[options])/row_km))
        for _ in range(24):
            middle=(low_many+high_many)/2
            points=np.array([column,row])+direction_many*middle[:,None]
            dry=terrain_field.sample_points(points[:,0],points[:,1])>0
            low_many=np.where(dry,middle,low_many);high_many=np.where(dry,high_many,middle)
        origin=np.array([column,row])
        landings=origin+direction_many*np.maximum(0.,low_many-.07)[:,None]
        approaches=shapely.linestrings(np.stack((np.broadcast_to(origin,landings.shape),landings),axis=1))
        reachable=shapely.covers(road_surface,approaches)
        if not np.any(reachable):
            raise ValueError(f'{city.identifier}: port has no shore connected to its dry road approach')
        nearby=reachable & (high_many<=high_many[reachable].min()+harbor_scale*2)
        options,low_many,high_many,direction_many=(a[nearby] for a in (options,low_many,high_many,direction_many))
        coast_many=origin+direction_many*((low_many+high_many)/2)[:,None]
        probe_angles=angles[options,None]+np.linspace(-np.pi/2,np.pi/2,17)
        shelter_samples=[]
        for reach in (harbor_scale*.6,harbor_scale*1.2,harbor_scale*2):
            height=terrain_field.sample_points(coast_many[:,0,None]+np.cos(probe_angles)*reach/column_km,
                coast_many[:,1,None]+np.sin(probe_angles)*reach/row_km)
            shelter_samples.append((height>0).mean(axis=1))
        shelter=np.mean(shelter_samples,axis=0)
        score=high_many+harbor_scale*(1-shelter)*1.8
        choice=int(score.argmin());index=int(options[choice])
        low, high = distances[first[index]-1], distances[first[index]]
        direction = np.array([np.cos(angles[index])/column_km, np.sin(angles[index])/row_km])
        origin = np.array([column, row])
        for _ in range(24):
            middle = (low+high)/2
            p = origin+direction*middle
            if terrain_field.sample_points(p[0], p[1]) > 0:
                low = middle
            else:
                high = middle
        coast = origin+direction*(low+high)/2
        shore = origin+direction*max(0., low-.07)
        sea = origin+direction*(high+.12)
        # Native cells are many kilometres wide. Their edge must not keep an
        # ordinary port inland: refine its anchor within the same polity, on a
        # usable coastal terrace. A real cliff can still require an upper town.
        setback=min(2., max(.10, .65*math.sqrt(population/15000)))
        centre=origin
        for retreat in (setback,setback*1.25,setback*1.65):
            candidate=origin+direction*max(0., low-retreat)
            cy,cx=int(candidate[1]),int(candidate[0])%grid.shape[1]
            if not 0<=cy<grid.shape[0]:
                continue
            owner=society.politics.state_id[cy,cx]
            coastal_remnant=owner<=0 and grid.water[cy,cx]>0
            if owner!=society.politics.state_id[city.row,city.column] and not coastal_remnant:
                continue
            probe=.05
            elevations=terrain_field.sample_points(
                candidate[0]+np.array([0,probe,-probe,0,0])/column_km,
                candidate[1]+np.array([0,0,0,probe,-probe])/row_km)
            grade=np.hypot(elevations[1]-elevations[2],elevations[3]-elevations[4])/(probe*2000)
            if np.all(elevations>0) and grade<.10:
                centre=candidate
                break
        locations[city.identifier]=(float(centre[1]),float(centre[0]))
        prefix = city.name[:3]
        access = [centre, shore]
        result[city.identifier] = {
            'kind': 'lake' if city.site_type == 'lake-port' else 'sea',
            'name': prefix+('湖港' if city.site_type == 'lake-port' else '港'),
            'shore': {'column': float(coast[0]), 'row': float(coast[1])},
            'landPoint': {'column': float(shore[0]), 'row': float(shore[1])},
            'seaPoint': {'column': float(sea[0]), 'row': float(sea[1])},
            'outward': {'x': float(np.cos(angles[index])), 'y': float(np.sin(angles[index]))},
            'distanceKm': float(np.hypot((shore[0]-centre[0])*column_km, (shore[1]-centre[1])*row_km)),
            'access': [{'column': float(p[0]), 'row': float(p[1])} for p in access],
            'shorelineElevationMetres': float(terrain_field.sample_points(coast[0], coast[1])),
            'shelter':float(shelter[choice]),
            'setting':'sheltered-bay' if shelter[choice]>.35 else 'open-coast',
        }
    return result


def connect_harbor_routes(routes, harbors, terrain_field):
    """Join sea routes to real berths with a continuously checked wet approach."""
    result=[]
    for route in routes:
        points=list(route.path)
        if route.mode=='sea':
            for identifier,reverse in ((route.source_settlement_id,False),(route.target_settlement_id,True)):
                harbor=harbors.get(identifier)
                if not harbor or harbor['kind']!='sea':
                    continue
                values=points[::-1] if reverse else points
                berth=(harbor['seaPoint']['column'],harbor['seaPoint']['row'])
                for index,p in enumerate(values):
                    if terrain_field.sample_points(*p)>0:
                        continue
                    t=np.linspace(0.,1.,65)
                    x=berth[0]+(p[0]-berth[0])*t
                    y=berth[1]+(p[1]-berth[1])*t
                    if np.any(terrain_field.sample_points(x,y)>0):
                        continue
                    values=[berth]+values[index:]
                    sailing=[{'column':float(x),'row':float(y)} for x,y in values[:2]]
                    harbor['sailing']=sailing
                    break
                points=values[::-1] if reverse else values
        result.append(replace(route,path=tuple(points)))
    return tuple(result)
