"""Harbor sites on the world's continuous shoreline, shared by both maps."""

import math

import numpy as np
import shapely
from dataclasses import replace
from .polygon_navigation import PolygonNavigator


def derive_harbors(grid, society, locations, terrain_field, *, road_surface, land_surface):
    """Choose a shore reachable from the city on the shared dry road surface."""
    shapely.prepare(road_surface)
    sailing_obstacle=land_surface.buffer(1e-6)
    shapely.prepare(sailing_obstacle)
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
        # The source cell is a coarse site address, not a fixed town centre.
        # Choose a usable coastal terrace and its dock on the same actual
        # bank before fixing the canonical display city location.
        local=road_surface.intersection(shapely.box(
            column-distances[-1]/column_km,row-distances[-1]/row_km,
            column+distances[-1]/column_km,row+distances[-1]/row_km))
        parts=shapely.get_parts(local)
        parts=np.asarray([part for part in parts if part.geom_type=='Polygon'],dtype=object)
        banks=shapely.covers(parts[:,None],shapely.points(landings)[None,:])
        bank_indices=np.argmax(banks,axis=0)
        reachable=banks.any(axis=0)
        seaward=origin+direction_many*(high_many+.12)[:,None]
        reachable &= ~shapely.covers(sailing_obstacle,shapely.points(seaward))
        setback=min(2., max(.10, .65*math.sqrt(population/15000)))
        terraces=np.full_like(landings,np.nan)
        owner=society.politics.state_id[city.row,city.column]
        for index in np.flatnonzero(reachable):
            bank=parts[int(bank_indices[index])]
            for retreat in (setback,setback*1.25,setback*1.65):
                candidate=landings[index]-direction_many[index]*retreat
                cy,cx=int(candidate[1]),int(candidate[0])%grid.shape[1]
                if not 0<=cy<grid.shape[0]:
                    continue
                candidate_owner=society.politics.state_id[cy,cx]
                coastal_remnant=(candidate_owner<=0 and grid.water[cy,cx]>0 and
                    abs(cy-city.row)<=1 and abs(cx-city.column)<=1)
                if candidate_owner!=owner and not coastal_remnant:
                    continue
                probe=.05
                elevations=terrain_field.sample_points(
                    candidate[0]+np.array([0,probe,-probe,0,0])/column_km,
                    candidate[1]+np.array([0,0,0,probe,-probe])/row_km)
                grade=np.hypot(elevations[1]-elevations[2],elevations[3]-elevations[4])/(probe*2000)
                if np.all(elevations>0) and grade<.10 and bank.covers(
                        shapely.LineString((candidate,landings[index]))):
                    terraces[index]=candidate
                    break
            # A genuine cliff port can retain its upper town. Its approach
            # must still fit the local dry bank, including any necessary turn.
            if not np.isfinite(terraces[index]).all() and bank.covers(shapely.Point(origin)):
                terraces[index]=origin
        reachable &= np.isfinite(terraces).all(axis=1)
        original_banks=shapely.covers(parts,shapely.Point(origin))
        on_original_bank=original_banks[bank_indices]
        if np.any(reachable & on_original_bank):
            # Preserve the bank served by the source city's licensed roads
            # whenever that bank offers a real coastal terrace of its own.
            reachable &= on_original_bank
        if not np.any(reachable):
            raise ValueError(f'{city.identifier}: port has no coastal terrace connected to a real shore')
        nearby=reachable & (high_many<=high_many[reachable].min()+harbor_scale*2)
        terraces,bank_indices=terraces[nearby],bank_indices[nearby]
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
        centre=terraces[choice]
        bank=parts[int(bank_indices[choice])]
        access=PolygonNavigator(bank).path(centre,shore)
        metric=np.array([column_km,row_km])
        metric_access=shapely.LineString(access*metric)
        locations[city.identifier]=(float(centre[1]),float(centre[0]))
        prefix = city.name[:3]
        result[city.identifier] = {
            'kind': 'lake' if city.site_type == 'lake-port' else 'sea',
            'name': prefix+('湖港' if city.site_type == 'lake-port' else '港'),
            'shore': {'column': float(coast[0]), 'row': float(coast[1])},
            'landPoint': {'column': float(shore[0]), 'row': float(shore[1])},
            'seaPoint': {'column': float(sea[0]), 'row': float(sea[1])},
            'outward': {'x': float(np.cos(angles[index])), 'y': float(np.sin(angles[index]))},
            'distanceKm': float(metric_access.length),
            'access': [{'column': float(p[0]), 'row': float(p[1])} for p in access],
            'shorelineElevationMetres': float(terrain_field.sample_points(coast[0], coast[1])),
            'shelter':float(shelter[choice]),
            'setting':'sheltered-bay' if shelter[choice]>.35 else 'open-coast',
        }
    return result


def _sea_corridor(grid, points, sea_body, margin):
    """Use the coarse voyage for support, and the actual shore for eligibility."""
    from .society.transport import _path_cells

    cells=set(_path_cells(tuple(map(tuple,points)),grid.shape))
    bounds=shapely.union_all([shapely.box(x,y,x+1,y+1) for y,x in sorted(cells)])
    return bounds.buffer(margin,join_style='mitre').intersection(sea_body)


def connect_harbor_routes(routes, harbors, grid, *, land_surface):
    """Navigate whole sea routes around real headlands and islands to berths."""
    from .polygon_navigation import polygon_path
    from .transport_geometry import _split_sailing_seam

    # Offshore clearance is larger than navigation's eight-decimal coordinate
    # serialization error. It changes the sailing domain, never the shore.
    sea_obstacles=land_surface.buffer(1e-6)
    sea_bodies=shapely.get_parts(shapely.box(0,0,grid.shape[1],grid.shape[0]).difference(sea_obstacles))
    sea_index=shapely.STRtree(sea_bodies)
    result=[]
    for route in routes:
        if route.mode!='sea':
            result.append(route)
            continue
        points=list(route.path)
        for identifier,index in ((route.source_settlement_id,0),(route.target_settlement_id,-1)):
            harbor=harbors.get(identifier)
            if harbor is None or harbor['kind'] not in {'sea','lake'}:
                raise ValueError(f'{route.identifier}: sailing endpoint has no real berth')
            berth=(harbor['seaPoint']['column'],harbor['seaPoint']['row'])
            if index==0:points.insert(0,berth)
            else:points.append(berth)
        parts=[]
        for part in _split_sailing_seam(points,grid.shape[1]):
            owners=set(sea_index.query(shapely.Point(part[0]),predicate='covered_by')) & set(
                sea_index.query(shapely.Point(part[-1]),predicate='covered_by'))
            if len(owners)!=1:
                raise ValueError(f'{route.identifier}: sailing berths belong to disconnected water bodies')
            sea_body=sea_bodies[next(iter(owners))]
            anchors=shapely.MultiPoint((part[0],part[-1]))
            margin=1
            while True:
                corridor=_sea_corridor(grid,part,sea_body,margin)
                connected=[polygon for polygon in shapely.get_parts(corridor)
                           if polygon.geom_type=='Polygon' and polygon.covers(anchors)]
                if len(connected)==1:break
                if margin>=max(grid.shape):
                    raise ValueError(f'{route.identifier}: sailing corridor does not connect its real berths')
                margin=min(max(grid.shape),margin*2)
            path=polygon_path(connected[0],part[0],part[-1])
            if shapely.LineString(path).intersects(land_surface):
                raise ValueError(f'{route.identifier}: sailing route enters continuous land')
            parts.extend(map(tuple,path))
        for identifier,values in ((route.source_settlement_id,parts),(route.target_settlement_id,parts[::-1])):
            harbors[identifier]['sailing']=[{'column':float(x),'row':float(y)} for x,y in values[:2]]
        result.append(replace(route,path=tuple(parts)))
    return tuple(result)
