"""Export shared road, sea and port networks and world travel capabilities.

The browser consumes delivered road geometry. It never reconstructs a network
from SVG strokes, joins nearby banks, or invents a crossing facility.
"""

import json
from pathlib import Path

import numpy as np
import shapely

from .road_engineering import segment_lengths_km
from .society.transport import _technology_era
from .transport_geometry import _anchor_parts


def travel_profile(era, *, capabilities=()):
    if era not in {"tribal","ancient","medieval","early-modern","preindustrial","industrial","contemporary"}:
        raise ValueError('unsupported navigation era')
    modes=[dict(id='walk',label='步行',speedKmh=5,dailyHours=8,maximumGrade=.45,kind='road'),
           dict(id='horse',label='骑马',speedKmh=7,dailyHours=8,maximumGrade=.25,kind='road')]
    if era!='tribal':
        modes.append(dict(id='carriage',label='马车',speedKmh=6,dailyHours=8,maximumGrade=.10,kind='road'))
    ship = ('独木舟',4,8) if era=='tribal' else ('动力船',18,20) if era in {'industrial','contemporary'} else ('帆船',8,16)
    modes.append(dict(id='boat',label='步行＋'+ship[0],speedKmh=ship[1],dailyHours=ship[2],maximumGrade=.45,kind='ship'))
    if 'magic-flight' in capabilities:
        modes.append(dict(id='magic-flight',label='魔法飞行',speedKmh=40,dailyHours=3,maximumGrade=0,kind='flight',rarity='rare'))
    if 'dragon' in capabilities:
        modes.append(dict(id='dragon',label='骑龙',speedKmh=65,dailyHours=6,maximumGrade=0,kind='flight',rarity='rare'))
    flights=[f"{mode['label']}每日 {mode['speedKmh']*mode['dailyHours']} km"
             for mode in modes if mode['kind']=='flight']
    assumptions='地面交通每日行进 8 小时；步行计入爬升，马车避开超过 10% 的坡度。'
    assumptions+=f'船程按{ship[0]}平均 {ship[1]} km/h、每日航行 {ship[2]} 小时估算；在真实港口上下船各计 1 小时，陆路接驳按步行计算，途中可在船上休息。'
    if flights:
        assumptions+='，'.join(flights)+'；沿两点间球面最短航线飞行，每日额度用完后休息至次日，不要求中途经过城镇。'
    return dict(schema='world-travel-profile-v1',technologyEra=era,modes=modes,
        assumptions=assumptions+'交通速度为世界模拟设定。')


def _metrics(points, terrain_field, shape, radius):
    """Grade observations belong to the exact delivered curve, in model metres."""
    points=np.asarray(points,dtype=float)
    samples=[];boundaries=[0]
    for first,last in zip(points[:-1],points[1:],strict=True):
        count=max(1,int(np.ceil(np.max(np.abs(last-first))*64)))
        samples.append(first+np.arange(count)[:,None]/count*(last-first))
        boundaries.append(boundaries[-1]+count)
    sampled=np.vstack((*samples,points[-1:]))
    z=terrain_field.sample_points(sampled[:,0],sampled[:,1])
    lengths=segment_lengths_km(sampled,shape,radius)
    dz=np.diff(z)
    grades=np.divide(dz,lengths*1000,out=np.zeros_like(dz),where=lengths>1e-12)
    total=float(lengths.sum())
    hours={};time_segments={}
    for mode in ('walk','horse','carriage'):
        if mode=='walk':
            # Naismith/Langmuir: 5 km/h, +1h/600m ascent, steep downhill cost.
            first=lengths/5+np.maximum(dz,0)/600+np.where(grades<-.2125,np.maximum(-dz,0)/1800,0)
            last=lengths/5+np.maximum(-dz,0)/600+np.where(grades>.2125,np.maximum(dz,0)/1800,0)
        else:
            speed=7 if mode=='horse' else 6
            multiplier=1+np.abs(grades)/(.12 if mode=='horse' else .06)
            first=lengths*multiplier/speed
            last=first
        time_segments[mode]=(first,last)
        hours[mode]=[float(first.sum()),float(last.sum())]
    segments=[]
    for a,b in zip(boundaries[:-1],boundaries[1:],strict=True):
        segments.append([float(lengths[a:b].sum()),float(np.maximum(dz[a:b],0).sum()),
            float(np.maximum(-dz[a:b],0).sum()),float(np.max(abs(grades[a:b]),initial=0)),
            float(time_segments['walk'][0][a:b].sum()),float(time_segments['walk'][1][a:b].sum()),
            float(time_segments['horse'][0][a:b].sum()),float(time_segments['carriage'][0][a:b].sum())])
    return dict(distanceKm=total,ascentM=float(np.maximum(dz,0).sum()),descentM=float(np.maximum(-dz,0).sum()),
                maximumGrade=float(np.max(np.abs(grades),initial=0)),hours=hours,segments=segments)


def build_navigation(grid, society, road_geometry, *, terrain_field, settlement_locations, profile, sea_paths, land_surface, harbors):
    era=_technology_era(grid)
    if profile['schema']!='world-travel-profile-v1' or profile['technologyEra']!=era:
        raise ValueError('travel profile must match the saved world era')
    radius=float(grid.metadata['planet']['radiusKm'])
    anchors=shapely.STRtree([shapely.Point(column,row) for row,column in settlement_locations.values()])
    network=shapely.line_merge(shapely.union_all(shapely.get_parts(road_geometry)))
    nodes=[];indices={};edges=[]
    def key(point,kind):
        x,y=tuple(round(float(value),8) for value in point)
        return (kind, x%grid.shape[1] if kind=='sea' else x, y)
    def node(point,kind='road'):
        # This precision is the delivered transport codec, not proximity snapping.
        identity=key(point,kind)
        if identity not in indices:
            indices[identity]=len(nodes);nodes.append(list(identity[1:]))
        return indices[identity]
    for line in shapely.get_parts(network):
        if line.geom_type!='LineString':
            continue
        for points in _anchor_parts(line,anchors):
            if len(points)<2 or np.array_equal(points[0],points[-1]):
                continue
            a,b=node(points[0]),node(points[-1])
            edges.append(dict(kind='road',a=a,b=b,points=np.round(points,8).tolist(),
                              **_metrics(points,terrain_field,grid.shape,radius)))
    road_nodes={identifier:indices.get(key((column,row),'road'))
                for identifier,(row,column) in settlement_locations.items()}
    sea_lines=[shapely.LineString(points) for points in sea_paths if len(points)>=2]
    sea=shapely.line_merge(shapely.union_all(sea_lines).difference(land_surface))
    berths={identifier:(harbor['seaPoint']['column'],harbor['seaPoint']['row'])
            for identifier,harbor in harbors.items() if harbor['kind']=='sea'}
    berth_index=shapely.STRtree([shapely.Point(point) for point in berths.values()])
    for line in shapely.get_parts(sea):
        if line.geom_type!='LineString':
            continue
        for points in _anchor_parts(line,berth_index):
            if len(points)<2:
                continue
            lengths=segment_lengths_km(points,grid.shape,radius)
            edges.append(dict(kind='sea',a=node(points[0],'sea'),b=node(points[-1],'sea'),
                points=np.round(points,8).tolist(),distanceKm=float(lengths.sum()),ascentM=0.,descentM=0.,
                maximumGrade=0.,segments=[[float(length),0,0,0,0,0,0,0] for length in lengths]))
    port_nodes={}
    for identifier,berth in berths.items():
        sea_node=indices.get(key(berth,'sea'))
        if sea_node is None:
            continue
        harbor=harbors[identifier]
        row,column=settlement_locations[identifier]
        dry=[(column,row),*((p['column'],p['row']) for p in harbor['access'][1:]),
             (harbor['shore']['column'],harbor['shore']['row'])]
        dry=np.asarray(dry,dtype=float)
        dry=dry[np.r_[True,np.any(np.diff(dry,axis=0)!=0,axis=1)]]
        metrics=_metrics(dry,terrain_field,grid.shape,radius)
        berth_km=float(segment_lengths_km([dry[-1],berth],grid.shape,radius).sum())
        metrics['distanceKm']+=berth_km
        # The wet pier/berth span is covered by boarding, never by walking on water.
        metrics['segments'].append([berth_km,0,0,0,0,0,0,0])
        access=[*dry.tolist(),list(berth)]
        port_nodes[identifier]=node((column,row))
        edges.append(dict(kind='port',a=port_nodes[identifier],b=sea_node,points=access,
            port={'cityId':identifier,'name':harbor['name']},**metrics))
    cities=[]
    for city in society.settlements:
        row,column=settlement_locations[city.identifier]
        cities.append(dict(id=city.identifier,name=city.name,native=[column,row],node=road_nodes[city.identifier],
                           portNode=port_nodes.get(city.identifier)))
    return dict(schema='world-atlas-navigation-v3',gridDigest=grid.content_digest(),
                shape=list(grid.shape),radiusKm=radius,dayHours=float(grid.metadata['planet']['rotationPeriodHours']),
                profile=profile,nodes=nodes,edges=edges,cities=cities)


def write_navigation_assets(output, grid, society, road_geometry, *, terrain_field, settlement_locations, profile, sea_paths, land_surface, harbors):
    output=Path(output)
    result=build_navigation(grid,society,road_geometry,terrain_field=terrain_field,
                            settlement_locations=settlement_locations,profile=profile,sea_paths=sea_paths,
                            land_surface=land_surface,harbors=harbors)
    (output/'navigation-network.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    (output/'travel-profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result
