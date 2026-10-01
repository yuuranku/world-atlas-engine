"""Persist the one accepted road/river drawing and its crossing evidence."""
import json
from pathlib import Path
import numpy as np
import shapely


def write_transport_sources(output, grid, prepared, *, river_geometry, channel_geometry):
    roads=shapely.MultiLineString([points for mode,_,points in prepared.paths if mode in {'road','rail'}])
    points=shapely.points(np.asarray(list(prepared.positions.values())).reshape(-1,2))
    road_errors=int(np.count_nonzero(shapely.distance(points,roads)>1e-7))
    river_errors=int(np.count_nonzero(shapely.distance(points,river_geometry)>1e-7))
    if road_errors or river_errors:
        raise ValueError('a saved bridge is detached from its final road or river')
    approaches=[]
    for route in prepared.routes:
        if route.mode not in {'road','rail'}:
            continue
        p=np.asarray(route.path,dtype=float)
        parts=np.split(p,np.flatnonzero(np.abs(np.diff(p[:,0]))>grid.shape[1]*.5)+1)
        geometry=shapely.MultiLineString([part for part in parts if len(part)>=2])
        approaches.append(dict(identifier=route.identifier,geometry=json.loads(shapely.to_geojson(geometry))))
    result=dict(schema='shared-road-river-crossings-v1',crossings=prepared.crossings,endpointTouches=prepared.endpoint_touches,
        bridges=[dict(identifier=b.identifier,routeIdentifier=b.route_identifier,position=prepared.positions[b.identifier],
                      tangent=prepared.tangents[b.identifier],bankSpan=prepared.spans[b.identifier]) for b in prepared.bridges],
        roadGeometry=json.loads(shapely.to_geojson(roads)),preparedRoadGeometry=approaches,
        drawnTransportPaths=[dict(mode=mode,importance=importance,
            geometry=json.loads(shapely.to_geojson(shapely.LineString(points)))) for mode,importance,points in prepared.paths],
        riverGeometry=json.loads(shapely.to_geojson(river_geometry)),channelGeometry=json.loads(shapely.to_geojson(channel_geometry)),
        bankBoundaryRoundoff=prepared.bank_roundoff,checks=dict(bridgesOffRoad=road_errors,bridgesOffRiver=river_errors))
    (Path(output)/'transport-crossings.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    return roads,result
