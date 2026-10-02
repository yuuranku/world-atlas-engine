"""Shared physical road and sailing geometry from accepted simulation routes.

Logical connections remain in TransportLayers. Atlas ink is drawn once per
shared corridor, with city, bridge and junction anchors retained exactly.
Turns use one sampled curve checked against the shared land and water faces;
straight routes never acquire decorative waves.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
import shapely
from shapely.ops import substring
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from .polygon_navigation import polygon_path

from .river_crossings import (
    endpoint_bank_direction, river_bank_owner as _river_bank_owner,
    source_crossing_points, source_crossing_transitions,
)


_RANK = {"local": 0, "regional": 1, "trunk": 2}


def native_river_geometry(grid, *, raw_elevation_m):
    """The actual recorded flow edges, rather than a river-cell mask."""
    active = np.asarray(grid.river_order).ravel() > 0
    source = np.flatnonzero(active)
    from .river_network import hydrologic_outlet_targets
    outlets = hydrologic_outlet_targets(grid,raw_elevation_m)
    target = np.asarray(grid.flow_to).ravel()[source].copy()
    for index,cell in enumerate(source):
        if int(cell) in outlets:
            target[index]=outlets[int(cell)]
    present = (target >= 0) & (target < active.size)
    source, target = source[present], target[present]
    present = active[target] | (np.asarray(grid.water).ravel()[target] > 0)
    source, target = source[present], target[present]
    width = grid.shape[1]
    first = np.column_stack((source % width + .5, source // width + .5))
    last = np.column_stack((target % width + .5, target // width + .5))
    ordinary = np.abs(first[:, 0] - last[:, 0]) <= width / 2
    return shapely.line_merge(shapely.multilinestrings(
        shapely.linestrings(np.stack((first[ordinary], last[ordinary]), axis=1))))


def _point_parts(geometry):
    if geometry.geom_type == "Point":
        return (geometry,)
    if hasattr(geometry, "geoms"):
        return tuple(point for part in geometry.geoms for point in _point_parts(part))
    return ()


def source_transport_crossings(grid, routes, river_geometry, *, raw_elevation_m):
    """Yield source road, point, native river cell and order for real crossings."""
    from .river_network import hydrologic_outlet_targets
    outlets=hydrologic_outlet_targets(grid,raw_elevation_m)
    parts = shapely.get_parts(river_geometry)
    index = shapely.STRtree(parts)
    endpoint_stars={}
    def support(point):
        row,column=int(np.floor(point.y)),int(np.floor(point.x))
        crossed=[]
        for y in range(max(0,row-1),min(grid.shape[0],row+2)):
            for x in range(column-1,column+2):
                cell=(y,x%grid.shape[1])
                if grid.river_order[cell]<=0:
                    continue
                source_cell=cell[0]*grid.shape[1]+cell[1]
                target=outlets.get(source_cell,int(grid.flow_to[cell]))
                if not 0<=target<grid.shape[0]*grid.shape[1]:
                    continue
                last_row,last_column=divmod(target,grid.shape[1])
                edge=shapely.LineString(((cell[1]+.5,cell[0]+.5),(last_column+.5,last_row+.5)))
                if edge.distance(point)<1e-7:
                    crossed.append(cell)
        if not crossed:
            raise ValueError("a crossing has no recorded native flow edge")
        cell=min(crossed,key=lambda cell:(-int(grid.river_order[cell]),point.distance(shapely.Point(cell[1]+.5,cell[0]+.5)),cell))
        return cell,int(grid.river_order[cell])
    for route in routes:
        if route.mode not in {"road", "rail"}:
            continue
        for points in _split_seam(route.path, grid.shape[1]):
            road = shapely.LineString(points)
            matches = index.query(road, predicate="intersects")
            if not len(matches):
                continue
            river = shapely.union_all(parts[matches])
            for distance,sample_distance in ((0.,.025),(road.length,max(0.,road.length-.025))):
                point=road.interpolate(distance)
                if point.distance(river)<1e-8:
                    endpoint_stars.setdefault((round(point.x,7),round(point.y,7)),[]).append((route,point,road.interpolate(sample_distance)))
            for point in source_crossing_points(road, river):
                cell,order=support(point)
                yield route,point,cell,order
    for entries in endpoint_stars.values():
        point=entries[0][1]
        nearby=index.query(point.buffer(.1),predicate="intersects")
        river=shapely.union_all(parts[nearby])
        owners=_river_bank_owner(river,point,[entry[2]for entry in entries])
        if len({owner for owner in owners if owner is not None})>=2:
            cell,order=support(point)
            for route,_point,_sample in entries:
                yield route,point,cell,order


def _river_navigation_support(grid, routes):
    from .society.transport import _path_cells

    supported = {}
    for route in routes:
        if route.mode != "river":
            continue
        for row, column in _path_cells(route.path, grid.shape):
            if grid.water[row, column] == 0 and grid.river_order[row, column] > 0:
                supported.setdefault((row, column), set()).add(route.identifier)
    return supported


def river_navigation_attributes(grid, source_paths, routes):
    """Attach navigation to physical reaches, without drawing another river.

    The transport network retains every logical route. A reach is styled as
    navigable only when its entire represented native river support belongs
    to that network; touching or crossing a single river cell is insufficient.
    Partial support is retained as route metadata, without extending a route
    designation beyond its accepted cells.
    """
    supported = _river_navigation_support(grid, routes)
    attributes = []
    height, width = grid.shape
    for path in source_paths:
        cells = {
            (min(height-1,max(0,int(np.floor(y)))), int(np.floor(x)) % width)
            for x, y in path
        }
        cells = {cell for cell in cells if grid.water[cell] == 0 and grid.river_order[cell] > 0}
        matched = cells.intersection(supported)
        if not matched:
            attributes.append({})
            continue
        identifiers = sorted({identifier for cell in matched for identifier in supported[cell]})
        attributes.append({"navigation-route-ids": ",".join(identifiers),
                           "navigation-cell-count": str(len(matched)),
                           "navigation-support-cell-count": str(len(cells)),
                           "navigable": str(matched == cells).lower()})
    return attributes


def river_navigation_segments(grid, source_paths, curve_paths, routes):
    """Divide the existing readable line into contiguous navigation sections.

    All original sampled edges occur exactly once. Colours are attributes of
    that one physical line, not another independently smoothed transport path.
    Projecting each edge midpoint back to the native flow support prevents a
    rounded physical bend from borrowing navigation from an adjacent valley.
    """
    supported = _river_navigation_support(grid, routes)
    height, width = grid.shape
    result = []
    for source, curve in zip(source_paths, curve_paths, strict=True):
        curve = np.asarray(curve)
        midpoints = (curve[:-1] + curve[1:]) * .5
        physical = shapely.LineString(source)
        positions = shapely.line_locate_point(physical, shapely.points(midpoints))
        feet = shapely.get_coordinates(shapely.line_interpolate_point(physical, positions))
        owners = [supported.get((min(height-1,max(0,int(np.floor(y)))),int(np.floor(x)) % width),set())
                  for x,y in feet]
        flags = [bool(identifiers) for identifiers in owners]
        sections = []
        start = 0
        while start < len(flags):
            end = start + 1
            while end < len(flags) and flags[end] == flags[start]:
                end += 1
            identifiers = sorted({identifier for support in owners[start:end] for identifier in support})
            sections.append((curve[start:end+1],{"navigable":str(flags[start]).lower(),
                              "navigation-route-ids":",".join(identifiers)}))
            start = end
        result.append(sections)
    return result


def _parts(geometry):
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "LineString":
        return (geometry,)
    if hasattr(geometry,"geoms"):
        return tuple(line for part in geometry.geoms for line in _parts(part))
    return ()


def _split_seam(points, width):
    source = np.asarray(points, dtype=np.float64)
    breaks = np.flatnonzero(np.abs(np.diff(source[:, 0])) > width * .5) + 1
    return tuple(part for part in np.split(source, breaks) if len(part) > 1)


def _split_sailing_seam(points, width):
    """Keep both ends of a wrapped sea edge at the map's date-line portals."""
    source=np.asarray(points,dtype=float)
    parts=[];current=[source[0]]
    for first,last in zip(source[:-1],source[1:],strict=True):
        delta=last[0]-first[0]
        if abs(delta)>width*.5:
            wrapped=last[0]+(width if delta<0 else -width)
            edge=float(width if delta<0 else 0)
            fraction=(edge-first[0])/(wrapped-first[0]) if wrapped!=first[0] else 0.
            latitude=float(first[1]+fraction*(last[1]-first[1]))
            current.append(np.array([edge,latitude]))
            parts.append(np.asarray(current))
            current=[np.array([width-edge,latitude])]
        current.append(last)
    parts.append(np.asarray(current))
    return tuple(part for part in parts if len(part)>1 and np.any(np.diff(part,axis=0)!=0))


def _river_safe_line(line, river_geometry):
    intersection = shapely.intersection(line, river_geometry)
    if intersection.is_empty:
        return True
    points = _point_parts(intersection)
    if intersection.geom_type not in {"Point", "MultiPoint"}:
        return False
    ends = (shapely.Point(line.coords[0]), shapely.Point(line.coords[-1]))
    return all(min(point.distance(end) for end in ends) < 1e-7 for point in points)


def _simplify(points, mode, land_geometry, river_geometry, *, grade_check=None):
    """Replace redundant navigation steps with visible, terrain-safe chords."""

    if len(points) <= 2:
        return points.copy()
    chosen = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        selected = anchor + 1
        for candidate in range(len(points) - 1, anchor + 1, -1):
            chord = shapely.LineString((points[anchor], points[candidate]))
            if mode == "sea":
                safe = not shapely.intersects(land_geometry, chord)
            else:
                safe = shapely.covers(land_geometry, chord)
                if safe:
                    # A road must continue following its accepted valley,
                    # rather than cutting across it to obtain a shorter line.
                    corridor = shapely.LineString(points[anchor:candidate + 1]).buffer(.35)
                    safe = shapely.covers(corridor, chord)
                    safe &= _river_safe_line(chord, river_geometry)
                    if safe and grade_check is not None:
                        safe = grade_check(points[anchor:candidate+1], points[[anchor,candidate]])
            if safe:
                selected = candidate
                break
        chosen.append(points[selected])
        anchor = selected
    return np.asarray(chosen)


def _anchor_parts(line, anchors):
    """Split a chain at fixed route nodes and projected bridge crossings."""

    coordinates = np.asarray(line.coords)
    deltas = np.diff(coordinates, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    if not anchors:
        return (coordinates,)
    distances = [0., float(line.length)]
    candidates = anchors.query(line, predicate="dwithin", distance=1e-8)
    for index in candidates:
        point = anchors.geometries[index]
        distance = float(shapely.line_locate_point(line, point))
        if 0. < distance < line.length:
            distances.append(distance)
    distances = np.unique(np.asarray(distances))
    result = []
    for start, end in zip(distances[:-1], distances[1:]):
        if end <= start:
            continue
        interior = coordinates[(cumulative > start) & (cumulative < end)]
        first = np.asarray(shapely.line_interpolate_point(line, start).coords[0])
        last = np.asarray(shapely.line_interpolate_point(line, end).coords[0])
        part=np.vstack((first,interior,last))
        part=part[np.r_[True,np.any(np.diff(part,axis=0)!=0.,axis=1)]]
        if len(part)>1:
            result.append(part)
    return tuple(result)


def _network_paths(lines, importances, fixed):
    """Extract attributed chains from one shared network."""

    source_index = shapely.STRtree(lines)
    # Union and linear interpolation can introduce floating-point vertices
    # a few ulps off their source segment. A subpixel tolerance identifies
    # its owning route without treating a transverse crossing as overlap.
    source_corridors = shapely.buffer(lines, 1e-7)
    anchor_index = shapely.STRtree([shapely.Point(point) for point in sorted(fixed)])
    network = shapely.line_merge(shapely.union_all(lines))
    result = []
    for line in _parts(network):
        for points in _anchor_parts(line, anchor_index):
            piece = shapely.LineString(points)
            covering = [
                index for index in source_index.query(piece.buffer(1e-7), predicate="intersects")
                if shapely.covers(source_corridors[index], piece)
            ]
            if not covering:
                raise ValueError(f"network segment has no source route: {piece.wkt}")
            importance = max((importances[index] for index in covering), key=_RANK.__getitem__)
            result.append((importance, points))
    return tuple(result)


def shared_transport_paths(grid, routes: Iterable, *, land_geometry, road_surface, river_geometry, anchors, grade_check=None, engineer_path=None):
    """Return ``(mode, importance, points)`` for unique road/sea ink chains.

    ``land_geometry`` is the same continuous land surface used by atlas fills
    and shorelines. ``anchors`` contains the displayed city centres and bridge
    locations in (column, row) coordinates. Route endpoints and all true
    junctions are fixed as well; physical routes and input arrays are unchanged.
    """

    shapely.prepare(land_geometry)
    routes = tuple(routes)
    result = []
    for mode in ("road", "rail", "sea"):
        lines, importances = [], []
        fixed = set(tuple(map(float, point)) for point in anchors)
        for route in routes:
            if route.mode != mode:
                continue
            split=_split_sailing_seam if mode=='sea' else _split_seam
            for points in split(route.path, grid.shape[1]):
                line = shapely.LineString(points)
                if line.length < 1e-9:
                    continue
                lines.append(line)
                importances.append(route.importance)
                fixed.update((tuple(points[0]), tuple(points[-1])))
        if not lines:
            continue
        # Union nodes crossings and shared prefixes before any simplification.
        # A junction therefore belongs to every incident curve exactly once.
        simplified, strengths = [], []
        for importance, points in _network_paths(lines, importances, fixed):
            navigable=land_geometry if mode=="sea" else road_surface
            if mode == 'road' and engineer_path is not None:
                points = engineer_path(points)
            simplified.append(shapely.LineString(_simplify(points, mode, navigable, river_geometry,
                grade_check=grade_check if mode=='road' else None)))
            strengths.append(importance)
            fixed.update((tuple(points[0]), tuple(points[-1])))
        # Redundant sailing bends can simplify onto the very same chord.
        # Node and deduplicate the final geometry as well as the source grid.
        result.extend((mode, importance, points)
                      for importance, points in _network_paths(simplified, strengths, fixed))
    return tuple(result)


def _curved_transport_points(points, mode, land_geometry, river_geometry, *, grade_check=None):
    """Sample one checked curve; the same points own SVG and crossing audits."""
    values = np.asarray(points, dtype=np.float64)
    result = [values[0]]
    for index in range(1, len(values)-1):
        corner = values[index]
        incoming, outgoing = values[index-1]-corner, values[index+1]-corner
        before, after = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        if min(before, after) < 1e-9:
            # Overlay noding can retain a tiny edge next to a real bank turn.
            # Removing both of its corners connects the previous approach
            # straight to the next one and cuts through the water obstacle.
            result.append(corner)
            continue
        radius = min(1.5 if mode == "sea" else .45, before*.35, after*.35)
        for _attempt in range(8):
            entry, exit_point = corner+incoming/before*radius, corner+outgoing/after*radius
            triangle = shapely.Polygon((entry, corner, exit_point))
            safe = (not shapely.intersects(land_geometry, triangle) if mode == "sea"
                    else shapely.covers(land_geometry, triangle)
                    and not shapely.intersects(river_geometry, triangle))
            if safe:
                fraction = np.linspace(0., 1., 9)[1:, None]
                curve = np.vstack((entry,(1-fraction)**2*entry + 2*(1-fraction)*fraction*corner + fraction**2*exit_point))
                if grade_check is None or grade_check(np.vstack((entry,corner,exit_point)),curve):
                    result.extend(curve)
                    break
            radius *= .5
        else:
            result.append(corner)
    result.append(values[-1])
    result = np.asarray(result)
    return result[np.r_[True, np.any(np.diff(result, axis=0)!=0.,axis=1)]]


def _road_native_corridor(grid, points, land_surface, *, endpoint_access):
    """Reconstruct the cells admitted by the original wrapped D8 road walk.

    Source path compression removes intermediate centre stations. An ink
    buffer around that compressed chord is not the native navigable domain.
    Diagonal steps require both orthogonal land cells in the source router;
    retain that same corner access before clipping by actual physical ground.
    These cells constrain path finding only and are never painted as boxes.
    """
    from .society.transport import _path_cells
    def walk_cells(walk):
        path=_path_cells(tuple(map(tuple,walk)),grid.shape)
        cells=set(path)
        for first,last in zip(path[:-1],path[1:]):
            if first[0]!=last[0] and first[1]!=last[1]:
                cells.update(((first[0],last[1]),(last[0],first[1])))
        # Floor sampling can record one orthogonal cell between two diagonal
        # cells. Retain both sides of that same grid-corner transition.
        for first,last in zip(path[:-2],path[2:]):
            if abs(first[0]-last[0])==1 and abs(first[1]-last[1])==1:
                cells.update(((first[0],last[1]),(last[0],first[1])))
        return cells
    cells={cell for cell in walk_cells(points) if grid.water[cell]==0}
    # Refined coastal anchors may lie on real dry ground within a coarse
    # water cell. Admit only the explicit settlement access walk there, then
    # clip it to the same physical land as the accepted native corridor.
    for access in endpoint_access:
        for part in _split_seam(access,grid.shape[1]):
            cells.update(walk_cells(part))
    cells=sorted(cells)
    if not cells:
        raise ValueError("an accepted road has no native land corridor")
    rows,columns=np.asarray(cells).T
    boxes=shapely.box(columns,rows,columns+1,rows+1)
    return shapely.intersection(shapely.coverage_union_all(boxes),land_surface)


def _bank_route(points, channel_geometry, passages, corridor):
    """Navigate dry banks between the source route's explicit bridge portals.

    A facility centre alone does not select either approach bank: a shortest
    path can enter and leave that centre through the same half deck. Each
    accepted native bank transition therefore supplies its actual incoming
    and outgoing bank points, joined only by its straight physical deck.
    """
    source=shapely.LineString(points)
    if not passages and not source.intersects(channel_geometry):
        return np.asarray(points)
    free=corridor.difference(channel_geometry)
    polygons=[polygon for polygon in shapely.get_parts(free)if polygon.geom_type=="Polygon"]
    def bank_port(coordinates,centre):
        point=shapely.Point(coordinates)
        if any(polygon.covers(point)for polygon in polygons):
            return tuple(coordinates)
        # An intersection of two floating-point edges can round one ulp
        # inside the water while its actual construction lies on the bank.
        # Continue the same straight deck to the first representable dry
        # coordinate. No city or facility anchor is shifted, and no physical
        # width, polygon buffer or acceptance tolerance is introduced.
        bound=16*float(np.spacing(max(1.,max(abs(value)for value in coordinates))))
        if not polygons or min(polygon.distance(point)for polygon in polygons)>bound:
            raise ValueError("a bridge portal does not touch its accepted native dry corridor")
        values=np.asarray(coordinates,dtype=float)
        direction=values-np.asarray(centre,dtype=float)
        direction/=np.linalg.norm(direction)
        for iteration in range(1,17):
            sample=values+direction*(bound*iteration/16)
            point=shapely.Point(sample)
            if any(polygon.covers(point)for polygon in polygons):
                return tuple(sample)
        raise ValueError("a computed physical bank intersection has no representable dry portal")
    actual_passages=[]
    for station,deck in passages:
        deck=list(deck)
        if len(deck)==3:
            deck[0]=bank_port(deck[0],deck[1]);deck[2]=bank_port(deck[2],deck[1])
        elif station==0.:
            deck[-1]=bank_port(deck[-1],deck[0])
        else:
            deck[0]=bank_port(deck[0],deck[-1])
        actual_passages.append((station,deck))
    result=[tuple(points[0])]
    def navigate(last):
        first=result[-1]
        if first==tuple(last):
            return
        candidates=[polygon for polygon in polygons if polygon.covers(shapely.MultiPoint((first,last)))]
        if not candidates:
            raise ValueError(f"accepted corridor has disconnected banks between portals {first} and {last}")
        polygon=max(candidates,key=lambda polygon:polygon.area)
        result.extend(map(tuple,polygon_path(polygon,first,last)[1:]))
    for _station,deck in sorted(actual_passages,key=lambda entry:entry[0]):
        navigate(deck[0])
        result.extend(tuple(point)for point in deck[1:]if tuple(point)!=result[-1])
    navigate(points[-1])
    return np.asarray(result)


def _channel_span(point, direction, channel):
    """The connected water interval across a recorded facility's station."""
    direction=np.asarray(direction,dtype=float)
    direction/=np.linalg.norm(direction)
    radius=max(float(point.distance(channel.boundary))*2,1e-7)
    coordinates=np.asarray(point.coords[0])
    while True:
        ends=coordinates+np.asarray((-radius,radius))[:,None]*direction
        if not any(channel.covers(shapely.Point(end))for end in ends):
            break
        radius*=2
    line=shapely.LineString(ends)
    pieces=[part for part in _parts(shapely.intersection(line,channel))if part.distance(point)<1e-8]
    if not pieces:
        raise ValueError("a river facility has no physical channel span")
    span=max(pieces,key=lambda part:part.length)
    return tuple(map(tuple,(span.coords[0],span.coords[-1])))


def _facility_decks(point, river, channel):
    """Straight decks from a recorded junction into its actual bank sectors.

    At a confluence a normal to just one reach misses the narrow bank between
    tributaries. The incident channel rays, rather than an arbitrary circle
    opening, determine the crossing's possible bank approaches.
    """
    angles=[]
    for line in _parts(river):
        station=float(line.project(point))
        if line.distance(point)>1e-7:
            continue
        for distance in (max(0.,station-.0001),min(line.length,station+.0001)):
            sample=line.interpolate(distance)
            vector=np.asarray((sample.x-point.x,sample.y-point.y))
            if np.linalg.norm(vector)>1e-10:
                angles.append(float(np.arctan2(vector[1],vector[0])))
    angles=sorted(set(round(angle,10)for angle in angles))
    if not angles:
        raise ValueError("a recorded flow facility has no incident channel rays")
    decks=[]
    for first,last in zip(angles,(*angles[1:],angles[0]+2*np.pi),strict=True):
        angle=(first+last)*.5
        direction=np.asarray((np.cos(angle),np.sin(angle)))
        span=_channel_span(point,direction,channel)
        end=max(span,key=lambda coordinates:np.dot(np.asarray(coordinates)-np.asarray(point.coords[0]),direction))
        decks.append(shapely.LineString((point.coords[0],end)))
    return tuple(decks)


def _bank_deck(direction, native, point, source_parts, display_parts, decks):
    """Map a native bank sector through the shared reach identities.

    A global angle comparison loses left/right bank identity when a terrain
    channel bends. The two sides of each same source/display reach retain
    their order; the native sector's clockwise ray selects the matching
    displayed sector and its already constructed physical deck.
    """
    rays=[]
    for source,display in zip(source_parts,display_parts,strict=True):
        if source.distance(native)>1e-7 or display.distance(point)>1e-7:
            continue
        native_station=float(source.project(native));display_station=float(display.project(point))
        for sign in (-1.,1.):
            native_sample=source.interpolate(np.clip(native_station+sign*.0001,0.,source.length))
            sample=display.interpolate(np.clip(display_station+sign*.0001,0.,display.length))
            first=np.asarray(native_sample.coords[0])-np.asarray(native.coords[0])
            last=np.asarray(sample.coords[0])-np.asarray(point.coords[0])
            if np.linalg.norm(first)>1e-10 and np.linalg.norm(last)>1e-10:
                rays.append((round(float(np.arctan2(first[1],first[0])),10),
                             round(float(np.arctan2(last[1],last[0])),10)))
    rays=sorted(set(rays))
    if not rays:
        raise ValueError("a source bank portal has no shared physical channel ray")
    angle=float(np.arctan2(direction[1],direction[0]))
    clockwise=max(rays,key=lambda ray:-((angle-ray[0])%(2*np.pi)))
    display_angles=sorted(set(ray[1]for ray in rays))
    if len(display_angles)!=len(decks):
        raise ValueError("source and displayed facility bank fans differ")
    return decks[display_angles.index(clockwise[1])]


def _bank_boundary_roundoff(water,channel):
    """Certify binary64 bank limits without accepting a physical water chord.

    Clipping a straight bank edge can produce an intersection coordinate a
    few ulps off the original edge. For each candidate road segment, both
    ends must lie within the same scale-bound distance of one actual straight
    channel edge. Distance to that convex edge is convex along the segment,
    so this proves the whole segment is a numerical bank limit; no sampling
    or fixed map-space tolerance substitutes for a physical crossing.
    """
    if water.is_empty:
        return ()
    segments=np.concatenate([np.stack((np.asarray(line.coords)[:-1],np.asarray(line.coords)[1:]),axis=1)
                             for line in _parts(channel.boundary)])
    bank_edges=shapely.linestrings(segments)
    index=shapely.STRtree(bank_edges)
    certified=[]
    for line in _parts(water):
        coordinates=np.asarray(line.coords)
        for first,last in zip(coordinates[:-1],coordinates[1:],strict=True):
            pair=np.asarray((first,last))
            scale=max(1.,float(np.max(np.abs(pair))))
            bound=16*float(np.spacing(scale))
            low=np.min(pair,axis=0)-bound;high=np.max(pair,axis=0)+bound
            matches=index.query(shapely.box(*low,*high))
            if not len(matches):
                continue
            errors=np.max(shapely.distance(bank_edges[matches,None],shapely.points(pair)[None,:]),axis=1)
            valid=np.flatnonzero(errors<=bound)
            if not len(valid):
                continue
            selected=int(valid[np.argmin(errors[valid])]);edge=int(matches[selected])
            certified.append({"geometry":shapely.geometry.mapping(shapely.LineString(pair)),
                              "bankSegment":segments[edge].tolist(),"maxDistance":float(errors[selected]),
                              "ulpBound":bound})
    return tuple(certified)


def _uncertified_water(water,certificates):
    """Retain physical water chords, without overlaying certified sublines."""
    banks={tuple(map(tuple,record["geometry"]["coordinates"]))for record in certificates}
    unsafe=[]
    for line in _parts(water):
        coordinates=tuple(line.coords)
        for first,last in zip(coordinates[:-1],coordinates[1:],strict=True):
            if (first,last)not in banks:
                unsafe.append(shapely.LineString((first,last)))
    return shapely.union_all(unsafe)


def _water_components(water_roads):
    """Connected bridge graphs, including shared junctions within a deck."""
    lines=np.asarray(_parts(shapely.line_merge(shapely.node(water_roads))),dtype=object)
    endpoints={}
    for index,line in enumerate(lines):
        for coordinates in (line.coords[0],line.coords[-1]):
            endpoints.setdefault(tuple(coordinates),[]).append(index)
    edges=[(members[0],member)for members in endpoints.values()for member in members[1:]]
    if edges:
        rows,columns=np.asarray(edges).T
        graph=csr_array((np.ones(len(rows)),(rows,columns)),shape=(len(lines),len(lines)))
        count,labels=connected_components(graph,directed=False)
    else:
        count,labels=len(lines),np.arange(len(lines))
    return np.asarray([shapely.line_merge(shapely.union_all(lines[labels==label]))for label in range(count)],dtype=object)


def _crossing_span_payload(point, water_lines, water_index, channel):
    matches=water_index.query(point.buffer(1e-7),predicate="intersects")
    pieces=[line for line in water_lines[matches]if line.distance(point)<1e-7]
    if not pieces:
        raise ValueError("a recorded river access has no final physical water span")
    geometry=shapely.line_merge(shapely.union_all(pieces))
    terminals={}
    for line in _parts(geometry):
        station=float(line.project(point))
        for start,end in ((0.,station),(station,line.length)):
            if end-start<1e-7:
                continue
            deck=substring(line,start,end)
            first,last=shapely.Point(deck.coords[0]),shapely.Point(deck.coords[-1])
            if abs(deck.length-first.distance(last))>1e-7:
                raise ValueError("a facility's water span must be a straight bank deck")
            direction=np.asarray(last.coords[0])-np.asarray(first.coords[0])
            offset=np.asarray(point.coords[0])-np.asarray(first.coords[0])
            perpendicular=abs(direction[0]*offset[1]-direction[1]*offset[0])/np.linalg.norm(direction)
            if perpendicular>1e-7:
                raise ValueError("all bridge deck arms must share the recorded crossing station")
        for coordinates in (line.coords[0],line.coords[-1]):
            terminals[tuple(coordinates)]=terminals.get(tuple(coordinates),0)+1
    banks=set()
    for coordinates,degree in terminals.items():
        if degree==1:
            terminal=shapely.Point(coordinates)
            if terminal.distance(point)<1e-7:
                continue
            if terminal.distance(channel.boundary)>1e-7:
                raise ValueError(f"a crossing deck must terminate at the actual water bank: {point.wkt}; {terminal.wkt}; {terminal.distance(channel.boundary)}")
            banks.add(tuple(coordinates))
    return {"geometry":shapely.geometry.mapping(geometry),"bankPoints":[list(point)for point in sorted(banks)]}


@dataclass(frozen=True)
class PreparedTransportGeometry:
    routes: tuple
    paths: tuple
    bridges: tuple
    positions: dict
    tangents: dict
    crossings: tuple
    endpoint_touches: tuple
    spans: dict
    bank_roundoff: tuple


def prepare_transport_geometry(grid, routes, bridges, *, locations, land_surface,
                               river_source_geometry, river_geometry, river_channel_geometry, raw_elevation_m,
                               terrain_field):
    """Resolve river-bank roads and actual crossing facilities once, before SVG.

    Every crossing has a transverse intersection in the recorded native flow
    graph. Shared source crossings use one facility. Overlapping river/road
    centre lines become bank corridors; checked simplification and curves
    cannot manufacture another crossing. No settlement endpoint is moved.
    """
    from .city_transport import refined_land_routes
    from .society.transport import _route_can_bridge, derive_bridges
    from .road_engineering import RoadEngineering
    engineering = RoadEngineering(raw_elevation_m,float(grid.metadata['planet']['radiusKm']),
        grid.metadata['worldProfile']['technologyEra'])
    grade_check = lambda original, proposed: engineering.preserves_profile(original,proposed,terrain_field=terrain_field)
    from .mountain_roads import engineer_mountain_path
    engineer_path = lambda points: engineer_mountain_path(points,engineering,
        terrain_field=terrain_field,road_surface=road_surface)

    routes = tuple(routes)
    bridges = tuple(bridges)
    expected = derive_bridges(grid, routes,raw_elevation_m=raw_elevation_m)
    if bridges != expected:
        raise ValueError("recorded bridges must exactly match the accepted native road/flow crossings")
    source_parts = shapely.get_parts(river_source_geometry)
    display_parts = shapely.get_parts(river_geometry)
    if len(source_parts) != len(display_parts):
        raise ValueError("source and displayed river reaches must have shared identities")
    source_index, display_index = shapely.STRtree(source_parts), shapely.STRtree(display_parts)
    channel_parts=shapely.get_parts(river_channel_geometry)
    channel_index=shapely.STRtree(channel_parts)
    def crossing_channel(point):
        matches=channel_index.query(point,predicate="intersects")
        if not len(matches):
            raise ValueError("a displayed flow crossing has no physical water channel")
        return shapely.union_all(channel_parts[matches])
    native_geometry=native_river_geometry(grid,raw_elevation_m=raw_elevation_m)
    native_parts=shapely.get_parts(native_geometry)
    native_index=shapely.STRtree(native_parts)
    transitions={}
    for route in routes:
        if route.mode not in {"road","rail"}:
            continue
        for coordinates in _split_seam(route.path,grid.shape[1]):
            road=shapely.LineString(coordinates)
            matches=native_index.query(road,predicate="intersects")
            river=shapely.union_all(native_parts[matches])
            for point,incoming,outgoing in source_crossing_transitions(road,river):
                transitions[(route.identifier,round(point.x,7),round(point.y,7))]=(incoming,outgoing)
    events = list(source_transport_crossings(grid,routes,native_geometry,raw_elevation_m=raw_elevation_m))
    grouped = {}
    for route, point, cell, order in events:
        grouped.setdefault((round(point.x,7),round(point.y,7)),[]).append((route,point,cell,order))
    # Replay the same deterministic source ordering. Two crossings may share
    # one river support cell; cell/route matching alone cannot identify them.
    selected = []
    for entries in grouped.values():
        legal = [entry for entry in entries if _route_can_bridge(entry[0].importance,entry[3])]
        if not legal:
            raise ValueError(f"{entries[0][0].identifier} crosses an unlicensed major river")
        selected.append((min(legal,key=lambda entry:(-entry[3],-_RANK[entry[0].importance],entry[0].identifier,entry[2])),entries))
    selected.sort(key=lambda item:(item[0][2][0],item[0][2][1],item[0][0].identifier))
    if len(selected) != len(bridges):
        raise ValueError("source facilities must cover each distinct native crossing exactly once")
    positions, facilities, audit, spans, decks = {}, [], [], {}, {}
    by_route = {}
    for (chosen, entries), record in zip(selected,bridges,strict=True):
        route, native, cell, order = chosen
        reaches = source_index.query(native.buffer(1e-7),predicate="intersects")
        if not len(reaches):
            raise ValueError("a native crossing has no corresponding physical river reach")
        physical = shapely.union_all(display_parts[reaches])
        # The accepted native flow station owns the display station on that
        # same reach. Intersecting the old road with a displaced channel can
        # select a neighbouring bend, and therefore the wrong bank portal.
        reach=min(_parts(physical),key=native.distance)
        point=reach.interpolate(reach.project(native))
        position = (float(point.x),float(point.y))
        local_channel=crossing_channel(point)
        bridge_decks=_facility_decks(point,physical,local_channel)
        facilities.append(record)
        positions[record.identifier] = position
        decks[record.identifier]=[]
        audit.append({"id":record.identifier,"kind":"bridge","sourceRoadIds":sorted({entry[0].identifier for entry in entries}),
                      "nativePoint":[float(native.x),float(native.y)],"displayPoint":list(position),
                      "riverCell":[int(cell[0]),int(cell[1])],"riverOrder":order,
                      "evidence":"recorded-bridge-and-native-flow-crossing","sourceBankTransitions":[]})
        for member,member_point,_cell,_order in entries:
            key=(member.identifier,round(member_point.x,7),round(member_point.y,7))
            directions=transitions.get(key)
            if directions is None:
                original=min((shapely.LineString(part)for part in _split_seam(member.path,grid.shape[1])),key=member_point.distance)
                direction=endpoint_bank_direction(original,native_geometry,member_point)
                if direction is None:
                    raise ValueError("a recorded source endpoint bridge has no native bank approach")
                selected_deck=_bank_deck(direction,native,point,source_parts[reaches],display_parts[reaches],bridge_decks)
                passage=(position,tuple(selected_deck.coords[-1]))
                if original.project(member_point)>original.length*.5:
                    passage=tuple(reversed(passage))
                used_decks=(selected_deck,)
            else:
                used_decks=tuple(_bank_deck(direction,native,point,source_parts[reaches],display_parts[reaches],bridge_decks)for direction in directions)
                if used_decks[0].equals(used_decks[1]):
                    raise ValueError("a recorded source bank transition maps to one physical bank")
                passage=(tuple(used_decks[0].coords[-1]),position,tuple(used_decks[1].coords[-1]))
            decks[record.identifier].extend(used_decks)
            audit[-1]["sourceBankTransitions"].append({"routeIdentifier":member.identifier,
                "nativeDirections":[list(direction)for direction in (directions if directions is not None else (direction,))],
                "bankPoints":[list(deck.coords[-1])for deck in used_decks]})
            by_route.setdefault(member.identifier,[]).append((member_point,passage))
    access_spans={}
    for route in routes:
        if route.mode in {"road","rail"}:
            for coordinates in (route.path[0],route.path[-1]):
                point=shapely.Point(coordinates)
                if point.distance(native_geometry)<1e-8 and point.distance(river_geometry)<1e-8:
                    original=min((shapely.LineString(part)for part in _split_seam(route.path,grid.shape[1])),key=point.distance)
                    direction=endpoint_bank_direction(original,native_geometry,point)
                    if direction is None:
                        raise ValueError("a river-city endpoint has no accepted native bank approach")
                    matches=source_index.query(point.buffer(1e-7),predicate="intersects")
                    possible=_facility_decks(point,shapely.union_all(display_parts[matches]),crossing_channel(point))
                    access_spans[(route.identifier,tuple(coordinates))]=(_bank_deck(direction,point,point,source_parts[matches],display_parts[matches],possible),)
    opening_parts=[deck.buffer(1e-8,cap_style="square")
                   for group in (*decks.values(),*access_spans.values())for deck in group]
    openings=shapely.union_all(opening_parts)
    road_surface=land_surface.difference(river_channel_geometry).union(openings)
    native_routes={route.identifier:route for route in routes}
    prepared_routes = []
    for route in refined_land_routes(routes,locations):
        if route.mode not in {"road","rail"}:
            prepared_routes.append(route)
            continue
        route_parts = _split_seam(route.path,grid.shape[1])
        native=native_routes[route.identifier]
        endpoint_access=tuple((old,new) for old,new in (
            (native.path[0],route.path[0]),(native.path[-1],route.path[-1])) if old!=new)
        prepared_parts = []
        for points in route_parts:
            line = shapely.LineString(points)
            passages=[]
            for endpoint,is_start in ((tuple(points[0]),True),(tuple(points[-1]),False)):
                for deck in access_spans.get((route.identifier,endpoint),()):
                    passage=tuple(deck.coords)if is_start else tuple(reversed(deck.coords))
                    passages.append((0. if is_start else line.length,passage))
            for native,passage in by_route.get(route.identifier,()):
                if line.distance(native)<1e-7:
                    distance=float(line.project(native))
                    # A source endpoint facility already owns that exact
                    # city access; do not navigate its half deck twice.
                    passages=[entry for entry in passages if abs(entry[0]-distance)>1e-8]
                    passages.append((distance,passage))
            try:
                corridor=_road_native_corridor(grid,points,land_surface,endpoint_access=endpoint_access)
                local_channel=shapely.union_all(channel_parts[channel_index.query(corridor,predicate="intersects")])
                banked = _bank_route(points,local_channel,passages,corridor)
            except ValueError as error:
                raise ValueError(f"{route.identifier}: {error}") from error
            prepared_parts.append(banked)
        if len(prepared_parts)!=1:
            # Preserve the longitude break explicitly; the network splits it
            # before union, so the source's wrapped connection is never ink.
            path = tuple(tuple(point) for part in prepared_parts for point in part)
        else:
            path = tuple(map(tuple,prepared_parts[0]))
        prepared_routes.append(replace(route,path=path))
    anchors = [(column,row) for row,column in locations.values()]+list(positions.values())+[tuple(deck.coords[-1])for group in (*decks.values(),*access_spans.values())for deck in group]
    sea_obstacles=land_surface.buffer(1e-6)
    paths = tuple((mode,importance,_curved_transport_points(points,mode,sea_obstacles if mode=="sea" else road_surface,river_geometry,
                      grade_check=grade_check if mode=='road' else None))
                  for mode,importance,points in shared_transport_paths(
                      grid,prepared_routes,land_geometry=sea_obstacles,road_surface=road_surface,river_geometry=river_geometry,anchors=anchors,
                      grade_check=grade_check,engineer_path=engineer_path))
    lines = [shapely.LineString(points) for mode,_importance,points in paths if mode in {"road","rail"}]
    network = shapely.union_all(lines)
    water_roads=network.intersection(river_channel_geometry).difference(river_channel_geometry.boundary)
    unlicensed_water=water_roads.difference(openings)
    bank_roundoff=_bank_boundary_roundoff(unlicensed_water,river_channel_geometry)
    # A second GEOS difference of floating-point collinear sublines can
    # reproduce the bank sliver; classify the actual segments directly.
    unexplained_water=_uncertified_water(unlicensed_water,bank_roundoff)
    raw_water_roads=water_roads
    water_roads=water_roads.intersection(openings)
    if unexplained_water.length>1e-7:
        parts=sorted(_parts(unexplained_water),key=lambda part:-part.length)
        raise ValueError(f"rendered land transport enters the physical water channel without a recorded span: {unexplained_water.length}; {[part.wkt for part in parts[:3]]}")
    water_lines=_water_components(water_roads)
    water_index=shapely.STRtree(water_lines)
    for identifier,position in positions.items():
        spans[identifier]=_crossing_span_payload(shapely.Point(position),water_lines,water_index,river_channel_geometry)
        if len(spans[identifier]["bankPoints"])<2:
            raise ValueError("a recorded bridge must reach at least two physical banks")
        members=next(proof["sourceRoadIds"]for proof in audit if proof["id"]==identifier)
        source_routes={route.identifier:shapely.MultiLineString(_split_seam(route.path,grid.shape[1]))
                       for route in prepared_routes if route.identifier in members}
        portals=[]
        for bank in spans[identifier]["bankPoints"]:
            point=shapely.Point(bank)
            owners=sorted(identifier for identifier,line in source_routes.items()if line.distance(point)<1e-7)
            if not owners:
                raise ValueError("a physical bridge-bank portal has no licensed source road")
            portals.append({"position":bank,"sourceRoadIds":owners})
        spans[identifier]["bankPortals"]=portals
        proof=next(proof for proof in audit if proof["id"]==identifier)
        for transition in proof["sourceBankTransitions"]:
            banks=[]
            for intended in transition["bankPoints"]:
                bank=min(spans[identifier]["bankPoints"],key=lambda bank:shapely.Point(bank).distance(shapely.Point(intended)))
                if shapely.Point(bank).distance(shapely.Point(intended))>1e-7:
                    raise ValueError("a final facility omits a native route's required bank portal")
                if source_routes[transition["routeIdentifier"]].distance(shapely.Point(bank))>1e-7:
                    raise ValueError("a source member route omits its recorded bank transition")
                banks.append(bank)
            if len(banks)==2 and banks[0]==banks[1]:
                raise ValueError("a native two-bank route transition uses one final bank")
            transition["bankPoints"]=banks
    licensed_points = shapely.MultiPoint(list(positions.values()))
    intersections = shapely.intersection(network,river_geometry)
    if any(part.length>1e-7 for part in _parts(intersections)):
        raise ValueError("rendered land transport must not overlap a river channel")
    unexplained = [point for point in _point_parts(intersections)
                   if not positions or point.distance(licensed_points)>1e-6]
    endpoint_touches=[]
    endpoints={}
    for route in routes:
        if route.mode in {"road","rail"}:
            for point,settlement_id in ((route.path[0],route.source_settlement_id),
                                        (route.path[-1],route.target_settlement_id)):
                if settlement_id is not None:
                    endpoints.setdefault(tuple(point),[]).append((route,settlement_id))
    line_index=shapely.STRtree(lines)
    remaining=[]
    for point in unexplained:
        entries=next((entries for coordinates,entries in endpoints.items()
                      if point.distance(shapely.Point(coordinates))<1e-7),None)
        if not entries or point.distance(native_geometry)>1e-7:
            remaining.append(point)
            continue
        source_samples=[]
        for route,_settlement_id in entries:
            original=min((shapely.LineString(part)for part in _split_seam(route.path,grid.shape[1])),key=point.distance)
            distance=original.project(point)
            source_samples.append(original.interpolate(.025 if distance<.025 else max(0.,original.length-.025)))
        display_samples=[]
        for index in line_index.query(point.buffer(1e-7),predicate="intersects"):
            line=lines[index]
            distance=line.project(point)
            if distance<1e-7:
                display_samples.append(line.interpolate(min(.025,line.length)))
            elif line.length-distance<1e-7:
                display_samples.append(line.interpolate(max(0.,line.length-.025)))
            else:
                display_samples.extend((line.interpolate(distance-.025),line.interpolate(distance+.025)))
        source_owners={owner for owner in _river_bank_owner(native_geometry,point,source_samples)if owner is not None}
        display_owners={owner for owner in _river_bank_owner(river_geometry,point,display_samples)if owner is not None}
        if len(source_owners)>1 or len(display_owners)!=1:
            remaining.append(point)
            continue
        access=_crossing_span_payload(point,water_lines,water_index,river_channel_geometry)
        endpoint_touches.append({"kind":"same-bank-river-access","nativePoint":[float(point.x),float(point.y)],
                                 "displayPoint":[float(point.x),float(point.y)],
                                 "sourceRoadIds":sorted({route.identifier for route,_settlement_id in entries}),
                                 "sourceSettlementIds":sorted({identifier for _route,identifier in entries}),
                                 "nativeBankCount":len(source_owners),"displayBankCount":len(display_owners),
                                 "channelAccessGeometry":access["geometry"],"bankPoints":access["bankPoints"]})
    if remaining:
        raise ValueError(f"rendered road/river intersections lack facilities: {len(remaining)}; {[point.wkt for point in remaining[:8]]}")
    # The published numerical-bank certificates use the actual completed
    # physical spans as permission, rather than navigation's roundoff slots.
    permitted=shapely.union_all([shapely.geometry.shape(payload["geometry"])for payload in spans.values()]+
                               [shapely.geometry.shape(proof["channelAccessGeometry"])for proof in endpoint_touches])
    final_unlicensed=raw_water_roads.difference(permitted)
    bank_roundoff=_bank_boundary_roundoff(final_unlicensed,river_channel_geometry)
    unsafe_water=_uncertified_water(final_unlicensed,bank_roundoff)
    if unsafe_water.length>1e-7:
        raise ValueError(f"final physical water roads omit their exact facility spans: {unsafe_water.length}")
    tangents = {}
    for identifier,position in positions.items():
        point = shapely.Point(position)
        line = lines[int(line_index.nearest(point))]
        distance = float(line.project(point))
        first,last = line.interpolate(max(0.,distance-.05)),line.interpolate(min(line.length,distance+.05))
        direction = np.array((last.x-first.x,last.y-first.y))
        direction /= max(float(np.linalg.norm(direction)),1e-15)
        tangents[identifier] = tuple(map(float,direction))
    return PreparedTransportGeometry(tuple(prepared_routes),paths,tuple(facilities),positions,tangents,tuple(audit),tuple(endpoint_touches),spans,bank_roundoff)
