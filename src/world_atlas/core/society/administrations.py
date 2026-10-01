"""Administrative ownership and shared geometry from actual arrival fronts.

Formation chooses institutions and their recorded settlement cores. Country
fronts compete on terrain and actual marine routes; provincial fronts compete
inside those countries. The nested shared faces supply both fills and borders.
"""
from dataclasses import dataclass,replace
import hashlib
import json
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage

from ..suitability import relative_land_slope
from .administrative_front import administrative_front
from .administrative_coverage import administrative_coverage
from .politics import _state_transition_penalties
from .population import population_density, cell_areas_km2
from .territorial_simulation import TerritorySeed, TerritorySimulation, bridge_transition_discounts
from .transport import road_network_fields
from .provinces import derive_provinces
from .administrative_front import AdministrativeFront


@dataclass(frozen=True,slots=True)
class AdministrativeHierarchy:
    """Country arrival faces and nested provincial arrivals share one paint."""

    countries: AdministrativeFront
    provinces: AdministrativeFront
    province_to_state: np.ndarray


def _core_seeds(records, society):
    """Only actual administrative seats originate zero-time authority."""
    by_id = {city.identifier: city for city in society.settlements}
    return tuple(TerritorySeed(by_id[record.core_settlement_id].row,
        by_id[record.core_settlement_id].column,record.identifier) for record in records)


def _maritime_links(grid,society):
    """Travel along complete recorded shipping paths, in native metric units."""
    by_id={city.identifier:city for city in society.settlements}
    links=[]
    for route in society.transport.routes:
        if route.mode!='sea' or route.target_settlement_id is None:continue
        first=by_id[route.source_settlement_id];last=by_id[route.target_settlement_id]
        delta=np.diff(np.asarray(route.path,dtype=float),axis=0)
        delta[:,0]=(delta[:,0]+grid.shape[1]*.5)%grid.shape[1]-grid.shape[1]*.5
        cost=float(np.hypot(delta[:,0],delta[:,1]).sum())
        if cost>0:links.append(((first.row,first.column),(last.row,last.column),cost))
    return tuple(links)


def _travel_simulation(grid, thematic, society, *, domains=None, valid=None):
    land = np.asarray(grid.water) == 0
    valid = np.asarray(society.politics.state_id) > 0 if valid is None else np.asarray(valid,dtype=bool)
    roads, _ = road_network_fields(grid.shape, society.transport.routes)
    bridges = bridge_transition_discounts(grid.river_order, society.transport.routes, society.transport.bridges)
    elevation = np.asarray(grid.elevation, dtype=np.float64)
    density = population_density(grid, society.population)
    friction = np.maximum(.25,
        1. + 7.2 * elevation**2 + 18. * relative_land_slope(elevation, land)
        + 1.9 * (1.-np.clip(thematic.land_potential, 0., 1.))
        + 2.6 * (1.-np.clip(thematic.habitability, 0., 1.))
        - .08 * (grid.river_order > 0)
        - .22 * density / max(float(density.max(initial=0.)), 1e-15)
        - .68 * np.clip(society.transport.accessibility, 0., 1.)).astype(np.float32)
    transitions = _state_transition_penalties(elevation, grid.river_order,
        society.cultures.language_id, roads, land_mask=land)
    return TerritorySimulation(valid, friction, transitions, roads, bridges, owner_constraint=domains)


def administrative_source(grid, thematic, society):
    """Reproduce continuous national fronts and provincial fronts inside them."""
    countries = administrative_front(_travel_simulation(grid, thematic, society),
        _core_seeds(society.politics.states, society),maritime_links=_maritime_links(grid,society))
    states = np.where(grid.water == 0, np.maximum(countries.owner, 0), -1).astype(np.int16)
    seeds = tuple(replace(seed, domain=int(states[seed.row,seed.column]))
        for seed in _core_seeds(society.provinces.provinces, society))
    for record in society.provinces.provinces:
        city = next(city for city in society.settlements if city.identifier == record.core_settlement_id)
        if states[city.row,city.column] != record.state_identifier:
            raise ValueError('provincial seat must belong to its current national arrival domain')
    provinces = administrative_front(_travel_simulation(grid,thematic,society,domains=states,valid=states>0), seeds,maritime_links=_maritime_links(grid,society))
    parent = np.zeros(len(society.provinces.provinces)+1,dtype=np.int16)
    for record in society.provinces.provinces:
        parent[record.identifier] = record.state_identifier
    return AdministrativeHierarchy(countries, provinces, parent)


def derive_administrations(grid, thematic, society):
    """Return institutions with their native arrival winners and source front."""
    countries = administrative_front(_travel_simulation(grid,thematic,society),
        _core_seeds(society.politics.states,society),maritime_links=_maritime_links(grid,society))
    states = np.where(grid.water == 0,np.maximum(countries.owner,0),-1).astype(np.int16)
    politics = replace(society.politics,state_id=states,frontier=(grid.water==0)&(states==0))
    provinces = derive_provinces(grid,thematic,society.population,society.settlements,
                                society.transport,society.cultures,politics)
    society = replace(society,politics=politics,provinces=provinces)
    seeds = tuple(replace(seed,domain=int(states[seed.row,seed.column]))
        for seed in _core_seeds(provinces.provinces,society))
    province_front = administrative_front(_travel_simulation(grid,thematic,society,domains=states),seeds,maritime_links=_maritime_links(grid,society))
    land = np.asarray(grid.water) == 0
    provinces = np.where(land, np.maximum(province_front.owner, 0), -1).astype(np.int32)
    if np.any((society.politics.state_id > 0) & (provinces <= 0)):
        raise ValueError('governed component has no evidenced administrative source')
    parent = np.zeros(len(society.provinces.provinces)+1, dtype=np.int16)
    for record in society.provinces.provinces:
        parent[record.identifier] = record.state_identifier
    states = np.where(land, parent[np.maximum(provinces, 0)], -1).astype(np.int16)
    front = AdministrativeHierarchy(countries,province_front,parent)
    weights = society.population.population_weight
    state_weight = np.bincount(np.maximum(states, 0).ravel(),
        weights=weights.ravel(), minlength=len(society.politics.states)+1)
    state_records = []
    for record in society.politics.states:
        share = float(state_weight[record.identifier])
        lower = max(1000, round(share*society.population.population_min/10000)*10000)
        upper = max(lower+1000, round(share*society.population.population_max/10000)*10000)
        state_records.append(replace(record, population_min=lower, population_max=upper))
    counts = np.bincount(np.maximum(provinces, 0).ravel(), minlength=len(parent))
    areas = cell_areas_km2(grid)
    controlled = states > 0
    mean_density = float(weights[controlled].sum()/areas[controlled].sum()) if controlled.any() else 0.
    province_records = []
    for record in society.provinces.provinces:
        region = provinces == record.identifier
        ratio = float(weights[region].sum()/areas[region].sum()) / max(mean_density, 1e-15)
        province_records.append(replace(record, area_cells=int(counts[record.identifier]),
            population_density_class='dense' if ratio >= 1.35 else 'settled' if ratio >= .62 else 'sparse'))
    updated = replace(society,
        politics=replace(society.politics, state_id=states,
            frontier=land & (states <= 0), states=tuple(state_records)),
        provinces=replace(society.provinces, province_id=provinces,
            provinces=tuple(province_records)))
    return updated, front


def administrative_paint_coverage(front, land):
    """Extend source arrivals into water-only paint, then trace one envelope.

    The true physical shore clips the working paint. Land observations,
    including reserved stateless land, are never changed by this extension.
    """
    if isinstance(front,AdministrativeHierarchy):
        from shapely.affinity import translate
        country_faces,country_ids = administrative_paint_coverage(front.countries,land)
        faces=[]; labels=[]
        for identifier in np.unique(country_ids):
            country=shapely.union_all([face for face,label in zip(country_faces,country_ids,strict=True) if label==identifier])
            if identifier == 0:
                faces.extend(shapely.get_parts(country));labels.extend([0]*len(shapely.get_parts(country)));continue
            # An interpolation halo carries this country's own observations
            # beyond its source arc. It is paint support, never new ownership.
            west,north,east,south=country.bounds
            height,width=front.provinces.valid.shape
            x0=max(0,int(np.floor(west))-8);x1=min(width,int(np.ceil(east))+8)
            y0=max(0,int(np.floor(north))-8);y1=min(height,int(np.ceil(south))+8)
            region=(front.countries.owner[y0:y1,x0:x1]==identifier)
            distance,indices=ndimage.distance_transform_edt(~region,return_indices=True)
            rr,cc=indices
            source=front.provinces
            paint=AdministrativeFront(source.owners[:,y0:y1,x0:x1][:,rr,cc].copy(),
                source.arrivals[:,y0:y1,x0:x1][:,rr,cc].copy(),
                source.edge_costs[:,y0:y1,x0:x1][:,rr,cc].copy(),
                distance<=6.,source.absent_time,None,source.domains)
            paint.owners[:,~paint.valid]=0;paint.arrivals[:,~paint.valid]=np.inf
            province_faces,province_ids=administrative_coverage(paint)
            painted=[]
            for face,label in zip(province_faces,province_ids,strict=True):
                if label==0:continue
                geometry=shapely.intersection(country,translate(face,xoff=x0,yoff=y0),grid_size=1e-8)
                painted.append(geometry)
                for part in shapely.get_parts(geometry):
                    if shapely.get_type_id(part)==3 and part.area>0:
                        faces.append(part);labels.append(label)
            missing=shapely.difference(country,shapely.union_all(painted),grid_size=1e-8)
            if not missing.is_empty:
                raise ValueError(f'provincial interpolation halo failed: {identifier=}, {missing.area=}, {missing.bounds=}, {x0=}, {y0=}')
        from ..cartographic_features import shared_display_coverage
        identifiers=np.unique(labels)
        grouped=shared_display_coverage([shapely.union_all([face for face,label in zip(faces,labels,strict=True) if label==identifier])
                                        for identifier in identifiers])
        faces=[];labels=[]
        for identifier,geometry in zip(identifiers,grouped,strict=True):
            parts=shapely.get_parts(geometry)
            faces.extend(parts);labels.extend([identifier]*len(parts))
        expected=shapely.union_all(country_faces)
        difference=shapely.symmetric_difference(expected,shapely.union_all(faces),grid_size=1e-8)
        if not difference.is_empty:
            raise ValueError(f'nested provincial paint must cover every national source face: {difference.area=}, {difference.bounds=}')
        return tuple(faces),np.asarray(labels,dtype=np.int32)
    land = np.asarray(land, dtype=bool)
    if not land.any():
        return administrative_coverage(front)
    distance, indices = ndimage.distance_transform_edt(
        ~np.pad(land, ((0,0),(4,4)), mode='wrap'), return_indices=True)
    distance = distance[:,4:-4]; indices = indices[:,:,4:-4]
    rows = indices[0]; cols = (indices[1]-4)%land.shape[1]
    water_paint = ~land & (distance <= 4.)
    owners = front.owners.copy(); arrivals = front.arrivals.copy()
    valid = front.valid.copy(); edges = front.edge_costs.copy()
    owners[:,water_paint] = front.owners[:,rows[water_paint],cols[water_paint]]
    arrivals[:,water_paint] = front.arrivals[:,rows[water_paint],cols[water_paint]]
    edges[:,water_paint] = front.edge_costs[:,rows[water_paint],cols[water_paint]]
    valid[water_paint] = front.valid[rows[water_paint],cols[water_paint]]
    parent = None if front.parent is None else front.parent.copy()
    if parent is not None:
        parent[water_paint] = front.parent[rows[water_paint],cols[water_paint]]
    paint = replace(front, owners=owners, arrivals=arrivals,
        edge_costs=edges, valid=valid, parent=parent)
    return administrative_coverage(paint)


def write_administrative_stage(directory, grid, society, front):
    """Persist source arrivals and their single shared, full-frame coverage."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name,source in (('country-arrivals',front.countries),('province-arrivals',front.provinces)):
        np.savez_compressed(directory/f'{name}.npz',owners=source.owners,
            arrivals=source.arrivals,edge_costs=source.edge_costs,valid=source.valid,
            absent_time=np.array(source.absent_time),parent=np.empty((0,0),dtype=np.int32) if source.parent is None else source.parent,
            domain_ids=np.array(list(source.domains)),domain_values=np.array(list(source.domains.values())))
    faces, labels = administrative_paint_coverage(front, grid.water == 0)
    parent = np.zeros(len(society.provinces.provinces)+1, dtype=np.int16)
    for record in society.provinces.provinces:
        parent[record.identifier] = record.state_identifier
    # Validate every native centre against the published shared faces using an
    # indexed query. A dissolved country inherits exactly this same partition.
    tree = shapely.STRtree(faces)
    wrong = missing = overlap = 0
    height, width = grid.shape
    for north in range(0, height, 32):
        rows, cols = np.indices((min(32, height-north), width))
        dry = np.asarray(grid.water[north:north+len(rows)] == 0).ravel()
        points = shapely.points((cols+.5).ravel()[dry], (rows+north+.5).ravel()[dry])
        point_ids, face_ids = tree.query(points, predicate='covered_by')
        expected = np.maximum(society.provinces.province_id[north:north+len(rows)], 0).ravel()[dry]
        observed = labels[face_ids]
        good = observed == expected[point_ids]
        covered = np.bincount(point_ids[good], minlength=len(points))
        missing += int(np.count_nonzero(covered == 0))
        overlap += int(np.count_nonzero(covered > 1))
        wrong += int(np.count_nonzero(~good))
    if wrong or missing or overlap:
        raise ValueError(f'administrative arrival coverage native mismatch: {wrong=}, {missing=}, {overlap=}')
    wkb = shapely.to_wkb(faces)
    lengths = np.array([len(item) for item in wkb], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    data = np.frombuffer(b''.join(wkb), dtype=np.uint8)
    fingerprint = hashlib.sha256()
    for value in (society.provinces.province_id,parent,front.countries.owners,front.countries.arrivals,
                  front.provinces.owners,front.provinces.arrivals,data,labels):
        fingerprint.update(np.ascontiguousarray(value).tobytes())
    np.savez_compressed(directory/'shared-coverage.npz', wkb=data, offsets=offsets,
        province_ids=labels, state_ids=parent[labels], province_to_state=parent)
    report = dict(schema='administrative-arrival-stage-v2', status='complete',
        source='nested-administrative-core-travel-arrivals', fingerprint=fingerprint.hexdigest(),
        gridDigest=grid.content_digest(), shape=list(grid.shape),
        faces=len(faces), states=len(society.politics.states), provinces=len(society.provinces.provinces),
        nativeCentresClassified=height*width, nativeLandCentresChecked=int(np.count_nonzero(grid.water == 0)),
        nativeWaterCentresExcludedByPhysicalClip=int(np.count_nonzero(grid.water != 0)),
        nativeWrong=wrong, nativeMissing=missing,
        nativeOverlap=overlap, coverageValid=bool(shapely.coverage_is_valid(faces)),
        frameArea=float(shapely.union_all(faces).area),
        ownership='national core front intersected by intrastate provincial core fronts; fill and ink share these faces')
    (directory/'manifest.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return faces, labels, report


def load_administrative_coverage(directory):
    """Read the canonical WKB faces and exact native province/state bindings."""
    directory = Path(directory)
    report = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if report.get('schema') != 'administrative-arrival-stage-v2' or report.get('status') != 'complete':
        raise ValueError('administrative coverage requires a complete current arrival stage')
    with np.load(directory/'shared-coverage.npz', allow_pickle=False) as source:
        data = source['wkb'].tobytes(); offsets = source['offsets']
        faces = tuple(shapely.from_wkb([data[int(a):int(b)] for a,b in zip(offsets, offsets[1:])]))
        return faces, source['province_ids'].copy(), source['province_to_state'].copy()
