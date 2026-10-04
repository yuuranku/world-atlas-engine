"""Shared administrative drawing from authoritative institutional ownership.

Formation allocates countries and provinces. Drawing reads those observations;
it never replaces political control with another race from administrative seats.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import shapely

from ..coastal_partition import extend_coastal_partition
from ..raster_topology import categorical_coverage
from .population import cell_areas_km2


@dataclass(frozen=True, slots=True)
class AdministrativeSource:
    province_id: np.ndarray
    province_to_state: np.ndarray


def administrative_source(grid, society):
    """Snapshot formed provinces and their sole national hierarchy."""
    land = np.asarray(grid.water) == 0
    provinces = np.asarray(society.provinces.province_id)
    states = np.asarray(society.politics.state_id)
    if provinces.shape != land.shape or states.shape != land.shape:
        raise ValueError('administrative observations must match the physical grid')
    records = society.provinces.provinces
    parent = np.zeros(max((record.identifier for record in records), default=0)+1, dtype=np.int16)
    for record in records:
        if record.identifier <= 0 or parent[record.identifier] != 0:
            raise ValueError('province records require unique positive identifiers')
        parent[record.identifier] = record.state_identifier
    positive = provinces > 0
    if np.any(provinces[positive] >= len(parent)) or np.any(parent[provinces[positive]] <= 0):
        raise ValueError('formed province observations require recorded national parents')
    if np.any((states > 0) != positive) or np.any(parent[np.maximum(provinces[land], 0)] != states[land]):
        raise ValueError('formed countries must equal the union of their recorded provinces')
    province_id = np.where(land, np.maximum(provinces, 0), -1).astype(np.int32)
    province_id.setflags(write=False)
    parent.setflags(write=False)
    return AdministrativeSource(province_id, parent)


def administrative_paint_coverage(source, land):
    """Trace formed province observations with a water-only working band.

    The physical shore clips this paint in the renderer. Country fill and ink
    derive from the same province faces via ``province_to_state``.
    """
    land = np.asarray(land, dtype=bool)
    if land.shape != source.province_id.shape:
        raise ValueError('administrative paint and physical land must share a shape')
    if not np.any(land):
        return (), np.empty(0, dtype=np.int32)
    extended, domain = extend_coastal_partition(source.province_id, land,
        margin=4., wrap_longitude=True)
    return categorical_coverage(extended, domain, category_count=len(source.province_to_state))


def synchronize_administrative_statistics(grid, society, *, state_id, province_id):
    """Bind ownership, population and province statistics to formed arrays."""
    states = np.asarray(state_id, dtype=np.int16)
    provinces = np.asarray(province_id, dtype=np.int32)
    updated = replace(society, politics=replace(society.politics, state_id=states),
                      provinces=replace(society.provinces, province_id=provinces))
    source = administrative_source(grid, updated)
    land = np.asarray(grid.water) == 0
    weights = society.population.population_weight
    state_weight = np.bincount(np.maximum(states, 0).ravel(),
        weights=weights.ravel(), minlength=max(record.identifier for record in society.politics.states)+1)
    state_records = []
    for record in society.politics.states:
        share = float(state_weight[record.identifier])
        lower = max(1000, round(share*society.population.population_min/10000)*10000)
        upper = max(lower+1000, round(share*society.population.population_max/10000)*10000)
        state_records.append(replace(record, population_min=lower, population_max=upper))
    counts = np.bincount(np.maximum(provinces, 0).ravel(), minlength=len(source.province_to_state))
    areas = cell_areas_km2(grid)
    controlled = states > 0
    mean_density = float(weights[controlled].sum()/areas[controlled].sum()) if controlled.any() else 0.
    province_records = []
    for record in society.provinces.provinces:
        region = provinces == record.identifier
        area = float(areas[region].sum())
        ratio = float(weights[region].sum()/area) / max(mean_density, 1e-15) if area else 0.
        province_records.append(replace(record, area_cells=int(counts[record.identifier]),
            population_density_class='dense' if ratio >= 1.35 else 'settled' if ratio >= .62 else 'sparse'))
    return replace(updated,
        politics=replace(updated.politics, frontier=land & (states <= 0), states=tuple(state_records)),
        provinces=replace(updated.provinces, provinces=tuple(province_records)))


def write_administrative_stage(directory, grid, society, source):
    """Persist formed native observations and their one shared coverage."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    faces, labels = administrative_paint_coverage(source, grid.water == 0)
    parent = source.province_to_state
    tree = shapely.STRtree(faces)
    wrong = missing = overlap = 0
    height, width = grid.shape
    for north in range(0, height, 32):
        rows, cols = np.indices((min(32, height-north), width))
        dry = np.asarray(grid.water[north:north+len(rows)] == 0).ravel()
        points = shapely.points((cols+.5).ravel()[dry], (rows+north+.5).ravel()[dry])
        point_ids, face_ids = tree.query(points, predicate='covered_by')
        expected = np.maximum(source.province_id[north:north+len(rows)], 0).ravel()[dry]
        good = labels[face_ids] == expected[point_ids]
        covered = np.bincount(point_ids[good], minlength=len(points))
        missing += int(np.count_nonzero(covered == 0))
        overlap += int(np.count_nonzero(covered > 1))
        wrong += int(np.count_nonzero(~good))
    if wrong or missing or overlap:
        raise ValueError(f'administrative formed coverage native mismatch: {wrong=}, {missing=}, {overlap=}')
    wkb = shapely.to_wkb(faces)
    offsets = np.r_[0, np.cumsum([len(item) for item in wkb], dtype=np.int64)]
    data = np.frombuffer(b''.join(wkb), dtype=np.uint8)
    fingerprint = hashlib.sha256()
    for value in (source.province_id, parent, data, labels):
        fingerprint.update(np.ascontiguousarray(value).tobytes())
    np.savez_compressed(directory/'shared-coverage.npz', wkb=data, offsets=offsets,
        native_province_id=source.province_id, province_ids=labels,
        state_ids=parent[labels], province_to_state=parent)
    report = dict(schema='administrative-ownership-stage-v1', status='complete',
        source='formed-institutional-province-ownership', fingerprint=fingerprint.hexdigest(),
        gridDigest=grid.content_digest(), shape=list(grid.shape),
        faces=len(faces), states=len(society.politics.states), provinces=len(society.provinces.provinces),
        nativeCentresClassified=height*width, nativeLandCentresChecked=int(np.count_nonzero(grid.water == 0)),
        nativeWaterCentresExcludedByPhysicalClip=int(np.count_nonzero(grid.water != 0)),
        nativeWrong=wrong, nativeMissing=missing, nativeOverlap=overlap,
        coverageValid=bool(shapely.coverage_is_valid(faces)), frameArea=float(shapely.union_all(faces).area),
        ownership='formed province observations; countries are parent unions; fill and ink share these faces')
    (directory/'manifest.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return faces, labels, report


def load_administrative_coverage(directory):
    """Read current shared geometry and exact province-to-state bindings."""
    directory = Path(directory)
    report = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if report.get('schema') != 'administrative-ownership-stage-v1' or report.get('status') != 'complete':
        raise ValueError('administrative coverage requires a complete current ownership stage')
    with np.load(directory/'shared-coverage.npz', allow_pickle=False) as source:
        data = source['wkb'].tobytes(); offsets = source['offsets']
        faces = tuple(shapely.from_wkb([data[int(a):int(b)] for a,b in zip(offsets, offsets[1:])]))
        return faces, source['province_ids'].copy(), source['province_to_state'].copy()
