"""Strict two-file persistence for generated society layers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .model import (
    Bridge,
    Civilization,
    CultureLayers,
    FrontierGroup,
    GeographicFeature,
    GovernmentForm,
    Language,
    PoliticalLayers,
    PoliticalEntity,
    PopulationLayers,
    Province,
    ProvinceLayers,
    Religion,
    ReligionLayers,
    Settlement,
    SocietyLayers,
    State,
    TransportLayers,
    TransportRoute,
)


_FORMAT = "eirenor-society"
_SCHEMA = "society-v7"
_ARRAYS = (
    "population_weight",
    "population_band",
    "civilization_id",
    "civilization_influence",
    "language_family_id",
    "language_id",
    "language_contact",
    "religion_id",
    "state_id",
    "province_id",
    "frontier",
    "transport_accessibility",
)


def save_society(
    society: SocietyLayers,
    output_directory: str | Path,
    *,
    grid_digest: str,
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    write_society_files(
        society,
        output / "society.npz",
        output / "society.json",
        grid_digest=grid_digest,
    )


def write_society_files(
    society: SocietyLayers,
    npz_path: str | Path,
    json_path: str | Path,
    *,
    grid_digest: str,
) -> None:
    """Write the strict payload to explicitly owned paths."""

    arrays = {
        "population_weight": society.population.population_weight,
        "population_band": society.population.population_band,
        "civilization_id": society.cultures.civilization_id,
        "civilization_influence": society.cultures.civilization_influence,
        "language_family_id": society.cultures.language_family_id,
        "language_id": society.cultures.language_id,
        "language_contact": society.cultures.language_contact,
        "religion_id": society.religions.religion_id,
        "state_id": society.politics.state_id,
        "province_id": society.provinces.province_id,
        "frontier": society.politics.frontier,
        "transport_accessibility": society.transport.accessibility,
    }
    with Path(npz_path).open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    metadata = {
        "format": _FORMAT,
        "schema": _SCHEMA,
        "gridDigest": str(grid_digest),
        "populationRange": [
            society.population.population_min,
            society.population.population_max,
        ],
        "settlements": [asdict(item) for item in society.settlements],
        "transportRoutes": [asdict(item) for item in society.transport.routes],
        "bridges": [asdict(item) for item in society.transport.bridges],
        "civilizations": [asdict(item) for item in society.cultures.civilizations],
        "languages": [asdict(item) for item in society.cultures.languages],
        "religions": [asdict(item) for item in society.religions.religions],
        "geographicFeatures": [asdict(item) for item in society.geographic_features],
        "states": [asdict(item) for item in society.politics.states],
        "governmentForms": [
            asdict(item) for item in society.politics.government_forms
        ],
        "politicalEntities": [
            asdict(item) for item in society.politics.political_entities
        ],
        "frontierGroups": [
            asdict(item) for item in society.politics.frontier_groups
        ],
        "provinces": [asdict(item) for item in society.provinces.provinces],
    }
    Path(json_path).write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def load_society(
    input_directory: str | Path,
    *,
    expected_grid_digest: str | None = None,
) -> SocietyLayers:
    source = Path(input_directory)
    metadata = json.loads((source / "society.json").read_text(encoding="utf-8"))
    if metadata.get("format") != _FORMAT or metadata.get("schema") != _SCHEMA:
        raise ValueError("unsupported society bundle")
    if expected_grid_digest is not None and metadata.get("gridDigest") != expected_grid_digest:
        raise ValueError("society bundle belongs to a different WorldGrid")
    with np.load(source / "society.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(_ARRAYS):
            raise ValueError("society archive has an invalid array set")
        arrays = {name: np.array(archive[name], copy=True) for name in _ARRAYS}
    population_range = metadata["populationRange"]
    population = PopulationLayers(
        population_weight=arrays["population_weight"],
        population_band=arrays["population_band"],
        population_min=int(population_range[0]),
        population_max=int(population_range[1]),
    )
    cultures = CultureLayers(
        civilization_id=arrays["civilization_id"],
        civilization_influence=arrays["civilization_influence"],
        language_family_id=arrays["language_family_id"],
        language_id=arrays["language_id"],
        language_contact=arrays["language_contact"],
        civilizations=tuple(Civilization(**item) for item in metadata["civilizations"]),
        languages=tuple(Language(**item) for item in metadata["languages"]),
    )
    religions = ReligionLayers(
        religion_id=arrays["religion_id"],
        religions=tuple(Religion(**item) for item in metadata["religions"]),
    )
    politics = PoliticalLayers(
        state_id=arrays["state_id"],
        frontier=arrays["frontier"],
        states=tuple(State(**item) for item in metadata["states"]),
        government_forms=tuple(
            GovernmentForm(**item) for item in metadata["governmentForms"]
        ),
        political_entities=tuple(
            PoliticalEntity(**item) for item in metadata["politicalEntities"]
        ),
        frontier_groups=tuple(
            FrontierGroup(**item) for item in metadata["frontierGroups"]
        ),
    )
    transport = TransportLayers(
        accessibility=arrays["transport_accessibility"],
        routes=tuple(
            TransportRoute(
                **{
                    **item,
                    "path": tuple(tuple(point) for point in item["path"]),
                }
            )
            for item in metadata["transportRoutes"]
        ),
        bridges=tuple(Bridge(**item) for item in metadata["bridges"]),
    )
    provinces = ProvinceLayers(
        province_id=arrays["province_id"],
        provinces=tuple(Province(**item) for item in metadata["provinces"]),
    )
    return SocietyLayers(
        population=population,
        settlements=tuple(Settlement(**item) for item in metadata["settlements"]),
        transport=transport,
        cultures=cultures,
        religions=religions,
        geographic_features=tuple(
            GeographicFeature(**item) for item in metadata["geographicFeatures"]
        ),
        politics=politics,
        provinces=provinces,
    )
