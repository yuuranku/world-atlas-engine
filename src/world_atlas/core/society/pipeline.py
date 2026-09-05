"""One-way causal society generation from canonical physical geography."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import logging

import numpy as np

from ..model import WorldGrid
from ..thematic import ThematicLayers
from .culture import align_culture_names, assign_culture_lineages, derive_cultures
from .administrative_centres import derive_administrative_centres
from .institutions import derive_state_formation_profiles
from .model import (
    CultureLayers,
    NameLexicon,
    PoliticalLayers,
    ProvinceLayers,
    ReligionLayers,
    SocietyLayers,
    TransportLayers,
)
from .names import extract_geographic_features, load_name_lexicon, name_settlements
from .politics import derive_politics
from .population import derive_population, derive_settlements
from .provinces import derive_provinces
from .religion import derive_religions
from .strategic_sites import derive_strategic_sites, promote_border_settlements
from .transport import conform_transport_to_politics, derive_transport
from .world_identity import assign_world_identity


def derive_society_layers(
    grid: WorldGrid,
    thematic: ThematicLayers,
    name_source: str | Path | NameLexicon,
    *,
    settlement_count: int | None = None,
    civilization_count: int | None = None,
    minimum_civilization_count: int | None = None,
    language_count: int | None = None,
    religion_count: int | None = None,
    state_count: int | None = None,
    population_min: int = 450_000_000,
    population_max: int = 600_000_000,
) -> SocietyLayers:
    """Derive every human layer without reading legacy geography or politics."""

    progress = logging.getLogger(__name__)
    progress.info("Society: population and settlement sites")
    population = derive_population(
        grid,
        thematic,
        population_min=population_min,
        population_max=population_max,
    )
    lexicon = (
        name_source
        if isinstance(name_source, NameLexicon)
        else load_name_lexicon(name_source)
    )
    settlements = derive_settlements(
        grid,
        thematic,
        population,
        target_count=settlement_count,
    )
    if not settlements:
        unassigned = np.where(grid.water == 0, 0, -1).astype(np.int16)
        cultures = CultureLayers(
            civilization_id=unassigned,
            civilization_influence=np.zeros(grid.shape, dtype=np.uint8),
            language_family_id=unassigned,
            language_id=unassigned,
            language_contact=np.zeros(grid.shape, dtype=bool),
            civilizations=(),
            languages=(),
        )
        politics = PoliticalLayers(
            state_id=unassigned,
            frontier=np.zeros(grid.shape, dtype=bool),
            states=(),
            government_forms=(),
            political_entities=(),
            frontier_groups=(),
        )
        return SocietyLayers(
            population=population,
            settlements=(),
            transport=TransportLayers(
                accessibility=np.zeros(grid.shape, dtype=np.float32),
                routes=(),
            ),
            cultures=cultures,
            religions=ReligionLayers(
                religion_id=np.where(grid.water == 0, 0, -1).astype(np.int16),
                religions=(),
            ),
            geographic_features=(),
            politics=politics,
            provinces=ProvinceLayers(
                province_id=np.where(grid.water == 0, 0, -1).astype(np.int32),
                provinces=(),
            ),
        )
    progress.info("Society: initial transport, %d settlements", len(settlements))
    transport = derive_transport(grid, thematic, population, settlements)
    progress.info("Society: civilizations and languages")
    cultures = derive_cultures(
        grid,
        thematic,
        population,
        settlements,
        transport=transport,
        civilization_count=civilization_count,
        minimum_civilization_count=minimum_civilization_count,
        language_count=language_count,
        lexicon=lexicon,
    )
    cultures = assign_culture_lineages(cultures, settlements, grid, thematic)
    settlements = name_settlements(settlements, cultures, lexicon, grid, thematic)
    cultures = align_culture_names(cultures, settlements)
    progress.info("Society: religions and holy cities")
    religions, settlements = derive_religions(
        grid,
        thematic,
        population,
        settlements,
        cultures,
        transport,
        religion_count=religion_count,
    )
    # Holy-city identity is now part of route demand. Recompute the network
    # before states and provinces derive their expansion costs from it.
    transport = derive_transport(grid, thematic, population, settlements)
    state_formation_profiles = derive_state_formation_profiles(
        grid,
        thematic,
        population,
        settlements,
        cultures,
        transport,
    )
    progress.info("Society: countries")
    politics = derive_politics(
        grid,
        thematic,
        population,
        settlements,
        cultures,
        transport,
        lexicon,
        state_count=state_count,
        state_formation_profiles=state_formation_profiles,
    )
    transport = conform_transport_to_politics(
        grid,
        thematic,
        settlements,
        transport,
        politics,
    )
    for refinement_pass in range(2):
        progress.info("Society: administrative network refinement %d", refinement_pass + 1)
        administrative_centres = derive_administrative_centres(
            grid,
            thematic,
            population,
            settlements,
            transport,
            politics,
            maximum_additions=72 if refinement_pass == 0 else 36,
        )
        if not administrative_centres:
            break
        settlements = name_settlements(
            settlements + administrative_centres,
            cultures,
            lexicon,
            grid,
            thematic,
        )
        transport = derive_transport(
            grid,
            thematic,
            population,
            settlements,
        )
        state_formation_profiles = derive_state_formation_profiles(
            grid,
            thematic,
            population,
            settlements,
            cultures,
            transport,
        )
        politics = derive_politics(
            grid,
            thematic,
            population,
            settlements,
            cultures,
            transport,
            lexicon,
            state_count=len(politics.states),
            state_formation_profiles=state_formation_profiles,
        )
        transport = conform_transport_to_politics(
            grid,
            thematic,
            settlements,
            transport,
            politics,
        )
    settlements = promote_border_settlements(
        grid,
        thematic,
        cultures,
        politics,
        transport,
        settlements,
    )
    strategic_sites = derive_strategic_sites(
        grid,
        thematic,
        population,
        settlements,
        cultures,
        transport,
        politics,
    )
    if strategic_sites:
        settlements = name_settlements(
            settlements + strategic_sites,
            cultures,
            lexicon,
            grid,
            thematic,
        )
    progress.info("Society: provinces")
    provinces = derive_provinces(
        grid,
        thematic,
        population,
        settlements,
        transport,
        cultures,
        politics,
    )
    progress.info("Society: geographic names and canonical world identity")
    features = extract_geographic_features(grid, thematic, cultures, lexicon)
    society = SocietyLayers(
        population=population,
        settlements=settlements,
        transport=transport,
        cultures=cultures,
        religions=religions,
        geographic_features=features,
        politics=politics,
        provinces=provinces,
    )
    request = grid.metadata.get("societyGeneration", {})
    if isinstance(request, Mapping) and request.get("namingProfile") == "procedural":
        society = assign_world_identity(
            society, seed=int(request["seed"]), forbidden=request.get("forbiddenNames", ()),
        )
    return society
