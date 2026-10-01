"""Allocate a country's urban population after administrative centres exist."""

from collections import Counter, defaultdict
from dataclasses import replace
import math

import numpy as np


def allocate_urban_population(grid, society):
    """Share finite country budgets using hierarchy, hinterland and transport.

    The population raster remains the total population, including these urban
    residents. Settlement figures are a subset, never additional headcounts.
    Allocation does not use old settlement figures and is safe to recompute.
    """
    era = grid.metadata['worldProfile']['technologyEra']
    urban_fraction = {'tribal': .015, 'ancient': .08, 'medieval': .07,
                      'early-modern': .11, 'preindustrial': .10,
                      'industrial': .35, 'contemporary': .72}[era]
    capitals = {s.core_settlement_id for s in society.politics.states}
    regional = {p.core_settlement_id for p in society.provinces.provinces}
    connections = defaultdict(Counter)
    for route in society.transport.routes:
        for identifier in {route.source_settlement_id, route.target_settlement_id}-{None}:
            connections[identifier][route.mode] += 1
    groups = defaultdict(list)
    weights = society.population.population_weight
    height, width = grid.shape
    for city in society.settlements:
        owner = int(society.politics.state_id[city.row, city.column])
        nearby = weights[max(0,city.row-4):min(height,city.row+5),
                         np.arange(city.column-4,city.column+5)%width]
        food = float(nearby.mean())
        access = float(society.transport.accessibility[city.row, city.column])
        rank = .2+max(.2,city.score)**4
        institution = 3.6 if city.identifier in capitals else 1.55 if city.identifier in regional else 1.
        trade = 1+min(1.8, connections[city.identifier]['sea']*.18
                     +connections[city.identifier]['river']*.15+connections[city.identifier]['road']*.07)
        water = 1.25 if city.site_type in {'river-city','port','island-port','lake-port'} else .8 if city.site_type=='oasis' else 1.
        capacity = math.sqrt(max(food,1e-12))*(.5+access)*water
        merit = rank*institution*trade*capacity*max(.2,city.score)**1.4
        groups[owner].append((city,merit))
    countries = {s.identifier:s for s in society.politics.states}
    updated = {}
    for owner, cities in groups.items():
        total = sum(merit for _,merit in cities)
        if owner in countries:
            state = countries[owner]
            minimum,maximum = state.population_min,state.population_max
        else:
            fraction = float(weights[society.politics.state_id==owner].sum())/max(1.,float(weights.sum()))
            minimum = society.population.population_min*fraction
            maximum = society.population.population_max*fraction
        for city,merit in cities:
            share = urban_fraction*merit/total
            low = int(minimum*share//10)*10
            high = max(low,int(maximum*share//10)*10)
            estimate=(low+high)/2
            tier='metropolis' if estimate>=80000 else 'city' if estimate>=10000 else 'town' if estimate>=1800 else 'site'
            updated[city.identifier] = replace(city,population_min=low,population_max=high,tier=tier)
    return replace(society,settlements=tuple(updated[s.identifier] for s in society.settlements))
