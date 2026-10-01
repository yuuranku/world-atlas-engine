"""Continuous administrative arrival fronts, before categorical ownership.

Two distinct competitors per observation suffice for a shared, owner-independent
travel metric. No country-count-sized matrix or geometric boundary noise is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import numpy as np

from .territorial_simulation import TerritorySimulation, TerritorySeed, _NEIGHBORS


@dataclass(frozen=True, slots=True)
class AdministrativeFront:
    owners: np.ndarray
    arrivals: np.ndarray
    edge_costs: np.ndarray
    valid: np.ndarray
    absent_time: float
    parent: np.ndarray | None
    domains: dict[int, int]

    @property
    def owner(self):
        return self.owners[0]


def administrative_front(simulation: TerritorySimulation, seeds: tuple[TerritorySeed, ...], *, maritime_links=()) -> AdministrativeFront:
    """Solve the first two distinct-owner arrival times on the same terrain.

    Each candidate obeys the actual edge crossing cost. Accepted same-owner
    axial fronts solve the isotropic, causal two-dimensional Eikonal update.
    The metric is shared: seed strength must be one, and a third owner at an
    observation cannot outrun its two predecessors by following that point.
    Unseeded components remain unowned; there is no proximity allocation.
    """
    shape = simulation.valid.shape
    height, width = shape
    friction = simulation.friction.astype(np.float64)*(1-.15*simulation.road_access)
    transitions = simulation.transition_penalty.astype(np.float64)*(1-simulation.bridge_edges)
    edges = np.empty((8, *shape), dtype=np.float64)
    for direction, (dy, dx, _) in enumerate(_NEIGHBORS):
        edges[direction] = .5*(friction+np.roll(friction, (-dy,-dx), (0,1)))+transitions[direction]
    owners = np.zeros((2, *shape), dtype=np.int32)
    arrivals = np.full((2, *shape), math.inf)
    accepted = np.zeros((2, *shape), dtype=bool)
    domains = {}
    pending = []
    ports={}
    for first,last,cost in maritime_links:
        if not np.isfinite(cost) or cost<=0:
            raise ValueError('maritime arrival links require a positive actual route cost')
        ports.setdefault(tuple(first),[]).append((*last,float(cost)))
        ports.setdefault(tuple(last),[]).append((*first,float(cost)))

    def offer(row, column, owner, time):
        if owner == owners[0,row,column]:
            rank = 0
        elif owner == owners[1,row,column]:
            rank = 1
        elif (time,owner) < (arrivals[1,row,column], int(owners[1,row,column])):
            rank = 1
        else:
            return
        if accepted[rank,row,column] or time >= arrivals[rank,row,column]:
            return
        owners[rank,row,column] = owner
        arrivals[rank,row,column] = time
        if (arrivals[1,row,column],int(owners[1,row,column])) < (arrivals[0,row,column],int(owners[0,row,column])):
            owners[:,row,column] = owners[::-1,row,column]
            arrivals[:,row,column] = arrivals[::-1,row,column]
            accepted[:,row,column] = accepted[::-1,row,column]
        heapq.heappush(pending,(time,owner,row,column))

    occupied = set()
    for seed in seeds:
        if seed.strength != 1:
            raise ValueError('administrative competitors require one shared travel metric')
        if seed.row >= height or seed.column >= width or not simulation.valid[seed.row,seed.column]:
            raise ValueError('administrative core must lie in its actual allowed domain')
        if (seed.row,seed.column) in occupied:
            raise ValueError('administrative cores must have distinct native locations')
        occupied.add((seed.row,seed.column))
        if simulation.owner_constraint is not None:
            required=int(simulation.owner_constraint[seed.row,seed.column])
            if required>0 and required!=seed.domain:
                raise ValueError('administrative core does not belong to its parent domain')
        if seed.owner in domains and domains[seed.owner] != seed.domain:
            raise ValueError('one administrative owner cannot have two parent domains')
        domains[seed.owner] = seed.domain
        offer(seed.row,seed.column,seed.owner,0.)

    while pending:
        time, owner, row, column = heapq.heappop(pending)
        rank = 0 if owners[0,row,column] == owner else 1
        if owners[rank,row,column] != owner or arrivals[rank,row,column] != time or accepted[rank,row,column]:
            continue
        accepted[rank,row,column] = True
        for next_row,next_column,cost in ports.get((row,column),()):
            if not simulation.valid[next_row,next_column]:continue
            if simulation.owner_constraint is not None:
                required=int(simulation.owner_constraint[next_row,next_column])
                if required>0 and required!=domains[owner]:continue
            candidate=time+cost
            if (candidate,owner)<(arrivals[0,next_row,next_column],int(owners[0,next_row,next_column])) and rank!=0:
                continue
            offer(next_row,next_column,owner,candidate)
        for direction,(dy,dx,distance) in enumerate(_NEIGHBORS):
            if dy and dx:
                continue
            next_row = row+dy
            if not 0 <= next_row < height:
                continue
            next_column = (column+dx)%width
            if not simulation.valid[next_row,next_column]:
                continue
            if simulation.owner_constraint is not None:
                required = int(simulation.owner_constraint[next_row,next_column])
                if required > 0 and required != domains[owner]:
                    continue
            slowness = edges[direction,row,column]
            candidate = time+slowness*distance
            # Solve the two-axis Eikonal equation on a causal accepted
            # front. Axial paths establish actual administrative contact;
            # the quadratic update retains an isotropic travel metric.
            for peer_dy,peer_dx in (((0,-1),(0,1)) if dy else ((-1,0),(1,0))):
                peer_row = next_row+peer_dy
                if not 0 <= peer_row < height:
                    continue
                peer_column = (next_column+peer_dx)%width
                peer_rank = 0 if owners[0,peer_row,peer_column] == owner else 1
                if owners[peer_rank,peer_row,peer_column] != owner or not accepted[peer_rank,peer_row,peer_column]:
                    continue
                peer_time = arrivals[peer_rank,peer_row,peer_column]
                peer_direction = _NEIGHBORS.index((-peer_dy,-peer_dx,1.))
                bound = max(slowness, edges[peer_direction,peer_row,peer_column])
                difference = abs(time-peer_time)
                if difference < bound:
                    candidate = min(candidate,(time+peer_time+math.sqrt(2*bound*bound-difference*difference))*.5)
            # A runner-up is allowed to cross an occupied region to supply
            # the competing potential, but cannot start a disconnected win.
            if (candidate,owner) < (arrivals[0,next_row,next_column],int(owners[0,next_row,next_column])):
                if owners[0,row,column] != owner:
                    continue
            offer(next_row,next_column,owner,candidate)
    owners[:,~simulation.valid] = -1
    absent_time=float(np.max(arrivals[np.isfinite(arrivals)],initial=0.)+np.max(edges)*4+1.)
    parent=None if simulation.owner_constraint is None else np.array(simulation.owner_constraint,copy=True)
    return AdministrativeFront(owners,arrivals,edges,np.array(simulation.valid,copy=True),absent_time,parent,domains)
