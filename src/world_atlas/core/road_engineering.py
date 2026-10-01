"""Metric road construction and travel on the world's saved physical ground."""

from dataclasses import dataclass
import math

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
# These are explicit simulation design standards, not universal historical laws.
# An unengineered cart road has less capacity for sustained steep grades.
ROAD_STANDARDS = {
    "tribal": (.035, .10), "ancient": (.04, .10), "medieval": (.04, .10),
    "early-modern": (.045, .10), "preindustrial": (.045, .10),
    "industrial": (.055, .12), "contemporary": (.06, .12),
}


@dataclass(frozen=True)
class RoadCosts:
    values: np.ndarray
    heuristic_scale: float
    radius_km: float
    minimum_friction: float

    def heuristic(self, goal):
        h,w=self.values.shape[:2]
        lat=math.pi/2-(np.arange(h)+.5)*math.pi/h
        lon=(np.arange(w)+.5)*2*math.pi/w
        cy,sy=tuple(np.cos(lat)),tuple(np.sin(lat))
        cx,sx=tuple(np.cos(lon)),tuple(np.sin(lon))
        r,c=goal;gx,gy,gz=cy[r]*cx[c],cy[r]*sx[c],sy[r]
        factor=self.radius_km*self.minimum_friction
        return lambda point: factor*math.sqrt((cy[point[0]]*cx[point[1]]-gx)**2+
            (cy[point[0]]*sx[point[1]]-gy)**2+(sy[point[0]]-gz)**2)


def segment_lengths_km(points, shape, radius_km):
    """Great-circle length of each actual segment, including longitude wrap."""
    p = np.asarray(points, dtype=float)
    lon = p[:, 0] * (2 * math.pi / shape[1])
    lat = math.pi / 2 - p[:, 1] * (math.pi / shape[0])
    dlon, dlat = np.diff(lon), np.diff(lat)
    a = np.sin(dlat / 2)**2 + np.cos(lat[:-1])*np.cos(lat[1:])*np.sin(dlon / 2)**2
    return 2 * radius_km * np.arctan2(np.sqrt(np.clip(a, 0, 1)), np.sqrt(np.clip(1-a, 0, 1)))


@dataclass
class RoadEngineering:
    elevation_m: np.ndarray
    radius_km: float
    era: str

    def __post_init__(self):
        self.elevation_m = np.asarray(self.elevation_m, dtype=float)
        if (self.elevation_m.ndim != 2 or not np.isfinite(self.elevation_m).all()
                or self.era not in ROAD_STANDARDS or not math.isfinite(self.radius_km) or self.radius_km <= 0):
            raise ValueError("road engineering requires metric ground, planet radius and a supported era")
        self.preferred_grade, self.maximum_grade = ROAD_STANDARDS[self.era]

    def edge_costs(self, friction):
        """Bidirectional construction cost; inaccessible grades have infinite cost.

        Friction describes ground occupation. Grade describes travel direction.
        A road that traverses a hillside therefore differs from one climbing it.
        """
        h, w = self.elevation_m.shape
        latitude = math.pi/2-(np.arange(h)+.5)*math.pi/h
        costs = np.full((h, w, 8), np.inf, dtype=np.float64)
        for direction, (dy, dx) in enumerate(DIRECTIONS):
            rows = np.arange(max(0, -dy), min(h, h-dy))
            other = rows+dy
            a = np.sin(dy*math.pi/h/2)**2 + np.cos(latitude[rows])*np.cos(latitude[other])*np.sin(dx*math.pi/w)**2
            distance = 2*self.radius_km*np.arctan2(np.sqrt(np.clip(a, 0, 1)), np.sqrt(np.clip(1-a, 0, 1)))
            target_z = np.roll(self.elevation_m[other], -dx, axis=1)
            grade = np.abs(target_z-self.elevation_m[rows])/(distance[:, None]*1000)
            target_f = np.roll(friction[other], -dx, axis=1)
            cost = distance[:, None]*(.5*(friction[rows]+target_f)+4*(grade/self.preferred_grade)**2)
            costs[rows, :, direction] = np.where(grade <= self.maximum_grade, cost, np.inf)
        steps = np.asarray([math.hypot(dy, dx) for dy, dx in DIRECTIONS])
        lower = float(np.min(np.divide(costs, steps), where=np.isfinite(costs), initial=math.inf))
        return RoadCosts(costs, lower,self.radius_km,float(np.min(friction)))

    def profile(self, points, *, terrain_field):
        """Sample a candidate line on the same continuous ground used by the map."""
        p = np.asarray(points, dtype=float)
        pieces = []
        for first, last in zip(p[:-1], p[1:], strict=True):
            delta = last-first
            delta[0] = (delta[0]+self.elevation_m.shape[1]/2)%self.elevation_m.shape[1]-self.elevation_m.shape[1]/2
            count = max(1, math.ceil(float(np.max(np.abs(delta)))*64))
            pieces.append(first+np.arange(count)[:, None]/count*delta)
        sampled = np.vstack((*pieces, p[-1:]))
        height = terrain_field.sample_points(sampled[:, 0], sampled[:, 1])
        lengths = segment_lengths_km(sampled, self.elevation_m.shape, self.radius_km)
        changes = np.diff(height)
        grade = np.divide(changes, lengths*1000, out=np.zeros_like(changes), where=lengths > 1e-12)
        return sampled, height, lengths, grade

    def preserves_profile(self, original, proposed, *, terrain_field):
        """A display chord cannot introduce a steeper or more costly climb."""
        _, _, old_d, old_g = self.profile(original, terrain_field=terrain_field)
        _, _, new_d, new_g = self.profile(proposed, terrain_field=terrain_field)
        old_peak = np.max(np.abs(old_g), initial=0)
        new_peak = np.max(np.abs(new_g), initial=0)
        old_cost = np.sum(old_d*(1+4*(old_g/self.preferred_grade)**2))
        new_cost = np.sum(new_d*(1+4*(new_g/self.preferred_grade)**2))
        return bool(new_peak <= old_peak+1e-6 and new_cost <= old_cost*(1+1e-6))
