"""Physical-coordinate displacement of structural coastal sectors."""
from __future__ import annotations

import numpy as np


def motion_budget_km(speed_cm_per_year, duration_myr):
    """Bound reactivation without flattening different speeds to one clip value."""
    distance = np.asarray(speed_cm_per_year) * 10. * duration_myr
    return 280. * -np.expm1(-distance / 280.)


def margin_motion(along_km, inland_km, length_km, influence_km, slip_km, *, seed, family):
    """Finite structural offset of a whole margin, rather than a water stamp.

    Knots are relay/transfer zones; between them motion is affine, so a long
    smooth coast gains changes of direction at continental scale. Compact
    transverse support transports the coast together with its local relief.
    """
    if family not in ('rift','active','passive','transform'):
        raise ValueError(f'unknown margin motion family: {family}')
    rng = np.random.Generator(np.random.PCG64(int(seed) & 0xFFFFFFFFFFFFFFFF))
    x, y = np.broadcast_arrays(np.asarray(along_km,dtype=float), np.asarray(inland_km,dtype=float))
    knots = np.array([-1.,-.68,-.30,.08,.30,.70,1.])
    knots[1:-1] += rng.uniform(-.065,.065,5)
    if family in ('rift','transform'):
        offsets = np.array([0,.35,1.0,.82,-.74,-.45,0])
    elif family == 'active':
        offsets = np.array([0,-.30,-.72,.75,1.0,.32,0])
    else:
        offsets = np.array([0,.30,.85,-.42,-.68,-.20,0])
    offsets[1:-1] *= rng.uniform(.88,1.12,5)
    offsets *= float(rng.choice((-1,1))) * slip_km
    transverse = np.square(np.clip(1-np.square(y/influence_km),0,1))
    normal = np.interp(x/length_km,knots,offsets,left=0,right=0) * transverse
    tangent = normal * (.70 if family == 'transform' else .18)
    support = transverse * np.clip(1-np.abs(x/length_km),0,1)
    return normal.astype(np.float32), tangent.astype(np.float32), support.astype(np.float32)
