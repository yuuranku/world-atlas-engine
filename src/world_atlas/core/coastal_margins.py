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
    # Relay zones need unequal spacing and unequal displacement. The former
    # fixed seven-knot profile made a long coast look like one softened arc
    # after displacement. Seeded controls preserve finite fault blocks while
    # treating sea and lake shores by the same continuous-surface rule.
    knots = np.array([-1.0, -.77, -.52, -.22, .07, .34, .63, .84, 1.0])
    knots[1:-1] += rng.uniform(-.075, .075, len(knots) - 2)
    knots[1:-1] = np.sort(knots[1:-1])
    if family in ('rift','transform'):
        template = np.array([0.0, .30, .92, .58, -.46, -.96, -.28, .24, 0.0])
    elif family == 'active':
        template = np.array([0.0, -.28, -.76, .18, .94, .62, -.36, -.60, 0.0])
    else:
        template = np.array([0.0, .22, .78, .40, -.54, -.82, -.18, .30, 0.0])
    offsets = template.copy()
    offsets[1:-1] += rng.uniform(-.28, .28, len(offsets) - 2)
    offsets[1:-1] *= rng.uniform(.72, 1.30, len(offsets) - 2)
    offsets *= float(rng.choice((-1,1))) * slip_km
    # Bound strain, not just absolute slip: a short sector must not fold a
    # shoreline into a row of teeth. Smoothstep's maximum derivative is 1.5.
    along_strain = 1.5 * np.max(np.abs(np.diff(offsets)) / np.diff(knots)) / length_km
    transverse_strain = 1.54 * np.max(np.abs(offsets)) / influence_km
    offsets *= min(1.0, 0.45 / max(along_strain + transverse_strain, 1.0e-12))
    transverse = np.square(np.clip(1-np.square(y/influence_km),0,1))
    position = x / length_km
    segment = np.clip(np.searchsorted(knots, position, side="right") - 1, 0, len(knots) - 2)
    fraction = np.clip(
        (position - knots[segment]) / np.maximum(knots[segment + 1] - knots[segment], 1.0e-9),
        0.0,
        1.0,
    )
    smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    normal_profile = offsets[segment] + (
        offsets[segment + 1] - offsets[segment]
    ) * smooth_fraction
    normal_profile = np.where(
        position < knots[0],
        0.0,
        np.where(position > knots[-1], 0.0, normal_profile),
    )
    normal = normal_profile * transverse
    tangent = normal * (.70 if family == 'transform' else .18)
    support = transverse * np.clip(1-np.abs(x/length_km),0,1)
    return normal.astype(np.float32), tangent.astype(np.float32), support.astype(np.float32)
