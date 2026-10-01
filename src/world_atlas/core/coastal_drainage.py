"""Connected drowned valleys cut from the existing drainage surface."""
import numpy as np
from scipy import ndimage


def shoreline_distance(inside_distance, outside_distance):
    """Combine complementary distances, whose source-side values are zero."""
    return np.maximum(inside_distance, outside_distance)


def drown_coastal_valleys(surface, accumulation, quiet, cell_km):
    """Approximate transgression into eroded valleys, without new lake holes.

    The existing catchments control cuts; a low-frequency coast noise does
    not prescribe their shape. Ocean-connected flooding is mandatory.
    """
    z = np.asarray(surface, dtype=np.float32)
    land = z > 0
    labels, count = ndimage.label(~land)
    if not count or not land.any():
        return z.copy(), {"drownedCells": 0}
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    ocean = labels == int(np.argmax(sizes))
    distance = ndimage.distance_transform_edt(~ocean) * cell_km
    a = np.maximum(np.asarray(accumulation), 0)
    flow = np.clip(np.log1p(a) / np.log(90.0), 0, 1)
    valley = np.maximum(flow, ndimage.gaussian_filter(flow, 1.2) * 0.72)
    support = np.exp(-0.5 * (distance / 130.0)**2)
    # Quiet depositional plains resist drowning; rocky, dissected coasts
    # retain stronger branching valley relief.
    depth = (95.0 + 260.0 * valley**1.6) * support
    depth *= (0.30 + 0.70 * flow) * (1 - 0.65 * np.clip(quiet, 0, 1))
    depth *= land
    proposed = z - depth
    wet = ndimage.binary_propagation(ocean, mask=ocean | (proposed <= 0))
    isolated = land & (proposed <= 0) & ~wet
    proposed[isolated] = z[isolated]
    return proposed.astype(np.float32), {
        "model": "drainage-controlled-connected-transgression",
        "drownedCells": int(np.count_nonzero(land & wet)),
        "maximumLoweringMeters": float(np.max(z - proposed)),
    }
