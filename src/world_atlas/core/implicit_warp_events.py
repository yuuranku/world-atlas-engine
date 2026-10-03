"""Transport actual source PCHIP switch sections through accepted W disks.

Every compact ground deformation is a homeomorphism of its own disk and
fixes its boundary. Thus the same disk bounds the inverse image of every
source switch inside it. Source polynomial roots and source native latitude
intervals own these candidates; no sampled inverse coordinate field does.
"""

import numpy as np
import shapely

from .implicit_pchip import pchip_native_switch_abscissae


def _disk_sections(field, patch):
    patches = field._landforms
    centre, radius = patches.centres[patch], float(patches.radius[patch])
    first_column = int(np.floor(centre[0]-radius-.5))
    last_column = int(np.floor(centre[0]+radius-.5))
    first_row = max(-1, int(np.floor(centre[1]-radius-.5)))
    last_row = min(field.height-1, int(np.floor(centre[1]+radius-.5)))
    columns, rows = np.meshgrid(np.arange(first_column, last_column+1),
                                np.arange(first_row, last_row+1))
    columns, rows = columns.ravel(), rows.ravel()
    primary = columns % field.width
    # The primary interval straddling the longitude cut is [-.5,.5]. Keep
    # this source root's fraction; adding then subtracting a world width can
    # otherwise give two periodic copies different floating cache keys.
    primary = np.where(primary == field.width-1, -1, primary)
    periods = columns-primary
    lower = np.column_stack((primary+.5, np.maximum(0., rows+.5)))
    upper = np.column_stack((primary+1.5, np.minimum(field.height, rows+1.5)))
    roots = pchip_native_switch_abscissae(field.base, lower, upper)
    sections, seen = [], set()
    for interval, (events, period) in enumerate(zip(roots, periods, strict=True)):
        for canonical_x in events:
            source_x = float(canonical_x+period)
            distance = source_x-centre[0]
            # Bound the real disk slice outward. These directed-rounding
            # endpoints bound a source parameter domain; they do not relax
            # a geometric delivery/coverage gate or alter the accepted W.
            distance_low = np.nextafter(distance, -np.inf)
            distance_high = np.nextafter(distance, np.inf)
            magnitude_low = (0. if distance_low <= 0 <= distance_high
                             else min(abs(distance_low), abs(distance_high)))
            distance_squared_low = max(0., np.nextafter(magnitude_low*magnitude_low, -np.inf))
            radius_squared_high = np.nextafter(radius*radius, np.inf)
            squared = np.nextafter(radius_squared_high-distance_squared_low, np.inf)
            if squared <= 0:
                continue
            extent = np.nextafter(np.sqrt(squared), np.inf)
            lower_y = max(float(lower[interval, 1]), float(np.nextafter(centre[1]-extent, -np.inf)))
            upper_y = min(float(upper[interval, 1]), float(np.nextafter(centre[1]+extent, np.inf)))
            if lower_y >= upper_y:
                continue
            key = source_x, lower_y, upper_y
            if key in seen:
                continue
            seen.add(key)
            sections.append({"sourceX": source_x, "canonicalSourceX": float(canonical_x),
                             "sourceYLower": lower_y, "sourceYUpper": upper_y,
                             "patchIndex": patch})
    return sections


def warped_pchip_events(field, lower, upper):
    """Return model-owned inverse-W source-switch candidates per physical box.

    ``sourceYLower``/``sourceYUpper`` span at most one original native
    interval. ``patchIndex`` is the accepted field's original periodic disk
    index. The consumer must use its inverse-curve interval oracle and solve
    the joint actual height; a disk-box hit alone is not a contour event.
    This helper covers Base tensor-PCHIP slope switches only, including their
    native endpoint roots. It does not enumerate other gain/profile events.
    """
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.ndim != 2 or lower.shape[1:] != (2,) or upper.shape != lower.shape
            or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)):
        raise ValueError("warped PCHIP events require finite positive-area physical boxes")
    patches = field._landforms
    if not patches.count or not len(lower):
        return []
    boxes = shapely.box(lower[:, 0], lower[:, 1], upper[:, 0], upper[:, 1])
    owners, indices = patches._tree.query(boxes)
    order = np.lexsort((indices, owners))
    cached, result = {}, []
    for owner, patch in zip(owners[order], indices[order], strict=True):
        owner, patch = int(owner), int(patch)
        centre, radius = patches.centres[patch], float(patches.radius[patch])
        offset = np.maximum(np.maximum(lower[owner]-centre, centre-upper[owner]), 0.)
        if float(offset@offset) >= radius*radius:
            continue
        if patch not in cached:
            cached[patch] = _disk_sections(field, patch)
        result.extend({"owner": owner, **event} for event in cached[patch])
    return result
