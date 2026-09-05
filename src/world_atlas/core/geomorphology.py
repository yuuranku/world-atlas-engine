"""Conservative geomorphic transport on the existing spherical sample grid."""

from __future__ import annotations

import numpy as np


def relax_hillslopes(
    height_m: np.ndarray,
    land_mask: np.ndarray,
    low_relief_support: np.ndarray,
    latitude_degrees: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Move relief down-gradient before drainage incision and fine synthesis.

    This is a scale-aware geomorphic diffusion approximation, not a calibrated
    geological clock. Pairwise fluxes conserve latitude-weighted material;
    sea cells are closed boundaries, not sinks or shortcuts between islands.
    """
    original = np.asarray(height_m, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    quiet = np.asarray(low_relief_support, dtype=np.float64)
    latitude = np.asarray(latitude_degrees, dtype=np.float64)
    if original.shape != land.shape or quiet.shape != land.shape:
        raise ValueError('hillslope fields must share one shape')
    if latitude.shape != (land.shape[0],):
        raise ValueError('one latitude is required per hillslope row')
    area = np.maximum(np.cos(np.radians(latitude)), 0.025)[:, None]
    surface = original.copy()
    mobility = 0.10 * (1.0 - 0.60 * np.clip(quiet, 0.0, 1.0))
    east_land = land & np.roll(land, -1, axis=1)
    north_land = land[:-1] & land[1:]
    east_rate = np.minimum(
        np.maximum(mobility, np.roll(mobility, -1, axis=1)) / np.square(area),
        0.18,
    ) * east_land
    north_rate = np.maximum(mobility[:-1], mobility[1:]) * north_land
    north_area = (area[:-1] + area[1:]) * 0.5
    passes = max(4, int(round(80 * (land.shape[1] / 2176.0) ** 2)))
    for _ in range(passes):
        east_flux = (surface - np.roll(surface, -1, axis=1)) * east_rate * area
        change = np.roll(east_flux, 1, axis=1) - east_flux
        north_flux = (surface[:-1] - surface[1:]) * north_rate * north_area
        change[:-1] -= north_flux
        change[1:] += north_flux
        surface += change / area
    difference = surface - original
    transported = float(np.sum(np.maximum(difference, 0.0) * area))
    balance = float(np.sum(difference * area))
    return surface.astype(np.float32), {
        'model': 'paired-spherical-hillslope-flux',
        'passes': passes,
        'relativeMaterialBalanceError': abs(balance) / max(transported, 1.0),
        'maximumLoweringMeters': float(-np.min(difference, initial=0.0)),
        'maximumFootslopeAccretionMeters': float(np.max(difference, initial=0.0)),
    }
