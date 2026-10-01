"""Shared display classification of the unchanged relative elevation field.

The simulation's ordinal bands remain part of WorldGrid. Presentation uses
finer intervals so lowland relief is visible without changing the terrain or
claiming that the authored relative elevations are measured heights in metres.
"""

from __future__ import annotations

import numpy as np


ELEVATION_DISPLAY_LEVELS = 24
_ELEVATION_THRESHOLDS = np.power(
    np.arange(1, ELEVATION_DISPLAY_LEVELS, dtype=np.float64)
    / ELEVATION_DISPLAY_LEVELS,
    1.25,
)
_ELEVATION_THRESHOLDS.flags.writeable = False


def elevation_display_thresholds() -> list[float]:
    """Return the same relative-height boundaries for raster and vector maps."""

    return _ELEVATION_THRESHOLDS.tolist()


def elevation_display_indices(elevation: np.ndarray) -> np.ndarray:
    """Classify continuous elevations; a threshold belongs to its upper band."""

    return np.searchsorted(
        _ELEVATION_THRESHOLDS, np.asarray(elevation), side="right"
    ).astype(np.uint8)
