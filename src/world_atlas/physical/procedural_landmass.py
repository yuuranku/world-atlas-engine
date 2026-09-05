"""Shared deterministic geometry for continents, microcontinents and islands."""

from __future__ import annotations

import numpy as np


def rotated_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(angle_degrees)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    offset_x = x - center_x
    offset_y = y - center_y
    return (
        cosine * offset_x + sine * offset_y,
        -sine * offset_x + cosine * offset_y,
    )


def ellipse_margin(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    angle_degrees: float,
) -> np.ndarray:
    local_x, local_y = rotated_coordinates(
        x, y, center_x, center_y, angle_degrees
    )
    return 1.0 - np.sqrt((local_x / radius_x) ** 2 + (local_y / radius_y) ** 2)


def lattice_noise(
    x: np.ndarray,
    y: np.ndarray,
    *,
    spacing: float,
    seed: int,
) -> np.ndarray:
    """Return smooth deterministic value noise in an undistorted plane."""

    if spacing <= 0.0:
        raise ValueError("noise spacing must be positive")
    grid_x = x / spacing
    grid_y = y / spacing
    x0 = np.floor(grid_x)
    y0 = np.floor(grid_y)
    fraction_x = grid_x - x0
    fraction_y = grid_y - y0
    weight_x = fraction_x * fraction_x * (3.0 - 2.0 * fraction_x)
    weight_y = fraction_y * fraction_y * (3.0 - 2.0 * fraction_y)

    def hashed(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        phase = ix * 127.1 + iy * 311.7 + seed * 73.419
        raw = np.sin(phase) * 43758.5453123
        return 2.0 * (raw - np.floor(raw)) - 1.0

    lower_left = hashed(x0, y0)
    lower_right = hashed(x0 + 1.0, y0)
    upper_left = hashed(x0, y0 + 1.0)
    upper_right = hashed(x0 + 1.0, y0 + 1.0)
    lower = lower_left * (1.0 - weight_x) + lower_right * weight_x
    upper = upper_left * (1.0 - weight_x) + upper_right * weight_x
    return lower * (1.0 - weight_y) + upper * weight_y


def fractal_plane_noise(x: np.ndarray, y: np.ndarray, *, seed: int) -> np.ndarray:
    """Return the same five-octave coast texture used by polar continents."""

    result = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=np.float64)
    amplitude_total = 0.0
    for spacing, amplitude, offset in (
        (8.2, 0.48, 0),
        (4.1, 0.28, 17),
        (2.05, 0.15, 43),
        (1.02, 0.07, 89),
        (0.52, 0.02, 151),
    ):
        result += amplitude * lattice_noise(
            x, y, spacing=spacing, seed=seed + offset
        )
        amplitude_total += amplitude
    return result / amplitude_total


__all__ = [
    "ellipse_margin",
    "fractal_plane_noise",
    "lattice_noise",
    "rotated_coordinates",
]
