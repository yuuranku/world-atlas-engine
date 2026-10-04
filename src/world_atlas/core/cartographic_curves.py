"""Native river stations and topology-preserving transverse constraints."""

import numpy as np


def _forward_corridor_offsets(samples, normal, offsets):
    """Keep native station order while solving neighbouring valley sections.

    Independent transverse minima can fold the two banks of an elbow onto
    the same point. Constrain both consecutive stations and the two-station
    chord used by the channel normal: their forward projection retains at
    least half the accepted drainage chord. Only offsets that consume that
    corridor are reduced. Reducing any offset cannot increase its harmful
    contribution, so previously checked constraints remain satisfied. Each
    span's constraints are applied together, independently of flow direction.
    """
    for span in (1, 2):
        chord = samples[span:] - samples[:-span]
        budget = .5 * np.einsum("ij,ij->i", chord, chord)
        if np.any(budget <= 1e-24):
            raise ValueError("river drainage anchors cannot reverse at the same station")
        before = np.maximum(0.0, offsets[:-span] * np.einsum(
            "ij,ij->i", normal[:-span], chord))
        after = np.maximum(0.0, -offsets[span:] * np.einsum(
            "ij,ij->i", normal[span:], chord))
        harmful = before + after
        ratio = np.minimum(1.0, np.divide(budget, harmful,
            out=np.ones_like(budget), where=harmful > 0))
        factor = np.ones_like(offsets)
        factor[:-span] = np.minimum(factor[:-span], np.where(before > 0, ratio, 1.0))
        factor[span:] = np.minimum(factor[span:], np.where(after > 0, ratio, 1.0))
        offsets *= factor
    return offsets


def _source_stations(grid, points):
    source = np.asarray(points, dtype=np.float64)
    if (source.ndim != 2 or source.shape[1] != 2
            or not np.all(np.isfinite(source))):
        raise ValueError("river anchors must be finite (N, 2) coordinates")
    if len(source) < 2:
        return source.copy(), np.empty(0, dtype=np.int32), np.empty((0, 2, 2))
    edges = np.diff(source, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(np.abs(edges[:, 0]) > grid.shape[1] / 2):
        raise ValueError("river reaches must be split at the longitude seam")
    pieces = [source[:1]]
    identifiers = []
    segments = []
    for first, vector, length in zip(source[:-1], edges, lengths, strict=True):
        if length <= 1e-12:
            continue
        fraction = np.linspace(0, 1, max(1, int(np.ceil(length * 4))) + 1)[1:]
        pieces.append(first + fraction[:, None] * vector)
        identifiers.extend([len(segments)] * len(fraction))
        segments.append((first, first + vector))
    samples = np.vstack(pieces)
    return samples, np.asarray(identifiers, dtype=np.int32), np.asarray(segments).reshape(-1, 2, 2)
