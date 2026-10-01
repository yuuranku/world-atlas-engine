"""Deterministic atlas colors chosen for actual neighbouring territories."""
from collections import Counter
from collections.abc import Sequence
import numpy as np

POLITICAL_PALETTE = (
    "#eed775", "#78a4ef", "#f881ad", "#a0c86d", "#ba8be5",
    "#70c8b3", "#eb9269", "#bea9a6", "#add0df", "#bcf0a1",
)


def area_colors(values: np.ndarray, identifiers: Sequence[int]) -> dict[int, str]:
    """Maximize neighbour contrast, then balance palette reuse across the map."""
    adjacency = {int(identifier): set() for identifier in identifiers}
    for first, second in ((values[:, :-1], values[:, 1:]),
                          (values[:-1, :], values[1:, :]),
                          (values[:, -1:], values[:, :1])):
        changed = (first > 0) & (second > 0) & (first != second)
        pairs = np.unique(np.column_stack((first[changed], second[changed])), axis=0)
        for left, right in pairs:
            if int(left) in adjacency and int(right) in adjacency:
                adjacency[int(left)].add(int(right))
                adjacency[int(right)].add(int(left))
    colors = np.asarray([[int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
                         for color in POLITICAL_PALETTE])
    distance = np.linalg.norm(colors[:, None, :] - colors[None, :, :], axis=2)
    selected = {}
    used = Counter()
    while len(selected) < len(adjacency):
        identifier = min((i for i in adjacency if i not in selected), key=lambda i: (
            -len({selected[n] for n in adjacency[i] if n in selected}), -len(adjacency[i]), i))
        neighbours = {selected[n] for n in adjacency[identifier] if n in selected}
        available = [i for i in range(len(colors)) if i not in neighbours]
        if not available:
            available = list(range(len(colors)))
        selected[identifier] = max(available, key=lambda i: (
            (min(distance[i, n] for n in neighbours) if neighbours else .5) - .006 * used[i],
            -used[i], -((i - identifier * 3) % len(colors))))
        used[selected[identifier]] += 1
    return {identifier: POLITICAL_PALETTE[index] for identifier, index in selected.items()}
