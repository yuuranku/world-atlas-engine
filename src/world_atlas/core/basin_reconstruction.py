"""Continuous lowland reconstruction constrained by the surrounding rim."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def reconstruct_basin(surface, mask, target):
    """Solve a screened Laplace surface, avoiding nearest-rim Voronoi seams."""
    rows, cols = np.nonzero(mask)
    count = len(rows)
    if not count:
        return np.empty(0, dtype=np.float64)
    indices = np.full(mask.shape, -1, dtype=np.int32)
    indices[rows, cols] = np.arange(count)
    diagonal = np.full(count, 0.03)
    rhs = 0.03 * np.broadcast_to(target, (count,)).astype(np.float64).copy()
    matrix_rows, matrix_cols, values = [], [], []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = rows + dr, (cols + dc) % mask.shape[1]
        valid = (rr >= 0) & (rr < mask.shape[0])
        source = np.flatnonzero(valid)
        neighbor = indices[rr[valid], cc[valid]]
        diagonal[source] += 1
        internal = neighbor >= 0
        matrix_rows.extend(source[internal])
        matrix_cols.extend(neighbor[internal])
        values.extend(np.full(np.count_nonzero(internal), -1.0))
        edge = source[~internal]
        rhs[edge] += np.maximum(surface[rr[edge], cc[edge]], 32.0)
    matrix_rows.extend(np.arange(count))
    matrix_cols.extend(np.arange(count))
    values.extend(diagonal)
    matrix = sparse.csr_matrix((values, (matrix_rows, matrix_cols)), shape=(count, count))
    return spsolve(matrix, rhs)
