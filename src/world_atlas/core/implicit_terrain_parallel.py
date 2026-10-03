"""Run independent height batches on the unchanged physical model."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

import numpy as np

from .implicit_terrain import _refined_curves


_SOURCE=None


def _initialize(field,lower,upper,coarse_low,coarse_high):
    global _SOURCE
    _SOURCE=field,lower,upper,coarse_low,coarse_high


def _height_graphs(levels):
    field,lower,upper,low,high=_SOURCE
    # Keep original native batches and share level-independent certificates
    # between nearby cuts, just as the serial extractor does.
    return _refined_curves(field,levels,lower,upper,low,high)


def parallel_height_curves(field,levels,lower,upper,coarse_low,coarse_high):
    """Keep a complete source-port graph per height and preserve level order.

    Callers must enter through an importable, main-guarded Python program.
    Worker exceptions propagate; no alternative geometry is substituted.
    """
    workers=min(4, max(1,(os.process_cpu_count() or 1)//2),len(levels))
    if workers == 1:
        return _refined_curves(field,levels,lower,upper,coarse_low,coarse_high)
    batches=np.array_split(levels, workers)
    with ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,
            initargs=(field,lower,upper,coarse_low,coarse_high)) as executor:
        return [paths for batch in executor.map(_height_graphs,batches) for paths in batch]
