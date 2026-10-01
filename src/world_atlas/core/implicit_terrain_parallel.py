"""Run independent whole-world height graphs on the unchanged physical model."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

from .implicit_terrain import _refined_curves


_SOURCE=None


def _initialize(field,lower,upper,coarse_low,coarse_high):
    global _SOURCE
    _SOURCE=field,lower,upper,coarse_low,coarse_high


def _level_graph(level):
    field,lower,upper,low,high=_SOURCE
    selected=(low<=level)&(high>=level)
    return _refined_curves(field,np.asarray((level,)),lower[selected],upper[selected],
                           low[selected],high[selected])[0]


def parallel_height_curves(field,levels,lower,upper,coarse_low,coarse_high):
    """Keep a complete source-port graph per height and preserve level order.

    Callers must enter through an importable, main-guarded Python program.
    Worker exceptions propagate; no alternative geometry is substituted.
    """
    with ProcessPoolExecutor(max_workers=min(4,len(levels)),
            mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,
            initargs=(field,lower,upper,coarse_low,coarse_high)) as executor:
        return list(executor.map(_level_graph,map(float,levels)))
