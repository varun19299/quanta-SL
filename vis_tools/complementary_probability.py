import numpy as np

# from scipy.special import comb
from math import comb
from numba import types, njit
from numba.extending import overload


@overload(comb)
def comb_numba(p, q):
    # if isinstance(p, types.int64) and isinstance(q, types.int64):
    print(p)
    print(int(p))
    out = comb(p, q)

    def comb_implement(p, q):
        return out

    return comb_implement


@njit
def actual(p, q):
    return comb(p, q)


def hoeffding():
    pass


def gaussian():
    pass


actual(3, 1)
