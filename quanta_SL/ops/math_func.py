"""
General math funcs

All kept pure
"""
from typing import Callable

import numpy as np
from nptyping import NDArray
from numba import njit, vectorize
from numba import float64

from quanta_SL.encode import metaclass

LOOKUP_TABLE = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ],
    dtype="int64",
)


@njit(cache=True)
def fast_factorial(n):
    """
    Fast LUT based factorial
    :param n:
    :return:
    """
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]


@njit(cache=True)
def comb(n, r):
    """
    Binomial coefficient, nCr, aka the "choose" function
    n! / (r! * (n - r)!)

    Faster than math.comb XD!
    """
    p = 1
    for i in range(1, min(r, n - r) + 1):
        p *= n
        p //= i
        n -= 1
    return p


def list_error_correcting_capacity(bch_tuple: metaclass.BCH) -> int:
    """
    Based on BCH LECC limit.
    See Wu et al. 2008, https://arxiv.org/pdf/cs/0703105.pdf.

    :param bch_tuple: Describes BCH code as [n, k, t]
        n: Code length
        k: Message length
        t: Worst case correctable errors
    :return: List Error Correcting Capacity
    """
    n = bch_tuple.n
    d = bch_tuple.distance
    return n / 2 * (1 - (1 - 2 * d / n) ** 0.5)


def order_range(array: NDArray[float], axis: int = None) -> int:
    """
    Difference in order (powers of 10)
    between maximum and minimum elements of an array.

    :param array: float array
    :param axis: Which axis to operate on.
        Default None (across all).
    :return:
    """
    upper = np.log10(array.max(axis=axis))
    lower = np.log10(array.min(axis=axis))
    return upper.astype(int) - lower.astype(int) + 1


def periodically_continued(a: float, b: float) -> Callable:
    """
    Periodically continue a function
    :param a: Start point
    :param b: End point
    :return: Periodic function
    """

    interval = b - a
    return lambda f: lambda x: f((x - a) % interval + a)

@periodically_continued(-1, 1)
@vectorize([float64(float64)], nopython=True, target="parallel", cache=True)
def rect(t: float):
    # Discontinuous version
    # if abs(t) == 0.5:
    #     return 1
    if abs(t) > 0.5:
        return 0
    else:
        return 1
