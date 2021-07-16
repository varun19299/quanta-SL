import numpy as np
from nptyping import NDArray
from numba import njit

from vis_tools.strategies.metaclass import BCH

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


def list_error_correcting_capacity(bch_tuple: BCH) -> int:
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


def order_range(array: NDArray) -> int:
    upper = np.log10(array.max())
    lower = np.log10(array.min())
    return upper.astype(int) - lower.astype(int) + 1