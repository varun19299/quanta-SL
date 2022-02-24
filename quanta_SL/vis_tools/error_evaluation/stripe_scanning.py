from numba import njit
from quanta_SL.ops.math_func import comb


@njit
def prob_binom(n, p, i):
    return comb(n, i) * pow(p, i) * pow(1 - p, n - i)


@njit(cache=True)
def binom_x_geq_y(n, p, q):
    """
    Prob X >= Y

    X ~ Binom(n, p)
    Y ~ Binom(n, q)
    :param n:
    :param p: Binomial param for X
    :param q: Binomial param for Y
    :return: Prob{X \geq Y}
    """
    out = 0
    for i in range(n + 1):
        inner_sum = 0
        for j in range(i, n + 1):
            inner_sum += prob_binom(n, p, j)

        # P(Y=i)
        inner_sum *= prob_binom(n, q, i)
        out += inner_sum
    return out


@njit
def expected_abs_error(p_flip_bright, p_flip_dark):
    out = 0
    for r in range(16):
        out += 2 * r * binom_x_geq_y(r, p_flip_dark, 1 - p_flip_bright)
    return out
