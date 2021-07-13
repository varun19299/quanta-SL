import numpy as np
from matplotlib import pyplot as plt
from numba import njit

from utils.math import comb
from vis_tools.strategies import metaclass

from dataclasses import astuple


def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def z_capacity(p):
    s_p = entropy(p) / (1 - p)
    denom = 1 + 2 ** s_p
    term_1 = entropy(1 / denom)
    term_2 = s_p / denom
    return term_1 - term_2


def repetition_error(p, repetition_tuple: metaclass.Repetition):
    n, k, t = astuple(repetition_tuple)
    rep = repetition_tuple.repeat

    # @njit
    def _error():
        prob = np.zeros_like(p)
        print(t)
        for i in range(t + 1):
            prob += comb(rep, i) * np.power(1 - p, rep - i) * np.power(p, i)
        return 1 - prob ** k

    return _error()


def bch_error(p, bch_tuple: metaclass.BCH, shortened: int = 10):
    n, k, t = astuple(bch_tuple)
    n -= k - shortened

    # @njit()
    def _error():
        prob = np.zeros_like(p)
        print(n)
        for i in range(t + 1):
            prob += comb(n, i) * np.power(1 - p, n - i) * np.power(p, i)
        return 1 - prob

    return _error()


if __name__ == "__main__":
    repetititon_tuple = metaclass.Repetition(30, 10, 1)
    bch_tuple = metaclass.BCH(31, 11, 5)

    p = np.linspace(0, 1.0, num=10)

    plt.plot(
        p,
        repetition_error(p, repetititon_tuple),
        label=str(repetititon_tuple),
        linewidth=3,
        color="red",
    )
    plt.plot(p, bch_error(p, bch_tuple), label=str(bch_tuple), linewidth=3)
    plt.ylabel("Error Probability")
    plt.xlabel("Corruption Probability $p$")
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()

    # cap = z_capacity(p)
    # plt.plot(p, cap)
    # plt.show()
