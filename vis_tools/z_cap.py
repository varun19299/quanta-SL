import numpy as np
from matplotlib import pyplot as plt
from numba import jit

from math import comb


def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def z_capacity(p):
    s_p = entropy(p) / (1 - p)
    denom = 1 + 2 ** s_p
    term_1 = entropy(1 / denom)
    term_2 = s_p / denom
    return term_1 - term_2


@jit
def avg_error(p, n):
    prob = 0
    rep = n // 10
    for i in range((rep - 1) // 2):
        prob += comb(rep, i) * pow(1-p, rep - i) * pow(p, i)
    return prob ** 10


@jit
def bch_error(p, n, t):
    prob = 0
    for i in range(t + 1):
        prob += comb(n, i) * pow(1-p, n - i) * pow(p, i)
    return prob


if __name__ == "__main__":

    p = np.linspace(0, 1.0, num=10)
    plt.plot(p, avg_error(p, 60), label="Avg")
    plt.plot(p, bch_error(p, 63, 13), label="BCH-31-11-5")
    plt.legend()
    plt.show()

    cap = z_capacity(p)
    plt.plot(p, cap)
    plt.show()
