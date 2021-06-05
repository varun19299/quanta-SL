import numpy as np
from matplotlib import pyplot as plt


def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def z_capacity(p):
    s_p = entropy(p) / (1 - p)
    denom = 1 + 2 ** s_p
    term_1 = entropy(1 / denom)
    term_2 = s_p / denom
    return term_1 - term_2


p = np.linspace(0, 1.0, num=10)
cap = z_capacity(p)
plt.plot(p, cap)
plt.show()
