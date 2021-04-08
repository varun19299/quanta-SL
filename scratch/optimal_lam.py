from matplotlib import pyplot as plt
import numpy as np
from nptyping import NDArray


def to_min(lam: NDArray, p: float, q: float, n: float):
    frac_0 = (n / 2) * (lam - q) ** 2 / (q + lam / 3)
    min_0 = np.exp(-frac_0)

    frac_1 = (n / 2) * (p - lam) ** 2 / p
    min_1 = np.exp(-frac_1)

    return min_0 + min_1


if __name__ == "__main__":
    lam = np.linspace(0.01, 1, num=50)
    kwargs = dict(p=0.9, q=0, n=10)

    plt.plot(lam, to_min(lam, **kwargs))
    plt.show()
