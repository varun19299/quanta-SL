"""
Helper ops
"""
from typing import Any, Union

import numpy as np
from nptyping import NDArray


def normalized(a: NDArray, axis: int = -1, order: int = 2):
    """
    From https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy-to-a-unit-vector/21032099#21032099
    :param order: Lp norm order
    :return: normalized vector
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def sorted_eigendecomposition(A: NDArray[Any, Any]) -> Union[np.ndarray, np.ndarray]:
    eigen_values, eigen_vectors = np.linalg.eig(A)

    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    return eigen_values, eigen_vectors
