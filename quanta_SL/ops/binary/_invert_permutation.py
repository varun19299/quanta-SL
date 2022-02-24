import numpy as np
from nptyping import NDArray
from numba import njit, prange


@njit(cache=True, nogil=True, parallel=True)
def numba_invert_permutation(index_map: NDArray[int]) -> NDArray[int]:
    inverted_index_map = np.zeros_like(index_map)

    for i in prange(len(index_map)):
        # f o g = I
        # f(x) = y
        index = index_map[i]

        # g(y) = x
        inverted_index_map[index] = i

    return inverted_index_map


def invert_permutation(index_map: NDArray[int]) -> NDArray[int]:
    assert issubclass(
        index_map.dtype.type, np.integer
    ), "Index map must be integer dtype"
    assert index_map.ndim == 1, "Only 1 dimensional inverting supported"
    assert np.array_equal(
        np.unique(index_map), np.arange(len(index_map), dtype=index_map.dtype)
    ), "Index map must take unique values from 0, ..., len(index_map) - 1"

    inverted_index_map = numba_invert_permutation(index_map)

    return inverted_index_map
