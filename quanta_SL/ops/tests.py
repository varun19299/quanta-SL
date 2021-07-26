"""
Tests
"""

import numpy as np

from quanta_SL.ops.binary import packbits_strided, unpackbits_strided
from quanta_SL.ops.binary import invert_permutation


def test_packbits_unpackbits_strided():
    h, w, n = 100, 100, 30
    x = np.random.randint(0, 2, (h, w, n), dtype=int)

    packed_x = packbits_strided(x)
    unpacked_x = unpackbits_strided(packed_x)

    assert np.array_equal(x, unpacked_x)


def test_invert_permutation():
    test_array = np.random.rand(1024)

    index_map = np.random.permutation(len(test_array))
    permuted_test = test_array[index_map]

    inverted_index_map = invert_permutation(index_map)

    assert np.array_equal(
        index_map[inverted_index_map],
        np.arange(len(test_array), dtype=test_array.dtype),
    )
    assert np.array_equal(permuted_test[inverted_index_map], test_array)
