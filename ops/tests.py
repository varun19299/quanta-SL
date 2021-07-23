"""
Tests
"""

import numpy as np

from ops.binary import packbits_strided, unpackbits_strided


def test_packbits_unpackbits_strided():
    h, w, n, k = 100, 100, 30, 10
    x = np.random.randint(0, 2, (h, w, n), dtype=int)

    packed_x = packbits_strided(x)
    unpacked_x = unpackbits_strided(packed_x)

    breakpoint()
    assert np.array_equal(x, unpacked_x)
