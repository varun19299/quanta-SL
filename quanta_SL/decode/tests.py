import numpy as np
from pynndescent import NNDescent
from sklearn.neighbors._ball_tree import BallTree

from quanta_SL.decode.methods import brute_minimum_distance, \
    pyNNdescent_minimum_distance, numba_minimum_distance
from quanta_SL.decode.benchmark import bch_dataset_query_points
from quanta_SL.ops.coding import hamming_distance_8bit
from quanta_SL.encode import metaclass
from quanta_SL.ops.binary import packbits_strided


def test_minimum_distance():
    x, y, gt_indices = bch_dataset_query_points(metaclass.BCH(15, 7, 2), num_bits=7)

    # NNDescent index
    index = NNDescent(
        y,
        metric="hamming",
        n_neighbors=50,
        diversify_prob=0.0,
    )
    index.prepare()

    # Unpacked versions
    indices_binary = brute_minimum_distance(x, y)
    indices_approx = pyNNdescent_minimum_distance(x, y, index=index)

    tree = BallTree(y, metric="hamming")
    dist, indices_tree = tree.query(x, k=1)
    indices_tree = indices_tree[:, 0]

    # Check packed versions
    x = packbits_strided(x)
    y = packbits_strided(y)

    indices_packed = brute_minimum_distance(
        x, y, hamming_dist_LUT=hamming_distance_8bit()
    )
    indices_numba = numba_minimum_distance(
        x, y, hamming_dist_LUT=hamming_distance_8bit()
    )

    # Assertions
    assert np.array_equal(indices_binary, indices_packed)
    assert np.array_equal(indices_binary, indices_numba)
    assert np.array_equal(indices_binary, indices_tree)
    assert (
        indices_binary == indices_approx
    ).mean() > 0.5, "Error: Approximate Neighbours too noisy."