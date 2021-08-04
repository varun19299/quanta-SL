import numpy as np
from pynndescent import NNDescent

from quanta_SL.decode.minimum_distance.benchmark import (
    bch_dataset_query_points,
    benchmark_func,
)
from quanta_SL.decode.methods import repetition_decoding
from quanta_SL.decode.minimum_distance.factory import (
    brute_minimum_distance,
    faiss_minimum_distance,
    numba_minimum_distance,
    pyNNdescent_minimum_distance,
    sklearn_minimum_distance,
    faiss_flat_index,
    balltree_index,
)
from quanta_SL.encode.strategies import repetition_code_LUT
from quanta_SL.encode.message import binary_message
from quanta_SL.encode import metaclass
from quanta_SL.ops.coding import hamming_distance_8bit


def test_minimum_distance():
    x, y, gt_indices = bch_dataset_query_points(metaclass.BCH(15, 7, 2), num_bits=7)

    data_query_kwargs = dict(x=x, y=y, gt_indices=gt_indices)

    indices_numpy = benchmark_func(
        "Numpy",
        brute_minimum_distance,
        **data_query_kwargs,
    )
    indices_numpy_packed = benchmark_func(
        "Numpy byte-packed",
        brute_minimum_distance,
        **data_query_kwargs,
        pack=True,
        hamming_dist_LUT=hamming_distance_8bit(),
    )
    indices_numba_packed = benchmark_func(
        "Numba byte-packed",
        numba_minimum_distance,
        **data_query_kwargs,
        pack=True,
        hamming_dist_LUT=hamming_distance_8bit(),
    )
    indices_sklearn = benchmark_func(
        "Sklearn BallTree",
        sklearn_minimum_distance,
        **data_query_kwargs,
        index_func=balltree_index,
    )

    def _nndescent_index(y):
        index = NNDescent(
            y,
            metric="hamming",
        )
        index.prepare()
        return index

    indices_nndescent = benchmark_func(
        "NNDescent",
        pyNNdescent_minimum_distance,
        **data_query_kwargs,
        index_func=_nndescent_index,
    )
    indices_faiss = benchmark_func(
        "FAISS Flat",
        faiss_minimum_distance,
        **data_query_kwargs,
        index_func=faiss_flat_index,
        pack=True,
    )

    # Assertions
    assert np.array_equal(indices_numpy, gt_indices)
    assert np.array_equal(indices_numpy_packed, gt_indices)
    assert np.array_equal(indices_numba_packed, gt_indices)
    assert np.array_equal(indices_sklearn, gt_indices)
    assert np.array_equal(indices_faiss, gt_indices)

    assert (
        indices_nndescent == gt_indices
    ).mean() > 0.5, "Error: Approximate Neighbours too noisy."


def test_repetition_decoding():
    # Repeated code
    repetition_tuple = metaclass.Repetition(60, 10, 2)
    code_LUT = repetition_code_LUT(repetition_tuple, message_mapping=binary_message)

    assert np.array_equal(
        repetition_decoding(code_LUT, code_LUT, num_repeat=repetition_tuple.repeat),
        np.arange(len(code_LUT)),
    )
