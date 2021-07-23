"""
Profiling for Maximum Likelihood Decoding
"""

import logging

import galois
import numpy as np
from einops import rearrange, repeat
from matplotlib import pyplot as plt
from nptyping import NDArray
from numba import njit, prange
from pynndescent import NNDescent
from sklearn.neighbors import BallTree

from ops.binary import packbits_strided
from utils.mapping import binary_mapping
from utils.plotting import save_plot
from utils.timer import CatchTimer
from vis_tools.strategies import metaclass

plt.style.use(["science", "grid"])

logging.getLogger().setLevel(logging.INFO)

try:
    import cupy as cp
    from cupy.cuda import memory_hooks
    from cupyx import time

    xp = cp

    CUPY_INSTALLED = True
    logging.info(
        f"CuPy installation found, with {cp.cuda.runtime.getDeviceCount()} GPU(s)."
    )

except ImportError:
    xp = np
    CUPY_INSTALLED = False
    logging.warning("No CuPy installation detected. Using Numpy, may be slow.")


def hamming_distance_8bit():
    """
    Hamming distance of each 8-bit uint (in binary representation).

        >>> hamming_distance_8bit(5) # 3
        >>> hamming_distance_8bit(8) # 1
        >>> hamming_distance_8bit(0) # 0

    :return: array with Look Up Table
    """
    return xp.array([bin(i).count("1") for i in range(256)], dtype=xp.uint8)


@njit(parallel=True, nogil=True, fastmath=True, cache=True)
def numba_minimum_distance(x, y, hamming_dist_LUT):
    """
    Minimum distance between binary vectors.
    Computed brute-force in a loop.

     :param x: Shape N_x, n
     :param y: Shape N_y, n
     :param hamming_dist_LUT: precomputed hamming distance values.
         Useful if x or y are packed as uint8 or uint16.
     :return: Minimum distance indices of size N_x (with values in 0,...,N_y - 1)
    """
    N_x, n = x.shape
    N_y, n = y.shape

    z_min = np.zeros(N_x, dtype=np.uint16)

    for i in prange(N_x):
        x_vec = x[i, :]
        min_dist = 1e8

        for j in range(N_y):
            y_vec = y[j, :]

            dist = 0
            for l in range(n):
                dist += np.take(hamming_dist_LUT, x_vec[l] ^ y_vec[l])

            if dist < min_dist:
                z_min[i] = j
                min_dist = dist

    return z_min


def brute_minimum_distance(x, y, hamming_dist_LUT: NDArray[int] = None):
    """
    Brute force minimum distance between binary vectors

    :param x: Shape N_x, n
    :param y: Shape N_y, n
    :param hamming_dist_LUT: precomputed hamming distance values.
        Useful if x or y are packed as uint8 or uint16.
    :return: Minimum distance indices of size N_x (with values in 0,...,N_y - 1)
    """
    N_x, n = x.shape
    N_y, n = y.shape

    # Check if look-up-table is supplied
    # duck typing: check if numpy or cupy array
    # See: https://stackoverflow.com/questions/66453371/how-to-ensure-an-argument-is-like-a-numpy-array
    if hasattr(hamming_dist_LUT, "__array_function__"):
        # We are now implementing
        # hamming_dist_LUT[x ^ y].sum(axis=2).argmin(axis=-1)
        # in a memory efficient manner

        z = xp.zeros((N_x, N_y, n), dtype=xp.uint8)

        # Bitwise XOR
        x = rearrange(x, "N_x n -> N_x 1 n")
        y = rearrange(y, "N_y n -> 1 N_y n")
        xp.bitwise_xor(x, y, out=z)

        hamming_out = hamming_dist_LUT[z.ravel()]

        hamming_out = rearrange(
            hamming_out,
            "(N_x N_y n) -> N_x N_y n",
            N_x=N_x,
            N_y=N_y,
            n=n,
        )

        # Save memory. Below 255, we can make do with uint8
        if n < 256:
            summation_out = xp.zeros((N_x, N_y), dtype=xp.uint8)
        else:
            summation_out = xp.zeros((N_x, N_y), dtype=xp.uint16)

        # Direct sum seems faster on CuPy
        hamming_out.sum(axis=-1, out=summation_out)

        return summation_out.argmin(axis=-1)

    else:
        # Bitwise XOR
        x = rearrange(x, "N_x n -> N_x 1 n")
        y = rearrange(y, "N_y n -> 1 N_y n")
        z = x ^ y

        # Matrix mul faster than .sum()
        # https://github.com/numpy/numpy/issues/16158
        vec = xp.ones(n, dtype=xp.uint8)
        z = xp.matmul(z, vec)

        return z.argmin(axis=-1)


def pyNNdescent_minimum_distance(x, y, index: NNDescent = None):
    """
    Minimum distance using PyNNDescent
    See https://pynndescent.readthedocs.io.

    :param x: Shape N_x, n
    :param y: Shape N_y, n
    :param index: Search index object
    :return: Minimum distance indices of size N_x (with values in 0,...,N_y - 1)
    """
    logging.info("Preparing index")

    if not index:
        index = NNDescent(
            y,
            metric="hamming",
            n_neighbors=200,
            diversify_prob=0.0,
        )
        index.prepare()

    logging.info("Querying neighbours")

    # Returns top-k neighbours, distances
    query = index.query(x, k=1, epsilon=0.3)
    neighbors = query[0][:, 0]

    return neighbors


@njit(cache=True, parallel=True, nogil=True)
def bit_flip_noise(x: np.ndarray, noisy_bits: int):
    N, n = x.shape
    noise_mat = np.zeros_like(x)

    for i in prange(N):
        noise = np.zeros(n)
        indices = np.random.choice(
            np.arange(noise.size), replace=False, size=noisy_bits
        )
        noise[indices] = 1
        noise_mat[i] = noise

    return noise_mat


def bch_dataset_query_points(
    bch_tuple: metaclass.BCH,
    num_bits: int = 10,
    query_repeat: int = 1,
    verbose: bool = False,
):
    assert num_bits <= bch_tuple.k, "num_bits exceeds BCH message dimensions."
    message_ll = binary_mapping(num_bits)

    # BCH encode
    bch = galois.BCH(bch_tuple.n, bch_tuple.k)
    code_LUT = bch.encode(galois.GF2(message_ll))
    code_LUT = code_LUT.view(np.ndarray).astype(int)

    N, n = code_LUT.shape
    y = code_LUT

    # Corrupt query points
    x = repeat(code_LUT, "N n -> (N repeat) n", repeat=query_repeat)
    gt_indices = np.arange(N)
    gt_indices = repeat(gt_indices, "N -> (N repeat)", repeat=query_repeat)

    with CatchTimer() as t:
        noise_mat = bit_flip_noise(x, noisy_bits=bch.t - 1)

    if verbose:
        print(f"Noise generation {t}")
    x ^= noise_mat

    return x, y, gt_indices


def cpu_minimum_distance(x: NDArray[int], y: NDArray[int], gt_indices: NDArray[int]):
    timing_dict = {}

    # NNDescent index
    with CatchTimer() as t:
        index = NNDescent(
            y,
            metric="hamming",
            n_neighbors=200,
            diversify_prob=0.0,
        )
        index.prepare()

    print(f"Index preparation time (NN Descent) {t.elapsed_time}")

    # Unpacked versions
    with CatchTimer() as t:
        indices_binary = brute_minimum_distance(x, y)

    # Numpy
    timing_dict["Numpy"] = t.elapsed_time
    print(f"Query time (Brute) {t.elapsed_time}")

    # NNDescent
    with CatchTimer() as t:
        indices_approx = pyNNdescent_minimum_distance(x, y, index=index)

    timing_dict["NNDescent"] = t.elapsed_time
    print(f"Query time (NN Descent) {t.elapsed_time}")

    # Sklearn
    tree = BallTree(y, metric="hamming")
    with CatchTimer() as t:
        dist, indices_tree = tree.query(x, k=1)
        indices_tree = indices_tree[:, 0]

    timing_dict["Sklearn BallTree"] = t.elapsed_time
    print(f"Query time (Sklearn Ball Tree) {t.elapsed_time}")

    # Check packed versions
    x = packbits_strided(x)
    y = packbits_strided(y)

    with CatchTimer() as t:
        indices_packed = brute_minimum_distance(
            x, y, hamming_dist_LUT=hamming_distance_8bit()
        )

    # Numpy
    timing_dict["Numpy byte-packed"] = t.elapsed_time
    print(f"Query time (Brute, packed) {t.elapsed_time}")

    # Numba
    with CatchTimer() as t:
        indices_numba = numba_minimum_distance(
            x, y, hamming_dist_LUT=hamming_distance_8bit()
        )
    timing_dict["Numba byte-packed"] = t.elapsed_time
    print(f"Query time (Numba, packed) {t.elapsed_time}")

    # Stats
    print(f"Approximate NN accuracy {(indices_binary == indices_approx).mean()}")
    print(f"Ball Tree accuracy {(indices_binary == indices_tree).mean()}")
    print(f"Minimum distance accuracy {(indices_binary == gt_indices).mean()}")

    # Assertions
    assert np.array_equal(indices_binary, indices_packed)
    assert np.array_equal(indices_binary, indices_numba)
    assert np.array_equal(indices_binary, indices_tree)

    return timing_dict


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


if __name__ == "__main__":
    bch_tuple = metaclass.BCH(63, 10, 13)
    num_bits = 10

    # 2^10 x 10 query points
    query_repeat = 100

    x, y, gt_indices = bch_dataset_query_points(bch_tuple, num_bits, query_repeat)
    timing_dict = cpu_minimum_distance(x, y, gt_indices)
    timing_dict = dict(sorted(timing_dict.items(), key=lambda item: item[1]))

    plt.figure(figsize=(6, 4))

    for name, timing in timing_dict.items():
        plt.barh(name, timing, label=name, align="center")

    plt.legend()
    plt.tight_layout()
    plt.xscale("log")
    plt.xlabel("Time (in seconds)")
    plt.title(
        f"Querying {pow(2, num_bits) * query_repeat} points from {bch_tuple} dataset"
    )
    save_plot(savefig=True, fname="outputs/benchmarks/cpu-KNN.pdf")

    if CUPY_INSTALLED:
        hook = memory_hooks.LineProfileHook()

        with hook:
            x = xp.asarray(x)
            y = xp.asarray(y)
            hamming_dist_LUT = xp.asarray(hamming_distance_8bit())

        hook.print_report()
        # print(time.repeat(minimum_distance, (x, y), n_repeat=10))
        print(
            time.repeat(brute_minimum_distance, (x, y, hamming_dist_LUT), n_repeat=10)
        )
