"""
Maximum Likelihood Decoding

(Minimum Distance Decoding methods).
"""

import faiss
import numpy as np
from einops import rearrange
from loguru import logger
from nptyping import NDArray
from numba import njit, prange
from pynndescent import NNDescent
from sklearn.neighbors._ball_tree import BallTree

from quanta_SL.utils.package_gpu_checker import (
    xp,
    CUPY_INSTALLED,
    KEOPS_GPU_INSTALLED,
    FAISS_GPU_INSTALLED,
)
from quanta_SL.ops.coding import hamming_distance_8bit
from types import ModuleType


@njit(parallel=True, nogil=True, fastmath=True, cache=True)
def numba_minimum_distance(x, y, hamming_dist_LUT: NDArray[int]):
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


def brute_minimum_distance(
    x, y, hamming_dist_LUT: NDArray[int] = None, xp: ModuleType = np
):
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

    x = rearrange(x, "N_x n -> N_x 1 n")
    y = rearrange(y, "N_y n -> 1 N_y n")

    # Check if look-up-table is supplied
    # duck typing: check if numpy or cupy array
    # See: https://stackoverflow.com/questions/66453371/how-to-ensure-an-argument-is-like-a-numpy-array

    if hasattr(hamming_dist_LUT, "__array_function__"):
        z = xp.zeros((N_x, N_y, n), dtype=xp.uint8)

        # Bitwise XOR
        xp.bitwise_xor(x, y, out=z)

        # We are now implementing
        # hamming_dist_LUT[x ^ y].sum(axis=-1).argmin(axis=-1)
        # in a memory efficient manner

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
        z = x ^ y

        # Matrix mul faster than .sum()
        # https://github.com/numpy/numpy/issues/16158
        vec = xp.ones(n, dtype=xp.uint8)
        z = xp.matmul(z, vec)

        return z.argmin(axis=-1)


def cupy_minimum_distance(x, y, hamming_dist_LUT: NDArray[int] = None):
    """
    Brute force minimum distance between binary vectors.
    Via CuPy.

    :param x: Shape N_x, n
    :param y: Shape N_y, n
    :param hamming_dist_LUT: precomputed hamming distance values.
        Useful if x or y are packed as uint8 or uint16.
    :return: Minimum distance indices of size N_x (with values in 0,...,N_y - 1)
    """
    assert CUPY_INSTALLED, "CuPy is not installed"
    x = xp.asarray(x)
    y = xp.asarray(y)

    if hasattr(hamming_dist_LUT, "__array_function__"):
        hamming_dist_LUT = xp.asarray(hamming_distance_8bit())

    try:
        indices = brute_minimum_distance(x, y, hamming_dist_LUT, xp=xp)
    except MemoryError as e:
        logger.warning(f"CuPy ran out of memory. {e}.")

        # Array of NANs
        indices = xp.empty(x.shape)
        indices[:] = xp.nan

    # Move to CPU from GPU
    return indices.get()


def sklearn_minimum_distance(x, y, index: BallTree = None):
    """
    Minimum distance using Sklearn Ball Tree
    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html

    :param x: Shape N_x, n
    :param y: Shape N_y, n
    :param index: BallTree instance (or compatible APIs, such as KDTree)
    :return: Minimum distance indices of size N_x (with values in 0,...,N_y - 1)
    """
    if not index:
        logger.info("Initializing NNDescent index")
        index = BallTree(y, metric="hamming")

    # Returns top-k neighbours, distances
    dist, indices_tree = index.query(x, k=1)
    neighbors = indices_tree[:, 0]

    return neighbors


def pyNNdescent_minimum_distance(x, y, index: NNDescent = None):
    """
    Minimum distance using PyNNDescent
    See https://pynndescent.readthedocs.io.

    :param x: Shape N_x, n
    :param y: Shape N_y, n
    :param index: Search index object
    :return: Minimum distance indices of size N_x (with values in 0,...,N_y - 1)
    """
    if not index:
        logger.info("Initializing NNDescent index")
        index = NNDescent(
            y,
            metric="hamming",
            n_neighbors=200,
            diversify_prob=0.0,
        )
        index.prepare()

    # Returns top-k neighbours, distances
    query = index.query(x, k=1, epsilon=0.3)
    neighbors = query[0][:, 0]

    return neighbors


def keops_minimum_distance(x, y):
    """
    Keops KNN with hamming distance

    :param x: Shape N_x, n
    :param y: Shape N_y, n
    :param hamming_dist_LUT: precomputed hamming distance values.
        Useful if x or y are packed as uint8 or uint16.
    :return: Minimum distance indices of size N_x (with values in 0,...,N_y - 1)
    """
    from pykeops.numpy import LazyTensor

    x_i = LazyTensor(x[:, None, :].astype(float))
    y_j = LazyTensor(y[None, :, :].astype(float))

    # a ^ b == (a + b) mod 2
    D_ij = (x_i + y_j).mod(2).sum(axis=-1)

    # Keops is lazy
    # All the computation happens now (reduction step)
    # Squeeze extra axis
    return D_ij.argmin(axis=1).squeeze(-1)


def faiss_minimum_distance(x, y, index: faiss.IndexBinaryFlat = None):
    """
    Minimum distance using FAISS
    See https://pynndescent.readthedocs.io.

    :param x: Shape N_x, n
    :param y: Shape N_y, n
    :param index: Search index object
    :return: Minimum distance indices of size N_x (with values in 0,...,N_y - 1)
    """
    if not index:
        logger.info("Initializing FAISS Flat index")
        index = faiss.IndexBinaryFlat(y.shape[1] * 8)
        index.add(y)
        index = NNDescent(
            y,
            metric="hamming",
            n_neighbors=200,
            diversify_prob=0.0,
        )
        index.prepare()

    # Returns top-k neighbours, distances
    distances, neighbors = index.search(x, k=1)
    neighbors = neighbors[:, 0]

    return neighbors


"""
Index functions
"""


def nndescent_index(y):
    index = NNDescent(
        y,
        metric="hamming",
        n_neighbors=200,
        diversify_prob=0.0,
    )
    index.prepare()
    return index


def balltree_index(y):
    return BallTree(y, metric="hamming")


def faiss_flat_index(y):
    d = y.shape[1] * 8
    faiss_index = faiss.IndexBinaryFlat(d)

    # Adding the database vectors.
    faiss_index.add(y)

    nthreads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(nthreads)

    return faiss_index


def faiss_flat_gpu_index(y):
    res = faiss.StandardGpuResources()

    d = y.shape[1] * 8
    config = faiss.GpuIndexBinaryFlatConfig()

    # Store values using 32-bit indices instead
    # config.indicesOptions = faiss.INDICES_32_BIT

    gpu_faiss_index = faiss.GpuIndexBinaryFlat(res, d, config)

    # Adding the database vectors.
    gpu_faiss_index.add(y)

    return gpu_faiss_index


def faiss_IVF_index(y):
    d = y.shape[1] * 8

    # Initializing the quantizer.
    quantizer = faiss.IndexBinaryFlat(d)

    # Number of clusters.
    ncluster = 4

    # Initializing index.
    faiss_index = faiss.IndexBinaryIVF(quantizer, d, ncluster)
    faiss_index.nprobe = 4  # Number of nearest clusters to be searched per query.

    # Training the quantizer.
    # Our dataset is small and training is a one-time step
    # So train on entire
    faiss_index.train(y)

    # Adding the database vectors.
    faiss_index.add(y)
    return faiss_index


def faiss_HNSW_index(y):
    d = y.shape[1] * 8
    # NOTE(hoss): Ensure the HNSW construction is deterministic.
    faiss.omp_set_num_threads(1)

    index_hnsw_bin = faiss.IndexBinaryHNSW(d, 64)

    # Adding the database vectors.
    index_hnsw_bin.add(y)

    return index_hnsw_bin
