"""
Profiling for Maximum Likelihood Decoding
"""

import logging
from types import ModuleType

import galois
import numpy as np
import pynndescent
from einops import rearrange, repeat
from nptyping import NDArray
from numba import njit, prange

from ops.binary import packbits_strided
from utils.mapping import gray_mapping
from utils.profiler import profile

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


@njit(parallel=True, nogil=True)
def numba_minimum_distance(x, y, hamming_dist_LUT):
    h, w, n, _ = x.shape
    _, _, _, dim_k = y.shape

    z_min = np.zeros((h, w), dtype=np.uint16)

    for i in prange(h):
        for j in prange(w):
            x_vec = x[i, j, :, :].flatten()
            min_dist = 1e8

            for k in range(dim_k):
                y_vec = y[:, :, :, k].flatten()

                dist = 0
                for l in range(n):
                    dist += np.take(hamming_dist_LUT, x_vec[l] ^ y_vec[l])

                if dist < min_dist:
                    z_min[i, j] = k
                    min_dist = dist

    return z_min


def minimum_distance(x, y, hamming_dist_LUT: NDArray[int] = None):
    """

    :param x: Shape h, w, n, 1
    :param y: Shape 1, 1, n, 2**k
    :param hamming_dist_LUT: precomputed hamming distance values.
        Useful if x or y are packed as uint8 or uint16.
    :return: Minimum distances h, w, 1 (channel denotes index in 0...2**k -1)
    """

    # Check if look-up-table is supplied
    # duck typing: check if numpy or cupy array
    # See: https://stackoverflow.com/questions/66453371/how-to-ensure-an-argument-is-like-a-numpy-array
    if hasattr(hamming_dist_LUT, "__array_function__"):
        # We are now implementing
        # hamming_dist_LUT[x ^ y].sum(axis=2).argmin(axis=-1)
        # in a memory efficient manner

        h, w, n, _ = x.shape
        _, _, _, dim_k = y.shape

        z = xp.zeros((h, w, n, dim_k), dtype=xp.uint8)

        # Bitwise XOR
        xp.bitwise_xor(x, y, out=z)
        hamming_out = hamming_dist_LUT[z.ravel()]

        hamming_out = rearrange(
            hamming_out,
            "(height width num_bytes samples) -> height width samples num_bytes",
            height=h,
            width=w,
            num_bytes=n,
            samples=dim_k,
        )

        # Save memory. Below 255, we can make do with uint8
        if n < 256:
            summation_out = xp.zeros((h, w, dim_k), dtype=xp.uint8)
        else:
            summation_out = xp.zeros((h, w, dim_k), dtype=xp.uint16)

        # Direct sum seems faster on CuPy
        hamming_out.sum(axis=-1, out=summation_out)

        return summation_out.argmin(axis=-1)

    else:
        h, w, n, _ = x.shape
        _, _, _, dim_k = y.shape

        z = x ^ y
        z = rearrange(
            z, "height width num_bytes samples -> height width samples num_bytes"
        )

        # Matrix mul faster than .sum()
        # https://github.com/numpy/numpy/issues/16158
        vec = np.ones(n, dtype=np.uint8)
        z = xp.matmul(z, vec)

        return z.argmin(axis=-1)


def hamming_distance_8bit(xp: ModuleType = np):
    return xp.array([bin(i).count("1") for i in range(256)], dtype=xp.uint8)


@profile
def pydescent_minimum_distance(x, y):
    h, w, n, _ = x.shape
    x = rearrange(x, "h w n 1 -> (h w) n")

    _, _, _, s = y.shape
    y = rearrange(y, "1 1 n s -> s n")

    logging.info("Preparing index")
    index = pynndescent.NNDescent(
        y,
        metric="hamming",
        n_neighbors=200,
        diversify_prob=0.0,
        pruning_degree_multiplier=6.0,
    )
    index.prepare()

    logging.info("Querying neighbours")
    query = index.query(x, k=1, epsilon=0.6)
    neighbors = query[0][:, 0]
    neighbors = rearrange(neighbors, "(h w)-> h w", h=h, w=w)

    return neighbors


def test_minimum_distance():
    num_bits = 10
    message_ll = gray_mapping(num_bits)
    bch = galois.BCH(63, 10)

    code_LUT = bch.encode(galois.GF2(message_ll))
    code_LUT = code_LUT.view(np.ndarray).astype(int)

    N, n = code_LUT.shape
    h = w = int(np.sqrt(N))
    x = rearrange(code_LUT, "(h w) c -> h w c 1", h=h, w=w).copy()

    # Moderate corruption
    p = 0.26
    noise = np.random.binomial(1, p, x.shape).astype(bool)
    x ^= noise
    y = rearrange(code_LUT, "N c -> 1 1 c N")

    # Check unpacked version
    indices_binary = minimum_distance(x, y)

    indices_approx = pydescent_minimum_distance(x, y)

    # Stats
    print(f"Approximate NN accuracy {(indices_binary == indices_approx).mean()}")
    print(
        f"Minimum distance accuracy {(indices_binary == np.arange(N).reshape(h, w)).mean()}"
    )

    # Check packed versions
    x = packbits_strided(x)
    y = packbits_strided(y)
    indices_packed = minimum_distance(x, y, hamming_dist_LUT=hamming_distance_8bit())

    indices_numba = numba_minimum_distance(
        x, y, hamming_dist_LUT=hamming_distance_8bit()
    )

    # breakpoint()
    assert np.array_equal(indices_binary, indices_packed)
    assert np.array_equal(indices_binary, indices_numba)
    # assert np.array_equal(indices_binary, indices_approx)


if __name__ == "__main__":
    # Params
    num_bits = 10
    bch = galois.BCH(63, 10)
    p = 0.26

    # BCH encoding of Gray codes
    message_ll = gray_mapping(num_bits)
    code_LUT = bch.encode(galois.GF2(message_ll))
    code_LUT = code_LUT.view(np.ndarray).astype(int)

    N, n = code_LUT.shape

    if CUPY_INSTALLED:
        h = w = N
    else:
        h = w = int(np.sqrt(N))
        # h = w = N

    x = rearrange(code_LUT, "(h w) c -> h w c 1", h=h, w=w).copy()
    x = repeat(x, "h w c 1-> (h_repeat h) (w_repeat w) c 1", h_repeat=10, w_repeat=10)

    # Corruption
    noise = np.random.binomial(1, p, x.shape).astype(bool)
    x ^= noise
    y = rearrange(code_LUT, "N c -> 1 1 c N")

    pydescent_minimum_distance(x, y)
    exit(1)

    x = packbits_strided(x)
    y = packbits_strided(y)
    hamming_dist_LUT = hamming_distance_8bit()

    if CUPY_INSTALLED:
        hook = memory_hooks.LineProfileHook()

        with hook:
            x = xp.asarray(x)
            y = xp.asarray(y)
            hamming_dist_LUT = xp.asarray(hamming_dist_LUT)

        hook.print_report()
        # print(time.repeat(minimum_distance, (x, y), n_repeat=10))
        print(time.repeat(minimum_distance, (x, y, hamming_dist_LUT), n_repeat=10))

    else:
        print(minimum_distance(x, y, hamming_dist_LUT))
