"""
Coding theory related
"""
from types import ModuleType

import numpy as np
from nptyping import NDArray
from numba import njit, prange
from collections import namedtuple

from quanta_SL.ops.linalg import mean

StripeWidthStats = namedtuple(
    "StripeWidthStats", ["minSW", "framewise_minSW", "framewise_avgSW"]
)


@njit(cache=True, fastmath=True, nogil=True)
def stripe_width_stats(code_LUT: np.ndarray):
    """
    Stripe width statistics (run length) of a coding strategy
    :param code_LUT: Look up table describing the code.
    The code frames are
        c_0[0], c_1[0], ..., c_N[0]
        .                        .
        .                        .
        .                        .
        c_0[n-1], c_1[n-1], ..., c_N[n-1]

    Where N is the number of messages, n is the code dimension.
    Typically, N= 2^k
    C : F^k \to F^n

    We account for circular shift while calculating stripe width

    :return:
        minimum stripe width across all frames
        frame-wise minimum stripe width
        frame-wise average stripe width
    """
    assert code_LUT.ndim == 2, "Must be message space x code dim"

    # Per frame statistics
    min_stripe_ll = []
    mean_stripe_ll = []

    for frame in code_LUT.transpose():
        # Intra-frame statistics
        stripe_ll = []
        stripe_width = 1

        for curr_elem, next_elem in zip(frame, np.roll(frame, -1)):
            if curr_elem == next_elem:
                stripe_width += 1
            else:
                stripe_ll.append(stripe_width)
                # Reset
                stripe_width = 1

        # If the frame happens to be full black / full white
        if not stripe_ll:
            stripe_ll.append(stripe_width)

        # Consider minimum stripe
        # After circular shifting
        if len(stripe_ll) >= 2 and (frame[0] == frame[-1]):
            stripe_ll[0] += stripe_ll[-1]
            stripe_ll = stripe_ll[:-1]

        min_stripe_ll.append(min(stripe_ll))
        mean_stripe_ll.append(mean(stripe_ll))

    return StripeWidthStats(min(min_stripe_ll), min_stripe_ll, mean_stripe_ll)


@njit(cache=True, fastmath=True, nogil=True)
def minimum_hamming_distance(code_LUT: np.ndarray):
    minimum_hamming_distance = 0
    N, n = code_LUT.shape

    for i in range(N - 1):
        for j in range(i + 1, N):
            code_i = code_LUT[i]
            code_j = code_LUT[j]

            distance_ij = (code_i ^ code_j).sum()

            if not minimum_hamming_distance:
                minimum_hamming_distance = distance_ij
            else:
                if distance_ij < minimum_hamming_distance:
                    minimum_hamming_distance = distance_ij

    return minimum_hamming_distance


def hamming_distance_8bit(xp: ModuleType = np):
    """
    Hamming distance of each 8-bit uint (in binary representation).

        >>> hamming_distance_8bit(5) # 3
        >>> hamming_distance_8bit(8) # 1
        >>> hamming_distance_8bit(0) # 0

    :return: array with Look Up Table
    """
    return xp.array([bin(i).count("1") for i in range(256)], dtype=xp.uint8)


@njit(cache=True, parallel=True, nogil=True)
def bit_flip_noise(x: NDArray[int], noisy_bits: int) -> NDArray[int]:
    """
    Generate noise pattern with fixed bit flips

    :param x: 2D array (code_words x code_dim)
    :param noisy_bits: Corrupted bits per code
    :return: Noise pattern
    """
    assert x.ndim == 2
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

