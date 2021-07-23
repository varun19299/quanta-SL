"""
Coding theory related
"""
import numpy as np
from numba import njit

from ops.linalg import mean


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

    return min(min_stripe_ll), min_stripe_ll, mean_stripe_ll


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