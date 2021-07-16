"""
Strategies whose expected error
is derived in closed-form
"""

from copy import copy
from dataclasses import astuple
from math import ceil
from typing import Any, Union, Tuple

import galois
import numpy as np
from einops import repeat, rearrange
from nptyping import NDArray
from numba import njit, prange
from scipy.special import erf, comb

import utils.math_ops
from vis_tools.strategies import metaclass
from utils.array_ops import unpackbits


def naive(phi_P: NDArray, phi_A: NDArray, t_exp: float, num_bits: int = 10):
    prob_y_1_x_1 = 1 - np.exp(-phi_P * t_exp)
    prob_y_0_x_0 = np.exp(-phi_A * t_exp)

    return no_coding(prob_y_1_x_1, prob_y_0_x_0, num_bits)


def naive_conventional(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    threshold: float = 1,
    Q_e: float = 1,
    N_r: float = 1e4,
    num_bits: int = 10,
):
    sigma_1 = np.sqrt(Q_e * phi_P * t_exp + N_r ** 2)
    mu_1 = Q_e * phi_P * t_exp
    frac_1 = (threshold - mu_1) / (sigma_1 * np.sqrt(2))
    prob_y_1_x_1 = 0.5 * (1 - erf(frac_1))

    sigma_0 = np.sqrt(Q_e * phi_A * t_exp + N_r ** 2)
    mu_0 = Q_e * phi_A * t_exp
    frac_0 = (threshold - mu_0) / (sigma_0 * np.sqrt(2))
    prob_y_0_x_0 = 0.5 * (1 + erf(frac_0))

    return no_coding(prob_y_1_x_1, prob_y_0_x_0, num_bits)


"""
Optimal Threshold
"""


def optimal_threshold(
    phi_P,
    phi_A,
    t_exp: float,
    num_frames: 10,
) -> NDArray:
    """
    Determine optimal threshold for avg strategy
    (known phi_P, phi_A)
    """
    N_p = phi_P * t_exp
    N_a = phi_A * t_exp

    num = N_p - N_a

    p = 1 - np.exp(-N_p / num_frames)
    q = 1 - np.exp(-N_a / num_frames)
    denom = N_p - N_a + num_frames * (np.log(p) - np.log(q))

    tau = num / denom
    return tau


"""
Average strategies
"""


def average_conventional(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    num_frames: 10,
    threshold: float = 1,
    Q_e: float = 1,
    N_r: float = 1e4,
    num_bits: int = 10,
):
    sigma_1 = np.sqrt(Q_e * phi_P * t_exp / num_frames + N_r ** 2)
    mu_1 = Q_e * phi_P * t_exp / num_frames
    frac_1 = (threshold - mu_1) / (sigma_1 * np.sqrt(2))
    prob_y_1_x_1 = 0.5 * (1 - erf(frac_1))

    sigma_0 = np.sqrt(Q_e * phi_A * t_exp / num_frames + N_r ** 2)
    mu_0 = Q_e * phi_A * t_exp / num_frames
    frac_0 = (threshold - mu_0) / (sigma_0 * np.sqrt(2))
    prob_y_0_x_0 = 0.5 * (1 + erf(frac_0))

    return no_coding(prob_y_1_x_1, prob_y_0_x_0, num_bits)


def average_probabilities(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    num_frames: int = 1,
    tau: float = 0.5,
    **unused_kwargs,
) -> Tuple[NDArray, NDArray]:
    if not isinstance(tau, np.ndarray):
        threshold = np.array(ceil((num_frames + 1) / 2)).reshape(-1)
        threshold = repeat(threshold, "1 -> h w 1", h=phi_P.shape[0], w=phi_P.shape[1])
    else:
        threshold = np.ceil(tau * num_frames)

    j_ll = np.arange(start=0, stop=num_frames + 1)
    comb_ll = rearrange(comb(num_frames, j_ll), "d -> 1 1 d")

    # 10_C_0, 10_C_1, ..., 10_C_5
    mask_ll = j_ll < threshold

    prob_naive_y_0_x_0 = np.exp(-phi_A * t_exp)

    prob_frame_y_1_x_0 = 1 - np.exp(-phi_A * t_exp / num_frames)
    prob_frame_y_0_x_0 = 1 - prob_frame_y_1_x_0

    # Conditioning
    dtype = prob_frame_y_0_x_0.dtype
    prob_frame_y_0_x_0 = np.maximum(prob_frame_y_0_x_0, np.finfo(dtype).eps)

    frac = prob_frame_y_1_x_0 / prob_frame_y_0_x_0
    frac = [frac ** j for j in j_ll]
    frac = np.stack(frac, axis=-1)
    mult = (frac * comb_ll * mask_ll).sum(axis=-1)
    prob_y_0_x_0 = prob_naive_y_0_x_0 * mult

    # 10_C_6, 10_C_7, ..., 10_C_10
    mask_ll = j_ll >= threshold

    prob_naive_y_0_x_1 = np.exp(-phi_P * t_exp)
    prob_frame_y_1_x_1 = 1 - np.exp(-phi_P * t_exp / num_frames)
    prob_frame_y_0_x_1 = 1 - prob_frame_y_1_x_1

    frac = prob_frame_y_0_x_1 / prob_frame_y_1_x_1
    frac = [frac ** (num_frames - j) for j in j_ll]
    frac = np.stack(frac, axis=-1)
    mult = (frac * comb_ll * mask_ll).sum(axis=-1)
    prob_y_1_x_1 = (prob_frame_y_1_x_1 ** num_frames) * mult

    return prob_y_1_x_1, prob_y_0_x_0


def average_fixed(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    num_frames: 10,
    threshold: int = None,
    num_bits: int = 10,
):
    kwargs = copy(locals())
    prob_y_1_x_1, prob_y_0_x_0 = average_probabilities(**kwargs)

    return no_coding(prob_y_1_x_1, prob_y_0_x_0, num_bits)


def average_optimal(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    num_frames: 10,
    num_bits: int = 10,
):
    tau = optimal_threshold(phi_P, phi_A, t_exp, num_frames)
    tau = rearrange(tau, "h w -> h w 1")
    kwargs = copy(locals())

    return average_probabilities(**kwargs)


"""
Bounded Error models
"""


def no_coding(
    p: NDArray[(Any, Any), float], q: NDArray[(Any, Any), float], num_bits: int = 10
):
    """
    Error when no coding scheme is used

    :param p: P(y=1|x=1)
    :param q: P(y=0|x=0)
    :param num_bits: message dimension
    :return: Expected error
    """
    return 1 - pow((p + q) / 2, num_bits)


@njit(parallel=True, fastmath=True)
def bounded_error_coding(
    p: NDArray[(Any, Any), float], q: NDArray[(Any, Any), float], code_LUT, t: int
):
    """
    Evaluate a general coding scheme using the Bounded error model

    :param p: P(y=1|x=1)
    :param q: P(y=0|x=0)
    :param code_LUT: Look up table for coding scheme [Message num x code length]
    :param t: Correctable errors (can be worst-case or LECC)
    :return: Expected error
    """
    num_message, n = code_LUT.shape

    p = np.asarray(p)
    q = np.asarray(q)

    assert p.shape == q.shape
    p_correct = np.zeros_like(p)

    for c in prange(num_message):
        code = code_LUT[c]
        num_ones = (code == 1).sum()
        num_zeros = (code == 0).sum()

        for e in range(t + 1):
            for r in range(min(e, num_ones) + 1):
                s = e - r
                term_0 = (
                        utils.math_ops.comb(num_zeros, s)
                        * np.power(q, num_zeros - s)
                        * np.power(1 - q, s)
                )
                term_1 = (
                        utils.math_ops.comb(num_ones, r)
                        * np.power(p, num_ones - r)
                        * np.power(1 - p, r)
                )
                p_correct += term_0 * term_1

    return 1 - p_correct / num_message


def bch_coding(
    phi_P: NDArray[(Any, Any), float],
    phi_A: NDArray[(Any, Any), float],
    t_exp: Union[float, NDArray[(Any, Any), float]],
    bch_tuple: metaclass.BCH,
    num_bits: int = 10,
    num_frames: int = 1,
    tau: Union[float, NDArray[(Any, Any), float]] = 0.5,
    use_optimal_tau: bool = False,
    use_complementary: bool = False,
) -> NDArray:
    if use_optimal_tau:
        tau = optimal_threshold(phi_P, phi_A, t_exp, num_frames)
        tau = rearrange(tau, "h w -> h w 1")
    kwargs = copy(locals())

    prob_y_1_x_1, prob_y_0_x_0 = average_probabilities(**kwargs)

    # Code LUT
    message_ll = np.arange(pow(2, num_bits))
    message_ll = galois.GF2(unpackbits(message_ll))

    # BCH encode
    n, k, t = astuple(bch_tuple)
    bch = galois.BCH(n, k)
    code_LUT = bch.encode(message_ll)

    return bounded_error_coding(prob_y_1_x_1, prob_y_0_x_0, code_LUT, t)


if __name__ == "__main__":

    num_bits = 11
    message_ll = np.arange(pow(2, num_bits))
    message_ll = galois.GF2(unpackbits(message_ll))

    bch = galois.BCH(31, 11)
    code_LUT = bch.encode(message_ll)
