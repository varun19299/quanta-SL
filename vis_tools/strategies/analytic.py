"""
Strategies whose expected error
is derived in closed-form
"""

from math import ceil

import numpy as np
from einops import repeat, rearrange
from nptyping import NDArray
from scipy.special import erf, comb


def naive(phi_P: NDArray, phi_A: NDArray, t_exp: float, bits: int = 10):
    prob_y_1_x_1 = 1 - np.exp(-phi_P * t_exp)
    prob_y_0_x_0 = np.exp(-phi_A * t_exp)

    return 1 - (prob_y_0_x_0 / 2 + prob_y_1_x_1 / 2) ** bits


def naive_conventional(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    threshold: float = 1,
    Q_e: float = 1,
    N_r: float = 1e4,
    bits: int = 10,
):
    sigma_1 = np.sqrt(Q_e * phi_P * t_exp + N_r ** 2)
    mu_1 = Q_e * phi_P * t_exp
    frac_1 = (threshold - mu_1) / (sigma_1 * np.sqrt(2))
    prob_y_1_x_1 = 0.5 * (1 - erf(frac_1))

    sigma_0 = np.sqrt(Q_e * phi_A * t_exp + N_r ** 2)
    mu_0 = Q_e * phi_A * t_exp
    frac_0 = (threshold - mu_0) / (sigma_0 * np.sqrt(2))
    prob_y_0_x_0 = 0.5 * (1 + erf(frac_0))

    return 1 - (prob_y_0_x_0 / 2 + prob_y_1_x_1 / 2) ** bits


"""
Average strategies
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
    threshold = np.ceil(tau * num_frames)
    return threshold, tau


def average_conventional(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    num_frames: 10,
    threshold: float = 1,
    Q_e: float = 1,
    N_r: float = 1e4,
    bits: int = 10,
):
    sigma_1 = np.sqrt(Q_e * phi_P * t_exp / num_frames + N_r ** 2)
    mu_1 = Q_e * phi_P * t_exp / num_frames
    frac_1 = (threshold - mu_1) / (sigma_1 * np.sqrt(2))
    prob_y_1_x_1 = 0.5 * (1 - erf(frac_1))

    sigma_0 = np.sqrt(Q_e * phi_A * t_exp / num_frames + N_r ** 2)
    mu_0 = Q_e * phi_A * t_exp / num_frames
    frac_0 = (threshold - mu_0) / (sigma_0 * np.sqrt(2))
    prob_y_0_x_0 = 0.5 * (1 + erf(frac_0))

    return 1 - (prob_y_0_x_0 / 2 + prob_y_1_x_1 / 2) ** bits


def average(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    num_frames: 10,
    threshold: int = None,
    bits: int = 10,
):
    if not isinstance(threshold, np.ndarray):
        threshold = np.array([ceil((num_frames + 1) / 2)])
        threshold = repeat(threshold, "1 -> h w 1", h=phi_P.shape[0], w=phi_P.shape[1])

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

    return 1 - (prob_y_0_x_0 / 2 + prob_y_1_x_1 / 2) ** bits


def average_fixed(*args, **kwargs):
    return average(*args, **kwargs)


def average_optimal(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    num_frames: 10,
    bits: int = 10,
):
    threshold_ll, tau_ll = optimal_threshold(phi_P, phi_A, t_exp, num_frames)
    threshold_ll = rearrange(threshold_ll, "h w -> h w 1")
    return average(phi_P, phi_A, t_exp, num_frames, threshold_ll, bits)
