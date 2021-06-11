from math import ceil
from typing import Tuple, Union

try:
    import cupy as cp

    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False

import numpy as np
from einops import repeat, rearrange
from nptyping import NDArray
from scipy.io import loadmat
from scipy.special import erf, comb
from tqdm import tqdm


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


def average_optimal_threshold(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    num_frames: 10,
    bits: int = 10,
):
    threshold_ll, tau_ll = optimal_threshold(phi_P, phi_A, t_exp, num_frames)
    threshold_ll = rearrange(threshold_ll, "h w -> h w 1")
    return average(phi_P, phi_A, t_exp, num_frames, threshold_ll, bits)


"""
BCH strategies
"""


def bch_LUT(
    phi_P: np.ndarray,
    phi_A: np.ndarray,
    t_exp: float,
    bch_tuple: Tuple[int, int, int],
    code_LUT: np.ndarray,
    num_frames: int = 1,
    threshold: Union[float, np.ndarray] = 0.5,
    monte_carlo_iter: int = 1,
) -> NDArray:
    """
    BCH strategy from a LUT

    :param phi_P: meshgrid of ambient + projector flux
    :param phi_A: meshgrid of ambient flux
    :param t_exp: exposure time
    :param bch_tuple: (n, k, t)
        n: code length
        k: message length
        t: error correcting length (typically \floor((d-1)/2), could be more with list decoding)

    :param code_LUT: Look Up Table for BCH code

    :param num_frames: Frames for averaging based strategy. For naive, set 1
    :param threshold: Threshold. Can be set as a function of \Phi_p, \Phi_a, \t_{exp}
    :param monte_carlo_iter: MC averaging iterations, 1 should suffice most of the times.
    :return:
    """
    # Ensure all arrays are on the same device
    if CUPY_INSTALLED:
        assert (
            cp.get_array_module(phi_P)
            == cp.get_array_module(phi_A)
            == cp.get_array_module(code_LUT)
        )
        xp = cp.get_array_module(phi_P)
    else:
        xp = np

    # Code parameters
    n, k, t = bch_tuple

    eval_error = xp.zeros_like(phi_P)

    h, w = phi_P.shape
    phi_A = rearrange(phi_A, "h w -> h w 1 1")
    phi_P = rearrange(phi_P, "h w -> h w 1 1")

    code_LUT = repeat(code_LUT, "N n -> (repeat N) n", repeat=monte_carlo_iter)
    for code in tqdm(code_LUT):
        zero_locations = xp.where(code == 0)[0]
        one_locations = xp.where(code == 1)[0]

        # Noisy transmit
        # Phi A
        # phi_A_arrived = xp.zeros((h, w, n), dtype=xp.int32)
        # phi_P_arrived = xp.zeros((h, w, n), dtype=xp.int32)
        #
        # for i in range(num_frames):
        #     # arrival_times = xp.random.exponential(1 / phi_A, (h, w, n))
        #     # phi_A_arrived += arrival_times < t_exp / num_frames
        #
        #     phi_A_arrived += (
        #         xp.random.exponential(phi_A * t_exp / num_frames, (h, w, n)).astype(
        #             xp.int32
        #         )
        #         > 0
        #     )
        #
        #     # Phi P
        #     # arrival_times = xp.random.exponential(1 / phi_P, (h, w, n))
        #     # phi_P_arrived += arrival_times < t_exp / num_frames
        #
        #     phi_P_arrived += (
        #         xp.random.exponential(phi_P * t_exp / num_frames, (h, w, n)).astype(
        #             xp.int32
        #         )
        #         > 0
        #     )

        phi_A_arrived = (
            xp.random.exponential(phi_A * t_exp / num_frames, (h, w, n, num_frames)) > 0
        ).sum(axis=3)

        phi_P_arrived = (
            xp.random.exponential(phi_P * t_exp / num_frames, (h, w, n, num_frames)) > 0
        ).sum(axis=3)

        phi_A_arrived = phi_A_arrived > num_frames * threshold
        phi_P_arrived = phi_P_arrived > num_frames * threshold

        phi_A_flips = phi_A_arrived
        phi_P_flips = 1 - phi_P_arrived

        # Flip em!
        zero_flips = phi_A_flips[:, :, zero_locations].sum(axis=2)
        one_flips = phi_P_flips[:, :, one_locations].sum(axis=2)

        distance = zero_flips + one_flips
        eval_error += distance > t

    eval_error /= code_LUT.shape[0]
    return eval_error


if __name__ == "__main__":
    phi_proj = np.logspace(4, 8, num=512)
    phi_A = np.logspace(0, 4, num=512)

    phi_P_mesh, phi_A_mesh = np.meshgrid(phi_A + phi_proj, phi_A, indexing="ij")

    # DMD framerate
    # 0.1 millisecond or 10^4 FPS
    t_exp = 1e-4

    n = 63
    k = 10
    t = 10
    LUT_path = "BCH/bch_LUT.mat"
    code_LUT = loadmat(LUT_path)[f"bch_{n}_{k}_code"].astype(int)

    if CUPY_INSTALLED:
        phi_P_mesh = cp.asarray(phi_P_mesh)
        phi_A_mesh = cp.asarray(phi_A_mesh)
        code_LUT = cp.asarray(code_LUT)

    bch_LUT(
        phi_P_mesh,
        phi_A_mesh,
        t_exp,
        bch_tuple=(n, k, t),
        code_LUT=code_LUT,
        num_frames=10,
    )
