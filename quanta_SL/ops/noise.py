import math
import random
from types import ModuleType
from typing import Any, Union, Tuple

import numpy as np
from einops import repeat
from nptyping import NDArray
from numba import vectorize, boolean, uint8, float64, int64, njit, prange
from quanta_SL.utils.package_gpu_checker import xp


@vectorize(
    [
        boolean(uint8, float64, float64, float64, int64, float64, boolean),
        boolean(int64, float64, float64, float64, int64, float64, boolean),
    ],
    cache=True,
    fastmath=True,
    target="parallel",
)
def shot_noise_corrupt(
    bit: int,
    phi_P: float,
    phi_A: float,
    t_exp: float,
    num_frames: int,
    tau: float,
    use_complementary: bool,
):
    """
    Corrupt a code word with shot noise

    :param bit: Binary or Uint8
    :param phi_P: Photon flux in "on" state
    :param phi_A: Photon flux in "off" state
    :param t_exp: SPAD exposure time
    :param num_frames: SPAD oversampling per projected bit.
    :param tau: Threshold (normalized by num_frames).
        1 if num_photons > num_frames * tau
        0 otherwise
    :param use_complementary: Use complementary strategy to compare.
        Involves projecting a code and its complement.
        Compare the #photons to figure out.
        Effective only when num_frames > 1.
    :return: Corrupted code
    """
    # Ambient vs "On" illum.
    phi = phi_P if bit else phi_A
    prob = 1 - math.exp(-phi * t_exp / num_frames)

    out = 0
    for _ in range(num_frames):
        out += random.random() < prob
    out /= num_frames

    if use_complementary:
        phi_comp = phi_A if bit else phi_P
        prob_comp = 1 - math.exp(-phi_comp * t_exp / num_frames)

        out_comp = 0
        for _ in range(num_frames):
            out_comp += random.random() < prob_comp
        out_comp /= num_frames

        # Compare to complementary pattern
        return out > out_comp
    else:
        # Threshold
        return out > tau


@njit(cache=True, parallel=True, nogil=True)
def fixed_bit_flip_corrupt(x: NDArray[bool], noisy_bits: int) -> NDArray[bool]:
    """
    Generate noise pattern with fixed no of bit flips

    :param x: 2D array (code_words x code_dim)
    :param noisy_bits: Corrupted bits per code
    :return: Corrupted x
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

    return x ^ noise_mat


def photon_arrival(
    num_frames: int,
    phi: NDArray[(Any, Any), float],
    t_exp: Union[float, NDArray[(Any, Any), float]],
    size: Tuple[int],
) -> NDArray[(Any, Any), int]:
    """
    Simulate SPAD photon arrival.
    Operating in Quanta SPAD mode.
    Each frames thresholded if it receives atleast 1 photon

    :param num_frames: Oversampling rate
    :param phi: Incident photon flux (in seconds^-1)
    :param t_exp: Exposure time (in seconds)
    :param size: Number of samples to draw, usually, (h x w x vec_length)
    :return: Number of frames active
    """
    return xp.random.binomial(
        n=num_frames,
        p=1 - xp.exp(-phi * t_exp / num_frames),
        size=size,
        dtype=xp.int32,
    )


def shot_noise_corrupt_gpu(
    code: NDArray[int],
    phi_P: NDArray[(Any, Any), float],
    phi_A: NDArray[(Any, Any), float],
    t_exp: Union[float, NDArray[(Any, Any), float]],
    num_frames: int,
    tau: Union[float, NDArray[(Any, Any), float]] = 0.5,
    use_complementary: bool = False,
) -> NDArray[int]:
    """
    Corrupt a code word with shot noise

    :param code: Code word \in F_2^n
    :param phi_P: Photon flux in "on" state
    :param phi_A: Photon flux in "off" state
    :param t_exp: SPAD exposure time
    :param num_frames: Number of frames to repetitively capture at the SPAD end
    :param tau: Threshold (normalized by num_frames).
        1 if #photons > num_frames * tau
        0 otherwise
    :param use_complementary: Use complementary strategy to compare.
        Involves projecting a code and its complement.
        Compare the #photons to figure out.
        Effective only when num_frames > 1.
    :return:
    """
    h, w, _ = phi_P.shape

    corrupt_code = repeat(code, "1 1 n -> h w n", h=h, w=w)

    zero_locations = xp.where(code == 0)[0]
    one_locations = xp.where(code == 1)[0]

    # Sample from Binom(num_frames, 1 - exp(-Phi x t_exp))
    # Simulates single cycle photon arrival (ie, atleast 1 photon arrives)
    phi_A_arrived = photon_arrival(
        num_frames, phi_A, t_exp, (h, w, len(zero_locations))
    )

    phi_P_arrived = photon_arrival(num_frames, phi_P, t_exp, (h, w, len(one_locations)))

    if use_complementary:
        # Complementary frames
        phi_A_complementary = photon_arrival(
            num_frames, phi_P, t_exp, (h, w, len(zero_locations))
        )

        phi_P_complementary = photon_arrival(
            num_frames, phi_A, t_exp, (h, w, len(one_locations))
        )

        # Averaging strategy (sensor side)
        phi_A_flips = ~(phi_A_arrived < phi_A_complementary)
        phi_P_flips = ~(phi_P_arrived > phi_P_complementary)
    else:
        # Averaging strategy (sensor side)
        phi_A_arrived = phi_A_arrived > num_frames * tau
        phi_P_arrived = phi_P_arrived > num_frames * tau

        # Flips are when error occurs
        # ie, if a "0" arrives
        # or a "1" doesn't
        phi_A_flips = phi_A_arrived
        phi_P_flips = ~phi_P_arrived

    # GF2: addition is XOR
    corrupt_code[:, :, zero_locations] ^= phi_A_flips
    corrupt_code[:, :, one_locations] ^= phi_P_flips

    return corrupt_code
