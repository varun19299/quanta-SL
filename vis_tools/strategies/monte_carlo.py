"""
Strategies whose expected error is obtained
via Monte Carlo methods
"""
import dataclasses
import logging
from typing import Union, Any

import numpy as np
from einops import rearrange, repeat
from galois import GF2, BCH
from nptyping import NDArray
from tqdm import tqdm

from vis_tools.strategies import metaclass

FORMAT = "[%(filename)s-%(funcName)s():%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

try:
    import cupy as cp

    CUPY_INSTALLED = True
    logging.info(
        f"CuPy installation found, with {cp.cuda.runtime.getDeviceCount()} GPU(s)."
    )

except ImportError:
    CUPY_INSTALLED = False
    logging.warning("No CuPy installation detected. Using Numpy, may be slow.")

from vis_tools.strategies.utils import unpackbits


def shot_noise_corrupt(
    code: NDArray[int],
    phi_P: NDArray[(Any, Any), float],
    phi_A: NDArray[(Any, Any), float],
    t_exp: Union[float, NDArray[(Any, Any), float]],
    num_frames: int,
    threshold: Union[float, NDArray[(Any, Any), float]] = 0.5,
    use_complementary: bool = False,
) -> NDArray[int]:
    if CUPY_INSTALLED:
        xp = cp.get_array_module(phi_P)
    else:
        xp = np

    h, w = phi_P.shape
    phi_A = rearrange(phi_A, "h w -> h w 1 1")
    phi_P = rearrange(phi_P, "h w -> h w 1 1")

    corrupt_code = repeat(code, "n -> h w n", h=h, w=w)

    zero_locations = xp.where(code == 0)[0]
    one_locations = xp.where(code == 1)[0]

    # Sample from poisson
    # Threshold if atleast 1 photon arrives
    phi_A_arrived = xp.random.poisson(
        phi_A * t_exp / num_frames, (h, w, len(zero_locations), num_frames)
    )
    phi_A_arrived = (phi_A_arrived > 0).sum(axis=3, dtype=np.uint8)

    phi_P_arrived = xp.random.poisson(
        phi_P * t_exp / num_frames, (h, w, len(one_locations), num_frames)
    )
    phi_P_arrived = (phi_P_arrived > 0).sum(axis=3, dtype=np.uint8)

    if use_complementary:
        # Complementary frames
        phi_A_complementary = xp.random.poisson(
            phi_P * t_exp / num_frames, (h, w, len(zero_locations), num_frames)
        )
        phi_A_complementary = (phi_A_complementary > 0).sum(axis=3, dtype=np.uint8)

        phi_P_complementary = xp.random.poisson(
            phi_A * t_exp / num_frames, (h, w, len(one_locations), num_frames)
        )
        phi_P_complementary = (phi_P_complementary > 0).sum(axis=3, dtype=np.uint8)

        # Averaging strategy (sensor side)
        phi_A_flips = ~(phi_A_arrived < phi_A_complementary)
        phi_P_flips = ~(phi_P_arrived > phi_P_complementary)
    else:
        # Averaging strategy (sensor side)
        phi_A_arrived = phi_A_arrived > num_frames * threshold
        phi_P_arrived = phi_P_arrived > num_frames * threshold

        # Flips are when error occurs
        # ie, if a "0" arrives
        # or a "1" doesn't
        phi_A_flips = phi_A_arrived
        phi_P_flips = ~phi_P_arrived

    corrupt_code[:, :, zero_locations] += phi_A_flips
    corrupt_code[:, :, one_locations] += phi_P_flips
    corrupt_code = corrupt_code % 2

    return corrupt_code


def bch_LUT(
    phi_P: NDArray[(Any, Any), float],
    phi_A: NDArray[(Any, Any), float],
    t_exp: Union[float, NDArray[(Any, Any), float]],
    bch_tuple: metaclass.BCH,
    num_bits: int = 10,
    num_frames: int = 1,
    threshold: Union[float, NDArray[(Any, Any), float]] = 0.5,
    use_complementary: bool = False,
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

    :param num_bits: No of structured light frames to transmit
    :param num_frames: Frames for averaging based strategy. For naive, set 1
    :param threshold: Threshold. Can be set as a function of \Phi_p, \Phi_a, \t_{exp}
    :param use_complementary: Use Complementary strategy.
        Should be done with num_frames > 1
        Captures double the number of frames

    :param monte_carlo_iter: MC averaging iterations, 1 should suffice most of the times.
    :return: eval_error, MC estimate of expected error.
    """
    # Ensure all arrays are on the same device
    if CUPY_INSTALLED:
        # Move arrays to GPU
        phi_P = cp.asarray(phi_P)
        phi_A = cp.asarray(phi_A)
        xp = cp
    else:
        xp = np

    # Code parameters
    n, k, t = dataclasses.astuple(bch_tuple)
    bch = BCH(n, k)

    # Cannot send a message longer than bch message length
    assert (
        num_bits <= k
    ), f"Num-bits {num_bits} must be lesser than BCH message dim {k}."

    # Generate BCH codes
    logging.info("Building BCH code space...")
    message_ll = np.arange(pow(2, num_bits))
    message_ll = GF2(unpackbits(message_ll, k))
    code_LUT = xp.asarray(bch.encode(message_ll))
    logging.info(
        rf"Generated code space in subset of C: F_2^{k} \to F_2^{n} with {num_bits} bits."
    )

    eval_error = xp.zeros_like(phi_P)

    code_LUT = repeat(code_LUT, "N n -> (repeat N) n", repeat=monte_carlo_iter)
    N, n = code_LUT.shape

    for code in tqdm(code_LUT):
        corrupt_code = shot_noise_corrupt(
            code,
            phi_P,
            phi_A,
            t_exp,
            num_frames,
            threshold,
            use_complementary=use_complementary,
        )

        # Hamming distance
        # GF2: addition == subtraction
        distance = rearrange(code, "n -> 1 1 n") + corrupt_code
        distance = (distance % 2).sum(axis=2)

        ## TODO: Use syndrome or MLD decoder to get more realistic limits (~ LECC)
        eval_error += distance > t

    eval_error /= N

    # Move from device to host
    if CUPY_INSTALLED:
        eval_error = eval_error.get()

    return eval_error


if __name__ == "__main__":
    phi_proj = np.logspace(3, 5, num=512)
    phi_A = np.logspace(3, 5, num=512)

    phi_P_mesh, phi_A_mesh = np.meshgrid(phi_A + phi_proj, phi_A, indexing="ij")

    # DMD framerate
    # 0.1 millisecond or 10^4 FPS
    t_exp = 1e-4

    n = 127
    k = 15
    t = 40

    bch_LUT(
        phi_P_mesh,
        phi_A_mesh,
        t_exp,
        bch_tuple=metaclass.BCH(n, k, t),
        num_frames=10,
        use_complementary=True,
    )
