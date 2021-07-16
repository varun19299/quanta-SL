"""
Strategies whose expected error is obtained
via Monte Carlo methods
"""
import dataclasses
import logging
from typing import Union, Any, Tuple

import numpy as np
from einops import rearrange, repeat, reduce
from galois import GF2, BCH
from nptyping import NDArray
from tqdm import tqdm

from vis_tools.strategies import metaclass

try:
    import cupy as cp

    CUPY_INSTALLED = True
    logging.info(
        f"CuPy installation found, with {cp.cuda.runtime.getDeviceCount()} GPU(s)."
    )

except ImportError:
    CUPY_INSTALLED = False
    logging.warning("No CuPy installation detected. Using Numpy, may be slow.")

from utils.array_ops import unpackbits


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
    if CUPY_INSTALLED:
        xp = cp.get_array_module(phi)
    else:
        xp = np
    return xp.random.binomial(
        n=num_frames,
        p=1 - xp.exp(-phi * t_exp / num_frames),
        size=size,
        dtype=xp.int32,
    )


def shot_noise_corrupt(
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
    if CUPY_INSTALLED:
        xp = cp.get_array_module(phi_P)
    else:
        xp = np

    h, w = phi_P.shape
    phi_A = rearrange(phi_A, "h w -> h w 1")
    phi_P = rearrange(phi_P, "h w -> h w 1")

    corrupt_code = repeat(code, "n -> h w n", h=h, w=w)

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


def _coding_LUT(
    phi_P: NDArray[(Any, Any), float],
    phi_A: NDArray[(Any, Any), float],
    t_exp: Union[float, NDArray[(Any, Any), float]],
    code_LUT: NDArray[(Any, Any), int],
    t_correctable: int,
    num_frames: int = 1,
    num_repeat: int = 1,
    tau: Union[float, NDArray[(Any, Any), float]] = 0.5,
    use_complementary: bool = False,
    monte_carlo_iter: int = 1,
    pbar_description: str = "",
    **unused_kwargs,
) -> NDArray:
    """
    Evaluate coding strategy using LUT

    :param phi_P: meshgrid of ambient + projector flux
    :param phi_A: meshgrid of ambient flux
    :param t_exp: exposure time
    :param t_correctable: correctable errors

    :param num_frames: Frames for averaging based strategy. For naive, set 1
    :param num_repeat: Number of repeated projections
    :param tau: Threshold (normalized by num_frames). Can be set as a function of \Phi_p, \Phi_a, \t_{exp}
    :param use_complementary: Use Complementary strategy.
        Should be done with num_frames > 1
        Captures double the number of frames

    :param monte_carlo_iter: MC averaging iterations, 1 should suffice most of the times.
    :param pbar_description: optional TQDM pbar description.
    :return: eval_error, MC estimate of expected error.
    """
    if CUPY_INSTALLED:
        # Move arrays to GPU
        phi_P = cp.asarray(phi_P)
        phi_A = cp.asarray(phi_A)
        xp = cp

        if isinstance(tau, np.ndarray):
            tau = cp.asarray(tau)
    else:
        xp = np

    eval_error = xp.zeros_like(phi_P)

    code_LUT = xp.asarray(code_LUT)
    code_LUT = repeat(
        code_LUT,
        "N n -> (mc_repeat N) n",
        mc_repeat=monte_carlo_iter,
    )
    N, n = code_LUT.shape

    for code in tqdm(code_LUT, desc=pbar_description, dynamic_ncols=True):
        corrupt_code = shot_noise_corrupt(
            repeat(
                code,
                "n -> (proj_repeat n)",
                proj_repeat=num_repeat,
            ),
            phi_P,
            phi_A,
            t_exp,
            num_frames,
            tau,
            use_complementary=use_complementary,
        )

        if num_repeat > 1:
            corrupt_code = rearrange(
                corrupt_code, "h w (repeat n) -> h w n repeat", repeat=num_repeat
            )
            corrupt_code = corrupt_code.mean(axis=3) > 0.5

        # Hamming distance
        # GF2: addition == XOR
        distance = rearrange(code, "n -> 1 1 n") ^ corrupt_code
        distance = reduce(distance, "h w n -> h w", "sum")

        ## TODO: Use syndrome or MLD decoder to get more realistic limits (~ LECC)
        eval_error += distance > t_correctable

    eval_error /= N

    # Move from device to host
    if CUPY_INSTALLED:
        eval_error_numpy = eval_error.get()

        # Free memory
        del phi_P, phi_A, tau, eval_error
        # xp.get_default_memory_pool().free_all_blocks()

        return eval_error_numpy
    else:
        return eval_error


def bch_coding(
    phi_P: NDArray[(Any, Any), float],
    phi_A: NDArray[(Any, Any), float],
    t_exp: Union[float, NDArray[(Any, Any), float]],
    bch_tuple: metaclass.BCH,
    num_bits: int = 10,
    num_frames: int = 1,
    num_repeat: int = 1,
    tau: Union[float, NDArray[(Any, Any), float]] = 0.5,
    use_complementary: bool = False,
    monte_carlo_iter: int = 1,
) -> NDArray:
    """
    BCH strategy evaluation (monte carlo)

    :param phi_P: meshgrid of ambient + projector flux
    :param phi_A: meshgrid of ambient flux
    :param t_exp: exposure time
    :param bch_tuple: (n, k, t)
        n: code length
        k: message length
        t: error correcting length (typically \floor((d-1)/2), could be more with list decoding)

    :param num_bits: No of structured light frames to transmit
    :param num_frames: Frames for averaging based strategy. For naive, set 1
    :param num_repeat: Number of repeated projections
    :param tau: Threshold (normalized by num_frames). Can be set as a function of \Phi_p, \Phi_a, \t_{exp}
    :param use_complementary: Use Complementary strategy.
        Should be done with num_frames > 1
        Captures double the number of frames

    :param monte_carlo_iter: MC averaging iterations, 1 should suffice most of the times.
    :return: eval_error, MC estimate of expected error.
    """
    kwargs = locals()

    # Code parameters
    n, k, t = dataclasses.astuple(bch_tuple)
    bch = BCH(n, k)

    # Cannot send a message longer than bch message length
    assert (
        num_bits <= k
    ), f"Num-bits {num_bits} must be lesser than BCH message dim {k}."

    # Generate BCH codes
    logging.debug("Building BCH code space...")
    message_ll = np.arange(pow(2, num_bits))
    message_ll = GF2(unpackbits(message_ll, k))
    code_LUT = bch.encode(message_ll)

    # Puncture
    code_LUT = code_LUT[:, k - num_bits :]
    logging.debug(
        rf"Generated code space in subset of C: F_2^{k} \to F_2^{n} with {num_bits} bits."
    )

    return _coding_LUT(
        code_LUT=code_LUT,
        t_correctable=t,
        pbar_description=rf"{bch_tuple}"
        rf"{'-list' if 2*t + 1 > bch_tuple.distance else ''}"
        rf"{'-comp' if use_complementary else ''}: F_2^{k} \to F_2^{n}",
        **kwargs,
    )


def repetition_coding(
    phi_P: NDArray[(Any, Any), float],
    phi_A: NDArray[(Any, Any), float],
    t_exp: Union[float, NDArray[(Any, Any), float]],
    repetition_tuple: metaclass.Repetition,
    num_bits: int = 10,
    num_frames: int = 1,
    tau: Union[float, NDArray[(Any, Any), float]] = 0.5,
    use_complementary: bool = False,
    monte_carlo_iter: int = 1,
) -> NDArray:
    """
    Repetition strategy MC evaluation.

    :param phi_P: meshgrid of ambient + projector flux
    :param phi_A: meshgrid of ambient flux
    :param t_exp: exposure time
    :param repetition_tuple: (n, k, t)
        n: code length
        k: message length
        t: error correcting length (typically \floor((d-1)/2), could be more with list decoding)

    :param num_bits: No of structured light frames to transmit
    :param num_frames: Frames for averaging based strategy. For naive, set 1
    :param tau: Threshold (normalized by num_frames). Can be set as a function of \Phi_p, \Phi_a, \t_{exp}
    :param use_complementary: Use Complementary strategy.
        Should be done with num_frames > 1
        Captures double the number of frames

    :param monte_carlo_iter: MC averaging iterations, 1 should suffice most of the times.
    :return: eval_error, MC estimate of expected error.
    """
    kwargs = locals()

    # Code parameters
    n, k, t = dataclasses.astuple(repetition_tuple)

    # Cannot send a message longer than repetition message length
    assert num_bits <= k, f"Num-bits {num_bits} must be lesser than message dim {k}."

    # Generate Repetition codes
    message_ll = np.arange(pow(2, num_bits))
    message_ll = GF2(unpackbits(message_ll, k))
    logging.debug(
        rf"Generated Repetition code space in subset of C: F_2^{k} \to F_2^{n} with {num_bits} bits."
    )

    return _coding_LUT(
        code_LUT=message_ll,
        t_correctable=0,
        num_repeat=n // k,
        pbar_description=rf"{repetition_tuple}"
        rf"{'-comp' if use_complementary else ''}: F_2^{k} \to F_2^{n}",
        **kwargs,
    )


if __name__ == "__main__":
    phi_proj = np.logspace(3, 5, num=32)
    phi_A = np.logspace(3, 5, num=32)

    phi_P_mesh, phi_A_mesh = np.meshgrid(phi_A + phi_proj, phi_A, indexing="ij")

    # DMD framerate
    # 0.1 millisecond or 10^4 FPS
    t_exp = 1e-4

    n = 15
    k = 11
    t = 1

    bch_coding(
        phi_P_mesh,
        phi_A_mesh,
        t_exp,
        bch_tuple=metaclass.BCH(n, k, t),
        num_frames=10,
        use_complementary=True,
    )
