"""
Strategies whose expected error is obtained
via Monte Carlo methods
"""
from functools import partial
from typing import Union, Callable

import numpy as np
from einops import rearrange
from loguru import logger
from nptyping import NDArray
from tqdm import tqdm

from quanta_SL.decode.methods import (
    minimum_distance_decoding,
    repetition_decoding,
    read_off_decoding,
)
from quanta_SL.decode.minimum_distance.factory import (
    faiss_flat_index,
    faiss_flat_gpu_index,
    faiss_minimum_distance,
)
from quanta_SL.encode import metaclass
from quanta_SL.encode.message import (
    gray_message,
    message_to_permuation,
    message_to_inverse_permuation,
)
from quanta_SL.encode.strategies import (
    repetition_code_LUT,
    bch_code_LUT,
)
from quanta_SL.ops.binary import packbits_strided
from quanta_SL.ops.metrics import exact_error, squared_error
from quanta_SL.ops.noise import shot_noise_corrupt, shot_noise_corrupt_gpu
from quanta_SL.utils.gpu_status import FAISS_GPUs, CUPY_GPUs, move_to_gpu
from quanta_SL.vis_tools.error_evaluation.plotting import mesh_plot_2d


def coding_LUT(
    phi_P: NDArray[float],
    phi_A: NDArray[float],
    t_exp: Union[float, NDArray[float]],
    code_LUT: NDArray[int],
    decoding_func: Callable,
    error_metric: Callable = exact_error,
    num_frames: int = 1,
    tau: Union[float, NDArray[float]] = 0.5,
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

    :param code_LUT: Look Up Table, mapping projector columns (int) to code vectors.
    :param decoding_func: Decode projector column from code vector.
    :param error_metric: Error quantifier (L1, L2, exact, etc.)

    :param num_frames: Frames for averaging based strategy. For naive, set 1
    :param tau: Threshold (normalized by num_frames). Can be set as a function of \Phi_p, \Phi_a, \t_{exp}
    :param use_complementary: Use Complementary strategy.
        Should be done with num_frames > 1
        Captures double the number of frames

    :param monte_carlo_iter: MC averaging iterations, 1 should suffice most of the times.
    :param pbar_description: optional TQDM pbar description.
    :return: eval_error, MC estimate of expected error.
    """
    eval_error = np.zeros_like(phi_A, dtype=float)

    # Rearrange
    h, w = phi_A.shape
    N, n = code_LUT.shape
    phi_A = rearrange(phi_A, "h w -> h w 1")
    phi_P = rearrange(phi_P, "h w -> h w 1")

    # If CuPy installed
    if CUPY_GPUs:
        shot_noise_func = shot_noise_corrupt_gpu
        phi_P = move_to_gpu(phi_P)
        phi_A = move_to_gpu(phi_A)
        t_exp = move_to_gpu(t_exp)
        tau = move_to_gpu(tau)
    else:
        shot_noise_func = shot_noise_corrupt

    pbar = tqdm(
        total=len(code_LUT) * monte_carlo_iter,
        desc=pbar_description,
        dynamic_ncols=True,
    )

    for _ in range(monte_carlo_iter):
        for e, code in enumerate(code_LUT):
            pbar.update(1)
            code = rearrange(code, "n -> 1 1 n")

            corrupt_code = shot_noise_func(
                code, phi_P, phi_A, t_exp, num_frames, tau, use_complementary
            )

            corrupt_code = rearrange(corrupt_code, "h w n -> (h w) n")
            decoded_index = decoding_func(corrupt_code, code_LUT)
            decoded_index = rearrange(decoded_index, "(h w) -> h w", h=h, w=w)

            eval_error += error_metric(decoded_index, e)

    # Take average
    eval_error /= N * monte_carlo_iter

    eval_error = error_metric.post_mean_func(eval_error)

    return eval_error


def bch_coding(
    phi_P: NDArray[float],
    phi_A: NDArray[float],
    t_exp: Union[float, NDArray[float]],
    bch_tuple: metaclass.BCH,
    message_mapping: Callable = gray_message,
    **kwargs,
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
    :param message_mapping: Describes message
        m: [num_cols] -> F_2^k

    :param kwargs: _coding_LUT args & kwargs
    :return: eval_error, MC estimate of expected error.
    """
    num_bits = kwargs.get("num_bits", 10)
    use_complementary = kwargs.get("use_complementary")  # Optional arg

    # Generate BCH codes
    code_LUT = bch_code_LUT(bch_tuple, num_bits, message_mapping)

    # FAISS indexing
    if FAISS_GPUs:
        index = faiss_flat_gpu_index(packbits_strided(code_LUT))
    else:
        index = faiss_flat_index(packbits_strided(code_LUT))

    decoding_func = partial(
        minimum_distance_decoding,
        func=faiss_minimum_distance,
        index=index,
        pack=True,
    )

    return coding_LUT(
        phi_P,
        phi_A,
        t_exp,
        code_LUT=code_LUT,
        decoding_func=decoding_func,
        pbar_description=rf"{bch_tuple}"
        rf"{'-list' if bch_tuple.is_list_decoding else ''}"
        rf"{'-comp' if use_complementary else ''}: F_2^{bch_tuple.k} \to F_2^{bch_tuple.n}",
        **kwargs,
    )


def repetition_coding(
    phi_P: NDArray[float],
    phi_A: NDArray[float],
    t_exp: Union[float, NDArray[float]],
    repetition_tuple: metaclass.Repetition,
    message_mapping: Callable = gray_message,
    **kwargs,
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
    :param message_mapping: Describes message
        m: [num_cols] -> F_2^k

    :param kwargs: _coding_LUT args & kwargs
    :return: eval_error, MC estimate of expected error.
    """
    # Optional args
    num_bits = kwargs.get("num_bits", 10)
    use_complementary = kwargs.get("use_complementary")

    # Generate Repetition codes
    message_ll = message_mapping(num_bits)
    code_LUT = repetition_code_LUT(repetition_tuple, num_bits, message_mapping)

    decoding_func = partial(
        repetition_decoding,
        num_repeat=repetition_tuple.repeat,
        inverse_permuation=message_to_inverse_permuation(message_ll),
    )

    return coding_LUT(
        phi_P,
        phi_A,
        t_exp,
        code_LUT=code_LUT,
        decoding_func=decoding_func,
        pbar_description=rf"{repetition_tuple}"
        rf"{'-comp' if use_complementary else ''}: F_2^{repetition_tuple.k} \to F_2^{repetition_tuple.n}",
        **kwargs,
    )


def no_coding(
    phi_P: NDArray[float],
    phi_A: NDArray[float],
    t_exp: Union[float, NDArray[float]],
    message_mapping: Callable = gray_message,
    **kwargs,
) -> NDArray:
    """
    Evaluation without any coding strategy (monte carlo)

    :param phi_P: meshgrid of ambient + projector flux
    :param phi_A: meshgrid of ambient flux
    :param t_exp: exposure time

    :param message_mapping: Describes message
        m: [num_cols] -> F_2^k

    :param kwargs: _coding_LUT args & kwargs
    :return: eval_error, MC estimate of expected error.
    """
    num_bits = kwargs.get("num_bits", 10)

    # Generate BCH codes
    code_LUT = message_mapping(num_bits)

    decoding_func = partial(
        read_off_decoding, inverse_permuation=message_to_inverse_permuation(code_LUT)
    )

    return coding_LUT(
        phi_P,
        phi_A,
        t_exp,
        code_LUT=code_LUT,
        decoding_func=decoding_func,
        gt_permutation=message_to_permuation(code_LUT),
        pbar_description=rf"No Coding" rf": F_2^{num_bits} \to F_2^{num_bits}",
        **kwargs,
    )


if __name__ == "__main__":
    phi_proj = np.logspace(3, 5, num=64)
    phi_A = np.logspace(2, 4, num=64)

    phi_P_mesh, phi_A_mesh = np.meshgrid(phi_A + phi_proj, phi_A, indexing="ij")

    # DMD framerate
    # 0.1 millisecond or 10^4 FPS
    t_exp = 1e-4

    eval_error = bch_coding(
        phi_P_mesh,
        phi_A_mesh,
        t_exp,
        bch_tuple=metaclass.BCH(31, 11, 5),
        error_metric=squared_error,
    )
    mesh_plot_2d(eval_error, phi_proj, phi_A, error_metric=squared_error)

    logger.info("Repetition")
    eval_error = repetition_coding(
        phi_P_mesh,
        phi_A_mesh,
        t_exp,
        repetition_tuple=metaclass.Repetition(30, 10, 1),
        error_metric=squared_error,
    )

    mesh_plot_2d(eval_error, phi_proj, phi_A, error_metric=squared_error)

    eval_error = no_coding(
        phi_P_mesh,
        phi_A_mesh,
        t_exp,
        error_metric=squared_error,
    )

    mesh_plot_2d(eval_error, phi_proj, phi_A, error_metric=squared_error)
