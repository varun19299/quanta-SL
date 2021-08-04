"""
Exposes API for decoding based methods
"""
from typing import Callable

import numpy as np
from einops import reduce
from nptyping import NDArray

from quanta_SL.ops.binary import (
    packbits,
    packbits_strided,
    unpackbits_strided,
    unpackbits,
)


def _pack_or_unpack(
    queries,
    code_LUT,
    pack: bool = False,
    unpack: bool = False,
):
    if pack:
        queries = packbits_strided(queries)
        code_LUT = packbits_strided(code_LUT)

    elif unpack:
        queries = unpackbits_strided(queries)
        code_LUT = unpackbits_strided(code_LUT)

    return queries, code_LUT


def repetition_decoding(
    queries,
    code_LUT,
    num_repeat: int,
    inverse_permuation: NDArray[int] = None,
    pack: bool = False,
    unpack: bool = False,
):
    """
    Decoding a repetition code-word

    :param queries:  (N, n) sized binary matrix
    :param code_LUT: Not used, kept for consistency across funcs.
    :param num_repeat: repetitions per bit

    :param inverse_permuation: Mapping from message int to binary
        Useful when evaluating strategies with MSE / MAE (where locality matters).

    :param pack: Pack bits into bytes for memory efficiency
    :param unpack: Unpack bytes into bits if a certain algorithm needs.
    :return: Decoded indices.
    """
    # Pack or Unpack bits appropriately
    queries, code_LUT = _pack_or_unpack(queries, code_LUT, pack, unpack)

    queries = reduce(queries, "N (c repeat) -> N c", "sum", repeat=num_repeat)
    queries = queries > 0.5 * num_repeat
    indices = packbits(queries.astype(int))

    # Inverse mapping from permuted message to binary
    if isinstance(inverse_permuation, np.ndarray):
        indices = inverse_permuation[indices]

    return indices


def minimum_distance_decoding(
    queries,
    code_LUT,
    func: Callable,
    pack: bool = False,
    unpack: bool = False,
    **func_kwargs,
):
    """
    Decoding via MLE

    :param queries:  (N, n) sized binary matrix
    :param code_LUT: Not used, kept for consistency across funcs.
    :param func: Minimum Distance implementation

    :param inverse_permuation: Mapping from message int to binary
        Useful when evaluating strategies with MSE / MAE (where locality matters).

    :param pack: Pack bits into bytes for memory efficiency
    :param unpack: Unpack bytes into bits if a certain algorithm needs.
    :param func_kwargs: Additional kwargs to MLE
    :return: Decoded indices.
    """
    # Pack or Unpack bits appropriately
    queries, code_LUT = _pack_or_unpack(queries, code_LUT, pack, unpack)

    indices = func(queries, code_LUT, **func_kwargs)

    # inverse_permutation not needed
    # since code_LUT is a mapping from binary to codes
    # MLE returns this index

    return indices


def read_off_decoding(
    queries,
    code_LUT,
    inverse_permuation: NDArray[int] = None,
    pack: bool = False,
    unpack: bool = False,
):
    """
    Decoding by simply reading off binary values.
    No coding scheme used.

    :param queries:  (N, n) sized binary matrix
    :param code_LUT: Not used, kept for consistency across funcs.

    :param inverse_permuation: Mapping from message int to binary
        Useful when evaluating strategies with MSE / MAE (where locality matters).

    :param pack: Pack bits into bytes for memory efficiency
    :param unpack: Unpack bytes into bits if a certain algorithm needs.
    :return: Decoded indices.
    """
    # Pack or Unpack bits appropriately
    queries, code_LUT = _pack_or_unpack(queries, code_LUT, pack, unpack)

    # Pack into projector cols
    indices = packbits(queries)

    # Inverse mapping from permuted message to binary
    if isinstance(inverse_permuation, np.ndarray):
        indices = inverse_permuation[indices]

    return indices


def square_wave_phase_unwrap(
    phase: NDArray[float], stripe_width: int = 8
) -> NDArray[float]:
    """
    Phase unwrap function for square wave
    :param phase: In radians
    :param stripe_width: width of 1's / 0's
        Periodicity is 2 * stripe_width    :return:
    """
    alpha = phase[..., 1] * stripe_width / np.pi

    # Phase of first spike / discrete delta
    offset = -(stripe_width + 1) / 2
    return (offset - alpha) % (2 * stripe_width)


def phase_decoding(
    queries,
    code_LUT,
    stripe_width: int = 8,
    inverse_permuation: NDArray[int] = None,
    pack: bool = False,
    unpack: bool = False,
):
    """
    Phase shift based decoding.
    For stripe scan.

    :param queries:  (N, n) sized binary matrix
    :param code_LUT: Not used, kept for consistency across funcs.
    :param stripe_width: width of 1's / 0's
        Periodicity is 2 * stripe_width

    :param inverse_permuation: Mapping from message int to binary
        Useful when evaluating strategies with MSE / MAE (where locality matters).

    :param pack: Pack bits into bytes for memory efficiency
    :param unpack: Unpack bytes into bits if a certain algorithm needs.
    :return: Decoded indices.
    :return:
    """
    # Pack or Unpack bits appropriately
    queries, code_LUT = _pack_or_unpack(queries, code_LUT, pack, unpack)

    queries_fft = np.fft.fft(queries, axis=-1)
    queries_phase = np.angle(queries_fft)

    # Phase unwrap
    indices = square_wave_phase_unwrap(queries_phase, stripe_width)

    # Inverse mapping from permuted message to binary
    if isinstance(inverse_permuation, np.ndarray):
        indices = inverse_permuation[indices]

    return indices


def hybrid_decoding(
    queries,
    code_LUT,
    func: Callable,
    bch_message_bits: int,
    overlap_bits: int = 1,
    inverse_permuation: NDArray[int] = None,
    pack: bool = False,
    unpack: bool = False,
    **func_kwargs,
):
    """
    Decoding Hybrid (BCH + Stripe) patterns

    :param queries:  (N, n) sized binary matrix
    :param code_LUT: Not used, kept for consistency across funcs.
    :param func: Minimum Distance implementation

    :param bch_message_bits: BCH message dims
    :param overlap_bits: Bits that BCH and Stripe encode

    :param inverse_permuation: Mapping from message int to binary
        Useful when evaluating strategies with MSE / MAE (where locality matters).

    :param pack: Pack bits into bytes for memory efficiency
    :param unpack: Unpack bytes into bits if a certain algorithm needs.

    :param func_kwargs: Additional kwargs to MLE
    :return: Decoded indices.
    """
    # First bits are BCH
    bch_indices = minimum_distance_decoding(
        queries[:, :bch_message_bits],
        code_LUT,
        func,
        inverse_permuation,
        pack,
        unpack,
        **func_kwargs,
    )

    # Followed by stripe scan
    # No inverse perm for stripe scan
    stripe_width = (queries.shape[1] - bch_message_bits) // 2
    stripe_indices = phase_decoding(
        queries[:, :bch_message_bits], code_LUT, stripe_width, pack, unpack
    )

    # Unpack both
    bch_indices = unpackbits(bch_indices)
    stripe_indices = unpackbits(stripe_indices)[:, overlap_bits:]
    indices = np.concatenate([bch_indices, stripe_indices], axis=-1)
    indices = packbits(indices)

    return indices
