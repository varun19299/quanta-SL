"""
Exposes API for decoding based method
"""
from typing import Callable

from math import floor
import numpy as np
from einops import reduce
from nptyping import NDArray
from numba import vectorize, float64, int64

from quanta_SL.encode import metaclass
from quanta_SL.ops.binary import (
    packbits,
    packbits_strided,
    unpackbits_strided,
)
from quanta_SL.encode.message import (
    message_to_inverse_permuation,
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


@vectorize(
    [float64(int64, float64, int64, int64, int64)],
    cache=True,
    fastmath=True,
    target="parallel",
)
def merge_bch_stripe_indices(
    bch_index, stripe_index, code_bits, bch_code_bits, overlap_bits
):
    # Twice the stripe width
    stripe_period = code_bits - bch_code_bits
    stripe_bits = floor(np.log2(stripe_period)) - overlap_bits
    stripe_width = stripe_period // 2

    if bch_index % pow(2, overlap_bits) == 0:
        # Circular wrap error can occur
        if stripe_index > stripe_width:
            # Wrap around
            unwrapped_index = stripe_index - stripe_period
        else:
            unwrapped_index = stripe_index
    elif bch_index % pow(2, overlap_bits) == pow(2, overlap_bits) - 1:
        # Circular wrap error can occur
        if stripe_index < stripe_width:
            # Wrap around
            unwrapped_index = stripe_index + stripe_period - 1
            unwrapped_index = unwrapped_index % stripe_width
        else:
            unwrapped_index = stripe_index % stripe_width
    else:
        unwrapped_index = stripe_index % stripe_width

    bch_index = bch_index << stripe_bits
    index = bch_index + unwrapped_index

    return index


def hybrid_decoding(
    queries,
    code_LUT,
    func: Callable,
    bch_tuple: metaclass.BCH,
    bch_message_bits: int,
    overlap_bits: int = 1,
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

    :param pack: Pack bits into bytes for memory efficiency
    :param unpack: Unpack bytes into bits if a certain algorithm needs.

    :param func_kwargs: Additional kwargs to MLE
    :return: Decoded indices.
    """
    # First bits are BCH
    # Account for puncturing (if any)
    bch_code_bits = bch_tuple.n - (bch_tuple.k - bch_message_bits)
    bch_indices = minimum_distance_decoding(
        queries[:, :bch_code_bits],
        code_LUT[:, :bch_code_bits],
        func,
        pack,
        unpack,
        **func_kwargs,
    )

    # Followed by stripe scan
    # No inverse perm for stripe scan
    stripe_width = (queries.shape[1] - bch_code_bits) // 2
    stripe_indices = phase_decoding(
        queries[:, bch_code_bits:],
        code_LUT[:, bch_code_bits:],
        stripe_width,
        pack,
        unpack,
    )

    # Merge BCH and stripe
    code_bits = code_LUT.shape[1]

    indices = merge_bch_stripe_indices(
        bch_indices, stripe_indices, code_bits, bch_code_bits, overlap_bits
    )

    return indices


def gray_stripe_decoding(
    queries,
    code_LUT,
    gray_message_bits: int,
    overlap_bits: int = 1,
    pack: bool = False,
    unpack: bool = False,
):
    """
    Decoding (Gray + Stripe) patterns

    :param queries:  (N, n) sized binary matrix
    :param code_LUT: Not used, kept for consistency across funcs.

    :param gray_message_bits: Gray message dims
    :param overlap_bits: Bits that BCH and Stripe encode

    :param pack: Pack bits into bytes for memory efficiency
    :param unpack: Unpack bytes into bits if a certain algorithm needs.

    :return: Decoded indices.
    """
    # First bits are BCH
    # Account for puncturing (if any)
    gray_code_LUT = code_LUT[:, :gray_message_bits]
    code_bits = code_LUT.shape[1]

    stripe_width = (code_bits - gray_message_bits) // 2
    stripe_bits = int(np.log2(stripe_width)) + 1

    message_bits = gray_message_bits + stripe_bits - overlap_bits

    inverse_permuation = message_to_inverse_permuation(
        gray_code_LUT[:: pow(2, message_bits - gray_message_bits), :]
    )

    bch_indices = read_off_decoding(
        queries[:, :gray_message_bits],
        gray_code_LUT,
        inverse_permuation,
        pack,
        unpack,
    )

    # Followed by stripe scan
    # No inverse perm for stripe scan
    stripe_indices = phase_decoding(
        queries[:, gray_message_bits:],
        code_LUT[:, gray_message_bits:],
        stripe_width,
        pack,
        unpack,
    )

    # Merge BCH and stripe
    indices = merge_bch_stripe_indices(
        bch_indices, stripe_indices, code_bits, gray_message_bits, overlap_bits
    )

    return indices


if __name__ == "__main__":
    from quanta_SL.encode.strategies import gray_stripe_code_LUT

    code_LUT = gray_stripe_code_LUT(8, 11)
    decoded_LUT = gray_stripe_decoding(code_LUT, code_LUT, 8)
