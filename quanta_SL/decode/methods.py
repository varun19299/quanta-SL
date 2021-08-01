"""
Exposes API for decoding based methods
"""
from types import ModuleType
from typing import Callable

import numpy as np
from einops import reduce
from nptyping import NDArray
from quanta_SL.ops.binary import packbits, packbits_strided, unpackbits_strided


def repetition_decoding(
    queries,
    code_LUT,
    num_repeat: int = 1,
    inverse_permuation: NDArray[int] = None,
    pack: bool = False,
    unpack: bool = False,
    xp: ModuleType = np,
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
    :param xp: Numpy or Cupy
    :return: Decoded indices.
    """
    if pack:
        queries = packbits_strided(queries, xp=xp)
    elif unpack:
        queries = unpackbits_strided(queries, xp=xp)

    queries = reduce(queries, "N (c repeat) -> N c", "sum", repeat=num_repeat)
    queries = queries > 0.5 * num_repeat

    if xp != np:
        queries = queries.get()
    indices = packbits(queries.astype(int))

    # Inverse mapping from permuted message to binary
    if isinstance(inverse_permuation, xp.ndarray):
        indices = inverse_permuation[indices]

    return indices


def minimum_distance_decoding(
    queries,
    code_LUT,
    func: Callable,
    inverse_permuation: NDArray[int] = None,
    pack: bool = False,
    unpack: bool = False,
    xp: ModuleType = np,
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
    :param xp: Numpy or Cupy
    :return: Decoded indices.
    """
    if pack:
        queries = packbits_strided(queries, xp=xp)
        code_LUT = packbits_strided(code_LUT, xp=xp)

    elif unpack:
        queries = unpackbits_strided(queries, xp=xp)
        code_LUT = unpackbits_strided(code_LUT, xp=xp)

    indices = func(queries, code_LUT, **func_kwargs)

    # Inverse mapping from permuted message to binary
    if isinstance(inverse_permuation, xp.ndarray):
        indices = inverse_permuation[indices]

    return indices


def read_off_decoding(queries, code_LUT, unpack: bool = False):
    """
    Decoding by simply reading off binary values.
    No coding scheme used.

    :param queries:  (N, n) sized binary matrix
    :param code_LUT: Not used, kept for consistency across funcs.
    :return: Decoded indices.
    """
    if unpack:
        # Undo byte conversion
        queries = unpackbits_strided(queries)

    # Pack into projector cols
    return packbits(queries)
