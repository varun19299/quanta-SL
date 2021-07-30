"""
Exposes API for decoding based methods
"""
from types import ModuleType
from typing import Callable

import numpy as np
from einops import reduce

from quanta_SL.ops.binary import packbits, packbits_strided, unpackbits_strided


def repetition_decoding(
    queries,
    code_LUT,
    num_repeat: int = 1,
    pack: bool = False,
    unpack: bool = False,
    xp: ModuleType = np,
    **unused_kwargs,
):
    """
    Decoding a repetition code-word

    :param queries:  (N, n) sized binary matrix
    :param code_LUT: Not used, kept for consistency across funcs.
    :param num_repeat: repetitions per bit
    :param pack: Pack bits into bytes for memory efficiency
    :param unpack: Unpack bytes into bits if a certain algorithm needs.
    :param xp: Numpy or Cupy
    :return: Decoded indices. May have to permute if encoded message was Gray, antipodal etc. (non-binary code).
    """
    if pack:
        queries = packbits_strided(queries, xp=xp)
    elif unpack:
        queries = unpackbits_strided(queries, xp=xp)

    queries = reduce(queries, "N (c repeat) -> N c", "sum", repeat=num_repeat)
    queries = queries > 0.5 * num_repeat
    indices = packbits(queries.astype(int))

    return indices


def minimum_distance_decoding(
    queries,
    code_LUT,
    func: Callable,
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
    :param pack: Pack bits into bytes for memory efficiency
    :param unpack: Unpack bytes into bits if a certain algorithm needs.
    :param xp: Numpy or Cupy
    :return: Decoded indices. May have to permute if encoded message was Gray, antipodal etc. (non-binary code).
    """
    if pack:
        queries = packbits_strided(queries, xp=xp)
        code_LUT = packbits_strided(code_LUT, xp=xp)

    elif unpack:
        queries = unpackbits_strided(queries, xp=xp)
        code_LUT = unpackbits_strided(code_LUT, xp=xp)

    indices = func(queries, code_LUT, **func_kwargs)
    return indices

