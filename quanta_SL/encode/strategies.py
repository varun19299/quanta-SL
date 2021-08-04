"""
code LUT for various strategies
"""

from typing import Callable

import galois
import numpy as np
from einops import repeat
from nptyping import NDArray

from quanta_SL.encode import metaclass
from quanta_SL.encode.message import gray_message
from quanta_SL.encode.precision_bits import circular_shifted_stripes


def repetition_code_LUT(
    repetition_tuple: metaclass.Repetition,
    message_bits: int = 0,
    message_mapping: Callable = gray_message,
) -> NDArray[int]:
    """
    Repetition code LUT.

    :param repetition_tuple: Repetition code [n,k] parameters
    :param message_bits: message dimension
    :param message_mapping: Describes message
        m: [2^message_bits] -> F_2^message_bits

    :return Code Look-Up Table
    """
    if not message_bits:
        message_bits = repetition_tuple.k
    assert (
        message_bits == repetition_tuple.k
    ), f"Cannot encode {message_bits} bits must equal {repetition_tuple.k}."

    message_ll = message_mapping(message_bits)

    # Generate repetition codes
    # Repeats consecutive frames
    code_LUT = repetition_tuple.encode(message_ll)

    return code_LUT


def bch_code_LUT(
    bch_tuple: metaclass.BCH,
    message_bits: int,
    message_mapping: Callable = gray_message,
) -> NDArray[int]:
    """
    BCH code LUT.

    :param bch_tuple: BCH code [n,k,t] parameters
    :param message_bits: message dimension
    :param message_mapping: Describes message
        m: [2^message_bits] -> F_2^message_bits

    :return Code Look-Up Table
    """
    assert (
        message_bits <= bch_tuple.k
    ), f"Cannot encode {message_bits} bits, exceeds {bch_tuple.k}."
    message_ll = message_mapping(message_bits)

    # Generate BCH codes
    message_ll = galois.GF2(message_ll)
    code_LUT = bch_tuple.encode(message_ll)

    return code_LUT


def hybrid_code_LUT(
    bch_tuple: metaclass.BCH,
    bch_bits: int,
    message_bits: int,
    overlap_bits: int = 1,
    message_mapping: Callable = gray_message,
):
    """
    Hybrid code LUT (BCH + stripe scanning).

    :param bch_tuple: BCH code [n,k,t] parameters
    :param bch_bits: message bits (from MSB) encoded by BCH

    :param message_bits: message dimension
    :param overlap_bits: message bits encoded by both BCH and stripe

    :param message_mapping: Describes message
        m: [2^message_bits] -> F_2^message_bits

    :return Code Look-Up Table
    """
    assert (
        bch_bits < message_bits
    ), f"Bits coded by BCH ({bch_bits} bits) must be lesser than {message_bits} bits."

    stripe_bits = message_bits - bch_bits + overlap_bits

    # message mapping used only for BCH
    code_LUT = bch_code_LUT(bch_tuple, bch_bits, message_mapping)

    # Precision bits
    stripe_LUT = circular_shifted_stripes(pow(2, stripe_bits - 1))
    stripe_LUT = repeat(
        stripe_LUT, "N c -> (repeat N) c", repeat=pow(2, message_bits - stripe_bits)
    )

    # Concatenate along time axis
    code_LUT = np.concatenate([code_LUT, stripe_LUT], axis=-1)

    return code_LUT
