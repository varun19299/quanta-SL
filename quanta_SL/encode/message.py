"""
Base projector patterns or messages
"""
import json
from pathlib import Path

import graycode
import numpy as np

from quanta_SL.encode.graycodes import monotonic_graycode
from quanta_SL.utils.plotting import plot_code_LUT
from quanta_SL.utils.loader import load_code
from quanta_SL.ops.binary import packbits, unpackbits, invert_permutation
from quanta_SL.ops.coding import stripe_width_stats
from quanta_SL.utils.decorators import named_func

long_run_gray_cache_dict = {}

# Map func names, named func names, long names to dict
registry = {}


def register(func):
    registry[func.__name__] = func

    if hasattr(func, "name"):
        registry[func.name] = func

    return func


@register
@named_func("Binary Code")
def binary_message(num_bits: int):
    return unpackbits(np.arange(pow(2, num_bits))).astype(int)


@register
@named_func("Gray Code")
def gray_message(num_bits: int):
    binary_ll = binary_message(num_bits)
    graycode_indices = graycode.gen_gray_codes(num_bits)

    return binary_ll[graycode_indices, :]


@register
@named_func("max-minSW Code")
def max_minSW_message(num_bits: int):
    assert num_bits == 10, "Max Min code only designed for 10 bits"

    return load_code("MaxMinSWGray")


@register
@named_func("XOR2 Code")
def xor2_message(num_bits: int):
    assert num_bits == 10, "XOR2 code only designed for 10 bits"

    return load_code("XOR02")


@register
@named_func("XOR4 Code")
def xor4_message(num_bits: int):
    assert num_bits == 10, "XOR4 code only designed for 10 bits"

    return load_code("XOR04")


@register
@named_func("Long-run Gray Code")
def long_run_gray_message(num_bits: int):
    from pkg_resources import resource_filename

    global long_run_gray_cache_dict
    if not long_run_gray_cache_dict:
        filepath = resource_filename(
            "quanta_SL", "encode/graycodes/long_run_graycode.json"
        )

        with open(filepath) as f:
            long_run_gray_cache_dict = json.load(f)["codes"]

    long_run_gray_cache_dict = {int(k): v for k, v in long_run_gray_cache_dict.items()}

    assert (
        num_bits in long_run_gray_cache_dict
    ), f"Balanced gray codes not evaluated for {num_bits} bits."
    mapping = np.array(long_run_gray_cache_dict[num_bits], dtype=int)

    return unpackbits(mapping)


@register
@named_func("Monotonic Gray Code")
def monotonic_gray_message(num_bits: int):
    code_LUT = []
    for code in monotonic_graycode.monotonic(num_bits):
        code_LUT.append(code)

    return np.array(code_LUT)


def message_to_permuation(message_ll, **packbits_kwargs):
    """
    Convert message LUT to packed integers.
    Useful to define a mapping between binary and the code.

    :param message_ll: message LUT
    :return: integer array
    """
    return packbits(message_ll, **packbits_kwargs)


def message_to_inverse_permuation(message_ll, **packbits_kwargs):
    """
    Convert message LUT inverse to packed integers.
    Useful to define a mapping between code and binary listing (np.arange).

    :param message_ll: message LUT
    :return: integer array
    """
    return invert_permutation(message_to_permuation(message_ll, **packbits_kwargs))


def _save_code_img():
    path = Path("outputs/code_images/message")
    plot_kwargs = dict(show=False, savefig=True)

    num_bits = 10
    for mapping in [
        gray_message,
        xor4_message,
        xor2_message,
        long_run_gray_message,
        monotonic_gray_message,
        max_minSW_message,
    ]:
        print(mapping.__name__)
        code_LUT = mapping(num_bits)

        fname = mapping.__name__.replace("_message", "")
        fname = f"{fname}[{num_bits} bits].png"
        plot_code_LUT(
            code_LUT,
            fname=path / fname,
            **plot_kwargs,
        )
        print(stripe_width_stats(code_LUT))
        print("\n")


if __name__ == "__main__":
    _save_code_img()
