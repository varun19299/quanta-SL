"""
Mapping functions
"""
from pathlib import Path

import cv2
import graycode
import numpy as np
from einops import repeat
from matplotlib import pyplot as plt
from tqdm import tqdm
import json

from utils.array_ops import unpackbits
from utils.monotonic_graycode import monotonic

mapping_dict = {}


def binary_mapping(num_bits: int):
    return unpackbits(np.arange(pow(2, num_bits)))


def gray_mapping(num_bits: int):
    binary_ll = unpackbits(np.arange(pow(2, num_bits)))
    graycode_indices = graycode.gen_gray_codes(num_bits)

    return binary_ll[graycode_indices, :]


def max_min_SW_mapping(num_bits: int):
    assert num_bits == 10, "Max Min code only designed for 10 bits"

    return load_code("MaxMinSWGray")


def xor2_mapping(num_bits: int):
    assert num_bits == 10, "XOR2 code only designed for 10 bits"

    return load_code("XOR02")


def xor4_mapping(num_bits: int):
    assert num_bits == 10, "XOR4 code only designed for 10 bits"

    return load_code("XOR04")


def long_run_gray_mapping(num_bits: int):
    global mapping_dict
    if not mapping_dict:
        with open("utils/long_run_graycode.json") as f:
            mapping_dict = json.load(f)["codes"]

    mapping_dict = {int(k): v for k, v in mapping_dict.items()}

    assert (
        num_bits in mapping_dict
    ), f"Balanced gray codes not evaluated for {num_bits} bits."
    mapping = np.array(mapping_dict[num_bits], dtype=int)

    return unpackbits(mapping)


def monotonic_mapping(num_bits: int):
    code_LUT = []
    for code in monotonic(num_bits):
        code_LUT.append(code)

    return np.array(code_LUT)


def load_code(
    pattern_name: str = "MaxMinSWGray", ignore_complement: bool = True
) -> np.ndarray:
    """
    Load code from projector patterns

    :param pattern_name: Folder in data/patterns
    :param ignore_complement: Skip even frames
    :return: Code Look-Up-Table
    """
    path = Path("data/patterns") / pattern_name

    frame_ll = []
    path_ll = list(path.glob("*.exr"))
    for e, file in enumerate(tqdm(path_ll, desc=f"Loading {path}")):
        if ignore_complement and (e % 2 == 0):
            continue
        img = cv2.imread(str(file), 0)
        frame_ll.append(img)

    frame_ll = np.stack(frame_ll, axis=-1)

    # Ignore height
    code_LUT = frame_ll[0]
    return code_LUT


def plot_code_LUT(
    code_LUT: np.ndarray, show: bool = True, aspect_ratio: float = 3.0, **kwargs
):
    """
    Image illustrating coding scheme
    :param code_LUT: Code Look-Up-Table
    """
    h, c = code_LUT.shape

    num_repeat =  kwargs.get("num_repeat", int(h / c / aspect_ratio))

    code_img = repeat(code_LUT, "h c -> (c repeat) h", repeat=num_repeat)

    if kwargs.get("savefig") or kwargs.get("fname"):
        assert kwargs.get("fname")
        cv2.imwrite(str(kwargs["fname"]), code_img * 255)

    if show:
        plt.imshow(code_img, cmap="gray")
        plt.show()


def _save_code_img():
    path = Path("outputs/projector_frames/code_images")
    plot_kwargs = dict(show=False, savefig=True)

    num_bits = 10
    for mapping in [
        gray_mapping,
        xor4_mapping,
        xor2_mapping,
        long_run_gray_mapping,
        max_min_SW_mapping,
    ]:
        code_LUT = mapping(num_bits)
        plot_code_LUT(
            code_LUT,
            fname=path / f"{mapping.__name__}-{num_bits}_bits.png",
            **plot_kwargs,
        )
        print(min_stripe_width(code_LUT))


if __name__ == "__main__":
    from utils.array_ops import min_stripe_width

    code_LUT = long_run_gray_mapping(8)
    m = min_stripe_width(code_LUT)
