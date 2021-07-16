import logging
from math import ceil, log2
from pathlib import Path
from typing import Tuple, Callable

import cv2
import galois
import numpy as np
from einops import repeat, rearrange
from galois import GF2
from matplotlib import pyplot as plt
from nptyping import NDArray
from tqdm import tqdm

from utils.array_ops import min_stripe_width, packbits
from utils.mapping import (
    gray_mapping,
    binary_mapping,
    max_min_SW_mapping,
    long_run_gray_mapping,
    xor2_mapping,
    xor4_mapping,
    monotonic_mapping,
    plot_code_LUT,
)
from utils.math_ops import fast_factorial
from vis_tools.strategies import metaclass
import itertools

FORMAT = "%(asctime)s [%(filename)s : %(funcName)2s() : %(lineno)2s] %(message)s"
logging.basicConfig(format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)


"""
Strategy to Projector frames
"""


def code_LUT_to_projector_frames(
    code_LUT: NDArray[int],
    projector_resolution: Tuple[int],
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    folder_name: str = "",
    **unused_kwargs,
):
    """
    Generate projector frames from an arbitrary code

    :param code_LUT: Describes F_2^k \to F_2^n code
    :param projector_resolution: width x height
    :param use_complementary: whether to save / show complementary (1-frame_i)
    :param show: Plot in pyplot
    :param save: Save as png
    :param folder_name: Folder name to save to
    """
    width, height = projector_resolution
    N, n = code_LUT.shape

    assert (
        N >= width
    ), f"Coding scheme has only {N} codes, not sufficient to cover {width} projector columns."

    # Crop to fit projector width
    code_LUT = code_LUT[(N - width) // 2 : (N + width) // 2]

    if use_complementary:
        code_ll_gray_comp = np.zeros((width, 2 * n))

        # Interleave
        # Can do more cleverly via np.ravel
        # but this is most readable
        code_ll_gray_comp[:, ::2] = code_LUT
        code_ll_gray_comp[:, 1::2] = 1 - code_LUT
        code_LUT = code_ll_gray_comp

    code_LUT = repeat(code_LUT, "width n -> height width n", height=height)

    if save:
        assert folder_name, f"No output folder provided"
        out_dir = Path("outputs/projector_frames") / folder_name
        out_dir.mkdir(exist_ok=True, parents=True)

    # Write out files
    if show or save:
        logging.info(f"Saving / showing strategy {folder_name}")

        pbar = tqdm(rearrange(code_LUT, "height width n -> n height width"))

        for e, frame in enumerate(pbar, start=1):
            pbar.set_description(f"Frame {e}")

            if show:
                plt.imshow(frame, cmap="gray")
                plt.title(f"Frame {e}")
                plt.show()

            if save:
                out_path = out_dir / f"frame-{e}.png"
                cv2.imwrite(str(out_path), frame * 255.0)


def gray_code_to_projector_frames(
    projector_resolution: Tuple[int],
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    folder_name: str = "",
):
    """
    Gray codes to projector frames

    :param projector_resolution: width x height
    :param show: Plot in pyplot
    :param save: Save as png
    :param folder_name: Folder name to save to
    """
    # Find bits required to represent columns
    width, height = projector_resolution
    num_bits = ceil(log2(width))

    if not folder_name:
        folder_name = f"Gray-Code-{num_bits}-bits"

    if use_complementary:
        folder_name += "-comp"

    kwargs = locals().copy()

    # Generate gray code mapping on projector resolution
    code_LUT_to_projector_frames(code_LUT=gray_mapping(num_bits), **kwargs)


def bch_to_projector_frames(
    bch_tuple: metaclass.BCH,
    projector_resolution: Tuple[int],
    message_mapping: Callable = gray_mapping,
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    folder_name: str = "",
):
    """
    Convert BCH codes to projector frames.

    :param bch_tuple: BCH code [n,k,t] parameters
    :param projector_resolution: width x height
    :param use_complementary: whether to save / show complementary (1-frame_i)
    :param show: Plot in pyplot
    :param save: Save as png
    :param folder_name: Folder name to save to
    """
    if not folder_name:
        folder_name = f"{bch_tuple}"

    if use_complementary:
        folder_name += "-comp"

    kwargs = locals().copy()

    # Find bits required to represent columns
    width, height = projector_resolution
    num_bits = ceil(log2(width))

    message_ll = message_mapping(num_bits)

    # BCH encoder
    bch = galois.BCH(bch_tuple.n, bch_tuple.k)
    binary_bch_codes = bch.encode(GF2(binary_mapping(num_bits)))
    binary_bch_codes = binary_bch_codes.view(np.ndarray).astype(int)

    # Try permuting
    pbar = tqdm(total=fast_factorial(10))

    most_acceptable_dict = {
        "num": 0,
        "perm": [],
        "min_stripe_ll": [],
        "mean_stripe_ll": [],
    }

    update_interval = 100

    try:
        for e, perm in enumerate(itertools.permutations(range(num_bits))):
            pbar.update(1)

            # Generate BCH_matlab codes
            perm_message = packbits(message_ll[:, perm])
            code_LUT = binary_bch_codes[perm_message, :]

            min_stripe, min_stripe_ll, mean_stripe_ll = min_stripe_width(code_LUT)

            acceptable = len([stripe for stripe in min_stripe_ll if stripe >= 4])

            if acceptable >= most_acceptable_dict["num"]:
                most_acceptable_dict.update(
                    {
                        "num": acceptable,
                        "perm": perm,
                        "min_stripe_ll": min_stripe_ll,
                        "mean_stripe_ll": mean_stripe_ll,
                    }
                )

            if e % update_interval == 0:
                pbar.set_description(
                    f"Min Stripe {min_stripe} | Max Acceptable so far {most_acceptable_dict['num']}"
                )

            if min_stripe > 1:
                break
    except KeyboardInterrupt:
        perm_message = packbits(message_ll[:, perm])
        acceptable_code_LUT = binary_bch_codes[perm_message, :]

        path = Path("outputs/projector_frames/code_images")
        plot_code_LUT(
            acceptable_code_LUT,
            show=False,
            fname=path / f"acceptable-{bch_tuple}-{message_mapping.__name__}.png",
        )

        breakpoint()

    # Generate BCH_matlab codes
    code_LUT = bch.encode(GF2(message_ll))
    code_LUT = code_LUT.view(np.ndarray).astype(int)

    if save:
        path = Path("outputs/projector_frames/code_images")
        plot_code_LUT(
            code_LUT, show=False, fname=path / f"{bch_tuple}-{message_mapping.__name__}.png"
        )

    code_LUT_to_projector_frames(code_LUT=code_LUT, **kwargs)


def repetition_to_projector_frames(
    repetition_tuple: metaclass.Repetition,
    projector_resolution: Tuple[int],
    message_mapping: Callable = gray_mapping,
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    folder_name: str = "",
):
    """
    Convert Repetition codes to projector frames.

    :param repetition_tuple: Repetition code [n,k] parameters
    :param projector_resolution: width x height
    :param use_complementary: whether to save / show complementary (1-frame_i)
    :param puncture: transmit only n - (k - log2(projector_cols))
    :param show: Plot in pyplot
    :param save: Save as png
    :param folder_name: Folder name to save to
    """
    if not folder_name:
        folder_name = f"{repetition_tuple}"

    if use_complementary:
        folder_name += "-comp"

    kwargs = locals().copy()

    # Find bits required to represent columns
    width, height = projector_resolution
    num_bits = ceil(log2(width))

    message_ll = message_mapping(num_bits)

    # Generate repetition codes
    # Repeats consecutive frames
    code_LUT = repeat(message_ll, "N k -> N (k repeat)", repeat=repetition_tuple.repeat)

    code_LUT_to_projector_frames(code_LUT=code_LUT, **kwargs)


if __name__ == "__main__":
    num_bits = 11
    projector_resolution = (1920, 1080)

    num_bits = 10
    projector_resolution = (1024, 1080)

    kwargs = {"show": False, "save": False}

    # gray_code_to_projector_frames(projector_resolution, **kwargs)
    bch_to_projector_frames(
        metaclass.BCH(31, 11, 5),
        projector_resolution,
        message_mapping=gray_mapping,
        **kwargs,
    )
    # bch_to_projector_frames(
    #     metaclass.BCH(31, 11, 5),
    #     projector_resolution,
    #     use_complementary=True,
    #     **kwargs,
    # )
    # repetition_to_projector_frames(
    #     metaclass.Repetition(66, 11, 2), projector_resolution, **kwargs
    # )
