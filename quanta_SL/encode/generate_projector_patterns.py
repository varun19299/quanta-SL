from math import ceil, log2
from pathlib import Path
from typing import Tuple, Callable

import cv2
import numpy as np
from einops import repeat, rearrange
from loguru import logger
from matplotlib import pyplot as plt
from nptyping import NDArray
from tqdm import tqdm

from quanta_SL.encode import metaclass
from quanta_SL.encode.message import gray_message, long_run_gray_message
from quanta_SL.encode.strategies import (
    repetition_code_LUT,
    bch_code_LUT,
    hybrid_code_LUT,
)
from quanta_SL.utils.plotting import plot_code_LUT

"""
Strategy to Projector frames
"""


def code_LUT_to_projector_frames(
    code_LUT: NDArray[int],
    projector_resolution: Tuple[int, int],
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
        logger.info(f"Saving / showing strategy {folder_name}")

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
    projector_resolution: Tuple[int, int],
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
        folder_name = f"Gray Code [{num_bits} bits]"

    if use_complementary:
        folder_name += " comp"

    kwargs = locals().copy()

    code_LUT = gray_message(num_bits)

    # Generate gray code mapping on projector resolution
    code_LUT_to_projector_frames(code_LUT, **kwargs)

def long_run_gray_code_to_projector_frames(
    projector_resolution: Tuple[int, int],
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    folder_name: str = "",
):
    """
    Long Run Gray codes to projector frames

    :param projector_resolution: width x height
    :param show: Plot in pyplot
    :param save: Save as png
    :param folder_name: Folder name to save to
    """
    # Find bits required to represent columns
    width, height = projector_resolution
    num_bits = ceil(log2(width))

    if not folder_name:
        folder_name = f"Long Run Gray Code [{num_bits} bits]"

    if use_complementary:
        folder_name += " comp"

    kwargs = locals().copy()

    code_LUT = long_run_gray_message(num_bits)

    # Generate gray code mapping on projector resolution
    code_LUT_to_projector_frames(code_LUT, **kwargs)

def bch_to_projector_frames(
    bch_tuple: metaclass.BCH,
    projector_resolution: Tuple[int, int],
    message_mapping: Callable = gray_message,
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    folder_name: str = "",
):
    """
    Convert BCH codes to projector frames.

    :param bch_tuple: BCH code [n,k,t] parameters
    :param projector_resolution: width x height
    :param message_mapping: Describes message
        m: [num_cols] -> F_2^k
    :param use_complementary: whether to save / show complementary (1-frame_i)
    :param show: Plot in pyplot
    :param save: Save as png
    :param folder_name: Folder name to save to
    """
    if not folder_name:
        folder_name = f"{bch_tuple} [{message_mapping.__name__}]"

    if use_complementary:
        folder_name += " comp"

    kwargs = locals().copy()

    # Find bits required to represent columns
    width, height = projector_resolution
    message_bits = ceil(log2(width))

    code_LUT = bch_code_LUT(bch_tuple, message_bits, message_mapping)

    if save:
        path = Path("outputs/code_images/bch")
        plot_code_LUT(
            code_LUT,
            show,
            fname=path / f"{folder_name}.png",
        )

    code_LUT_to_projector_frames(code_LUT=code_LUT, **kwargs)


def repetition_to_projector_frames(
    repetition_tuple: metaclass.Repetition,
    projector_resolution: Tuple[int, int],
    message_mapping: Callable = gray_message,
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    folder_name: str = "",
):
    """
    Convert Repetition codes to projector frames.

    :param repetition_tuple: Repetition code [n,k] parameters
    :param projector_resolution: width x height
    :param message_mapping: Describes message
        m: [num_cols] -> F_2^k
    :param use_complementary: whether to save / show complementary (1-frame_i)
    :param show: Plot in pyplot
    :param save: Save as png
    :param folder_name: Folder name to save to
    """
    if not folder_name:
        folder_name = f"{repetition_tuple} [{message_mapping.__name__}]"

    if use_complementary:
        folder_name += " comp"

    kwargs = locals().copy()

    # Find bits required to represent columns
    width, height = projector_resolution
    message_bits = ceil(log2(width))

    code_LUT = repetition_code_LUT(repetition_tuple, message_bits, message_mapping)

    if save:
        path = Path("outputs/code_images/repetition")
        plot_code_LUT(
            code_LUT,
            show,
            fname=path / f"{folder_name}.png",
        )

    code_LUT_to_projector_frames(code_LUT=code_LUT, **kwargs)


def hybrid_to_projector_frames(
    bch_tuple: metaclass.BCH,
    bch_message_bits: int,
    projector_resolution: Tuple[int, int],
    overlap_bits: int = 1,
    message_mapping: Callable = gray_message,
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    folder_name: str = "",
):
    """
    Generate projector frames for "Hybrid" (BCH + stripe scan) strategy

    :param bch_tuple: BCH code [n,k,t] parameters
    :param bch_message_bits: message bits (from MSB) encoded by BCH

    :param projector_resolution: width x height
    :param overlap_bits: message bits encoded by both BCH and stripe
    :param message_mapping: Describes message
        m: [num_cols] -> F_2^k

    :param use_complementary: whether to save / show complementary (1-frame_i)
    :param show: Plot in pyplot
    :param save: Save as png
    :param folder_name: Folder name to save to
    """
    if not folder_name:
        folder_name = f"Hybrid {bch_tuple} [{message_mapping.__name__}]"

    if use_complementary:
        folder_name += " comp"

    kwargs = locals().copy()

    # Find bits required to represent columns
    width, height = projector_resolution
    message_bits = ceil(log2(width))

    code_LUT = hybrid_code_LUT(
        bch_tuple,
        bch_message_bits,
        message_bits,
        overlap_bits=1,
        message_mapping=message_mapping,
    )

    if save:
        path = Path("outputs/code_images/hybrid")
        plot_code_LUT(
            code_LUT,
            show,
            fname=path / f"{folder_name}.png",
        )

    code_LUT_to_projector_frames(code_LUT=code_LUT, **kwargs)


if __name__ == "__main__":
    num_bits = 11
    projector_resolution = (1920, 1080)

    kwargs = {"show": False, "save": True, "use_complementary": True}

    # gray_code_to_projector_frames(projector_resolution, **kwargs)

    long_run_gray_code_to_projector_frames(projector_resolution, **kwargs)

    # hybrid_to_projector_frames(
    #     metaclass.BCH(63, 10, 13),
    #     bch_message_bits=8,
    #     projector_resolution=projector_resolution,
    #     message_mapping=gray_message,
    #     **kwargs,
    # )

    # bch_to_projector_frames(
    #     metaclass.BCH(63, 16, 11),
    #     projector_resolution,
    #     message_mapping=gray_message,
    #     **kwargs,
    # )

    # repetition_to_projector_frames(
    #     metaclass.Repetition(77, 11, 3),
    #     projector_resolution,
    #     message_mapping=long_run_gray_message,
    #     **kwargs,
    # )
