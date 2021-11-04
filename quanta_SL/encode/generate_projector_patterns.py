from math import ceil, log2
from pathlib import Path
from typing import Callable, Union, Tuple

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
    gray_stripe_code_LUT,
)
from quanta_SL.utils.plotting import plot_code_LUT

"""
Strategy to Projector frames
"""


def code_LUT_to_projector_frames(
    code_LUT: NDArray[int],
    projector_resolution: Tuple[int, int],
    encoded_dim: Union[int, str] = "columns",
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    out_dir: str = "",
    include_all_white: bool = False,
    **unused_kwargs,
):
    """
    Generate projector frames from an arbitrary code

    :param code_LUT: Describes F_2^k \to F_2^n code
    :param projector_resolution: width x height

    :param encoded_dim: Whether to encode rows or columns
    :param use_complementary: whether to save / show complementary (1-frame_i)
    :param show: Plot in pyplot
    :param save: Save as png
    :param out_dir: Folder name to save to
    :param include_all_white: Include an all-white frame
    """
    assert encoded_dim in [0, 1, "columns", "rows"]
    width, height = projector_resolution
    N, n = code_LUT.shape

    if encoded_dim in [0, "columns"]:
        assert (
            N >= width
        ), f"Coding scheme has only {N} codes, not sufficient to cover {width} projector columns."

        encoded_len = width

    else:
        assert (
            N >= height
        ), f"Coding scheme has only {N} codes, not sufficient to cover {height} projector rows."

        encoded_len = height
        out_dir += " [rows]"

    # Crop to fit projector width / height
    code_LUT = code_LUT[(N - encoded_len) // 2 : (N + encoded_len) // 2]

    if use_complementary:
        code_ll_gray_comp = np.zeros((encoded_len, 2 * n))

        # Interleave
        # Can do more cleverly via np.ravel
        # but this is most readable
        code_ll_gray_comp[:, ::2] = code_LUT
        code_ll_gray_comp[:, 1::2] = 1 - code_LUT
        code_LUT = code_ll_gray_comp

    if include_all_white:
        all_white = np.ones((encoded_len, 1), dtype=int)
        code_LUT = np.concatenate((all_white, code_LUT), axis=1)

    if encoded_dim in [0, "columns"]:
        code_LUT = repeat(code_LUT, "width n -> height width n", height=height)
    else:
        code_LUT = repeat(code_LUT, "height n -> height width n", width=width)

    if save:
        assert out_dir, f"No output folder provided"
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

    # Write out files
    if show or save:
        logger.info(f"Saving / showing strategy {out_dir}")

        pbar = tqdm(rearrange(code_LUT, "height width n -> n height width"))

        start = 1 if not include_all_white else 0
        for e, frame in enumerate(pbar, start=start):
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
    encoded_dim: Union[int, str] = "columns",
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    out_dir: str = "",
    include_all_white: bool = False,
):
    """
    Gray codes to projector frames

    :param projector_resolution: width x height

    :param encoded_dim: Whether to encode rows or columns
    :param use_complementary: whether to save / show complementary (1-frame_i)

    :param show: Plot in pyplot
    :param save: Save as png
    :param out_dir: Folder to save to
    :param include_all_white: Include an all-white frame
    """
    # Find bits required to represent columns
    width, height = projector_resolution
    num_bits = ceil(log2(width))

    folder_name = f"Gray Code [{num_bits} bits]"
    if not out_dir:
        out_dir = f"outputs/projector_frames/lcd/{folder_name}"
    else:
        out_dir = f"{out_dir}/{folder_name}"

    if use_complementary:
        out_dir += " comp"

    kwargs = locals().copy()

    code_LUT = gray_message(num_bits)

    # Generate gray code mapping on projector resolution
    code_LUT_to_projector_frames(code_LUT, **kwargs)


def long_run_gray_code_to_projector_frames(
    projector_resolution: Tuple[int, int],
    encoded_dim: Union[int, str] = "columns",
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    out_dir: str = "",
    include_all_white: bool = False,
):
    """
    Long Run Gray codes to projector frames

    :param projector_resolution: width x height

    :param encoded_dim: Whether to encode rows or columns
    :param use_complementary: whether to save / show complementary (1-frame_i)

    :param show: Plot in pyplot
    :param save: Save as png
    :param out_dir: Folder name to save to
    :param include_all_white: Include an all-white frame
    """
    # Find bits required to represent columns
    width, height = projector_resolution
    num_bits = ceil(log2(width))

    folder_name = f"Long Run Gray Code [{num_bits} bits]"
    if not out_dir:
        out_dir = f"outputs/projector_frames/lcd/{folder_name}"
    else:
        out_dir = f"{out_dir}/{folder_name}"

    if use_complementary:
        out_dir += " comp"

    kwargs = locals().copy()

    code_LUT = long_run_gray_message(num_bits)

    # Generate gray code mapping on projector resolution
    code_LUT_to_projector_frames(code_LUT, **kwargs)


def bch_to_projector_frames(
    bch_tuple: metaclass.BCH,
    projector_resolution: Tuple[int, int],
    message_mapping: Callable = gray_message,
    encoded_dim: Union[int, str] = "columns",
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    out_dir: str = "",
    include_all_white: bool = False,
):
    """
    Convert BCH codes to projector frames.

    :param bch_tuple: BCH code [n,k,t] parameters
    :param projector_resolution: width x height
    :param message_mapping: Describes message
        m: [num_cols] -> F_2^k

    :param encoded_dim: Whether to encode rows or columns
    :param use_complementary: whether to save / show complementary (1-frame_i)
    :param show: Plot in pyplot
    :param save: Save as png
    :param out_dir: Folder name to save to
    :param include_all_white: Include an all-white frame
    """

    folder_name = f"{bch_tuple} [{message_mapping.__name__}]"
    if not out_dir:
        out_dir = f"outputs/projector_frames/lcd/{folder_name}"
    else:
        out_dir = f"{out_dir}/{folder_name}"

    if use_complementary:
        out_dir += " comp"

    kwargs = locals().copy()

    # Find bits required to represent columns
    width, height = projector_resolution
    message_bits = ceil(log2(width))

    code_LUT = bch_code_LUT(bch_tuple, message_bits, message_mapping)

    if save:
        path = Path(f"outputs/code_images/{projector_resolution}/bch")
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
    encoded_dim: Union[int, str] = "columns",
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    out_dir: str = "",
    include_all_white: bool = False,
):
    """
    Convert Repetition codes to projector frames.

    :param repetition_tuple: Repetition code [n,k] parameters
    :param projector_resolution: width x height
    :param message_mapping: Describes message
        m: [num_cols] -> F_2^k

    :param encoded_dim: Whether to encode rows or columns
    :param use_complementary: whether to save / show complementary (1-frame_i)
    :param show: Plot in pyplot
    :param save: Save as png
    :param out_dir: Folder name to save to
    :param include_all_white: Include an all-white frame
    """
    folder_name = f"{repetition_tuple} [{message_mapping.__name__}]"
    if not out_dir:
        out_dir = f"outputs/projector_frames/lcd/{folder_name}"
    else:
        out_dir = f"{out_dir}/{folder_name}"

    if use_complementary:
        out_dir += " comp"

    kwargs = locals().copy()

    # Find bits required to represent columns
    width, height = projector_resolution
    message_bits = ceil(log2(width))

    code_LUT = repetition_code_LUT(repetition_tuple, message_bits, message_mapping)

    if save:
        path = Path(f"outputs/code_images/{projector_resolution}/repetition")
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
    encoded_dim: Union[int, str] = "columns",
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    out_dir: str = "",
    include_all_white: bool = False,
):
    """
    Generate projector frames for "Hybrid" (BCH + stripe scan) strategy

    :param bch_tuple: BCH code [n,k,t] parameters
    :param bch_message_bits: message bits (from MSB) encoded by BCH

    :param projector_resolution: width x height
    :param overlap_bits: message bits encoded by both BCH and stripe
    :param message_mapping: Describes message
        m: [num_cols] -> F_2^k

    :param encoded_dim: Whether to encode rows or columns
    :param use_complementary: whether to save / show complementary (1-frame_i)
    :param show: Plot in pyplot
    :param save: Save as png
    :param out_dir: Folder name to save to
    :param include_all_white: Include an all-white frame
    """

    folder_name = f"Hybrid {bch_tuple} [{message_mapping.__name__}]"
    if not out_dir:
        out_dir = f"outputs/projector_frames/lcd/{folder_name}"
    else:
        out_dir = f"{out_dir}/{folder_name}"

    if use_complementary:
        out_dir += " comp"

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
        path = Path(f"outputs/code_images/{projector_resolution}/hybrid")
        plot_code_LUT(
            code_LUT,
            show,
            fname=path / f"{folder_name}.png",
        )

    code_LUT_to_projector_frames(code_LUT=code_LUT, **kwargs)


def gray_stripe_to_projector_frames(
    gray_message_bits: int,
    projector_resolution: Tuple[int, int],
    overlap_bits: int = 1,
    message_mapping: Callable = gray_message,
    encoded_dim: Union[int, str] = "columns",
    use_complementary: bool = False,
    show: bool = False,
    save: bool = False,
    out_dir: str = "",
    include_all_white: bool = False,
):
    """
    Generate projector frames for (Gray + stripe scan) strategy

    :param gray_message_bits: message bits (from MSB) encoded by BCH

    :param projector_resolution: width x height
    :param overlap_bits: message bits encoded by both BCH and stripe
    :param message_mapping: Describes message
        m: [num_cols] -> F_2^k

    :param encoded_dim: Whether to encode rows or columns
    :param use_complementary: whether to save / show complementary (1-frame_i)
    :param show: Plot in pyplot
    :param save: Save as png
    :param out_dir: Folder name to save to
    :param include_all_white: Include an all-white frame
    """
    # Find bits required to represent columns
    width, height = projector_resolution
    num_bits = ceil(log2(width))

    folder_name = f"GrayStripe [{num_bits} bits] [{message_mapping.__name__}]"
    if not out_dir:
        out_dir = f"outputs/projector_frames/lcd/{folder_name}"
    else:
        out_dir = f"{out_dir}/{folder_name}"

    if use_complementary:
        out_dir += " comp"

    kwargs = locals().copy()

    code_LUT = gray_stripe_code_LUT(
        gray_message_bits,
        num_bits,
        overlap_bits=1,
        message_mapping=message_mapping,
    )

    if save:
        path = Path(f"outputs/code_images/{projector_resolution}/graystripe")
        plot_code_LUT(
            code_LUT,
            show,
            fname=path / f"{folder_name}.png",
        )

    code_LUT_to_projector_frames(code_LUT=code_LUT, **kwargs)


def lcd_patterns():
    projector_resolution = (1920, 1080)

    kwargs = {
        "projector_resolution": projector_resolution,
        "show": False,
        "save": True,
        "use_complementary": True,
        "out_dir": "outputs/projector_frames/lcd",
    }

    # Gray stripe (for calibration)
    gray_stripe_to_projector_frames(8, **kwargs)
    gray_stripe_to_projector_frames(8, encoded_dim="rows", **kwargs)

    # No coding
    gray_code_to_projector_frames(**kwargs)

    long_run_gray_code_to_projector_frames(**kwargs)

    bch_tuple_ll = [
        metaclass.BCH(63, 16, 11),
        metaclass.BCH(127, 15, 27),
        metaclass.BCH(255, 13, 59),
    ]
    hybrid_bch_tuple_ll = [
        metaclass.BCH(63, 10, 13),
        metaclass.BCH(127, 8, 31),
        metaclass.BCH(255, 9, 63),
    ]
    repetition_tuple_ll = [
        metaclass.Repetition(77, 11, 3),
        metaclass.Repetition(143, 11, 6),
        metaclass.Repetition(275, 11, 12),
    ]
    bch_message_bits = 8

    # BCH, Hybrid, Repetition for various redundancies
    for bch_tuple, hybrid_bch_tuple, repetition_tuple in zip(
        bch_tuple_ll, hybrid_bch_tuple_ll, repetition_tuple_ll
    ):
        # BCH
        bch_to_projector_frames(
            bch_tuple,
            message_mapping=gray_message,
            **kwargs,
        )

        # Hybrid with gray
        hybrid_to_projector_frames(
            hybrid_bch_tuple,
            bch_message_bits=bch_message_bits,
            message_mapping=gray_message,
            **kwargs,
        )

        # Hybrid with long run
        hybrid_to_projector_frames(
            hybrid_bch_tuple,
            bch_message_bits=bch_message_bits,
            message_mapping=long_run_gray_message,
            **kwargs,
        )

        # Repeat Conv. Gray
        repetition_to_projector_frames(
            repetition_tuple,
            message_mapping=gray_message,
            **kwargs,
        )

        # Repeat long run
        repetition_to_projector_frames(
            repetition_tuple,
            message_mapping=long_run_gray_message,
            **kwargs,
        )


def dlp_patterns():
    projector_resolution = (1024, 768)

    kwargs = {
        "projector_resolution": projector_resolution,
        "show": False,
        "save": True,
        "use_complementary": True,
        "include_all_white": True,
        "out_dir": "outputs/projector_frames/dlp",
    }

    # Gray stripe (for calibration)
    gray_stripe_to_projector_frames(7, **kwargs)
    gray_stripe_to_projector_frames(7, encoded_dim="rows", **kwargs)

    kwargs["use_complementary"] = False

    # Gray code
    gray_code_to_projector_frames(**kwargs)

    # Hybrid
    hybrid_bch_tuple_ll = [
        metaclass.BCH(31, 11, 5),
        metaclass.BCH(63, 7, 15),
        metaclass.BCH(127, 8, 31),
        metaclass.BCH(255, 9, 63),
        metaclass.BCH(511, 10, 127),
    ]
    bch_message_bits = 7

    # BCH, Hybrid, Repetition for various redundancies
    for hybrid_bch_tuple in hybrid_bch_tuple_ll:
        # Hybrid with gray
        hybrid_to_projector_frames(
            hybrid_bch_tuple,
            bch_message_bits=bch_message_bits,
            message_mapping=gray_message,
            **kwargs,
        )

        # Hybrid with long run
        hybrid_to_projector_frames(
            hybrid_bch_tuple,
            bch_message_bits=bch_message_bits,
            message_mapping=long_run_gray_message,
            **kwargs,
        )


if __name__ == "__main__":
    dlp_patterns()
    # lcd_patterns()
