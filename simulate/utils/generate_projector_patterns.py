import logging
from math import ceil, log2
from pathlib import Path
from typing import Tuple

import cv2
import galois
import graycode
import numpy as np
from einops import repeat, rearrange
from galois import GF2
from matplotlib import pyplot as plt
from nptyping import NDArray
from tqdm import tqdm

from vis_tools.strategies import metaclass
from vis_tools.strategies.utils import unpackbits

FORMAT = "%(asctime)s [%(filename)s : %(funcName)2s() : %(lineno)2s] %(message)s"
logging.basicConfig(format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)


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

    # Generate gray code mapping on projector resolution
    column_bits = ceil(log2(N))
    graycode_ll = graycode.gen_gray_codes(column_bits)

    # Map gray codes to coded equivalents
    code_ll_gray_mapped = code_LUT[graycode_ll, :]

    # Crop to fit projector width
    code_ll_gray_mapped = code_ll_gray_mapped[(N - width) // 2 : (N + width) // 2]

    if use_complementary:
        code_ll_gray_comp = np.zeros((width, 2 * n))

        # Interleave
        # Can do more cleverly via np.ravel
        # but this is most readable
        code_ll_gray_comp[:, ::2] = code_ll_gray_mapped
        code_ll_gray_comp[:, 1::2] = 1 - code_ll_gray_mapped
        code_ll_gray_mapped = code_ll_gray_comp

    code_ll_gray_mapped = repeat(
        code_ll_gray_mapped, "width n -> height width n", height=height
    )

    if save:
        assert folder_name, f"No output folder provided"
        out_dir = Path("outputs/projector_frames") / folder_name
        out_dir.mkdir(exist_ok=True, parents=True)

    # Write out files
    if show or save:
        logging.info(f"Saving / showing strategy {folder_name}")

        pbar = tqdm(rearrange(code_ll_gray_mapped, "height width n -> n height width"))

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
    :param puncture: transmit only n - (k - log2(projector_cols))
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

    code_LUT = unpackbits(np.arange(pow(2, num_bits)))

    code_LUT_to_projector_frames(code_LUT=code_LUT, **kwargs)


def bch_to_projector_frames(
    bch_tuple: metaclass.BCH,
    projector_resolution: Tuple[int],
    use_complementary: bool = False,
    puncture: bool = True,
    show: bool = False,
    save: bool = False,
    folder_name: str = "",
):
    """
    Convert BCH_matlab codes to projector frames.

    :param bch_tuple: BCH_matlab code [n,k,t] parameters
    :param projector_resolution: width x height
    :param use_complementary: whether to save / show complementary (1-frame_i)
    :param puncture: transmit only n - (k - log2(projector_cols))
    :param show: Plot in pyplot
    :param save: Save as png
    :param folder_name: Folder name to save to
    """
    if not folder_name:
        folder_name = f"{bch_tuple}-non-sys"

    if use_complementary:
        folder_name += "-comp"

    kwargs = locals().copy()

    # Find bits required to represent columns
    width, height = projector_resolution
    num_bits = ceil(log2(width))

    message_ll = np.arange(pow(2, num_bits))
    message_ll = unpackbits(message_ll, bch_tuple.k)

    # Generate BCH_matlab codes
    code_LUT = galois.BCH(bch_tuple.n, bch_tuple.k, systematic=False).encode(GF2(message_ll))
    code_LUT = code_LUT.view(np.ndarray).astype(int)

    # Puncture
    logging.info(f"Puncturing by {bch_tuple.k - num_bits} bits")
    code_LUT = code_LUT[:, bch_tuple.k - num_bits :]

    code_LUT_to_projector_frames(code_LUT=code_LUT, **kwargs)


def repetition_to_projector_frames(
    repetition_tuple: metaclass.Repetition,
    projector_resolution: Tuple[int],
    use_complementary: bool = False,
    puncture: bool = True,
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

    message_ll = np.arange(pow(2, num_bits))
    message_ll = unpackbits(message_ll, repetition_tuple.k)

    # Generate repetition codes
    # Repeats consecutive frames
    code_LUT = repeat(message_ll, "N k -> N (k repeat)", repeat=repetition_tuple.repeat)

    code_LUT_to_projector_frames(code_LUT=code_LUT, **kwargs)


if __name__ == "__main__":
    num_bits = 11
    projector_resolution = (1920, 1080)

    kwargs = {"show": False, "save": True}

    gray_code_to_projector_frames(projector_resolution, **kwargs)
    bch_to_projector_frames(metaclass.BCH(63, 16, 11), projector_resolution, **kwargs)
    bch_to_projector_frames(
        metaclass.BCH(31, 11, 5),
        projector_resolution,
        use_complementary=True,
        **kwargs,
    )
    repetition_to_projector_frames(
        metaclass.Repetition(66, 11, 2), projector_resolution, **kwargs
    )
