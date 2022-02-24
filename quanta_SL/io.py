from pathlib import Path

import cv2
import numpy as np
from einops import rearrange
from loguru import logger
from matplotlib import pyplot as plt
from nptyping import NDArray
from scipy.io import loadmat

from quanta_SL.ops.binary import unpackbits


def load_swiss_spad_bin(
    folder_name: Path,
    file_name: str = "filename",
    bin_suffix: int = 0,
    num_rows: int = 256,
    num_cols: int = 512,
) -> NDArray[int]:
    """
    Load a single SwissSPAD2 binary (.bin) file
    Dumped by MATLAB code.

    Each capture consists of .bin files, nested as:

    `folder_name/outer/inner/file_name.bin`

    Where `outer` goes from 0...2^11,
        `inner` goes from 0...2^6

    :param folder_name: Capture folder name
    :param file_name: Binary file name
    :param bin_suffix: Bin index, will be
        sorted accordingly into inner and outer
    :param num_rows: Sensor height
    :param num_cols: Sensor width
    :return: SPAD frames [burst x rows x cols]
    """
    # 8 MB
    file_size = 8 * 1024 * 1024 * 8
    num_frames = file_size // num_rows // num_cols

    if not isinstance(folder_name, Path):
        folder_name = Path(folder_name)

    outer_folder = bin_suffix // pow(2, 11)
    inner_folder = bin_suffix // pow(2, 6)
    path = folder_name / f"{outer_folder}/{inner_folder}/{file_name}{bin_suffix}.bin"

    # Flat array
    # (N,)
    array = np.fromfile(path, dtype=np.uint8)

    # Unpack uint8 to (N, 8)
    array = unpackbits(array, num_bits=8).astype(np.uint8)

    array = np.reshape(array.T, (file_size, 1), order="F")

    # Extract frames
    array = np.reshape(array, (num_frames, num_rows, num_cols), order="F")

    # Fix read-out order of bits
    # FIXME: exact reasoning unclear here. Changes welcome!
    array = np.reshape(array, (4, 2, num_frames * num_rows * num_cols // 8), order="F")
    array = np.flip(array, 0)
    array = array[:, ::-1, ::]
    array = np.reshape(array, (num_cols, num_rows, num_frames), order="F")
    array = rearrange(array, "c r n -> n r c")

    logger.info(f"Loaded {array.shape} [n x r x c] burst from {path}")

    return array


def load_swiss_spad_sequence(
    folder_name: Path,
    file_name: str = "filename",
    num_rows: int = 256,
    num_cols: int = 512,
    bin_suffix_range: range = range(1),
    block_range: range = range(1),
    use_gamma_correction: bool = False,
    pre_scale: int = 1,
    post_scale: int = 1,
    gamma: float = 0.4,
    show: bool = False,
) -> NDArray[int]:
    """
    Load multiple bursts (or subset of) of Swiss SPAD frames.

    Each capture consists of .bin files, nested as:

    `folder_name/outer/inner/file_name{bin_index}.bin`

    Where `outer` goes from 0...2^11,
        `inner` goes from 0...2^6

    :param folder_name: Folder name for captures
    :param file_name: Prefix for the .bin files
    :param num_rows: Sensor height
    :param num_cols: Sensor width
    :param bin_suffix_range: bin_index start and stop
    :param block_range: To index a burst
        (eg: 25 burst frames from indices starting at 0, 100, 200, ...)
    :param use_gamma_correction: Use gamma correction while displaying
    :param pre_scale: Linear scaling before gamma exponentiation
    :param post_scale: Linear scaling after gamma exponentiation
    :param gamma: Gamma exponential power
    :param show: Display via Pyplot or not
    :return: Return burst image
    """
    image_ll = []

    for bin_suffix in bin_suffix_range:
        frame = np.zeros((num_rows, num_cols))

        # To pick a certain burst of frames
        for block_index in block_range:
            frame += load_swiss_spad_bin(
                folder_name, file_name, bin_suffix + block_index, num_rows, num_cols
            ).mean(axis=0)

        frame /= len(block_range)

        image_ll.append(frame)

    image = np.stack(image_ll, axis=0).mean(axis=0)

    if show:
        image_disp = image.copy()

        if use_gamma_correction:
            image_disp = pow((image_disp * pre_scale), gamma) * post_scale
            image_disp = cv2.equalizeHist(image_disp)

        plt.imshow(image_disp, cmap="gray")
        plt.show()

    return image


def test_load_swiss_spad_burst():
    spad_dump_foldername = "data/test_spad_matlab_dump"
    spad_dump_filename = "filename"

    matlab_filename = "data/test_spad_io.mat"
    array = load_swiss_spad_bin(
        spad_dump_foldername,
        spad_dump_filename,
        bin_suffix=0,
        num_rows=256,
        num_cols=512,
    )

    matlab_array = loadmat(matlab_filename)["frames"]

    assert np.array_equal(
        array, matlab_array
    ), "SPAD IO failed. Check if .bin file corresponds to .mat test file."
