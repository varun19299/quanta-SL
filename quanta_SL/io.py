import numpy as np
from quanta_SL.ops.binary import unpackbits
from einops import rearrange
from scipy.io import loadmat
from pathlib import Path
from nptyping import NDArray


def load_swiss_spad_frame(
    folder_name: Path,
    file_name: str,
    bin_suffix: int,
    num_rows: int = 256,
    num_cols: int = 512,
) -> NDArray[int]:
    """
    Load swiss SPAD frames.
    Dumped by MATLAB code.

    Each capture consists of .bin files, nested as:

    `folder_name/outer/inner/file_name.bin`

    Where `outer` goes from 0...2^11,
        `inner` goes from 0...2^6

    :param folder_name: Capture folder name
    :param file_name:
    :param bin_suffix:
    :param num_rows:
    :param num_cols:
    :return:
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

    return array


def test_load_swiss_spad_frame():
    spad_dump_foldername = "data/test_spad_matlab_dump"
    spad_dump_filename = "filename"

    matlab_filename = "data/test_spad_io.mat"
    array = load_swiss_spad_frame(
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
