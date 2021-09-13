from math import floor, log2

import numpy as np
from nptyping import NDArray


def unpackbits(x: NDArray[int], num_bits: int = 0, mode: str = "left-msb") -> NDArray:
    """
    Unpack an integer array into its binary representation.
    Similar to MATLAB's `de2bi`

    :param x: Integer array.
        For large integers use `object` dtype, python allows storing arbitrarily large ints. Numpy ctypes cannot do so.
        For more info, see here:
        https://stackoverflow.com/questions/18296035/how-to-extract-the-bits-of-larger-numeric-numpy-data-types
    :param num_bits: Number of bits in the final representation
        If less than ceil(log2(x.max())), we will drop the MSB bits accordingly.
        Else, we pad sufficiently (to the MSB).
    :param mode: Left MSB (default) or Right MSB.
    :return:
    """
    assert mode in ["left-msb", "right-msb"]

    # Object mode for large integers
    if not (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, object)):
        raise ValueError("numpy data type needs to be int-like")

    if np.iinfo(np.maximum_sctype(int)).max < x.max():
        raise ValueError(
            f"Detected large binary representation. Too many bits. Use `object` dtype instead of {x.dtype}"
        )

    if not num_bits:
        num_bits = floor(log2(x.max())) + 1

    assert (
        isinstance(x, np.ndarray) and x.ndim >= 1
    ), "Input must be atleast 1D numpy array"

    xshape = list(x.shape)
    x = x.reshape([-1, 1])

    mask_size = floor(log2(max(1, x.max()))) + 1
    mask = 2 ** np.arange(mask_size, dtype=x.dtype).reshape([1, mask_size])

    binary_rep = (x & mask).astype(bool).astype(int).reshape(xshape + [mask_size])

    if num_bits < mask_size:
        binary_rep = binary_rep[:, :num_bits]
    elif num_bits > mask_size:
        binary_rep = np.concatenate(
            (binary_rep, np.zeros((xshape[0], num_bits - mask_size), dtype=x.dtype)),
            axis=1,
        )

    if mode == "left-msb":
        binary_rep = binary_rep[:, ::-1]

    return binary_rep


def packbits(x: NDArray[int], mode: str = "left-msb") -> NDArray:
    """
    Pack an binary array into its integer representation.
    Axis assumed to be last
    Unlike numpy, returns an integer, not uint8.
    Similar to MATLAB's `de2bi`

    :param x: Binary array.
    :param mode: Left MSB (default) or Right MSB.
    :return:
    """
    assert (
        isinstance(x, np.ndarray) and x.ndim >= 1
    ), "Input must be atleast 1D numpy array"

    if x.ndim == 1:
        x = x.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    assert mode in ["left-msb", "right-msb"]
    num_bits = x.shape[-1]

    power_vec = pow(2, np.arange(num_bits))

    if mode == "left-msb":
        power_vec = power_vec[::-1]

    power_vec = power_vec.reshape(-1, 1)

    decimal_rep = np.matmul(x, power_vec).squeeze(-1)

    if squeeze:
        decimal_rep.squeeze()

    return decimal_rep
