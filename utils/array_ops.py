from math import floor, log2

import numpy as np
from nptyping import NDArray
from numba import njit


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

    mask_size = floor(log2(x.max())) + 1
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


@njit(cache=True, fastmath=True, nogil=True)
def mean(array):
    out = 0
    for i in array:
        out += i
    return out / len(array)


@njit(cache=True, fastmath=True, nogil=True)
def min_stripe_width(code_LUT: np.ndarray):
    assert code_LUT.ndim == 2, "Must be message space x code dim"

    # Per frame statistics
    min_stripe_ll = []
    mean_stripe_ll = []

    for frame in code_LUT.transpose():
        # Intra-frame statistics
        stripe_ll = []
        stripe_width = 1

        for curr_elem, next_elem in zip(frame, np.roll(frame, -1)):
            if curr_elem == next_elem:
                stripe_width += 1
            else:
                stripe_ll.append(stripe_width)
                # Reset
                stripe_width = 1

        # If the frame happens to be full black / full white
        if not stripe_ll:
            stripe_ll.append(stripe_width)

        # Consider minimum stripe
        # After circular shifting
        if len(stripe_ll) >= 2 and (frame[0] == frame[-1]):
            stripe_ll[0] += stripe_ll[-1]
            stripe_ll = stripe_ll[:-1]

        min_stripe_ll.append(min(stripe_ll))
        mean_stripe_ll.append(mean(stripe_ll))

    return min(min_stripe_ll), min_stripe_ll, mean_stripe_ll


def left_bit_roll(x: NDArray[int], d: int, num_bits: int = None) -> NDArray[int]:
    """
    Roll x to left by d bits

    In n<<d, last d bits are 0.
    To put first d bits of n at
    last, do bitwise or of n<<d
    with n >>(INT_BITS - d)

    :param x: integer array
    :param d: roll
    :param num_bits: total bits. If None, set as log2(x.max())
    :return:
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x).astype(int)

    if not num_bits:
        num_bits = np.floor(np.log2(x.max())).astype(int) + d + 1
    else:
        # Modulo shift
        d %= num_bits

    y = (x << d) | (x >> (num_bits - d))

    # Chop bits
    y &= pow(2, num_bits) - 1

    return y


def right_bit_roll(x: NDArray[int], d: int, num_bits: int = None) -> NDArray[int]:
    """
    Roll x to right by d bits

    In n>>d, first d bits are 0.
    To put last d bits of at
    first, do bitwise or of n>>d
    with n <<(INT_BITS - d)

    :param x: integer array
    :param d: roll
    :param num_bits: total bits. If None, set as log2(x.max())
    :return:
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x).astype(int)

    if not num_bits:
        num_bits = np.floor(np.log2(x.max())).astype(int) + 1
    else:
        # Modulo shift
        d %= num_bits

    return (x >> d) | (x << (num_bits - d)) & (pow(2, num_bits) - 1)


def bit_roll(x: NDArray[int], d: int) -> NDArray[int]:
    """
    Roll x to left / right by d bits

    :param x: integer array
    :param d: roll
    :return:
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x).astype(int)

    num_bits = np.floor(np.log2(x.max())).astype(int) + 1

    if d > 0:
        return right_bit_roll(x, d, num_bits)

    elif d < 0:
        return left_bit_roll(x, -d, num_bits)

    else:
        return x
