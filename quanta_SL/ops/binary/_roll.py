import numpy as np
from nptyping import NDArray


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