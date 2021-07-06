from math import ceil, log2
from pathlib import Path
from typing import Callable, Union

import numpy as np
from matplotlib import pyplot as plt
from nptyping import NDArray

from vis_tools.strategies.metaclass import BCH


def list_error_correcting_capacity(bch_tuple: BCH) -> int:
    """
    Based on BCH LECC limit.
    See Wu et al. 2008, https://arxiv.org/pdf/cs/0703105.pdf.

    :param bch_tuple: Describes BCH code as [n, k, t]
        n: Code length
        k: Message length
        t: Worst case correctable errors
    :return: List Error Correcting Capacity
    """
    n = bch_tuple.n
    d = bch_tuple.distance
    return n / 2 * (1 - (1 - 2 * d / n) ** 0.5)


def func_name(func: Union[Callable, str]) -> str:
    if isinstance(func, Callable):
        func_str = func.__name__
    elif isinstance(func, str):
        func_str = func
    return func_str.replace("_", " ").title()


def order_range(array: NDArray) -> int:
    upper = np.log10(array.max())
    lower = np.log10(array.min())
    return upper.astype(int) - lower.astype(int) + 1


def save_plot(savefig, show: bool, **kwargs):
    """
    Helper function for saving plots
    :param savefig: Whether to save the figure
    :param show: Display in graphical window or just close the plot
    :param kwargs: fname, close
    :return:
    """
    if "close" in kwargs:
        close = kwargs["close"]
    else:
        close = not show

    if savefig:
        path = Path(kwargs["fname"])
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(kwargs["fname"], dpi=150, bbox_inches="tight", transparent=True)

    if show:
        plt.show()
    if close:
        plt.close()


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
        num_bits = ceil(log2(x.max()))

    xshape = list(x.shape)
    x = x.reshape([-1, 1])

    mask_size = ceil(log2(x.max()))
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
