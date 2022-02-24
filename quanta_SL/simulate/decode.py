from typing import Any

import graycode
import numpy as np
from nptyping import NDArray, Int


def conventional_gray_code(
    binary_codes: NDArray[(Any, Any, Any), Int], mask: NDArray[(Any, Any), Int]
) -> NDArray[(Any, Any), Int]:
    """

    :param binary_codes: Thresholded captures (h x w x num_captures)
    :param mask: RoI
    :return: Decoded map
    """
    # Convert binary_codes to gray code (in decimal) nos
    h, w, binary_width = binary_codes.shape
    gray_decimal = np.zeros((h, w))
    for i in range(binary_width):
        gray_decimal += binary_codes[:, :, i] * pow(2, binary_width - i - 1)
    gray_decimal = gray_decimal.astype(int)

    # Convert gray code (in decimal) to 2's complement positions (assume reflected gray codes)
    correspondence = np.array(
        list(map(graycode.gray_code_to_tc, gray_decimal.flatten()))
    ).reshape(gray_decimal.shape)

    masked_indices = np.where(mask == 0)
    correspondence[masked_indices] = -1

    return correspondence
