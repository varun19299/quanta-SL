"""
Compare precision frame encoding and decoding
"""

import numpy as np
from nptyping import NDArray


def circular_shifted_stripes(stripe_width: int = 8) -> NDArray[int]:
    """
    Generate circularly shifted stripe pattern

    :param stripe_width: width of 1's / 0's
        Periodicity is 2 * stripe_width
    :return: code Look Up Table (LUT),
        columns x time
    """

    pattern = [0] * stripe_width + [1] * stripe_width
    pattern = np.array(pattern)

    code_LUT = []
    for i in range(stripe_width * 2):
        code_LUT.append(np.roll(pattern, i))

    code_LUT = np.stack(code_LUT, axis=-1).astype(int)

    return code_LUT
