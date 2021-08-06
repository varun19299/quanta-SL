"""
Long run vs Conventional gray
"""

from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from quanta_SL.encode.message import long_run_gray_message, gray_message
from quanta_SL.ops.binary import invert_permutation
from quanta_SL.ops.noise import fixed_bit_flip_corrupt
from quanta_SL.utils.plotting import save_plot

plt.style.use(["science", "grid"])


def decoding_rmse(
    message_mapping: Callable,
    bit_flips: int,
    num_bits: int = 10,
    monte_carlo_iters: int = None,
):
    code_LUT = message_mapping(num_bits)
    pow_vector = np.power(
        2, np.arange(start=code_LUT.shape[1] - 1, stop=-1, step=-1)
    ).astype(int)
    pow_vector = pow_vector

    inverse_perm = invert_permutation(code_LUT @ pow_vector)
    gt_indices = np.arange(len(code_LUT), dtype=int)

    if not monte_carlo_iters:
        monte_carlo_iters = pow(2, num_bits)

    pbar = tqdm(total=monte_carlo_iters)
    error = 0

    for e in range(1, monte_carlo_iters + 1):
        noisy_code_LUT = fixed_bit_flip_corrupt(code_LUT, bit_flips)

        decoded_indices = inverse_perm[noisy_code_LUT @ pow_vector]

        mse = ((decoded_indices - gt_indices) ** 2).mean()
        rmse = mse ** 0.5
        error += (rmse - error) / (e + 1)

        pbar.update(1)
        pbar.set_description(
            f"{message_mapping.name} | {bit_flips} bit flip(s) | RMSE {error}"
        )

    return error


def _compare(show: bool = True, savefig: bool = True):
    num_bits = 10
    mapping_ll = [long_run_gray_message, gray_message]
    rmse_dict = {mapping.name: {} for mapping in mapping_ll}

    for mapping in mapping_ll:
        for bit_flips in range(0, 6):
            rmse_dict[mapping.name][bit_flips] = decoding_rmse(
                mapping, bit_flips, num_bits
            )

    # Dataframe
    cols = rmse_dict[mapping_ll[0].name].keys()

    df = pd.DataFrame.from_dict(rmse_dict, orient="index", columns=cols).transpose()

    df.plot.bar(
        figsize=(7, 4),
        rot=0,
        xlabel="Corrupt bits",
        ylabel="RMSE",
        title=f"Fixed Bit Corruption ({num_bits} bit message)",
    )
    plt.legend()

    plt.tight_layout()

    save_plot(
        fname=f"outputs/plots/longrun_vs_conventional[fixed bit corruption].pdf",
        show=show,
        savefig=savefig,
    )


if __name__ == "__main__":
    _compare()
