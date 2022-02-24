"""
Understand distance variation vs locality for coding strategies

Good locality properties will result in robust codes
"""
from math import ceil
from typing import Callable

import matplotlib as mpl
import numpy as np
import pandas as pd
from einops import rearrange
from matplotlib import pyplot as plt

from quanta_SL.encode.message import gray_message, long_run_gray_message
from quanta_SL.encode.metaclass import BCH
from quanta_SL.encode.strategies import bch_code_LUT
from quanta_SL.utils.plotting import save_plot

plt.style.use(["science", "grid"])

params = {
    "legend.fontsize": "x-large",
    "figure.titlesize": "xx-large",
    "axes.labelsize": "xx-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
plt.rcParams.update(params)


# BCH code tuple
bch_tuple_ll = [
    BCH(15, 11, 1),
    BCH(31, 11, 5),
    BCH(63, 16, 11),
    BCH(127, 15, 27),
    BCH(255, 13, 59),
]


def error_matrix(vec):
    x = rearrange(vec, "n code_dim -> n 1 code_dim")
    y = rearrange(vec, "n code_dim -> 1 n code_dim")
    return (x ^ y).sum(axis=-1)


def _neighbour_distance(code_LUT):
    x = code_LUT[:-1]
    y = code_LUT[1:]

    distance_vec = (x ^ y).sum(axis=-1)
    return distance_vec


def _get_proportion(distance_vec, d):
    return (distance_vec == d).mean()


def distance_matrix(
    num_bits: int = 8, message_mapping: Callable = gray_message, **plot_options
):
    # Get message LUT
    message_LUT = message_mapping(num_bits)

    width = pow(2, num_bits - 1)
    centre_crop = np.s_[
        pow(2, num_bits - 1) - width // 2 : pow(2, num_bits - 1) + width // 2 + 1,
        pow(2, num_bits - 1) - width // 2 : pow(2, num_bits - 1) + width // 2 + 1,
    ]

    bch_code_LUT_ll = [
        bch_code_LUT(bch_tuple, num_bits, message_mapping) for bch_tuple in bch_tuple_ll
    ]

    subplot_cols = 3
    subplot_rows = ceil((1 + len(bch_tuple_ll)) / subplot_cols)
    fig, ax_ll = plt.subplots(
        subplot_rows,
        subplot_cols,
        figsize=(9, 9),
    )

    title_ll = [
        message_mapping.name,
        *[str(bch_tuple) for bch_tuple in bch_tuple_ll],
    ]
    code_LUT_ll = [message_LUT, *bch_code_LUT_ll]

    for e, (code_LUT, title) in enumerate(zip(code_LUT_ll, title_ll)):
        ax = ax_ll.flatten()[e]
        img = error_matrix(code_LUT)[centre_crop]
        im = ax.imshow(img)
        ax.set_title(title)

    plt.suptitle(f"Pair-wise Distances ({num_bits} bit message)", fontsize=28, y=0.87)

    # Set the ticks and ticklabels for all axes
    h, w = img.shape
    xticks = [(i * h) // 6 for i in range(1, 6)]
    yticks = [(i * w) // 6 for i in range(1, 6)]

    offset = (pow(2, num_bits) - width) // 2

    xticklabels = [rf"$c_{'{' + str(i + offset) + '}'}$" for i in xticks]
    yticklabels = [rf"$c_{'{' + str(i + offset) + '}'}$" for i in yticks]
    plt.setp(
        ax_ll,
        xticks=xticks,
        xticklabels=xticklabels,
        yticks=yticks,
        yticklabels=yticklabels,
    )

    plt.tight_layout()
    fig.subplots_adjust(hspace=-0.4)

    # Common color bar
    cax, kw = mpl.colorbar.make_axes(
        [ax for ax in ax_ll.flatten()[: len(bch_tuple_ll) + 1]],
        fraction=0.046,
        shrink=0.67,
        aspect=30,
        pad=0.02,
    )
    plt.colorbar(im, cax=cax, **kw)

    # Save
    fname = message_mapping.name.replace(" ", "_").lower()
    save_plot(
        fname=f"outputs/plots/locality_{fname}_vs_bch_encoded.pdf", **plot_options
    )


def neighbourhood_analysis(
    num_bits: int = 10, message_mapping: Callable = gray_message, **plot_options
):
    proportion_dict = {}
    rel_dist = 6

    index_ll = ["$d$"]
    index_ll += [f"$d+{i}$" for i in range(1, rel_dist)]

    for bch_tuple in bch_tuple_ll:
        code_LUT = bch_code_LUT(bch_tuple, num_bits, message_mapping)
        distance_vec = _neighbour_distance(code_LUT)

        t = bch_tuple.t
        bch_dist = 2 * t + 1

        proportion_dict[str(bch_tuple)] = [
            _get_proportion(distance_vec, d)
            for d in (bch_dist + i for i in range(rel_dist))
        ]

    df = pd.DataFrame(proportion_dict, index=index_ll)
    df.plot.bar(
        xlabel="Neighbour Distance",
        ylabel="Proportion",
        ylim=(0, 1),
        rot=0,
        grid=True,
        figsize=(10, 5),
    )
    plt.tight_layout()

    # Save
    fname = message_mapping.name.replace(" ", "_").lower()
    save_plot(
        fname=f"outputs/plots/bch_{fname}_neighbourhood_analysis.pdf", **plot_options
    )


if __name__ == "__main__":
    plot_options = {
        "show": True,
        "savefig": True,
    }

    distance_matrix(num_bits=8, message_mapping=gray_message, **plot_options)
    distance_matrix(num_bits=8, message_mapping=long_run_gray_message, **plot_options)

    # neighbourhood_analysis(**plot_options)
