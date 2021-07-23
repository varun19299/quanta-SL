"""
Understand distance variation vs locality for coding strategies

Good locality properties will result in robust codes
"""
from math import ceil, log2

import galois
import graycode
import matplotlib as mpl
import numpy as np
import pandas as pd
from einops import rearrange
from galois import GF2
from matplotlib import pyplot as plt

from vis_tools.strategies import metaclass
from utils.plotting import save_plot
from ops.binary import unpackbits

params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (8, 6),
    "figure.titlesize": "xx-large",
    "axes.labelsize": "xx-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
plt.rcParams.update(params)


def error_matrix(vec):
    x = rearrange(vec, "n code_dim -> n 1 code_dim")
    y = rearrange(vec, "n code_dim -> 1 n code_dim")
    return (x ^ y).sum(axis=-1)


def graycode_mapping(projector_cols):
    num_bits = ceil(log2(projector_cols))
    message_ll = unpackbits(np.arange(pow(2, num_bits)))

    graycode_indices = graycode.gen_gray_codes(num_bits)

    # Permute according to Gray code
    gray_code_LUT = message_ll[graycode_indices, :]

    # Crop to fit projector width
    gray_code_LUT = gray_code_LUT[:projector_cols]

    return gray_code_LUT


def get_bch_codes(gray_code_LUT, bch_tuple_ll):
    bch_tuple_ll = [metaclass.BCH(*code_param) for code_param in bch_tuple_ll]

    for bch_tuple in bch_tuple_ll:
        bch = galois.BCH(bch_tuple.n, bch_tuple.k)
        code_LUT = bch.encode(GF2(gray_code_LUT))
        code_LUT = code_LUT.view(np.ndarray).astype(int)

        yield bch_tuple, code_LUT


def distance_matrix(projector_cols: int = 1920, **plot_options):
    num_bits = ceil(log2(projector_cols))

    # Get gray codes
    gray_code_LUT = graycode_mapping(1920)

    width = 20
    centre_crop = np.s_[
        pow(2, num_bits - 1) - width // 2 : pow(2, num_bits - 1) + width // 2 + 1,
        pow(2, num_bits - 1) - width // 2 : pow(2, num_bits - 1) + width // 2 + 1,
    ]

    # BCH code tuple
    bch_tuple_ll = [
        (15, 11, 1),
        (31, 11, 5),
        (63, 16, 11),
        (127, 15, 27),
        (255, 13, 59),
    ]

    subplot_cols = 3
    subplot_rows = ceil((1 + len(bch_tuple_ll)) / subplot_cols)
    fig, ax_ll = plt.subplots(
        subplot_rows,
        subplot_cols,
        # sharex=True,
        # sharey=True,
        figsize=(12, 12),
    )

    # Gray code
    ax_0 = ax_ll.flatten()[0]
    gray_cropped_img = error_matrix(gray_code_LUT)[centre_crop]
    ax_0.imshow(gray_cropped_img)
    img_shape = gray_cropped_img.shape
    ax_0.set_title("Gray Codes")

    # Generate BCH codes
    for e, (bch_tuple, code_LUT) in enumerate(
        get_bch_codes(gray_code_LUT, bch_tuple_ll), start=1
    ):
        ax = ax_ll.flatten()[e]
        im = ax.imshow(error_matrix(code_LUT)[centre_crop])
        ax.set_title(f"{bch_tuple}")

    plt.suptitle("Pair-wise distance mesh", fontsize=28, y=0.87)

    # Set the ticks and ticklabels for all axes
    h, w = img_shape
    xticks = [(i * h) // 6 for i in range(1, 6)]
    yticks = [(i * w) // 6 for i in range(1, 6)]

    offset = (projector_cols - width) // 2

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
    save_plot(fname="outputs/plots/locality_gray_code_vs_bch.pdf", **plot_options)


def _neighbour_distance(code_LUT):
    x = code_LUT[:-1]
    y = code_LUT[1:]

    distance_vec = (x ^ y).sum(axis=-1)
    return distance_vec


def _get_proportion(distance_vec, d):
    return (distance_vec == d).mean()


def neighbourhood_analysis(**plot_options):
    gray_code_LUT = graycode_mapping(1920)

    # BCH code tuple
    bch_tuple_ll = [
        (15, 11, 1),
        (31, 11, 5),
        (63, 16, 11),
        (127, 15, 27),
        (255, 13, 59),
    ]

    # Generate BCH codes
    proportion_dict = {}
    rel_dist = 6

    index_ll = ["$d$"]
    index_ll += [f"$d+{i}$" for i in range(1, rel_dist)]

    for bch_tuple, code_LUT in get_bch_codes(gray_code_LUT, bch_tuple_ll):
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

    save_plot(fname="outputs/plots/bch_neighbourhood_analysis.pdf", **plot_options)


if __name__ == "__main__":
    plot_options = {
        "show": True,
        "savefig": True,
    }

    # distance_matrix(**plot_options)
    neighbourhood_analysis(**plot_options)
