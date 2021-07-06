"""
Understand distance variation vs locality for coding strategies

Good locality properties will result in robust codes
"""
from einops import rearrange
from matplotlib import pyplot as plt
import matplotlib as mpl
from vis_tools.strategies.utils import unpackbits
from vis_tools.strategies import metaclass
import numpy as np
import galois
from galois import GF2
import graycode
from math import ceil

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


if __name__ == "__main__":
    num_bits = 10
    message_ll = unpackbits(np.arange(pow(2, num_bits)))

    graycode_indices = graycode.gen_gray_codes(num_bits)
    gray_code_LUT = message_ll[graycode_indices, :]

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
    bch_tuple_ll = [metaclass.BCH(*code_param) for code_param in bch_tuple_ll]

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
    for ax, bch_tuple in zip(ax_ll.flatten()[1:], bch_tuple_ll):
        code_LUT = galois.BCH(bch_tuple.n, bch_tuple.k).encode(GF2(gray_code_LUT))
        code_LUT = code_LUT.view(np.ndarray).astype(int)

        im = ax.imshow(error_matrix(code_LUT)[centre_crop])
        ax.set_title(f"{bch_tuple}")

    plt.suptitle("Pair-wise distance mesh", fontsize=28, y=0.87)

    # Set the ticks and ticklabels for all axes
    h, w = img_shape
    xticks = [(i * h) // 6 for i in range(1, 6)]
    yticks = [(i * w) // 6 for i in range(1, 6)]

    offset = pow(2, num_bits - 1) - width // 2

    xticklabels = [rf"$c_{'{' + str(i + offset) +'}'}$" for i in xticks]
    yticklabels = [rf"$c_{'{' + str(i + offset) +'}'}$" for i in yticks]
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
    plt.savefig(
        "outputs/plots/locality_gray_code_vs_bch.pdf",
        dpi=200,
        bbox_inches="tight",
        transparent=True,
    )

    plt.show()
