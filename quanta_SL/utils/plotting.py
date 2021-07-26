from pathlib import Path

import cv2
import numpy as np
from einops import repeat

from matplotlib import pyplot as plt


def save_plot(savefig: bool = False, show: bool = True, **kwargs):
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


def arrowed_spines(
    ax,
    x_width_fraction=0.05,
    x_height_fraction=0.05,
    lw=None,
    ohg=0.3,
    locations=("bottom right", "left up"),
    remove_spine: bool = False,
    **arrow_kwargs
):
    """
    Add arrows to the requested spines
    Code originally sourced here: https://3diagramsperpage.wordpress.com/2014/05/25/arrowheads-for-axis-in-matplotlib/
    And interpreted here by @Julien Spronck: https://stackoverflow.com/a/33738359/1474448
    Then corrected and adapted by me for more general applications.
    :param ax: The axis being modified
    :param x_{height,width}_fraction: The fraction of the **x** axis range used for the arrow height and width
    :param lw: Linewidth. If not supplied, default behaviour is to use the value on the current left spine.
    :param ohg: Overhang fraction for the arrow.
    :param locations: Iterable of strings, each of which has the format "<spine> <direction>". These must be orthogonal
    (e.g. "left left" will result in an error). Can specify as many valid strings as required.
    :param arrow_kwargs: Passed to ax.arrow()
    :return: Dictionary of FancyArrow objects, keyed by the location strings.
    """
    # set/override some default plotting parameters if required
    arrow_kwargs.setdefault("overhang", ohg)
    arrow_kwargs.setdefault("clip_on", False)
    arrow_kwargs.update({"length_includes_head": True})

    # axis line width
    if lw is None:
        # FIXME: does this still work if the left spine has been deleted?
        lw = ax.spines["left"].get_linewidth()

    annots = {}

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if remove_spine:
        # removing the default axis on all sides:
        for side in ["bottom", "right", "top", "left"]:
            ax.spines[side].set_visible(False)

    # get width and height of axes object to compute
    # matching arrowhead length and width
    fig = ax.get_figure()
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = x_width_fraction * (ymax - ymin)
    hl = x_height_fraction * (xmax - xmin)

    # compute matching arrowhead length and width
    yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
    yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

    # draw x and y axis
    for loc_str in locations:
        side, direction = loc_str.split(" ")
        assert side in {"top", "bottom", "left", "right"}, "Unsupported side"
        assert direction in {"up", "down", "left", "right"}, "Unsupported direction"

        if side in {"bottom", "top"}:
            if direction in {"up", "down"}:
                raise ValueError(
                    "Only left/right arrows supported on the bottom and top"
                )

            dy = 0
            head_width = hw
            head_length = hl

            y = ymin if side == "bottom" else ymax

            if direction == "right":
                x = xmin
                dx = xmax - xmin
            else:
                x = xmax
                dx = xmin - xmax

        else:
            if direction in {"left", "right"}:
                raise ValueError("Only up/downarrows supported on the left and right")
            dx = 0
            head_width = yhw
            head_length = yhl

            x = xmin if side == "left" else xmax

            if direction == "up":
                y = ymin
                dy = ymax - ymin
            else:
                y = ymax
                dy = ymin - ymax

        # arrow_kwargs.update("")

        annots[loc_str] = ax.arrow(
            x,
            y,
            dx,
            dy,
            # fc="k",
            # ec="k",
            lw=lw,
            head_width=head_width,
            head_length=head_length,
            **arrow_kwargs
        )

    return annots


def plot_code_LUT(
    code_LUT: np.ndarray, show: bool = True, aspect_ratio: float = 3.0, **kwargs
):
    """
    Image illustrating coding scheme
    :param code_LUT: Code Look-Up-Table
    """
    h, c = code_LUT.shape

    num_repeat = kwargs.get("num_repeat", int(h / c / aspect_ratio))

    code_img = repeat(code_LUT, "h c -> (c repeat) h", repeat=num_repeat)

    if kwargs.get("savefig") or kwargs.get("fname"):
        assert kwargs.get("fname")
        cv2.imwrite(str(kwargs["fname"]), code_img * 255)

    if show:
        plt.imshow(code_img, cmap="gray")
        plt.show()

    return code_img