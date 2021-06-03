from matplotlib import pyplot as plt


# Matplotlib font sizes
TINY_SIZE = 8
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def arrowed_spines(
    ax,
    x_width_fraction=0.05,
    x_height_fraction=0.05,
    lw=None,
    ohg=0.3,
    locations=("bottom right", "left up"),
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

        annots[loc_str] = ax.arrow(
            x,
            y,
            dx,
            dy,
            fc="k",
            ec="k",
            lw=lw,
            head_width=head_width,
            head_length=head_length,
            **arrow_kwargs
        )

    return annots


fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

# removing the default axis on all sides:
for side in ["right", "top", "bottom", "left"]:
    ax.spines[side].set_visible(False)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_yticks([0, 1, 2, 3, 4])

ax.set_xticklabels(["1Hz", "10Hz", "100Hz", "1000Hz", "10kHz"])
# ax.set_xlabel("Speed")


ax.set_yticklabels(["100cm", "10cm", "1cm", "1mm", "$100\mu m$"])
# ax.set_ylabel("Depth Resolution")
annots = arrowed_spines(ax, locations=("bottom right", "left up"))

plt.tight_layout()
plt.grid()

plt.savefig("outputs/plots/broader_impact.pdf", dpi=150)
plt.show()
