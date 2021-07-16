from pathlib import Path

from matplotlib import pyplot as plt


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