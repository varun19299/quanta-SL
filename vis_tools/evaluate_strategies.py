from typing import Callable, List

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from nptyping import NDArray
from pathlib import Path

from vis_tools.strategies import (
    naive,
    naive_conventional,
    average,
    average_conventional,
    optimal_threshold,
    average_optimal_threshold,
)

params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (8, 6),
    "axes.labelsize": "xx-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
}
plt.rcParams.update(params)

LINEWIDTH = 3


def order_range(array: NDArray):
    upper = np.log10(array.max())
    lower = np.log10(array.min())
    return upper.astype(int) - lower.astype(int) + 1


def _save_plot(savefig, show: bool, **kwargs):
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


def plot_optimal_threshold(
    phi_proj,
    phi_A,
    t_exp: float,
    num_frames: int = 10,
    savefig: bool = False,
    show: bool = True,
    **kwargs,
):
    print("Plotting optimal threshold \n")
    FIGSIZE = (8, 4)

    # varying phi P, different lines for Phi A
    plt.figure(figsize=FIGSIZE)

    idx_A = np.round(np.linspace(0, len(phi_A) - 1, 4)).astype(int)
    for a in phi_A[idx_A]:
        # Phi_P = phi_proj + a
        thresh_ll, tau_ll = optimal_threshold(phi_proj + a, a, t_exp, num_frames)
        plt.semilogx(
            phi_proj + a, tau_ll, label=f"$\Phi_a=${a:.0f}", linewidth=LINEWIDTH
        )

    def _plt_properties(xlabel):
        plt.xlabel(xlabel)
        plt.ylabel(r"Threshold $(\frac{\tau^*}{N_\mathrm{frames}})$")
        plt.legend(loc="upper right")
        plt.grid()
        plt.tight_layout()

    _plt_properties("$\Phi_p$")
    _save_plot(
        savefig, show, fname=f"outputs/plots/strategy_plots/optimal_thresh_vs_phi_p.pdf"
    )

    # varying phi A, different lines for Phi P
    plt.figure(figsize=FIGSIZE)

    idx_P = np.round(np.linspace(0, len(phi_proj) - 1, 7)).astype(int)
    for p in phi_proj[idx_P]:
        thresh_ll, tau_ll = optimal_threshold(p + phi_A, phi_A, t_exp, num_frames)
        plt.semilogx(
            phi_A, tau_ll, label=f"$\Phi_p - \Phi_a=${p:.0e}", linewidth=LINEWIDTH
        )

    _plt_properties("$\Phi_a$")
    _save_plot(
        savefig, show, fname=f"outputs/plots/strategy_plots/optimal_thresh_vs_phi_A.pdf"
    )


def plot_strategy_3d(
    eval_error,
    phi_proj,
    phi_A,
    savefig: bool = False,
    outname: str = "",
    show: bool = True,
):
    phi_proj_mesh, phi_A_mesh = np.meshgrid(phi_proj, phi_A, indexing="ij")

    # Plot the surface.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(
        np.log10(phi_proj_mesh),
        np.log10(phi_A_mesh),
        eval_error,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )

    # X, Y axis
    ax.set_xlabel("\n$\Phi_p - \Phi_a$ Projector Flux", labelpad=5, rotation=0)
    xticks = np.log10(phi_proj).astype(int)
    xticklabels = [f"$10^{x}$" for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_ylabel("\n$\Phi_a$ Ambient Flux", rotation=-60)
    yticks = np.log10(phi_A).astype(int)
    yticklabels = [f"$10^{y}$" for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # Z axis
    ax.set_zlabel("Error Probability", labelpad=5)
    zticks = [p / 10 for p in range(0, 11, 2)]
    ax.set_zticks(zticks)
    ax.set_zticklabels([f"{z:.1f}" for z in zticks])

    _save_plot(
        savefig,
        show=False,
        close=False,
        fname=f"outputs/plots/strategy_plots/{outname}/surface_plot_no_colorbar.pdf",
    )

    # Colorbar
    def surface_colorbar(fig, mappable):
        cbar = fig.colorbar(mappable, pad=0.1)
        cbar.ax.set_title("P(error)")

    surface_colorbar(fig, surf)
    plt.title(f"{outname.replace('_', ' ').capitalize()}")
    _save_plot(
        savefig,
        show,
        fname=f"outputs/plots/strategy_plots/{outname}/surface_plot_with_colorbar.pdf",
    )

    # draw a new figure and replot the colorbar there
    fig, ax = plt.subplots()
    surface_colorbar(fig, surf)
    ax.remove()
    _save_plot(
        savefig,
        show=False,
        fname=f"outputs/plots/strategy_plots/{outname}/surface_plot_only_colorbar.pdf",
    )


def plot_strategy_2d(
    eval_error,
    phi_proj,
    phi_A,
    savefig: bool = False,
    outname: str = "",
    show: bool = True,
):
    # Plot image
    plt.figure()
    eval_image = plt.imshow(eval_error, cmap="plasma")
    plt.gca().invert_yaxis()

    # Label x axis
    xticks = np.round(np.linspace(0, len(phi_A) - 1, order_range(phi_A))).astype(int)
    xticklabels = [f"{label:.0e}" for label in phi_A[xticks]]
    plt.xticks(xticks, xticklabels)
    plt.xlabel("$\Phi_a$ Ambient Flux", labelpad=5)

    # Label y axis
    yticks = np.round(np.linspace(0, len(phi_proj) - 1, order_range(phi_proj))).astype(
        int
    )
    yticklabels = [f"{label:.0e}" for label in phi_proj[yticks]]
    plt.yticks(yticks, yticklabels)
    plt.ylabel("$\Phi_p - \Phi_a$ Projector Flux")

    plt.grid()
    _save_plot(
        savefig,
        show=False,
        close=False,
        fname=f"outputs/plots/strategy_plots/{outname}/mesh_no_colorbar.pdf",
    )

    # Colorbar
    def img_colorbar(**kwargs):
        cbar = plt.colorbar(**kwargs)
        cbar.ax.set_title("P(error)")
        cbar.ax.locator_params(nbins=5)
        cbar.update_ticks()

    img_colorbar()
    plt.clim(0, 1)
    plt.title(f"{outname.replace('_',' ').capitalize()}")
    _save_plot(
        savefig,
        show,
        fname=f"outputs/plots/strategy_plots/{outname}/mesh_with_colorbar.pdf",
    )

    # draw a new figure and replot the colorbar there
    fig, ax = plt.subplots()
    img_colorbar(mappable=eval_image, ax=ax)
    ax.remove()
    _save_plot(
        savefig,
        show=False,
        fname=f"outputs/plots/strategy_plots/{outname}/mesh_only_colorbar.pdf",
    )


def plot_strategy(
    phi_proj,
    phi_A,
    t_exp: float,
    strategy: Callable,
    savefig: bool = False,
    show: bool = True,
    plot_3d: bool = False,
    outname: str = "",
    **kwargs,
):
    print(f"Plotting strategy {outname if outname else strategy.__name__}")

    # Meshgrid
    phi_P_mesh, phi_A_mesh = np.meshgrid(phi_proj + phi_A, phi_A, indexing="ij")

    # Evaluate strategy
    eval_error = strategy(phi_P_mesh, phi_A_mesh, t_exp, **kwargs)

    # Outname
    if not outname:
        outname = strategy.__name__

    if plot_3d:
        plot_strategy_3d(eval_error, phi_proj, phi_A, savefig, outname, show)

    plot_strategy_2d(eval_error, phi_proj, phi_A, savefig, outname, show)

    return eval_error


def average_vs_frames(
    strategy_ll: List[Callable],
    frames_ll: List[int],
    phi_proj: NDArray,
    phi_A: NDArray,
    t_exp: float,
    **kwargs,
):
    for strategy in strategy_ll:
        for frames in frames_ll:
            plot_strategy(
                phi_proj,
                phi_A,
                t_exp,
                strategy,
                outname=f"{strategy.__name__}/frames_{frames}",
                num_frames=frames,
                **kwargs,
            )


if __name__ == "__main__":
    phi_proj = np.logspace(1, 8, num=512)
    phi_A = np.logspace(0, 5, num=512)

    # DMD framerate
    # 0.1 millisecond or 10^4 FPS
    t_exp = 1e-4

    plot_options = {
        "show": False,
        "plot_3d": True,
        "savefig": True,
    }

    plot_optimal_threshold(
        phi_proj,
        phi_A,
        t_exp=t_exp,
        num_frames=10,
        **plot_options,
    )

    # Naive
    plot_strategy(phi_proj, phi_A, t_exp, naive, **plot_options)

    # Average
    plot_strategy(
        phi_proj,
        phi_A,
        t_exp,
        average,
        num_frames=10,
        **plot_options,
    )

    # Average Optimal
    plot_strategy(
        phi_proj,
        phi_A,
        t_exp,
        average_optimal_threshold,
        num_frames=10,
        **plot_options,
    )

    # Conventional Naive
    plot_strategy(
        phi_proj,
        phi_A,
        t_exp,
        naive_conventional,
        **{
            "threshold": 0.5,
            "Q_e": 0.5,
            "N_r": 1e-1,
        },
        **plot_options,
    )

    # Averaging vs Frames Quanta
    average_vs_frames(
        strategy_ll=[average, average_optimal_threshold],
        frames_ll=[2, 5, 10, 20, 100],
        phi_proj=phi_proj,
        phi_A=phi_A,
        t_exp=t_exp,
        **plot_options,
    )

    # Averaging vs Frames Naive
    average_vs_frames(
        strategy_ll=[average_conventional],
        frames_ll=[5, 10, 100],
        phi_proj=phi_proj,
        phi_A=phi_A,
        t_exp=t_exp,
        **plot_options,
        **{
            "threshold": 0.5,
            "Q_e": 0.5,
            "N_r": 1e-1,
        },
    )
