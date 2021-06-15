import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from vis_tools.strategies.metaclass import Eval, CallableEval, BCH, Repetition
from vis_tools.strategies.analytic import (
    naive,
    naive_conventional,
    average_fixed,
    average_optimal,
    average_conventional,
    optimal_threshold,
)
from vis_tools.strategies.monte_carlo import bch_coding, repetition_coding
from vis_tools.strategies.utils import order_range, save_plot, func_name

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
    save_plot(
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
    save_plot(
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

    save_plot(
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
    save_plot(
        savefig,
        show,
        fname=f"outputs/plots/strategy_plots/{outname}/surface_plot_with_colorbar.pdf",
    )

    # draw a new figure and replot the colorbar there
    fig, ax = plt.subplots()
    surface_colorbar(fig, surf)
    ax.remove()
    save_plot(
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
    save_plot(
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
    save_plot(
        savefig,
        show,
        fname=f"outputs/plots/strategy_plots/{outname}/mesh_with_colorbar.pdf",
    )

    # draw a new figure and replot the colorbar there
    fig, ax = plt.subplots()
    img_colorbar(mappable=eval_image, ax=ax)
    ax.remove()
    save_plot(
        savefig,
        show=False,
        fname=f"outputs/plots/strategy_plots/{outname}/mesh_only_colorbar.pdf",
    )


def plot_strategy(
    phi_proj,
    phi_A,
    t_exp: float,
    strategy: Eval,
    savefig: bool = False,
    show: bool = True,
    plot_3d: bool = False,
    outname: str = "",
    **kwargs,
):
    # Outname
    if not outname:
        outname = strategy.name

    print(f"Plotting strategy {outname}")

    # Meshgrid
    phi_P_mesh, phi_A_mesh = np.meshgrid(phi_proj + phi_A, phi_A, indexing="ij")

    # Evaluate strategy
    eval_error = strategy(phi_P_mesh, phi_A_mesh, t_exp)

    if plot_3d:
        plot_strategy_3d(eval_error, phi_proj, phi_A, savefig, outname, show)

    plot_strategy_2d(eval_error, phi_proj, phi_A, savefig, outname, show)

    return eval_error


def plot_strategies_3d(
    phi_proj,
    phi_A,
    t_exp: float,
    strategy_ll,
    savefig: bool = False,
    show: bool = True,
    outname: str = "",
    **kwargs,
):
    names = [strategy.name for strategy in strategy_ll]
    print(f"Comparing strategies {','.join(names)}")

    # Meshgrid
    phi_P_mesh, phi_A_mesh = np.meshgrid(phi_proj + phi_A, phi_A, indexing="ij")
    phi_proj_mesh, phi_A_mesh = np.meshgrid(phi_proj, phi_A, indexing="ij")

    # Plot the surface.
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})

    limit_dict = {
        "naive": (np.s_[224:320, 192:320], np.s_[224:320], np.s_[256:320]),
        "avg_fixed": (np.s_[256:320, 400:], np.s_[280:320], np.s_[420:]),
        "avg_optimal": (np.s_[224:320, 192:320], np.s_[224:320], np.s_[256:320]),
    }

    for strategy in strategy_ll:
        assert isinstance(strategy, Eval)
        # Call the strategy and plot
        eval_error = strategy(phi_P_mesh, phi_A_mesh, t_exp)

        surf = ax.plot_surface(
            np.log10(phi_proj_mesh)[256:320, 400:],
            np.log10(phi_A_mesh)[256:320, 400:],
            eval_error[256:320, 400:],
            alpha=0.8,
            label=strategy.name,
            linewidth=0,
            antialiased=False,
        )

        ## TODO: Hackish, from https://stackoverflow.com/questions/55531760/is-there-a-way-to-label-multiple-3d-surfaces-in-matplotlib/55534939
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

    # X, Y axis
    ax.set_xlabel("\n$\Phi_p - \Phi_a$ Projector Flux")
    xticks = np.log10(phi_proj).astype(int)[224:320]
    xticklabels = [f"$10^{x}$" for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_ylabel("\n$\Phi_a$ Ambient Flux")
    yticks = np.log10(phi_A).astype(int)[420:]
    yticklabels = [f"$10^{y}$" for y in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # Z axis
    ax.set_zlabel("Error Probability")
    zticks = [p / 10 for p in range(0, 11, 2)]
    ax.set_zticks(zticks)
    ax.set_zticklabels([f"{z:.1f}" for z in zticks])

    ax.legend()
    plt.grid()

    # ax.view_init(0, azim=-90)

    save_plot(
        savefig,
        show,
        fname=f"outputs/plots/strategy_plots/{outname}/comparison_of_{'_'.join(names)}.pdf",
    )


if __name__ == "__main__":
    phi_proj = np.logspace(1, 8, num=64)
    phi_A = np.logspace(0, 5, num=64)

    # DMD framerate
    # 0.1 millisecond or 10^4 FPS
    t_exp = 1e-4

    plot_options = {
        "show": False,
        "plot_3d": True,
        "savefig": True,
    }

    avg_kwargs = {"num_frames": 10}
    conventional_sensor_kwargs = {"threshold": 0.5, "Q_e": 0.5, "N_r": 1e-1}

    bch_ll = [BCH(15, 11, 1), BCH(31, 11, 5), BCH(63, 10, 13), BCH(127, 15, 27)]
    # bch_ll = [BCH(15, 11, 1), BCH(31, 11, 7), BCH(63, 10, 18), BCH(127, 15, 40)]

    bch_kwargs_ll = [
        {"bch_tuple": bch_tuple, "num_frames": 1, "use_complementary": False}
        for bch_tuple in bch_ll
    ]

    repetition_ll = [
        Repetition(30, 10, 1),
        Repetition(60, 10, 2),
        Repetition(130, 10, 5),
    ]
    repetition_kwargs_ll = [
        {
            "repetition_tuple": repetition_tuple,
            "num_frames": 1,
            "use_complementary": False,
        }
        for repetition_tuple in repetition_ll
    ]

    func = naive
    strategy_ll = [CallableEval(func_name(func), func)]
    strategy_ll += [
        CallableEval(f"{func_name(func)}/{bch_tuple}", bch_coding, bch_kwargs)
        for bch_tuple, bch_kwargs in zip(bch_ll, bch_kwargs_ll)
    ]
    # strategy_ll += [
    #     CallableEval(
    #         f"{func_name(func)}/{repetition_tuple}",
    #         repetition_coding,
    #         repetition_kwargs,
    #     )
    #     for repetition_tuple, repetition_kwargs in zip(
    #         repetition_ll, repetition_kwargs_ll
    #     )
    # ]
    for strategy in strategy_ll:
        plot_strategy(
            phi_proj,
            phi_A,
            t_exp,
            strategy,
            **plot_options,
        )
    breakpoint()
    """
    strategy_ll = [CallableEval("Avg Fixed", average_fixed, avg_kwargs)]
    strategy_ll = [
        CallableEval("Avg Optimal", average_optimal, avg_kwargs)
    ]
    
    plot_strategies_3d(phi_proj, phi_A, t_exp, strategy_ll, **plot_options)
    """

    """
    plot_optimal_threshold(
        phi_proj,
        phi_A,
        t_exp=t_exp,
        num_frames=10,
        **plot_options,
    )

    strategy_ll = [
        CallableEval(func_name(naive), naive),
        CallableEval(func_name(average_fixed), average_fixed, avg_kwargs),
        CallableEval(func_name(average_optimal), average_optimal, avg_kwargs),
        CallableEval(
            func_name(naive_conventional),
            naive_conventional,
            conventional_sensor_kwargs,
        ),
    ]
    strategy_ll += [
        MatlabEval(
            f"{func_name(strategy)}/BCH {code}",
            f"BCH/eval_{strategy}_bch_{code[0]}_{code[1]}_texp_1e-04_128x128.mat",
        )
        for code in bch_ll
        for strategy in ["naive", "average_fixed", "average_optimal"]
    ]
    for strategy in strategy_ll:
        plot_strategy(
            phi_proj,
            phi_A,
            t_exp,
            strategy,
            **plot_options,
        )
    """

    # Averaging vs Frames Quanta
    num_frame_ll = [2, 5, 10, 20, 100]

    strategy_ll = [
        CallableEval(
            f"{func_name(strategy)}/frames_{num_frames}",
            strategy,
            avg_kwargs,
        )
        for strategy in [average_fixed, average_optimal]
        for num_frames in num_frame_ll
    ]
    strategy_ll += [
        CallableEval(
            f"{func_name(average_conventional)}/frames_{num_frames}",
            average_conventional,
            conventional_sensor_kwargs,
        )
        for num_frames in num_frame_ll
    ]

    for strategy in strategy_ll:
        plot_strategy(
            phi_proj,
            phi_A,
            t_exp,
            strategy,
            **plot_options,
        )
