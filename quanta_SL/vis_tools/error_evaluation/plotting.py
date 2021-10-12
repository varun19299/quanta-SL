from pathlib import Path
from typing import List, Union, Callable

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from matplotlib import pyplot as plt, cm
from nptyping import NDArray

from quanta_SL.encode.metaclass import Eval
from quanta_SL.ops.math_func import order_range
from quanta_SL.ops.metrics import exact_error
from quanta_SL.utils.plotting import save_plot
from quanta_SL.vis_tools.error_evaluation.analytic import optimal_threshold

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
    plot_dir: Path = Path("outputs/strategy_comparison/"),
    **unused_kwargs,
):
    logger.info("Plotting optimal threshold \n")
    FIGSIZE = (8, 4)

    # varying phi P, different lines for Phi A
    plt.figure(figsize=FIGSIZE)

    idx_A = np.round(np.linspace(0, len(phi_A) - 1, 4)).astype(int)
    for a in phi_A[idx_A]:
        # Phi_P = phi_proj + a
        tau_ll = optimal_threshold(phi_proj + a, a, t_exp, num_frames)
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
    save_plot(savefig, show, fname=plot_dir / "optimal_thresh_vs_phi_p.pdf")

    # varying phi A, different lines for Phi P
    plt.figure(figsize=FIGSIZE)

    idx_P = np.round(np.linspace(0, len(phi_proj) - 1, 7)).astype(int)
    for p in phi_proj[idx_P]:
        thresh_ll, tau_ll = optimal_threshold(p + phi_A, phi_A, t_exp, num_frames)
        plt.semilogx(
            phi_A, tau_ll, label=f"$\Phi_p - \Phi_a=${p:.0e}", linewidth=LINEWIDTH
        )

    _plt_properties("$\Phi_a$")
    save_plot(savefig, show, fname=plot_dir / "optimal_thresh_vs_phi_A.pdf")


def surface_plot_3d(
    eval_error,
    phi_proj,
    phi_A,
    error_metric: Callable = exact_error,
    savefig: bool = False,
    outname: str = "",
    show: bool = True,
    plot_dir: Path = Path("outputs/strategy_comparison/oversampling_benefit"),
    **unused_kwargs,
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
    ax.set_xlabel("\n$\Phi_p - \Phi_a$ Projector Flux")
    indices = np.round(np.linspace(0, len(phi_proj) - 1, 4)).astype(int)
    xticks = np.log10(phi_proj)[indices]
    xticklabels = [f"{label:.1e}" for label in phi_proj[indices]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_ylabel("\n$\Phi_a$ Ambient Flux")
    indices = np.round(np.linspace(0, len(phi_A) - 1, 4)).astype(int)
    yticks = np.log10(phi_A)[indices]
    yticklabels = [f"{label:.1e}" for label in phi_A[indices]]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # Z axis
    ax.set_zlabel(error_metric.long_name, labelpad=5)
    zticks = [p / 10 for p in range(0, 11, 2)]
    ax.set_zticks(zticks)
    ax.set_zticklabels([f"{z:.1f}" for z in zticks])

    save_plot(
        savefig,
        show=False,
        close=False,
        fname=plot_dir / f"{outname}/surface_plot_no_colorbar.pdf",
    )

    # Colorbar
    def surface_colorbar(fig, ax, mappable):
        cbar = fig.colorbar(mappable, pad=0.1, ax=ax)
        cbar.ax.set_title("P(error)")

    surface_colorbar(fig, ax, surf)
    plt.title(f"{outname.replace('_', ' ').capitalize()}")
    save_plot(
        savefig,
        show,
        fname=plot_dir / f"{outname}/surface_plot_with_colorbar.pdf",
    )

    # draw a new figure and replot the colorbar there
    fig, ax = plt.subplots()
    surface_colorbar(fig, ax, surf)
    ax.remove()
    save_plot(
        savefig,
        show=False,
        fname=plot_dir / f"{outname}/surface_plot_only_colorbar.pdf",
    )


def mesh_plot_2d(
    eval_error,
    phi_proj,
    phi_A,
    error_metric: Callable = exact_error,
    savefig: bool = False,
    outname: str = "",
    show: bool = True,
    plot_dir: Path = Path("outputs/strategy_comparison/oversampling_benefit"),
    **unused_kwargs,
):
    # Plot image
    plt.figure()
    eval_image = plt.imshow(eval_error, cmap="plasma")
    plt.gca().invert_yaxis()

    # Label x axis
    xticks = np.round(np.linspace(0, len(phi_A) - 1, order_range(phi_A))).astype(int)
    xticklabels = [f"{label:.1e}" for label in phi_A[xticks]]
    plt.xticks(xticks, xticklabels)
    plt.xlabel("$\Phi_a$ Ambient Flux", labelpad=5)

    # Label y axis
    yticks = np.round(np.linspace(0, len(phi_proj) - 1, order_range(phi_proj))).astype(
        int
    )
    yticklabels = [f"{label:.1e}" for label in phi_proj[yticks]]
    plt.yticks(yticks, yticklabels)
    plt.ylabel("$\Phi_p - \Phi_a$ Projector Flux")

    plt.grid()
    save_plot(
        savefig,
        show=False,
        close=False,
        fname=plot_dir / f"{outname}/mesh_no_colorbar.pdf",
    )

    # Colorbar
    def img_colorbar(**kwargs):
        cbar = plt.colorbar(**kwargs)
        cbar.ax.set_title(error_metric.name)
        cbar.ax.locator_params(nbins=5)
        cbar.update_ticks()

    img_colorbar()
    # plt.clim(0, 1)
    plt.title(f"{outname.replace('_',' ').capitalize()}", y=1.12)
    save_plot(
        savefig,
        show,
        fname=plot_dir / f"{outname}/mesh_with_colorbar.pdf",
    )

    # draw a new figure and replot the colorbar there
    fig, ax = plt.subplots()
    img_colorbar(mappable=eval_image, ax=ax)
    ax.remove()
    save_plot(
        savefig,
        show=False,
        fname=plot_dir / f"{outname}/mesh_only_colorbar.pdf",
    )


def mesh_and_surface_plot(
    phi_proj,
    phi_A,
    t_exp: float,
    strategy: Eval,
    eval_error: NDArray = None,
    error_metric: Callable = exact_error,
    savefig: bool = False,
    show: bool = True,
    plot_3d: bool = False,
    outname: str = "",
    plot_dir: Path = Path("outputs/strategy_comparison/oversampling_benefit"),
    **unused_kwargs,
):
    # Outname
    if not outname:
        outname = strategy.name

    logger.info(f"Individual plotting: strategy {outname}")

    # If not supplied, call
    if not isinstance(eval_error, np.ndarray):
        # Meshgrid
        phi_P_mesh, phi_A_mesh = np.meshgrid(phi_proj + phi_A, phi_A, indexing="ij")

        # Call the strategy and plot
        assert isinstance(strategy, Eval)
        eval_error = strategy(phi_P_mesh, phi_A_mesh, t_exp)

    kwargs = locals().copy()

    if plot_3d:
        surface_plot_3d(**kwargs)

    mesh_plot_2d(**kwargs)

    return eval_error


def multiple_surface_pyplot_3d(
    phi_proj,
    phi_A,
    t_exp: float,
    strategy_ll: List[Eval],
    eval_error_ll: List = [],
    error_metric: Callable = exact_error,
    savefig: bool = False,
    show: bool = True,
    outname: str = "",
    title: str = "",
    plot_dir: Path = Path("outputs/strategy_comparison/oversampling_benefit"),
    **unused_kwargs,
):
    names = [strategy.name for strategy in strategy_ll]
    logger.info(f"Comparative plotting: {', '.join(names)}")

    # Meshgrid
    phi_P_mesh, phi_A_mesh = np.meshgrid(phi_proj + phi_A, phi_A, indexing="ij")
    phi_proj_mesh, phi_A_mesh = np.meshgrid(phi_proj, phi_A, indexing="ij")

    # Plot the surface.
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})

    if eval_error_ll:
        assert len(eval_error_ll) == len(
            strategy_ll
        ), "Supplied eval errors must match # strategies"

    for e, strategy in enumerate(strategy_ll):
        assert isinstance(strategy, Eval)

        # If not supplied, call
        if not eval_error_ll:
            # Call the strategy and plot
            assert isinstance(strategy, Eval)
            eval_error = strategy(phi_P_mesh, phi_A_mesh, t_exp)
        else:
            eval_error = eval_error_ll[e]

        surf = ax.plot_surface(
            np.log10(phi_proj_mesh),
            np.log10(phi_A_mesh),
            np.round(eval_error, decimals=4),
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
    indices = np.round(np.linspace(0, len(phi_proj) - 1, 4)).astype(int)
    xticks = np.log10(phi_proj)[indices]
    xticklabels = [f"{label:.1e}" for label in phi_proj[indices]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_ylabel("\n$\Phi_a$ Ambient Flux")
    indices = np.round(np.linspace(0, len(phi_A) - 1, 4)).astype(int)
    yticks = np.log10(phi_A)[indices]
    yticklabels = [f"{label:.1e}" for label in phi_A[indices]]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # Z axis
    ax.set_zlabel(error_metric.long_name)
    # zticks = [p / 10 for p in range(0, 11, 2)]
    # ax.set_zticks(zticks)
    # ax.set_zticklabels([f"{z:.1f}" for z in zticks])

    ax.legend()
    plt.grid()

    if title:
        plt.title(title)

    save_plot(
        savefig,
        show,
        fname=plot_dir / f"{outname}/comparison_of_{'_'.join(names)}.pdf",
    )


def multiple_surface_plotly_3d(
    phi_proj,
    phi_A,
    t_exp: float,
    strategy_ll: List[Eval],
    eval_error_ll: List = [],
    error_metric: Callable = exact_error,
    savefig: bool = False,
    show: bool = True,
    outname: str = "",
    title: str = "",
    plot_dir: Path = Path("outputs/strategy_comparison/oversampling_benefit"),
    **unused_kwargs,
):

    names = [strategy.name for strategy in strategy_ll]
    logger.info(f"Comparative plotting: {', '.join(names)}")

    # Meshgrid
    phi_P_mesh, phi_A_mesh = np.meshgrid(phi_proj + phi_A, phi_A, indexing="ij")
    phi_proj_mesh, phi_A_mesh = np.meshgrid(phi_proj, phi_A, indexing="ij")

    COLORS_ll = ["red", "orange", "green", "blue", "purple", "brown", "grey"]
    COLORS_discrete = [[(0, color), (1, color)] for color in COLORS_ll]

    fig = go.Figure(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            autosize=False,
            width=900,
            height=900,
        )
    )

    if eval_error_ll:
        assert len(eval_error_ll) == len(
            strategy_ll
        ), "Supplied eval errors must match # strategies"

    for e, strategy in enumerate(strategy_ll):
        assert isinstance(strategy, Eval)

        # If not supplied, call
        if not eval_error_ll:
            # Call the strategy and plot
            assert isinstance(strategy, Eval)
            eval_error = strategy(phi_P_mesh, phi_A_mesh, t_exp)
        else:
            eval_error = eval_error_ll[e]

        fig.add_trace(
            go.Surface(
                x=phi_P_mesh,
                y=phi_A_mesh,
                z=np.round(eval_error, decimals=3),
                name=strategy.name,
                colorscale=COLORS_discrete[e],
            )
        )

    fig.update_layout(
        showlegend=True,
        title=dict(text=title, x=0.5, y=0.9, xanchor="center", yanchor="top"),
        scene=dict(
            xaxis=dict(
                title=r"Projector Flux",
                tickfont_size=12,
                dtick="D2",
                type="log",
                exponentformat="power",
            ),
            yaxis=dict(
                title=r"Ambient Flux",
                tickfont_size=12,
                dtick="D2",
                type="log",
                exponentformat="power",
            ),
            zaxis=dict(title=error_metric.long_name, tickfont_size=12),
        ),
        scene_aspectmode="cube",
        scene_camera_eye=dict(x=1.61, y=1.61, z=0.25),
        legend=dict(yanchor="top", y=0.9, xanchor="right", x=0.87),
        font=dict(size=18),
    )

    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)

    fig.update_traces(showlegend=True, showscale=False)

    if show:
        fig.show(renderer="browser")

    if savefig:
        out_path = plot_dir / f"{outname}/comparison_of_{'_'.join(names)}"
        out_path.parent.mkdir(exist_ok=True, parents=True)
        fig.write_image(str(out_path) + ".pdf", scale=4)
        fig.write_image(str(out_path) + ".png", scale=4)
        fig.write_html(str(out_path) + ".html", include_plotlyjs="cdn")


def individual_and_multiple_plots(
    phi_proj,
    phi_A,
    t_exp: float,
    strategy_ll: List[Eval],
    error_metric: Callable = exact_error,
    savefig: bool = False,
    show: bool = True,
    plot_3d: str = "",
    outname: str = "",
    title: str = "",
    backend_3d: str = "plotly",
    plot_dir: Path = Path("outputs/strategy_comparison/oversampling_benefit"),
    **kwargs,
):
    kwargs = locals().copy()
    eval_error_ll = []

    # Meshgrid
    phi_P_mesh, phi_A_mesh = np.meshgrid(phi_proj + phi_A, phi_A, indexing="ij")

    logger.info("Evaluating expected error...")

    for e, strategy in enumerate(strategy_ll):
        # Call the strategy and plot
        assert isinstance(strategy, Eval)
        eval_error = strategy(phi_P_mesh, phi_A_mesh, t_exp)

        eval_error_ll.append(eval_error)

    logger.info("\n")

    if backend_3d == "plotly":
        multiple_surface_plotly_3d(eval_error_ll=eval_error_ll, **kwargs)
    elif backend_3d == "matplotlib":
        multiple_surface_pyplot_3d(eval_error_ll=eval_error_ll, **kwargs)

    del kwargs["outname"]
    for strategy, eval_error in zip(strategy_ll, eval_error_ll):
        mesh_and_surface_plot(
            eval_error=eval_error,
            strategy=strategy,
            outname=f"{outname}/{strategy.name}",
            **kwargs,
        )


def func_name(func: Union[Callable, str]) -> str:
    if isinstance(func, Callable):
        func_str = func.__name__
    elif isinstance(func, str):
        func_str = func
    return func_str.replace("_", " ").title()
