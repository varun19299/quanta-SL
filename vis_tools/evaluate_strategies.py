import numpy as np
from matplotlib import pyplot as plt, ticker
from typing import Callable, Dict
from scipy.special import comb, erf
from einops import rearrange, repeat
from nptyping import NDArray

from math import floor, ceil

from typing import Callable, List

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


def naive(phi_P: NDArray, phi_A: NDArray, t_exp: float, bits: int = 10):
    prob_y_1_x_1 = 1 - np.exp(-phi_P * t_exp)
    prob_y_0_x_0 = np.exp(-phi_A * t_exp)

    return 1 - (prob_y_0_x_0 / 2 + prob_y_1_x_1 / 2) ** bits


def naive_conventional(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    threshold: float = 1,
    Q_e: float = 1,
    N_r: float = 1e4,
    bits: int = 10,
):
    sigma_1 = np.sqrt(Q_e * phi_P * t_exp + N_r ** 2)
    mu_1 = Q_e * phi_P * t_exp
    frac_1 = (threshold - mu_1) / (sigma_1 * np.sqrt(2))
    prob_y_1_x_1 = 0.5 * (1 - erf(frac_1))

    sigma_0 = np.sqrt(Q_e * phi_A * t_exp + N_r ** 2)
    mu_0 = Q_e * phi_A * t_exp
    frac_0 = (threshold - mu_0) / (sigma_0 * np.sqrt(2))
    prob_y_0_x_0 = 0.5 * (1 + erf(frac_0))

    return 1 - (prob_y_0_x_0 / 2 + prob_y_1_x_1 / 2) ** bits


def average(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    num_frames: 10,
    threshold: int = None,
    bits: int = 10,
):
    if not isinstance(threshold, np.ndarray):
        threshold = np.array([ceil((num_frames + 1) / 2)])
        threshold = repeat(threshold, "1 -> h w 1", h=phi_P.shape[0], w=phi_P.shape[1])

    j_ll = np.arange(start=0, stop=num_frames + 1)
    comb_ll = rearrange(comb(num_frames, j_ll), "d -> 1 1 d")

    # 10_C_0, 10_C_1, ..., 10_C_5
    mask_ll = j_ll < threshold

    prob_naive_y_0_x_0 = np.exp(-phi_A * t_exp)

    prob_frame_y_1_x_0 = 1 - np.exp(-phi_A * t_exp / num_frames)
    prob_frame_y_0_x_0 = 1 - prob_frame_y_1_x_0

    # Conditioning
    dtype = prob_frame_y_0_x_0.dtype
    prob_frame_y_0_x_0 = np.maximum(prob_frame_y_0_x_0, np.finfo(dtype).eps)

    frac = prob_frame_y_1_x_0 / prob_frame_y_0_x_0
    frac = [frac ** j for j in j_ll]
    frac = np.stack(frac, axis=-1)
    mult = (frac * comb_ll * mask_ll).sum(axis=-1)
    prob_y_0_x_0 = prob_naive_y_0_x_0 * mult

    # 10_C_6, 10_C_7, ..., 10_C_10
    mask_ll = j_ll >= threshold

    prob_naive_y_0_x_1 = np.exp(-phi_P * t_exp)
    prob_frame_y_1_x_1 = 1 - np.exp(-phi_P * t_exp / num_frames)
    prob_frame_y_0_x_1 = 1 - prob_frame_y_1_x_1

    frac = prob_frame_y_0_x_1 / prob_frame_y_1_x_1
    frac = [frac ** (num_frames - j) for j in j_ll]
    frac = np.stack(frac, axis=-1)
    mult = (frac * comb_ll * mask_ll).sum(axis=-1)
    prob_y_1_x_1 = (prob_frame_y_1_x_1 ** num_frames) * mult

    return 1 - (prob_y_0_x_0 / 2 + prob_y_1_x_1 / 2) ** bits


def average_conventional(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    num_frames: 10,
    threshold: float = 1,
    Q_e: float = 1,
    N_r: float = 1e4,
    bits: int = 10,
):
    sigma_1 = np.sqrt(Q_e * phi_P * t_exp / num_frames + N_r ** 2)
    mu_1 = Q_e * phi_P * t_exp / num_frames
    frac_1 = (threshold - mu_1) / (sigma_1 * np.sqrt(2))
    prob_y_1_x_1 = 0.5 * (1 - erf(frac_1))

    sigma_0 = np.sqrt(Q_e * phi_A * t_exp / num_frames + N_r ** 2)
    mu_0 = Q_e * phi_A * t_exp / num_frames
    frac_0 = (threshold - mu_0) / (sigma_0 * np.sqrt(2))
    prob_y_0_x_0 = 0.5 * (1 + erf(frac_0))

    return 1 - (prob_y_0_x_0 / 2 + prob_y_1_x_1 / 2) ** bits


def optimal_threshold(
    phi_P,
    phi_A,
    t_exp: float,
    num_frames: 10,
) -> NDArray:
    """
    Determine optimal threshold for avg strategy
    (known phi_P, phi_A)
    """
    N_p = phi_P * t_exp
    N_a = phi_A * t_exp

    num = N_p - N_a

    p = 1 - np.exp(-N_p / num_frames)
    q = 1 - np.exp(-N_a / num_frames)
    denom = N_p - N_a + num_frames * np.log(p / q)

    tau = num / denom
    threshold = np.ceil(tau * num_frames)
    return threshold, tau


def average_optimal_threshold(
    phi_P: NDArray,
    phi_A: NDArray,
    t_exp: float,
    num_frames: 10,
    bits: int = 10,
):
    threshold_ll, tau_ll = optimal_threshold(phi_P, phi_A, t_exp, num_frames)
    threshold_ll = rearrange(threshold_ll, "h w -> h w 1")
    return average(phi_P, phi_A, t_exp, num_frames, threshold_ll, bits)


def plot_optimal_threshold(
    phi_proj,
    phi_A,
    t_exp: float,
    num_frames: int = 10,
    savefig: bool = False,
    show: bool = True,
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

    plt.xlabel("$\Phi_p$")
    plt.ylabel(r"Threshold $(\frac{\tau^*}{N_\mathrm{frames}})$")
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()

    if savefig:
        plt.savefig(f"outputs/plots/optimal_thresh_vs_phi_p.pdf", dpi=150)

    if show:
        plt.show()

    # varying phi A, different lines for Phi P
    plt.figure(figsize=FIGSIZE)

    idx_P = np.round(np.linspace(0, len(phi_proj) - 1, 7)).astype(int)
    for p in phi_proj[idx_P]:
        thresh_ll, tau_ll = optimal_threshold(p + phi_A, phi_A, t_exp, num_frames)
        plt.semilogx(
            phi_A, tau_ll, label=f"$\Phi_p - \Phi_a=${p:.0e}", linewidth=LINEWIDTH
        )

    plt.xlabel("$\Phi_a$")
    plt.ylabel(r"Threshold $(\frac{\tau^*}{N_\mathrm{frames}})$")
    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()

    if savefig:
        plt.savefig(f"outputs/plots/optimal_thresh_vs_phi_A.pdf", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def plot_strategy(
    phi_proj,
    phi_A,
    t_exp: float,
    strategy: Callable,
    savefig: bool = False,
    outname: str = "",
    show: bool = True,
    **kwargs,
):
    print(f"Plotting strategy {outname if outname else strategy.__name__}")

    plt.figure()

    # Meshgrid
    phi_P_mesh, phi_A_mesh = np.meshgrid(phi_proj + phi_A, phi_A, indexing="ij")
    eval = strategy(phi_P_mesh, phi_A_mesh, t_exp, **kwargs)

    eval_image = plt.imshow(eval, cmap="plasma")
    plt.gca().invert_yaxis()

    # Label x axis
    idx_A = np.round(np.linspace(0, len(phi_A) - 1, order_range(phi_A))).astype(int)
    xticks = [f"{label:.0e}" for label in phi_A[idx_A]]
    plt.xticks(idx_A, xticks)
    plt.xlabel("$\Phi_a$ Ambient Flux", labelpad=5)

    # Label y axis
    idx_P = np.round(np.linspace(0, len(phi_proj) - 1, order_range(phi_proj))).astype(
        int
    )
    yticks = [f"{label:.0e}" for label in phi_proj[idx_P]]
    plt.yticks(idx_P, yticks)
    plt.ylabel("$\Phi_p - \Phi_a$ Projector Flux")

    plt.grid()
    plt.tight_layout()

    if not outname:
        outname = strategy.__name__

    if savefig:
        plt.savefig(f"outputs/plots/{outname}_no_colorbar.pdf", dpi=150)

    # Colorbar
    cbar = plt.colorbar()
    cbar.ax.set_title("P(error)")
    cbar.ax.locator_params(nbins=5)
    cbar.update_ticks()
    plt.clim(0, 1)
    plt.title(f"{outname.replace('_',' ').capitalize()}")
    plt.tight_layout()

    if savefig:
        plt.savefig(f"outputs/plots/{outname}_with_colorbar.pdf", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    # draw a new figure and replot the colorbar there
    fig, ax = plt.subplots()
    cbar = plt.colorbar(eval_image, ax=ax)
    cbar.ax.set_title("P(error)")
    cbar.ax.locator_params(nbins=5)
    cbar.update_ticks()
    ax.remove()

    if savefig:
        plt.savefig(f"outputs/plots/only_colorbar.pdf", dpi=150, bbox_inches="tight")

    plt.close()

    return eval


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
                outname=f"{strategy.__name__}_frames_{frames}",
                savefig=True,
                **{"num_frames": frames, **kwargs},
            )


if __name__ == "__main__":
    phi_proj = np.logspace(1, 8, num=512)
    phi_A = np.logspace(0, 5, num=512)

    # 1 millisecond
    t_exp = 1e-3

    SHOW = False

    plot_optimal_threshold(
        phi_proj,
        phi_A,
        t_exp=t_exp,
        num_frames=10,
        savefig=True,
        show=SHOW,
    )

    # Naive
    plot_strategy(
        phi_proj,
        phi_A,
        t_exp,
        naive,
        savefig=True,
        show=SHOW,
    )

    # Average
    plot_strategy(
        phi_proj,
        phi_A,
        t_exp,
        average,
        savefig=True,
        num_frames=10,
        show=SHOW,
    )

    # Average Optimal
    plot_strategy(
        phi_proj,
        phi_A,
        t_exp,
        average_optimal_threshold,
        savefig=True,
        show=SHOW,
        **{"num_frames": 10},
    )

    # Conventional Naive
    plot_strategy(
        phi_proj,
        phi_A,
        t_exp,
        naive_conventional,
        savefig=True,
        show=SHOW,
        **{
            "threshold": 0.5,
            "Q_e": 0.5,
            "N_r": 1e-1,
        },
    )

    # Averaging vs Frames Quanta
    average_vs_frames(
        strategy_ll=[average, average_optimal_threshold],
        frames_ll=[2, 5, 10, 20, 100],
        phi_proj=phi_proj,
        phi_A=phi_A,
        t_exp=t_exp,
        show=SHOW,
    )

    # Averaging vs Frames Naive
    average_vs_frames(
        strategy_ll=[average_conventional],
        frames_ll=[5, 10, 100],
        phi_proj=phi_proj,
        phi_A=phi_A,
        t_exp=t_exp,
        show=SHOW,
        **{
            "threshold": 0.5,
            "Q_e": 0.5,
            "N_r": 1e-1,
        },
    )
