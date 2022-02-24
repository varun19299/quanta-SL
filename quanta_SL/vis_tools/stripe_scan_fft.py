"""
Noise tolerance when using stripe scanning

"""
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple, Iterable

import numpy as np
from einops import repeat
from matplotlib import axes, pyplot as plt

from quanta_SL.decode.methods import square_wave_phase_unwrap
from quanta_SL.encode.precision_bits import circular_shifted_stripes
from quanta_SL.ops.coding import minimum_hamming_distance, stripe_width_stats
from quanta_SL.utils.plotting import plot_code_LUT, save_plot
from nptyping import NDArray

params = {
    "axes.labelsize": "xx-large",
    "axes.titlesize": "x-large",
    "font.family": "Times New Roman"
}
plt.rcParams.update(params)


def magnitude_phase_plot(
    signal: NDArray[int],
    t: NDArray[float] = None,
    ax_ll: NDArray[axes.Axes] = None,
    noise: NDArray[int] = None,
    phase_decode_func: Callable = None,
):
    if not isinstance(t, np.ndarray):
        t = np.arange(len(signal))

    if not isinstance(ax_ll, np.ndarray):
        fig, ax_ll = plt.subplots(3, figsize=(8, 12))

    # Plot signal
    stem_line_fmt = "k-"
    marker_fmt = "ko"
    basefmt = "0.3"
    color="gray"
    ax_ll[0].stem(t, signal, linefmt=stem_line_fmt, markerfmt=marker_fmt,basefmt=basefmt) #, label="Signal")
    # ax_ll[0].plot(t, signal, color=color) #label="Interpolated",
    ax_ll[0].set_xlabel("Time")
    ax_ll[0].set_ylabel(r"Pixel Value")
    ax_ll[0].set_yticks([])

    # if isinstance(noise, np.ndarray):
    #     ax_ll[0].stem(t, noise, linefmt="y-", label="Noise")

    square_f = np.fft.fft(signal)

    freq = np.fft.fftfreq(t.shape[0], d=t[1] - t[0])
    freq = np.fft.fftshift(freq)

    # Magnitude
    mag = np.abs(square_f)
    mag = np.fft.fftshift(mag)

    # Phase
    phase = np.angle(square_f)

    # Print phase stats
    print_str = [f"Slope: {phase[1]:.5g}"]

    if phase_decode_func:
        print_str.append(f"Decoded as: {phase_decode_func(phase):.3g}")

        if len(ax_ll) == 4:
            ax_ll[3].text(
                0.5,
                0.5,
                f"{phase_decode_func(phase):.2g}",
                size=32,
                verticalalignment="center",
                horizontalalignment="center",
            )
            ax_ll[3].axis("off")

    print(" | ".join(print_str))

    phase = np.fft.fftshift(phase)

    ax_ll[1].stem(freq, mag, linefmt=stem_line_fmt, markerfmt=marker_fmt, basefmt=basefmt) #, label="Magnitude plot")
    ax_ll[1].plot(freq, mag, color=color)
    ax_ll[1].set_ylabel(r"Magnitude")
    ax_ll[1].set_xlabel(r"Frequency $\omega$")

    ax_ll[2].stem(freq, phase, linefmt=stem_line_fmt, markerfmt=marker_fmt,basefmt=basefmt) #, label="Phase plot")
    ax_ll[2].set_ylabel(r"Phase")
    ax_ll[2].set_xlabel(r"Frequency $\omega$")
    ax_ll[2].set_yticks([-np.pi, 0, np.pi])
    ax_ll[2].set_yticklabels(["$-\pi$", "0", "$\pi$"])

    # Remove splines
    for i in range(3):
        ax_ll[i].spines['right'].set_visible(False)
        ax_ll[i].spines['top'].set_visible(False)

    # for ax in ax_ll[:3]:
    #     ax.legend(loc="upper right")


def _plot_stripe(
    code_LUT: NDArray[int],
    name: str,
    padding_func: Callable = None,
    noisy_bits: List = [],
    figsize: Tuple = (14, 2),
    savefig: bool = True,
    show: bool = True,
    **plot_kwargs,
):
    save_path = Path("outputs/code_images/precision/")

    print(
        f"Minimum Hamming Distance of {name}: \t {minimum_hamming_distance(code_LUT)}"
    )
    print(
        f"Stripe width stats (min-SW, framewise min, framewise mean): \t {stripe_width_stats(code_LUT)}"
    )

    code_img = plot_code_LUT(code_LUT, num_repeat=1, show=False)
    # plt.imshow(code_img, cmap="gray")
    #
    # # X axis
    # plt.xlabel("Pixels")
    #
    # # Y axis
    # yticks = np.arange(code_LUT.shape[1])
    # plt.yticks(yticks, yticks + 1)
    # plt.ylabel("Time / Frames")
    #
    # save_plot(savefig, show, fname=save_path / f"{name}.pdf")

    # Repeat to cover 64 columns
    code_img_64 = repeat(
        code_LUT, "N c -> (repeat N) c", repeat=pow(2, 6) // code_LUT.shape[0]
    )
    code_img_64 = repeat(
        code_img_64, "N c -> (N N_repeat) (c c_repeat)", N_repeat=10, c_repeat=10
    )
    plot_code_LUT(
        code_img_64, num_repeat=2, show=False, fname=save_path / f"{name}[64].png"
    )

    # Repeat to cover 1024 columns
    code_img_1024 = repeat(
        code_LUT, "N c -> (repeat N) c", repeat=pow(2, 10) // code_LUT.shape[0]
    )
    plot_code_LUT(
        code_img_1024, num_repeat=32, show=False, fname=save_path / f"{name}[1024].png"
    )

    def _phase_decoding(corrupt_bits: int = 0):
        for e, code in enumerate(code_LUT[2:3]):
            noise = np.zeros_like(code_LUT[0])
            indices = np.random.choice(
                np.arange(noise.size), replace=False, size=corrupt_bits
            )
            noise[indices] = 1

            # Corrupt
            print(f"Pixel {e}, noise pattern: {noise}")
            code = code ^ noise

            # Pad
            if padding_func:
                code = padding_func(code)

            magnitude_phase_plot(code, ax_ll=ax_ll[:], **plot_kwargs)

        #     ax_ll[e, 0].set_ylabel(f"Pixel {e}")
        #
        # if _phase_decoding:
        #     ax_ll[0, 3].set_title("Decoded As")

    # Noiseless
    subplot_kwargs = dict(
        nrows=code_LUT[2:3].shape[0], ncols=3, figsize=figsize, sharex="col", sharey="col"
    )
    if _phase_decoding:
        subplot_kwargs["ncols"] = 4

    fig, ax_ll = plt.subplots(**subplot_kwargs)

    _phase_decoding(0)

    fig.tight_layout()
    save_plot(
        savefig,
        show,
        fname=save_path / f"{name}-phase-plot.pdf",
    )

    # Noisy
    if not isinstance(noisy_bits, Iterable):
        if noisy_bits:
            noisy_bits_ll = [noisy_bits]
    else:
        noisy_bits_ll = noisy_bits

    for noisy_bits in noisy_bits_ll[:3]:
        print("\n")
        fig, ax_ll = plt.subplots(**subplot_kwargs)

        _phase_decoding(noisy_bits)

        fig.tight_layout()
        save_plot(
            savefig,
            show,
            fname=save_path / f"{name}-phase-plot[corrupted bits {noisy_bits}].pdf",
        )


def plot_circular_shifted(stripe_width: int = 8, show: bool = False):
    """
    Plot phase decoding for circular shifted stripes

    :param stripe_width: Width of rectangle (1 or 0). Period = 2 x width
    :param show: Display in window (per corrupted bit)
    """
    code_LUT = circular_shifted_stripes(stripe_width)

    _phase_decode_func = partial(square_wave_phase_unwrap, stripe_width=stripe_width)

    _plot_stripe(
        code_LUT,
        name=f"circ-SW-{stripe_width}",
        noisy_bits=range(1, stripe_width + 1),
        show=show,
        phase_decode_func=_phase_decode_func,
    )


if __name__ == "__main__":
    for stripe_width in [8]:
        plot_circular_shifted(stripe_width, show=True)
