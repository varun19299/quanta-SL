"""
Compare precision frame encoding and decoding
"""
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from einops import repeat
from matplotlib import axes
from matplotlib import pyplot as plt
from nptyping import NDArray

from quanta_SL.ops.coding import minimum_hamming_distance
from quanta_SL.ops.coding import stripe_width_stats
from quanta_SL.utils.plotting import save_plot, plot_code_LUT

params = {
    "axes.labelsize": "large",
    "axes.titlesize": "x-large",
}
plt.rcParams.update(params)


"""
Phase-Mag plots
"""


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
    ax_ll[0].stem(t, signal, linefmt="r-", label="Signal")
    ax_ll[0].plot(t, signal, label="Interpolated", color="blue")
    ax_ll[0].set_xlabel("Time")
    ax_ll[0].set_yticks([])

    if isinstance(noise, np.ndarray):
        ax_ll[0].stem(t, noise, linefmt="y-", label="Noise")

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

    ax_ll[1].stem(freq, mag, linefmt="r-", label="Magnitude plot")
    ax_ll[1].plot(freq, mag, color="blue")
    ax_ll[1].set_xlabel(r"Frequency $\omega$")

    ax_ll[2].stem(freq, phase, linefmt="r-", label="Phase plot")
    ax_ll[2].set_xlabel(r"Frequency $\omega$")
    ax_ll[2].set_yticks([-np.pi, 0, np.pi])
    ax_ll[2].set_yticklabels(["$-\pi$", "0", "$\pi$"])

    for ax in ax_ll[:3]:
        ax.legend(loc="upper right")


def _plot_stripe(
    code_LUT: NDArray[int],
    name: str,
    padding_func: Callable = None,
    noisy_bits: int = 0,
    figsize: Tuple = (14, 14),
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
    plt.imshow(code_img, cmap="gray")

    # X axis
    plt.xlabel("Pixels")

    # Y axis
    yticks = np.arange(code_LUT.shape[1])
    plt.yticks(yticks, yticks + 1)
    plt.ylabel("Time / Frames")

    save_plot(savefig, show, fname=save_path / f"{name}.pdf")

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
        for e, code in enumerate(code_LUT):
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

            magnitude_phase_plot(code, ax_ll=ax_ll[e, :], **plot_kwargs)

            ax_ll[e, 0].set_ylabel(f"Pixel {e}")

        if _phase_decoding:
            ax_ll[0, 3].set_title("Decoded As")

    # Noiseless
    subplot_kwargs = dict(
        nrows=code_LUT.shape[0], ncols=3, figsize=figsize, sharex="col", sharey="col"
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
        noisy_bits_ll = [noisy_bits]
    else:
        noisy_bits_ll = noisy_bits

    for noisy_bits in noisy_bits_ll:
        print("\n")
        fig, ax_ll = plt.subplots(**subplot_kwargs)

        _phase_decoding(noisy_bits)

        fig.tight_layout()
        save_plot(
            savefig,
            show,
            fname=save_path / f"{name}-phase-plot[corrupted bits {noisy_bits}].pdf",
        )


def circular_shifted_sw_4(show: bool = False):
    num_pixels = 8
    pattern = [0] * (num_pixels // 2) + [1] * (num_pixels // 2)
    pattern = np.array(pattern)

    code_LUT = []
    for i in range(num_pixels):
        code_LUT.append(np.roll(pattern, i))

    code_LUT = np.stack(code_LUT, axis=-1).astype(int)

    def _phase_decode_func(phase):
        alpha = phase[1] * 4 / np.pi
        return (-2.5 - alpha) % 8

    _plot_stripe(
        code_LUT,
        name="circ-SW-4",
        noisy_bits=[1, 2, 3, 4],
        show=show,
        phase_decode_func=_phase_decode_func,
    )


def circular_shifted_sw_8(show: bool = False):
    num_pixels = 16
    pattern = [0] * (num_pixels // 2) + [1] * (num_pixels // 2)
    pattern = np.array(pattern)

    code_LUT = []
    for i in range(num_pixels):
        code_LUT.append(np.roll(pattern, i))

    code_LUT = np.stack(code_LUT, axis=-1).astype(int)

    def _phase_decode_func(phase):
        alpha = phase[1] * 8 / np.pi
        return (-4.5 - alpha) % 16

    _plot_stripe(
        code_LUT,
        name="circ-SW-8",
        noisy_bits=list(range(1, 8)),
        show=show,
        figsize=(14, 28),
        phase_decode_func=_phase_decode_func,
    )


if __name__ == "__main__":
    circular_shifted_sw_4(show=False)
    circular_shifted_sw_8(show=False)
