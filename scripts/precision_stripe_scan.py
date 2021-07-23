"""
Compare precision frame encoding and decoding
"""
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes
from nptyping import NDArray
from numba import vectorize, float64

from ops.coding import minimum_hamming_distance
from utils.mapping import plot_code_LUT
from utils.plotting import save_plot

params = {
    "axes.labelsize": "large",
    "axes.titlesize": "x-large",
}
plt.rcParams.update(params)

# plt.style.use("science")


def periodically_continued(a: float, b: float) -> Callable:
    """
    Periodically continue a function
    :param a: Start point
    :param b: End point
    :return: Periodic function
    """

    interval = b - a
    return lambda f: lambda x: f((x - a) % interval + a)

"""
Functions
"""

@periodically_continued(-1, 1)
@vectorize([float64(float64)], nopython=True, target="parallel", cache=True)
def rect(t: float):
    # Discontinuous version
    # if abs(t) == 0.5:
    #     return 1
    if abs(t) > 0.5:
        return 0
    else:
        return 1


def discrete_square(n: int, N: int, W: int):
    """
    https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Shift_theorem
    :param n:
    :param N:
    :param W:
    :return:
    """
    mask = (2 * n < W) | (2 * (N - n) < W)
    out = np.zeros_like(n)

    out[mask] = 1

    return out

"""
Phase-Mag plots
"""

def magnitude_phase_plot(
    signal: NDArray[int],
    t: NDArray[float] = None,
    ax_ll: NDArray[axes.Axes] = None,
    noise: NDArray[int] = None,
    phase_unwrap_func: Callable = None,
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
    print_str = [f"Slope: {phase[1]:.3g}"]

    if phase_unwrap_func:
        print_str.append(f"Decoded as: {phase_unwrap_func(phase):.3g}")

    print(" | ".join(print_str))

    phase = np.fft.fftshift(phase)

    ax_ll[1].stem(freq, mag, linefmt="r-", label="Magnitude plot")
    ax_ll[1].plot(freq, mag, color="blue")
    ax_ll[1].set_xlabel(r"Frequency $\omega$")

    ax_ll[2].stem(freq, phase, linefmt="r-", label="Phase plot")
    ax_ll[2].set_xlabel(r"Frequency $\omega$")
    ax_ll[2].set_yticks([-np.pi, 0, np.pi])
    ax_ll[2].set_yticklabels(["$-\pi$", "0", "$\pi$"])

    for ax in ax_ll:
        ax.legend()



def rect_fft():
    num = 100
    t = np.linspace(-1, 1, num=num)

    for i in range(3):
        magnitude_phase_plot(rect(t - i), t)
        plt.show()
        plt.close()


def square_pulse_fft():
    N = 8
    W = 5
    n = np.arange(N)

    for i in range(8):
        magnitude_phase_plot(np.roll(discrete_square(n, N, W), i), n)
        plt.show()
        plt.close()


def _plot_stripe(
    code_LUT: NDArray[int],
    name: str,
    padding_func: Callable,
    noisy_bits: int = 0,
    savefig: bool = True,
    show: bool = True,
    **plot_kwargs,
):
    print(
        f"Minimum Hamming Distance of {name}: \t {minimum_hamming_distance(code_LUT)}"
    )

    code_img = plot_code_LUT(code_LUT, num_repeat=1, show=False)
    plt.imshow(code_img, cmap="gray")

    # X axis
    plt.xlabel("Pixels")

    # Y axis
    yticks = np.arange(8)
    plt.yticks(yticks, yticks + 1)
    plt.ylabel("Time / Frames")

    save_plot(savefig, show, fname=f"outputs/projector_frames/code_images/{name}.pdf")

    # Noiseless
    fig, ax_ll = plt.subplots(
        nrows=code_LUT.shape[0], ncols=3, figsize=(14, 14), sharex="col", sharey="col"
    )

    for e, code in enumerate(code_LUT):
        # Zero pad
        code = padding_func(code)

        magnitude_phase_plot(code, ax_ll=ax_ll[e, :], **plot_kwargs)

        ax_ll[e, 0].set_ylabel(f"Pixel {e}")

    fig.tight_layout()
    save_plot(
        savefig,
        show,
        fname=f"outputs/projector_frames/code_images/{name}-phase-plot.pdf",
    )

    # Noisy
    print("\n")
    fig, ax_ll = plt.subplots(
        nrows=code_LUT.shape[0], ncols=3, figsize=(14, 14), sharex="col", sharey="col"
    )

    for e, code in enumerate(code_LUT):
        noise = np.zeros_like(code_LUT[0])
        indices = np.random.choice(
            np.arange(noise.size), replace=False, size=noisy_bits
        )
        noise[indices] = 1

        # Corrupt
        print(f"Pixel {e}: {noise}")
        code = code ^ noise

        # Zero pad
        code = padding_func(code)

        magnitude_phase_plot(code, ax_ll=ax_ll[e, :], **plot_kwargs)

        ax_ll[e, 0].set_ylabel(f"Pixel {e}")

    fig.tight_layout()

    save_plot(
        savefig,
        show,
        fname=f"outputs/projector_frames/code_images/{name}-phase-plot[corrupted bits {noisy_bits}].pdf",
    )


def stripe_4():
    num_pixels = 8
    pattern = [0] * (num_pixels // 2) + [1] * (num_pixels // 2)
    pattern = np.array(pattern)

    code_LUT = []
    for i in range(num_pixels):
        code_LUT.append(np.roll(pattern, i))

    code_LUT = np.stack(code_LUT, axis=-1).astype(int)

    def _padding_func(code):
        return code
        code_l = code[:4]
        code_r = code[4:]
        padding = np.zeros(16, dtype=int)
        return np.concatenate([code_l, padding, code_r])

    def _phase_unwrap_func(phase):
        alpha = phase[1] * 4 / np.pi
        return (-2.5 - alpha) % 8

        return ((-112.5 - deg) % 360) / 45

    _plot_stripe(
        code_LUT,
        name="stripe-width-4",
        padding_func=_padding_func,
        noisy_bits=4,
        show=False,
        phase_unwrap_func=_phase_unwrap_func,
    )


def stripe_8():
    num_pixels = 8
    pattern = [0] * num_pixels + [1] * num_pixels + [0] * num_pixels
    pattern = np.array(pattern)

    # Compose LUT
    code_LUT = []
    for i in range(num_pixels):
        code_LUT.append(np.roll(pattern, i)[num_pixels:-num_pixels])

    code_LUT = np.stack(code_LUT, axis=-1).astype(int)

    def _padding_func(code):
        return np.pad(code, (0, 16))

    _plot_stripe(
        code_LUT, name="stripe-width-8", noisy_bits=1, padding_func=_padding_func
    )


if __name__ == "__main__":
    # square_pulse_fft()
    stripe_4()
    stripe_8()
