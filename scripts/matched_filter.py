"""
Matched filter approach using circular convolution
"""
import matplotlib.pyplot as plt

from quanta_SL.encode.precision_bits import circular_shifted_stripes
import numpy as np

stripe_width = pow(2, 3)
code_LUT = circular_shifted_stripes(stripe_width)

col_index = 2
# num_bitflips
num_bitflips = 1

for col_index in range(len(code_LUT)):
    code = code_LUT[col_index]

    # Random bit flips
    noise = np.zeros_like(code)
    indices = np.random.choice(np.arange(noise.size), replace=False, size=num_bitflips)
    noise[indices] = 1

    corrupt_code = code ^ noise

    # Template, reverse
    template = code_LUT[0][::-1]

    # Circular conv
    conv_out = np.fft.fft(corrupt_code) * np.fft.fft(template)
    conv_out = np.real(np.fft.ifft(conv_out))

    # Plot!
    fig, ax_ll = plt.subplots(4, 1)

    ax_ll[0].plot(code)
    ax_ll[0].set_title("Original code")

    ax_ll[1].plot(corrupt_code)
    ax_ll[1].set_title(f"Noisy code | #{num_bitflips} bitflips")

    ax_ll[2].plot(template)
    ax_ll[2].set_title("Template")

    decoded = (np.argmax(conv_out) + 1) % code_LUT.shape[1]
    ax_ll[3].plot(conv_out)
    ax_ll[3].set_title(f"Conv | Max at {np.argmax(conv_out)} | Decoded as {decoded}")
    plt.suptitle(f"Col index {col_index}")
    plt.tight_layout()
    plt.show()
