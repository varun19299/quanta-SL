from pathlib import Path

import numpy as np
from dotmap import DotMap
from matplotlib import pyplot as plt

from quanta_SL.encode.message import gray_message
from quanta_SL.encode.metaclass import BCH
from quanta_SL.lcd.acquire.decode_correspondences import (
    get_code_LUT_decoding_func,
    decode_2d_code,
)
from quanta_SL.utils.plotting import save_plot

method_cfg = DotMap(
    {
        "bch_tuple": BCH(511, 10, 127),
        "bch_message_bits": 7,
        "message_bits": 10,
        "message_mapping": gray_message,
        "overlap_bits": 1,
    }
)
from quanta_SL.io import load_swiss_spad_bin
from tqdm import tqdm

# Code LUT, decoding func
hybrid_code_LUT, hybrid_decoding_func = get_code_LUT_decoding_func(
    method_cfg, method="hybrid"
)

capture_date = "29th_October"
folder_name = "buddha_bust"
folder = Path(f"outputs/real_captures/DLP_projector/{capture_date}/{folder_name}/")

frame_start = 490
bursts_per_bin = 512
bursts_per_pattern = 1024

num_patterns = hybrid_code_LUT.shape[1] + 1
frame_end = frame_start + num_patterns * bursts_per_pattern - 1

bin_suffix_start = frame_start // bursts_per_bin + 1
bin_start_modulous = frame_start % bursts_per_bin
bin_suffix_end = frame_end // bursts_per_bin + 1
bin_end_modulous = frame_end % bursts_per_bin

frames_to_add = 10

binary_sequence = []
for bin_suffix in tqdm(range(bin_suffix_start, bin_suffix_end + 1, 2)):
    binary_burst = load_swiss_spad_bin(folder, bin_suffix=bin_suffix)[255: 255 + frames_to_add]
    binary_frame = binary_burst[255: 255 + frames_to_add].sum(axis=0) > 0

    binary_sequence.append(binary_frame)

binary_sequence = np.concatenate(binary_sequence[1:], axis=0)

# Decode
binary_decoded = decode_2d_code(binary_sequence, hybrid_code_LUT, hybrid_decoding_func)

# Rotate 180
binary_decoded = binary_decoded[::-1, ::-1]

# Plot
plt.figure(figsize=(6, 6))
plt.imshow(binary_decoded)
plt.colorbar()
plt.title(f"Correspondences | Mary Bust | {capture_date.replace('_',' ')}")
save_plot(savefig=True, show=True, fname=folder / "correspondences.pdf")
