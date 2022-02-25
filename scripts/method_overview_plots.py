"""
Coarse and precision correspondences
"""
from matplotlib import pyplot as plt
import numpy as np
from math import ceil, log2
from quanta_SL.ops.binary import unpackbits, packbits
from pathlib import Path
from loguru import logger

projector = "LCD_projector"
date = "30th_September"
object = "tyre"
method = "hybrid_255"
main_folder = Path("outputs/real_captures") / projector / date / object

method_folder = main_folder / "results" / method
correspondence_file = method_folder / "correspondences.npz"

# Load file, discretize, split
num_cols = 1920
bch_message_bits = 8
message_bits = ceil(log2(num_cols))
correspondence_array = np.load(correspondence_file)["binary_decoded"]
correspondence_array = np.clip(correspondence_array, 0, num_cols)
correspondence_int_array = np.round(correspondence_array)

roi_map = np.load(main_folder / "roi_mask.npy")

correspondence_binary_array = unpackbits(
    correspondence_int_array.astype(int), num_bits=message_bits
)
correspondence_bch_array = correspondence_binary_array[:, :, :bch_message_bits]
correspondence_precision_array = correspondence_binary_array[:, :, bch_message_bits:]
correspondence_precision_array = packbits(correspondence_precision_array)
breakpoint()
plt.imshow(correspondence_precision_array * roi_map)
plt.show()

breakpoint()
