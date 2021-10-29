from pathlib import Path

from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm
from quanta_SL.io import load_swiss_spad_bin

import cv2

logger.disable("quanta_SL")

bin_range = range(0, 3)
bin_offset = 0
capture_date = "29th_October"
folder_name = "gray_stripe_2pm"


folder = Path(f"outputs/real_captures/DLP_projector/{capture_date}/{folder_name}/")
out_folder = folder / "binary_frames"
out_folder.mkdir(exist_ok=True, parents=True)

cumulative_frame = 0
for bin_suffix in tqdm(bin_range):
    burst = load_swiss_spad_bin(folder, bin_suffix=bin_suffix)

    for img in burst:
        # plt.imshow(img, cmap="gray")
        # plt.title(f"Frame {cumulative_frame}")
        # plt.show()

        cv2.imwrite(str(out_folder / f"frame_{cumulative_frame}.jpg"), img * 255)

        cumulative_frame += 1
