from pathlib import Path

import cv2
from loguru import logger
from tqdm import tqdm

from quanta_SL.io import load_swiss_spad_bin

logger.disable("quanta_SL")

bin_range = range(115, 120)
bin_offset = 0
capture_date = "29th_October"
folder_name = "white_cloth/Hybrid BCH [255, 9] [gray_message]"

folder = Path(f"outputs/real_captures/DLP_projector/{capture_date}/{folder_name}/")
out_folder = folder / f"binary_frames{bin_range}"
out_folder.mkdir(exist_ok=True, parents=True)

cumulative_frame = bin_range.start * 512
for bin_suffix in tqdm(bin_range):
    burst = load_swiss_spad_bin(folder, bin_suffix=bin_suffix)

    for img in burst:
        cv2.imwrite(str(out_folder / f"frame_{cumulative_frame}.jpg"), img * 255)

        cumulative_frame += 1
