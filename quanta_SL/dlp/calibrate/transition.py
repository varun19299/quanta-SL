from pathlib import Path

import cv2

from quanta_SL.io import load_swiss_spad_bin
from loguru import logger
from tqdm import tqdm

logger.disable("quanta_SL")

pose_index_range = range(41, 46)

bin_suffix_range = range(0, 30)
capture_date = "29th_October"

for pose_index in pose_index_range:
    folder = Path(
        f"outputs/real_captures/DLP_Projector/{capture_date}/calibration/pose{pose_index:02d}/"
    )

    transition_folder = folder / "transition"
    transition_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Pose {pose_index}")
    for i in tqdm(bin_suffix_range):
        burst = load_swiss_spad_bin(folder, bin_suffix=i).mean(axis=0)

        path = transition_folder / f"frame-{i}.jpg"
        cv2.imwrite(str(path), burst * 255.0)
