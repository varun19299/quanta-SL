from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_code(
    pattern_name: str = "MaxMinSWGray", ignore_complement: bool = True
) -> np.ndarray:
    """
    Load code from projector patterns

    :param pattern_name: Folder in data/patterns
    :param ignore_complement: Skip even frames
    :return: Code Look-Up-Table
    """
    path = Path("data/patterns") / pattern_name

    frame_ll = []
    path_ll = list(path.glob("*.exr"))
    for e, file in enumerate(tqdm(path_ll, desc=f"Loading {path}")):
        if ignore_complement and (e % 2 == 0):
            continue
        img = cv2.imread(str(file), 0)
        frame_ll.append(img)

    frame_ll = np.stack(frame_ll, axis=-1)

    # Ignore height
    code_LUT = frame_ll[0]
    return code_LUT
