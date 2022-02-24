from pathlib import Path
from roipoly import RoiPoly
from loguru import logger
from quanta_SL.io import load_swiss_spad_sequence
from matplotlib import pyplot as plt
from quanta_SL.lcd.acquire.reconstruct import inpaint_func
from scipy.io import loadmat
import numpy as np


def get_mask(
    mat_file,
    mat_key: str = "bpIndices",
    height: int = 256,
    width: int = 512,
    order: str = "col-major",
    rotate_180: bool = True,
):
    mat_file = loadmat(mat_file)
    bp_indices = mat_file[mat_key]

    mask = np.zeros((height, width), dtype=np.uint8)

    assert order in ["col-major", "row-major"]

    if order == "col-major":
        i_ll = (bp_indices - 1) % height
        j_ll = bp_indices // height
    else:
        j_ll = (bp_indices - 1) % width
        i_ll = bp_indices // width

    mask[i_ll, j_ll] = 1

    if rotate_180:
        mask = mask[::-1, ::-1]

    return mask


scene = "mary_bust_work_light_mini"
method = "Gray Code [11 bits] comp"
capture_folder = Path(f"outputs/real_captures/LCD_projector/30th_September/")

folder = capture_folder / f"{scene}/{method}"

inpaint_mat_path = capture_folder / "calibration/bpIndex_3000_0.7_8.mat"
inpaint_mask = get_mask(inpaint_mat_path)

bin_suffix_range = range(16, 20)
t_exp = 1e-4

img = load_swiss_spad_sequence(folder, bin_suffix_range=bin_suffix_range)[::-1, ::-1]

# Inpaint
img = inpaint_func(img, inpaint_mask)

logger.info("Select white region")
plt.imshow(img)
my_roi = RoiPoly(color="r")
my_roi.display_roi()
white_mask = my_roi.get_mask(img)

logger.info("Select black region")

plt.imshow(img)
my_roi = RoiPoly(color="r")
my_roi.display_roi()
black_mask = my_roi.get_mask(img)

# Averaged
prob_flip_bright = 1 - img[white_mask].mean()
prob_flip_dark = img[black_mask].mean()
num_bits = 10
gray_code_error_prob = 1 - pow(1 - (prob_flip_dark + prob_flip_bright) / 2, num_bits)

logger.info(
    f"Scene {scene} | P_flip_bright {prob_flip_bright:.3f} | P_flip_dark {prob_flip_dark:.3f} | 10-bit gray code error {gray_code_error_prob:3f}"
)
# plt.imshow(img)
# plt.show()
