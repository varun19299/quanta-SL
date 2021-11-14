from pathlib import Path

import cv2
import hydra
import numpy as np
from dotmap import DotMap
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

from quanta_SL.io import load_swiss_spad_bin
from quanta_SL.lcd.acquire.decode_correspondences import get_code_LUT_decoding_func
from quanta_SL.lcd.acquire.reconstruct import (
    get_intrinsic_extrinsic,
    inpaint_func,
)
from quanta_SL.dlp.acquire.decode_correspondences import setup_args
from quanta_SL.utils.plotting import save_plot

# Disable inner logging
logger.disable("quanta_SL")

plt.style.use(["science", "grid"])
params = {
    "legend.fontsize": "x-large",
    "figure.titlesize": "xx-large",
    "axes.labelsize": "xx-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
plt.rcParams.update(params)


def get_all_white_frames(cfg, code_LUT):
    """
    Get Binary frames for a strategy

    :param cfg: Overall conf
    :param code_LUT: Look Up Table for code
    :return:
    """

    frame_start = cfg.frame_start
    num_frames = 78
    num_patterns = code_LUT.shape[1] + 2

    img_all_white = np.zeros((cfg.spad.height, cfg.spad.width))

    pbar = tqdm(range(0, num_frames))
    for i in pbar:
        frame_index = frame_start + num_patterns * cfg.bursts_per_pattern * i
        bin_suffix = frame_index // cfg.bursts_per_bin
        bin_suffix_modulous = frame_index % cfg.bursts_per_bin

        pbar.set_description(f"Frame {frame_index} | Bin Suffix {bin_suffix}")

        binary_frame = load_swiss_spad_bin(cfg.method.folder, bin_suffix=bin_suffix)[
            np.array(
                [
                    max(bin_suffix_modulous - 3, 0),
                    max(bin_suffix_modulous - 2, 0),
                    bin_suffix_modulous,
                    bin_suffix_modulous + 1,
                ]
            )
        ]
        img_all_white += binary_frame.mean(axis=0)

    img_all_white /= num_frames

    if cfg.method.rotate_180:
        img_all_white = img_all_white[::-1, ::-1]

    return img_all_white


@hydra.main(config_path="../../conf/dlp/acquire", config_name=Path(__file__).stem)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    cfg = setup_args(cfg)

    logger.info("Generating code LUT")
    if "Hybrid" in cfg.method.name:
        code_LUT, decoding_func = get_code_LUT_decoding_func(cfg.method, "hybrid")

    elif "Conv Gray" in cfg.method.name:
        code_LUT, decoding_func = get_code_LUT_decoding_func(cfg.method, "repetition")

    logger.info("Reading all white frames")
    img = get_all_white_frames(cfg, code_LUT)

    inpaint_mask = get_intrinsic_extrinsic.get_mask(cfg)
    img = inpaint_func(img, inpaint_mask)

    plt.imshow(img, cmap="gray")
    save_plot(
        savefig=cfg.savefig,
        show=cfg.show,
        fname=f"blur_img.pdf",
    )
    cv2.imwrite(
        f"blur_img.png",
        img * 255,
    )


if __name__ == "__main__":
    main()
