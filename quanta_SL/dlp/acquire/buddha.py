from pathlib import Path

import cv2
import hydra
import numpy as np
from dotmap import DotMap
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from roipoly import RoiPoly
from scipy.signal import medfilt2d
from tqdm import tqdm
from math import floor

from quanta_SL.encode import metaclass
from quanta_SL.encode.message import (
    registry as message_registry,
)
from quanta_SL.io import load_swiss_spad_bin
from quanta_SL.lcd.acquire.decode_correspondences import get_code_LUT_decoding_func

# Disable inner logging
from quanta_SL.lcd.decode_helper import decode_2d_code
from quanta_SL.lcd.acquire.reconstruct import (
    get_intrinsic_extrinsic,
    inpaint_func,
    reconstruct_3d,
    stereo_setup,
)
from quanta_SL.utils.plotting import save_plot, plot_image_and_colorbar

logger.disable("quanta_SL")
logger.add(f"logs/lcd_scenes_{Path(__file__).stem}.log", rotation="daily", retention=3)

# plt.style.use(["science", "grid"])
params = {
    "legend.fontsize": "x-large",
    "figure.titlesize": "xx-large",
    "axes.labelsize": "xx-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
plt.rcParams.update(params)


def setup_args(cfg):
    """
    Performs config interpolation and str to python objects
    :param cfg:
    :return:
    """

    cfg = DotMap(OmegaConf.to_object(cfg))

    if "Hybrid" in cfg.method.name:
        cfg.method.bch_tuple = metaclass.BCH(*cfg.method.bch_tuple.values())
    elif "Gray Code" in cfg.method.name:
        cfg.method.repetition_tuple = metaclass.Repetition(
            *cfg.method.repetition_tuple.values()
        )

    cfg.method.message_mapping = message_registry[cfg.method.message_mapping]

    # Convert range string to python range
    cfg.frame_range = eval(cfg.frame_range)

    return cfg


def get_binary_sequence(cfg, code_LUT):
    """
    Get Binary frames for a strategy

    :param cfg: Overall conf
    :param code_LUT: Look Up Table for code
    :return:
    """

    frame_sequence = []

    frame_start = cfg.frame_start
    num_patterns = code_LUT.shape[1] + 1
    frame_end = frame_start + num_patterns * cfg.bursts_per_pattern - 1

    bin_suffix_start = frame_start // cfg.bursts_per_bin + 1
    bin_suffix_end = frame_end // cfg.bursts_per_bin

    bin_suffix_range = range(bin_suffix_start, bin_suffix_end + 1, 2)
    pbar = tqdm(bin_suffix_range)

    frames_to_add = 20

    for bin_suffix in pbar:
        pbar.set_description(f"Bin Suffix {bin_suffix}")
        binary_burst = load_swiss_spad_bin(cfg.method.folder, bin_suffix=bin_suffix)
        binary_frame = binary_burst[255: 255 + frames_to_add].sum(axis=0) > 0

        frame_sequence.append(binary_frame)

    frame_sequence = np.stack(frame_sequence, axis=0)

    # First is all-white
    logger.info(f"All White")
    img = frame_sequence.mean(axis=0)

    logger.info("Binary frames")
    binary_sequence = frame_sequence[1:]

    if cfg.method.visualize_frame:
        logger.info("Saving frames")
        plt.imshow(img, cmap="gray")
        save_plot(
            savefig=cfg.savefig,
            show=cfg.show,
            fname=f"frame_wise/mean_image.pdf",
        )
        cv2.imwrite(
            f"frame_wise/mean_image.png",
            img * 255,
        )

        pbar = tqdm(total=len(binary_sequence))
        for e, binary_frame in enumerate(binary_sequence):
            pbar.set_description(f"Saving binary frame {e}")
            plt.imshow(binary_frame, cmap="gray")
            save_plot(
                savefig=cfg.savefig,
                show=cfg.show,
                fname=f"frame_wise/binary_frame{e}.pdf",
            )
            cv2.imwrite(
                f"frame_wise/binary_frame{e}.png",
                binary_frame * 255,
            )
            pbar.update(1)

    if cfg.method.rotate_180:
        img = img[::-1, ::-1]
        binary_sequence = binary_sequence[:, ::-1, ::-1]

    return img, binary_sequence


@hydra.main(config_path="../../conf/dlp/acquire", config_name=Path(__file__).stem)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    cfg = setup_args(cfg)

    logger.info("Generating code LUT")
    if "Hybrid" in cfg.method.name:
        code_LUT, decoding_func = get_code_LUT_decoding_func(cfg.method, "hybrid")

    elif "Conv Gray" in cfg.method.name:
        code_LUT, decoding_func = get_code_LUT_decoding_func(cfg.method, "repetition")

    logger.info("Reading binary sequence")
    img, binary_sequence = get_binary_sequence(cfg, code_LUT)

    logger.info(f"Decoding {cfg.method.name}")
    binary_decoded = decode_2d_code(binary_sequence, code_LUT, decoding_func)

    # Median filter binary decoded
    logger.info(f"Median filter")
    binary_decoded = medfilt2d(binary_decoded, kernel_size=5)

    inpaint_mask = get_intrinsic_extrinsic.get_mask(cfg)
    img = inpaint_func(img, inpaint_mask)

    plt.imshow(img, cmap="gray")
    save_plot(
        savefig=cfg.savefig,
        show=cfg.show,
        fname=f"mean_image.pdf",
    )
    cv2.imwrite(
        f"mean_image.png",
        img * 255,
    )

    # Regions to ignore
    # Custom RoI
    mask_path = Path(cfg.mask_path)
    if mask_path.exists():
        logger.info(f"RoI Mask found at {mask_path}")
        mask = np.load(mask_path)

    else:
        plt.imshow(img)
        my_roi = RoiPoly(color="r")
        my_roi.display_roi()

        mask = my_roi.get_mask(img)

        # Ignore shadows and black areas, white pixels
        # Valid regions are 1
        mask_shadows = (img > 0.2) | (img < 0.90)
        mask = mask & mask_shadows

        np.save(mask_path, mask)

    # Save img
    # cv2.imwrite(cfg.img, (img * 255).astype(int))

    # Save mask
    plot_image_and_colorbar(
        mask,
        f"mask",
        show=cfg.show,
        savefig=cfg.savefig,
        cmap="jet",
    )

    # Save correspondences
    plot_image_and_colorbar(
        binary_decoded * mask,
        f"correspondences",
        show=cfg.show,
        savefig=cfg.savefig,
        title=f"Correspondences",
    )

    # Save binary and gt correspondences
    np.savez(f"correspondences.npz", binary_decoded=binary_decoded)

    # Reconstruction

    binary_decoded = inpaint_func(binary_decoded, inpaint_mask)

    # Obtain stereo calibration params
    logger.info("Loading stereo params")
    camera_matrix, projector_matrix, camera_pixels = stereo_setup(cfg)

    # 3d reconstruction
    cfg.show = True
    method_points_3d = reconstruct_3d(
        cfg,
        binary_decoded,
        camera_pixels,
        mask,
        camera_matrix,
        projector_matrix,
        img,
        depth_map_vmin=cfg.scene.get("depth_map_vmin"),
        depth_map_vmax=cfg.scene.get("depth_map_vmax"),
        fname=f"reconstruction",
    )


if __name__ == "__main__":
    main()
