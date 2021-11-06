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


def get_binary_frame(folder: Path, bin_suffix: int, samples: int = 1, **kwargs):
    kwargs = {**dict(num_rows=256, num_cols=512), **kwargs}
    binary_burst = load_swiss_spad_bin(folder, bin_suffix=bin_suffix, **kwargs)

    indices = np.random.choice(len(binary_burst), size=samples)

    return binary_burst[indices]


def get_binary_sequence(cfg, code_LUT):
    """
    Get Binary frames for a strategy

    :param cfg: Overall conf
    :param code_LUT: Look Up Table for code
    :return:
    """

    frame_start = cfg.frame_start
    bin_suffix_start = frame_start // cfg.bursts_per_bin
    bin_start_modulous = frame_start % cfg.bursts_per_bin

    # Include all-white
    num_patterns = code_LUT.shape[1] + 1
    frame_end = frame_start + num_patterns * cfg.bursts_per_pattern - 1
    bin_suffix_end = frame_end // cfg.bursts_per_bin
    bin_end_modulous = frame_end % cfg.bursts_per_bin + 1

    bin_suffix_range = range(bin_suffix_start, bin_suffix_end + 1)
    pbar = tqdm(bin_suffix_range)

    frame_sequence = []

    for bin_suffix in pbar:
        pbar.set_description(f"Bin Suffix {bin_suffix}")
        binary_burst = load_swiss_spad_bin(cfg.method.folder, bin_suffix=bin_suffix)

        if bin_suffix == bin_suffix_start:
            if bin_suffix == bin_suffix_end:
                frame_sequence.append(binary_burst[bin_start_modulous:bin_end_modulous])
            else:
                frame_sequence.append(binary_burst[bin_start_modulous:])
        elif bin_suffix == bin_suffix_end:
            frame_sequence.append(binary_burst[:bin_end_modulous])
        else:
            frame_sequence.append(binary_burst)

    # Range adding
    frame_sequence = np.concatenate(frame_sequence, axis=0)
    post_adding_frame_sequence = np.zeros_like(frame_sequence)[:num_patterns]
    for j in cfg.frame_range:
        post_adding_frame_sequence += frame_sequence[j :: cfg.bursts_per_pattern]

    # Num photons > 0
    post_adding_frame_sequence = post_adding_frame_sequence > 0

    # First is all-white
    logger.info(f"All White")
    img = post_adding_frame_sequence.mean(axis=0)

    logger.info("Binary frames")
    binary_sequence = post_adding_frame_sequence[1:]

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

    if "Hybrid" in cfg.method.name:
        code_LUT, decoding_func = get_code_LUT_decoding_func(cfg.method, "hybrid")

    elif "Conv Gray" in cfg.method.name:
        code_LUT, decoding_func = get_code_LUT_decoding_func(cfg.method, "repetition")

    img, binary_sequence = get_binary_sequence(cfg, code_LUT)

    binary_decoded = decode_2d_code(binary_sequence, code_LUT, decoding_func)

    # Median filter binary decoded
    binary_decoded = medfilt2d(binary_decoded)

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
    inpaint_mask = get_intrinsic_extrinsic.get_mask(cfg)
    img = inpaint_func(img, inpaint_mask)

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
        fname=f"reconstruction",
    )


if __name__ == "__main__":
    main()
