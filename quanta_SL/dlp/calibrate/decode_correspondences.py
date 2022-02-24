from functools import partial
from pathlib import Path

import hydra
import numpy as np
from dotmap import DotMap
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from scipy.io import savemat

from quanta_SL.decode.methods import gray_stripe_decoding
from quanta_SL.encode import strategies
from quanta_SL.encode.message import (
    registry as message_registry,
)
from quanta_SL.lcd.calibrate.decode_correspondences import get_frame, get_code_LUT_decoding_func
from quanta_SL.lcd.decode_helper import decode_2d_code
from quanta_SL.utils.plotting import save_plot, ax_imshow_with_colorbar

# Disable inner logging
logger.disable("quanta_SL")
logger.add(
    f"logs/lcd_calibrate_{Path(__file__).stem}.log", rotation="daily", retention=3
)

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
    cfg = DotMap(OmegaConf.to_object(cfg))

    # Setup args
    cfg.poses.bin_suffix_range = eval(cfg.poses.bin_suffix_range)

    # Get from
    cfg.gray_stripe.message_mapping = message_registry[cfg.gray_stripe.message_mapping]

    return cfg


def get_sequence(cfg, code_LUT, pose_index: int):
    col_sequence = []
    row_sequence = []

    # Pose cfg
    pose_cfg = cfg.poses.get(f"pose{pose_index:02d}")

    bin_suffix_offset = pose_cfg.bin_suffix_offset
    bursts_per_pattern = cfg.poses.bursts_per_pattern
    bin_suffix_range = np.array(cfg.poses.bin_suffix_range)
    pattern_range = range(1, code_LUT.shape[1] + 1)

    # First frame is all white
    pose_folder = Path(cfg.outfolder) / f"pose{pose_index:02d}"
    logger.info(f"All White")
    img = get_frame(pose_folder, bin_suffix_range=bin_suffix_range + bin_suffix_offset)
    bin_suffix_offset += bursts_per_pattern

    # Column encoding
    for pattern_index in pattern_range:
        logger.info(f"Column GrayStripe #{pattern_index}/{len(pattern_range)}")

        # Filename with leading zeros
        frame_bin_suffix_range = bin_suffix_range + bin_suffix_offset
        bin_suffix_offset += bursts_per_pattern

        comp_bin_suffix_range = bin_suffix_range + bin_suffix_offset
        bin_suffix_offset += bursts_per_pattern

        col_frame = get_frame(
            pose_folder, frame_bin_suffix_range, comp_bin_suffix_range
        )
        col_sequence.append(col_frame)

        if cfg.gray_stripe.visualize_frame:
            plt.imshow(col_frame, cmap="gray")
            save_plot(
                savefig=cfg.savefig,
                show=cfg.show,
                fname=f"{cfg.outfolder}/decoded_correspondences/pose{pose_index:02d}/col_frame{pattern_index:02d}.pdf",
            )

    # Row encoding
    for pattern_index in pattern_range:
        logger.info(f"Row GrayStripe #{pattern_index}/{len(pattern_range)}")

        # Filename with leading zeros
        frame_bin_suffix_range = bin_suffix_range + bin_suffix_offset
        bin_suffix_offset += bursts_per_pattern

        comp_bin_suffix_range = bin_suffix_range + bin_suffix_offset
        bin_suffix_offset += bursts_per_pattern

        row_frame = get_frame(
            pose_folder, frame_bin_suffix_range, comp_bin_suffix_range
        )
        row_sequence.append(row_frame)

        if cfg.gray_stripe.visualize_frame:
            plt.imshow(row_frame, cmap="gray")
            save_plot(
                savefig=cfg.savefig,
                show=cfg.show,
                fname=f"{cfg.outfolder}/decoded_correspondences/pose{pose_index:02d}/row_frame{pattern_index:02d}.pdf",
            )

    # Stack
    col_sequence = np.stack(col_sequence, axis=0)
    row_sequence = np.stack(row_sequence, axis=0)

    if cfg.gray_stripe.rotate_180:
        img = img[::-1, ::-1]
        col_sequence = col_sequence[:, ::-1, ::-1]
        row_sequence = row_sequence[:, ::-1, ::-1]

    return img, col_sequence, row_sequence


@hydra.main(
    config_path="../../conf/dlp/calibrate",
    config_name=Path(__file__).stem,
)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    cfg = setup_args(cfg)

    code_LUT, decoding_func = get_code_LUT_decoding_func(cfg)

    img_ll = []
    col_correspondence_ll = []
    row_correspondence_ll = []

    pose_list = [
        int(pose_key[-2:]) for pose_key in cfg.poses if pose_key.startswith("pose")
    ]
    pose_list = sorted(pose_list)

    for pose_index in pose_list:
        logger.info(f"Pose{pose_index:02d}")

        img, col_sequence, row_sequence = get_sequence(cfg, code_LUT, pose_index)

        # Images
        img_ll.append(img)

        # Decode
        logger.info(f"Pose {pose_index} | Decoding Cols")
        col_correspondence = decode_2d_code(col_sequence, code_LUT, decoding_func)

        logger.info(f"Pose {pose_index} | Decoding Rows")
        row_correspondence = decode_2d_code(row_sequence, code_LUT, decoding_func)

        if cfg.projector.crop_mode == "center":
            col_correspondence -= (code_LUT.shape[0] - cfg.projector.width) // 2
            row_correspondence -= (code_LUT.shape[0] - cfg.projector.height) // 2

        col_correspondence = 1023 - col_correspondence
        col_correspondence_ll.append(col_correspondence)
        row_correspondence_ll.append(row_correspondence)

        # Plotting
        fig, ax_ll = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(16, 6))

        # All White image
        ax = ax_ll[0]
        ax_imshow_with_colorbar(img, ax, fig, cmap="gray")
        ax.set_title("All White Image")

        # Column correspondences
        ax = ax_ll[1]
        ax_imshow_with_colorbar(
            col_correspondence, ax, fig, vmin=0, vmax=cfg.projector.width
        )
        ax.set_title("Column Correspondences")

        ax = ax_ll[2]
        ax_imshow_with_colorbar(
            row_correspondence, ax, fig, vmin=0, vmax=cfg.projector.height
        )
        ax.set_title("Row Correspondences")

        plt.suptitle(
            f"Pose {pose_index} | Date {cfg.capture_date.replace('_', ' ')}",
            fontsize=28,
            y=0.75,
        )
        plt.tight_layout()
        fig.subplots_adjust(hspace=-0.65)

        save_plot(
            savefig=cfg.savefig,
            show=cfg.show,
            fname=f"{cfg.outfolder}/decoded_correspondences/pose{pose_index:02d}.pdf",
        )

    img_ll = np.stack(img_ll, axis=0)
    col_correspondence_ll = np.stack(col_correspondence_ll, axis=0)
    row_correspondence_ll = np.stack(row_correspondence_ll, axis=0)
    save_dict = {
        "all-white": img_ll,
        "col-correspondences": col_correspondence_ll,
        "row-correspondences": row_correspondence_ll,
    }

    savemat(f"{cfg.outfolder}/correspondence_data.mat", save_dict)


if __name__ == "__main__":
    main()
