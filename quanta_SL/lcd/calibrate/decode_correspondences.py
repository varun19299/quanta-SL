from functools import partial
from pathlib import Path

import hydra
import numpy as np
from dotmap import DotMap
from hydra.utils import get_original_cwd
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from scipy.io import savemat

from quanta_SL.decode.methods import gray_stripe_decoding
from quanta_SL.encode import strategies
from quanta_SL.encode.message import (
    registry as message_registry,
)
from quanta_SL.io import load_swiss_spad_sequence
from quanta_SL.utils.memoize import MemoizeNumpy
from quanta_SL.utils.plotting import save_plot, ax_imshow_with_colorbar

from quanta_SL.lcd.decode_helper import decode_2d_code

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
    cfg.gray_stripe.binary_frame.bin_suffix_range = eval(
        cfg.gray_stripe.binary_frame.bin_suffix_range
    )
    cfg.pose.range = eval(cfg.pose.range)

    lcd_captures_folder = Path(
        f"{get_original_cwd()}/outputs/real_captures/LCD_projector"
    )
    cfg.gray_stripe.folder = lcd_captures_folder / cfg.gray_stripe.folder

    # Get from
    cfg.gray_stripe.message_mapping = message_registry[cfg.gray_stripe.message_mapping]

    return cfg


def get_code_LUT_decoding_func(cfg):
    """
    Generate GrayStripe coding LUT, decoding func
    :param cfg:
    :return:
    """
    logger.info(f"Generating Code LUT")

    code_LUT = strategies.gray_stripe_code_LUT(**cfg.gray_stripe)

    gray_message_bits = cfg.gray_stripe.gray_message_bits

    decoding_func = partial(
        gray_stripe_decoding,
        gray_message_bits=gray_message_bits,
    )
    return code_LUT, decoding_func


@MemoizeNumpy
def get_frame(
    pattern_folder,
    bin_suffix_range: range = range(0, 10),
    comp_bin_suffix_range: range = [],
    **kwargs,
):
    # Groundtruth
    pos_frame = load_swiss_spad_sequence(
        pattern_folder,
        bin_suffix_range=bin_suffix_range,
        **kwargs,
    )

    if len(comp_bin_suffix_range):
        # Complementary
        neg_frame = load_swiss_spad_sequence(
            pattern_folder,
            bin_suffix_range=comp_bin_suffix_range,
            **kwargs,
        )

        return pos_frame > neg_frame

    return pos_frame


def get_sequence(cfg, code_LUT, pose_index: int, rotate_180: bool = False):
    col_sequence = []
    row_sequence = []

    bin_suffix_offset = 0
    pattern_range = range(1, code_LUT.shape[1] + 1)

    # First frame is all white
    pose_folder = (
        cfg.gray_stripe.folder / f"{cfg.gray_stripe.subfolder}{pose_index:02d}"
    )
    logger.info(f"All White")
    img = get_frame(
        pose_folder, bin_suffix_range=cfg.gray_stripe.binary_frame.bin_suffix_range
    )
    bin_suffix_offset += cfg.gray_stripe.binary_frame.bursts_per_pattern

    # Column encoding
    for pattern_index in pattern_range:
        logger.info(f"Column GrayStripe #{pattern_index}/{len(pattern_range)}")

        # Filename with leading zeros
        bin_suffix_range = (
            np.array(cfg.gray_stripe.binary_frame.bin_suffix_range) + bin_suffix_offset
        )
        bin_suffix_offset += cfg.gray_stripe.binary_frame.bursts_per_pattern

        comp_bin_suffix_range = (
            np.array(cfg.gray_stripe.binary_frame.bin_suffix_range) + bin_suffix_offset
        )
        bin_suffix_offset += cfg.gray_stripe.binary_frame.bursts_per_pattern

        col_frame = get_frame(pose_folder, bin_suffix_range, comp_bin_suffix_range)
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
        bin_suffix_range = (
            np.array(cfg.gray_stripe.binary_frame.bin_suffix_range) + bin_suffix_offset
        )
        bin_suffix_offset += cfg.gray_stripe.binary_frame.bursts_per_pattern

        comp_bin_suffix_range = (
            np.array(cfg.gray_stripe.binary_frame.bin_suffix_range) + bin_suffix_offset
        )
        bin_suffix_offset += cfg.gray_stripe.binary_frame.bursts_per_pattern

        row_frame = get_frame(pose_folder, bin_suffix_range, comp_bin_suffix_range)
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

    if rotate_180:
        img = img[::-1, ::-1]
        col_sequence = col_sequence[:, ::-1, ::-1]
        row_sequence = row_sequence[:, ::-1, ::-1]

    return img, col_sequence, row_sequence


@hydra.main(
    config_path="../../conf/lcd/calibrate",
    config_name=Path(__file__).stem,
)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    cfg = setup_args(cfg)

    code_LUT, decoding_func = get_code_LUT_decoding_func(cfg)

    img_ll = []
    col_correspondence_ll = []
    row_correspondence_ll = []

    for pose_index in cfg.pose.range:
        logger.info(f"Pose{pose_index:02d}")
        img, col_sequence, row_sequence = get_sequence(
            cfg, code_LUT, pose_index, cfg.gray_stripe.rotate_180
        )

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
        ax_imshow_with_colorbar(col_correspondence, ax, fig, vmin=0, vmax=1920)
        ax.set_title("Column Correspondences")

        ax = ax_ll[2]
        ax_imshow_with_colorbar(row_correspondence, ax, fig, vmin=0, vmax=1080)
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
