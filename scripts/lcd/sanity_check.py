from functools import partial
from pathlib import Path

import hydra
import numpy as np
from dotmap import DotMap
from einops import rearrange, repeat
from hydra.utils import get_original_cwd
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from quanta_SL.decode.methods import hybrid_decoding, repetition_decoding
from quanta_SL.decode.minimum_distance.factory import (
    faiss_flat_index,
    faiss_minimum_distance,
)
from quanta_SL.encode import metaclass
from quanta_SL.encode import strategies
from quanta_SL.encode.message import (
    registry as message_registry,
    message_to_inverse_permuation,
)
from quanta_SL.io import load_swiss_spad_sequence, load_swiss_spad_bin
from quanta_SL.ops.binary import packbits_strided
from quanta_SL.utils.memoize import MemoizeNumpy
from quanta_SL.utils.plotting import save_plot, ax_imshow_with_colorbar

# Disable inner logging
logger.disable("quanta_SL")
logger.add(f"logs/lcd_{Path(__file__).stem}{{time}}.log", rotation="daily", retention=3)

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
    cfg.hybrid.bch_tuple = metaclass.BCH(*cfg.hybrid.bch_tuple)
    cfg.repetition.repetition_tuple = metaclass.Repetition(
        *cfg.repetition.repetition_tuple
    )
    cfg.groundtruth.bin_suffix_range = eval(cfg.groundtruth.bin_suffix_range)

    lcd_captures_folder = Path(
        f"{get_original_cwd()}/outputs/real_captures/LCD_projector"
    )
    cfg.groundtruth.folder = lcd_captures_folder / cfg.groundtruth.folder
    cfg.hybrid.folder = lcd_captures_folder / cfg.hybrid.folder
    cfg.repetition.folder = lcd_captures_folder / cfg.repetition.folder

    # Get from
    cfg.hybrid.message_mapping = message_registry[cfg.hybrid.message_mapping]
    cfg.repetition.message_mapping = message_registry[cfg.repetition.message_mapping]

    return cfg


def get_code_LUT_decoding_func(cfg, method: str = "hybrid"):
    # Generate hybrid LUT, decoding func
    logger.info(f"Generating {method} LUT")

    if method == "hybrid":
        hybrid_code_LUT = strategies.hybrid_code_LUT(**cfg.hybrid)
        bch_subset = strategies.bch_code_LUT(
            **{**cfg.hybrid, "message_bits": cfg.hybrid.bch_message_bits}
        )
        index = faiss_flat_index(packbits_strided(bch_subset))
        hybrid_decoding_func = partial(
            hybrid_decoding,
            func=faiss_minimum_distance,
            bch_tuple=cfg.hybrid.bch_tuple,
            bch_message_bits=cfg.hybrid.bch_message_bits,
            overlap_bits=cfg.hybrid.overlap_bits,
            index=index,
            pack=True,
        )
        return (
            hybrid_code_LUT,
            hybrid_decoding_func,
        )

    elif method == "repetition":

        repetition_code_LUT = strategies.repetition_code_LUT(**cfg.repetition)
        message_ll = cfg.repetition.message_mapping(cfg.repetition.message_bits)

        repetition_decoding_func = partial(
            repetition_decoding,
            num_repeat=cfg.repetition.repetition_tuple.repeat,
            inverse_permuation=message_to_inverse_permuation(message_ll),
        )
        return (
            repetition_code_LUT,
            repetition_decoding_func,
        )


@MemoizeNumpy
def get_gt_frame(pattern_folder, pattern_comp_folder, bin_suffix_range, **kwargs):
    # Groundtruth
    pos_frame = load_swiss_spad_sequence(
        pattern_folder,
        bin_suffix_range=bin_suffix_range,
        **kwargs,
    )

    # Complementary
    neg_frame = load_swiss_spad_sequence(
        pattern_comp_folder,
        bin_suffix_range=bin_suffix_range,
        **kwargs,
    )

    return pos_frame > neg_frame


def get_hybrid_binary_gt_sequence(cfg, hybrid_code_LUT, rotate_180: bool = False):
    gt_sequence = []
    binary_sequence = []

    pattern_range = range(1, hybrid_code_LUT.shape[1] + 1)
    for pattern_index in pattern_range:
        logger.info(f"Sequence #{pattern_index}/{len(pattern_range)}")

        # Filename with leading zeros
        pattern_folder = cfg.hybrid.folder / f"pattern{2 * pattern_index - 1:03d}"
        pattern_comp_folder = cfg.hybrid.folder / f"pattern{2 * pattern_index:03d}"

        gt_sequence.append(
            get_gt_frame(
                pattern_folder, pattern_comp_folder, cfg.groundtruth.bin_suffix_range
            )
        )

        # Single binary sample
        binary_frame = load_swiss_spad_bin(
            pattern_folder,
            bin_suffix=cfg.hybrid.binary_frame.bin_suffix,
            num_rows=256,
            num_cols=512,
        )
        binary_sequence.append(binary_frame[cfg.hybrid.binary_frame.index])

    # Stack
    gt_sequence = np.stack(gt_sequence, axis=0)
    binary_sequence = np.stack(binary_sequence, axis=0)

    if rotate_180:
        gt_sequence = gt_sequence[:, ::-1, ::-1]
        binary_sequence = binary_sequence[:, ::-1, ::-1]

    return gt_sequence, binary_sequence


def get_repetition_binary_gt_sequence(
    cfg, repetition_code_LUT, rotate_180: bool = False
):
    gt_sequence = []
    binary_sequence = []

    if cfg.repetition.multi_sample:
        pattern_range = range(1, cfg.repetition.message_bits + 1)
    else:
        pattern_range = range(1, repetition_code_LUT.shape[1] + 1)

    for pattern_index in pattern_range:
        logger.info(f"Sequence #{pattern_index}/{len(pattern_range)}")

        # Filename with leading zeros
        pattern_folder = cfg.repetition.folder / f"pattern{2 * pattern_index - 1:03d}"
        pattern_comp_folder = cfg.repetition.folder / f"pattern{2 * pattern_index:03d}"

        gt_sequence.append(
            get_gt_frame(
                pattern_folder, pattern_comp_folder, cfg.groundtruth.bin_suffix_range
            )
        )

        # Single binary sample
        binary_frame = load_swiss_spad_bin(
            pattern_folder,
            bin_suffix=cfg.repetition.binary_frame.bin_suffix,
            num_rows=256,
            num_cols=512,
        )

        # Pick multiple samples
        if cfg.repetition.multi_sample:
            binary_sequence.append(
                binary_frame[
                    cfg.repetition.binary_frame.index : cfg.repetition.binary_frame.index
                    + cfg.repetition.repetition_tuple.repeat
                ]
            )
        else:
            binary_sequence.append(
                binary_frame[
                    cfg.repetition.binary_frame.index : cfg.repetition.binary_frame.index
                    + 1
                ]
            )

    # Stack
    gt_sequence = np.stack(gt_sequence, axis=0)
    binary_sequence = np.concatenate(binary_sequence, axis=0)

    if rotate_180:
        gt_sequence = gt_sequence[:, ::-1, ::-1]
        binary_sequence = binary_sequence[:, ::-1, ::-1]

    if cfg.repetition.multi_sample:
        gt_sequence = repeat(
            gt_sequence,
            "n r c -> (n repeat) r c",
            repeat=cfg.repetition.repetition_tuple.repeat,
        )

    return gt_sequence, binary_sequence


def decode_2d_code(sequence_array, code_LUT, decoding_func):
    n, r, c = sequence_array.shape

    sequence_flat = rearrange(sequence_array, "n r c -> (r c) n")
    decoded_flat = decoding_func(sequence_flat, code_LUT)
    decoded_array = rearrange(decoded_flat, "(r c) -> r c", r=r, c=c)

    return decoded_array


@hydra.main(config_path="../../conf/scripts", config_name=f"lcd_{Path(__file__).stem}")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    cfg = setup_args(cfg)

    hybrid_code_LUT, hybrid_decoding_func = get_code_LUT_decoding_func(cfg, "hybrid")
    repetition_code_LUT, repetition_decoding_func = get_code_LUT_decoding_func(
        cfg, "repetition"
    )

    hybrid_gt_sequence, hybrid_binary_sequence = get_hybrid_binary_gt_sequence(
        cfg, hybrid_code_LUT, rotate_180=cfg.hybrid.rotate_180
    )

    (
        repetition_gt_sequence,
        repetition_binary_sequence,
    ) = get_repetition_binary_gt_sequence(
        cfg, repetition_code_LUT, rotate_180=cfg.repetition.rotate_180
    )

    # Regions to ignore
    # Ignore shadows and black areas, white pixels
    mean_hybrid_binary = hybrid_binary_sequence.mean(axis=0)
    mask = (mean_hybrid_binary < 0.2) | (mean_hybrid_binary > 0.90)
    # plt.figure(figsize=(8, 6))
    # plt.imshow(mean_hybrid_binary)
    # plt.colorbar()
    # plt.show()

    # Decode
    logger.info("Decoding Hybrid GT")
    hybrid_gt_decoded = decode_2d_code(
        hybrid_gt_sequence, hybrid_code_LUT, hybrid_decoding_func
    )

    logger.info("Decoding Hybrid Binary")
    hybrid_binary_decoded = decode_2d_code(
        hybrid_binary_sequence, hybrid_code_LUT, hybrid_decoding_func
    )

    logger.info("Decoding Repetition GT")
    repetition_gt_decoded = decode_2d_code(
        repetition_gt_sequence, repetition_code_LUT, repetition_decoding_func
    )

    logger.info("Decoding Repetition Binary")
    repetition_binary_decoded = decode_2d_code(
        repetition_binary_sequence, repetition_code_LUT, repetition_decoding_func
    )

    logger.info("Evaluating Accuracy")

    hybrid_rmse = np.sqrt(
        ((hybrid_gt_decoded[~mask] - hybrid_binary_decoded[~mask]) ** 2).mean()
    )
    logger.info(f"RMSE Hybrid {hybrid_rmse}")

    repetition_rmse = np.sqrt(
        ((repetition_gt_decoded[~mask] - repetition_binary_decoded[~mask]) ** 2).mean()
    )
    logger.info(f"RMSE Repetition {repetition_rmse}")

    logger.info("Plotting...")

    # Plotting
    n_row = 2
    n_col = 3
    fig, ax_ll = plt.subplots(n_row, n_col, sharey=True, sharex=True, figsize=(16, 12))

    # Row 1: Groundtruths, Mask
    ax = ax_ll[0, 0]
    ax_imshow_with_colorbar(hybrid_gt_decoded, ax, fig, cmap="gray")
    ax.set_title("Groundtruth est. from Hybrid")

    ax = ax_ll[0, 1]
    ax_imshow_with_colorbar(repetition_gt_decoded, ax, fig, cmap="gray")
    ax.set_title("Groundtruth est. from Repeat")

    ax = ax_ll[0, 2]
    ax_imshow_with_colorbar(mask, ax, fig, cmap="jet")
    ax.set_title("Mask (exclude shadow + hot pixels)")

    # Row 2: Hybrid ABS, Repeat ABS, GT discrepancy
    hybrid_abs_error = np.abs(hybrid_gt_decoded - hybrid_binary_decoded)
    hybrid_abs_error[mask] = 0
    ax = ax_ll[1, 0]
    ax_imshow_with_colorbar(hybrid_abs_error, ax, fig)
    ax.set_title(f"Hybrid Absolute error | RMSE {hybrid_rmse:.2f}")

    repetition_abs_error = np.abs(repetition_gt_decoded - repetition_binary_decoded)
    repetition_abs_error[mask] = 0
    ax = ax_ll[1, 1]
    ax_imshow_with_colorbar(repetition_abs_error, ax, fig)
    ax.set_title(f"Repetition Absolute error | RMSE {repetition_rmse:.2f}")

    gt_discrepancy = np.abs(hybrid_gt_decoded - repetition_gt_decoded)
    gt_discrepancy[mask] = 0
    ax = ax_ll[1, 2]
    ax_imshow_with_colorbar(gt_discrepancy, ax, fig)
    ax.set_title("GT Discrepancy (Hybrid, Repetition)")

    plt.suptitle(
        f"LCD Captures from {cfg.capture_date.replace('_',' ')}", fontsize=28, y=0.75
    )
    plt.tight_layout()
    fig.subplots_adjust(hspace=-0.65)

    save_plot(savefig=cfg.savefig, show=cfg.show, fname=cfg.fname)


if __name__ == "__main__":
    main()
