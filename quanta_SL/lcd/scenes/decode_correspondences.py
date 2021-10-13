from functools import partial
from pathlib import Path

import hydra
import numpy as np
from dotmap import DotMap
from einops import repeat
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from roipoly import RoiPoly
from sklearn.metrics import median_absolute_error, mean_squared_error

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

# Disable inner logging
from quanta_SL.lcd.decode_helper import decode_2d_code
from quanta_SL.ops.binary import packbits_strided
from quanta_SL.utils.memoize import MemoizeNumpy
from quanta_SL.utils.plotting import save_plot, ax_imshow_with_colorbar

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

    # Setup args
    for method_key in cfg.methods.keys():
        method_cfg = cfg.methods[method_key]

        # Evaluate tuples
        if "hybrid" in method_key:
            method_cfg.bch_tuple = metaclass.BCH(*method_cfg.bch_tuple.values())
        elif "repetition" in method_key:
            method_cfg.repetition_tuple = metaclass.Repetition(
                *method_cfg.repetition_tuple.values()
            )

        # Convert range string to python range
        method_cfg.binary_frame.bin_suffix_range = eval(
            method_cfg.binary_frame.bin_suffix_range
        )

        # Obtain message vector from registry
        method_cfg.message_mapping = message_registry[method_cfg.message_mapping]

    return cfg


def get_code_LUT_decoding_func(method_cfg, method: str = "hybrid"):
    # Generate hybrid LUT, decoding func
    logger.info(f"Generating {method} LUT")

    if method == "hybrid":
        hybrid_code_LUT = strategies.hybrid_code_LUT(**method_cfg)
        bch_subset = strategies.bch_code_LUT(
            **{**method_cfg, "message_bits": method_cfg.bch_message_bits}
        )
        index = faiss_flat_index(packbits_strided(bch_subset))
        hybrid_decoding_func = partial(
            hybrid_decoding,
            func=faiss_minimum_distance,
            bch_tuple=method_cfg.bch_tuple,
            bch_message_bits=method_cfg.bch_message_bits,
            overlap_bits=method_cfg.overlap_bits,
            index=index,
            pack=True,
        )
        return (
            hybrid_code_LUT,
            hybrid_decoding_func,
        )

    elif method == "repetition":

        repetition_code_LUT = strategies.repetition_code_LUT(**method_cfg)
        message_ll = method_cfg.message_mapping(method_cfg.message_bits)

        repetition_decoding_func = partial(
            repetition_decoding,
            num_repeat=method_cfg.repetition_tuple.repeat,
            inverse_permuation=message_to_inverse_permuation(message_ll),
        )
        return (
            repetition_code_LUT,
            repetition_decoding_func,
        )


@MemoizeNumpy
def get_gt_frame(
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


def get_binary_gt_sequence(method_cfg, cfg, code_LUT):
    """
    Get Groundtruth (averaged and thresholded via complementary)
    and
    Binary frames for a strategy

    :param method_cfg: method conf
    :param cfg: Overall conf
    :param code_LUT: Look Up Table for code
    :return:
    """
    gt_sequence = []
    binary_sequence = []

    bin_suffix_offset = 0
    pattern_range = range(1, code_LUT.shape[1] + 1)

    # For repetition, just sample frames multiple times
    if method_cfg.multi_sample:
        pattern_range = range(1, method_cfg.message_bits + 1)

    # First frame is all white
    logger.info(f"All White")

    folder = method_cfg.folder
    img = get_gt_frame(
        folder, bin_suffix_range=method_cfg.binary_frame.bin_suffix_range
    )
    bin_suffix_offset += method_cfg.binary_frame.bursts_per_pattern

    for pattern_index in pattern_range:
        logger.info(f"Sequence #{pattern_index}/{len(pattern_range)}")

        # Single binary sample
        binary_frame_ll = load_swiss_spad_bin(
            folder,
            bin_suffix=method_cfg.binary_frame.bin_suffix + bin_suffix_offset,
            num_rows=256,
            num_cols=512,
        )

        # Pick multiple samples
        if method_cfg.multi_sample:
            indices = np.random.choice(
                len(binary_frame_ll), size=method_cfg.repetition_tuple.repeat
            )
            binary_frame_ll = binary_frame_ll[indices]
            binary_frame = binary_frame_ll[0]
            binary_sequence += [binary_frame for binary_frame in binary_frame_ll]
        else:
            binary_frame = binary_frame_ll[method_cfg.binary_frame.index]
            binary_sequence.append(binary_frame)

        # Filename with leading zeros
        bin_suffix_range = (
            np.array(method_cfg.binary_frame.bin_suffix_range) + bin_suffix_offset
        )
        bin_suffix_offset += method_cfg.binary_frame.bursts_per_pattern

        comp_bin_suffix_range = (
            np.array(method_cfg.binary_frame.bin_suffix_range) + bin_suffix_offset
        )
        bin_suffix_offset += method_cfg.binary_frame.bursts_per_pattern

        # Groundtruth Frame
        gt_frame = get_gt_frame(folder, bin_suffix_range, comp_bin_suffix_range)
        gt_sequence.append(gt_frame)

        if method_cfg.visualize_frame:
            plt.imshow(gt_frame, cmap="gray")
            save_plot(
                savefig=cfg.savefig,
                show=cfg.show,
                fname=f"{cfg.outfolder}/decoded_correspondences/{method_cfg.name}/gt_frame{pattern_index:02d}.pdf",
            )
            plt.imshow(binary_frame, cmap="gray")
            save_plot(
                savefig=cfg.savefig,
                show=cfg.show,
                fname=f"{cfg.outfolder}/decoded_correspondences/{method_cfg.name}/binary_frame{pattern_index:02d}.pdf",
            )

    # Stack
    gt_sequence = np.stack(gt_sequence, axis=0)
    binary_sequence = np.stack(binary_sequence, axis=0)

    # Repeat GT
    if method_cfg.multi_sample:
        gt_sequence = repeat(
            gt_sequence,
            "n r c -> (n repeat) r c",
            repeat=method_cfg.repetition_tuple.repeat,
        )

    if method_cfg.rotate_180:
        img = img[::-1, ::-1]
        gt_sequence = gt_sequence[:, ::-1, ::-1]
        binary_sequence = binary_sequence[:, ::-1, ::-1]

    return img, gt_sequence, binary_sequence


@hydra.main(config_path="../../conf/lcd/scenes", config_name=Path(__file__).stem)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    cfg = setup_args(cfg)

    methods_dict = {}

    for method_key in cfg.methods.keys():
        method_cfg = cfg.methods[method_key]
        if "hybrid" in method_key:
            code_LUT, decoding_func = get_code_LUT_decoding_func(method_cfg, "hybrid")

        elif "repetition" in method_key:
            code_LUT, decoding_func = get_code_LUT_decoding_func(
                method_cfg, "repetition"
            )

        img, gt_sequence, binary_sequence = get_binary_gt_sequence(
            method_cfg, cfg, code_LUT
        )

        logger.info(f"Decoding {method_key} GT")
        gt_decoded = decode_2d_code(gt_sequence, code_LUT, decoding_func)

        logger.info(f"Decoding {method_key} binary")
        binary_decoded = decode_2d_code(binary_sequence, code_LUT, decoding_func)

        methods_dict[method_key] = {
            "code_LUT": code_LUT,
            "decoding_func": decoding_func,
            "img": img,
            "gt_sequence": gt_sequence,
            "binary_sequence": binary_sequence,
            "gt_decoded": gt_decoded,
            "binary_decoded": binary_decoded,
        }

    # Regions to ignore
    # Custom RoI
    mask_path = Path(cfg.outfolder) / "roi_mask.npy"
    if mask_path.exists():
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

    logger.info("Evaluating Accuracy")

    # Find hybrid method with greatest redundancy for GT
    hybrid_key = max(
        [method_key for method_key in methods_dict],
        key=lambda s: int(s.split("_")[-1] if "hybrid" in s else 0),
    )

    gt_decoded = methods_dict[hybrid_key]["gt_decoded"]

    for method_key in methods_dict:
        binary_decoded = methods_dict[method_key]["binary_decoded"]
        rmse = mean_squared_error(gt_decoded[mask], binary_decoded[mask], squared=False)
        logger.info(f"RMSE {method_key} {rmse}")

        mae = median_absolute_error(gt_decoded[mask], binary_decoded[mask])
        logger.info(f"MAE {method_key} {mae}")

        logger.info("Plotting...")

        # Plotting
        n_row = 2
        n_col = 2
        fig, ax_ll = plt.subplots(
            n_row, n_col, sharey=True, sharex=True, figsize=(12, 12)
        )

        # Row 1: Groundtruths, Mask
        ax = ax_ll[0, 0]
        ax_imshow_with_colorbar(gt_decoded, ax, fig, cmap="gray")
        ax.set_title(
            f"Groundtruth est. from {hybrid_key.replace('_',' ').capitalize()}"
        )

        ax = ax_ll[0, 1]
        ax_imshow_with_colorbar(mask, ax, fig, cmap="jet")
        ax.set_title("Mask (exclude shadow + hot pixels)")

        # Row 2: Correspondences, Absolute error map
        ax = ax_ll[1, 0]
        ax_imshow_with_colorbar(binary_decoded, ax, fig)
        ax.set_title(f"Method Correspondences")

        ax = ax_ll[1, 1]
        abs_error_map = np.abs(gt_decoded - binary_decoded)
        abs_error_map[~mask] = 0
        ax_imshow_with_colorbar(abs_error_map, ax, fig)
        ax.set_title(f"Absolute Error Map | MAE {mae:.2f} | RMSE {rmse:.2f} ")

        title = rf"{method_key.replace('_', ' ').capitalize()} | {cfg.capture_date.replace('_', ' ')}"
        plt.suptitle(title, fontsize=28, y=0.775)
        plt.tight_layout()
        fig.subplots_adjust(hspace=-0.65)

        save_plot(
            savefig=cfg.savefig,
            show=cfg.show,
            fname=f"{cfg.outfolder}/results/{method_key}.pdf",
        )


if __name__ == "__main__":
    main()
