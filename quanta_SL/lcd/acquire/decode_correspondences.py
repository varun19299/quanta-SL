from functools import partial
from pathlib import Path

import cv2
import hydra
import numpy as np
from dotmap import DotMap
from einops import repeat, reduce
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from roipoly import RoiPoly
from scipy.signal import medfilt2d
from sklearn.metrics import median_absolute_error, mean_squared_error

from quanta_SL.decode.methods import (
    hybrid_decoding,
    repetition_decoding,
    read_off_decoding,
    minimum_distance_decoding,
)
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

    # Setup args
    for method_key in cfg.methods.keys():
        method_cfg = cfg.methods[method_key]

        # Evaluate tuples
        if ("hybrid" in method_key) or ("bch" in method_key):
            method_cfg.bch_tuple = metaclass.BCH(*method_cfg.bch_tuple.values())
        elif ("repetition" in method_key) or ("gray" in method_key):
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
    # Generate code LUT, decoding func

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

    elif method == "bch":
        bch_code_LUT = strategies.bch_code_LUT(**method_cfg)
        index = faiss_flat_index(packbits_strided(bch_code_LUT))
        bch_decoding_func = partial(
            minimum_distance_decoding,
            func=faiss_minimum_distance,
            index=index,
            pack=True,
        )
        return (
            bch_code_LUT,
            bch_decoding_func,
        )

    elif method == "no_coding":
        code_LUT = method_cfg.message_mapping(method_cfg.message_bits)

        decoding_func = partial(
            read_off_decoding,
            inverse_permuation=message_to_inverse_permuation(code_LUT),
        )
        return (
            code_LUT,
            decoding_func,
        )

    else:
        raise NotImplementedError


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


def get_binary_frame(
    folder: Path, bin_suffix: int, frames_to_add: int = 1, samples: int = 1, **kwargs
):
    kwargs = {**dict(num_rows=256, num_cols=512), **kwargs}
    binary_burst = load_swiss_spad_bin(folder, bin_suffix=bin_suffix, **kwargs)

    # Simulate a brighter source (& ambient consequently)
    clipped_length = (len(binary_burst) // frames_to_add) * frames_to_add
    binary_burst = binary_burst[:clipped_length]
    binary_burst = (
        reduce(binary_burst, "(n add) h w -> n h w", "sum", add=frames_to_add) > 0
    )
    indices = np.random.choice(len(binary_burst), size=samples)

    return binary_burst[indices]


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
        if not method_cfg.multi_sample:
            samples = 1
        else:
            if method_cfg.use_complementary:
                samples = method_cfg.repetition_tuple.repeat // 2
            else:
                samples = method_cfg.repetition_tuple.repeat

        binary_burst = get_binary_frame(
            folder,
            bin_suffix=method_cfg.binary_frame.bin_suffix + bin_suffix_offset,
            frames_to_add=cfg.scene.get("frames_to_add", 1),
            samples=samples,
        )

        if method_cfg.use_complementary:
            comp_burst = get_binary_frame(
                folder,
                bin_suffix=method_cfg.binary_frame.bin_suffix
                + bin_suffix_offset
                + method_cfg.binary_frame.bursts_per_pattern,
                frames_to_add=cfg.scene.get("frames_to_add", 1),
                samples=samples,
            )

            binary_burst = binary_burst.mean(axis=0) > comp_burst.mean(axis=0)
            binary_burst = repeat(
                binary_burst, "h w -> n h w", n=method_cfg.repetition_tuple.repeat
            )

        # Pick multiple samples
        binary_frame = binary_burst.mean(axis=0)
        binary_sequence += [frame for frame in binary_burst]

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
            method_folder = Path(
                f"{cfg.outfolder}/decoded_correspondences/{method_cfg.name}"
            )
            save_plot(
                savefig=cfg.savefig,
                show=cfg.show,
                fname=method_folder / f"gt_frame{pattern_index:02d}.pdf",
            )
            cv2.imwrite(
                str(method_folder / f"gt_frame{pattern_index:02d}.png"), gt_frame * 255
            )

            plt.imshow(binary_frame, cmap="gray")
            save_plot(
                savefig=cfg.savefig,
                show=cfg.show,
                fname=method_folder / f"binary_frame{pattern_index:02d}.pdf",
            )
            cv2.imwrite(
                str(method_folder / f"binary_frame{pattern_index:02d}.png"),
                binary_frame * 255,
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


@hydra.main(config_path="../../conf/lcd/acquire", config_name=Path(__file__).stem)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    cfg = setup_args(cfg)

    methods_dict = {}

    for method_key in cfg.methods.keys():
        method_cfg = cfg.methods[method_key]
        logger.info(f"Generating {method_key} code LUT, decoding func")
        if "hybrid" in method_key:
            code_LUT, decoding_func = get_code_LUT_decoding_func(method_cfg, "hybrid")

        elif "repetition" in method_key:
            code_LUT, decoding_func = get_code_LUT_decoding_func(
                method_cfg, "repetition"
            )

        elif "bch" in method_key:
            code_LUT, decoding_func = get_code_LUT_decoding_func(method_cfg, "bch")
        else:
            code_LUT, decoding_func = get_code_LUT_decoding_func(
                method_cfg, "no_coding"
            )

        img, gt_sequence, binary_sequence = get_binary_gt_sequence(
            method_cfg, cfg, code_LUT
        )

        logger.info(f"Decoding {method_key} GT")
        gt_decoded = decode_2d_code(gt_sequence, code_LUT, decoding_func)

        logger.info(f"Decoding {method_key} binary")
        binary_decoded = decode_2d_code(binary_sequence, code_LUT, decoding_func)

        if cfg.projector.crop_mode == "center":
            gt_decoded -= (code_LUT.shape[0] - cfg.projector.width) // 2
            binary_decoded -= (code_LUT.shape[0] - cfg.projector.width) // 2

        # Median filter binary decoded
        binary_decoded = medfilt2d(binary_decoded)

        methods_dict[method_key] = {
            "code_LUT": code_LUT,
            "decoding_func": decoding_func,
            "img": img,
            "gt_sequence": gt_sequence,
            "binary_sequence": binary_sequence,
            "gt_decoded": gt_decoded,
            "binary_decoded": binary_decoded,
        }

    # Find hybrid method with greatest redundancy for GT
    hybrid_key = max(
        [method_key for method_key in methods_dict],
        key=lambda s: int(s.split("_")[-1] if "hybrid" in s else 0),
    )

    gt_decoded = methods_dict[hybrid_key]["gt_decoded"]
    hybrid_binary_decoded = methods_dict[hybrid_key]["binary_decoded"]
    img = methods_dict[hybrid_key]["img"]

    # Ignore high error regions of hybrid_key
    high_error_hybrid = np.abs(gt_decoded - hybrid_binary_decoded) > 100

    # Regions to ignore
    # Custom RoI
    mask_path = Path(cfg.groundtruth.mask_path)
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

    # mask *= ~high_error_hybrid
    # np.save(mask_path.parent / f"{mask_path.stem}_hybrid_filtered.npy", mask)
    # cv2.imwrite(
    #     str(mask_path.parent / f"{mask_path.stem}_hybrid_filtered.png"),
    #     (mask * 255.0).astype(int),
    # )

    logger.info("Evaluating Accuracy")

    # Save mask, GT
    plot_image_and_colorbar(
        gt_decoded,
        f"{cfg.outfolder}/results/groundtruth",
        savefig=cfg.savefig,
        show=cfg.show,
    )
    plot_image_and_colorbar(
        mask,
        f"{cfg.outfolder}/results/mask",
        savefig=cfg.savefig,
        show=cfg.show,
        cmap="jet",
    )

    cv2.imwrite(cfg.groundtruth.img, (img * 255).astype(int))
    np.save(cfg.groundtruth.correspondences, gt_decoded)

    for method_key in methods_dict:
        binary_decoded = methods_dict[method_key]["binary_decoded"]

        # Correspondence error
        rmse = mean_squared_error(gt_decoded[mask], binary_decoded[mask], squared=False)
        logger.info(f"RMSE {method_key} {rmse}")

        mae = median_absolute_error(gt_decoded[mask], binary_decoded[mask])
        logger.info(f"MAE {method_key} {mae}")

        logger.info("Plotting...")

        # Correspondences
        plot_image_and_colorbar(
            binary_decoded * mask,
            f"{cfg.outfolder}/results/{method_key}/correspondences",
            savefig=cfg.savefig,
            show=cfg.show,
            title=f"Correspondences",
        )

        # Errors
        abs_error_map = np.abs(gt_decoded - binary_decoded)
        abs_error_map[~mask] = 0

        plot_image_and_colorbar(
            abs_error_map,
            f"{cfg.outfolder}/results/{method_key}/error_map",
            savefig=cfg.savefig,
            show=cfg.show,
            title=f"MAE {mae:.2f} | RMSE {rmse:.2f}",
        )

        # Save binary and gt correspondences
        np.savez(
            f"{cfg.outfolder}/results/{method_key}/correspondences.npz",
            binary_decoded=binary_decoded,
        )


if __name__ == "__main__":
    main()
