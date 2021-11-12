from pathlib import Path

import cv2
import hydra
import numpy as np
from dotmap import DotMap
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from scipy.signal import medfilt2d
from tqdm import tqdm
from scipy.signal import medfilt

from quanta_SL.encode import metaclass
from quanta_SL.encode.message import (
    registry as message_registry,
)
from quanta_SL.reconstruct.project3d import triangulate_ray_plane
from einops import repeat
from quanta_SL.io import load_swiss_spad_bin
from quanta_SL.lcd.acquire.decode_correspondences import get_code_LUT_decoding_func
from quanta_SL.lcd.acquire.reconstruct import (
    get_intrinsic_extrinsic,
    inpaint_func,
    stereo_setup,
)

# Disable inner logging
from quanta_SL.lcd.decode_helper import decode_2d_code
from quanta_SL.utils.plotting import save_plot, plot_image_and_colorbar

import imageio
import matplotlib

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


def get_binary_sequence(frame_start: int, cfg, code_LUT):
    """
    Get Binary frames for a strategy

    :param cfg: Overall conf
    :param code_LUT: Look Up Table for code
    :return:
    """

    bin_suffix_start = frame_start // cfg.bursts_per_bin
    bin_start_modulous = frame_start % cfg.bursts_per_bin

    # Include all-white
    num_patterns = code_LUT.shape[1] + 1
    frame_end = frame_start + num_patterns * cfg.bursts_per_pattern - 1
    bin_suffix_end = frame_end // cfg.bursts_per_bin
    bin_end_modulous = frame_end % cfg.bursts_per_bin + 1

    bin_suffix_range = range(bin_suffix_start, bin_suffix_end + 1)
    frame_sequence = []

    for bin_suffix in bin_suffix_range:
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
    img = post_adding_frame_sequence.mean(axis=0)
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

    logger.info("Generating code LUT")
    if "Hybrid" in cfg.method.name:
        code_LUT, decoding_func = get_code_LUT_decoding_func(cfg.method, "hybrid")

    elif "Conv Gray" in cfg.method.name:
        code_LUT, decoding_func = get_code_LUT_decoding_func(cfg.method, "repetition")

    num_video_frames = 62
    frame_gap = (code_LUT.shape[1] + 2) * cfg.bursts_per_pattern

    # Obtain stereo calibration params
    logger.info("Loading stereo params")
    camera_matrix, projector_matrix, camera_pixels = stereo_setup(cfg)

    valid_indices = np.where(np.ones((cfg.spad.height, cfg.spad.width)))
    camera_pixels = np.stack(
        [camera_pixels[0][valid_indices], camera_pixels[1][valid_indices]], axis=1
    )

    depth_map_ll = []

    for i in range(num_video_frames):
        frame_start = cfg.frame_start + i * frame_gap
        logger.info(
            f"Video frame {i} | Frame-start {frame_start}| Reading binary sequence"
        )
        img, binary_sequence = get_binary_sequence(frame_start, cfg, code_LUT)

        logger.info(
            f"Video frame {i} | Frame-start {frame_start}|Decoding {cfg.method.name}"
        )
        binary_decoded = decode_2d_code(binary_sequence, code_LUT, decoding_func)

        # Median filter binary decoded
        logger.info(f"Video frame {i} | Frame-start {frame_start}| Median filter")
        binary_decoded = medfilt2d(binary_decoded)

        inpaint_mask = get_intrinsic_extrinsic.get_mask(cfg)

        binary_decoded = inpaint_func(binary_decoded, inpaint_mask)

        # Save correspondences
        plt.imshow(binary_decoded)
        plt.colorbar()
        save_plot(
            cfg.savefig,
            show=cfg.show,
            fname=f"depth_maps/correspondence_{frame_start}.pdf",
        )

        # 3d reconstruction
        projector_pixels = binary_decoded[valid_indices]
        projector_pixels = np.stack(
            [projector_pixels, np.zeros_like(projector_pixels)], axis=1
        )

        # Triangulate, remove invalid intersections
        points_3d = triangulate_ray_plane(
            camera_matrix, projector_matrix, camera_pixels, projector_pixels, axis=0
        )

        # Median filtering
        points_3d[:, 2] = medfilt(points_3d[:, 2], kernel_size=5)

        # Depth map
        depth_map = np.zeros((cfg.spad.height, cfg.spad.width))
        depth_map[valid_indices] = points_3d[:, 2]
        mask = (depth_map >= cfg.scene.get("depth_map_vmin")) & (
            depth_map <= cfg.scene.get("depth_map_vmax")
        )
        depth_map[~mask] = np.nan

        cmap = matplotlib.cm.get_cmap("jet").copy()
        cmap.set_bad("black", alpha=1.0)
        plt.imshow(
            depth_map,
            vmin=cfg.scene.get("depth_map_vmin"),
            vmax=cfg.scene.get("depth_map_vmax"),
            cmap=cmap,
        )
        plt.colorbar()
        save_plot(
            cfg.savefig,
            show=cfg.show,
            fname=f"depth_maps/depth_map_{frame_start}.png",
        )

        depth_map_ll.append(depth_map)

    with imageio.get_writer("mygif.gif", mode="I") as writer:
        for i in range(num_video_frames):
            frame_start = cfg.frame_start + i * frame_gap
            image = imageio.imread(f"depth_maps/depth_map_{frame_start}.png")
            writer.append_data(image)


if __name__ == "__main__":
    main()
