from pathlib import Path

import hydra
import imageio
import matplotlib
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from scipy.signal import medfilt
from scipy.signal import medfilt2d

from quanta_SL.dlp.acquire.decode_correspondences import setup_args, get_binary_sequence
from quanta_SL.lcd.acquire.decode_correspondences import get_code_LUT_decoding_func
from quanta_SL.lcd.acquire.reconstruct import (
    get_intrinsic_extrinsic,
    inpaint_func,
    stereo_setup,
)
# Disable inner logging
from quanta_SL.lcd.decode_helper import decode_2d_code
from quanta_SL.reconstruct.project3d import triangulate_ray_plane
from quanta_SL.utils.plotting import save_plot

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
