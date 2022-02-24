from pathlib import Path

import cv2
import hydra
import matplotlib
import numpy as np
from einops import repeat
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from scipy.signal import medfilt
from sklearn.metrics import mean_squared_error

from quanta_SL.lcd.calibrate import get_intrinsic_extrinsic
from quanta_SL.reconstruct.project3d import CameraMatrix, triangulate_ray_plane
from quanta_SL.utils.plotting import (
    visualize_point_cloud,
    save_plot,
    plot_image_and_colorbar,
)

# Disable inner logging
logger.disable("quanta_SL")
logger.add(f"logs/lcd_acquire_{Path(__file__).stem}.log", rotation="daily", retention=3)

# plt.style.use(["science"])
params = {
    "legend.fontsize": "x-large",
    "figure.titlesize": "xx-large",
    "axes.labelsize": "xx-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
plt.rcParams.update(params)


def inpaint_func(img, mask):
    return cv2.inpaint(
        src=img.astype(np.float32),
        inpaintMask=mask,
        inpaintRadius=5,
        flags=cv2.INPAINT_NS,
    )


def reject_outliers(data, m=2.0):
    """
    Reject outliers based on robust median

    :param data:
    :param m: Std dev ratio
    :return:
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0
    return data[s < m]


def stereo_setup(cfg):
    f = np.load(cfg.scene.calibration.npz_file)
    camera_matrix = CameraMatrix(K=f["camera_matrix"])
    projector_matrix = CameraMatrix(
        K=f["projector_matrix"], R=f["R"], T=f["T"].squeeze()
    )

    # Select valid camera points
    # Exclude shadows etc
    width_range = np.arange(cfg.spad.width)
    height_range = np.arange(cfg.spad.height)
    camera_pixels = np.meshgrid(width_range, height_range)

    return camera_matrix, projector_matrix, camera_pixels


def reconstruct_3d(
    cfg,
    col_correspondences,
    camera_pixels,
    mask,
    camera_matrix,
    projector_matrix,
    img,
    fname: str,
    depth_map_vmin: float = None,
    depth_map_vmax: float = None,
):
    valid_indices = np.where(mask)
    camera_pixels = np.stack(
        [camera_pixels[0][valid_indices], camera_pixels[1][valid_indices]], axis=1
    )

    # 3d reconstruction
    projector_pixels = col_correspondences[valid_indices]
    projector_pixels = np.stack(
        [projector_pixels, np.zeros_like(projector_pixels)], axis=1
    )

    gray_colors = img[valid_indices]
    gray_colors = repeat(gray_colors, "n -> n c", c=3)

    # Triangulate, remove invalid intersections
    points_3d = triangulate_ray_plane(
        camera_matrix, projector_matrix, camera_pixels, projector_pixels, axis=0
    )

    # Median filtering
    points_3d[:, 2] = medfilt(points_3d[:, 2], kernel_size=5)

    points_3d_to_return = points_3d.copy()

    # Depth map
    depth_map = np.zeros((cfg.spad.height, cfg.spad.width))
    depth_map[:] = np.nan
    depth_map[valid_indices] = points_3d_to_return[:, 2]

    # Save depth map
    if not depth_map_vmin or not depth_map_vmax:
        inlier_m = 2
        num_attempt = 1
        while True:
            depth_inliers = reject_outliers(points_3d_to_return[:, 2], inlier_m)
            if len(depth_inliers):
                break
            inlier_m *= 2
            num_attempt += 1

            # Too noisy, don't filter. Do it manually
            if num_attempt > 5:
                depth_inliers = points_3d_to_return[:, 2]
                break

    if not depth_map_vmin:
        depth_map_vmin = depth_inliers.min()
    if not depth_map_vmax:
        depth_map_vmax = depth_inliers.max()

    cmap = matplotlib.cm.get_cmap("jet").copy()
    cmap.set_bad("black", alpha=1.0)
    plot_image_and_colorbar(
        depth_map,
        savefig=cfg.savefig,
        show=False,
        fname=Path(fname).parent / "depth_map.pdf",
        cmap=cmap,
        vmin=depth_map_vmin,
        vmax=depth_map_vmax,
        cbar_title="Depth (in cm)",
    )

    # NaN removal
    point_mask = ~np.isnan(points_3d).any(axis=1)
    points_3d = points_3d[point_mask]
    gray_colors = gray_colors[point_mask]

    # Range filtering
    distance_range = cfg.scene.get("distance_range", (20, 60))
    point_mask = (points_3d[:, 2] > distance_range[0]) & (
        points_3d[:, 2] < distance_range[1]
    )
    points_3d = points_3d[point_mask]
    gray_colors = gray_colors[point_mask]

    # Save meshes
    visualize_point_cloud(
        points_3d,
        gray_colors,
        view_kwargs=cfg.scene.view_kwargs,
        savefig=cfg.savefig,
        show=cfg.show,
        create_mesh=cfg.create_mesh,
        fname=fname,
    )

    return points_3d_to_return, depth_map, depth_map_vmin, depth_map_vmax


@hydra.main(
    config_path="../../conf/lcd/acquire",
    config_name=Path(__file__).stem,
)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    methods_dict = {}

    logger.info("Loading groundtruth")
    img = cv2.imread(cfg.groundtruth.img, -1) / 255.0
    gt_decoded = np.load(cfg.groundtruth.correspondences)

    # Inpaint img, correspondences
    inpaint_mask = get_intrinsic_extrinsic.get_mask(cfg)
    img = inpaint_func(img, inpaint_mask)

    gt_decoded = inpaint_func(gt_decoded, inpaint_mask)
    mask = np.load(cfg.groundtruth.hybrid_filtered_mask)

    # Obtain stereo calibration params
    logger.info("Loading stereo params")
    camera_matrix, projector_matrix, camera_pixels = stereo_setup(cfg)

    # 3d reconstruction
    gt_points_3d, gt_depth_map, gt_depth_vmin, gt_depth_vmax = reconstruct_3d(
        cfg,
        gt_decoded,
        camera_pixels,
        mask,
        camera_matrix,
        projector_matrix,
        img,
        fname=f"{cfg.outfolder}/results/groundtruth",
    )

    for method_key in cfg.methods.keys():
        logger.info(f"Method {method_key}")
        method_cfg = cfg.methods[method_key]

        binary_decoded = np.load(
            f"{cfg.outfolder}/results/{method_key}/correspondences.npz"
        )["binary_decoded"]

        binary_decoded = inpaint_func(binary_decoded, inpaint_mask)

        methods_dict[method_cfg] = {"binary_decoded": binary_decoded}

        # 3d reconstruction
        method_points_3d, method_depth_map, _, _ = reconstruct_3d(
            cfg,
            binary_decoded,
            camera_pixels,
            mask,
            camera_matrix,
            projector_matrix,
            img,
            depth_map_vmin=gt_depth_vmin,
            depth_map_vmax=gt_depth_vmax,
            fname=f"{cfg.outfolder}/results/{method_key}/reconstruction",
        )

        valid_points = ~(
            np.isnan(method_points_3d).any(axis=1) | np.isnan(gt_points_3d).any(axis=1)
        )
        depth_rmse = mean_squared_error(
            gt_points_3d[valid_points, 2],
            method_points_3d[valid_points, 2],
            squared=False,
        )

        # Inliers, inlier rmse
        abs_errors = np.abs(
            gt_points_3d[valid_points, 2] - method_points_3d[valid_points, 2]
        )
        inliers = abs_errors < 0.5
        percentage_inliers = inliers.mean() * 100

        inliers_rmse = mean_squared_error(
            gt_points_3d[valid_points, 2][inliers],
            method_points_3d[valid_points, 2][inliers],
            squared=False,
        )

        logger.info(
            f"{method_key} total rmse (in cm) {depth_rmse:.4g} | % inliers {percentage_inliers:.2f} | inliers rmse (in cm) {inliers_rmse:.4g} "
        )


if __name__ == "__main__":
    main()
