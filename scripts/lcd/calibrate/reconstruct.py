from pathlib import Path

import hydra
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from scipy.io import loadmat
import numpy as np
import cv2
import open3d
from quanta_SL.reconstruct.project3d import CameraMatrix, triangulate_ray_plane
from scripts.lcd.calibrate import get_intrinsic_extrinsic

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


@hydra.main(
    config_path="../../../conf/scripts",
    config_name=f"lcd_calibrate_{Path(__file__).stem}",
)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    # Obtain image, correspondences
    f = loadmat(cfg.correspondence.mat_file)
    img_ll = f["all-white"]
    row_correspondences_ll = f["row-correspondences"]
    col_correspondences_ll = f["col-correspondences"]

    img = img_ll[cfg.pose.index - 1]
    row_correspondences = row_correspondences_ll[cfg.pose.index - 1]
    col_correspondences = col_correspondences_ll[cfg.pose.index - 1]

    # Inpaint img, correspondences
    inpaint_mask = get_intrinsic_extrinsic.get_mask(cfg)
    img = cv2.inpaint(
        src=img.astype(np.float32),
        inpaintMask=inpaint_mask,
        inpaintRadius=5,
        flags=cv2.INPAINT_NS,
    )
    col_correspondences = cv2.inpaint(
        src=col_correspondences.astype(np.float32),
        inpaintMask=inpaint_mask,
        inpaintRadius=5,
        flags=cv2.INPAINT_NS,
    )
    row_correspondences = cv2.inpaint(
        src=row_correspondences.astype(np.float32),
        inpaintMask=inpaint_mask,
        inpaintRadius=5,
        flags=cv2.INPAINT_NS,
    )

    # Obtain stereo calibration params
    f = np.load(cfg.calibration.npz_file)
    camera_matrix = CameraMatrix(K=f["camera_matrix"])
    projector_matrix = CameraMatrix(
        K=f["projector_matrix"], R=f["R"], T=f["T"].squeeze()
    )

    # Select valid camera points
    # Exclude shadows etc
    width_range = np.arange(cfg.spad.width)
    height_range = np.arange(cfg.spad.height)
    camera_pixels = np.meshgrid(width_range, height_range)

    mask = img > 0.07
    valid_indices = np.where(mask)

    camera_pixels = np.stack(
        [camera_pixels[0][valid_indices], camera_pixels[1][valid_indices]], axis=1
    )
    projector_pixels = col_correspondences[valid_indices]
    projector_pixels = np.stack(
        [projector_pixels, np.zeros_like(projector_pixels)], axis=1
    )

    # Triangulate, remove invalid intersections
    points_3d = triangulate_ray_plane(
        camera_matrix, projector_matrix, camera_pixels, projector_pixels, axis=0
    )
    points_3d = points_3d[~np.isnan(points_3d).any(axis=1)]

    # 3D plot
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points_3d)
    open3d.visualization.draw_geometries([pcd])

    if cfg.savefig:
        path = Path(f"{cfg.outfolder}/point_cloud/pose{cfg.pose.index:02d}.ply")
        path.parent.mkdir(exist_ok=True, parents=True)
        open3d.io.write_point_cloud(str(path), pcd)


if __name__ == "__main__":
    main()
