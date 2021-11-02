from pathlib import Path

import hydra
from dotmap import DotMap
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf, DictConfig
from nptyping import NDArray
from scipy.io import loadmat, savemat
import numpy as np
import cv2
from einops import rearrange
from copy import copy

from quanta_SL.lcd.calibrate.get_intrinsic_extrinsic import get_mask, _plot_inpainted, find_projector_corners

from collections import namedtuple

from quanta_SL.utils.plotting import ax_imshow_with_colorbar, save_plot

# Disable inner logging
logger.disable("quanta_SL")
logger.add(
    f"logs/dlp_calibrate_{Path(__file__).stem}.log", rotation="daily", retention=3
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

CalibrateCameraOutput = namedtuple(
    "CalibrateCameraOutput",
    ["ret", "matrix", "distortion", "rotation_vecs", "translation_vecs"],
)

StereoCalibrateOutput = namedtuple(
    "StereoCalibrateOutput",
    [
        "ret",
        "camera_matrix",
        "distortion_camera",
        "projector_matrix",
        "distortion_projector",
        "R",
        "T",
        "E",
        "F",
    ],
)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100_000, 1e-6)

camera_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100_000, 1e-10)
calib_flags = (
    cv2.CALIB_ZERO_TANGENT_DIST
    + cv2.CALIB_FIX_K1
    + cv2.CALIB_FIX_K2
    + cv2.CALIB_FIX_K3
    + cv2.CALIB_FIX_K4
    + cv2.CALIB_FIX_K5
    + cv2.CALIB_FIX_K6
)

stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100_000, 1e-56)
stereo_flags = (
    cv2.CALIB_ZERO_TANGENT_DIST
    + cv2.CALIB_FIX_K1
    + cv2.CALIB_FIX_K2
    + cv2.CALIB_FIX_K3
    + cv2.CALIB_FIX_K4
    + cv2.CALIB_FIX_K5
    + cv2.CALIB_FIX_K6
)

@hydra.main(
    config_path="../../conf/dlp/calibrate",
    config_name=Path(__file__).stem,
)
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

    # cfg = setup_args(cfg)

    # Open mat file
    f = loadmat(cfg.correspondence.mat_file)
    img_ll = f["all-white"]
    row_correspondences_ll = f["row-correspondences"]
    col_correspondences_ll = f["col-correspondences"]

    # See: https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cfg.chessboard.height * cfg.chessboard.width, 3), np.float32)
    objp[:, :2] = np.mgrid[
        0 : cfg.chessboard.height, 0 : cfg.chessboard.width
    ].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    proj_points = []

    # Load mask
    mask = get_mask(cfg)

    for pose_index, (img, col_correspondences, row_correspondences) in enumerate(
        zip(img_ll, col_correspondences_ll, row_correspondences_ll), start=1
    ):
        if pose_index in cfg.correspondence.exclude_poses:
            logger.info(f"Excluding Pose{pose_index:02d}")
            continue

        # Inpainting
        inpainted_img = cv2.inpaint(
            src=img.astype(np.float32),
            inpaintMask=mask,
            inpaintRadius=5,
            flags=cv2.INPAINT_NS,
        )
        col_correspondences = cv2.inpaint(
            src=col_correspondences.astype(np.float32),
            inpaintMask=mask,
            inpaintRadius=5,
            flags=cv2.INPAINT_NS,
        )
        row_correspondences = cv2.inpaint(
            src=row_correspondences.astype(np.float32),
            inpaintMask=mask,
            inpaintRadius=5,
            flags=cv2.INPAINT_NS,
        )

        # Find the chess board corners
        inpainted_img_8bit = (inpainted_img * 255).astype(np.uint8)
        ret, corners = cv2.findChessboardCorners(
            inpainted_img_8bit,
            (cfg.chessboard.height, cfg.chessboard.width),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH,
        )

        # If found, add object points, image points (after refining them)
        if ret:
            logger.info(f"Pose{pose_index:02d} | Chessboard detected")

            obj_points.append(objp)
            corners_subpixel = cv2.cornerSubPix(
                inpainted_img,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=criteria,
            )
            img_points.append(corners_subpixel)

            # Draw and display the corners
            marked_inpainted_img = np.stack([inpainted_img] * 3, axis=-1)
            marked_inpainted_img = cv2.drawChessboardCorners(
                marked_inpainted_img, (7, 6), corners_subpixel, ret
            )

            # Find corresponding projector corners
            proj_points.append(
                find_projector_corners(
                    corners_subpixel, col_correspondences, row_correspondences, cfg
                )
            )

            if cfg.plot:
                # Plot image and inpainted version
                _plot_inpainted(
                    pose_index,
                    img,
                    inpainted_img,
                    mask,
                    marked_inpainted_img,
                    col_correspondences,
                    row_correspondences,
                    cfg,
                )
        else:
            logger.info(f"Pose{pose_index:02d} | Chessboard NOT detected")

    # Calibrate Intrinsics
    logger.info("Calibrating Intrinsics")
    camera_intrinsics = CalibrateCameraOutput(
        *cv2.calibrateCamera(
            obj_points,
            img_points,
            img.shape[::-1],
            None,
            None,
            flags=calib_flags,
            criteria=camera_criteria,
        )
    )
    projector_intrinsics = CalibrateCameraOutput(
        *cv2.calibrateCamera(
            obj_points,
            proj_points,
            (cfg.projector.width, cfg.projector.height),
            None,
            None,
            flags=calib_flags,
            criteria=camera_criteria,
        )
    )

    # Stereo Calibrate
    logger.info("Calibrating Stereo")
    stereo_model = StereoCalibrateOutput(
        *cv2.stereoCalibrate(
            obj_points,
            img_points,
            proj_points,
            camera_intrinsics.matrix,
            None,
            projector_intrinsics.matrix,
            None,
            img.shape[::-1],
            flags=stereo_flags,
            criteria=stereo_criteria,
        )
    )

    # Reprojection Error
    logger.info(f"Reprojection Error {stereo_model.ret}")

    # Save
    np.savez(f"{cfg.outfolder}/stereo_model.npz", **stereo_model._asdict())
    savemat(f"{cfg.outfolder}/stereo_model.mat", stereo_model._asdict())


if __name__ == "__main__":
    main()
