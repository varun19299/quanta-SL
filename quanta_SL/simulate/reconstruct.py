from copy import deepcopy
from typing import Any, Dict, List, Tuple

import cv2
import hydra
import logging
from matplotlib import pyplot as plt
import numpy as np
import open3d
from nptyping import NDArray, Float32, Int
from omegaconf import DictConfig, ListConfig
import pandas as pd

import quanta_SL.ops.linalg
from quanta_SL.simulate.pbrt import parser
from quanta_SL.simulate import decode
from quanta_SL.simulate import sensor
from quanta_SL.simulate.utils import project3d
from quanta_SL.simulate.pbrt import set_paths

def load_exr2grayscale(path: str) -> NDArray[(Any, Any, 3), Float32]:
    """
    Open exr and convert to grayscale using CV2

    :param path: EXR file path
    :return: Image read as is
    """
    img = cv2.imread(path, -1)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def conventional_SL_threshold(
    captures_dict: Dict[str, List[NDArray]], cfg: DictConfig
) -> Tuple[NDArray[(Any, Any, Any), Int], NDArray[(Any, Any, Any), Int]]:
    """
    Uses captures to compute binary codes

    :param captures_dict: Contains all_white, direct and inverse codes
    :param cfg: OmegaConf
    :return: Binary codes (channels represent bits), Mask
    """
    # Conventional SL
    assert cfg.reconstruct.include_inverse, "More robust with inverse captures"
    coded_captures = np.stack(
        [captures_dict["coded"][i] for i in range(cfg.reconstruct.num_codes)], axis=-1
    )
    inverse_coded_captures = np.stack(
        [captures_dict["inverse_coded"][i] for i in range(cfg.reconstruct.num_codes)],
        axis=-1,
    )
    binary_codes = (coded_captures > inverse_coded_captures).astype(int)

    # TODO: will need a threshold
    if cfg.reconstruct.include_all_white:
        mask = (captures_dict["all_white"] > 0).astype(int)
    else:
        # Every point covered by projector receives light at some point
        mask = ((coded_captures + inverse_coded_captures).sum(axis=-1) > 0).astype(int)
    return binary_codes, mask


def quanta_SL_threshold(
    captures_dict: Dict[str, List[NDArray]], cfg: DictConfig
) -> Tuple[NDArray[(Any, Any, Any), Int], NDArray[(Any, Any, Any), Int]]:
    """
    Uses captures to compute binary codes
    Transforms captures to Quanta captures

    :param captures_dict: Contains all_white, direct and inverse codes
    :param cfg: Omegaconf
    :return: Binary codes (channels represent bits), Mask
    """
    # Conventional SL
    assert cfg.reconstruct.include_all_white, "Quanta SL needs all white"
    binary_codes = np.stack(
        [
            sensor.spad(
                captures_dict["coded"][i] * cfg.sensor.intensity_multiplier,
                t_exp=cfg.sensor.exposure,
            )
            for i in range(cfg.reconstruct.num_codes)
        ],
        axis=-1,
    )

    mask = sensor.spad(
        captures_dict["all_white"] * cfg.sensor.intensity_multiplier,
        t_exp=cfg.sensor.exposure,
    )

    return binary_codes, mask


def load_captures(
    name: str,
    extension: str,
    num_codes: int,
    include_inverse: bool = True,
    include_all_white: bool = True,
) -> Dict[str, List[NDArray]]:
    """
    All captures are stored in the same output directory
    :return:
    """
    if include_inverse:
        projector_indices = list(range(1, num_codes * 2 + 1, 2))
        inverse_projector_indices = list(range(2, num_codes * 2 + 1, 2))
    else:
        projector_indices = list(range(1, num_codes + 1))
        inverse_projector_indices = []

    captures_dict = {}
    if include_all_white:
        captures_dict["all_white"] = load_exr2grayscale(f"{name}_0.{extension}")

    captures_dict["coded"] = [
        load_exr2grayscale(f"{name}_{i}.{extension}") for i in projector_indices
    ]
    captures_dict["inverse_coded"] = [
        load_exr2grayscale(f"{name}_{i}.{extension}") for i in inverse_projector_indices
    ]

    return captures_dict


def reconstruct_3d(
    correspondence: NDArray[(Any, Any), int],
    mask: NDArray[(Any, Any), int],
    camera_matrix: project3d.CameraMatrix,
    projector_matrix: project3d.CameraMatrix,
    axis: int = 0,
    filename: str = "",
):
    """
    Compute 3D reconstruction from correspondence

    :param correspondence: Correspondence map
    :param mask: Region of valid correspondence (1 if valid, 0 otherwise)
    :param camera_matrix: Camera matrix, K[R | T]
    :param projector_matrix: Projector matrix, K[R | T]
    :param axis: vertical patterns (x const) or horizontal (y const) patterns
    :return:
    """
    assert axis in [0, 1]

    width_range = np.arange(mask.shape[1])
    height_range = np.arange(mask.shape[0])
    camera_pixels = np.meshgrid(width_range, height_range)

    # Select valid correspondences
    valid_indices = np.where(mask == 1)
    camera_pixels = np.stack(
        [camera_pixels[0][valid_indices], camera_pixels[1][valid_indices]], axis=1
    )
    projector_pixels = correspondence[valid_indices]

    if axis == 0:
        projector_pixels = np.stack(
            [projector_pixels, np.zeros_like(projector_pixels)], axis=1
        )
    else:
        projector_pixels = np.stack(
            [np.zeros_like(projector_pixels), projector_pixels], axis=1
        )

    # Triangulate, remove invalid intersections
    points_3d = project3d.triangulate_ray_plane(
        camera_matrix, projector_matrix, camera_pixels, projector_pixels, axis=axis
    )
    points_3d = points_3d[~np.isnan(points_3d).any(axis=1)]

    # 3D plot
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points_3d)
    open3d.visualization.draw_geometries([pcd])

    if filename:
        open3d.io.write_point_cloud(filename, pcd)


def single_exposure_run(cfg: DictConfig) -> float:
    captures_dict = load_captures(**cfg.output, **cfg.reconstruct)

    if cfg.visualize.show_captures:
        for img in captures_dict["coded"]:
            plt.imshow(img, cmap="gray")
            plt.show()

    pbrt_path, _, _ = set_paths(cfg)
    with open(pbrt_path) as f:
        pbrt_file = f.read()

    camera_matrix, projector_matrix = parser.get_camera_projector_matrices(
        pbrt_file
    )

    binary_codes, mask = conventional_SL_threshold(captures_dict, cfg)
    conventional_correspondence = decode.conventional_gray_code(binary_codes, mask)

    if cfg.visualize.reconstruct_3d:
        reconstruct_3d(
            conventional_correspondence,
            mask,
            camera_matrix,
            projector_matrix,
            filename="conventional.ply",
        )

    binary_codes, mask = quanta_SL_threshold(captures_dict, cfg)
    quanta_correspondence = decode.conventional_gray_code(binary_codes, mask)

    if cfg.visualize.reconstruct_3d:
        reconstruct_3d(
            quanta_correspondence,
            mask,
            camera_matrix,
            projector_matrix,
            filename="quanta.ply",
        )

    if cfg.visualize.show_correspondences:
        plt.imshow(conventional_correspondence, cmap="gray", interpolation="none")
        plt.show()
        plt.imshow(quanta_correspondence, cmap="gray", interpolation="none")
        plt.show()

    if cfg.visualize.show_abs_error:
        plt.imshow(
            np.abs(conventional_correspondence - quanta_correspondence),
            cmap="gray",
            interpolation="none",
        )
        plt.title("Absolute error in correspondences")
        plt.colorbar()
        plt.savefig(
            f"abs_error_exp_{cfg.sensor.exposure}_int_mul_{cfg.sensor.intensity_multiplier}.pdf",
            dpi=150,
        )
        plt.show()

    # rmse = np.sqrt(((conventional_correspondence - quanta_correspondence) ** 2).mean())
    mean_l1_error = quanta_SL.ops.linalg.mean()
    logging.info(f"Mean L1 between Quanta and Conventional {mean_l1_error}")
    logging.info(
        f"SPAD: exposure {cfg.sensor.exposure} intenstiy multiplier {cfg.sensor.intensity_multiplier}"
    )

    return mean_l1_error


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    if isinstance(cfg.sensor.get("exposure", None), ListConfig):
        mean_l1_error_ll = []
        for exposure in cfg.sensor.exposure:
            run_cfg = deepcopy(cfg)
            run_cfg.sensor.exposure = exposure
            mean_l1_error = single_exposure_run(run_cfg)
            mean_l1_error_ll.append(mean_l1_error)

        exposure_ll = np.array(list(cfg.sensor.exposure))
        intensity_multiplier_ll = (
            np.ones_like(exposure_ll) * cfg.sensor.intensity_multiplier
        )
        plt.semilogx(exposure_ll, mean_l1_error_ll, "b", marker="o")
        plt.grid()
        plt.xlabel(f"T exposure (s)")
        plt.ylabel(f"Mean L1 in correspondences (conventional, quanta)")
        plt.savefig("Mean_L1_vs_exposure.pdf", dpi=150)
        plt.show()

        plt.semilogx(
            exposure_ll * intensity_multiplier_ll, mean_l1_error_ll, "b", marker="o"
        )
        plt.grid()
        plt.xlabel(r"$t_\mathrm{exp} \times \Phi_\mathrm{max}$")
        plt.ylabel(f"Mean L1 in correspondences (conventional, quanta)")
        plt.savefig("Mean_L1_vs_exposure_phi_prod.pdf", dpi=150)
        plt.show()

        df = pd.DataFrame(columns=["Exposure", "Intensity Multiplier", "Mean L1"])
        df["Exposure"] = exposure_ll
        df["Intensity Multiplier"] = intensity_multiplier_ll
        df["Mean L1"] = mean_l1_error_ll
        df.to_csv("mean_l1.csv")

        return sum(mean_l1_error_ll) / len(mean_l1_error_ll)
    else:
        mean_l1_error = single_exposure_run(cfg)
        return mean_l1_error


if __name__ == "__main__":
    main()
