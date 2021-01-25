from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import cv2
import hydra
import logging
from matplotlib import pyplot as plt
import numpy as np
from nptyping import NDArray, Float32
from omegaconf import DictConfig, ListConfig
import pandas as pd

import pypbrt.decode as decode
import pypbrt.sensor as sensor


def exr2grayscale(path: str) -> NDArray[(Any, Any, 3), Float32]:
    """
    Open exr and convert to grayscale using CV2
    :param path:
    :return:
    """
    img = cv2.imread(path, -1)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def conventional_SL_threshold(captures_dict: Dict[str, List[NDArray]], cfg: DictConfig):
    """
    Uses captures to compute binary codes
    :param captures_dict: Contains all_white, direct and inverse codes
    :param cfg:
    :return:
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


def quanta_SL_threshold(captures_dict: Dict[str, List[NDArray]], cfg: DictConfig):
    """
    Uses captures to compute binary codes
    :param captures_dict: Contains all_white, direct and inverse codes
    :param cfg:
    :return:
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
    output_folder = Path(".")

    if include_inverse:
        projector_indices = list(range(1, num_codes * 2 + 1, 2))
        inverse_projector_indices = list(range(2, num_codes * 2 + 1, 2))
    else:
        projector_indices = list(range(1, num_codes + 1))
        inverse_projector_indices = []

    captures_dict = {}
    if include_all_white:
        captures_dict["all_white"] = exr2grayscale(f"{name}_0.{extension}")

    captures_dict["coded"] = [
        exr2grayscale(f"{name}_{i}.{extension}") for i in projector_indices
    ]
    captures_dict["inverse_coded"] = [
        exr2grayscale(f"{name}_{i}.{extension}") for i in inverse_projector_indices
    ]

    return captures_dict


def single_exposure_run(cfg: DictConfig) -> float:
    captures_dict = load_captures(**cfg.output, **cfg.reconstruct)

    # Visualise
    # for img in captures_dict["coded"]:
    #     plt.imshow(img, cmap="gray")
    #     plt.show()

    binary_codes, mask = conventional_SL_threshold(captures_dict, cfg)
    conventional_correspondence = decode.conventional_gray_code(binary_codes, mask)

    binary_codes, mask = quanta_SL_threshold(captures_dict, cfg)
    quanta_correspondence = decode.conventional_gray_code(binary_codes, mask)

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
    mean_l1_error = np.abs(conventional_correspondence - quanta_correspondence).mean()
    logging.info(f"Mean L1 between Quanta and Conventional {mean_l1_error}")
    logging.info(
        f"SPAD: exposure {cfg.sensor.exposure} intenstiy multiplier {cfg.sensor.intensity_multiplier}"
    )

    return mean_l1_error


@hydra.main(config_path="../conf", config_name="config")
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
