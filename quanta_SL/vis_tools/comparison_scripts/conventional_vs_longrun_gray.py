"""
Repeated Conventional Gray
vs
Repeated Long-run Gray

Comparison based on RMSE.
Both strategies are fairly robust to short-range effects
(projector defocus, subsurface scattering, etc.)
"""

from pathlib import Path
from typing import Dict

import numpy as np
from einops import rearrange
from loguru import logger

from quanta_SL.encode.message import long_run_gray_message, gray_message
from quanta_SL.encode.metaclass import CallableEval, Repetition
from quanta_SL.ops.metrics import exact_error, root_mean_squared_error
from quanta_SL.vis_tools.error_evaluation import analytic
from quanta_SL.vis_tools.error_evaluation.monte_carlo import (
    repetition_coding,
)
from quanta_SL.vis_tools.error_evaluation.plotting import (
    individual_and_multiple_plots,
)


def _get_strategies(
    repetition_tuple: Repetition,
    **coding_kwargs,
):
    strategy_ll = [
        CallableEval(
            f"Gray {repetition_tuple}",
            repetition_coding,
            {
                "repetition_tuple": repetition_tuple,
                "message_mapping": gray_message,
                "pbar_header": "Conventional Gray",
                **coding_kwargs,
            },
        ),
        CallableEval(
            f"Max-minSW {repetition_tuple}",
            repetition_coding,
            {
                "repetition_tuple": repetition_tuple,
                "message_mapping": long_run_gray_message,
                "pbar_header": "LongRun Gray",
                **coding_kwargs,
            },
        ),
    ]

    return strategy_ll


def _compare(
    redundancy_factor: int = 1,
    oversampling_factor: int = 1,
    use_optimal_threshold: bool = True,
    coding_kwargs: Dict = {},
    plot_kwargs: Dict = {},
):
    global phi_proj, phi_A, t_exp

    # Threshold
    if (oversampling_factor > 1) and use_optimal_threshold:
        phi_P_mesh, phi_A_mesh = np.meshgrid(phi_proj + phi_A, phi_A, indexing="ij")
        tau = analytic.optimal_threshold(
            phi_P_mesh, phi_A_mesh, t_exp, oversampling_factor
        )
        tau = rearrange(tau, "h w -> h w 1")
    else:
        tau = 0.5

    # Kwargs
    error_metric = plot_kwargs.get("error_metric", exact_error)
    _plot_kwargs = {"error_metric": error_metric, **plot_kwargs}

    _coding_kwargs = {
        "tau": tau,
        "num_frames": oversampling_factor,
        "error_metric": error_metric,
        **coding_kwargs,
    }

    # Dump dir
    config_str = f"over={oversampling_factor} | redun={redundancy_factor}"

    # Plot title
    title = f"Redundancy {redundancy_factor} | Oversampling {oversampling_factor}"

    if oversampling_factor > 1:
        title += f" | Threshold {'optimal' if use_optimal_threshold else 'fixed'}"

    log_str = f"{title} | {error_metric.long_name}"
    logger.info(log_str)

    repetition_tuple_ll = [
        Repetition(11, 11, 0),
        Repetition(33, 11, 1),
        Repetition(66, 11, 2),
        Repetition(143, 11, 6),
        Repetition(275, 11, 12),
    ]
    redundancy_ll = [1, 3, 6, 13, 25]
    redundancy_index = redundancy_ll.index(redundancy_factor)

    assert redundancy_index, "Comparing at redundancy 1 not supported"

    strategy_ll = _get_strategies(
        repetition_tuple_ll[redundancy_index],
        **_coding_kwargs,
    )

    outfolder = f"{config_str}/{error_metric.long_name}"
    individual_and_multiple_plots(
        phi_proj,
        phi_A,
        t_exp,
        strategy_ll,
        outname=outfolder,
        title=title,
        **_plot_kwargs,
    )
    logger.info("\n\n")


if __name__ == "__main__":
    from quanta_SL.utils.gpu_status import FAISS_GPUs

    if FAISS_GPUs:
        num = 256
    else:
        num = 64

    phi_proj = np.logspace(4, 5, num=num)
    phi_A = np.logspace(3, 4, num=num)

    # DMD framerate
    # 0.1 millisecond or 10^4 FPS
    t_exp = 1e-4

    plot_kwargs = dict(
        show=False,
        plot_3d=True,
        savefig=True,
        error_metric=root_mean_squared_error,
        plot_dir=Path("outputs/strategy_comparison/conventional_vs_longrun_gray/"),
    )

    coding_kwargs = dict(monte_carlo_iter=1, num_bits=11)

    if FAISS_GPUs:
        coding_kwargs["monte_carlo_iter"] = 5

    # Repetition vs BCH
    redundancy_ll = [3, 6, 13, 25]
    oversampling_ll = [1, 5]

    # Only RMSE makes sense here
    for redundancy_factor in redundancy_ll:
        for oversampling_factor in oversampling_ll:
            _compare(
                redundancy_factor,
                oversampling_factor,
                coding_kwargs=coding_kwargs,
                plot_kwargs=plot_kwargs,
            )
