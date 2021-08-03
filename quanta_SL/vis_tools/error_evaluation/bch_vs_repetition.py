"""
With arbitrary error metrics
"""

import numpy as np
from einops import rearrange
from loguru import logger
from typing import Dict

from quanta_SL.encode.metaclass import CallableEval, BCH, Repetition
from quanta_SL.ops.metrics import exact_error, squared_error
from quanta_SL.vis_tools.error_evaluation import analytic
from quanta_SL.vis_tools.error_evaluation.monte_carlo import (
    no_coding,
    repetition_coding,
    bch_coding,
)
from quanta_SL.vis_tools.error_evaluation.plotting import (
    individual_and_multiple_plots,
)
from copy import copy


def _get_strategies(
    bch_tuple: BCH,
    repetition_tuple: Repetition,
    bch_comp_tuple: BCH = (),
    **coding_kwargs,
):
    strategy_ll = [
        CallableEval(
            f"No Coding [{repetition_tuple.k} bits]", no_coding, coding_kwargs
        ),
        CallableEval(
            f"{repetition_tuple}",
            repetition_coding,
            {"repetition_tuple": repetition_tuple, **coding_kwargs},
        ),
        CallableEval(
            f"{bch_tuple}", bch_coding, {"bch_tuple": bch_tuple, **coding_kwargs}
        ),
    ]

    # Compare the complementary idea too
    if bch_comp_tuple:
        bch_comp_kwargs = {
            "bch_tuple": bch_comp_tuple,
            "use_complementary": True,
            **coding_kwargs,
        }
        strategy_ll += [
            CallableEval(
                f"{bch_comp_tuple} comp",
                bch_coding,
                bch_comp_kwargs,
            ),
        ]

    return strategy_ll


def _compare_repetition_bch(
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

    bch_tuple_ll = [
        BCH(15, 11, 1),
        BCH(31, 11, 5),
        BCH(63, 10, 13),
        BCH(127, 15, 27),
        BCH(255, 13, 59),
    ]
    repetition_tuple_ll = [
        Repetition(10, 10, 0),
        Repetition(30, 10, 1),
        Repetition(60, 10, 2),
        Repetition(130, 10, 6),
        Repetition(260, 10, 12),
    ]
    redundancy_ll = [1, 3, 6, 13, 25]
    redundancy_index = redundancy_ll.index(redundancy_factor)

    assert redundancy_index, "Comparing at redundancy 1 not supported"

    bch_comp_tuple = ()

    # Complementary makes sense
    # only with averaging
    if oversampling_factor > 1:
        bch_comp_tuple = bch_tuple_ll[redundancy_index - 1]

    strategy_ll = _get_strategies(
        bch_tuple_ll[redundancy_index],
        repetition_tuple_ll[redundancy_index],
        bch_comp_tuple,
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

    plot_kwargs = dict(show=False, plot_3d=True, savefig=True)
    coding_kwargs = dict(monte_carlo_iter=5)

    # Repetition vs BCH
    redundancy_ll = [3, 6, 13, 25]
    # redundancy_ll = [25]
    oversampling_ll = [1, 5]

    for redundancy_factor in redundancy_ll:
        for oversampling_factor in oversampling_ll:
            _compare_repetition_bch(
                redundancy_factor,
                oversampling_factor,
                coding_kwargs=coding_kwargs,
                plot_kwargs=plot_kwargs,
            )
            _compare_repetition_bch(
                redundancy_factor,
                oversampling_factor,
                coding_kwargs=coding_kwargs,
                plot_kwargs={**plot_kwargs, "error_metric": squared_error},
            )
