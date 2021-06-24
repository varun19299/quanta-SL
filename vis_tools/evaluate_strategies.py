import logging

import numpy as np

FORMAT = "%(asctime)s [%(filename)s : %(funcName)2s() : %(lineno)2s] %(message)s"
logging.basicConfig(format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)

from vis_tools.strategies.analytic import (
    naive,
    average_fixed,
    average_optimal,
    average_conventional,
    naive_conventional,
)
from vis_tools.strategies.metaclass import CallableEval, BCH, Repetition
from vis_tools.strategies.monte_carlo import bch_coding, repetition_coding
from vis_tools.strategies.plot import (
    plot_optimal_threshold,
    mesh_and_surface_plot,
)
from vis_tools.strategies.utils import func_name

if __name__ == "__main__":
    phi_proj = np.logspace(1, 8, num=512)
    phi_A = np.logspace(0, 5, num=512)

    # DMD framerate
    # 0.1 millisecond or 10^4 FPS
    t_exp = 1e-4

    plot_options = {
        "show": False,
        "plot_3d": True,
        "savefig": True,
    }
    avg_kwargs = {"num_frames": 10}
    conventional_sensor_kwargs = {"threshold": 0.5, "Q_e": 0.5, "N_r": 1e-1}

    bch_ll = [
        BCH(15, 11, 1),
        BCH(31, 11, 5),
        BCH(63, 10, 13),
        BCH(127, 15, 27),
    ]
    # bch_ll = [BCH(15, 11, 1), BCH(31, 11, 7), BCH(63, 10, 18), BCH(127, 15, 40)]

    bch_kwargs_ll = [
        {"bch_tuple": bch_tuple, "num_frames": 1, "use_complementary": False}
        for bch_tuple in bch_ll
    ]

    repetition_ll = [
        Repetition(30, 10, 1),
        Repetition(60, 10, 2),
        Repetition(130, 10, 5),
    ]
    repetition_kwargs_ll = [
        {
            "repetition_tuple": repetition_tuple,
            "num_frames": 1,
            "use_complementary": False,
        }
        for repetition_tuple in repetition_ll
    ]

    # Optimal threshold
    plot_optimal_threshold(
        phi_proj,
        phi_A,
        t_exp=t_exp,
        num_frames=10,
        **plot_options,
    )

    # Effect of averaging
    strategy_ll = [
        # CallableEval(func_name(naive), naive),
        # CallableEval(func_name(average_fixed), average_fixed, avg_kwargs),
        CallableEval(func_name(average_optimal), average_optimal, avg_kwargs),
        CallableEval(
            func_name(naive_conventional),
            naive_conventional,
            conventional_sensor_kwargs,
        ),
    ]

    # BCH Coding
    strategy_ll += [
        CallableEval(f"{func_name(func)}/{bch_tuple}", bch_coding, bch_kwargs)
        for bch_tuple, bch_kwargs in zip(bch_ll, bch_kwargs_ll)
        for code in bch_ll
        for func in [naive, average_fixed, average_optimal]
    ]

    # Repetition Coding
    strategy_ll += [
        CallableEval(
            f"{func_name(func)}/{repetition_tuple}",
            repetition_coding,
            repetition_kwargs,
        )
        for repetition_tuple, repetition_kwargs in zip(
            repetition_ll, repetition_kwargs_ll
        )
        for func in [naive, average_fixed, average_optimal]
    ]

    for strategy in strategy_ll:
        mesh_and_surface_plot(
            phi_proj,
            phi_A,
            t_exp,
            strategy,
            **plot_options,
        )

    # Averaging vs Frames Quanta
    num_frame_ll = [2, 5, 10, 20, 100]

    strategy_ll = [
        CallableEval(
            f"{func_name(strategy)}/frames_{num_frames}",
            strategy,
            avg_kwargs,
        )
        for strategy in [average_fixed, average_optimal]
        for num_frames in num_frame_ll
    ]
    strategy_ll += [
        CallableEval(
            f"{func_name(average_conventional)}/frames_{num_frames}",
            average_conventional,
            conventional_sensor_kwargs,
        )
        for num_frames in num_frame_ll
    ]

    for strategy in strategy_ll:
        mesh_and_surface_plot(
            phi_proj,
            phi_A,
            t_exp,
            strategy,
            **plot_options,
        )
