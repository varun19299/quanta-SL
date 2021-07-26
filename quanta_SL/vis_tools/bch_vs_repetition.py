import logging
from typing import Callable, Dict

FORMAT = "%(asctime)s [%(filename)s : %(funcName)2s() : %(lineno)2s] %(message)s"
logging.basicConfig(format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)

import numpy as np
from einops import rearrange

from quanta_SL.vis_tools.strategies import analytic
from quanta_SL.vis_tools import CallableEval, BCH, Repetition
from quanta_SL.vis_tools import naive, average_fixed, average_optimal
from quanta_SL.vis_tools import repetition_coding, bch_coding
from quanta_SL.vis_tools.strategies.plot import individual_and_multiple_plots, func_name


def _get_strategies(
    func,
    bch_tuple: BCH,
    repetition_tuple: Repetition,
    bch_list_tuple: BCH,
    bch_comp_tuple: BCH = (),
    bch_comp_list_tuple: BCH = (),
    func_kwargs: Dict = {},
    coding_kwargs: Dict = {},
):
    num_bits = repetition_tuple.k
    strategy_ll = [
        CallableEval(f"{func_name(func)}-{num_bits}-bits", func, func_kwargs),
    ]
    strategy_ll += [
        CallableEval(
            f"{repetition_tuple}",
            repetition_coding,
            {"repetition_tuple": repetition_tuple, **coding_kwargs},
        ),
        CallableEval(
            f"{bch_tuple}", bch_coding, {"bch_tuple": bch_tuple, **coding_kwargs}
        ),
    ]

    # Add list decoding version
    strategy_ll += [
        CallableEval(
            f"{bch_list_tuple}-list",
            bch_coding,
            {"bch_tuple": bch_list_tuple, **coding_kwargs},
        ),
    ]

    # Compare the complementary idea too
    if bch_comp_tuple:
        strategy_ll += [
            CallableEval(
                f"{bch_comp_tuple}-comp",
                bch_coding,
                {
                    "bch_tuple": bch_comp_tuple,
                    "use_complementary": True,
                    **coding_kwargs,
                },
            ),
        ]

    if bch_comp_list_tuple:
        # Add list decoding version
        strategy_ll += [
            CallableEval(
                f"{bch_comp_list_tuple}-list-comp",
                bch_coding,
                {
                    "bch_tuple": bch_comp_list_tuple,
                    "use_complementary": True,
                    **coding_kwargs,
                },
            ),
        ]

    return strategy_ll


def _compare_repetition_bch(
    func: Callable,
    redundancy_factor: int = 1,
    oversampling_factor: int = 1,
    num_bits: int = 10,
):
    global phi_proj, phi_A, t_exp, plot_options

    # Use optimal threshold
    optimal_threshold = "optimal" in func.__name__
    is_averaging = "average" in func.__name__
    avg_kwargs = {"num_frames": oversampling_factor * num_bits}

    if optimal_threshold:
        phi_P_mesh, phi_A_mesh = np.meshgrid(phi_proj + phi_A, phi_A, indexing="ij")
        _, tau = analytic.optimal_threshold(
            phi_P_mesh, phi_A_mesh, t_exp, avg_kwargs["num_frames"]
        )
        tau = rearrange(tau, "h w -> h w 1")
    else:
        tau = 0.5

    func_kwargs = avg_kwargs if is_averaging else {}
    coding_kwargs = {"tau": tau, **func_kwargs}

    # Dump dir
    config_str = f"redundancy-{redundancy_factor}-oversampling-{oversampling_factor}"

    # Plot title
    title = f"Redundancy {redundancy_factor} | Oversampling {oversampling_factor}"
    if is_averaging:
        title += f" | Threshold {'optimal' if optimal_threshold else 'fixed'}"

    logging.info(title)

    # Custom runs at each redundancy
    if redundancy_factor == 3:
        complementary_kwargs = (
            {"bch_comp_tuple": BCH(15, 11, 1)} if is_averaging else {}
        )
        strategy_ll = _get_strategies(
            func,
            BCH(31, 11, 5),
            Repetition(30, 10, 1),
            bch_list_tuple=BCH(31, 11, 7),
            **complementary_kwargs,
            func_kwargs=func_kwargs,
            coding_kwargs=coding_kwargs,
        )

    elif redundancy_factor == 6:
        complementary_kwargs = (
            {"bch_comp_tuple": BCH(31, 11, 5), "bch_comp_list_tuple": BCH(31, 11, 7)}
            if is_averaging
            else {}
        )
        strategy_ll = _get_strategies(
            func,
            BCH(63, 10, 13),
            Repetition(60, 10, 2),
            bch_list_tuple=BCH(63, 10, 18),
            **complementary_kwargs,
            func_kwargs=func_kwargs,
            coding_kwargs=coding_kwargs,
        )

    elif redundancy_factor == 13:
        complementary_kwargs = (
            {"bch_comp_tuple": BCH(63, 10, 13), "bch_comp_list_tuple": BCH(63, 10, 18)}
            if is_averaging
            else {}
        )
        strategy_ll = _get_strategies(
            func,
            BCH(127, 15, 27),
            Repetition(130, 10, 6),
            bch_list_tuple=BCH(127, 15, 38),
            **complementary_kwargs,
            func_kwargs=func_kwargs,
            coding_kwargs=coding_kwargs,
        )
    elif redundancy_factor == 25:
        complementary_kwargs = (
            {
                "bch_comp_tuple": BCH(127, 15, 27),
                "bch_comp_list_tuple": BCH(127, 15, 38),
            }
            if is_averaging
            else {}
        )
        strategy_ll = _get_strategies(
            func,
            BCH(255, 13, 59),
            Repetition(260, 10, 12),
            bch_list_tuple=BCH(255, 13, 93),
            **complementary_kwargs,
            func_kwargs=func_kwargs,
            coding_kwargs=coding_kwargs,
        )
    elif redundancy_factor == 51:
        complementary_kwargs = (
            {
                "bch_comp_tuple": BCH(255, 13, 59),
                "bch_comp_list_tuple": BCH(255, 13, 93),
            }
            if is_averaging
            else {}
        )
        strategy_ll = _get_strategies(
            func,
            BCH(511, 10, 127),
            Repetition(510, 10, 25),
            bch_list_tuple=BCH(511, 10, 243),
            **complementary_kwargs,
            func_kwargs=func_kwargs,
            coding_kwargs=coding_kwargs,
        )

    elif redundancy_factor == 102:
        complementary_kwargs = (
            {
                "bch_comp_tuple": BCH(511, 10, 127),
                "bch_comp_list_tuple": BCH(511, 10, 243),
            }
            if is_averaging
            else {}
        )
        strategy_ll = _get_strategies(
            func,
            BCH(1023, 11, 255),
            Repetition(1020, 10, 50),
            bch_list_tuple=BCH(1023, 11, 493),
            **complementary_kwargs,
            func_kwargs=func_kwargs,
            coding_kwargs=coding_kwargs,
        )
    else:
        raise ValueError(f"Redundancy factor of {redundancy_factor} is invalid")

    outfolder = f"{func_name(func)}/{config_str}"
    individual_and_multiple_plots(
        phi_proj,
        phi_A,
        t_exp,
        strategy_ll,
        outname=outfolder,
        title=title,
        **plot_options,
    )
    print("\n\n")


if __name__ == "__main__":
    phi_proj = np.logspace(3, 6, num=256)
    phi_A = np.logspace(2, 4, num=256)

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

    # Repetition vs BCH
    redundancy_ll = [3, 6, 13, 25, 51, 102]
    redundancy_ll = [102]
    for redundancy_factor in redundancy_ll:
        _compare_repetition_bch(naive, redundancy_factor)

        # Oversampling based
        for oversampling_factor in [5, 10]:
            _compare_repetition_bch(
                average_fixed, redundancy_factor, oversampling_factor
            )
            _compare_repetition_bch(
                average_optimal, redundancy_factor, oversampling_factor
            )
