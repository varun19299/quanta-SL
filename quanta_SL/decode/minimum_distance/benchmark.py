from typing import Callable, Dict, Type

import numpy as np
import pandas as pd
from einops import repeat
from loguru import logger
from matplotlib import pyplot as plt
from nptyping import NDArray

from quanta_SL.decode.minimum_distance.factory import (
    brute_minimum_distance,
    cupy_minimum_distance,
    faiss_minimum_distance,
    galois_berlekamp_massey,
    keops_minimum_distance,
    numba_minimum_distance,
    pyNNdescent_minimum_distance,
    sklearn_minimum_distance,
    balltree_index,
    faiss_flat_index,
    faiss_flat_gpu_index,
    nndescent_index,
)
from quanta_SL.encode import metaclass
from quanta_SL.encode.message import binary_message
from quanta_SL.encode.strategies import bch_code_LUT
from quanta_SL.ops.binary import packbits_strided
from quanta_SL.ops.coding import hamming_distance_8bit
from quanta_SL.ops.noise import fixed_bit_flip_corrupt
from quanta_SL.utils.gpu_status import (
    xp,
    CUPY_GPUs,
    KEOPS_GPUs,
    FAISS_GPUs,
    GPU_AVAILABLE,
)
from quanta_SL.utils.plotting import save_plot
from quanta_SL.utils.timer import Timer, CPUTimer, CuPyTimer

plt.style.use(["science", "grid"])


def bch_dataset_query_points(
    bch_tuple: metaclass.BCH,
    num_bits: int = 10,
    query_repeat: int = 1,
):
    """
    Generate bch dataset and query points

    :param bch_tuple: BCH[n, k, t]
    :param num_bits: message dimensionality
    :param query_repeat: Query points are 2^k x query_repeat
    :return:
    """
    # BCH encode
    code_LUT = bch_code_LUT(bch_tuple, num_bits, message_mapping=binary_message)

    N, n = code_LUT.shape
    y = code_LUT

    x = repeat(code_LUT, "N n -> (N repeat) n", repeat=query_repeat)
    gt_indices = np.arange(N)
    gt_indices = repeat(gt_indices, "N -> (N repeat)", repeat=query_repeat)

    # Corrupt query points
    with CPUTimer() as t:
        x = fixed_bit_flip_corrupt(x, noisy_bits=bch_tuple.t)

    logger.info(f"\tNoise generation time: {t:.4g}")
    return x, y, gt_indices


def benchmark_func(
    name: str,
    func: Callable,
    x: NDArray[int],
    y: NDArray[int],
    gt_indices: NDArray[int],
    pack: bool = False,
    index_func: Callable = None,
    benchmark_dict: Dict = {},
    Timer: Type[Timer] = CPUTimer,
    num_repeat: int = 1,
    **func_kwargs,
) -> Dict:
    f"""
    Benchmark a function, with indexing separately

    :param name: Method name
    :param func: f(x, y)
    :param x: Binary / uint8 (packed) query vector
    :param y: Binary / uint8 (packed) query vector
    :param gt_indices: Groundtruth for min dist of y indices for each x
    :param pack: Whether to pack bits into bytes.
        Saves space, since the smallest size on most Prog. Langs is 1 byte.
    :param index_func: Indexing func, index(y)
    :param benchmark_dict: Adds benchmark_dict[name], with accuracy and timing
    :param func_kwargs: optional keyword kwargs to func
    :return: benchmark_dict updated
    """
    logger.info(f"\tBenchmarking {name}")

    if pack:
        x = packbits_strided(x)
        y = packbits_strided(y)

    sub_dict = {}
    if index_func:
        with Timer() as t:
            index = index_func(y)

        # Indexing time
        logger.info(f"\t\tIndex preparation time: {t.elapsed_time:.6g}")
        func_kwargs.update({"index": index})
        sub_dict["indexing_time"] = t.elapsed_time

    # Run MDD
    for _ in range(num_repeat):
        with CPUTimer() as t:
            indices = func(x, y, **func_kwargs)

    if np.isnan(indices).any():
        logger.info(f"\t\tNaNs returned")
        sub_dict["querying_time"] = np.nan
        sub_dict["accuracy"] = np.nan

    else:
        # Query time
        logger.info(f"\t\tQuery time: {t.elapsed_time:.6g}")
        sub_dict["querying_time"] = t.elapsed_time

        # Accuracy
        accuracy = (indices == gt_indices).mean()
        logger.info(f"\t\tAccuracy: {accuracy:.6g}")
        sub_dict["accuracy"] = accuracy

    benchmark_dict[name] = sub_dict
    return indices


def cpu_minimum_distance(
    x: NDArray[int], y: NDArray[int], gt_indices: NDArray[int], **kwargs
):
    # Dirty way to store x, y, gt_indices
    data_query_kwargs = dict(x=x, y=y, gt_indices=gt_indices)
    benchmark_dict = {}

    benchmark_func(
        "Berlekamp Massey",
        galois_berlekamp_massey,
        **data_query_kwargs,
        benchmark_dict=benchmark_dict,
        bch_tuple=kwargs["bch_tuple"],
    )
    # benchmark_func(
    #     "Numpy",
    #     brute_minimum_distance,
    #     **data_query_kwargs,
    #     benchmark_dict=benchmark_dict,
    # )
    benchmark_func(
        "Numpy byte-packed",
        brute_minimum_distance,
        **data_query_kwargs,
        pack=True,
        benchmark_dict=benchmark_dict,
        hamming_dist_LUT=hamming_distance_8bit(),
    )
    benchmark_func(
        "Numba byte-packed",
        numba_minimum_distance,
        **data_query_kwargs,
        pack=True,
        benchmark_dict=benchmark_dict,
        hamming_dist_LUT=hamming_distance_8bit(),
    )
    benchmark_func(
        "Sklearn BallTree",
        sklearn_minimum_distance,
        **data_query_kwargs,
        index_func=balltree_index,
        benchmark_dict=benchmark_dict,
    )
    benchmark_func(
        "NNDescent",
        pyNNdescent_minimum_distance,
        **data_query_kwargs,
        index_func=nndescent_index,
        benchmark_dict=benchmark_dict,
    )
    benchmark_func(
        "FAISS Flat",
        faiss_minimum_distance,
        **data_query_kwargs,
        index_func=faiss_flat_index,
        pack=True,
        benchmark_dict=benchmark_dict,
    )

    return benchmark_dict


def gpu_minimum_distance(
    x: NDArray[int], y: NDArray[int], gt_indices: NDArray[int], **kwargs
):
    # Dirty way to store x, y, gt_indices
    data_query_kwargs = dict(x=x, y=y, gt_indices=gt_indices)
    benchmark_dict = {}

    if CUPY_GPUs:
        benchmark_func(
            "CuPy byte-packed",
            cupy_minimum_distance,
            **data_query_kwargs.copy(),
            pack=True,
            benchmark_dict=benchmark_dict,
            hamming_dist_LUT=hamming_distance_8bit(),
            Timer=CuPyTimer,
            num_repeat=1,
        )

        mempool = xp.get_default_memory_pool()
        pinned_mempool = xp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if KEOPS_GPUs:
        benchmark_func(
            "KeOps",
            keops_minimum_distance,
            **data_query_kwargs,
            benchmark_dict=benchmark_dict,
        )

    if FAISS_GPUs:
        benchmark_func(
            "FAISS Flat",
            faiss_minimum_distance,
            **data_query_kwargs,
            index_func=faiss_flat_gpu_index,
            pack=True,
            benchmark_dict=benchmark_dict,
        )

    return benchmark_dict


def plot_timing(
    df: pd.DataFrame,
    title: str,
    show: bool = True,
    savefig: bool = True,
    fname: str = "cpu-KNN",
):
    df.plot.barh(figsize=(6, 4), grid=True)

    plt.legend()
    plt.tight_layout()
    plt.xscale("log")
    plt.xlabel("Time (in seconds)")
    plt.title(title)
    save_plot(savefig=savefig, show=show, fname=f"outputs/benchmarks/{fname}.pdf")


def run_benchmark(
    device: str = "cpu", query_repeat: int = 100, num_bits: int = 10, **plot_kwargs
):
    """
    Run benchmarks for CPU / GPU

    :param device: CPU / GPU
    :param query_repeat: Query Points = 2^num_bits x query_repeat
    :param num_bits: Message dimensions
    :return:
    """
    assert device in ["cpu", "gpu"]

    bch_tuple_ll = [
        metaclass.BCH(31, 11, 5),
        metaclass.BCH(63, 10, 13),
        metaclass.BCH(127, 15, 27),
        metaclass.BCH(255, 13, 59),
    ]

    benchmark_dict = {}

    minimum_distance = gpu_minimum_distance if device == "gpu" else cpu_minimum_distance

    for bch_tuple in bch_tuple_ll:
        logger.info(
            f"Dataset from {bch_tuple}, {pow(2, num_bits) * query_repeat} queries"
        )
        x, y, gt_indices = bch_dataset_query_points(bch_tuple, num_bits, query_repeat)

        # Benchmark
        timing_dict = minimum_distance(x, y, gt_indices, bch_tuple=bch_tuple)

        timing_dict = {k: v["querying_time"] for k, v in timing_dict.items()}
        benchmark_dict[f"{bch_tuple}"] = timing_dict
        logger.info("\n")

    cols = benchmark_dict[f"{bch_tuple}"].keys()

    # Dataframe
    df = pd.DataFrame.from_dict(
        benchmark_dict, orient="index", columns=cols
    ).transpose()
    df = df[::-1]

    # Plotting and saving
    num_queries = pow(2, num_bits) * query_repeat
    fname = f"{device}-KNN[Queries {num_queries:,}]"

    df.to_csv(f"outputs/benchmarks/{fname}.csv")

    plot_timing(
        df,
        title=f"Querying {num_queries:,} Points",
        fname=fname,
        **plot_kwargs,
    )


if __name__ == "__main__":
    run_benchmark("cpu", query_repeat=100)

    if GPU_AVAILABLE:
        run_benchmark("gpu", query_repeat=1024, show=False)
