"""
Checks for GPU support (for various packages)
"""

import faiss
import pykeops
from loguru import logger
from numba import cuda


# CuPy
# Unfortunately, can't be installed on CPU only systems
try:
    import cupy as cp

    xp = cp
    CUPY_GPUs = cp.cuda.runtime.getDeviceCount()

except ImportError:
    import numpy as np

    xp = np
    CUPY_GPUs = 0

# FAISS
FAISS_GPUs = faiss.get_num_gpus()

# KeOps
KEOPS_GPUs = pykeops.config.get_gpu_number()

# Numba
# Exception since cuda.gpus raises error for CPU systems
try:
    NUMBA_GPUs = len(cuda.gpus)
except cuda.cudadrv.error.CudaSupportError:
    NUMBA_GPUs = 0


# Logging messages
for name, var in zip(
    ["CuPy", "FAISS", "KeOps", "Numba"], [CUPY_GPUs, FAISS_GPUs, KEOPS_GPUs, NUMBA_GPUs]
):
    if var:
        logger.debug(f"{name} installed with {var} GPUs detected")
    else:
        logger.debug(f"No GPU install for {name} found.")

GPU_AVAILABLE = CUPY_GPUs or KEOPS_GPUs or FAISS_GPUs or NUMBA_GPUs


def free_cupy_gpu():
    if CUPY_GPUs:
        mempool = xp.get_default_memory_pool()
        pinned_mempool = xp.get_default_pinned_memory_pool()

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()


def move_to_gpu(array):
    """
    Move to GPU if not already
    """
    if hasattr(array, "__array_function__"):
        if not isinstance(array, xp.ndarray):
            array = xp.asarray(array)
    return array
