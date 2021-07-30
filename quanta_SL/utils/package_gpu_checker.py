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
    CUPY_INSTALLED = True
    logger.debug(
        f"CuPy installation found, with {cp.cuda.runtime.getDeviceCount()} GPU(s)."
    )

except ImportError:
    import numpy as np

    xp = np
    CUPY_INSTALLED = False
    logger.debug("No CuPy installation detected. Using Numpy.")

# KeOps
KEOPS_GPU_INSTALLED = pykeops.config.gpu_available

# FAISS
FAISS_GPU_INSTALLED = faiss.get_num_gpus() > 0

# Numba
NUMBA_GPU_INSTALLED = cuda.is_available()

GPU_AVAILABLE = (
    CUPY_INSTALLED or KEOPS_GPU_INSTALLED or FAISS_GPU_INSTALLED or NUMBA_GPU_INSTALLED
)


def free_cupy_gpu():
    if CUPY_INSTALLED:
        mempool = xp.get_default_memory_pool()
        pinned_mempool = xp.get_default_pinned_memory_pool()

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
