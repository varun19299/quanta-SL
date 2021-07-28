"""
Checks for GPU support (for various packages)
"""

from loguru import logger
import pykeops
import faiss

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

GPU_AVAILABLE = CUPY_INSTALLED or KEOPS_GPU_INSTALLED or FAISS_GPU_INSTALLED
