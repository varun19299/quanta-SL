import time
from quanta_SL.utils.package_gpu_checker import CUPY_INSTALLED, xp

class Timer:
    @property
    def elapsed_time(self):
        raise NotImplementedError

    def __float__(self):
        return float(self.elapsed_time)

    def __coerce__(self, other):
        return float(self), other

    def __str__(self):
        return str(float(self))

    def __format__(self, format_spec):
        return f"{float(self) :{format_spec}}"

    def __repr__(self):
        return str(float(self))

class CPUTimer(Timer):
    """
    Context manager to measure CPU time.
    Example ::

        >>> with CPUTimer() as t:
        >>>      time.sleep(2)
        >>> t.elapsed_time # 2 ish, highest precision afforded by your machine
    """

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.end_time = time.perf_counter()

    @property
    def elapsed_time(self):
        if not hasattr(self, "end_time"):
            raise RuntimeError("`elapsed_time` attribute accessed within decorator.")
        return self.end_time - self.start_time


class CuPyTimer(Timer):
    """
    Context manager to measure CUDA Stream time.
    Example ::

        >>> with CuPyTimer() as t:
        >>>      time.sleep(2)
        >>> t.elapsed_time # 2 ish, highest precision afforded by your machine
    """

    def __init__(self):
        assert CUPY_INSTALLED, "CuPy must be installed"

    def __enter__(self):
        self.start_gpu = xp.cuda.Event()
        self.start_gpu.record()

        return self

    def __exit__(self, type, value, traceback):
        self.end_gpu = xp.cuda.Event()
        self.end_gpu.record()
        self.end_gpu.synchronize()

    @property
    def elapsed_time(self):
        if not hasattr(self, "end_gpu"):
            raise RuntimeError("`elapsed_time` attribute accessed within decorator.")
        # Convert from microseconds to seconds
        return xp.cuda.get_elapsed_time(self.start_gpu, self.end_gpu) / 1_000_000

if __name__ == "__main__":
    with CPUTimer() as t:
        time.sleep(1)

    print(f"Elapsed time {t}")

    with CuPyTimer() as t:
        time.sleep(1)

    print(f"Elapsed time {t}")