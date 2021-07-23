import time


class CatchTimer:
    """
    Context manager to measure time.
    Example ::

        >>> with CatchTimer() as t:
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

    def __float__(self):
        return float(self.elapsed_time)

    def __coerce__(self, other):
        return float(self), other

    def __str__(self):
        return str(float(self))

    def __repr__(self):
        return str(float(self))


if __name__ == "__main__":
    with CatchTimer() as t:
        time.sleep(1)

    print(f"Elapsed time {t}")
