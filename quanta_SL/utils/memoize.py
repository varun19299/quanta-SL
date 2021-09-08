import atexit
import inspect
import shelve
from pathlib import Path

Path(".cache").mkdir(exist_ok=True, parents=True)


class memoize(object):
    def __init__(self, func):
        self.func = func
        self.cache = shelve.open(f".cache/{func.__name__}", "c")
        atexit.register(self.cache.close)

    def __call__(self, *args, **kwargs):
        key = self.key(args, kwargs)
        if key not in self.cache:
            self.cache[key] = self.func(*args, **kwargs)
        return self.cache[key]

    def normalize_args(self, args, kwargs):
        spec = inspect.getargs(self.func.__code__).args
        return {**kwargs, **dict(zip(spec, args))}

    def key(self, args, kwargs):
        a = self.normalize_args(args, kwargs)
        return str(sorted(a.items()))


def test_memoize():
    from quanta_SL.utils.timer import CPUTimer
    from quanta_SL.io import load_swiss_spad_sequence

    Path(".cache/load_swiss_spad_sequence.db").unlink()

    # Should be slow
    with CPUTimer() as t:
        load_swiss_spad_sequence(
            Path(
                r"outputs/real_captures/LCD projector/27th August/0827-hybridBCH/pattern001"
            ),
            bin_suffix_range=range(10),
        )

    non_memoized_time = t.elapsed_time

    # Should be fast
    with CPUTimer() as t:
        load_swiss_spad_sequence(
            Path(
                r"outputs/real_captures/LCD projector/27th August/0827-hybridBCH/pattern001"
            ),
            bin_suffix_range=range(10),
        )

    memoized_time = t.elapsed_time

    assert (
        memoized_time < non_memoized_time
    ), "Memoization is slow. Either func is fast or issue in memoization."
