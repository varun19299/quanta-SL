import galois
from galois import GF2

from vis_tools.strategies.utils import unpackbits

import numpy as np
from einops import rearrange

from tqdm import tqdm

from dataclasses import dataclass, astuple
import logging

FORMAT = "[%(filename)s-%(funcName)s():%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

try:
    import cupy as cp
    xp = cp
    CUPY_INSTALLED = True
    logging.info(
        f"CuPy installation found, with {cp.cuda.runtime.getDeviceCount()} GPU(s)."
    )

except ImportError:
    xp = np
    CUPY_INSTALLED = False
    logging.warning("No CuPy installation detected. Using Numpy, may be slow.")


@dataclass
class _BCHTuple:
    n: int
    k: int
    LECC: int


if __name__ == "__main__":
    # Corrects upto 13 errors
    # LECC 19

    bch_ll = [
        _BCHTuple(15, 11, 1),
        _BCHTuple(31, 11, 7),
        _BCHTuple(63, 10, 17),
        _BCHTuple(127, 15, 38),
        _BCHTuple(127, 8, 45),
    ]
    index = 2

    n, k, LECC = astuple(bch_ll[index])
    bch = galois.BCH(n, k)

    # What is the probability of incorrect list decoding at LECC?
    message_ll = np.arange(pow(2, bch.k))
    message_ll = GF2(unpackbits(message_ll, bch.k))
    code_ll = bch.encode(message_ll)

    code_ll = xp.asarray(code_ll, dtype=int)

    cum_incorrect_prob = 0
    error = xp.zeros(code_ll.shape, dtype=int)

    pbar = tqdm(range(100_000))

    logging.info(f"BCH code {bch_ll[index]}")

    for i in pbar:
        # Zero error matrix
        error ^= error

        col_indices = xp.random.choice(code_ll.shape[1], LECC, replace=False)
        error[:, col_indices] = 1

        corrupted_code_ll = code_ll ^ error
        # Try MLE recovery
        distance = rearrange(corrupted_code_ll, "N n -> N n 1") ^ rearrange(
            code_ll, "N n -> 1 n N"
        )
        distance = distance.sum(axis=1)
        min_dist_vector = xp.argmin(distance, axis=1)

        recovered_code_ll = code_ll[min_dist_vector, :]

        incorrect_prob = xp.equal(recovered_code_ll, code_ll)

        # Cumulative moving average
        # https://en.wikipedia.org/wiki/Moving_average
        cum_incorrect_prob += (incorrect_prob - cum_incorrect_prob) / (i + 1)

        pbar.set_description(f"Success probability {cum_incorrect_prob.mean():.6f}")
