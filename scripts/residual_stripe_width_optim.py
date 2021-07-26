import itertools
from tqdm import tqdm
import galois
from galois import GF2
from quanta_SL.encode import binary_message, gray_message, metaclass
from quanta_SL.ops.math_func import fast_factorial
from quanta_SL.ops.binary import packbits
from quanta_SL.ops import stripe_width_stats

import numpy as np

if __name__ == "__main__":
    # BCH codes
    bch_tuple = metaclass.BCH(7, 4, 1)
    num_bits = 3

    bch = galois.BCH(bch_tuple.n, bch_tuple.k)
    binary_bch_codes = bch.encode(GF2(binary_message(num_bits)))
    binary_bch_codes = binary_bch_codes.view(np.ndarray).astype(int)

    message_ll = gray_message(num_bits)

    # Try permuting
    pbar = tqdm(total=fast_factorial(pow(2, num_bits)))

    most_acceptable_dict = {
        "num": 0,
        "perm": [],
        "min_stripe_ll": [],
        "mean_stripe_ll": [],
    }

    update_interval = 100

    for e, perm in enumerate(itertools.permutations(range(message_ll.shape[0]))):
        pbar.update(1)

        # Generate BCH_matlab codes
        perm_message = packbits(message_ll[perm, :])
        code_LUT = binary_bch_codes[perm_message, :]

        min_stripe, min_stripe_ll, mean_stripe_ll = stripe_width_stats(code_LUT)

        acceptable = len([stripe for stripe in min_stripe_ll if stripe >= 2])

        if acceptable >= most_acceptable_dict["num"]:
            most_acceptable_dict.update(
                {
                    "num": acceptable,
                    "perm": perm,
                    "min_stripe_ll": min_stripe_ll,
                    "mean_stripe_ll": mean_stripe_ll,
                }
            )

        if e % update_interval == 0:
            pbar.set_description(
                f"Min Stripe {min_stripe} | Max Acceptable so far {most_acceptable_dict['num']}"
            )

        if min_stripe > 1:
            break
