import networkx as nx
from utils.numba_converter import python_to_numba_dict
from numba import njit
from numba.typed import List
import numba
from utils.profiler import profile

from vis_tools.strategies import metaclass
from utils.array_ops import unpackbits

import galois

import numpy as np

import logging

FORMAT = "%(asctime)s [%(filename)s : %(funcName)2s() : %(lineno)2s] %(message)s"
logging.basicConfig(format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)


# @profile
@njit(cache=True, fastmath=True)
def find_longest_path(traversal_dict):
    longest_path_len = 0
    longest_path = List.empty_list(numba.types.int64)

    for s in traversal_dict:
        for t in traversal_dict[s]:
            if len(traversal_dict[s][t]) > longest_path_len:
                longest_path = traversal_dict[s][t]
                longest_path_len = len(traversal_dict[s][t])

    return longest_path


def bch_longest_path(bch_tuple: metaclass.BCH, num_bits: int = 11):
    message_ll = unpackbits(np.arange(pow(2, num_bits)))

    # Messages are binary codes
    bch = galois.BCH(bch_tuple.n, bch_tuple.k)
    code_LUT = bch.encode(galois.GF2(message_ll))
    code_LUT = code_LUT.view(np.ndarray).astype(int)

    G = nx.Graph()

    for e, code in enumerate(code_LUT):
        G.add_node(e)
        distances = (code_LUT ^ code.reshape(1, -1)).sum(axis=-1)
        breakpoint()

        min_dist_indices = np.where(distances == 2 * bch.t + 1)[0].tolist()

        for dist_index in min_dist_indices:
            G.add_edge(e, dist_index)

        min_dist_indices = np.where(distances == 2 * bch.t + 2)[0].tolist()

        for dist_index in min_dist_indices:
            G.add_edge(e, dist_index)

    return G


if __name__ == "__main__":

    G = bch_longest_path(metaclass.BCH(15, 11, 1))

    # G = nx.binomial_tree(10)

    logging.info("Computing all paths...")
    p = nx.shortest_path(G, weight=-1)
    logging.info("Done computing paths.")

    p = python_to_numba_dict(p)

    longest_path = find_longest_path(p)
    print(longest_path)
    breakpoint()
