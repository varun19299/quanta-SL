from vis_tools.strategies.metaclass import BCH


def list_error_correcting_capacity(bch_tuple: BCH) -> int:
    """
    Based on BCH LECC limit.
    See Wu et al. 2008, https://arxiv.org/pdf/cs/0703105.pdf.

    :param bch_tuple: Describes BCH code as [n, k, t]
        n: Code length
        k: Message length
        t: Worst case correctable errors
    :return: List Error Correcting Capacity
    """
    n = bch_tuple.n
    d = bch_tuple.distance
    return n / 2 * (1 - (1 - 2 * d / n) ** 0.5)
