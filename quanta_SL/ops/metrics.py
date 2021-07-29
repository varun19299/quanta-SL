"""
Error Metrics
"""
from numba import vectorize, float64, int64


@vectorize(
    [int64(int64, int64), float64(float64, float64)], cache=True, target="parallel"
)
def exact_error(decoded_index: int, gt_index: int):
    """
    Must exactly match
    """
    return decoded_index != gt_index


@vectorize(
    [int64(int64, int64), float64(float64, float64)],
    cache=True,
    fastmath=True,
    target="parallel",
)
def squared_error(decoded_index: int, gt_index: int):
    return (decoded_index - gt_index) ** 2


@vectorize(
    [int64(int64, int64), float64(float64, float64)],
    cache=True,
    fastmath=True,
    target="parallel",
)
def absolute_error(decoded_index: int, gt_index: int):
    return abs(decoded_index - gt_index)
