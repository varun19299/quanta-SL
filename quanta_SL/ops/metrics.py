"""
Error Metrics
"""
from nptyping import NDArray

from quanta_SL.utils.decorators import named_func


@named_func("P(error)", "Error Probability")
def exact_error(decoded_index: NDArray[int], gt_index: NDArray[int]):
    """
    Must exactly match
    """
    return decoded_index != gt_index


@named_func("RMSE", "Root Mean Square Error", lambda x: x ** 0.5)
def root_mean_squared_error(decoded_index: NDArray[int], gt_index: NDArray[int]):
    return (decoded_index - gt_index) ** 2


@named_func("MAE", "Mean Absolute Error")
def absolute_error(decoded_index: NDArray[int], gt_index: NDArray[int]):
    return (decoded_index - gt_index).abs()
