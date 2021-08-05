"""
Error Metrics
"""
from nptyping import NDArray
from typing import Callable


def named_func(name: str, long_name: str = "", post_mean_func: Callable = lambda x: x):
    def decorator(function):
        function.name = name
        if long_name:
            function.long_name = long_name

        # Any func after taking mean
        # Default is identity func
        function.post_mean_func = post_mean_func

        return function

    return decorator


@named_func("P(error)", "Error Probability")
def exact_error(decoded_index: NDArray[int], gt_index: NDArray[int]):
    """
    Must exactly match
    """
    return decoded_index != gt_index


@named_func("RMSE", "Root Mean Squared Error", lambda x: x ** 0.5)
def root_mean_squared_error(decoded_index: NDArray[int], gt_index: NDArray[int]):
    return (decoded_index - gt_index) ** 2


@named_func("MAE", "Mean Absolute Error")
def absolute_error(decoded_index: NDArray[int], gt_index: NDArray[int]):
    return (decoded_index - gt_index).abs()
