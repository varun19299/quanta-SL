"""
Error Metrics
"""
from nptyping import NDArray


def named_func(name: str, long_name: str = ""):
    def decorator(function):
        function.name = name
        if long_name:
            function.long_name = long_name
        return function

    return decorator


@named_func("P(error)", "Error Probability")
def exact_error(decoded_index: NDArray[int], gt_index: NDArray[int]):
    """
    Must exactly match
    """
    return (decoded_index != gt_index).mean(axis=-1)


@named_func("RMSE", "Root Mean Squared Error")
def squared_error(decoded_index: NDArray[int], gt_index: NDArray[int]):
    out = ((decoded_index - gt_index) ** 2).mean(axis=-1)
    return out ** 0.5


@named_func("MAE", "Mean Absolute Error")
def absolute_error(decoded_index: NDArray[int], gt_index: NDArray[int]):
    return (decoded_index - gt_index).abs().mean(axis=-1)
