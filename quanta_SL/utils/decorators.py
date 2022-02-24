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
