import numpy as np
from nptyping import NDArray


def _check_valid_points(points: NDArray):
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Input type is not a numpy array. Got {type(points)}")
    if len(points.shape) not in [1, 2]:
        raise ValueError(f"Input must be a 1D / 2D array. Got {points.shape}")


def convert_points_from_homogeneous(points: NDArray, eps: float = 1e-8) -> NDArray:
    r"""Function that converts points from homogeneous to Euclidean space.

    Examples::

        >>> input = np.random.rand(3)  # 3
        >>> output = convert_points_to_homogeneous(input)  # 2

        >>> input = np.random.rand(4, 3)  # Nx3
        >>> output = convert_points_to_homogeneous(input)  # Nx2
    """
    _check_valid_points(points)

    # we check for points at infinity
    z_vec: NDArray = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    scale = np.where(np.abs(z_vec) > eps, 1 / z_vec, 1)
    return scale * points[..., :-1]


def convert_points_to_homogeneous(points: NDArray) -> NDArray:
    r"""Function that converts points from Euclidean to homogeneous space.

    Examples::

        >>> input = np.random.rand(2)  # 2
        >>> output = convert_points_to_homogeneous(input)  # 3

        >>> input = np.random.rand(4, 3)  # Nx3
        >>> output = convert_points_to_homogeneous(input)  # Nx4
    """
    _check_valid_points(points)

    if len(points.shape) == 1:
        return np.pad(points, [(0, 1)], mode="constant", constant_values=1.0)

    elif len(points.shape) == 2:
        return np.pad(points, [(0, 0), (0, 1)], mode="constant", constant_values=1.0)
