from einops import rearrange
import logging
import numpy as np
from nptyping import NDArray
from scipy.spatial.transform import Rotation
from typing import Union
from utils.ops import normalized, sorted_eigendecomposition


def lookat_to_translate_rotate(
    pos: NDArray[3, float], look: NDArray[3, float], up: NDArray[3, float]
) -> Union[NDArray[3, float], NDArray[3, float], float]:
    """
    Convert look at to translation + rotation along an axis
    :param pos:
    :param look:
    :param up:
    :return:
    """
    dir = normalized(look - pos)

    up = normalized(up)

    right = np.cross(up, dir)
    assert np.linalg.norm(right), f"Up {up} and dir {dir} are in the same direction"
    right = normalized(right)

    up = np.cross(dir, right)

    # Make column vectors
    dir = rearrange(dir, "1 n -> n 1")
    up = rearrange(up, "1 n -> n 1")
    right = rearrange(right, "1 n -> n 1")

    rot_mat = np.hstack((right, up, dir))
    rotation = Rotation.from_matrix(rot_mat)
    rot_vector = rotation.as_rotvec()

    print(rot_mat)
    print(rot_vector)

    rot_axis = normalized(rot_vector)
    theta = np.rad2deg(np.linalg.norm(rot_vector))

    return pos, rot_axis, theta


def lookat_to_transform_inverse(
    pos: NDArray[3, float], look: NDArray[3, float], up: NDArray[3, float]
) -> NDArray[(4, 4), float]:
    """
    Convert look at to translation + rotation along an axis
    :param pos:
    :param look:
    :param up:
    :return:
    """
    dir = normalized(look - pos)

    up = normalized(up)

    right = np.cross(up, dir)
    assert np.linalg.norm(right), f"Up {up} and dir {dir} are in the same direction"
    right = normalized(right)

    up = np.cross(dir, right)

    # Make column vectors
    dir = rearrange(dir, "1 n -> n 1")
    up = rearrange(up, "1 n -> n 1")
    right = rearrange(right, "1 n -> n 1")

    rot_mat = np.hstack((right, up, dir))
    rot_mat_inv = rot_mat.transpose()

    logging.debug(f"Rotational Matrix {rot_mat}")

    # Now compute rot_mat_inv @ pos
    pos = rearrange(pos, "n -> n 1")
    pos = -rot_mat_inv @ pos

    transform_mat = np.block([[rot_mat_inv, pos], [np.zeros((1, 3)), 1]])

    return transform_mat


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pos = np.array([4, 0, 0])
    look = np.array([0, 0, 0])
    up = np.array([0, 0, 1])

    pos, rot_axis, theta = lookat_to_translate_rotate(pos, look, up)
    print(f"pos {pos} \n rot axis {rot_axis} \n theta {theta}")

    transform_mat = lookat_to_transform_inverse(pos, look, up)
    coord = np.array([4, 1, 0, 1]).reshape(-1, 1)
    # coord = np.array([4, 0, 0, 1]).reshape(-1, 1)
    print(transform_mat @ coord)
