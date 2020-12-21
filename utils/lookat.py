import logging
from dataclasses import astuple, dataclass
from typing import Union

import numpy as np
from einops import rearrange
from nptyping import NDArray
from scipy.spatial.transform import Rotation

from utils.conversions import (
    convert_points_from_homogeneous,
    convert_points_to_homogeneous,
)
from utils.ops import normalized


@dataclass
class LookAt(object):
    pos: NDArray[3, float]
    look: NDArray[3, float]
    up: NDArray[3, float]


def lookat_to_translate_rotate(
    coords: LookAt,
) -> Union[NDArray[3, float], NDArray[3, float], float]:
    """
    Convert look at to translation + rotation along an axis
    :param pos:
    :param look:
    :param up:
    :return:
    """
    pos, look, up = astuple(coords)
    dir = normalized(look - pos)

    up = normalized(coords.up)

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

    rot_axis = normalized(rot_vector)
    theta = np.rad2deg(np.linalg.norm(rot_vector))

    return pos, rot_axis, theta


def lookat_to_Tinv(coords: LookAt,) -> NDArray[(4, 4), float]:
    """
    Convert look at to inverse Transformation matrix (4x4)
    :param pos:
    :param look:
    :param up:
    :return:
    """
    pos, look, up = astuple(coords)
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

    Tinv = np.block([[rot_mat_inv, pos], [np.zeros((1, 3)), 1]])

    return Tinv


def lookat_camcoord(obj_coords: LookAt, cam_coords: LookAt) -> LookAt:
    """
    vec in camera coord
    :param vec:
    :param cam_pos:
    :param cam_look:
    :param cam_up:
    :return:
    """
    Tinv = lookat_to_Tinv(cam_coords)

    def _proj3d(Tmat, vec):
        vec = convert_points_to_homogeneous(vec)
        vec = rearrange(vec, "n -> n 1")
        vec = Tmat @ vec
        vec = rearrange(vec, "n 1 -> n")
        vec = convert_points_from_homogeneous(vec)
        return vec

    pos = _proj3d(Tinv, obj_coords.pos)
    look = _proj3d(Tinv, obj_coords.look)

    # Up vector only rotated, not translated
    Tinv_rot = np.zeros_like(Tinv)
    Tinv_rot[:3, :3] = Tinv[:3, :3]
    Tinv_rot[3, 3] = 1
    up = _proj3d(Tinv_rot, obj_coords.up)

    return LookAt(pos, look, up)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pos = np.array([4, 0, 0])
    look = np.array([0, 0, 0])
    up = np.array([0, 0, 1])
    cam_coords = LookAt(pos, look, up)
    print(f"Camera Coords {cam_coords}")

    Tinv = lookat_to_Tinv(cam_coords)
    obj_pos = np.array([4, 1, 0])
    obj_coords = LookAt(obj_pos, look, up)
    print(f"Object Coords {lookat_camcoord(obj_coords, cam_coords)}")
