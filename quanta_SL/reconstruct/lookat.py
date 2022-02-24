"""
Support for `LookAt_camcoord`

which is a `LookAt` directive
with world coordinate system.

(In PBRT, other directives such as
Rotate, Translate etc follow
world coordinates. LookAt is the exception).

So, if the camera is positioned at `cam` and
looks at `cam_look`, with up direction `up_dir`:

    ```
    LookAt_camcoord cam_x cam_y cam_z
        cam_look_x cam_look_y cam_look
        up_dir_x up_dir_y up_dir_z
    ```

colocates the object & the camera!

DONOT use before the scene description
(before WorldBegin),
will lead to unexpected behaviour.
"""
import logging
from dataclasses import astuple, dataclass
from typing import Union

import numpy as np
from einops import rearrange
from nptyping import NDArray
from scipy.spatial.transform import Rotation

from quanta_SL.reconstruct.conversions import (
    convert_points_from_homogeneous,
    convert_points_to_homogeneous,
)
from quanta_SL.ops.linalg import normalized


@dataclass
class LookAt:
    pos: NDArray[3, float]
    look: NDArray[3, float]
    up: NDArray[3, float]


def lookat_to_translate_rotmat(
    lookat: LookAt,
) -> Union[NDArray[3, float], NDArray[(3, 3), float]]:
    """
    Convert LookAt to translation vector + rotational matrix.

    Translation vector, Rot mat are from Camera (object) to World.

    :param lookat: LookAt instance holding position and orientation info
    :return: Translation vec, Rotation mat
    """

    pos, look, up = astuple(lookat)
    dir = normalized(look - pos)

    up = normalized(lookat.up)

    right = np.cross(up, dir)
    assert np.linalg.norm(right), f"Up {up} and dir {dir} are in the same direction"
    right = normalized(right)

    up = np.cross(dir, right)

    # Make column vectors
    dir = rearrange(dir, "1 n -> n 1")
    up = rearrange(up, "1 n -> n 1")
    right = rearrange(right, "1 n -> n 1")

    rot_mat = np.hstack((right, up, dir))

    return pos, rot_mat


def lookat_to_translate_rotvec(
    lookat: LookAt,
) -> Union[NDArray[3, float], NDArray[3, float], float]:
    """
    Convert LookAt to translation vector + rotation vector

    Translation, Rot vectors are from Camera (object) to World.

    :param lookat: LookAt instance holding position and orientation info
    :return: Translation vec, Rotation axis, Rotation angle
    """
    pos, rot_mat = lookat_to_translate_rotmat(lookat)
    rotation = Rotation.from_matrix(rot_mat)
    rot_vector = rotation.as_rotvec()

    rot_axis = normalized(rot_vector)
    theta = np.rad2deg(np.linalg.norm(rot_vector))

    return pos, rot_axis, theta


def lookat_to_T(
    lookat: LookAt,
) -> NDArray[(4, 4), float]:
    """
    Convert look at to Transformation matrix (4x4)

    Describes Camera (object in general) to World transformation.

    :param lookat: Position & orientation
        as described by a LookAt instance
    :return: Transformation matrix (homogeneous)
    """
    pos, rot_mat = lookat_to_translate_rotmat(lookat)

    return np.block([[rot_mat, pos], [np.zeros((1, 3)), 1]])


def lookat_to_Tinv(
    lookat: LookAt,
) -> NDArray[(4, 4), float]:
    """
    Convert look at to inverse Transformation matrix (4x4)

    Describes World to Camera (object in general) transformation.

    :param lookat: Position & orientation
        as described by a LookAt instance
    :return: Transformation matrix (homogeneous)
    """
    pos, rot_mat = lookat_to_translate_rotmat(lookat)
    rot_mat_inv = rot_mat.transpose()

    # Now compute rot_mat_inv @ pos
    pos = rearrange(pos, "n -> n 1")
    pos_inv = -rot_mat_inv @ pos

    Tinv = np.block([[rot_mat_inv, pos_inv], [np.zeros((1, 3)), 1]])

    return Tinv


def lookat_camcoord(obj_lookat: LookAt, cam_lookat: LookAt) -> LookAt:
    """
    Converts LookAt to camera coords

    :param obj_lookat: Object position & orientation described by LookAt
    :param cam_pos: Camera position & orientation described by LookAt
    :return: Object LookAt in camera coordinates
    """
    Tinv = lookat_to_Tinv(cam_lookat)

    def _proj3d(Tmat, vec):
        vec = convert_points_to_homogeneous(vec)
        vec = rearrange(vec, "n -> n 1")
        vec = Tmat @ vec
        vec = rearrange(vec, "n 1 -> n")
        vec = convert_points_from_homogeneous(vec)
        return vec

    pos = _proj3d(Tinv, obj_lookat.pos)
    look = _proj3d(Tinv, obj_lookat.look)

    # Up vector only rotated, not translated
    Tinv_rot = np.zeros_like(Tinv)
    Tinv_rot[:3, :3] = Tinv[:3, :3]
    Tinv_rot[3, 3] = 1
    up = _proj3d(Tinv_rot, obj_lookat.up)

    return LookAt(pos, look, up)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pos = np.array([4, 0, 0])
    look = np.array([0, 0, 0])
    up = np.array([0, 0, 1])
    cam_lookat = LookAt(pos, look, up)
    print(f"Camera Coords {cam_lookat}")

    Tinv = lookat_to_Tinv(cam_lookat)
    obj_pos = np.array([4, 1, 0])
    obj_lookat = LookAt(obj_pos, look, up)
    print(f"Object Coords {lookat_camcoord(obj_lookat, cam_lookat)}")
