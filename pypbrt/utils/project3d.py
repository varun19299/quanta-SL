"""
3D Projection utilities

TODO:
1. Generate Ray from camera pixels
2. Generate Plane from a pixel
"""

from dataclasses import dataclass
from einops import rearrange, repeat
import numpy as np
from pypbrt.utils import conversions, lookat, ops

from typing import Any
from nptyping import NDArray


@dataclass
class CameraMatrix(object):
    K: NDArray[(3, 3), float] = np.eye(3)
    R: NDArray[(3, 3), float] = np.eye(3)
    T: NDArray[(3), float] = np.zeros((3,))


def _add_axis(vec: NDArray[Any]) -> NDArray[(1, Any)]:
    if len(vec.shape) == 1:
        vec = rearrange(vec, "d -> 1 d")
    return vec


def _batch_dot_prod(a: NDArray[(Any, Any)], b: NDArray[(Any, Any)]):
    return (a * b).sum(axis=1)


def get_ray_through_pixel(pixel: NDArray[(Any, 2), float], camera_matrix: CameraMatrix):
    """
    Get ray (in world coord) through a camera's pixel

    :param pixel: Batched pixels, is implicitly batched otherwise
    :param camera_matrix: Camera Matrix K, R, T
    :return:
    """
    pixel = _add_axis(pixel)

    # Convert to homogenous, shape: N x 3
    pixel = conversions.convert_points_to_homogeneous(pixel)

    # Get ray in camera frame
    ray_dir = pixel @ np.linalg.inv(camera_matrix.K).T

    # Translate ray to world frame: rotate via cam rotation
    ray_dir = ray_dir @ camera_matrix.R

    ray_start = -camera_matrix.R.T @ rearrange(camera_matrix.T, "d-> d 1")
    ray_start = repeat(ray_start, "d 1 -> n d", n=pixel.shape[0])

    return ray_dir, ray_start


def intersect_ray_plane(
    ray_start: NDArray[(Any, 3), float],
    ray_dir: NDArray[(Any, 3), float],
    plane_coeff: NDArray[(Any, 4), float],
) -> NDArray[(Any, 3), float]:
    """
    Intersect ray with a plane in 3D
    :param ray_start: batched ray_start, if singular is batched
    :param ray_dir: batched ray_dir, if singular is batched
    :param plane_coeff: batched plane_coeff, if singular is batched
    :return: Coordinates of intersection, batched.
        NaN if ray doesnt intersect plane.
    """
    ray_start = _add_axis(ray_start)
    ray_dir = _add_axis(ray_dir)
    plane_coeff = _add_axis(plane_coeff)

    assert ray_start.shape[0] == ray_dir.shape[0] == plane_coeff.shape[0]
    assert ray_start.shape[1] == ray_dir.shape[1] == 3
    assert plane_coeff.shape[1] == 4

    t = np.zeros(ray_start.shape[:1])

    plane_normal = plane_coeff[:, :3]
    denom = _batch_dot_prod(plane_normal, ray_dir)

    invalid_denom_mask = np.abs(denom) < 1e-8

    # t is the distance across the ray
    t[~invalid_denom_mask] = -_batch_dot_prod(
        plane_normal[~invalid_denom_mask, :], ray_start[~invalid_denom_mask, :]
    )
    t[~invalid_denom_mask] -= plane_coeff[~invalid_denom_mask, 3]

    t[~invalid_denom_mask] /= denom[~invalid_denom_mask]

    invalid_t_mask = t <= 0
    invalid_mask = np.logical_or(invalid_t_mask, invalid_denom_mask)

    # Use NaN to mark where ray wont intersect
    # Hacky, but easiest?
    t[invalid_mask] = np.nan

    return np.squeeze(t * ray_dir + ray_start)


def test_intersect_ray_plane():
    # z = 0
    plane_coeff = np.array([0, 0, 1, 0])

    # -ve z
    ray_dir = np.array([0, 0, -1])

    # start at (0,0,2)
    ray_start = np.array([0, 0, 2])

    print(intersect_ray_plane(ray_start, ray_dir, plane_coeff))

    plane_coeff = repeat(plane_coeff, "d ->n d", n=3)
    ray_dir = repeat(ray_dir, "d ->n d", n=3)
    ray_start = repeat(ray_start, "d ->n d", n=3)

    print(intersect_ray_plane(ray_start, ray_dir, plane_coeff))


def test_camera_ray_through_pixel():
    # LookAt
    camera_loc = lookat.LookAt(
        pos=np.array([4, 0, 0]), look=np.zeros((3,)), up=np.array([0, 0, 1])
    )

    extrinsic_mat = lookat.lookat_to_Tinv(camera_loc)

    R = extrinsic_mat[:3, :3]
    T = extrinsic_mat[:3, 3]

    K = np.eye(3)
    # Assume focal length of 10cm, pixel size of 1 micron
    K[0, 0] = K[1, 1] = 1e5

    # Assume (1024,768) size image
    center_pixel = np.array([512, 384])
    K[:2, 2] = center_pixel

    camera_mat = CameraMatrix(K, R, T)

    print(camera_mat)
    ray_dir, ray_start = get_ray_through_pixel(center_pixel, camera_mat)

    # Ray through camera centre must hit look,
    # must start at camera centre
    assert (ray_start == camera_loc.pos).all()
    assert (
        ops.normalized(ray_dir) == ops.normalized(camera_loc.look - camera_loc.pos)
    ).all()


if __name__ == "__main__":
    test_intersect_ray_plane()
    test_camera_ray_through_pixel()
