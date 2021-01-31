"""
3D Projection utilities

TODO:
1. Check coordinate system. Should be Image[y,x]

"""

from dataclasses import dataclass
from einops import rearrange, repeat
import numpy as np
from pypbrt.utils import conversions, lookat, ops

from typing import Any, Tuple
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


def _batch_dot_prod(a: NDArray[(Any, Any)], b: NDArray[(Any, Any)]) -> NDArray[(Any,)]:
    """
    First axis is batch
    """
    return (a * b).sum(axis=1)


def plane_through_pixel(
    pixel: NDArray[(Any, 2), float], camera_matrix: CameraMatrix
) -> Tuple[NDArray[(Any, 4), float], NDArray[(Any, 4), float]]:
    """
    Get planes through x and y directions (of camera coord)

    :param pixel: Batched pixels, is implicitly batched otherwise
        Format: [[pixel_u, pixel_v]]
        https://learnopencv.com/wp-content/uploads/2020/02/camera-projection-3D-to-2D.png
    :param camera_matrix: Camera Matrix K, R, T
    :return:
    """
    pixel = _add_axis(pixel)

    # Convert to homogenous, shape: N x 3
    pixel = conversions.convert_points_to_homogeneous(pixel)

    x_dir = np.array([1, 0, 0])
    y_dir = np.array([0, 1, 0])

    # Get ray in camera frame
    ray_dir = pixel @ np.linalg.inv(camera_matrix.K).T

    # Plane normals
    plane_through_x_dir = np.cross(ray_dir, x_dir)
    plane_through_y_dir = np.cross(ray_dir, y_dir)

    # Transform dir to world coord (just rotate)
    plane_through_x_dir = plane_through_x_dir @ camera_matrix.R
    plane_through_y_dir = plane_through_y_dir @ camera_matrix.R

    # Camera Centre in world coord
    camera_centre = camera_matrix.R.T @ np.zeros(
        (3, 1)
    ) - camera_matrix.R.T @ rearrange(camera_matrix.T, "d-> d 1")
    camera_centre = repeat(camera_centre, "d 1 -> n d", n=pixel.shape[0])

    plane_through_x = np.block(
        [plane_through_x_dir, -_batch_dot_prod(plane_through_x_dir, camera_centre)]
    )
    plane_through_y = np.block(
        [plane_through_y_dir, -_batch_dot_prod(plane_through_y_dir, camera_centre)]
    )

    # Plane through u const is ydir
    # Plane through v const is xdir
    return plane_through_y, plane_through_x


def ray_through_pixel(
    pixel: NDArray[(Any, 2), float], camera_matrix: CameraMatrix
) -> Tuple[NDArray[(Any, 3), float], NDArray[(Any, 3), float]]:
    """
    Get ray (in world coord) through a camera's pixel

    :param pixel: Batched pixels, is implicitly batched otherwise
        Format: [[pixel_u, pixel_v]]
        https://learnopencv.com/wp-content/uploads/2020/02/camera-projection-3D-to-2D.png
    :param camera_matrix: Camera Matrix K, R, T
    :return: ray_start, ray_dir
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

    return ray_start, ray_dir


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


def triangulate_ray_plane(
    camera_matrix: CameraMatrix,
    projector_matrix: CameraMatrix,
    camera_pixel: NDArray[(Any, 2), float],
    projector_pixel: NDArray[(Any, 2), float],
    axis: str = 0,
):
    """
    Triangulate a given camera pixel
    with known projector column

    :param camera_matrix: Camera Matrix K, R, T
    :param projector_matrix: Projector Matrix K, R, T
    :param camera_pixel: Batched pixels, is implicitly batched otherwise
        Format: [[pixel_u, pixel_v]]
        https://learnopencv.com/wp-content/uploads/2020/02/camera-projection-3D-to-2D.png
    :param projector_pixel: Batched pixels, is implicitly batched otherwise
        Format: [[pixel_u, pixel_v]]. Use 0 for axis not considered
    :param axis: \in {0,1}. vertical or horizontal columns
    :return:
    """
    assert axis in [0, 1]
    planes = plane_through_pixel(projector_pixel, projector_matrix)
    ray_start, ray_dir = ray_through_pixel(camera_pixel, camera_matrix)
    print(ray_start, ray_dir)
    print(planes)
    return intersect_ray_plane(ray_start, ray_dir, planes[axis])


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


def _init_camera(pos: NDArray[3, float] = np.array([4, 0, 0])):
    # LookAt
    camera_loc = lookat.LookAt(pos=pos, look=np.zeros((3,)), up=np.array([0, 0, 1]))

    extrinsic_mat = lookat.lookat_to_Tinv(camera_loc)

    R = extrinsic_mat[:3, :3]
    T = extrinsic_mat[:3, 3]

    K = np.eye(3)
    # Assume focal length of 10cm, pixel size of 1 micron
    K[0, 0] = K[1, 1] = 1e5

    # Assume (1024 x 768) size image
    # width 1024
    # height 768
    center_pixel = np.array([512, 384])
    K[:2, 2] = center_pixel

    camera_mat = CameraMatrix(K, R, T)
    return camera_loc, camera_mat, center_pixel


def test_ray_through_pixel():
    camera_loc, camera_mat, center_pixel = _init_camera()
    print(f"Camera Matrix {camera_mat}")
    ray_start, ray_dir = ray_through_pixel(center_pixel, camera_mat)

    print(f"Ray through {center_pixel}: dir {ray_dir} start {ray_start}\n")
    # Ray through camera centre must hit look,
    # must start at camera centre
    assert (ray_start == camera_loc.pos).all()
    assert (
        ops.normalized(ray_dir) == ops.normalized(camera_loc.look - camera_loc.pos)
    ).all()


def test_plane_through_pixel():
    camera_loc, camera_mat, center_pixel = _init_camera()
    print(f"Camera Matrix {camera_mat}")
    plane_x, plane_y = plane_through_pixel(center_pixel + 1024, camera_mat)

    print(f"Plane through [{center_pixel[0]}, Any]: {plane_x}")
    print(f"Plane through [Any, {center_pixel[1]}]: {plane_y}\n")

    # Camera centre must lie on planes
    camera_loc_pos = conversions.convert_points_to_homogeneous(camera_loc.pos)
    assert np.dot(np.squeeze(plane_x), camera_loc_pos) == 0
    assert np.dot(np.squeeze(plane_y), camera_loc_pos) == 0


def test_triangulate_ray_plane():
    camera_loc, camera_mat, camera_center_pixel = _init_camera(pos=np.array([4, 0, 0]))
    projector_loc, projector_mat, projector_center_pixel = _init_camera(
        pos=np.array([4, 5, 0])
    )

    print(f"Camera Matrix {camera_mat}")
    print(f"Projector Matrix {projector_mat}")

    intersection = triangulate_ray_plane(
        camera_mat, projector_mat, camera_center_pixel, projector_center_pixel, axis=0
    )

    print(f"Intersection {intersection}")

    # Plane (parallel y dir), ray intersect
    # Must pass through origin
    # we set up the "look" that way
    assert (intersection == np.zeros((3,))).all()


if __name__ == "__main__":
    test_intersect_ray_plane()
    test_ray_through_pixel()
    test_plane_through_pixel()
    test_triangulate_ray_plane()
