"""
3D Projection utilities
"""

from einops import rearrange, repeat
import numpy as np
from pypbrt.utils import lookat

from typing import Any
from nptyping import NDArray


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

    def _add_axis(vec: NDArray[3]) -> NDArray[(1, 3)]:
        if len(vec.shape) == 1:
            vec = rearrange(vec, "d -> 1 d")
        return vec

    def _batch_dot_prod(a: NDArray[(Any, Any)], b: NDArray[(Any, Any)]):
        return (a * b).sum(axis=1)

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
    ray_start = np.array([0, 0, -2])

    print(intersect_ray_plane(ray_start, ray_dir, plane_coeff))

    plane_coeff = repeat(plane_coeff, "d ->n d", n=3)
    ray_dir = repeat(ray_dir, "d ->n d", n=3)
    ray_start = repeat(ray_start, "d ->n d", n=3)

    print(intersect_ray_plane(ray_start, ray_dir, plane_coeff))


if __name__ == "__main__":
    test_intersect_ray_plane()
