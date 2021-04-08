from dataclasses import astuple
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from omegaconf import ListConfig
import re
from pypbrt.utils import lookat, project3d


def get_list_repr(ll: List) -> str:
    """
    Convert list to string
    (which represents a List in PBRT files).

    [a, b, c] -> "[ a b c ]"
    :param ll: python list to convert
    :return: str representing list
    """
    ll = [str(e) for e in ll]
    _str = " ".join(ll)
    return f"[ {_str} ]"


def get_lookat(contents: str, lookat_directive: str = "LookAt") -> lookat.LookAt:
    """
    Get LookAt directive and arguments

    :param contents: .pbrt file
    :return: LookAt instance
    """
    # Extract camera coords
    regexp = re.compile(
        rf"{lookat_directive} ([-\d. ]+) *#?.*?\n([\d. ]+) *#?.*?\n([\d. ]+) *#?.*?"
    )
    pose = [group.strip(" ").split(" ") for group in regexp.search(contents).groups()]
    pose = [np.array([float(x) for x in coord]) for coord in pose]
    pose = lookat.LookAt(*pose)
    return pose


def get_intrinsic_matrix(
    contents: str,
    camera_directive: str = 'Camera "perspective"',
    film_directive: str = 'Film "rgb"',
) -> project3d.CameraMatrix:
    """
    Get intrinsic matrix (K) from pbrt file.
    Works for projector too (treats as an inverse camera).
    See https://en.wikipedia.org/wiki/Camera_resectioning for notation.

    Method:
        1. Extract FOV, sensor dimensions (physical), sensor resolution
        2. Infer focal length from FOV, sensor dimension
        3. Assume camera centre at image centre (in X, Y).

    Camera coordinate system:
        http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT9/node2.html

    :param contents: pbrt file
    :param camera_directive: Camera "perspective", LightSource "projection", etc.
        Used to get FOV.
    :param film_directive: Film "rgb" etc.
        Used to get image resolution, sensor size.
    :return:
    """
    regexp = re.compile(rf'{camera_directive} *[\n\S ]*?"float fov" \[([\d. ]+)\]\n')
    fov = float(regexp.search(contents).group(1))

    # Film is 35mm by default
    regexp = re.compile(rf'{film_directive} *[\n\S ]*?"float diagonal" \[([\d. ]+)\]\n')
    diagonal = (
        35e-3 if not regexp.search(contents) else regexp.search(contents).group(1)
    )

    # Width, height in pixels
    regexp = re.compile(
        rf'{film_directive} *[\n\S ]*?"integer xresolution" \[([\d. ]+)\]\n'
    )
    width = float(regexp.search(contents).group(1))

    regexp = re.compile(
        rf'{film_directive} *[\n\S ]*?"integer yresolution" \[([\d. ]+)\]\n'
    )
    height = float(regexp.search(contents).group(1))

    # Focal length from FOV, Film size
    # film_width^2 + film_height^2 = diagonal^2
    aspect_ratio = width / height
    film_height = diagonal / np.sqrt(1 + aspect_ratio ** 2)
    film_width = film_height * aspect_ratio

    # FOV = 2 * arctan(shorter_side / 2f)
    # f = shorter_side / (2 * tan(FOV/2))
    focal_length = min(film_height, film_width) / (2 * np.tan(np.deg2rad(fov / 2)))

    # See coordinate system here
    # http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT9/node2.html
    focal_length_in_pixels_u = focal_length / film_width * width
    focal_length_in_pixels_v = focal_length / film_height * height

    K = np.eye(3)

    # Main diagonal
    K[0, 0] = focal_length_in_pixels_u
    K[1, 1] = focal_length_in_pixels_v

    # Assume camera centre directly above image centre
    K[:2, 2] = [width / 2, height / 2]

    # TODO: allow projector to have different resolution.
    # Typically tends to be much lower.
    if "projector" in camera_directive:
        logging.warning(
            f"We assume projector and camera have same resolution of {width} x {height}. Values taken from Film directive."
        )

    return K


def get_camera_projector_matrices(
    contents: str,
) -> Tuple[project3d.CameraMatrix, project3d.CameraMatrix]:
    """
    Get camera, projector matrices (intrinsic & extrinsic) from PBRT file.

    :param contents: pbrt file
    :return: Camera Matrix, Projector Matrix
    """
    # Camera Matrix
    K = get_intrinsic_matrix(contents)
    camera_pos = get_lookat(contents)

    extrinsic_mat = lookat.lookat_to_Tinv(camera_pos)
    R = extrinsic_mat[:3, :3]
    T = extrinsic_mat[:3, 3]

    camera_matrix = project3d.CameraMatrix(K, R, T)

    # Projector Matrix
    K = get_intrinsic_matrix(contents, camera_directive='LightSource "projection"')
    projector_pos = get_lookat(contents, lookat_directive="LookAt_camcoord")
    extrinsic_mat = lookat.lookat_to_Tinv(projector_pos)
    R = extrinsic_mat[:3, :3]
    T = extrinsic_mat[:3, 3]

    projector_matrix = project3d.CameraMatrix(K, R, T)

    return camera_matrix, projector_matrix


def parse_lookat_camcoord(contents: str) -> str:
    """
    Convert LookAt_camcoord to LookAt in camera frame

    :param contents: pbrt file
    :return: parsed pbrt file
    """
    camera_pos = get_lookat(contents, lookat_directive="LookAt")
    logging.debug(f"Camera coords in world frame {camera_pos}")

    # Extract LookAt_camcoord values
    proj_pos = get_lookat(contents, lookat_directive="LookAt_camcoord")
    logging.debug(f"Proj coords in world frame {proj_pos}")

    # Rewrite wrt to cam-coords
    proj_pos = lookat.lookat_camcoord(proj_pos, camera_pos)
    proj_pos = astuple(proj_pos)
    proj_pos = [[str(x) for x in coord] for coord in proj_pos]
    proj_pos = [" ".join(coord) for coord in proj_pos]
    logging.debug(f"Proj coords in camera frame {proj_pos}")
    regexp = re.compile(r"LookAt_camcoord .*?( #.*?\n *).*?( #.*?\n *).*?( #.*?\n)")
    contents = regexp.sub(
        f"LookAt {proj_pos[0]}\g<1>{proj_pos[1]}\g<2>{proj_pos[2]}\g<3>",
        contents,
    )

    return contents


def parse_material(contents: str, material_dict: Dict) -> str:
    """
    Parse material.sub directive with chosen material

    :param contents: pbrt file
    :param material_dict: Dict/DictConfig describing material
    :return: parsed pbrt file
    """
    material_name = material_dict["name"]
    is_named_material = material_dict.get("is_named_material", False)

    if is_named_material:
        label = material_dict["label"]
        material_str = f'MakeNamedMaterial "{label}" "string type" "{material_name}"'
    else:
        material_str = f'Material "{material_name}"'

    for key, value in material_dict.items():
        if key in ["name", "label"]:
            continue
        attr_type = value["type"]
        attr_value = value["value"]

        if isinstance(attr_value, ListConfig):
            attr_value = get_list_repr(attr_value)
        elif isinstance(attr_value, str):
            attr_value = f'"{attr_value}"'

        material_str += f'\n\t"{attr_type} {key}" {attr_value}'

    contents = contents.replace("[ material.sub ]", material_str)
    return contents


def parse_paths(contents: str, projector_path: Path, output_path: Path) -> str:
    """
    Parse output.sub, pattern.sub

    :param contents: pbrt file
    :param projector_path: path to projector pattern (relative to code repo)
    :param output_path: path to output file (relative to code repo)
    :return: parsed pbrt file
    """
    contents = contents.replace("pattern.sub", str(projector_path.resolve()))
    contents = contents.replace("output.sub", str(output_path.resolve()))

    return contents
