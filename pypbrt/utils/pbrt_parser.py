from dataclasses import astuple
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List

from omegaconf import ListConfig
import re
from pypbrt.utils.lookat import LookAt, lookat_camcoord


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


def get_lookat(contents: str, lookat_directive: str = "LookAt") -> LookAt:
    """
    Get LookAt directive and arguments

    :param contents: .pbrt file
    :return: LookAt instance
    """
    # Extract camera coords
    regexp = re.compile(rf"{lookat_directive} (.*?) #.*?\n *(.*?) #.*?\n *(.*?) #.*?\n")
    pose = [group.split(" ") for group in regexp.search(contents).groups()]
    pose = [np.array([float(x) for x in coord]) for coord in pose]
    pose = LookAt(*pose)
    return pose


def parse_lookat_camcoord(contents: str) -> str:
    """
    Convert LookAt_camcoord to LookAt in camera frame

    :param contents: pbrt file
    :return: parsed pbrt file
    """
    camera_coords = get_lookat(contents, lookat_directive="LookAt")
    logging.debug(f"Camera coords in world frame {camera_coords}")

    # Extract LookAt_camcoord values
    proj_coords = get_lookat(contents, lookat_directive="LookAt_camcoord")
    logging.debug(f"Proj coords in world frame {proj_coords}")

    # Rewrite wrt to cam-coords
    proj_coords = lookat_camcoord(proj_coords, camera_coords)
    proj_coords = astuple(proj_coords)
    proj_coords = [[str(x) for x in coord] for coord in proj_coords]
    proj_coords = [" ".join(coord) for coord in proj_coords]
    logging.debug(f"Proj coords in camera frame {proj_coords}")
    regexp = re.compile(r"LookAt_camcoord .*?( #.*?\n *).*?( #.*?\n *).*?( #.*?\n)")
    contents = regexp.sub(
        f"LookAt {proj_coords[0]}\g<1>{proj_coords[1]}\g<2>{proj_coords[2]}\g<3>",
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
