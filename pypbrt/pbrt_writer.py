from dataclasses import astuple
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List

from omegaconf import ListConfig
import re
from utils.lookat import LookAt, lookat_camcoord


def configure_paths(contents: str, projector_path: Path, output_path: Path) -> str:
    contents = contents.replace("pattern.sub", str(projector_path.resolve()))
    contents = contents.replace("output.sub", str(output_path.resolve()))

    return contents


def modify_material(contents: str, material_dict: Dict) -> str:
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


def parse_lookat_camcoord(contents: str) -> str:
    # Extract camera coords
    regexp = re.compile(r"LookAt (.*?) #.*?\n *(.*?) #.*?\n *(.*?) #.*?\n")
    camera_coords = [group.split(" ") for group in regexp.search(contents).groups()]
    camera_coords = [np.array([float(x) for x in coord]) for coord in camera_coords]
    camera_coords = LookAt(*camera_coords)
    logging.debug(f"Camera coords in world frame {camera_coords}")

    # Extract LookAt_camcoord values
    regexp = re.compile(r"LookAt_camcoord (.*?) #.*?\n *(.*?) #.*?\n *(.*?) #.*?\n")
    proj_coords = [group.split(" ") for group in regexp.search(contents).groups()]
    proj_coords = [np.array([float(x) for x in coord]) for coord in proj_coords]
    proj_coords = LookAt(*proj_coords)
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


def get_list_repr(ll: List):
    ll = [str(e) for e in ll]
    _str = " ".join(ll)
    return f"[ {_str} ]"
