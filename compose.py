"""
1. Create .pbrt file in dump
2. Render with pbrt
3. Compute correspondences
"""

import hydra
import logging
from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path
import os
import subprocess

from typing import Dict, List, Union


def get_list_repr(ll: List):
    ll = [str(e) for e in ll]
    _str = " ".join(ll)
    return f"[ {_str} ]"


def set_paths(cfg: DictConfig) -> Union[Path, Path, Path]:
    root_dir = Path(hydra.utils.get_original_cwd())
    tag = cfg.pbrt.type

    pbrt_path = root_dir / "scenes" / cfg.pbrt.scene
    pbrt_path = pbrt_path / f"{tag}.pbrt"

    projector_path = (
        root_dir / "patterns" / cfg.projector.pattern / f"{cfg.projector.index}.exr"
    )

    output_path = Path(f"{cfg.output.name}_{cfg.projector.index}.exr")

    return pbrt_path, projector_path, output_path


def pbrt_configure_paths(contents: str, projector_path: Path, output_path: Path) -> str:
    contents = contents.replace("pattern.sub", str(projector_path.resolve()))
    contents = contents.replace("output.sub", str(output_path.resolve()))

    return contents


def pbrt_modify_material(contents: str, material_dict: Dict) -> str:
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


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    tag = cfg.pbrt.type
    pbrt_path, projector_path, output_path = set_paths(cfg)

    with open(pbrt_path) as f:
        pbrt_file = f.read()

    # Put in output render, projector image paths
    pbrt_file = pbrt_configure_paths(pbrt_file, projector_path, output_path)

    # Modify material
    pbrt_file = pbrt_modify_material(pbrt_file, cfg.material)

    # Write file
    pbrt_dump_path = (
        pbrt_path.parent
        / f"dump_{tag}_{projector_path.parent.name}_{projector_path.name}.pbrt"
    )
    with open(pbrt_dump_path, "w") as f:
        f.write(pbrt_file)

    # Render it
    subprocess.run([cfg.pbrt.executable, str(pbrt_dump_path.resolve())])

    # Remove pbrt file
    os.remove(pbrt_dump_path)


if __name__ == "__main__":
    main()
