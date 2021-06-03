"""
1. Create .pbrt file in dump
2. Render with pbrt
3. Compute correspondences
"""

import os
import subprocess
from pathlib import Path
from typing import Union

import hydra
from omegaconf import DictConfig, OmegaConf

from pypbrt.utils import pbrt_parser


def set_paths(cfg: DictConfig) -> Union[Path, Path, Path]:
    root_dir = Path(hydra.utils.get_original_cwd())

    pbrt_path = root_dir / "scenes" / cfg.pbrt.scene / f"{cfg.pbrt.filename}.pbrt"

    projector_path = (
        root_dir / "patterns" / cfg.projector.pattern / f"{cfg.projector.index}.exr"
    )

    output_path = Path(
        f"{cfg.output.name}_{cfg.projector.index}.{cfg.output.extension}"
    )

    return pbrt_path, projector_path, output_path


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    pbrt_path, projector_path, output_path = set_paths(cfg)

    with open(pbrt_path) as f:
        pbrt_file = f.read()

    # Put in output render, projector image paths
    pbrt_file = pbrt_parser.parse_paths(pbrt_file, projector_path, output_path)

    # Modify material
    pbrt_file = pbrt_parser.parse_material(pbrt_file, cfg.material)

    # LookAt_camcoord directive
    pbrt_file = pbrt_parser.parse_lookat_camcoord(pbrt_file)

    # Write file
    pbrt_dump_path = (
        pbrt_path.parent
        / f"dump_{pbrt_path.stem}_{projector_path.parent.name}_{projector_path.name}.pbrt"
    )
    with open(pbrt_dump_path, "w") as f:
        f.write(pbrt_file)

    # Render it
    pbrt_executable = Path(hydra.utils.get_original_cwd()) / cfg.pbrt.executable

    # resolve if symlinked
    if pbrt_executable.is_symlink():
        pbrt_executable = os.readlink(pbrt_executable)

    pbrt_executable = str(pbrt_executable)

    cmd = [
        pbrt_executable,
        str(pbrt_dump_path.resolve()),
        "--display-server",
        str(cfg.display_server),
    ]
    if cfg.device == "gpu":
        cmd = cmd + ["--gpu"]

    if cfg.pbrt.spp:
        cmd = cmd + ["--spp", f"{cfg.pbrt.spp}"]
    subprocess.run(cmd)

    # Remove pbrt file
    if not cfg.pbrt.save_dump:
        pbrt_dump_path.unlink()


if __name__ == "__main__":
    main()
