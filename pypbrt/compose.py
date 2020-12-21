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
from omegaconf import DictConfig

import pypbrt.pbrt_writer as pbrt_writer


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


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    tag = cfg.pbrt.type
    pbrt_path, projector_path, output_path = set_paths(cfg)

    with open(pbrt_path) as f:
        pbrt_file = f.read()

    # Put in output render, projector image paths
    pbrt_file = pbrt_writer.configure_paths(pbrt_file, projector_path, output_path)

    # Modify material
    pbrt_file = pbrt_writer.modify_material(pbrt_file, cfg.material)

    # LookAt_camcoord directive
    pbrt_file = pbrt_writer.parse_lookat_camcoord(pbrt_file)

    # Write file
    pbrt_dump_path = (
        pbrt_path.parent
        / f"dump_{tag}_{projector_path.parent.name}_{projector_path.name}.pbrt"
    )
    with open(pbrt_dump_path, "w") as f:
        f.write(pbrt_file)

    # Render it
    cmd = [
        cfg.pbrt.executable,
        str(pbrt_dump_path.resolve()),
        "--display-server",
        str(cfg.display_server),
    ]
    if cfg.device == "gpu":
        cmd = cmd + ["--gpu"]
    subprocess.run(cmd)

    # Remove pbrt file
    if not cfg.pbrt.save_dump:
        os.remove(pbrt_dump_path)


if __name__ == "__main__":
    main()
