from pathlib import Path

import matplotlib
import numpy as np
from einops import rearrange
from scipy.io import loadmat
from tqdm import tqdm
import imageio
from loguru import logger
import cv2
from matplotlib import pyplot as plt

from quanta_SL.utils.plotting import save_plot

kinect_path = Path("outputs/real_captures/kinect")
mat_file = kinect_path / "box.mat"
out_folder = kinect_path / mat_file.stem

logger.info(f"Loading mat file from {mat_file}")
mat_file = loadmat(mat_file)
rgb_frames = mat_file["imgColor"]
depth_frames = mat_file["imgDepth"]

rgb_frames = rearrange(rgb_frames, "h w c n -> n h w c")[:30]
depth_frames = rearrange(depth_frames, "h w 1 n -> n h w")[:30]

depth_folder = out_folder / f"depth"
depth_folder.mkdir(exist_ok=True, parents=True)
rgb_folder = out_folder / f"rgb"
rgb_folder.mkdir(exist_ok=True, parents=True)

depth_writer = imageio.get_writer(out_folder / "depth.mp4", fps=30)
rgb_writer = imageio.get_writer(out_folder / "rgb.mp4", fps=30)

vmin = 700
vmax = 800

for frame_index, (depth_frame, rgb_frame) in tqdm(
    enumerate(zip(depth_frames, rgb_frames)), desc="depth", total=len(depth_frames)
):
    depth_frame = depth_frame.astype(float)

    h, w, _ = rgb_frame.shape
    depth_mask = cv2.resize(depth_frame, (w, h))

    depth_frame[(depth_frame < vmin) | (depth_frame > vmax)] = np.nan
    cmap = matplotlib.cm.get_cmap("jet").copy()
    cmap.set_bad("black", alpha=1.0)

    plt.imshow(
        depth_frame,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    plt.grid(False)
    plt.axis("off")
    plt.colorbar()
    save_plot(
        savefig=True,
        show=False,
        fname=f"{depth_folder}/{frame_index}.png",
    )

    # Mask out rgb image
    depth_mask[(depth_mask < 600) | (depth_mask > 900)] = np.nan
    depth_mask = np.isnan(depth_mask)
    # rgb_frame[depth_mask, :] = 0

    cv2.imwrite(str(rgb_folder / f"{frame_index}.png"), rgb_frame[:, :, ::-1])

    depth_frame = imageio.imread(depth_folder / f"{frame_index}.png")
    depth_writer.append_data(depth_frame)
    rgb_writer.append_data(rgb_frame)
