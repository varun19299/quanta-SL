from pathlib import Path
from scipy.io import loadmat
from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
from roipoly import RoiPoly
from loguru import logger
from quanta_SL.utils.plotting import plot_image_and_colorbar
import matplotlib
import cv2

kinect_path = Path("outputs/real_captures/kinect")
mat_file = kinect_path / "kinect_results_1.mat"

out_folder = kinect_path / mat_file.stem
mask_folder = out_folder / "masks"
mask_folder.mkdir(exist_ok=True, parents=True)

frame_index = 5
mask_path = mask_folder / f"mask_{frame_index}.npy"

mat_file = loadmat(mat_file)
color_frames = mat_file["imgColor"]
depth_frames = mat_file["imgDepth"]

color_frames = rearrange(color_frames, "h w c n -> n h w c")
depth_frames = rearrange(depth_frames, "h w 1 n -> n h w")

color_frame = color_frames[frame_index]
depth_frame = depth_frames[frame_index]

# Regions to ignore
# Custom RoI
# if mask_path.exists():
#     logger.info(f"RoI Mask found at {mask_path}")
#     mask = np.load(mask_path)
#
# else:
#     plt.imshow(depth_frame)
#     my_roi = RoiPoly(color="r")
#     my_roi.display_roi()
#
#     mask = my_roi.get_mask(depth_frame)
#
#     np.save(mask_path, mask)
#     cv2.imwrite(str(mask_path.parent / f"{mask_path.stem}.jpg"), mask * 255.0)

# depth_frame = depth_frame * mask
depth_frame = depth_frame.astype(float)
# depth_frame[~mask] = np.nan
depth_frame[(depth_frame < 400) | (depth_frame > 550)] = np.nan
cmap = matplotlib.cm.get_cmap("jet").copy()
cmap.set_bad("black", alpha=1.0)

plot_image_and_colorbar(
    depth_frame,
    savefig=True,
    show=False,
    fname=out_folder / f"depth/{frame_index}.pdf",
    cmap=cmap,
    cbar_title="Depth (in mm)",
    vmin=495,
    vmax=505,
)

# (out_folder / "rgb").mkdir(exist_ok=True, parents=True)
# for frame_index, color_frame in enumerate(color_frames):
#     print(frame_index)
#     cv2.imwrite(
#         f"{out_folder}/rgb/color_frame_{frame_index}.png", color_frame[:, :, ::-1]
#     )
