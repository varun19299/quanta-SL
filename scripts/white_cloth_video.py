from pathlib import Path
from tqdm import tqdm
import imageio
from natsort import natsorted

frame_path = Path(
    "outputs/real_captures/DLP_projector/29th_October/white_cloth/results/depth_video_250_vm_44_49"
)

fps = 30
frame_writer = imageio.get_writer(frame_path.parent / f"{frame_path.stem}_fps_{fps}_2.mp4", fps=fps)


file_list = natsorted(list(frame_path.glob("*.png")))
pbar = tqdm(file_list[::4], total=len(file_list))

for file_path in pbar:
    pbar.set_description(f"File name {file_path.name}")
    frame = imageio.imread(file_path)

    frame_writer.append_data(frame)
