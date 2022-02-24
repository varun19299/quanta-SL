from pathlib import Path
from tqdm import tqdm
import imageio

pattern_path = Path(
    "outputs/projector_frames/lcd/Hybrid BCH [255, 9] [gray_message] comp"
)
frame_path = Path(
    "outputs/real_captures/LCD_projector/30th_September/mary_bust/decoded_correspondences/Hybrid BCH [255, 9] [gray_message] comp"
)

frame_range = range(1, 271)
pattern_range = range(1, 541, 2)

pattern_writer = imageio.get_writer(frame_path / "patterns.mp4", fps=4)
frame_writer = imageio.get_writer(frame_path / "frames.mp4", fps=4)

assert len(frame_range) == len(pattern_range)

pbar = tqdm(zip(frame_range, pattern_range), total=len(frame_range), desc="Frame index")
for frame_index, pattern_index in pbar:
    pattern = imageio.imread(pattern_path / f"frame-{pattern_index}.png")

    # Read and flip
    frame = imageio.imread(frame_path / f"binary_frame{frame_index:02d}.png")[
        ::-1, ::-1
    ]

    pattern_writer.append_data(pattern)
    frame_writer.append_data(frame)
