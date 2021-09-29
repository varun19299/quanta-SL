from pathlib import Path

import cv2

from quanta_SL.io import load_swiss_spad_sequence
from loguru import logger

logger.disable("quanta_SL")

pose_index_range = range(1, 31)

frame_range = range(1, 25)
frame_index = 1
bin_range = range(0, 10)
bursts_per_pattern = 10
bin_offset = 0

for pose_index in pose_index_range:
    folder = Path(
        f"outputs/real_captures/LCD_projector/22nd_September/calibration/pose{pose_index:02d}/"
    )

    transition_folder = folder / "transition"
    transition_folder.mkdir(parents=True, exist_ok=True)

    # All white
    logger.info(f"Pose {pose_index} | AllWhite")
    for i in bin_range:
        burst = load_swiss_spad_sequence(folder, bin_suffix_range=[bin_offset + i])

        path = transition_folder / "all-white"
        path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path / f"burst_{i}.jpg"), burst * 255.0)

    bin_offset = bursts_per_pattern

    # Row and Column
    for encoding in ["col", "row"]:
        for frame_index in frame_range:
            logger.info(f"Pose {pose_index} | {encoding}{frame_index}")
            for i in bin_range:
                burst = load_swiss_spad_sequence(
                    folder, bin_suffix_range=[bin_offset + i]
                )
                comp_burst = load_swiss_spad_sequence(
                    folder, bin_suffix_range=[bin_offset + bursts_per_pattern + i]
                )

                path = transition_folder / f"{encoding}{frame_index:02d}"
                path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path / f"burst_{i}.jpg"), burst * 255.0)
                cv2.imwrite(str(path / f"comp_burst_{i}.jpg"), comp_burst * 255.0)

            bin_offset += 2 * bursts_per_pattern

        # # Plotting
        # fig, ax_ll = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(12, 6))
        #
        # # Frame
        # ax = ax_ll[0]
        # ax_imshow_with_colorbar(burst, ax, fig, cmap="gray")
        # ax.set_title(f"Burst {i} | Frame {frame_index} | Pose {pose_index}")
        #
        # # Comp
        # ax = ax_ll[1]
        # ax_imshow_with_colorbar(comp_burst, ax, fig, cmap="gray")
        # ax.set_title(f"Comp Burst {i} | Frame {frame_index} | Pose {pose_index}")
        #
        # # Comparison
        # ax = ax_ll[2]
        # ax_imshow_with_colorbar(burst > comp_burst, ax, fig)
        # ax.set_title(f"Resultant {i} | Frame {frame_index} | Pose {pose_index}")
        #
        # plt.show()
