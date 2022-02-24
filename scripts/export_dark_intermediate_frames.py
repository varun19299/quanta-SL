from quanta_SL.io import load_swiss_spad_bin
import cv2
from pathlib import Path

scene = "coffee_mug"
strategy = r"Hybrid BCH [255, 9] [gray_message] comp"
folder_name = Path(
    f"outputs/real_captures/LCD_projector/30th_September/{scene}/{strategy}"
)

out_array = load_swiss_spad_bin(folder_name=folder_name, bin_suffix=84)
out_array = out_array[255:258].sum(axis=0) > 0

cv2.imwrite(f"{folder_name}/{scene}_intermediate.png", out_array * 255.0)
