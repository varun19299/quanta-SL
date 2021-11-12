from pathlib import Path
from roipoly import RoiPoly
from loguru import logger
from quanta_SL.io import load_swiss_spad_sequence

scene = "mary_bust"
method= "Gray Code [11 bits] comp"
folder = Path(f"outputs/real_captures/LCD_projector/30th_September/{scene}/{method}")

bin_suffix_range = range(16, 20)
t_exp = 1e-4

frame = load_swiss_spad_sequence(folder, bin_suffix_range=bin_suffix_range)

