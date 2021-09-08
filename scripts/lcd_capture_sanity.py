from quanta_SL.io import load_swiss_spad_sequence, load_swiss_spad_burst
from pathlib import Path
from quanta_SL.encode import metaclass
from quanta_SL.encode.strategies import hybrid_code_LUT
from quanta_SL.encode.message import gray_message
from quanta_SL.decode.methods import hybrid_decoding
from quanta_SL.decode.minimum_distance.factory import (
    faiss_flat_index,
    faiss_minimum_distance,
)
from quanta_SL.ops.binary import packbits_strided
from functools import partial

from loguru import logger
import numpy as np
from einops import rearrange

# Disable inner logging
logger.disable("quanta_SL")

hybrid_folder = Path(r"outputs/real_captures/LCD projector/27th August/0827-hybridBCH/")

gt_sequence = []
binary_sequence = []

# Generate hybrid LUT
logger.info("Generating code LUT")

# Params
bch_tuple = metaclass.BCH(63, 10, 13)
bch_message_bits = 8
message_bits = 11
overlap_bits = 1
gt_bin_suffix_range = range(10)

code_LUT = hybrid_code_LUT(
    bch_tuple,
    bch_message_bits,
    message_bits,
    overlap_bits,
    message_mapping=gray_message,
)

# Decoding func
index = faiss_flat_index(packbits_strided(code_LUT))
decoding_func = partial(
    hybrid_decoding,
    func=faiss_minimum_distance,
    bch_tuple=bch_tuple,
    bch_message_bits=bch_message_bits,
    overlap_bits=overlap_bits,
    index=index,
    pack=True,
)


pattern_range = range(1, code_LUT.shape[1] + 1)
for pattern_index in pattern_range:
    logger.info(f"Sequence #{pattern_index}/{len(pattern_range)}")

    # Filename with leading zeros
    pattern_folder = hybrid_folder / f"pattern{2 * pattern_index - 1:03d}"
    pattern_comp_folder = hybrid_folder / f"pattern{2 * pattern_index:03d}"

    # Groundtruth
    gt_frame = load_swiss_spad_sequence(
        pattern_folder, bin_suffix_range=gt_bin_suffix_range, num_rows=256, num_cols=512
    )

    # Complementary
    gt_frame_comp = load_swiss_spad_sequence(
        pattern_comp_folder,
        bin_suffix_range=gt_bin_suffix_range,
        num_rows=256,
        num_cols=512,
    )

    gt_frame = gt_frame > gt_frame_comp
    gt_sequence.append(gt_frame)

    # Single binary sample
    binary_frame = load_swiss_spad_burst(
        pattern_folder, bin_suffix=0, num_rows=256, num_cols=512
    )
    binary_sequence.append(binary_frame[0])

# Stack
gt_sequence = np.stack(gt_sequence, axis=0)
binary_sequence = np.stack(binary_sequence, axis=0)

n, r, c = binary_sequence.shape

# Decode
gt_sequence = rearrange(gt_sequence, "n r c -> (r c) n")
binary_sequence = rearrange(binary_sequence, "n r c -> (r c) n")

logger.info("Decoding GT")
decoded_gt = decoding_func(gt_sequence, code_LUT)
decoded_gt = rearrange(decoded_gt, "(r c) -> r c", r=r, c=c)

logger.info("Decoding Binary")
decoded_binary = decoding_func(binary_sequence, code_LUT)
decoded_binary = rearrange(decoded_binary, "(r c) -> r c", r=r, c=c)

logger.info("Evaluating Accuracy")
accuracy = decoded_binary == decoded_gt
print(accuracy.mean())
