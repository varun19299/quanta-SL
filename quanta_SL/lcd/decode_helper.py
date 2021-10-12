from einops import rearrange


def decode_2d_code(sequence_array, code_LUT, decoding_func):
    n, r, c = sequence_array.shape

    sequence_flat = rearrange(sequence_array, "n r c -> (r c) n")
    decoded_flat = decoding_func(sequence_flat, code_LUT)
    decoded_array = rearrange(decoded_flat, "(r c) -> r c", r=r, c=c)

    return decoded_array