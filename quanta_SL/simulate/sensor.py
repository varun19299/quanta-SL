import numpy as np
from nptyping import NDArray

import quanta_SL.ops.linalg


def spad(
    Flux: NDArray[float], t_exp: float = 1e-3, epsilon: float = 1e-8
) -> NDArray[int]:
    # Poisson process
    arrival_time = np.random.exponential(scale=1 / (Flux + epsilon))
    return (arrival_time < t_exp).astype(int)


if __name__ == "__main__":
    import cv2
    img = quanta_SL.ops.linalg.mean(axis=-1)
    spad_img = spad(10 ** 3 * img)
    cv2.imwrite("/tmp/spad_dump.jpeg", spad_img * 255)
