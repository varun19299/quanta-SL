import numpy as np
from nptyping import NDArray


def spad(
    Flux: NDArray[float], t_exp: float = 1e-3, epsilon: float = 1e-8
) -> NDArray[int]:
    # Poisson process
    arrival_time = np.random.exponential(scale=1 / (Flux + epsilon))
    return (arrival_time < t_exp).astype(int)


if __name__ == "__main__":
    import cv2
    img = cv2.imread("outputs/dragon_subsurface/diffuse_0.1/ConventionalGray/meas_13.exr", -1).mean(axis=-1)
    spad_img = spad(10 ** 3 * img)
    cv2.imwrite("/tmp/spad_dump.jpeg", spad_img * 255)
