import numpy as np
from nptyping import NDArray


def spad(
    Flux: NDArray[float], t_exp: float = 1e-3, epsilon: float = 1e-8
) -> NDArray[int]:
    # Poisson process
    arrival_time = np.random.exponential(scale=1 / (Flux + epsilon))
    return (arrival_time < t_exp).astype(int)
