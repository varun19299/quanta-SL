from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union, Callable

import galois
import numpy as np
import scipy.io as sio
from einops import repeat
from scipy import interpolate

from quanta_SL.ops.binary import dim_str


@dataclass
class Eval:
    name: str


@dataclass
class MatlabEval(Eval):
    fname: Union[Path, str]
    key: str = ""

    def load(self):
        mat_dict = sio.loadmat(self.fname)
        if not self.key:
            for k, v in mat_dict.items():
                if isinstance(v, np.ndarray):
                    self.key = k
        return mat_dict[self.key]

    def __call__(self, phi_P_mesh, phi_A_mesh, t_exp, *args, **kwargs):
        eval_error = self.load()

        # Resize it via linear interpolation
        ratio = int(phi_P_mesh.shape[0] / eval_error.shape[0])

        # Undo meshgrid: https://stackoverflow.com/questions/53385605/how-to-undo-or-reverse-np-meshgrid
        phi_P = phi_P_mesh[:, 0]
        phi_A = phi_A_mesh[0, :]
        phi_proj = phi_P - phi_A

        phi_P_subsampled = (phi_proj + phi_A)[::ratio]
        phi_A_subsampled = phi_A[::ratio]

        f = interpolate.interp2d(
            phi_P_subsampled, phi_A_subsampled, eval_error, kind="cubic"
        )
        return f(phi_proj + phi_A, phi_A)


@dataclass
class CallableEval(Eval):
    func: Callable
    kwargs: Dict = field(default_factory=dict)

    def __call__(self, phi_P_mesh, phi_A_mesh, t_exp, *args, **kwargs):
        return self.func(phi_P_mesh, phi_A_mesh, t_exp, *args, **kwargs, **self.kwargs)


@dataclass
class _Code:
    n: int
    k: int
    # Correctable errors. Can be > \floor(d - 1 / 2), via List Decoding
    t: int


@dataclass
class BCH(_Code):
    def __post_init__(self):
        self.galois_instance = galois.BCH(self.n, self.k)

    def __str__(self):
        return f"BCH [{self.n}, {self.k}, {self.t}]"

    def encode(self, array):
        """
        Encoding function

        :param array: Input message (1d or 2d)
        :return: code words
        """
        array = galois.GF2(array)
        out = self.galois_instance.encode(array)
        out = out.view(np.ndarray)
        return out

    @property
    def distance(self):
        """
        Minimum distance of the code
        Not the same as 2 * correctable errors + 1
        (esp if oracle or list decoded is employed).

        :return: minimum distance
        """
        return self.galois_instance.t * 2 + 1

    @property
    def is_list_decoding(self):
        return 2 * self.t + 1 > self.distance


@dataclass
class Repetition(_Code):
    def __post_init__(self):
        assert (
            self.n % self.k == 0
        ), f"Code length {self.n} must be a multiple of message length {self.k}."

    def __str__(self):
        return f"Repetition [{self.n}, {self.k}]"

    def encode(self, array):
        """
        Encoding function.
        Repeats in interleaved manner.
        [0, 1, 2] -> [0, 0, 0, 1, 1, 1, 2, 2, 2]

        :param array: Input message (1d or 2d)
        :return: code words
        """
        assert array.ndim in [1, 2]

        return repeat(
            array,
            f"{dim_str(array.ndim - 1)} n -> {dim_str(array.ndim - 1)} (n repeat)",
            repeat=self.repeat,
        )

    @property
    def repeat(self) -> int:
        return self.n // self.k
