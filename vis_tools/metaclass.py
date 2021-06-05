from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union, Callable

import numpy as np
import scipy.io as sio
from scipy import interpolate


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
