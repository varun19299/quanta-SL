from dataclasses import dataclass, field
from typing import Union, Dict, List

# general directives
import numpy as np


@dataclass
class Param:
    type: str
    name: str
    value: Union[int, float, str]


"""
Transforms
"""


@dataclass
class Transform:
    name: str = "toWorld"
    sequence: List = field(default_factory=list)


@dataclass
class Matrix:
    value: List = field(default_factory=np.eye(4).tolist)

    def __post_init__(self):
        assert np.array(self.value).shape == (4, 4), "Transform should be a 4x4 matrix"


@dataclass
class Rotate:
    angle: float = 0.0
    axis: List = field(default_factory=lambda: [0, 0, 1])

    def __post_init__(self):
        assert np.array(self.axis).shape == (3,), "Axis must be 3D"


@dataclass
class Translate:
    value: List = field(default_factory=np.zeros(3).tolist)

    def __post_init__(self):
        assert np.array(self.value).shape == (3,), "value must be 3D"


@dataclass
class Scale:
    value: List = field(default_factory=np.ones(3).tolist)

    def __post_init__(self):
        assert np.array(self.value).shape == (3,), "value must be 3D"


@dataclass
class LookAt:
    eye: List = field(default_factory=lambda: [0, 0, 0])
    look: List = field(default_factory=lambda: [0, 1, 0])
    up: List = field(default_factory=lambda: [0, 0, 1])

    def __post_init__(self):
        assert (
            len(self.eye) == len(self.look) == len(self.up) == 3
        ), "Eye, look, up must be 3D vectors."


"""
Scene directives
"""


@dataclass
class Integrator:
    type: str = "path"
    params: Dict = field(default_factory=dict)


@dataclass
class Sampler:
    type: str = "stratified"
    params: Dict = field(default_factory=dict)


@dataclass
class Film:
    type: str = "hdrfilm"
    filter: str = ""


@dataclass
class Sensor:
    type: str = "perspective"
    transform: Transform = Transform()
    sampler: Sampler = Sampler()
    integrator: Integrator = Integrator()
    film: Film = Film()
    params: Dict = field(default_factory=dict)


"""
World Components
"""


@dataclass
class Texture:
    name: str
    type: str
    params: Dict = field(default_factory=dict)

    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.params = {}


@dataclass
class Material:
    type: str
    id: str
    texture: Texture = None
    params: Dict = field(default_factory=dict)


@dataclass
class BumpMap:
    texture: Texture = None
    material: Material = Material(type="", id="")
    params: Dict = field(default_factory=dict)


@dataclass
class Emitter:
    type: str
    transform: Transform = None
    params: Dict = field(default_factory=dict)


@dataclass
class Shape:
    type: str
    emitter: Emitter = None
    material: Material = None
    transform: Transform = None
    params: Dict = field(default_factory=dict)


"""
Global
"""


@dataclass()
class Scene:
    sensor: Sensor = Sensor()
    materials: List[Material] = field(default_factory=list)
    shapes: List[Shape] = field(default_factory=list)
    lights: List = field(default_factory=list)
    mediums: List = field(default_factory=list)
