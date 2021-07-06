from dataclasses import dataclass, field
from typing import Union, Dict, List

# general directives


@dataclass
class Param:
    type: str
    name: str
    value: Union[int, float, str]


# scene directives


@dataclass
class Integrator:
    type: str = "path"
    params: Dict = field(default_factory=dict)


@dataclass
class Transform:
    name: str = "toWorld"
    matrix: List = field(default_factory=list)


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
    film: Film = Film()
    params: Dict = field(default_factory=dict)


# world components


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
    transform: Transform = Transform()
    params: Dict = field(default_factory=dict)


@dataclass
class Shape:
    type: str
    emitter: Emitter = None
    material: Material = None
    transform: Transform = None
    params: Dict = field(default_factory=dict)


# global


class Scene:
    def __init__(self):
        self.integrator = Integrator()
        self.sensor = Sensor()
        self.materials = []
        self.shapes = []
        self.lights = []
        self.mediums = []
