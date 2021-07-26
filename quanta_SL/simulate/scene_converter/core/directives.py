import textwrap
from dataclasses import dataclass, field
from typing import Union, Dict, List

# general directives
import numpy as np
from copy import copy


def _indent(lines: str):
    """
    Indent string (excluding first line) by 4 spaces
    :param lines: string with lines separated by \n
    :return: indented string
    """
    return textwrap.indent(
        lines,
        "    ",
        lambda line: not line.isspace() and lines.splitlines(True).index(line),
    )


@dataclass
class Param:
    type: str
    name: str
    value: Union[int, float, str, List]

    def __str__(self):
        value_str = str(self.value)
        if isinstance(self.value, str):
            value_str = f'"{self.value}"'

        elif isinstance(self.value, list):
            value_str = " ".join(str(e) for e in self.value)

        return f'"{self.type} {self.name}" [ {value_str} ]'


"""
Transforms
"""


@dataclass
class Transform:
    name: str = "toWorld"
    sequence: List = field(default_factory=list)

    def __post_init__(self):
        self.sequence = copy(self.sequence)

    def __str__(self):
        out_str = "\n".join(str(e) for e in self.sequence)
        return out_str

    def __len__(self):
        return len(self.sequence)


@dataclass
class Matrix:
    value: List = field(default_factory=np.eye(4).tolist)

    def __post_init__(self):
        assert np.array(self.value).shape == (4, 4), "Transform should be a 4x4 matrix"

    def __str__(self):
        array_str = []
        for row in self.value:
            array_str.append(" ".join(str(e) for e in row))
        array_str = "\n".join(array_str)

        out_str = f"Transform [ {array_str} ]"
        return out_str


@dataclass
class Rotate:
    angle: float = 0.0
    axis: List = field(default_factory=lambda: [0, 0, 1])

    def __post_init__(self):
        assert np.array(self.axis).shape == (3,), "Axis must be 3D"

    def __str__(self):
        return f"Rotate {self.angle} {' '.join(str(e) for e in self.axis)}"


@dataclass
class Translate:
    value: List = field(default_factory=np.zeros(3).tolist)

    def __post_init__(self):
        assert np.array(self.value).shape == (3,), "value must be 3D"

    def __str__(self):
        return f"Translate {' '.join(str(e) for e in self.value)}"


@dataclass
class Scale:
    value: List = field(default_factory=np.ones(3).tolist)

    def __post_init__(self):
        assert np.array(self.value).shape == (3,), "value must be 3D"

    def __str__(self):
        return f"Scale {' '.join(str(e) for e in self.value)}"


@dataclass
class LookAt:
    eye: List = field(default_factory=lambda: [0, 0, 0])
    look: List = field(default_factory=lambda: [0, 1, 0])
    up: List = field(default_factory=lambda: [0, 0, 1])

    def __post_init__(self):
        assert (
            len(self.eye) == len(self.look) == len(self.up) == 3
        ), "Eye, look, up must be 3D vectors."

    def __str__(self):
        out_str = f"LookAt {' '.join(str(e) for e in self.eye)}\n"
        out_str += f"{' '.join(str(e) for e in self.look)}\n"
        out_str += f"{' '.join(str(e) for e in self.look)}"
        return _indent(out_str)


"""
Scene directives
"""


@dataclass
class Integrator:
    type: str = "path"
    params: Dict = field(default_factory=dict)

    def __str__(self):
        out_str = f'Integrator "{self.type}"'
        for key, param in self.params.items():
            out_str += f"\n{param}"
        return _indent(out_str)


@dataclass
class Sampler:
    type: str = "stratified"
    params: Dict = field(default_factory=dict)

    def __str__(self):
        out_str = f'Sampler "{self.type}"'
        for key, param in self.params.items():
            out_str += f"\n{param}"
        return _indent(out_str)


@dataclass
class Film:
    type: str = "hdrfilm"
    filter: str = ""
    params: Dict = field(default_factory=dict)

    def __str__(self):
        out_str = f'Film "{self.type}"'
        for key, param in self.params.items():
            out_str += f"\n{param}"
        return _indent(out_str)


@dataclass
class Sensor:
    type: str = "perspective"
    transform: Transform = Transform()
    sampler: Sampler = Sampler()
    integrator: Integrator = Integrator()
    film: Film = Film()
    params: Dict = field(default_factory=dict)

    def __str__(self):
        out_str = f'Camera "{self.type}" '

        for key, param in self.params.items():
            out_str += f"\n{param}"

        out_str = _indent(out_str)

        for e in [self.transform, self.sampler, self.integrator, self.film]:
            out_str += f"\n{e}"

        return out_str


"""
World Components
"""


@dataclass
class Texture:
    name: str
    type: str
    texture_class: str
    params: Dict = field(default_factory=dict)

    def __str__(self):
        out_str = f'Texture "{self.name}" "{self.type}" "{self.texture_class}"'
        for key, param in self.params.items():
            out_str += f"\n{param}"
        return _indent(out_str)


@dataclass
class Material:
    type: str
    id: str
    texture: Texture = None
    params: Dict = field(default_factory=dict)

    def __str__(self):
        material_str = f'Material "{self.type}" '

        # Add other parameters
        for key, param in self.params.items():
            material_str += f"\n{param}"

        material_str = _indent(material_str)

        if self.texture:
            out_str = f"{self.texture}\n{material_str}"
        else:
            out_str = material_str
        return out_str


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

    def __str__(self):
        if type == "diffuse":
            light_str = "AreaLightSource"
        else:
            light_str = "LightSource"

        light_str = f'{light_str} "{self.type}" '

        for key, param in self.params.items():
            light_str += f"\n{param}"

        light_str = _indent(light_str)

        # Whether to enclose with TransformBegin
        if self.transform:
            out_str = "TransformBegin\n" + str(self.transform) + "\n" + light_str
            out_str = _indent(out_str)
            out_str += "\nTransformEnd"
        else:
            out_str = light_str

        return out_str


@dataclass
class Shape:
    type: str
    emitter: Emitter = None
    material: Material = None
    transform: Transform = Transform()
    params: Dict = field(default_factory=dict)

    def __str__(self):
        shape_str = f'Shape "{self.type}" '

        for key, param in self.params.items():
            shape_str += f"\n{param}"

        shape_str = _indent(shape_str)

        # Subproperties
        subproperty_str_ll = []

        for subproperty in [self.material, self.emitter, self.transform]:
            if subproperty:
                subproperty_str_ll.append(str(subproperty))

        subproperty_str = "\n".join(subproperty_str_ll)

        # Whether to enclose as an Attribute / Transform / Simple Shape
        if self.material or self.emitter:
            out_str = "AttributeBegin\n" + subproperty_str + "\n" + shape_str
            out_str = _indent(out_str)
            out_str += "\nAttributeEnd"

        elif self.transform:
            out_str = "TransformBegin\n" + subproperty_str + "\n" + shape_str
            out_str = _indent(out_str)
            out_str += "\nTransformEnd"

        else:
            out_str = shape_str

        return out_str


@dataclass
class Medium:
    pass


"""
Global
"""


@dataclass()
class Scene:
    sensor: Sensor = Sensor()
    world: List[Union[Material, Shape, Emitter, Medium]] = field(default_factory=list)

    def __str__(self):
        out_str_ll = [self.sensor]
        if self.world:
            out_str_ll = [*out_str_ll, "WorldBegin", *self.world, "WorldEnd"]
        out_str = "\n\n".join(str(e) for e in out_str_ll)

        return out_str
