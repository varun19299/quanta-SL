from simulate.scene_converter.core import pbrt_yacc
from simulate.scene_converter.core.directives import *
from pathlib import Path

from typing import Union, List, Tuple


class PBRTv3Loader:
    def __init__(self, filename, **kwargs):
        scene_struct = self.parse_file(filename)
        scene_struct = self.substitute_parameters(scene_struct, **kwargs)
        self.scene = self.load_scene(scene_struct)

    def substitute_parameters(self, scene_struct, **kwargs) -> List[Tuple]:
        """
        Pass arguments to the scene
        Similar to:
        https://mitsuba2.readthedocs.io/en/latest/src/python_interface/parsing_xml.html#passing-arguments-to-the-scene

        Arguments to substitute must be present as strings, and begin with '$'.

        :param scene_struct: Parsed scene structure (Nested list, tuple)
        :param kwargs: arguments to substitute
        :return: Scene structure
        """
        scene_struct = list(scene_struct)
        for e, struct in enumerate(scene_struct):
            if isinstance(struct, tuple) or isinstance(struct, list):
                if isinstance(struct, tuple):
                    base_class = tuple
                else:
                    base_class = list

                scene_struct[e] = base_class(
                    self.substitute_parameters(struct, **kwargs)
                )
            if isinstance(struct, str):
                if struct.startswith("$"):
                    scene_struct[e] = kwargs.get(struct.lstrip("$"), struct)
        return scene_struct

    def load_scene(self, scene_struct) -> Scene:
        """
        Obtain a Scene object from scene struct
        :param scene_struct:
        :return:
        """
        try:
            len(scene_struct)
        except TypeError:
            raise TypeError("Scene structure should be Sized")

        scene = Scene()

        if len(scene_struct) == 1:
            if scene_struct[0][0] in [
                "Integrator",
                "Sampler",
                "Film",
                "Filter",
                "Camera",
                "Transform",
            ]:
                scene = self.load_directives(scene_struct[0], scene)
            else:
                scene = self.load_world(scene_struct[0], scene)

        else:
            scene = self.load_directives(scene_struct[0], scene)
            scene = self.load_world(scene_struct[1], scene)

        return scene

    @staticmethod
    def _transform_directives(directive: str, struct, transform_sequence: List):
        if directive == "Transform":
            transform = Matrix()

            if struct[2]:
                transform.value = np.array(struct[2]).reshape(4, 4).tolist()

            transform_sequence.append(transform)

        elif directive == "LookAt":
            lookat = LookAt()

            if struct[2]:
                lookat.eye = struct[2][:3]
                lookat.look = struct[2][3:6]
                lookat.up = struct[2][6:]

            transform_sequence.append(lookat)

        elif directive == "Translate":
            translate = Translate(value=struct[2])
            transform_sequence.append(translate)

        elif directive == "Rotate":
            rotate = Rotate(angle=struct[2][0], axis=struct[2][1:])
            transform_sequence.append(rotate)

        elif directive == "Scale":
            scale = Scale(value=struct[2])
            transform_sequence.append(scale)

    def load_directives(self, directive_struct, scene) -> Scene:
        scene.sensor = Sensor()
        transform_sequence = []

        for struct in directive_struct:
            directive = struct[0]

            if directive == "Integrator":
                scene.sensor.integrator.type = struct[1]

                if struct[2]:
                    scene.sensor.integrator.params = self.load_params(struct[2])

            elif directive == "Camera":
                scene.sensor.type = struct[1]

                if struct[2]:
                    scene.sensor.params = self.load_params(struct[2])

            elif directive == "Sampler":
                scene.sensor.sampler.type = struct[1]

                if struct[2]:
                    scene.sensor.sampler.params = self.load_params(struct[2])

            elif directive == "Film":
                scene.sensor.film.type = struct[1]

                if struct[2]:
                    scene.sensor.film.params = self.load_params(struct[2])

            elif directive == "PixelFilter":
                scene.sensor.film.filter = struct[1]

            self._transform_directives(directive, struct, transform_sequence)

        scene.sensor.transform.sequence = transform_sequence
        return scene

    def load_world(self, world_struct, scene) -> Scene:
        material_ll = []
        shape_ll = []
        light_ll = []
        texture_dict = {}

        # To hold named materials
        current_ref_material = ""

        for struct in world_struct:
            directive = struct[0]

            if directive == "Texture":
                name = struct[1]
                material_type = struct[3]

                params = self.load_params(struct[4])

                texture = Texture(name, material_type)
                texture.params = params

                texture_dict[name] = texture

            elif directive == "MakeNamedMaterial":
                struct_id = struct[1]
                params = {}

                if struct[2]:
                    params = self.load_params(struct[2])

                # actually there's little need to check if type is specified, but for the sake of properness...
                material_type = params.pop("type", "")

                # I'M NOT SURE
                if "bumpmap" in params:
                    bump_texture_name = params["bumpmap"].value

                    material = BumpMap()
                    material.texture = texture_dict[bump_texture_name]

                    material.material = Material(material_type, struct_id)
                    material.material.params = params

                    material_ll.append(material)

                else:
                    material = Material(material_type, struct_id)
                    material.params = params

                    if "Kd" in params:
                        kd = params["Kd"]
                        if kd.type == "texture":
                            material.texture = texture_dict[kd.value]
                            material.params.pop("Kd")

                    material_ll.append(material)

            elif directive == "NamedMaterial":
                current_ref_material = struct[1]

            elif directive == "Shape":
                # simple shape, no emitter, embed material or transform
                shape = Shape(struct[1])
                shape.params = self.load_params(struct[2])

                # add reference material
                if current_ref_material:
                    shape.params["id"] = Param("string", "id", current_ref_material)

                shape_ll.append(shape)

            elif directive == "LightSource":
                _, emitter_type, params = struct
                # simple emitters, no transform or shape involved. they go into lights list
                emitter = Emitter(emitter_type)
                emitter.transform = None
                emitter.params = self.load_params(params)

                light_ll.append(emitter)

            elif directive == "AttributeBegin":
                material = None
                emitter = None
                transform = Transform()
                # Texture description within attribute
                local_texture = {}

                transform_sequence = []

                for modified_struct in struct[1]:
                    modified_directive = modified_struct[0]

                    if modified_directive == "AreaLightSource":
                        emitter = Emitter(modified_struct[1])
                        emitter.params = self.load_params(modified_struct[2])

                    elif modified_directive == "Texture":
                        name = modified_struct[1]
                        material_type = modified_struct[3]

                        params = self.load_params(modified_struct[4])

                        texture = Texture(name, material_type)
                        texture.params = params

                        local_texture[name] = texture

                    elif modified_directive == "Material":
                        material_type = modified_struct[1]
                        params = self.load_params(modified_struct[2])

                        material = Material(material_type, "")
                        material.params = params

                        if "Kd" in params:
                            kd = params["Kd"]
                            if kd.type == "texture":
                                material.texture = {**texture_dict, **local_texture}[
                                    kd.value
                                ]
                                material.params.pop("Kd")

                    self._transform_directives(
                        modified_directive, modified_struct, transform_sequence
                    )

                    if modified_directive == "Shape":
                        # simple shape, no emitter, embed material or transform
                        shape = Shape(modified_struct[1], transform=transform)
                        shape.params = self.load_params(modified_struct[2])

                        # add reference material
                        if current_ref_material:
                            shape.params["id"] = Param(
                                "string", "id", current_ref_material
                            )

                        shape.emitter = emitter
                        shape.material = material
                        shape.transform.sequence = transform_sequence

                        shape_ll.append(shape)

            elif directive == "TransformBegin":
                transform = None
                for modified_struct in struct[1]:
                    modified_directive = modified_struct[0]

                    if modified_directive == "Transform":
                        transform = Transform()
                        transform.matrix = modified_struct[2]
                        transform.matrix = (
                            np.asarray(modified_struct[2]).reshape(4, 4).tolist()
                        )

                    elif modified_directive == "Shape":
                        # simple shape, no emitter, embed material or transform
                        shape = Shape(modified_struct[1])
                        shape.params = self.load_params(modified_struct[2])

                        # add reference material
                        if current_ref_material:
                            shape.params["id"] = Param(
                                "string", "id", current_ref_material
                            )

                        shape.transform = transform

                        shape_ll.append(shape)

                    elif modified_directive == "LightSource":
                        # simple emitters, no transform or shape involved. they go into lights list
                        emitter = Emitter(modified_struct[1])
                        emitter.transform = transform
                        emitter.params = self.load_params(modified_struct[2])

                        light_ll.append(emitter)

        scene.materials = material_ll
        scene.lights = light_ll
        scene.shapes = shape_ll

        return scene

    @staticmethod
    def load_params(param_struct):
        params = {}

        for element in param_struct:
            element_type, name, value = element
            param = Param(element_type, name, value)
            params[name] = param

        return params

    @staticmethod
    def parse_file(filename: Union[str, Path]) -> List[Tuple]:
        """
        Open scene and parse with YACC
        :param filename:
        :return: Scene structure as a list of tuples
        """

        data = open(filename).read()
        scene_struct = pbrt_yacc.parse(data)
        return scene_struct
