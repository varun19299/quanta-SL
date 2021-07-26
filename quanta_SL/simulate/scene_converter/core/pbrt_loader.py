from quanta_SL.simulate.scene_converter.core import pbrt_yacc
from pathlib import Path

from typing import Union, List, Tuple


class PBRTv3Loader:
    def __init__(self, filename, **kwargs):
        scene_struct = self.parse_file(filename)
        scene_struct = self.substitute_parameters(scene_struct, **kwargs)
        self.scene = self.load_scene(scene_struct)

        with open("temp.pbrt", "w") as f:
            f.write(str(self.scene))
        breakpoint()

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
            if scene_struct[0][0][0] in [
                "Integrator",
                "Sampler",
                "Film",
                "Filter",
                "Camera",
                "Transform",
                "LookAt",
            ]:
                scene = self.load_directives(scene_struct[0], scene)
            else:
                scene = self.load_world(scene_struct[0], scene)

        else:
            scene = self.load_directives(scene_struct[0], scene)
            scene = self.load_world(scene_struct[1], scene)

        return scene

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

            self._load_transform_directive(directive, struct, transform_sequence)

        scene.sensor.transform.sequence = transform_sequence
        return scene

    @staticmethod
    def _load_transform_directive(directive: str, struct, transform_sequence: List):
        breakpoint()
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

    def _load_shape_directive(
        self,
        directive: str,
        struct,
        emitter=None,
        material=None,
        transform_sequence: List = [],
        current_ref_material=None,
        shape_ll: List = [],
        world_ll: List = [],
    ):
        if directive == "Shape":
            _, shape_type, params = struct

            params = self.load_params(params)

            shape = Shape(
                shape_type,
                emitter,
                material,
                Transform(sequence=transform_sequence),
                params,
            )

            # add reference material
            if current_ref_material:
                shape.params["id"] = Param("string", "id", current_ref_material)

            shape_ll.append(shape)
            world_ll.append(shape)

    def _load_texture_directive(
        self,
        directive: str,
        struct,
        texture_dict: Dict = {},
    ):
        if directive == "Texture":
            _, name, texture_class, texture_type, params = struct

            params = self.load_params(params)

            texture = Texture(name, texture_class, texture_type, params)
            texture.params = params

            texture_dict[name] = texture

    def _load_lightsource_directive(
        self,
        directive: str,
        struct,
        light_ll: List = [],
        world_ll: List = [],
    ):
        if directive in ["LightSource", "AreaLightSource"]:
            _, emitter_type, params = struct

            # simple emitters, no transform or shape involved. they go into lights list
            emitter = Emitter(emitter_type)
            emitter.params = self.load_params(params)

            light_ll.append(emitter)
            world_ll.append(emitter)

    def load_world(self, world_struct, scene) -> Scene:
        # Individual components
        # TODO: redundant, but kept for debugging
        material_ll = []
        shape_ll = []
        light_ll = []

        # Preserves overall ordering
        world_ll = []
        texture_dict = {}

        # To hold named materials
        current_ref_material = ""

        for struct in world_struct:
            directive = struct[0]

            if directive == "MakeNamedMaterial":
                struct_id, params = struct
                params = self.load_params(params)

                # actually there's little need to check if type is specified, but for the sake of properness...
                material_type = params.pop("type", "")

                # I'M NOT SURE
                if "bumpmap" in params:
                    bump_texture_name = params["bumpmap"].value

                    material = BumpMap()
                    material.texture = texture_dict[bump_texture_name]

                    material.material = Material(material_type, struct_id)
                    material.material.params = params

                else:
                    material = Material(material_type, struct_id)
                    material.params = params
                    if "Kd" in params:
                        kd = params["Kd"]
                        if kd.type == "texture":
                            material.texture = texture_dict[kd.value]
                            # TODO: why is this there?
                            # material.params.pop("Kd")

                material_ll.append(material)
                world_ll.append(material)

            elif directive == "NamedMaterial":
                current_ref_material = struct[1]

            elif directive == "AttributeBegin":
                material = None
                emitter = None

                # Texture description within attribute
                local_texture_dict = {}

                transform_sequence = []

                for modified_struct in struct[1]:
                    modified_directive = modified_struct[0]

                    if modified_directive == "Material":
                        material_type = modified_struct[1]
                        params = self.load_params(modified_struct[2])

                        material = Material(material_type, "")
                        material.params = params

                        if "Kd" in params:
                            kd = params["Kd"]
                            if kd.type == "texture":
                                material.texture = {
                                    **texture_dict,
                                    **local_texture_dict,
                                }[kd.value]
                                # TODO: why is this there? (pop)
                                # material.params.pop("Kd")

                    self._load_lightsource_directive(
                        modified_directive, modified_struct
                    )

                    self._load_texture_directive(
                        modified_directive, modified_struct, local_texture_dict
                    )

                    self._load_transform_directive(
                        modified_directive, modified_struct, transform_sequence
                    )

                    self._load_shape_directive(
                        modified_directive,
                        modified_struct,
                        emitter,
                        material,
                        transform_sequence,
                        current_ref_material,
                        shape_ll,
                        world_ll,
                    )

            elif directive == "TransformBegin":
                for modified_struct in struct[1]:
                    modified_directive = modified_struct[0]

                    self._load_lightsource_directive(
                        modified_directive, modified_struct, light_ll, world_ll
                    )

                    self._load_transform_directive(
                        modified_directive, modified_struct, transform_sequence
                    )

                    self._load_shape_directive(
                        modified_directive,
                        modified_struct,
                        current_ref_material=current_ref_material,
                        shape_ll=shape_ll,
                        world_ll=world_ll,
                    )

            self._load_lightsource_directive(directive, struct, light_ll, world_ll)

            self._load_texture_directive(directive, struct, texture_dict)

            self._load_shape_directive(
                directive,
                struct,
                current_ref_material=current_ref_material,
                shape_ll=shape_ll,
                world_ll=world_ll,
            )

        scene.world = world_ll

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
