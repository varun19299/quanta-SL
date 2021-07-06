from simulate.scene_converter.core import (
    pbrt_loader as pbrt,
    mitsuba_loader as mit,
    mitsuba_to_pbrt as mp,
    pbrt_to_mistuba as pm,
)

import sys

if __name__ == "__main__":
    filename = ""
    source = ""
    destination = ""
    output = "scene"

    if len(sys.argv) <= 1:
        print(
            "Please call PBR Scene converter using the following parameters: -s [source renderer] -d [destination render] -f [input filename] <option: -o [output filename]>"
        )
    elif not "-f" in sys.argv:
        print(
            "No input file specified. Please call PBR Scene converter using the following parameters: -s [source renderer] -d [destination render] -f [input filename] -o [output filename]"
        )
    elif not "-s" in sys.argv:
        print(
            "No source renderer specified. Please call PBR Scene Converter with the following parameters: -s [source renderer] -d [destination render] -f [input filename] -o [output filename]"
        )

    else:
        for i in range(1, len(sys.argv)):
            if sys.argv[i] == "-s":
                source = sys.argv[i + 1]

            elif sys.argv[i] == "-d":
                destination = sys.argv[i + 1]

            elif sys.argv[i] == "-f":
                filename = sys.argv[i + 1]

            elif sys.argv[i] == "-o":
                output = sys.argv[i + 1]

        # load scene from file
        if source == "mitsuba":
            loader = mit.MitsubaLoader(filename)

            if destination == "pbrt":
                if not output.endswith(".pbrt"):
                    output += ".pbrt"
                mp.MitsubaToPBRTv3(loader.scene, output)

            else:
                print(
                    "The output renderer informed is not valid. For a mitsuba input file, please type -d pbrt.\n"
                )

        elif source == "pbrt":
            loader = pbrt.PBRTv3Loader(filename, samples=128)

            if destination == "mitsuba":
                pm.PBRTv3ToMitsuba(loader.scene, output)

            else:
                print(
                    "The output renderer informed is not valid. For a pbrt input file, please type -d mitsuba.\n"
                )

        else:
            print(
                "The source renderer informed is not valid. Current valid source renderers are: pbrt, mitsuba. \n"
            )
