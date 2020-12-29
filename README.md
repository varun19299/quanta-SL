# README

## Install

* Compile PBRT-v4 (optionally with GPU support)
* `pip install -r requirements.txt`

Install locally as:
* `pip install -e .`

## Example Code

### Acquire Gray Code captures of Diffuse Sphere

```
python pypbrt/simulate.py pbrt.scene=sphere material=diffuse \
projector.index='range(0,21)' device=gpu \
pbrt.executable=/srv/home/varunsundar/pbrt-v4/build/pbrt -m
``` 