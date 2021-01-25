# README

## Install

* Compile PBRT-v4 (optionally with GPU support)
* `make install``

## Example Code

### Acquire Gray Code captures of Diffuse Sphere

```
make simulate
```

Uses as default:

```
SCENE := sphere
MATERIAL := diffuse_0.3
PROJ_INDEX := 12
PROJ_PATTERN := ConventionalGray
DEVICE := cpu
EXP_NAME := $(SCENE)
PBRT_EXEC := /Users/varun/Dev/pbrt-v4/build/pbrt
```