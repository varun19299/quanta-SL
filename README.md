# README

## Install

* Compile PBRT-v4 (optionally with GPU support)
* Simlink the PBRT binary
    ```
    ln -s <path-to-pbrt-v4-repo>/build/pbrt pbrt 
    ```
* `make install`

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
PBRT_EXEC := pbrt
```

Output should be present at `outputs/<EXP_NAME>/<PROJ_PATTERN>`.

### Reconstruct with Conventional & Quanta SL

```
make reconstruct
```

Uses as default:

```
SCENE := sphere
MATERIAL := diffuse_0.3
PROJ_INDEX := 12
PROJ_PATTERN := ConventionalGray
DEVICE := cpu
EXP_NAME := $(SCENE)
PBRT_EXEC := pbrt
SENSOR := SPAD
EXPOSURE := 100
```