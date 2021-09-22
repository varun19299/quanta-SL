# Define macros
UNAME_S := $(shell uname -s)
PYTHON := python

## HYDRA_FLAGS : set as -m for multirun
HYDRA_FLAGS := -m
SEED := 0

.PHONY: help docs
.DEFAULT: help

## install: Pip requirements, local development mode
install.cpu:
	conda install -f environment.yml
	pip install -r requirements.txt
	pip install -e .

install.gpu:
	conda install -f environment_gpu.yml
	pip install -r requirements.txt
	pip install -e .

help : Makefile
    ifeq ($(UNAME_S),Linux)
		@sed -ns -e '$$a\\' -e 's/^##//p' $^
    endif
    ifeq ($(UNAME_S),Darwin)
        ifneq (, $(shell which gsed))
			@gsed -sn -e 's/^##//p' -e '$$a\\' $^
        else
			@sed -n 's/^##//p' $^
        endif
    endif

## test: Run unit tests wherever present
test:
	@pytest -s quanta_SL/*

# Simulate defaults
SCENE := sphere
MATERIAL := diffuse_0.3
PROJ_INDEX := 12
PROJ_PATTERN := ConventionalGray
DEVICE := cpu
EXP_NAME := $(SCENE)
PBRT_EXEC := pbrt

## simulate.pbrt: Ray tracing with PBRT-v4
simulate.pbrt:
	${PYTHON} simulate/pbrt \
	pbrt.scene=$(SCENE) material=$(MATERIAL) \
 	projector.index=$(PROJ_INDEX) projector.pattern=$(PROJ_PATTERN) \
 	exp_name=$(EXP_NAME) \
 	device=$(DEVICE) pbrt.executable=$(PBRT_EXEC) $(KWARGS) $(HYDRA_FLAGS)

## reconstruct: Reconstruct from multiple captures outputs/<exp_name>/<proj_pattern>.
SENSOR := SPAD
EXPOSURE := 100

reconstruct:
	${PYTHON} simulate/reconstruct.py \
	pbrt.scene=$(SCENE) material=$(MATERIAL) \
	projector.index=$(PROJ_INDEX) projector.pattern=$(PROJ_PATTERN) \
 	exp_name=$(EXP_NAME) \
	sensor=$(SENSOR) sensor.exposure=$(EXPOSURE) $(KWARGS) $(HYDRA_FLAGS)

encode.generate_projector_patterns:
	${PYTHON} quanta_SL/encode/generate_projector_patterns.py

## docs: build HTML docs
docs :
	@cd docs; make html
	@ln -s docs/_build/html/index.html index.html