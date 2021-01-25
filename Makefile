# Define macros
UNAME_S := $(shell uname -s)
PYTHON := python

## HYDRA_FLAGS : set as -m for multirun
HYDRA_FLAGS := -m
SEED := 0

.PHONY: help
.DEFAULT: help

## install: Pip requirements, local development mode
install:
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

##clean: Remove all outputs
clean:
	@rm -rf outputs/*


# Simulate defaults
SCENE := sphere
MATERIAL := diffuse_0.3
PROJ_INDEX := 12
PROJ_PATTERN := ConventionalGray
DEVICE := cpu
EXP_NAME := $(SCENE)
PBRT_EXEC := /Users/varun/Dev/pbrt-v4/build/pbrt

## simulate: args (SCENE, MATERIAL, PROJ_INDEX, DEVICE, PBRT_EXEC)
simulate:
	${PYTHON} pypbrt/simulate.py pbrt.scene=$(SCENE)  material=$(MATERIAL) \
 	projector.index=$(PROJ_INDEX) projector.pattern=$(PROJ_PATTERN) \
 	exp_name=$(EXP_NAME) \
 	device=$(DEVICE) pbrt.executable=$(PBRT_EXEC)  $(HYDRA_FLAGS)
