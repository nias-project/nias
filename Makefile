DOCKER ?= docker
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

ci_requirements:
	# we run pip-compile in a container to ensure that the right Python version is used
	$(DOCKER) run --rm -it -v=$(THIS_DIR):/src python:3.11-bullseye /bin/bash -c "\
		cd /src && \
		pip install pip-tools==6.13.0 && \
		pip-compile --resolver backtracking \
			$(CI_EXTRAS) \
			-o requirements-ci.txt \
		"
