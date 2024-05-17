DOCKER ?= docker
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: ci_requirements docs_requirements docs

ci_requirements:
	# we run pip-compile in a container to ensure that the right Python version is used
	$(DOCKER) run --rm -it -v=$(THIS_DIR):/src python:3.11-bullseye /bin/bash -c "\
		cd /src && \
		pip install pip-tools==6.13.0 && \
		pip-compile --resolver backtracking \
			--extra tests \
			-o requirements-ci.txt \
	"

docs_requirements:
	# we run pip-compile in a container to ensure that the right Python version is used
	$(DOCKER) run --rm -it -v=$(THIS_DIR):/src python:3.11-bullseye /bin/bash -c "\
		cd /src && \
		pip install pip-tools==6.13.0 && \
		pip-compile --resolver backtracking \
			--extra docs \
			-o requirements-docs.txt \
	"

docs:
	sphinx-build -nW --keep-going -b html docs/ docs/_build/html
