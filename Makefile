SHELL = /bin/bash

# Help
.PHONY: help
help:
	@echo "Commands:"
	@echo "venv: creates a virtual environment."
	@echo "style: executes file formating."

# Styling
.PHONY: style
style:
	black .
	flake8 .
	python3 -m isort .

# Environment
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e .