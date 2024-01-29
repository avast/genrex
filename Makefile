.PHONY: help \
		setup \
		clean \
		tests \
		install \
		dev-install \

help:
	@echo "Use \`make <target>', where <target> is one of"
	@echo "  help                  -> print this help"
	@echo "  setup                 -> setup development environment (needs virtualenv activated)"
	@echo "  clean                 -> clean all the generated files"
	@echo "  format                -> Format all the python files with isort and black"
	@echo "  black                 -> Format all the python files with black"
	@echo "  black-lint            -> Check format of all the python files with black"
	@echo "  isort                 -> Sort order of imports in python files"
	@echo "  isort-lint            -> Check order of imports in python files"
	@echo "  lint                  -> check code with all linters"
	@echo "  flake                 -> check code style with flake8"
	@echo "  mypy                  -> check code with mypy"
	@echo "  tests                 -> run all tests"

# Setup development environment (needs virtualenv activated).
setup:
	@poetry install
	@poetry shell

clean:
	rm -rf .coverage* coverage
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.py[co]' -exec rm -f {} +

# Run all linters on genrex codebase and format it.
lint:
	@poetry run poe lint

# Format genrex codebase with all formatters.
format-check:
	@poetry run poe format-check

# Format genrex codebase with all formatters.
format:
	@poetry run poe format

tests: install
	@poetry run poe test

install:
	@poetry update
	@poetry install

dev-install: install
	poetry run pre-commit install