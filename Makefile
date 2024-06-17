# parameters
CURL := $(shell command -v curl 2> /dev/null)
POETRY := $(shell command -v poetry 2> /dev/null)
RUFF := $(shell command -v ruff 2> /dev/null)
ISORT := $(shell command -v isort 2> /dev/null)
MKFILEPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
DIRNAME := $(notdir $(CURDIR))

# default command when `make` is called
.DEFAULT_GOAL := help

# the main help
.PHONY: help
help:
	@echo "Please use 'make <target>', where <target> is one of"
	@echo ""
	@echo "  init        uses poetry to initialize a new project"
	@echo "  install     install packages and prepare environment"
	@echo "  lock        uses poetry to generate requirements.txt and poetry.lock files"
	@echo "  format      formats the code"
	@echo "  test        runs pytest on the suite of tests within tests/"
	@echo "  precommit   runs lock, format, and tests together"
	@echo ""
	@echo "Check the Makefile to know exactly what each target is doing."

# for checking that poetry exists
.PHONY: curl_check
curl_check:
	@if [ -z $(CURL) ]; then echo "Curl could not be found."; exit 2; fi
.PHONY: poetry_check
poetry_check:
	@if [ -z $(POETRY) ]; then echo "Poetry could not be found."; exit 2; fi
.PHONY: ruff_check

# for initializing the project from scratch
.PHONY: init
init: curl_check poetry_check
	# create the project in this directory
	$(POETRY) new $(DIRNAME)
	# cleanup things
	mv $(DIRNAME) temp
	mv temp/* ./
	rmdir temp
	rm -f README.rst
	# make some default directories and files
	mkdir data
	mkdir src
	@bash -c "echo -e 'if __name__ == \"__main__\":\n    pass' > src/main.py"
	touch .env
	# template readme
	rm -f README.md
	echo "# TODO" > README.md
	# gitignore
	$(CURL) https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore -o .gitignore
	# perform a git init
	git init -b main
	# because poetry pins pytest version to obsolescence...
	# we need to manual update
	$(POETRY) remove --dev pytest
	$(POETRY) add --dev pytest
	$(POETRY) add python-dotenv
	# run the install
	$(MAKE) install
	# test precommit
	$(MAKE) precommit

# installs the package
.PHONY: install
install: poetry_check
	$(POETRY) install

# freezes things
.PHONY: lock
lock: poetry_check
	$(POETRY) lock
	$(POETRY) export -f requirements.txt --output requirements.txt

# linting, formatting, import sorting
.PHONY: format
format: poetry_check
	@if [ -z $(RUFF) ]; then $(POETRY) run pip install --upgrade ruff; fi
	@if [ -z $(ISORT) ]; then $(POETRY) run pip install --upgrade ISORT; fi
	@echo ""
	@echo "Checking code for linting errors..."
	-$(POETRY) run ruff check .
	@echo ""
	@echo "Formatting code..."
	-$(POETRY) run ruff format .
	@echo ""
	@echo "Sorting dependencies..."
	-$(POETRY) run isort .

# tests
.PHONY: test
test: poetry_check
	$(POETRY) run pytest tests -vvv

# basic precommit
.PHONY: precommit
precommit: lock format test
