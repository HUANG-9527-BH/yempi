.PHONY: all format lint type help

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/nomic --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_nomic
UV_RUN_LINT = uv run --all-groups
UV_RUN_TYPE = uv run --all-groups
lint_package: UV_RUN_LINT = uv run --group lint

lint lint_diff lint_package:
	./scripts/lint_imports.sh
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && $(UV_RUN_TYPE) mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

type:
	mkdir -p $(MYPY_CACHE) && $(UV_RUN_TYPE) mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check --fix $(PYTHON_FILES)

check_imports: $(shell find langchain_nomic -name '*.py')
	$(UV_RUN_LINT) python ./scripts/check_imports.py $^

check_version:
	uv run python ./scripts/check_version.py

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'check_version                - validate version consistency'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'type                         - run type checking'
