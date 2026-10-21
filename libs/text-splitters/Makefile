.PHONY: all format lint type help

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

check_imports: $(shell find langchain_text_splitters -name '*.py')
	uv run python ./scripts/check_imports.py $^

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/text-splitters --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_text_splitters
UV_RUN_LINT = uv run --all-groups
UV_RUN_TYPE = uv run --all-groups
lint_package: UV_RUN_LINT = uv run --group lint
lint_package: UV_RUN_TYPE = uv run --group lint --group typing

lint lint_diff lint_package:
	./scripts/lint_imports.sh
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_TYPE) ty check $(PYTHON_FILES)

type:
	$(UV_RUN_TYPE) ty check $(PYTHON_FILES)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || $(UV_RUN_LINT) ruff check --fix $(PYTHON_FILES)

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'type                         - run type checking'
