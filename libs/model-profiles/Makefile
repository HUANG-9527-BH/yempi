.PHONY: all format lint type help refresh-profiles

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

######################
# MODEL PROFILE REFRESH
######################

# Provider map: partner directory name -> models.dev provider ID.
# Used by .github/workflows/refresh_model_profiles.yml via `make refresh-profiles`.
PROFILE_PROVIDERS := \
	anthropic=anthropic \
	deepseek=deepseek \
	fireworks=fireworks-ai \
	groq=groq \
	huggingface=huggingface \
	mistralai=mistral \
	openai=openai \
	openrouter=openrouter \
	perplexity=perplexity \
	xai=xai

# Refresh model profiles for all supported partners in libs/partners/.
# Requires network access, so UV_FROZEN is overridden for this target.
refresh-profiles:
	@for entry in $(PROFILE_PROVIDERS); do \
		partner=$${entry%%=*}; \
		provider=$${entry##*=}; \
		data_dir="../partners/$${partner}/langchain_$$(echo "$${partner}" | tr '-' '_')/data"; \
		echo "--- Refreshing $${partner} (provider: $${provider}) ---"; \
		echo y | UV_FROZEN=false uv run langchain-profiles refresh \
			--provider "$${provider}" \
			--data-dir "$${data_dir}"; \
	done

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/model-profiles --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_model_profiles
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

check_imports: $(shell find langchain_model_profiles -name '*.py')
	$(UV_RUN_LINT) python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'type                         - run type checking'
	@echo 'refresh-profiles             - refresh model profiles for all supported partners'
