PY ?= .venv/bin/python
SRMJ ?= .venv/bin/srmj

.PHONY: help setup doctor test lint subset roundtrip clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv, fetch the upstream converter, install everything
	git submodule update --init --recursive
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[all]"
	$(PY) -m pip install -r third_party/usd2mjcf/requirements.txt
	.venv/bin/pre-commit install || true
	@echo "\nDone. Run 'make doctor' to check the environment."

doctor: ## Check that every dependency is installed and working
	$(SRMJ) doctor

test: ## Run the test suite
	env -u PYTHONPATH $(PY) -m pytest

lint: ## Lint and format the code
	.venv/bin/ruff check --fix src tests
	.venv/bin/ruff format src tests

subset: ## Convert every asset listed in configs/subset.yaml
	$(SRMJ) batch configs/subset.yaml

roundtrip: ## Validate the converter: Menagerie FR3 -> USD -> MJCF, compared with the original
	PATH="$(CURDIR)/.venv/bin:$$PATH" scripts/roundtrip_menagerie.sh

clean: ## Delete conversion output (keeps downloaded assets)
	rm -rf outputs
