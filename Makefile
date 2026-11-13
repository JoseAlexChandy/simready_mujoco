PY ?= .venv/bin/python
SRMJ ?= .venv/bin/srmj

.PHONY: help setup doctor test lint subset scene roundtrip clean pack release fetch-assets

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

scene: ## Put all converted assets on a floor + table and open the viewer
	$(SRMJ) scene configs/scene.yaml --view

roundtrip: ## Validate the converter: Menagerie FR3 -> USD -> MJCF, compared with the original
	PATH="$(CURDIR)/.venv/bin:$$PATH" scripts/roundtrip_menagerie.sh

pack: ## Pack outputs/ and data/koyo into dist/ for a release (see scripts/pack_assets.sh)
	scripts/pack_assets.sh

TAG ?= assets-$(shell date +%Y-%m-%d)
release: pack ## Upload dist/ as GitHub Release $(TAG) (needs gh, logged in)
	gh release create $(TAG) dist/* --title "Converted assets $(TAG)" \
	  --notes "Converted MuJoCo models (outputs/) and downloaded assets (data/). Get them in a clone with: make fetch-assets. Licences and sources: ATTRIBUTION.md."

fetch-assets: ## Download and unpack the newest assets-* release into outputs/ and data/
	scripts/fetch_assets.sh

clean: ## Delete conversion output (keeps downloaded assets)
	rm -rf outputs
