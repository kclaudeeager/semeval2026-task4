# The venv lives OUTSIDE the synced folder on purpose: a .venv inside OneDrive
# means tens of thousands of files syncing, and sync can corrupt it mid-install.
VENV ?= $(HOME)/.venvs/task4
PY   := $(VENV)/bin/python

.PHONY: help setup data test ablation diagnostic lint clean
help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  %-12s %s\n", $$1, $$2}'

setup:      ## create the venv (outside OneDrive) and install the package
	uv venv $(VENV) && VIRTUAL_ENV=$(VENV) uv pip install -e ".[dev]"
	@echo "activate with: source $(VENV)/bin/activate"

data:       ## unpack the dataset into data/raw
	bash scripts/fetch_data.sh

doctor:     ## check which layers this environment can actually run
	$(PY) scripts/doctor.py

models:     ## install the layers that need downloads (L2, L3, L4, L5)
	VIRTUAL_ENV=$(VENV) uv pip install -e ".[embeddings,structure]" --upgrade
	$(PY) -m spacy download en_core_web_sm
	$(PY) scripts/doctor.py

test:       ## run the test suite
	$(PY) -m pytest -q

ablation:   ## run every layer on dev and write reports/results/*.json
	$(PY) -m task4.cli --multirun experiment=ablation

diagnostic: ## per-aspect diagnostic on unanimous strata, paired against L2
	$(PY) -m task4.cli experiment=diagnostic_outcomes

lint:       ## style check
	$(PY) -m ruff check src tests

clean:      ## remove run outputs, keep results
	rm -rf outputs multirun .pytest_cache src/*.egg-info
