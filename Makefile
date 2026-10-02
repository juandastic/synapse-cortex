PYTHON ?= .venv/bin/python

.PHONY: verify lint lint-fix format format-check test

verify: format-check lint test

lint:
	$(PYTHON) -m ruff check app tests

lint-fix:
	$(PYTHON) -m ruff check --fix app tests

format:
	$(PYTHON) -m ruff format app tests

format-check:
	$(PYTHON) -m ruff format --check app tests

test:
	$(PYTHON) -m pytest -q
