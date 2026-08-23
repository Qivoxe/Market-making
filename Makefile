.PHONY: install test lint

install:
	pip install -e .

test:
	pytest tests/

lint:
	ruff check src/
