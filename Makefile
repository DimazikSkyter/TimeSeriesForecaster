SHELL := /bin/bash
POETRY ?= poetry
PACKAGE_PATH := TimeSeriesApp/forecaster_main

.PHONY: help install format lint mypy test check docker-build docker-run docker-test

help:
	@echo "Available targets:"
	@echo "  install       Install project dependencies with dev extras"
	@echo "  format        Format codebase using isort and black"
	@echo "  lint          Run static checks (isort, black, mypy)"
	@echo "  mypy          Run mypy type checking"
	@echo "  test          Run pytest test suite"
	@echo "  check         Run all quality gates (lint + tests)"
	@echo "  docker-build  Build application Docker image"
	@echo "  docker-run    Run application container exposing Streamlit"
	@echo "  docker-test   Run tests inside a temporary Docker container"

install:
	$(POETRY) install --with dev

format:
	$(POETRY) run isort .
	$(POETRY) run black .

lint:
	$(POETRY) run isort --check-only .
	$(POETRY) run black --check .
	$(POETRY) run mypy $(PACKAGE_PATH)

mypy:
	$(POETRY) run mypy $(PACKAGE_PATH)

test:
	$(POETRY) run pytest

check: lint test

docker-build:
	docker build -t timeseries-forecaster .

docker-run:
	docker run --rm -p 8501:8501 --name timeseries-forecaster-app timeseries-forecaster

docker-test: docker-build
	docker run --rm timeseries-forecaster $(POETRY) run pytest
