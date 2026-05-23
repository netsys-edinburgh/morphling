# ---------------------------------------------------------------------------- #
#  Morphling (DeviceEmulator) — top-level Makefile                             #
# ---------------------------------------------------------------------------- #
#  Conventions: the canonical build/test environment is the Docker image       #
#  defined in ./Dockerfile. Native (non-Docker) targets exist for local        #
#  development but are not guaranteed reproducible across hosts.               #
# ---------------------------------------------------------------------------- #

IMAGE ?= device-emulator:latest

.PHONY: help install install-dev test docker-build docker-test format clean

# Default target — show available commands.
help:
	@echo "Morphling (DeviceEmulator) — Make targets"
	@echo ""
	@echo "  make install        pip install -e .  (no isolation; requires CUDA + libs)"
	@echo "  make install-dev    install + lint dependencies (pre-commit, clang-format, ...)"
	@echo "  make docker-build   docker build -t $(IMAGE) ."
	@echo "  make docker-test    docker run --rm --gpus all $(IMAGE) python3 -m pytest tests -v"
	@echo "  make test           alias for docker-build + docker-test (canonical CI path)"
	@echo "  make format         pre-commit run --all-files"
	@echo "  make clean          remove local build artefacts"
	@echo "  make help           show this message"

# Native install (matches README current path).
install:
	pip install --no-build-isolation -e .

# Native install + lint deps for local development.
install-dev: install
	pip install -r requirements-lint.txt
	pre-commit install --install-hooks

# Canonical test path per CLAUDE.md §1–2: rebuild image, run pytest in container.
test: docker-build docker-test

# Build the canonical Docker image.
docker-build:
	docker build -t $(IMAGE) .

# Run the Python test suite inside the Docker image (requires --gpus all).
docker-test:
	docker run --rm --gpus all $(IMAGE) python3 -m pytest tests -v

# Run all pre-commit hooks across the repo.
format:
	pre-commit run --all-files

# Remove local build artefacts. Does NOT touch the Docker image or .git.
clean:
	rm -rf build/ build-verify-msg/ dist/ *.egg-info/ .eggs/
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	rm -rf .pytest_cache/ .ruff_cache/
