# CLAUDE.md — DeviceEmulator (Morphling) agent notes

## 0) Directory knowledge (read this first)

**Top-level map:**

- `morphling/` — Python package implementation (runtime, entrypoints, utils, ops)
- `csrc/` — C++ sources
  - `csrc/backend/` — core server/proxy/mqtt components (high impact)
  - `csrc/base/` — common C++ utilities (logging, threading, etc.)
  - `csrc/checkpoint/` — checkpoint / IO components
- `proto/` — protobuf / gRPC schemas (`*.proto`) (API surface; change only when requested)
- `config/` — runtime configs
  - `config/proxy/` — proxy client/server INI configs
  - `config/emqx/`, `config/mosquitto/` — broker configs
- `scripts/` — orchestration scripts (run devices, server, profiling, analysis)
- `tests/python/` — Python tests
- `tests/cpp/` — C++ tests/benchmarks
- `cmake/` — CMake helpers
- `Dockerfile` — canonical dev/test environment (tests run in Docker only)

**Heuristics:**
- Runtime behavior / CLI / Python logic → `morphling/`, `scripts/`
- Perf / concurrency / networking → `csrc/backend/`
- Config-driven behavior → `config/**` + config loading callsites in `scripts/`/`morphling/`
- Wire format / RPC contracts → `proto/*.proto` (ask before changing)

## 1) Development workflow (Docker-only; GPU by default)

**Hard rule:** the container built from `Dockerfile` is the source of truth.

### Build (rebuild) the image

Rebuild is required after *any* code change (Python or C++).

```bash
docker build -t device-emulator:latest .
```

### Optional: interactive shell in the same environment

```bash
docker run --rm -it --gpus all device-emulator:latest bash
```

## 2) Testing policy: run tests in Docker only

**Hard rule:** run tests in a container created from the rebuilt image.

```bash
docker run --rm --gpus all device-emulator:latest python3 -m pytest tests -v
```

## 3) “Modifications need to be in sync” rules

Because the Docker image bakes the installed package + compiled artifacts, **changes must be synced by rebuilding the image**.

### Any code change (Python or C++)
1) Rebuild image
```bash
newgrp docker
docker build -t device-emulator:latest .
```

2) Run tests in container
```bash
newgrp docker
docker run --rm --gpus all device-emulator:latest python3 -m pytest tests -v
```

### Dependency / build changes
If you change `Dockerfile` / system deps / Python deps, rebuild and re-test the same way.

### Proto / wire contract changes
- Treat `proto/**` as public API.
- Ask/confirm before editing.
- If edited, update all consumers together (C++ + Python) and re-run the Docker tests.

## 4) Conventions

- Python lint: `ruff` (see `pyproject.toml`), line length 80.
- C++ format: `.clang-format`.
- Prefer minimal, targeted edits; follow existing patterns.

## 5) “Don’t do this without asking”

- Don't change `proto/*.proto` unless explicitly requested.
- Don't do broad refactors in `csrc/backend/`.
- Don't add heavy dependencies.
- Don't provide summary of task, only when asked
- Any runtime fix, please update corresponding readme