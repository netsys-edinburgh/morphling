# Contributing to Morphling (DeviceEmulator)

Thank you for considering a contribution! Morphling is the companion code for
the EdgeSys '26 paper *"Morphling: Emulator for Distributed Machine Learning
at the Edge"* and is maintained as a long-running research codebase. We
welcome bug reports, feature requests, documentation improvements, and code
contributions.

## Table of Contents

- [Community](#community)
- [How to Contribute](#how-to-contribute)
- [Development Environment](#development-environment)
- [Testing Policy (Docker-only)](#testing-policy-docker-only)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Reporting Bugs](#reporting-bugs)

## Community

This project adopts the [Contributor Covenant Code of
Conduct](CODE_OF_CONDUCT.md). By participating you agree to abide by its
terms. Report unacceptable behavior privately via GitHub Security
Advisories: <https://github.com/drunkcoding/DeviceEmulator/security/advisories/new>.

Security vulnerabilities follow the same private channel. See
[SECURITY.md](SECURITY.md) for the reporting process and response
timeline.

## How to Contribute

1. Check the [issue tracker](https://github.com/drunkcoding/DeviceEmulator/issues)
   for open issues, or open a new one to discuss your idea before writing code.
2. Follow the
   [Fork-and-Pull-Request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)
   workflow.
3. Ensure your change passes formatting, lint, and the Docker test suite (see
   below).
4. Submit a pull request:
   - Title follows the [Commit Message Guidelines](#commit-message-guidelines).
   - Description follows
     [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md).
   - Link any related issues (e.g., `closes #123`).

If you fix a bug, add a regression test under `tests/python/unit/` (Python) or
`tests/cpp/unit/` (C++), following the test organization in `CLAUDE.md` §8.
If you add a feature, include unit tests and update the relevant doc under
`docs/`.

## Development Environment

We use [`pre-commit`](https://pre-commit.com/) for formatting and lint hooks
(`ruff`, `ruff-format`, `clang-format`, `codespell`, plus standard
whitespace/EOF checks).

```bash
pip install -r requirements-lint.txt
pre-commit install --install-hooks
```

After installation, every `git commit` runs the hooks automatically. To run
them manually across the whole repo:

```bash
pre-commit run --all-files
# or
make format
```

## Testing Policy (Docker-only)

**Hard rule:** all tests run inside the canonical Docker image. The image
bakes the installed Python package plus compiled C++/CUDA artifacts, so any
code change requires a rebuild.

```bash
# Rebuild image after any code change (Python or C++)
docker build -t device-emulator:latest .
# or
make docker-build

# Run the full test suite
docker run --rm --gpus all device-emulator:latest python3 -m pytest tests -v
# or
make docker-test
```

### What CI does (and does not)

CI (`.github/workflows/build.yml`, "Build Sanity") validates build
*inputs* only: Dockerfile lint (hadolint), `pyproject.toml` validation,
`pip install --dry-run` of `requirements*.txt`, `MANIFEST.in` glob
coverage, `CITATION.cff` parsing, and presence of required community
files. It does **not** run `docker build`, does **not** import the
package, and does **not** run pytest (GPU or CPU). Every PR must
additionally pass `make docker-test` locally; CPU-only contributors
should label the PR `needs-gpu-verification` so a maintainer can run
the GPU suite.

### Local fast-iteration loop

For a faster local iteration loop, see [docs/DEV_README.md](docs/DEV_README.md).
The Docker path (`make docker-test`) remains the required local
pre-merge verification path; CI itself only validates build inputs (see
above).

If you change a `.proto` file, update both C++ and Python consumers in the
same PR and re-run the Docker tests. See [`CLAUDE.md`](CLAUDE.md) for
additional architectural conventions (worker pool, memory pools, CUDA green
contexts, cleanup ordering).

## Commit Message Guidelines

We follow the [Angular Commit Format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format).
Each commit consists of a **header** and an optional **body**:

```
<type>: <summary>
<BLANK LINE>
<body>(optional)
```

- **Type**: one of `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`,
  `test`, `chore`.
- **Summary**: brief description in the imperative mood (e.g., "add", not
  "added").
- **Body**: required for non-`docs` commits. Explain *why* the change is
  needed.

Examples:

```
fix: correct CUDA green context cleanup ordering

Destroying cuBLAS handles before restoring the primary context caused
SIGSEGV at process exit. Now we restore with cudaSetDevice() after green
context teardown so the runtime cleanup path finds a valid context.
```

```
docs: clarify zero-copy buffer ownership in scatter_gather
```

Signing off commits is recommended:

```bash
git commit -s -m "feat: add LoadBalancedSchedulingPolicy"
```

## Reporting Bugs

Use [`.github/ISSUE_TEMPLATE/bug_report.yml`](.github/ISSUE_TEMPLATE/bug_report.yml).
Provide GPU model, CUDA version, Docker image ID, `nvidia-smi` output, and the
exact command that reproduces the issue.

For feature requests, use
[`.github/ISSUE_TEMPLATE/feature_request.yml`](.github/ISSUE_TEMPLATE/feature_request.yml).
