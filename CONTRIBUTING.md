# Contributing to Morphling (DeviceEmulator)

Thank you for considering a contribution! Morphling is the companion code for
the EdgeSys '26 paper *"Morphling: Emulator for Distributed Machine Learning
at the Edge"* and is maintained as a long-running research codebase. We
welcome bug reports, feature requests, documentation improvements, and code
contributions.

## Table of Contents

- [How to Contribute](#how-to-contribute)
- [Merge Policy](#merge-policy)
- [Development Environment](#development-environment)
- [Testing Policy (Docker-only)](#testing-policy-docker-only)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Reporting Bugs](#reporting-bugs)

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

If you fix a bug, add a regression test under `tests/python/unit/` or
`tests/cpp/unit/` where possible.
If you add a feature, include unit tests and update the relevant doc under
`docs/`.

## Merge Policy

To keep `main` coherent and reviewed, only the project owner presses the
**Merge** button on pull requests. Contributors with write access should:

- Open pull requests targeting `main`.
- Push commits to feature/PR branches as needed.
- Review pull requests, leave comments, formally Approve / Request changes.
- **Not press the Merge button** on any PR, including their own. Wait for the
  owner to merge after approval.

Pull requests must have at least one approving review (from a Code Owner
where applicable, per [`.github/CODEOWNERS`](.github/CODEOWNERS)) and a green
CI status before the owner merges. Direct pushes to `main` are reserved for
the owner only. This policy is currently enforced socially and will be
enforced via GitHub branch protection once the repository goes fully public.

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

A GPU is required for the full suite. CPU-only contributors should still
verify that `docker build` succeeds and rely on CI for the GPU runs.

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
git commit -s -m "feat: add SchedulingPolicy::ShortestWait"
```

## Reporting Bugs

Use [`.github/ISSUE_TEMPLATE/bug_report.yml`](.github/ISSUE_TEMPLATE/bug_report.yml).
Provide GPU model, CUDA version, Docker image ID, `nvidia-smi` output, and the
exact command that reproduces the issue.

For feature requests, use
[`.github/ISSUE_TEMPLATE/feature_request.yml`](.github/ISSUE_TEMPLATE/feature_request.yml).
