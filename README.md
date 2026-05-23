<div align="center">

# Morphling

**Emulator for Distributed Machine Learning at the Edge**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Format](https://github.com/drunkcoding/DeviceEmulator/actions/workflows/format.yaml/badge.svg)](https://github.com/drunkcoding/DeviceEmulator/actions/workflows/format.yaml)
[![Build](https://github.com/drunkcoding/DeviceEmulator/actions/workflows/build.yml/badge.svg)](https://github.com/drunkcoding/DeviceEmulator/actions/workflows/build.yml)
[![EdgeSys '26](https://img.shields.io/badge/EdgeSys-2026-blue)](https://doi.org/10.1145/3812836.3814779)

</div>

---

## About

Morphling (DeviceEmulator) emulates distributed machine-learning training on
heterogeneous edge devices. It runs unmodified training scripts on a unified
backend — real or emulated — intercepts backend dispatch calls, fits
stride-aware and thermal-aware performance models from real measurements,
decouples memory demand from device count, and uses event-driven virtual time
to preserve execution semantics. The result is large-scale what-if studies of
edge training without a physical device fleet.

## Paper

> **Morphling: Emulator for Distributed Machine Learning at the Edge**
> Leyang Xue, Yufeng Xia, Eren Mendi, Ismaeel Bashir, Jiaxun Yang, Myungjin
> Lee, Mahesh K. Marina.
> *MobiSys Workshop '26 (EdgeSys '26), Cambridge, United Kingdom, June 2026.*
> [DOI: 10.1145/3812836.3814779](https://doi.org/10.1145/3812836.3814779)
> &middot; [Citation](#citation)
> &middot; [Figure inventory](docs/paper.md)

## Key Features

- **Per-GEMM CUDA green-context switching** — trace-driven SM partitioning
  routed through autograd hooks, see [`docs/green-context.md`](docs/green-context.md).
- **Worker pool** with pluggable scheduling policies (round-robin,
  shortest-wait); GPU path via `XtGemmWorker` (cuBLASXt), CPU path via
  `CpuWorker` (MKL).
- **Zero-copy scatter-gather buffers** for inter-device data transfer over
  libevent (`evbuffer_add_reference` with `shared_ptr` cleanup).
- **Pool-based memory management** — pinned host pool, aligned buffer pool,
  bucketed by power-of-2 sizes and `mlock`ed for stable latency.
- **Virtual + physical device emulation** under one runtime, with a single
  CLI surface.

## Hardware & Software Requirements

- Linux host with Docker and the NVIDIA Container Toolkit (`--gpus all`).
- NVIDIA GPU supported by the runtime path (development uses L40S; green
  contexts require driver 12.5+).
- CUDA toolkit 12.x inside the image (provided by the Dockerfile).

## Installation

The canonical environment is the Docker image defined in
[`Dockerfile`](Dockerfile). Rebuild after any code change.

```bash
docker build -t device-emulator:latest .
# or
make docker-build
```

For native (non-Docker) development on a configured host, see
[`docs/DEV_README.md`](docs/DEV_README.md) and
[`docs/troubleshooting.md`](docs/troubleshooting.md).

## Quick Start

```bash
morphling_cmd save --model "facebook/opt-125m" --output <ckpt-path>
morphling_emulator --ckpt_path <ckpt-path>
```

Multi-device deployments (virtual fleet on one host, or physical edge
devices behind an Nginx stream proxy) are documented in
[`docs/deployment.md`](docs/deployment.md).

## Testing

All tests run inside the Docker image (per [`CLAUDE.md`](CLAUDE.md) §1–2).

```bash
make docker-test
# or
docker run --rm --gpus all device-emulator:latest python3 -m pytest tests -v
```

The image builds **all** C++ test categories (unit, CUDA/cuBLAS,
XtGemm/worker, zerocopy, benchmarks). See
[`tests/cpp/README.md`](tests/cpp/README.md) for the full catalogue.

## Contributing

We welcome bug reports, feature requests, documentation, and code. Start with
[`CONTRIBUTING.md`](CONTRIBUTING.md) — it covers the merge policy,
pre-commit setup, Docker-only test policy, and Angular-style commit format.

## Documentation

- [`docs/paper.md`](docs/paper.md) — EdgeSys '26 paper companion (abstract,
  authors, figure inventory, BibTeX).
- [`docs/DEV_README.md`](docs/DEV_README.md) — native development notes.
- [`docs/DOCKER.md`](docs/DOCKER.md) — Docker workflow deep-dive.
- [`docs/green-context.md`](docs/green-context.md) — per-GEMM CUDA green
  context API.
- [`docs/deployment.md`](docs/deployment.md) — virtual + physical device
  deployments.
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — common errors and
  fixes.
- [`docs/GEMM_ID_ISSUES.md`](docs/GEMM_ID_ISSUES.md) — GEMM identification
  internals.
- [`docs/EARLIEST_vs_LATEST.md`](docs/EARLIEST_vs_LATEST.md) — scheduling
  policy notes.
- [`tests/cpp/README.md`](tests/cpp/README.md) — C++ test catalogue.
- [`docs/cuda/README.md`](docs/cuda/README.md) — offline CUDA Driver/Runtime
  API reference.

## Citation

If you use Morphling (DeviceEmulator) in your research, please cite our
EdgeSys '26 workshop paper:

```bibtex
@inproceedings{DBLP:conf/mobisys/XueXMBYLM26,
  author       = {Leyang Xue and
                  Yufeng Xia and
                  Eren Mendi and
                  Ismaeel Bashir and
                  Jiaxun Yang and
                  Myungjin Lee and
                  Mahesh K. Marina},
  title        = {Morphling: Emulator for Distributed Machine Learning at the Edge},
  booktitle    = {The 24th Annual International Conference on Mobile Systems,
                  Applications and Services, MobiSys Workshop '26,
                  Cambridge, United Kingdom, June 21-25, 2026},
  publisher    = {{ACM}},
  year         = {2026},
  url          = {https://doi.org/10.1145/3812836.3814779},
  doi          = {10.1145/3812836.3814779}
}
```

A machine-readable citation is also available in
[`CITATION.cff`](CITATION.cff) — GitHub renders a "Cite this repository"
button from it.

## Acknowledgements

Morphling builds on excellent open-source work:

- [PyTorch](https://pytorch.org/) — autograd backend hooks and CUDA cpp extensions.
- [libevent](https://libevent.org/) — event-driven networking and `evbuffer` zero-copy plumbing.
- [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) — CPU GEMM backend.
- [muduo](https://github.com/chenshuo/muduo) — vendored `muduo_base` for logging and threading utilities.
- [BatchGen](https://github.com/batchgen-project/batchgen) — reference open-source repo structure.
