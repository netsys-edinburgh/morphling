# Morphling: Emulator for Distributed Machine Learning at the Edge

> EdgeSys '26 paper companion notes. This document is the canonical place for
> paper metadata, abstract, and figure-reproduction commands. The README links
> here from its `## Paper` and `## Citation` sections.

## Venue

**The 24th Annual International Conference on Mobile Systems, Applications and
Services (MobiSys Workshop '26)**, Cambridge, United Kingdom, June 21–25, 2026.

- DOI: [`10.1145/3812836.3814779`](https://doi.org/10.1145/3812836.3814779)
- ISBN: `979-8-4007-2712-2/26/06`
- License: CC BY 4.0

## Authors

- Leyang Xue (The University of Edinburgh)
- Yufeng Xia (The University of Edinburgh)
- Eren Mendi (The University of Edinburgh)
- Ismaeel Bashir (The University of Edinburgh)
- Jiaxun Yang (The University of Edinburgh)
- Myungjin Lee (Cisco Research)
- Mahesh K. Marina (The University of Edinburgh)

## Abstract

Evaluating distributed ML training on heterogeneous edge devices at scale
demands an emulator that jointly provides system efficiency for scaling to a
large number of devices, single-device fidelity, and distributed-device
fidelity. No existing tool meets all three: device emulators incur high
per-device overhead, performance models lack a distributed runtime, and
training simulators omit device-level thermal dynamics and heterogeneity.

Morphling addresses these gaps with a measurement-driven approach and a
unified backend that runs unmodified training scripts on both real and
emulated devices: it intercepts backend dispatch calls, fits stride-aware and
thermal-aware models from real measurements, decouples memory demand from
device count, and uses event-driven virtual time to preserve execution
semantics. The performance model yields latency MAE on the order of
milliseconds on CPU and GPU, and temperature MAE within a few degrees
Celsius. The emulator scales to thousands of emulated devices on a single
GPU server, enabling practical what-if studies of edge training without
large physical device fleets.

## Figure inventory and data availability

The figures in the EdgeSys '26 camera-ready paper were produced by an
experiment pipeline that depended on a private companion baselines
repository and on per-device measurement data collected on hardware
that is not part of this release. The pipeline scripts were therefore
removed from this open-source distribution (see commit
`chore(release): purge paper figures, plot scripts, and morphling.evaluation`).

For all figures below the status is **[data not public]**: the source
data, the plotting scripts, and the rendered PDFs are not part of this
repository.

| ID | Caption (paper) | Status |
|---|---|---|
| `fig:cpu-gpu-shape` | GEMM sensitivity to shape on Samsung S24 Ultra | [data not public] |
| `fig:cpu-gpu-comm` | GEMM (left) and GEMV (right) results motivate joint modeling of compute and communication | [data not public] |
| `fig:thermal-throttling` | Thermal throttling on Samsung S24 Ultra (CPU left, GPU right) | [data not public] |
| `fig:num_devices_per_gpu_config` | Maximum emulated devices as host GPUs increase | [data not public] |
| `fig:emulator_latency_dilution` | Latency dilution factor versus number of emulated devices | [data not public] |
| `fig:cache_performance_by_gflops` | Execution-time decomposition across GFLOPS with and without GPU buffer reuse | [data not public] |
| `fig:latency-variability` (appendix) | Achieved efficiency vs. workload size (GFLOPs) | [data not public] |
| `fig:motiovation-rtt-layout` (appendix) | On-device memory access overhead vs. network delay | [data not public] |

If you need to reproduce or extend any figure, the runtime APIs that
the original pipeline exercised remain available in this repository
(`morphling/runtime/*`, `morphling/hooks/*`, the C++ backend under
`csrc/backend/`, and the CUDA green-context layer documented in
[`docs/green-context.md`](green-context.md)). The plotting layer must
be re-implemented against your own measurement data.

## BibTeX

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

The same metadata in CFF (machine-readable) form is at
[`CITATION.cff`](../CITATION.cff).
