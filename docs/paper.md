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

## Figure inventory

The table below lists figures referenced in the paper and the scripts /
artefacts that produce them. **Reproduction commands are TODO(owner)** — the
repository owner will fill these in based on the camera-ready figure list. For
each row, "Status" reflects whether the figure currently exists under
`figures/` in this repo.

| ID | Caption (paper) | Source / script | Output under `figures/` | Status |
|---|---|---|---|---|
| `fig:cpu-gpu-shape` | GEMM sensitivity to shape on Samsung S24 Ultra | TODO(owner) | TODO(owner) | TODO(owner) |
| `fig:cpu-gpu-comm` | GEMM (left) and GEMV (right) results motivate joint modeling of compute and communication | TODO(owner) | TODO(owner) | TODO(owner) |
| `fig:thermal-throttling` | Thermal throttling on Samsung S24 Ultra (CPU left, GPU right) | TODO(owner) | TODO(owner) | TODO(owner) |
| `fig:num_devices_per_gpu_config` | Maximum emulated devices as host GPUs increase | TODO(owner) | TODO(owner) | TODO(owner) |
| `fig:emulator_latency_dilution` | Latency dilution factor versus number of emulated devices | TODO(owner) | TODO(owner) | TODO(owner) |
| `fig:cache_performance_by_gflops` | Execution-time decomposition across GFLOPS with and without GPU buffer reuse | TODO(owner) | TODO(owner) | TODO(owner) |
| `fig:latency-variability` (appendix) | Achieved efficiency vs. workload size (GFLOPs) | TODO(owner) | TODO(owner) | TODO(owner) |
| `fig:motiovation-rtt-layout` (appendix) | On-device memory access overhead vs. network delay | TODO(owner) | TODO(owner) | TODO(owner) |

### `figures/` cleanup pending

The repository currently tracks ~3.4 MB of PDF/PNG outputs under `figures/`
including some that originated from an unrelated LDPC experiment
(`deadline_compliance`, `decode_latency_cdf`, `violation_inefficiency_events`,
`workload_vs_latency`, `timeline_decode_and_sm`, `fig_e2e_latency`,
`figures/comparison/`, `figures/evaluation/`). The owner will decide which
files correspond to EdgeSys '26 paper figures (keep, map to rows above) and
which to delete. Tracked as a follow-up in
[`.omo/plans/opensource-readiness.md`](../.omo/plans/opensource-readiness.md)
§W5 / §7.

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
