# EARLIEST vs LATEST Sync Strategy Comparison

## Executive Summary

Two strategies are used to align virtual time across devices:

- `EARLIEST`: align all devices to the earliest `vt_start` (fastest device).
- `LATEST`: align all devices to the latest `vt_start` (slowest device).

Both align device timelines, but they shift the global timestamp differently.
Choose based on whether you care about performance comparison or worst-case
system behavior.

## Strategy Comparison

| Dimension | EARLIEST | LATEST |
| --- | --- | --- |
| Baseline definition | `min(vt_start)` | `max(vt_start)` |
| Baseline device | Fastest | Slowest |
| Time shift direction | Backward (smaller timestamps) | Forward (larger timestamps) |
| Typical shift size | Smaller (conservative) | Larger (aggressive) |

## Worked example (illustrative)

Suppose three devices start the same GEMM at these virtual times:

```
Device 0: vt_start = 1,100 us (slowest)
Device 1: vt_start = 1,060 us
Device 2: vt_start = 1,000 us (fastest)

Spread = 100 us
```

EARLIEST (baseline = `min` = 1,000 us) pulls every device back to the
fastest start:

```
Device 0: 1,100 - 100 = 1,000 us
Device 1: 1,060 -  60 = 1,000 us
Device 2: 1,000 -   0 = 1,000 us
```

LATEST (baseline = `max` = 1,100 us) advances every device to the slowest
start:

```
Device 0: 1,100 +   0 = 1,100 us
Device 1: 1,060 +  40 = 1,100 us
Device 2: 1,000 + 100 = 1,100 us
```

## Key Differences

1. Global timestamp changes

- `EARLIEST` shifts all timestamps backward.
- `LATEST` shifts all timestamps forward.

2. How fast devices are treated

- `EARLIEST`: fast devices are pulled back to the earliest time.
- `LATEST`: fast devices are pushed forward to the slowest time.

3. How slow devices are represented

- `EARLIEST`: slow devices are pulled back, making them look less slow.
- `LATEST`: slow devices define the baseline, exposing their impact.

## Which should you use?

### Goal: performance comparison

Use `EARLIEST`.

- Aligns to the fastest device.
- Best for comparing device compute time fairly.
- Preserves relative ordering and uses a conservative shift.

### Goal: worst-case or SLA analysis

Use `LATEST`.

- Aligns to the slowest device.
- Highlights tail latency and bottlenecks.
- Useful for reliability and capacity planning.

### Goal: cross-experiment comparison

Neither is ideal.

- Both alter absolute timestamps.
- Consider keeping original logs and comparing deltas.
- Or use a fixed baseline device/time for all experiments.

## Recommended Approach

If your primary goal is performance comparison, `EARLIEST` is the right choice.
If you also care about worst-case impact, generate both and compare.

Example workflow:

```bash
python3 scripts/sync_virtual_time.py perf_merged.log --method earliest --output synced_earliest.log
python3 scripts/sync_virtual_time.py perf_merged.log --method latest --output synced_latest.log

python3 scripts/analyze_sync.py perf_merged.log synced_earliest.log > analysis_earliest.txt
python3 scripts/analyze_sync.py perf_merged.log synced_latest.log > analysis_latest.txt
```
